use crate::worker::{QueryRequest, WorkerMessage};
use crate::tasks::{ConsistentHashRing, RouterResult, RouterTask};
use anyhow::{anyhow, Context, Result};
use async_channel::Sender as AsyncSender;
use crossbeam_channel::Sender as CrossbeamSender;
use futures::channel::oneshot;
use futures::future::join_all;
use futures_lite::io::{AsyncReadExt, AsyncWriteExt};
use glommio::net::TcpListener;
use glommio::{LocalExecutorBuilder, Placement};
use log::{error, info};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::thread;

#[derive(Deserialize)]
struct NetQueryRequest {
    vector: Vec<f32>,
    #[serde(default = "NetQueryRequest::default_top_k")]
    top_k: usize,
    #[serde(default = "NetQueryRequest::default_router_top_k")]
    router_top_k: usize,
}

impl NetQueryRequest {
    fn default_top_k() -> usize {
        10
    }

    fn default_router_top_k() -> usize {
        200
    }
}

#[derive(Serialize)]
#[serde(tag = "status", rename_all = "lowercase")]
enum NetResponse {
    Ok { results: Vec<ResultItem> },
    Err { message: String },
}

#[derive(Serialize)]
struct ResultItem {
    id: u64,
    distance: f32,
}

/// Launch a single-threaded glommio reactor that owns all network I/O.
/// Protocol: length-prefixed (u32 LE) JSON frames:
/// Request: { "vector": [...], "top_k": 10, "router_top_k": 200 }
/// Response: { "status": "ok", "results": [{ "id": 1, "distance": 0.1 }] }
pub fn spawn_network_server(
    listen_addr: SocketAddr,
    router_tx: CrossbeamSender<RouterTask>,
    ring: ConsistentHashRing,
    worker_senders: Vec<AsyncSender<WorkerMessage>>,
) -> Result<thread::JoinHandle<()>> {
    let handle = thread::Builder::new()
        .name("net-io".to_string())
        .spawn(move || {
            let builder = LocalExecutorBuilder::new(Placement::Unbound).name("net-reactor");
            let executor = match builder.make() {
                Ok(ex) => ex,
                Err(e) => {
                    error!("Failed to start glommio executor for network: {:?}", e);
                    return;
                }
            };

            executor.run(async move {
                info!("Listening on {}", listen_addr);
                let listener = match TcpListener::bind(listen_addr) {
                    Ok(l) => l,
                    Err(e) => {
                        error!("Failed to bind {}: {:?}", listen_addr, e);
                        return;
                    }
                };

                let mut conn_id: u64 = 0;
                loop {
                    match listener.accept().await {
                        Ok(stream) => {
                            conn_id += 1;
                            let id = conn_id;
                            info!("Connection {} accepted", id);
                            let router_tx = router_tx.clone();
                            let ring = ring.clone();
                            let worker_senders = worker_senders.clone();
                            glommio::spawn_local(async move {
                                if let Err(e) =
                                    handle_connection(stream, router_tx, ring, worker_senders, id)
                                        .await
                                {
                                    error!("Connection {} error: {:?}", id, e);
                                }
                            })
                            .detach();
                        }
                        Err(e) => {
                            error!("Accept error: {:?}", e);
                            break;
                        }
                    }
                }
            });
        })?;

    Ok(handle)
}

async fn handle_connection(
    mut stream: glommio::net::TcpStream,
    router_tx: CrossbeamSender<RouterTask>,
    ring: ConsistentHashRing,
    worker_senders: Vec<AsyncSender<WorkerMessage>>,
    conn_id: u64,
) -> Result<()> {
    loop {
        let frame = match read_frame(&mut stream).await? {
            Some(f) => f,
            None => break, // EOF
        };

        let req: NetQueryRequest = match serde_json::from_slice(&frame) {
            Ok(r) => r,
            Err(e) => {
                write_frame(
                    &mut stream,
                    &NetResponse::Err {
                        message: format!("invalid request (conn {}): {}", conn_id, e),
                    },
                )
                .await?;
                continue;
            }
        };

        let (tx, rx) = oneshot::channel();
        let task = RouterTask {
            query_vec: req.vector.clone(),
            top_k: req.router_top_k,
            respond_to: tx,
        };
        if let Err(e) = router_tx.send(task) {
            write_frame(
                &mut stream,
                &NetResponse::Err {
                    message: format!("router channel closed (conn {}): {:?}", conn_id, e),
                },
            )
            .await?;
            continue;
        }

        let RouterResult {
            bucket_ids,
            routing_version,
            affected_buckets,
        } = match rx.await {
            Ok(Ok(res)) => res,
            Ok(Err(e)) => {
                write_frame(
                    &mut stream,
                    &NetResponse::Err {
                        message: format!("router error (conn {}): {:?}", conn_id, e),
                    },
                )
                .await?;
                continue;
            }
            Err(e) => {
                write_frame(
                    &mut stream,
                    &NetResponse::Err {
                        message: format!("router canceled (conn {}): {:?}", conn_id, e),
                    },
                )
                .await?;
                continue;
            }
        };

        let mut pending = Vec::new();
        let mut requests = Vec::new();
        for &bid in &bucket_ids {
            let shard = ring.node_for(bid);
            if shard >= worker_senders.len() {
                continue;
            }
            if requests.len() <= shard {
                requests.resize_with(shard + 1, Vec::new);
            }
            requests[shard].push(bid);
        }

        for (shard, bids) in requests.into_iter().enumerate() {
            if bids.is_empty() {
                continue;
            }
            let (tx, rx) = oneshot::channel();
            let req = QueryRequest {
                query_vec: req.vector.clone(),
                bucket_ids: bids,
                routing_version,
                affected_buckets: affected_buckets.clone(),
                respond_to: tx,
            };
            if worker_senders[shard]
                .send(WorkerMessage::Query(req))
                .await
                .is_ok()
            {
                pending.push(rx);
            }
        }

        let responses = join_all(pending).await;
        let mut all_results = Vec::new();
        for res in responses {
            if let Ok(Ok(candidates)) = res {
                all_results.extend(candidates);
            }
        }
        all_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        if all_results.len() > req.top_k {
            all_results.truncate(req.top_k);
        }

        let payload = NetResponse::Ok {
            results: all_results
                .into_iter()
                .map(|(id, distance)| ResultItem { id, distance })
                .collect(),
        };
        write_frame(&mut stream, &payload).await?;
    }

    Ok(())
}

async fn read_frame<R>(stream: &mut R) -> Result<Option<Vec<u8>>>
where
    R: AsyncReadExt + Unpin,
{
    let mut len_buf = [0u8; 4];
    match stream.read_exact(&mut len_buf).await {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e.into()),
    }
    let len = u32::from_le_bytes(len_buf) as usize;
    if len > 10 * 1024 * 1024 {
        return Err(anyhow!("frame too large: {} bytes", len));
    }
    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf).await?;
    Ok(Some(buf))
}

async fn write_frame<W>(stream: &mut W, resp: &NetResponse) -> Result<()>
where
    W: AsyncWriteExt + Unpin,
{
    let bytes = serde_json::to_vec(resp).context("serialize response")?;
    let len = bytes.len() as u32;
    stream.write_all(&len.to_le_bytes()).await?;
    stream.write_all(&bytes).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use futures_lite::io::Cursor;
    use serde_json::Value;

    #[test]
    fn write_and_read_frame_round_trips() {
        block_on(async {
            let mut cursor = Cursor::new(Vec::new());
            let resp = NetResponse::Ok {
                results: vec![ResultItem {
                    id: 1,
                    distance: 0.5,
                }],
            };
            write_frame(&mut cursor, &resp).await.unwrap();
            cursor.set_position(0);
            let frame = read_frame(&mut cursor).await.unwrap().unwrap();
            let v: Value = serde_json::from_slice(&frame).unwrap();
            assert_eq!(v["status"], "ok");
            assert_eq!(v["results"][0]["id"], 1);
        });
    }

    #[test]
    fn read_frame_rejects_oversize() {
        block_on(async {
            let max = 10 * 1024 * 1024 + 1;
            let mut data = Vec::new();
            data.extend_from_slice(&(max as u32).to_le_bytes());
            let mut cursor = Cursor::new(data);
            let err = read_frame(&mut cursor).await.unwrap_err();
            assert!(err.to_string().contains("frame too large"));
        });
    }
}
