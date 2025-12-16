use crate::executor::{Executor, WorkerCache};
use crate::ingest_counter;
use crate::storage::wal::runtime::Walrus;
use crate::storage::{Bucket, Storage, StorageExecMode, Vector};
use async_channel::Receiver;
use futures::channel::oneshot;
use log::{error, info};
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;

pub struct QueryRequest {
    pub query_vec: Vec<f32>,
    pub bucket_ids: Vec<u64>,
    pub routing_version: u64,
    pub affected_buckets: std::sync::Arc<Vec<u64>>,
    pub respond_to: oneshot::Sender<anyhow::Result<Vec<(u64, f32)>>>,
}

pub enum WorkerMessage {
    Query(QueryRequest),
    Upsert {
        bucket_id: u64,
        vector: Vector,
        respond_to: oneshot::Sender<anyhow::Result<()>>,
    },
    Ingest {
        bucket_id: u64,
        vectors: Vec<Vector>,
    },
    Flush {
        respond_to: oneshot::Sender<()>,
    },
    Shutdown {
        respond_to: oneshot::Sender<()>,
    },
}

pub async fn run_worker(id: usize, receiver: Receiver<WorkerMessage>, wal: Arc<Walrus>) {
    info!("Worker {} started.", id);

    let storage = Storage::new(wal.clone()).with_mode(StorageExecMode::Offload);
    let cache = WorkerCache::new(512, 64 * 1024 * 1024); // Larger local cache: 512 buckets, 64MB each
    let executor = Rc::new(Executor::new(storage.clone(), cache));
    // Shared topic cache not easily safe for concurrent access across spawns without RefCell,
    // but computing topic string is cheap. Let's drop the cache for simplicity in concurrent model.
    // Prewarm ingestion thread-local buffers to avoid growth reallocs on the hot path.
    Storage::prewarm_thread_locals(2048, 1024);

    const MAX_CONCURRENCY: usize = 32;
    // Use channel as a semaphore. Capacity = max concurrency.
    // Loop sends (acquires), Task receives (releases).
    let (limit_tx, limit_rx) = async_channel::bounded(MAX_CONCURRENCY);

    while let Ok(msg) = receiver.recv().await {
        match msg {
            WorkerMessage::Query(req) => {
                // Acquire permit (blocks if full)
                limit_tx.send(()).await.unwrap();
                let limit_rx = limit_rx.clone();
                let executor = executor.clone();
                glommio::spawn_local(async move {
                    let result = executor
                        .query(
                            &req.query_vec,
                            &req.bucket_ids,
                            100,
                            req.routing_version,
                            req.affected_buckets,
                        )
                        .await;
                    if req.respond_to.send(result).is_err() {
                        error!("Worker {} failed to send response back.", id);
                    }
                    // Release permit
                    let _ = limit_rx.recv().await;
                })
                .detach();
            }
            WorkerMessage::Upsert {
                bucket_id,
                vector,
                respond_to,
            } => {
                let mut bucket = Bucket::new(bucket_id, Vec::new());
                bucket.vectors = vec![vector];
                let result = storage
                    .put_chunk(&bucket)
                    .await;
                let _ = respond_to.send(result);
            }
            WorkerMessage::Ingest { bucket_id, vectors } => {
                limit_tx.send(()).await.unwrap();
                let limit_rx = limit_rx.clone();
                let storage = storage.clone();
                glommio::spawn_local(async move {
                    let topic = Storage::topic_for(bucket_id);
                    loop {
                        match storage
                            .put_chunk_raw_with_topic(bucket_id, &topic, &vectors)
                            .await
                        {
                            Ok(_) => {
                                ingest_counter::add(vectors.len() as u64);
                                break;
                            }
                            Err(e) => {
                                let is_would_block = e
                                    .downcast_ref::<std::io::Error>()
                                    .map(|io_err| io_err.kind() == std::io::ErrorKind::WouldBlock)
                                    .unwrap_or(false);

                                if is_would_block {
                                    glommio::timer::Timer::new(Duration::from_millis(5)).await;
                                    continue;
                                }

                                error!(
                                    "Worker {} failed to persist chunk for bucket {}: {:?}",
                                    id, bucket_id, e
                                );
                                break;
                            }
                        }
                    }
                    // Release permit
                    let _ = limit_rx.recv().await;
                })
                .detach();
            }
            WorkerMessage::Flush { respond_to } => {
                // Wait for all active tasks to drain
                while !limit_rx.is_empty() {
                    glommio::timer::Timer::new(Duration::from_millis(10)).await;
                }
                let _ = respond_to.send(());
            }
            WorkerMessage::Shutdown { respond_to } => {
                let _ = respond_to.send(());
                break;
            }
        }
    }

    info!("Worker {} shutting down.", id);
}
