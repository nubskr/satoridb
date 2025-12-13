use crate::executor::{Executor, WorkerCache};
use crate::ingest_counter;
use crate::storage::wal::runtime::Walrus;
use crate::storage::{Storage, StorageExecMode, Vector};
use async_channel::Receiver;
use futures::channel::oneshot;
use glommio::sync::Semaphore;
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
    Ingest {
        bucket_id: u64,
        vectors: Vec<Vector>,
    },
    Flush {
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
    let semaphore = Rc::new(Semaphore::new(MAX_CONCURRENCY as i64));

    while let Ok(msg) = receiver.recv().await {
        match msg {
            WorkerMessage::Query(req) => {
                let permit = semaphore.acquire_permit(1).await.unwrap();
                let executor = executor.clone();
                glommio::spawn_local(async move {
                    let _permit = permit;
                    let result = executor
                        .query(
                            &req.query_vec,
                            &req.bucket_ids,
                            100,
                            req.routing_version,
                            req.affected_buckets,
                        )
                        .await;
                    if let Err(_) = req.respond_to.send(result) {
                        error!("Worker {} failed to send response back.", id);
                    }
                })
                .detach();
            }
            WorkerMessage::Ingest { bucket_id, vectors } => {
                let permit = semaphore.acquire_permit(1).await.unwrap();
                let storage = storage.clone();
                glommio::spawn_local(async move {
                    let _permit = permit;
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
                                // Check for WouldBlock (batch write in progress)
                                let is_would_block = e
                                    .downcast_ref::<std::io::Error>()
                                    .map(|io_err| io_err.kind() == std::io::ErrorKind::WouldBlock)
                                    .unwrap_or(false);

                                if is_would_block {
                                    // Backoff and retry
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
                })
                .detach();
            }
            WorkerMessage::Flush { respond_to } => {
                // Acquire all permits to ensure we drain pending work
                let _all_permits = semaphore
                    .acquire_permits(MAX_CONCURRENCY as i64)
                    .await
                    .unwrap();
                let _ = respond_to.send(());
                // permits released when _all_permits drops at end of scope
            }
        }
    }

    info!("Worker {} shutting down.", id);
}
