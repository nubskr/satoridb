use crate::executor::{Executor, WorkerCache};
use crate::storage::wal::runtime::Walrus;
use crate::storage::Storage;
use crate::storage::{Bucket, Vector};
use async_channel::Receiver;
use futures::channel::oneshot;
use log::{error, info};
use std::collections::HashMap;
use std::sync::Arc;

pub struct QueryRequest {
    pub query_vec: Vec<f32>,
    pub bucket_ids: Vec<u64>,
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

    let storage = Storage::new(wal.clone());
    let cache = WorkerCache::new(64, 8 * 1024 * 1024); // Defaults: 64 buckets, 8MB each
    let executor = Executor::new(storage.clone(), cache);
    let mut ingest_buffers: HashMap<u64, Vec<Vector>> = HashMap::new();

    // Config: Flush when buffer hits 10k vectors (~1.2MB)
    const FLUSH_THRESHOLD: usize = 10_000;

    while let Ok(msg) = receiver.recv().await {
        match msg {
            WorkerMessage::Query(req) => {
                let result = executor.query(&req.query_vec, &req.bucket_ids, 100).await;
                if let Err(_) = req.respond_to.send(result) {
                    error!("Worker {} failed to send response back.", id);
                }
            }
            WorkerMessage::Ingest { bucket_id, vectors } => {
                let buffer = ingest_buffers.entry(bucket_id).or_default();
                buffer.extend(vectors);

                if buffer.len() >= FLUSH_THRESHOLD {
                    // Flush
                    let mut bucket = Bucket::new(bucket_id, vec![]);
                    // Move vectors out of buffer
                    bucket.vectors = std::mem::take(buffer);
                    if let Err(e) = storage.put_chunk(&bucket).await {
                        error!(
                            "Worker {} failed to flush chunk for bucket {}: {:?}",
                            id, bucket_id, e
                        );
                    }
                }
            }
            WorkerMessage::Flush { respond_to } => {
                for (bucket_id, vectors) in ingest_buffers.iter_mut() {
                    if vectors.is_empty() {
                        continue;
                    }
                    let mut bucket = Bucket::new(*bucket_id, vec![]);
                    bucket.vectors = std::mem::take(vectors);
                    if let Err(e) = storage.put_chunk(&bucket).await {
                        error!(
                            "Worker {} failed to flush bucket {} on flush signal: {:?}",
                            id, bucket_id, e
                        );
                    }
                }
                let _ = respond_to.send(());
            }
        }
    }

    // Shutdown: Flush remaining
    info!("Worker {} flushing remaining buffers...", id);
    for (bucket_id, vectors) in ingest_buffers {
        if !vectors.is_empty() {
            let mut bucket = Bucket::new(bucket_id, vec![]);
            bucket.vectors = vectors;
            if let Err(e) = storage.put_chunk(&bucket).await {
                error!(
                    "Worker {} failed to flush final chunk for bucket {}: {:?}",
                    id, bucket_id, e
                );
            }
        }
    }

    info!("Worker {} shutting down.", id);
}
