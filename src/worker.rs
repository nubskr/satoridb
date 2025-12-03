use crate::storage::{Vector, Bucket};
use crate::executor::Executor;
use crate::storage::Storage;
use futures::channel::oneshot;
use async_channel::Receiver;
use log::{info, error};
use std::path::PathBuf;
use std::collections::HashMap;

pub struct QueryRequest {
    pub query_vec: Vec<u8>,
    pub bucket_ids: Vec<u64>,
    pub respond_to: oneshot::Sender<anyhow::Result<Vec<(u64, f32)>>>,
}

pub enum WorkerMessage {
    Query(QueryRequest),
    Ingest { bucket_id: u64, vectors: Vec<Vector> },
}

pub async fn run_worker(id: usize, receiver: Receiver<WorkerMessage>, data_dir: PathBuf) {
    info!("Worker {} started.", id);

    let storage = Storage::new(data_dir);
    let executor = Executor::new(storage.clone());
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
            },
            WorkerMessage::Ingest { bucket_id, vectors } => {
                let buffer = ingest_buffers.entry(bucket_id).or_default();
                buffer.extend(vectors);
                
                if buffer.len() >= FLUSH_THRESHOLD {
                    // Flush
                    let mut bucket = Bucket::new(bucket_id, vec![]);
                    // Move vectors out of buffer
                    bucket.vectors = std::mem::take(buffer);
                    if let Err(e) = storage.put_chunk(&bucket).await {
                        error!("Worker {} failed to flush chunk for bucket {}: {:?}", id, bucket_id, e);
                    }
                }
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
                error!("Worker {} failed to flush final chunk for bucket {}: {:?}", id, bucket_id, e);
            }
        }
    }

    info!("Worker {} shutting down.", id);
}