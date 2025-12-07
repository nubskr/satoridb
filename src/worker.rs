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
    let cache = WorkerCache::new(512, 64 * 1024 * 1024); // Larger local cache: 512 buckets, 64MB each
    let executor = Executor::new(storage.clone(), cache);

    while let Ok(msg) = receiver.recv().await {
        match msg {
            WorkerMessage::Query(req) => {
                let result = executor.query(&req.query_vec, &req.bucket_ids, 100).await;
                if let Err(_) = req.respond_to.send(result) {
                    error!("Worker {} failed to send response back.", id);
                }
            }
            WorkerMessage::Ingest { bucket_id, vectors } => {
                let mut bucket = Bucket::new(bucket_id, vec![]);
                bucket.vectors = vectors;
                if let Err(e) = storage.put_chunk(&bucket).await {
                    error!(
                        "Worker {} failed to persist chunk for bucket {}: {:?}",
                        id, bucket_id, e
                    );
                }
            }
            WorkerMessage::Flush { respond_to } => {
                let _ = respond_to.send(());
            }
        }
    }

    info!("Worker {} shutting down.", id);
}
