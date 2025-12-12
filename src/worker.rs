use crate::executor::{Executor, WorkerCache};
use crate::ingest_counter;
use crate::storage::wal::runtime::Walrus;
use crate::storage::Storage;
use crate::storage::Vector;
use async_channel::Receiver;
use futures::channel::oneshot;
use log::{error, info};
use std::sync::Arc;

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

    let storage = Storage::new(wal.clone());
    let cache = WorkerCache::new(512, 64 * 1024 * 1024); // Larger local cache: 512 buckets, 64MB each
    let executor = Executor::new(storage.clone(), cache);
    let mut topic_cache = std::collections::HashMap::new();
    let mut scratch_batch: Vec<Vector> = Vec::with_capacity(2048);
    // Prewarm ingestion thread-local buffers to avoid growth reallocs on the hot path.
    Storage::prewarm_thread_locals(2048, 1024);

    while let Ok(msg) = receiver.recv().await {
        match msg {
            WorkerMessage::Query(req) => {
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
            }
            WorkerMessage::Ingest { bucket_id, vectors } => {
                scratch_batch.clear();
                scratch_batch.reserve(vectors.len());
                scratch_batch.extend(vectors.into_iter());
                let topic = topic_cache
                    .entry(bucket_id)
                    .or_insert_with(|| Storage::topic_for(bucket_id));
                match storage
                    .put_chunk_raw_with_topic(bucket_id, topic.as_str(), &scratch_batch)
                    .await
                {
                    Ok(_) => ingest_counter::add(scratch_batch.len() as u64),
                    Err(e) => error!(
                        "Worker {} failed to persist chunk for bucket {}: {:?}",
                        id, bucket_id, e
                    ),
                }
            }
            WorkerMessage::Flush { respond_to } => {
                let _ = respond_to.send(());
            }
        }
    }

    info!("Worker {} shutting down.", id);
}
