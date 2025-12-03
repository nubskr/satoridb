use crate::executor::Executor;
use crate::storage::Storage;
use futures::channel::oneshot;
use async_channel::Receiver;
use log::{info, error};
use std::path::PathBuf;

pub struct QueryRequest {
    pub query_vec: Vec<f32>,
    pub bucket_ids: Vec<u64>,
    pub respond_to: oneshot::Sender<anyhow::Result<Vec<(u64, f32)>>>,
}

pub async fn run_worker(id: usize, receiver: Receiver<QueryRequest>, data_dir: PathBuf) {
    info!("Worker {} started.", id);

    // Initialize Storage (Shared path, separate instance)
    let storage = Storage::new(data_dir);
    let executor = Executor::new(storage);

    while let Ok(msg) = receiver.recv().await {
        // info!("Worker {} received query for buckets: {:?}", id, msg.bucket_ids);
        
        // Execute Query
        let result = executor.query(&msg.query_vec, &msg.bucket_ids, 100).await;
        
        // Send Response
        if let Err(_) = msg.respond_to.send(result) {
            error!("Worker {} failed to send response back to dispatcher.", id);
        }
    }

    info!("Worker {} shutting down.", id);
}
