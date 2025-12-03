mod executor;
mod indexer;
mod router;
mod storage;
mod worker;
mod quantizer;

use glommio::{LocalExecutorBuilder, Placement};
use log::info;
use crate::storage::{Storage, Vector};
use crate::router::Router;
use crate::indexer::Indexer;
use crate::worker::{QueryRequest, run_worker};
use crate::quantizer::Quantizer;
use std::path::PathBuf;
use std::thread;
use rand::prelude::*;
use futures::channel::oneshot;
use futures::future::join_all;
use std::collections::HashMap;

fn main() -> anyhow::Result<()> {
    env_logger::init();

    // Configuration
    let num_shards = 2;
    let data_path = PathBuf::from("./satori_data");

    // Clean start
    if data_path.exists() {
        std::fs::remove_dir_all(&data_path).ok();
    }

    // Create Channels
    let mut senders = Vec::new();
    let mut worker_handles = Vec::new();

    for i in 0..num_shards {
        let (sender, receiver) = async_channel::bounded(100);
        senders.push(sender);
        
        let path_clone = data_path.clone();
        let handle = thread::spawn(move || {
            let builder = LocalExecutorBuilder::new(Placement::Unbound)
                .name(&format!("worker-{}", i));
            
            let result = builder.make().expect("failed to create executor")
                .run(run_worker(i, receiver, path_clone));
            
            result
        });
        worker_handles.push(handle);
    }

    // Run Dispatcher (Main Thread)
    let builder = LocalExecutorBuilder::new(Placement::Unbound).name("dispatcher");
    builder.make().map_err(|e| anyhow::anyhow!("Failed to create dispatcher: {:?}", e))?
        .run(async move {
        info!("Dispatcher started. Generating data...");

        // 1. Data Generation & Indexing (Performed by Dispatcher for simplicity)
        // In a real system, this might be distributed or a separate process.
        let storage = Storage::new(&data_path);
        
        let mut rng = rand::thread_rng();
        let mut vectors = Vec::new();
        let dim = 128;
        let num_vectors = 1000;
        let k = 10; // More clusters to distribute across shards

        for i in 0..num_vectors {
            let center_base = (i % k) as f32 * 10.0; 
            let data: Vec<f32> = (0..dim).map(|_| center_base + rng.gen::<f32>()).collect();
            vectors.push(Vector::new(i as u64, data));
        }

        let buckets = Indexer::build_clusters(vectors, k as usize);
        info!("Indexer created {} buckets.", buckets.len());

        // Compute Quantization Bounds
        let centroids: Vec<Vec<f32>> = buckets.iter().map(|b| b.centroid.clone()).collect();
        let (min, max) = Quantizer::compute_bounds(&centroids);
        info!("Quantizer bounds: min={:.4}, max={:.4}", min, max);
        let quantizer = Quantizer::new(min, max);

        let mut router = Router::new(1000, quantizer);

        for bucket in &buckets {
            storage.put(bucket).await.expect("Put failed");
            router.add_centroid(bucket.id, &bucket.centroid);
        }
        info!("Data persisted and indexed.");

        // 1.5. Dynamic Insert Test
        info!("Inserting dynamic vector (ID 9999) at [0.0...]");
        let new_vec_data = vec![0.0f32; dim];
        let new_vec = Vector::new(9999, new_vec_data.clone());
        
        // Find closest bucket
        let target_bucket_ids = router.query(&new_vec.data, 1).expect("Router query failed");
        let target_id = target_bucket_ids[0];
        info!("Assigning new vector to Bucket {}", target_id);
        
        // Load, Update, Save (with Balancing)
        let mut buckets = storage.get(&[target_id]).await.expect("Failed to load bucket");
        let mut bucket = buckets.pop().unwrap();
        bucket.add_vector(new_vec);
        
        if bucket.vectors.len() > 100 { // Low threshold to force split
            info!("Bucket {} too large ({} vectors). Splitting...", target_id, bucket.vectors.len());
            let new_parts = Indexer::split_bucket(bucket);
            
            // Part 1 inherits old ID
            let mut p1 = new_parts[0].clone(); 
            p1.id = target_id;
            storage.put(&p1).await.expect("Failed to save split bucket 1");
            
            // Part 2 gets new ID
            let mut p2 = new_parts[1].clone(); 
            p2.id = 20000; // Hardcoded ID for demo
            storage.put(&p2).await.expect("Failed to save split bucket 2");
            router.add_centroid(p2.id, &p2.centroid);
            
            info!("Split complete. Bucket {} kept, Bucket {} created.", p1.id, p2.id);
        } else {
            storage.put(&bucket).await.expect("Failed to save updated bucket");
            info!("Bucket {} updated with new vector.", target_id);
        }

        // 2. Query Dispatching
        let query_vec: Vec<f32> = vec![0.0; dim]; // Exact match for inserted vector
        info!("Dispatching query for exact 0.0 vector...");

        // Step A: Route
        let bucket_ids = router.query(&query_vec, 3).expect("Router query failed");
        info!("Router selected buckets: {:?}", bucket_ids);

        // Step B: Shard & Send
        let mut pending_replies = Vec::new();
        let mut requests_by_shard: HashMap<usize, Vec<u64>> = HashMap::new();

        for &bid in &bucket_ids {
            let shard_id = (bid as usize) % num_shards;
            requests_by_shard.entry(shard_id).or_default().push(bid);
        }

        for (shard_id, bids) in requests_by_shard {
            let (reply_tx, reply_rx) = oneshot::channel();
            let req = QueryRequest {
                query_vec: query_vec.clone(),
                bucket_ids: bids,
                respond_to: reply_tx,
            };

            info!("Sending request to Shard {} for buckets {:?}", shard_id, req.bucket_ids);
            if let Err(e) = senders[shard_id].send(req).await {
                log::error!("Failed to send to shard {}: {:?}", shard_id, e);
            } else {
                pending_replies.push(reply_rx);
            }
        }

        // Step C: Aggregate
        let responses = join_all(pending_replies).await;
        let mut all_results = Vec::new();

        for res in responses {
            match res {
                Ok(Ok(candidates)) => all_results.extend(candidates),
                Ok(Err(e)) => log::error!("Worker returned error: {:?}", e),
                Err(e) => log::error!("Worker channel canceled: {:?}", e),
            }
        }

        // Sort and Top-K
        all_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_k = 5;
        if all_results.len() > top_k {
            all_results.truncate(top_k);
        }

        info!("Final Aggregated Results:");
        for (id, dist) in all_results {
            info!("  ID: {}, Dist: {:.4}", id, dist);
        }

    });

    // Shutdown: Drop senders to close channels, allowing workers to exit
    // senders were moved into closure, so they are dropped when closure returns.
    // Explicit drop not needed here as variable `senders` is moved.

    // Wait for workers
    for h in worker_handles {
        if let Err(e) = h.join() {
            log::error!("Worker thread panicked: {:?}", e);
        }
    }

    info!("System shutdown complete.");
    Ok(())
}