mod bvecs;
mod executor;
mod gnd;
mod indexer;
mod router;
mod storage;
mod worker;
mod quantizer;

use glommio::{LocalExecutorBuilder, Placement};
use log::info;
use crate::storage::{Storage, Vector, Bucket};
use crate::router::Router;
use crate::indexer::Indexer;
use crate::worker::{QueryRequest, WorkerMessage, run_worker};
use crate::quantizer::Quantizer;
use crate::bvecs::BvecsReader;
use crate::gnd::GndReader;
use std::path::PathBuf;
use std::thread;
use rand::prelude::*;
use futures::channel::oneshot;
use futures::future::join_all;
use std::collections::HashMap;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    env_logger::init();

    // Configuration
    let num_shards = 15; 
    let data_path = PathBuf::from("./satori_data");
    let base_file = "bigann_base.bvecs.gz";
    let query_file = "bigann_query.bvecs.gz";
    let gnd_file = "bigann_gnd.tar.gz";

    // Clean start
    if data_path.exists() {
        std::fs::remove_dir_all(&data_path).ok();
    }

    // Create Channels
    let mut senders = Vec::new();
    let mut worker_handles = Vec::new();

    for i in 0..num_shards {
        let (sender, receiver) = async_channel::bounded(1000);
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
        info!("Dispatcher started with {} shards. Target: 100M Vectors.", num_shards);

        let storage = Storage::new(&data_path);
        let mut router: Option<Router> = None;
        let max_vectors = 100_000_000; // 100 Million

        if std::path::Path::new(base_file).exists() {
            // --- BIGANN BENCHMARK MODE ---
            info!("Found {}, running BigANN Benchmark Mode.", base_file);
            
            let mut reader = BvecsReader::new(base_file).expect("Failed to open base file");
            let initial_batch_size = 100_000;
            let stream_batch_size = 100_000;

            // 1. Initial Batch
            info!("Reading initial batch of {} vectors...", initial_batch_size);
            let vectors = reader.read_batch(initial_batch_size).expect("Failed to read batch");
            
            if vectors.is_empty() {
                panic!("Base file is empty!");
            }

            // Index
            let k = 1000; 
            let buckets = Indexer::build_clusters(vectors, k);
            info!("Indexer created {} buckets.", buckets.len());

            // Quantizer & Router
            let centroids: Vec<Vec<f32>> = buckets.iter().map(|b| b.centroid.clone()).collect();
            let (min, max) = Quantizer::compute_bounds(&centroids);
            info!("Quantizer bounds: min={:.4}, max={:.4}", min, max);
            let quantizer = Quantizer::new(min, max);
            
            let mut r = Router::new(100_000, quantizer);

            for bucket in &buckets {
                storage.put_chunk(bucket).await.expect("Put failed");
                r.add_centroid(bucket.id, &bucket.centroid);
            }
            router = Some(r);
            info!("Initial indexing complete.");

            // 2. Streaming Ingestion
            let mut total_processed = initial_batch_size;
            
            while total_processed < max_vectors {
                let batch = reader.read_batch(stream_batch_size).expect("Failed to read stream batch");
                if batch.is_empty() { break; }
                
                info!("Ingesting batch (total: {})...", total_processed);
                
                let mut updates: HashMap<u64, Vec<Vector>> = HashMap::new();
                let r_ref = router.as_ref().unwrap();

                for vec in batch {
                    if let Ok(ids) = r_ref.query(&vec.data, 1) {
                        if !ids.is_empty() {
                            updates.entry(ids[0]).or_default().push(vec);
                        }
                    }
                }

                for (bucket_id, new_vectors) in updates {
                    let shard = (bucket_id as usize) % num_shards;
                    let msg = WorkerMessage::Ingest {
                        bucket_id,
                        vectors: new_vectors,
                    };
                    // Backpressure applied here if workers are slow
                    if let Err(e) = senders[shard].send(msg).await {
                        log::error!("Failed to send ingest to shard {}: {:?}", shard, e);
                    }
                }
                total_processed += stream_batch_size;
            }
            info!("Ingestion complete. Total vectors: {}", total_processed);

        } else {
            info!("{} not found. Skipping Benchmark Ingestion.", base_file);
        }

        // 3. Query & Recall
        if std::path::Path::new(query_file).exists() {
            info!("Running queries from {}...", query_file);
            let mut reader = BvecsReader::new(query_file).expect("Failed to open query file");
            let queries = reader.read_batch(100).expect("Failed to read queries");
            
            let ground_truth = if std::path::Path::new(gnd_file).exists() {
                info!("Loading Ground Truth from {}...", gnd_file);
                Some(GndReader::new(gnd_file).expect("Failed to load GT"))
            } else {
                info!("Ground Truth {} not found. Recall will be skipped.", gnd_file);
                None
            };

            let start = Instant::now();
            let r_ref = router.as_ref().unwrap();
            let mut total_hits_at_10 = 0;
            let mut total_queries_checked = 0;
            let mut total_reachable_gt = 0;
            
            for (i, q) in queries.iter().enumerate() {
                let bucket_ids = r_ref.query(&q.data, 50).unwrap_or_default();
                
                let mut pending = Vec::new();
                let mut requests: HashMap<usize, Vec<u64>> = HashMap::new();
                for &bid in &bucket_ids {
                    let shard = (bid as usize) % num_shards;
                    requests.entry(shard).or_default().push(bid);
                }
                
                for (shard, bids) in requests {
                    let (tx, rx) = oneshot::channel();
                    let req = QueryRequest {
                        query_vec: q.data.clone(),
                        bucket_ids: bids,
                        respond_to: tx,
                    };
                    let msg = WorkerMessage::Query(req);
                    if let Ok(_) = senders[shard].send(msg).await {
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
                
                if let Some(ref gt) = ground_truth {
                    if i < gt.ground_truth.len() {
                        let true_neighbors = &gt.ground_truth[i];
                        let gt_top_10 = true_neighbors.iter().take(10).cloned().collect::<Vec<_>>();
                        
                        let reachable = gt_top_10.iter().filter(|&&id| (id as usize) < max_vectors).count();
                        total_reachable_gt += reachable;

                        let my_top_10: Vec<u32> = all_results.iter().take(10).map(|r| r.0 as u32).collect();
                        let hits = my_top_10.iter().filter(|id| gt_top_10.contains(id)).count();
                        total_hits_at_10 += hits;
                        total_queries_checked += 1;
                    }
                }

                if i % 10 == 0 { info!("Processed {} queries", i); }
            }
            
            let duration = start.elapsed();
            info!("Benchmark Complete. Processed {} queries in {:.2?}. QPS: {:.2}", 
                  queries.len(), duration, queries.len() as f64 / duration.as_secs_f64());
            
            if total_queries_checked > 0 {
                let avg_hits = total_hits_at_10 as f64 / total_queries_checked as f64;
                let avg_reachable = total_reachable_gt as f64 / total_queries_checked as f64;
                let adjusted_recall = if total_reachable_gt > 0 {
                    total_hits_at_10 as f64 / total_reachable_gt as f64
                } else { 0.0 };
                
                info!("Recall Analysis:");
                info!("  Average Hits@10: {:.2}", avg_hits);
                info!("  Average Reachable GT@10: {:.2} (max 10)", avg_reachable);
                info!("  Adjusted Recall@10: {:.2}%", adjusted_recall * 100.0);
            }
        }

    });

    for h in worker_handles { h.join().ok(); }
    Ok(())
}
