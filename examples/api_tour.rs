use std::sync::Arc;

use futures::executor::block_on;
use satoridb::wal::runtime::Walrus;
use satoridb::wal::{FsyncSchedule, ReadConsistency};
use satoridb::{SatoriDb, SatoriDbConfig};

fn main() -> anyhow::Result<()> {
    println!("--- SatoriDB API Tour ---");

    // 1. Setup (Sync)
    // Walrus and SatoriDb::start use blocking operations internally.
    let wal = Arc::new(Walrus::with_consistency_and_schedule_for_key(
        "api_tour",
        ReadConsistency::StrictlyAtOnce,
        FsyncSchedule::NoFsync,
    )?);
    let mut cfg = SatoriDbConfig::new(wal);
    cfg.workers = 2;
    let db = SatoriDb::start(cfg)?;
    let api = db.handle();

    println!("> Database started.");

    // 2. Async Operations
    // We enter the async context only for the API interactions.
    block_on(async {
        // 2a. Upsert
        let vectors = vec![
            (1, vec![1.0, 0.0, 0.0]),
            (2, vec![0.0, 1.0, 0.0]),
            (3, vec![0.0, 0.0, 1.0]),
        ];

        for (id, vec) in &vectors {
            api.upsert(*id, vec.clone(), None).await?;
        }
        println!("> Upserted {} vectors.", vectors.len());

        // 2b. Query (Basic)
        let query_vec = vec![1.0, 0.1, 0.1];
        let results = api.query(query_vec.clone(), 2, 10).await?;
        println!("> Query results (top_k=2):");
        for (id, dist) in &results {
            println!("  - ID: {}, Distance: {:.4}", id, dist);
        }

        // 2c. Query with Vectors
        let results_with_data = api.query_with_vectors(query_vec, 1, 10).await?;
        println!("> Query with vectors (top_k=1):");
        for (id, dist, vec) in results_with_data {
            println!("  - ID: {}, Distance: {:.4}, Vector: {:?}", id, dist, vec);
        }

        // 2d. Fetch Vectors by ID
        let fetched = api.fetch_vectors_by_id(vec![2, 3]).await?;
        println!("> Fetch by ID (Global) results: {} found", fetched.len());
        for (id, data) in fetched {
            println!("  - ID: {}, Vector: {:?}", id, data);
        }

        // 2e. Resolve Buckets & Fetch
        let locations = api.resolve_buckets_by_id(vec![1]).await?;
        if let Some((id, bucket_id)) = locations.first() {
            println!("> Resolved ID {} to Bucket {}", id, bucket_id);
            let worker_fetched = api.fetch_vectors(*bucket_id, vec![*id]).await?;
            println!(
                "> Fetch from Worker (Bucket {}) result: {:?}",
                bucket_id, worker_fetched
            );
        }

        // 2f. Delete
        println!("> Deleting vector ID 2...");
        api.delete(2).await?;
        // Verify delete
        let fetched_after = api.fetch_vectors_by_id(vec![2]).await?;
        if fetched_after.is_empty() {
            println!("> Verified: Vector ID 2 not found.");
        } else {
            println!("> Warning: Vector ID 2 still exists!");
        }

        // 2g. Flush & Stats
        api.flush().await?;
        let stats = api.stats().await;
        println!("> Stats after flush: {:?}", stats);

        Ok::<(), anyhow::Error>(())
    })?;

    // 3. Shutdown (Sync)
    db.shutdown()?;
    println!("> Shutdown complete.");
    Ok(())
}
