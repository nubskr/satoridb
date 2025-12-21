use std::sync::Arc;

use futures::executor::block_on;
use satoridb::wal::runtime::Walrus;
use satoridb::wal::{FsyncSchedule, ReadConsistency};
use satoridb::{SatoriDb, SatoriDbConfig};

#[allow(dead_code)]
struct TourOutputs {
    query_results: Vec<(u64, f32)>,
    query_with_vectors: Vec<(u64, f32, Vec<f32>)>,
    fetched_by_id: Vec<(u64, Vec<f32>)>,
    bucket_fetch: Vec<(u64, Vec<f32>)>,
    fetched_after_delete: Vec<(u64, Vec<f32>)>,
}

fn run_example(assertions: bool) -> anyhow::Result<TourOutputs> {
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
    let outputs = block_on(async {
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
        for (id, dist, vec) in &results_with_data {
            println!("  - ID: {}, Distance: {:.4}, Vector: {:?}", id, dist, vec);
        }

        // 2d. Fetch Vectors by ID
        let fetched = api.fetch_vectors_by_id(vec![2, 3]).await?;
        println!("> Fetch by ID (Global) results: {} found", fetched.len());
        for (id, data) in &fetched {
            println!("  - ID: {}, Vector: {:?}", id, data);
        }

        // 2e. Resolve Buckets & Fetch
        let locations = api.resolve_buckets_by_id(vec![1]).await?;
        let bucket_fetch = if let Some((id, bucket_id)) = locations.first() {
            println!("> Resolved ID {} to Bucket {}", id, bucket_id);
            let worker_fetched = api.fetch_vectors(*bucket_id, vec![*id]).await?;
            println!(
                "> Fetch from Worker (Bucket {}) result: {:?}",
                bucket_id, worker_fetched
            );
            worker_fetched
        } else {
            Vec::new()
        };

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

        if assertions {
            let ids: std::collections::HashSet<u64> = results.iter().map(|(id, _)| *id).collect();
            assert!(ids.contains(&1));
            assert_eq!(results_with_data.len(), 1);
            assert_eq!(results_with_data[0].0, 1);
            assert_eq!(results_with_data[0].2, vec![1.0, 0.0, 0.0]);
            let fetched_ids: std::collections::HashSet<u64> =
                fetched.iter().map(|(id, _)| *id).collect();
            assert!(fetched_ids.contains(&2) && fetched_ids.contains(&3));
            if !bucket_fetch.is_empty() {
                assert_eq!(bucket_fetch[0].0, 1);
            }
            assert!(fetched_after.is_empty());
            assert!(stats.buckets >= 1);
            assert!(stats.total_vectors >= 2);
        }

        Ok::<_, anyhow::Error>(TourOutputs {
            query_results: results,
            query_with_vectors: results_with_data,
            fetched_by_id: fetched,
            bucket_fetch,
            fetched_after_delete: fetched_after,
        })
    })?;

    // 3. Shutdown (Sync)
    db.shutdown()?;
    println!("> Shutdown complete.");
    Ok(outputs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use tempfile::tempdir;

    static LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn example_runs_and_validates_outputs() {
        let _g = LOCK.lock().unwrap();
        let dir = tempdir().unwrap();
        std::env::set_var("WALRUS_DATA_DIR", dir.path());
        let outputs = run_example(true).expect("api_tour example should run");
        let ids: std::collections::HashSet<u64> =
            outputs.query_results.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&1));
        assert!(outputs.fetched_after_delete.is_empty());
        if !outputs.bucket_fetch.is_empty() {
            assert_eq!(outputs.bucket_fetch[0].0, 1);
        }
        assert!(outputs
            .query_with_vectors
            .iter()
            .any(|(id, _, vec)| *id == 1 && vec == &vec![1.0, 0.0, 0.0]));
        let fetched_ids: std::collections::HashSet<u64> =
            outputs.fetched_by_id.iter().map(|(id, _)| *id).collect();
        assert!(fetched_ids.contains(&2) && fetched_ids.contains(&3));
    }
}

fn main() -> anyhow::Result<()> {
    run_example(false).map(|_| ())
}
