use std::sync::Arc;

use futures::executor::block_on;
use satoridb::wal::runtime::Walrus;
use satoridb::wal::{FsyncSchedule, ReadConsistency};
use satoridb::{SatoriDb, SatoriDbConfig};

fn run_example(assertions: bool) -> anyhow::Result<Vec<(u64, f32)>> {
    // Customize WAL durability and router topology for this example.
    let wal = Arc::new(Walrus::with_consistency_and_schedule_for_key(
        "embedded_async",
        ReadConsistency::AtLeastOnce { persist_every: 1 },
        FsyncSchedule::NoFsync,
    )?);

    let mut cfg = SatoriDbConfig::new(wal);
    cfg.workers = 2;
    cfg.virtual_nodes = 16;

    // Start DB synchronously
    let db = SatoriDb::start(cfg)?;
    let api = db.handle();

    // Run async API calls
    let results = block_on(async {
        // Async API: upsert with a fixed bucket, plus a router-chosen bucket.
        api.upsert(10, vec![0.2, 0.2, 0.2], Some(99)).await?;
        api.upsert(11, vec![0.9, 0.9, 0.9], None).await?;

        // Route to more buckets by bumping router_top_k for better recall.
        let results = api.query(vec![0.1, 0.1, 0.1], 5, 16).await?;

        // Flush worker state and fetch router stats.
        api.flush().await?;
        let stats = api.stats().await;

        if assertions {
            let ids: std::collections::HashSet<u64> = results.iter().map(|(id, _)| *id).collect();
            assert!(ids.contains(&10), "expected query to return id 10");
            assert!(ids.contains(&11), "expected query to return id 11");
            assert!(stats.buckets >= 1);
        }

        Ok::<_, anyhow::Error>(results)
    })?;

    // Shutdown synchronously
    db.shutdown()?;
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use tempfile::tempdir;

    static LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn example_runs_and_returns_inserted_ids() {
        let _g = LOCK.lock().unwrap();
        let dir = tempdir().unwrap();
        std::env::set_var("WALRUS_DATA_DIR", dir.path());
        let results = run_example(true).expect("embedded_async example should run");
        let ids: std::collections::HashSet<u64> = results.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&10) && ids.contains(&11));
    }
}

fn main() -> anyhow::Result<()> {
    let results = run_example(false)?;
    println!("async query results={:?}", results);
    Ok(())
}
