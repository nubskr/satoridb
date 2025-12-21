use std::sync::Arc;

use futures::executor::block_on;
use satoridb::wal::runtime::Walrus;
use satoridb::wal::{FsyncSchedule, ReadConsistency};
use satoridb::{SatoriDb, SatoriDbConfig};

fn main() -> anyhow::Result<()> {
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
    block_on(async {
        // Async API: upsert with a fixed bucket, plus a router-chosen bucket.
        api.upsert(10, vec![0.2, 0.2, 0.2], Some(99)).await?;
        api.upsert(11, vec![0.9, 0.9, 0.9], None).await?;

        // Route to more buckets by bumping router_top_k for better recall.
        let results = api.query(vec![0.1, 0.1, 0.1], 5, 16).await?;
        println!("async query results={:?}", results);

        // Flush worker state and fetch router stats.
        api.flush().await?;
        let stats = api.stats().await;
        println!("router stats after flush: {:?}", stats);

        Ok::<(), anyhow::Error>(())
    })?;

    // Shutdown synchronously
    db.shutdown()?;
    Ok(())
}
