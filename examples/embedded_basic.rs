use std::sync::Arc;

use satoridb::wal::runtime::Walrus;
use satoridb::wal::{FsyncSchedule, ReadConsistency};
use satoridb::{SatoriDb, SatoriDbConfig};

fn run_example(assertions: bool) -> anyhow::Result<Vec<(u64, f32)>> {
    // By default Walrus writes under `wal_files/<key>` (relative to the current working dir).
    // Set `WALRUS_DATA_DIR=/some/dir` to control the parent directory.
    let wal = Arc::new(Walrus::with_consistency_and_schedule_for_key(
        "embedded_basic",
        ReadConsistency::StrictlyAtOnce,
        FsyncSchedule::Milliseconds(200),
    )?);

    let mut cfg = SatoriDbConfig::new(wal);
    cfg.workers = 4;
    let db = SatoriDb::start(cfg)?;
    let api = db.handle();

    api.upsert_blocking(1, vec![0.0, 0.0, 0.0], None)?;
    api.upsert_blocking(2, vec![1.0, 1.0, 1.0], None)?;

    let results = api.query_blocking(vec![0.1, 0.1, 0.1], 10, 200)?;
    if assertions {
        let ids: std::collections::HashSet<u64> = results.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&1) && ids.contains(&2));
    }

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
    fn example_runs_and_queries_both_vectors() {
        let _g = LOCK.lock().unwrap();
        let dir = tempdir().unwrap();
        std::env::set_var("WALRUS_DATA_DIR", dir.path());
        let results = run_example(true).expect("embedded_basic example should run");
        let ids: std::collections::HashSet<u64> = results.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&1) && ids.contains(&2));
    }
}

fn main() -> anyhow::Result<()> {
    let results = run_example(false)?;
    println!("results={:?}", results);
    Ok(())
}
