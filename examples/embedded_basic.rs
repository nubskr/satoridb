use std::sync::Arc;

use satoridb::wal::runtime::Walrus;
use satoridb::wal::{FsyncSchedule, ReadConsistency};
use satoridb::{SatoriDb, SatoriDbConfig};

fn main() -> anyhow::Result<()> {
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
    println!("results={:?}", results);

    db.shutdown()?;
    Ok(())
}
