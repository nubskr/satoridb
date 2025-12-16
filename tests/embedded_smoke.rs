use std::sync::Arc;

use satoridb::wal::runtime::Walrus;
use satoridb::wal::{FsyncSchedule, ReadConsistency};
use satoridb::{SatoriDb, SatoriDbConfig};

#[test]
fn embedded_upsert_then_query_smoke() {
    let tmp = tempfile::tempdir().expect("tempdir");
    std::env::set_var("WALRUS_QUIET", "1");

    let wal = Arc::new(
        Walrus::with_data_dir_and_options(
            tmp.path().to_path_buf(),
            ReadConsistency::StrictlyAtOnce,
            FsyncSchedule::NoFsync,
        )
        .expect("walrus"),
    );

    let mut cfg = SatoriDbConfig::new(wal);
    cfg.workers = 1;
    let db = SatoriDb::start(cfg).expect("start");
    let api = db.handle();

    api.upsert_blocking(42, vec![1.0, 2.0, 3.0], None)
        .expect("upsert");
    let results = api
        .query_blocking(vec![1.0, 2.0, 3.0], 10, 1)
        .expect("query");
    assert!(results.iter().any(|(id, _)| *id == 42));

    db.shutdown().expect("shutdown");
}
