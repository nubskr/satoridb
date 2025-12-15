use std::sync::Arc;

use satoridb::wal::runtime::Walrus;
use satoridb::wal::{FsyncSchedule, ReadConsistency};
use satoridb::{SatoriDb, SatoriDbConfig};

#[test]
fn embedded_basic_example_works() {
    let tmp = tempfile::tempdir().unwrap();
    std::env::set_var("WALRUS_DATA_DIR", tmp.path());
    std::env::set_var("WALRUS_QUIET", "1");

    let wal = Arc::new(
        Walrus::with_consistency_and_schedule_for_key(
            "embedded_basic_example_works",
            ReadConsistency::StrictlyAtOnce,
            FsyncSchedule::NoFsync,
        )
        .expect("walrus init"),
    );

    let mut cfg = SatoriDbConfig::new(wal);
    cfg.workers = 2;
    let db = SatoriDb::start(cfg).expect("db start");
    let api = db.handle();

    api.upsert_blocking(1, vec![0.0, 0.0, 0.0], None)
        .expect("upsert 1");
    api.upsert_blocking(2, vec![1.0, 1.0, 1.0], None)
        .expect("upsert 2");

    let results = api
        .query_blocking(vec![0.1, 0.1, 0.1], 10, 200)
        .expect("query");
    assert!(
        results.iter().any(|(id, _)| *id == 1),
        "expected id=1 in results, got: {:?}",
        results
    );

    db.shutdown().expect("shutdown");
}

