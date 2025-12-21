use std::sync::Arc;

use satoridb::wal::runtime::Walrus;
use satoridb::wal::{FsyncSchedule, ReadConsistency};
use satoridb::{SatoriDb, SatoriDbConfig};

#[test]
fn query_can_return_vectors_inline() {
    let tmp = tempfile::tempdir().unwrap();
    std::env::set_var("WALRUS_QUIET", "1");

    let wal = Arc::new(
        Walrus::with_data_dir_and_options(
            tmp.path().to_path_buf(),
            ReadConsistency::StrictlyAtOnce,
            FsyncSchedule::NoFsync,
        )
        .expect("walrus init"),
    );

    let mut cfg = SatoriDbConfig::new(wal);
    cfg.workers = 1;

    let db = SatoriDb::start(cfg).expect("db start");
    let api = db.handle();

    api.upsert_blocking(10, vec![0.0, 0.0], Some(7))
        .expect("upsert 10");
    api.upsert_blocking(11, vec![1.0, 1.0], Some(7))
        .expect("upsert 11");

    let results = api
        .query_with_vectors_blocking(vec![0.0, 0.0], 2, 1)
        .expect("query");

    assert_eq!(results.len(), 2);
    let mut returned = std::collections::HashMap::new();
    for (id, _dist, vec) in results {
        returned.insert(id, vec);
    }
    assert_eq!(returned.get(&10).unwrap(), &vec![0.0, 0.0]);
    assert_eq!(returned.get(&11).unwrap(), &vec![1.0, 1.0]);

    db.shutdown().expect("shutdown");
}
