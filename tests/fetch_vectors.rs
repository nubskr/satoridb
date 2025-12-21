use std::sync::Arc;

use satoridb::wal::runtime::Walrus;
use satoridb::wal::{FsyncSchedule, ReadConsistency};
use satoridb::{SatoriDb, SatoriDbConfig};

#[test]
fn fetch_vectors_by_bucket_and_id() {
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

    api.upsert_blocking(1, vec![0.0, 1.0, 2.0], Some(99))
        .expect("upsert 1");
    api.upsert_blocking(2, vec![3.0, 4.0, 5.0], Some(99))
        .expect("upsert 2");
    api.upsert_blocking(3, vec![9.0, 9.0, 9.0], Some(100))
        .expect("upsert other bucket");

    let found = api
        .fetch_vectors_blocking(99, vec![2, 3, 999])
        .expect("fetch");

    assert_eq!(found.len(), 1, "only id=2 in bucket 99 should return");
    assert_eq!(found[0].0, 2);
    assert_eq!(found[0].1, vec![3.0, 4.0, 5.0]);

    db.shutdown().expect("shutdown");
}
