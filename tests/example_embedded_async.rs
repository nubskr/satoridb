use std::sync::Arc;

use futures::executor::block_on;
use satoridb::wal::runtime::Walrus;
use satoridb::wal::{FsyncSchedule, ReadConsistency};
use satoridb::{SatoriDb, SatoriDbConfig};

#[test]
fn embedded_async_example_works() {
    let tmp = tempfile::tempdir().unwrap();
    std::env::set_var("WALRUS_QUIET", "1");

    let wal = Arc::new(
        Walrus::with_data_dir_and_options(
            tmp.path().to_path_buf(),
            ReadConsistency::AtLeastOnce { persist_every: 1 },
            FsyncSchedule::NoFsync,
        )
        .expect("walrus init"),
    );

    let mut cfg = SatoriDbConfig::new(wal);
    cfg.workers = 2;
    cfg.virtual_nodes = 8;

    let db = SatoriDb::start(cfg).expect("db start");
    let api = db.handle();

    block_on(async {
        api.upsert(42, vec![0.05, 0.05, 0.05], Some(7))
            .await
            .expect("upsert with hint");
        api.upsert(43, vec![1.0, 1.0, 1.0], None)
            .await
            .expect("upsert routed");

        let results = api.query(vec![0.0, 0.0, 0.0], 5, 8).await.expect("query");
        assert!(
            results.iter().any(|(id, _)| *id == 42),
            "expected hinted id in results, got {:?}",
            results
        );

        api.flush().await.expect("flush");
        let stats = api.stats().await;
        assert!(stats.buckets >= 1, "expected at least one bucket");
        assert!(
            stats.total_vectors >= 2,
            "expected total vectors to reflect inserts, got {}",
            stats.total_vectors
        );
        assert_eq!(
            stats.pending_updates, 0,
            "flush should clear pending updates"
        );
    });

    db.shutdown().expect("shutdown");
}
