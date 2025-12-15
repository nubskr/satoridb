use std::sync::Arc;

use satoridb::wal::runtime::Walrus;
use satoridb::wal::{FsyncSchedule, ReadConsistency};
use satoridb::{SatoriDb, SatoriDbConfig};

#[test]
fn routing_and_data_survive_restart() {
    let tmp = tempfile::tempdir().expect("tempdir");
    std::env::set_var("WALRUS_DATA_DIR", tmp.path());
    std::env::set_var("WALRUS_QUIET", "1");

    let key = format!(
        "restart-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );

    {
        let wal = Arc::new(
            Walrus::with_consistency_and_schedule_for_key(
                &key,
                ReadConsistency::StrictlyAtOnce,
                FsyncSchedule::NoFsync,
            )
            .expect("walrus"),
        );
        let mut cfg = SatoriDbConfig::new(wal);
        cfg.workers = 1;
        let db = SatoriDb::start(cfg).expect("start");
        let api = db.handle();

        api.upsert_blocking(1, vec![0.0, 0.0], Some(1))
            .expect("upsert b1");
        api.upsert_blocking(2, vec![10.0, 10.0], Some(2))
            .expect("upsert b2");

        db.shutdown().expect("shutdown");
    }

    {
        let wal = Arc::new(
            Walrus::with_consistency_and_schedule_for_key(
                &key,
                ReadConsistency::StrictlyAtOnce,
                FsyncSchedule::NoFsync,
            )
            .expect("walrus"),
        );
        // Verify routing metadata made it to the WAL before we even start the DB.
        let snap_size = wal.get_topic_size("router_snapshot");
        let upd_size = wal.get_topic_size("router_updates");
        assert!(snap_size > 0, "router_snapshot empty after restart");
        assert!(upd_size > 0, "router_updates empty after restart");
        let snap_entries = wal
            .batch_read_for_topic("router_snapshot", snap_size as usize + 1024, false, Some(0))
            .expect("read router_snapshot");
        let upd_entries = wal
            .batch_read_for_topic("router_updates", upd_size as usize + 1024, false, Some(0))
            .expect("read router_updates");
        assert!(
            !snap_entries.is_empty(),
            "router_snapshot has size but no entries"
        );
        assert!(
            !upd_entries.is_empty(),
            "router_updates has size but no entries"
        );

        let mut cfg = SatoriDbConfig::new(wal);
        cfg.workers = 1;
        let db = SatoriDb::start(cfg).expect("start");
        let api = db.handle();

        let near_10 = api.query_blocking(vec![10.0, 10.0], 1, 1).expect("query");
        assert_eq!(near_10.len(), 1);
        assert_eq!(near_10[0].0, 2);

        let near_0 = api.query_blocking(vec![0.0, 0.0], 1, 1).expect("query");
        assert_eq!(near_0.len(), 1);
        assert_eq!(near_0[0].0, 1);

        db.shutdown().expect("shutdown");
    }
}
