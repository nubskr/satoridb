use satoridb::wal::runtime::Walrus;
use satoridb::SatoriDb;

#[test]
fn routing_and_data_survive_restart() {
    let tmp = tempfile::tempdir().expect("tempdir");

    // First session: insert data
    {
        let db = SatoriDb::builder("test")
            .workers(1)
            .data_dir(tmp.path())
            .build()
            .expect("start");

        db.insert(1, vec![0.0, 0.0]).expect("insert 1");
        db.insert(2, vec![10.0, 10.0]).expect("insert 2");

        db.shutdown().expect("shutdown");
    }

    // Verify WAL has data before restart
    {
        let wal = Walrus::with_data_dir(tmp.path().to_path_buf()).expect("walrus");
        let snap_size = wal.get_topic_size("router_snapshot");
        let upd_size = wal.get_topic_size("router_updates");
        assert!(snap_size > 0, "router_snapshot empty after restart");
        assert!(upd_size > 0, "router_updates empty after restart");
    }

    // Second session: verify data persists
    {
        let db = SatoriDb::builder("test")
            .workers(1)
            .data_dir(tmp.path())
            .build()
            .expect("start");

        let near_10 = db.query(vec![10.0, 10.0], 1).expect("query");
        assert_eq!(near_10.len(), 1);
        assert_eq!(near_10[0].0, 2);

        let near_0 = db.query(vec![0.0, 0.0], 1).expect("query");
        assert_eq!(near_0.len(), 1);
        assert_eq!(near_0[0].0, 1);
    }
}
