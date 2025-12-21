use std::sync::Arc;

use satoridb::wal::runtime::Walrus;
use satoridb::wal::{FsyncSchedule, ReadConsistency};
use satoridb::{SatoriDb, SatoriDbConfig};

#[test]
fn api_tour_example_works() {
    // Use a temporary directory for isolation, similar to other tests.
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
    cfg.workers = 2;
    let db = SatoriDb::start(cfg).expect("db start");
    let api = db.handle();

    futures::executor::block_on(async {
        // 1. Upsert
        // ID 1: [1.0, 0.0, 0.0]
        // ID 2: [0.0, 1.0, 0.0]
        // ID 3: [0.0, 0.0, 1.0]
        let vectors = vec![
            (1, vec![1.0, 0.0, 0.0]),
            (2, vec![0.0, 1.0, 0.0]),
            (3, vec![0.0, 0.0, 1.0]),
        ];

        for (id, vec) in &vectors {
            api.upsert(*id, vec.clone(), None).await.expect("upsert");
        }

        // 2. Query (Basic)
        // Search near [1.0, 0.1, 0.1]. This should match ID 1 best.
        let query_vec = vec![1.0, 0.1, 0.1];
        let results = api.query(query_vec.clone(), 2, 10).await.expect("query");
        assert!(!results.is_empty(), "expected query results");
        assert_eq!(results[0].0, 1, "expected ID 1 to be the top result");

        // 3. Query with Vectors (Inline Payload)
        let results_with_data = api
            .query_with_vectors(query_vec, 1, 10)
            .await
            .expect("query with vectors");
        assert_eq!(results_with_data.len(), 1, "expected top_k=1");
        let (id, _, vec) = &results_with_data[0];
        assert_eq!(*id, 1);
        assert_eq!(*vec, vec![1.0, 0.0, 0.0], "expected stored vector data");

        // 4. Fetch Vectors by ID (Global Index)
        // Retrieve ID 2 and 3 directly via the persistent RocksDB index.
        let fetched = api
            .fetch_vectors_by_id(vec![2, 3])
            .await
            .expect("fetch vectors by id");
        assert_eq!(fetched.len(), 2, "expected 2 vectors fetched");

        let fetched_map: std::collections::HashMap<_, _> = fetched.into_iter().collect();
        assert_eq!(fetched_map.get(&2), Some(&vec![0.0, 1.0, 0.0]));
        assert_eq!(fetched_map.get(&3), Some(&vec![0.0, 0.0, 1.0]));

        // 5. Resolve Buckets & Fetch from Worker (Low-level)
        let locations = api.resolve_buckets_by_id(vec![1]).await.expect("resolve");
        assert_eq!(locations.len(), 1, "expected to resolve 1 ID");
        let (id, bucket_id) = locations[0];
        assert_eq!(id, 1);

        // Fetch directly from that bucket's worker.
        let worker_fetched = api
            .fetch_vectors(bucket_id, vec![id])
            .await
            .expect("fetch from worker");
        assert_eq!(worker_fetched.len(), 1);
        assert_eq!(worker_fetched[0].0, 1);
        assert_eq!(worker_fetched[0].1, vec![1.0, 0.0, 0.0]);

        // 6. Delete
        api.delete(2).await.expect("delete");
        // Verify delete
        let fetched_after = api
            .fetch_vectors_by_id(vec![2])
            .await
            .expect("fetch deleted");
        assert!(fetched_after.is_empty(), "expected ID 2 to be deleted");

        // 7. Flush & Stats
        api.flush().await.expect("flush");
        let stats = api.stats().await;
        assert!(stats.buckets > 0, "expected buckets");
        // Total vectors stat might lag or not reflect delete instantly depending on implementation detail of stats
        // Router stats come from RouterManager which might not be aware of deletes in this mode yet.
        // But let's check it doesn't panic.
        assert!(stats.total_vectors > 0, "expected some vectors");
    });

    db.shutdown().expect("shutdown");
}
