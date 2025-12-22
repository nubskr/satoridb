use satoridb::SatoriDb;

#[test]
fn embedded_async_example_works() {
    let tmp = tempfile::tempdir().unwrap();

    let db = SatoriDb::builder("test")
        .workers(2)
        .virtual_nodes(8)
        .data_dir(tmp.path())
        .build()
        .expect("db start");

    futures::executor::block_on(async {
        db.insert_async(42, vec![0.05, 0.05, 0.05])
            .await
            .expect("insert 42");
        db.insert_async(43, vec![1.0, 1.0, 1.0])
            .await
            .expect("insert 43");

        let results = db.query_async(vec![0.0, 0.0, 0.0], 5).await.expect("query");
        assert!(
            results.iter().any(|(id, _)| *id == 42),
            "expected id 42 in results, got {:?}",
            results
        );
    });

    let stats = db.stats();
    assert!(stats.buckets >= 1, "expected at least one bucket");
    assert!(
        stats.vectors >= 2,
        "expected at least 2 vectors, got {}",
        stats.vectors
    );
}
