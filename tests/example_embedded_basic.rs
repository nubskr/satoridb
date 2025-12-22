use satoridb::SatoriDb;

#[test]
fn embedded_basic_example_works() {
    let tmp = tempfile::tempdir().unwrap();

    let db = SatoriDb::builder("test")
        .workers(2)
        .data_dir(tmp.path())
        .build()
        .expect("db start");

    db.insert(1, vec![0.0, 0.0, 0.0]).expect("insert 1");
    db.insert(2, vec![1.0, 1.0, 1.0]).expect("insert 2");

    let results = db.query(vec![0.1, 0.1, 0.1], 10).expect("query");
    assert!(
        results.iter().any(|(id, _)| *id == 1),
        "expected id=1 in results, got: {:?}",
        results
    );
}
