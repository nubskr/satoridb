use satoridb::SatoriDb;

#[test]
fn embedded_insert_then_query_smoke() {
    let tmp = tempfile::tempdir().expect("tempdir");

    let db = SatoriDb::builder("test")
        .workers(1)
        .data_dir(tmp.path())
        .build()
        .expect("start");

    db.insert(42, vec![1.0, 2.0, 3.0]).expect("insert");
    let results = db.query(vec![1.0, 2.0, 3.0], 10).expect("query");
    assert!(results.iter().any(|(id, _)| *id == 42));
}
