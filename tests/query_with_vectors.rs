use satoridb::SatoriDb;

#[test]
fn query_can_return_vectors_inline() {
    let tmp = tempfile::tempdir().unwrap();

    let db = SatoriDb::builder("test")
        .workers(1)
        .data_dir(tmp.path())
        .build()
        .expect("db start");

    db.insert(10, vec![0.0, 0.0]).expect("insert 10");
    db.insert(11, vec![1.0, 1.0]).expect("insert 11");

    let results = db.query_with_vectors(vec![0.0, 0.0], 2).expect("query");

    assert_eq!(results.len(), 2);
    let mut returned = std::collections::HashMap::new();
    for (id, _dist, vec) in results {
        returned.insert(id, vec);
    }
    assert_eq!(returned.get(&10).unwrap(), &vec![0.0, 0.0]);
    assert_eq!(returned.get(&11).unwrap(), &vec![1.0, 1.0]);
}
