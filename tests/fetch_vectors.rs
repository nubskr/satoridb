use satoridb::SatoriDb;

#[test]
fn fetch_vectors_by_id() {
    let tmp = tempfile::tempdir().unwrap();

    let db = SatoriDb::builder("test")
        .workers(1)
        .data_dir(tmp.path())
        .build()
        .expect("db start");

    db.insert(1, vec![0.0, 1.0, 2.0]).expect("insert 1");
    db.insert(2, vec![3.0, 4.0, 5.0]).expect("insert 2");
    db.insert(3, vec![9.0, 9.0, 9.0]).expect("insert 3");

    // Fetch vectors by ID
    let found = db.get(vec![2, 3, 999]).expect("get");

    assert_eq!(found.len(), 2, "should find 2 vectors (999 doesn't exist)");

    // Check that we got the right vectors
    let ids: Vec<u64> = found.iter().map(|(id, _)| *id).collect();
    assert!(ids.contains(&2));
    assert!(ids.contains(&3));
}
