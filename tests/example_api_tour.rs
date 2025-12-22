use satoridb::SatoriDb;

#[test]
fn api_tour_example_works() {
    let tmp = tempfile::tempdir().unwrap();

    let db = SatoriDb::builder("test")
        .workers(2)
        .data_dir(tmp.path())
        .build()
        .expect("db start");

    // 1. Insert vectors
    let vectors = vec![
        (1, vec![1.0, 0.0, 0.0]),
        (2, vec![0.0, 1.0, 0.0]),
        (3, vec![0.0, 0.0, 1.0]),
    ];

    for (id, vec) in &vectors {
        db.insert(*id, vec.clone()).expect("insert");
    }

    // 2. Query (Basic)
    // Search near [1.0, 0.1, 0.1]. This should match ID 1 best.
    let query_vec = vec![1.0, 0.1, 0.1];
    let results = db.query(query_vec.clone(), 2).expect("query");
    assert!(!results.is_empty(), "expected query results");
    assert_eq!(results[0].0, 1, "expected ID 1 to be the top result");

    // 3. Query with Vectors (Inline Payload)
    let results_with_data = db
        .query_with_vectors(query_vec, 1)
        .expect("query with vectors");
    assert_eq!(results_with_data.len(), 1, "expected top_k=1");
    let (id, _, vec) = &results_with_data[0];
    assert_eq!(*id, 1);
    assert_eq!(*vec, vec![1.0, 0.0, 0.0], "expected stored vector data");

    // 4. Get Vectors by ID
    let fetched = db.get(vec![2, 3]).expect("get");
    assert_eq!(fetched.len(), 2, "expected 2 vectors fetched");

    let fetched_map: std::collections::HashMap<_, _> = fetched.into_iter().collect();
    assert_eq!(fetched_map.get(&2), Some(&vec![0.0, 1.0, 0.0]));
    assert_eq!(fetched_map.get(&3), Some(&vec![0.0, 0.0, 1.0]));

    // 5. Delete
    db.delete(2).expect("delete");

    // Verify delete - ID should be reusable
    db.insert(2, vec![0.5, 0.5, 0.0])
        .expect("reinsert after delete");

    // 6. Stats
    let stats = db.stats();
    assert!(stats.buckets > 0, "expected buckets");
    assert!(stats.vectors > 0, "expected some vectors");
}
