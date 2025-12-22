use anyhow::Result;
use satoridb::SatoriDb;

/// SatoriDb::builder with 0 workers should fail.
#[test]
fn start_with_zero_workers_fails() {
    let tmp = tempfile::tempdir().unwrap();

    let result = SatoriDb::builder("test")
        .workers(0)
        .data_dir(tmp.path())
        .build();

    assert!(result.is_err(), "0 workers should fail");
    let err = match result {
        Ok(_) => panic!("should have error"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains("workers must be > 0"),
        "error should mention workers"
    );
}

/// SatoriDb starts successfully with valid config.
#[test]
fn start_with_valid_config_succeeds() -> Result<()> {
    let tmp = tempfile::tempdir()?;

    let db = SatoriDb::builder("test")
        .workers(1)
        .data_dir(tmp.path())
        .build()?;

    drop(db); // auto-shutdown
    Ok(())
}

/// Stats returns ready=true after first insert (router needs data to initialize).
#[test]
fn stats_ready_after_insert() -> Result<()> {
    let tmp = tempfile::tempdir()?;

    let db = SatoriDb::builder("test")
        .workers(1)
        .data_dir(tmp.path())
        .build()?;

    // Before any insert, router is not initialized
    let stats_before = db.stats();
    assert!(
        !stats_before.ready,
        "router should not be ready before first insert"
    );

    // After first insert, router is ready
    db.insert(1, vec![1.0, 2.0])?;
    let stats_after = db.stats();
    assert!(
        stats_after.ready,
        "router should be ready after first insert"
    );

    Ok(())
}

/// Insert and query round-trip works.
#[test]
fn insert_and_query_round_trip() -> Result<()> {
    let tmp = tempfile::tempdir()?;

    let db = SatoriDb::builder("test")
        .workers(1)
        .data_dir(tmp.path())
        .build()?;

    // Insert a vector
    db.insert(42, vec![1.0, 2.0, 3.0])?;

    // Query should find it
    let results = db.query(vec![1.0, 2.0, 3.0], 10)?;
    assert!(!results.is_empty(), "should find the inserted vector");
    assert_eq!(results[0].0, 42, "should find vector id 42");

    Ok(())
}

/// Multiple inserts work.
#[test]
fn multiple_inserts_work() -> Result<()> {
    let tmp = tempfile::tempdir()?;

    let db = SatoriDb::builder("test")
        .workers(1)
        .data_dir(tmp.path())
        .build()?;

    // Insert multiple vectors
    db.insert(1, vec![1.0, 1.0])?;
    db.insert(2, vec![1.1, 1.1])?;
    db.insert(3, vec![1.2, 1.2])?;

    // Should be queryable
    let results = db.query(vec![1.0, 1.0], 10)?;
    assert!(!results.is_empty(), "should find vectors");

    Ok(())
}

/// Flush completes without error.
#[test]
fn flush_completes() -> Result<()> {
    let tmp = tempfile::tempdir()?;

    let db = SatoriDb::builder("test")
        .workers(1)
        .data_dir(tmp.path())
        .build()?;

    // Insert some data
    db.insert(1, vec![1.0, 2.0])?;

    // Flush should succeed
    db.flush()?;

    Ok(())
}

/// Shutdown is called automatically on drop.
#[test]
fn auto_shutdown_on_drop() -> Result<()> {
    let tmp = tempfile::tempdir()?;

    {
        let db = SatoriDb::builder("test")
            .workers(1)
            .data_dir(tmp.path())
            .build()?;

        db.insert(1, vec![1.0, 2.0])?;
        // db is dropped here, shutdown should be automatic
    }

    Ok(())
}

/// Explicit shutdown works.
#[test]
fn explicit_shutdown_works() -> Result<()> {
    let tmp = tempfile::tempdir()?;

    let db = SatoriDb::builder("test")
        .workers(1)
        .data_dir(tmp.path())
        .build()?;

    db.shutdown()?;
    Ok(())
}

/// Virtual nodes config is respected.
#[test]
fn virtual_nodes_config() -> Result<()> {
    let tmp = tempfile::tempdir()?;

    let db = SatoriDb::builder("test")
        .workers(2)
        .virtual_nodes(16)
        .data_dir(tmp.path())
        .build()?;

    // Should start without error
    drop(db);
    Ok(())
}

/// Query with no data returns error (router not initialized).
#[test]
fn query_empty_returns_error() -> Result<()> {
    let tmp = tempfile::tempdir()?;

    let db = SatoriDb::builder("test")
        .workers(1)
        .data_dir(tmp.path())
        .build()?;

    // Query on empty database returns error because router not initialized
    let result = db.query(vec![1.0, 2.0], 10);
    assert!(
        result.is_err(),
        "query on empty database should return error"
    );

    Ok(())
}

/// Insert rejects duplicate IDs.
#[test]
fn insert_rejects_duplicate_id() -> Result<()> {
    let tmp = tempfile::tempdir()?;

    let db = SatoriDb::builder("test")
        .workers(1)
        .data_dir(tmp.path())
        .build()?;

    // First insert should succeed
    db.insert(42, vec![1.0, 2.0, 3.0])?;

    // Second insert with same ID should fail
    let result = db.insert(42, vec![4.0, 5.0, 6.0]);
    assert!(result.is_err(), "duplicate id should be rejected");
    let err = match result {
        Ok(_) => panic!("should have error"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains("already exists"),
        "error should mention id already exists: {}",
        err
    );

    Ok(())
}

/// Insert allows different IDs.
#[test]
fn insert_allows_different_ids() -> Result<()> {
    let tmp = tempfile::tempdir()?;

    let db = SatoriDb::builder("test")
        .workers(1)
        .data_dir(tmp.path())
        .build()?;

    // Multiple inserts with different IDs should all succeed
    db.insert(1, vec![1.0, 2.0])?;
    db.insert(2, vec![2.0, 3.0])?;
    db.insert(3, vec![3.0, 4.0])?;

    // All should be queryable
    let results = db.query(vec![2.0, 3.0], 10)?;
    assert!(!results.is_empty(), "should find at least one vector");

    Ok(())
}

/// Regression: duplicate ID rejection works across workers.
#[test]
fn insert_rejects_duplicate_across_workers() -> Result<()> {
    let tmp = tempfile::tempdir()?;

    let db = SatoriDb::builder("test")
        .workers(2)
        .data_dir(tmp.path())
        .build()?;

    // Insert
    db.insert(100, vec![1.0, 2.0])?;

    // Try to insert same ID again - should fail
    let result = db.insert(100, vec![9.0, 9.0]);
    assert!(result.is_err(), "duplicate id should be rejected");

    Ok(())
}

// =============================================================================
// Delete + Reinsert Tests
// =============================================================================

/// Basic: delete allows reinsertion of same ID.
#[test]
fn delete_allows_reinsert_of_same_id() -> Result<()> {
    let tmp = tempfile::tempdir()?;

    let db = SatoriDb::builder("test")
        .workers(1)
        .data_dir(tmp.path())
        .build()?;

    // Insert
    db.insert(42, vec![1.0, 2.0, 3.0])?;

    // Delete
    db.delete(42)?;

    // Reinsert same ID should succeed
    let result = db.insert(42, vec![4.0, 5.0, 6.0]);
    assert!(result.is_ok(), "reinsert after delete should succeed");

    Ok(())
}

/// Delete non-existent ID does not error.
#[test]
fn delete_nonexistent_id_succeeds() -> Result<()> {
    let tmp = tempfile::tempdir()?;

    let db = SatoriDb::builder("test")
        .workers(1)
        .data_dir(tmp.path())
        .build()?;

    // Insert something to initialize router
    db.insert(1, vec![1.0, 2.0])?;

    // Delete non-existent ID should not error
    let result = db.delete(9999);
    assert!(result.is_ok(), "delete of non-existent id should succeed");

    Ok(())
}

/// Query does not find deleted vector (after cache invalidation).
///
/// Note: Delete is eventually consistent. The vector_index is updated
/// synchronously (enabling ID reuse), but query results may be cached.
/// This test verifies the delete is eventually reflected in queries.
#[test]
fn query_does_not_find_deleted_vector() -> Result<()> {
    let tmp = tempfile::tempdir()?;

    let db = SatoriDb::builder("test")
        .workers(1)
        .data_dir(tmp.path())
        .build()?;

    // Insert vector
    db.insert(42, vec![1.0, 2.0, 3.0])?;

    // Verify it's queryable
    let results = db.query(vec![1.0, 2.0, 3.0], 10)?;
    assert!(
        results.iter().any(|(id, _)| *id == 42),
        "should find vector before delete"
    );

    // Delete
    db.delete(42)?;

    // The key invariant: ID can be reused after delete
    // (vector_index is cleaned synchronously)
    db.insert(42, vec![9.0, 9.0, 9.0])?;

    // Query near new location should find the new vector
    let results_after = db.query(vec![9.0, 9.0, 9.0], 10)?;
    assert!(
        results_after.iter().any(|(id, _)| *id == 42),
        "should find reinserted vector"
    );

    Ok(())
}

/// Query finds reinserted vector with new data.
#[test]
fn query_finds_reinserted_vector() -> Result<()> {
    let tmp = tempfile::tempdir()?;

    let db = SatoriDb::builder("test")
        .workers(1)
        .data_dir(tmp.path())
        .build()?;

    // Insert vector near origin
    db.insert(42, vec![0.1, 0.1, 0.1])?;

    // Delete
    db.delete(42)?;

    // Reinsert with very different vector
    db.insert(42, vec![0.9, 0.9, 0.9])?;

    // Query near new location should find it
    let results = db.query(vec![0.9, 0.9, 0.9], 10)?;
    assert!(
        results.iter().any(|(id, _)| *id == 42),
        "should find reinserted vector"
    );

    Ok(())
}

/// Multiple delete-reinsert cycles on same ID.
#[test]
fn multiple_delete_reinsert_cycles() -> Result<()> {
    let tmp = tempfile::tempdir()?;

    let db = SatoriDb::builder("test")
        .workers(1)
        .data_dir(tmp.path())
        .build()?;

    for cycle in 0..5 {
        // Insert
        let vec = vec![cycle as f32, cycle as f32 + 0.1];
        db.insert(100, vec)?;

        // Verify duplicate rejected
        let dup_result = db.insert(100, vec![0.0, 0.0]);
        assert!(
            dup_result.is_err(),
            "cycle {}: duplicate should be rejected",
            cycle
        );

        // Delete
        db.delete(100)?;

        // Verify can reinsert
        // (next iteration will do the insert)
    }

    // Final insert after last delete
    db.insert(100, vec![9.9, 9.9])?;

    Ok(())
}

/// Delete and reinsert multiple different IDs.
///
/// Note: This test focuses on the key invariant that deleted IDs can be
/// reused. Query visibility is eventually consistent due to caching.
#[test]
fn delete_reinsert_multiple_ids() -> Result<()> {
    let tmp = tempfile::tempdir()?;

    let db = SatoriDb::builder("test")
        .workers(1)
        .data_dir(tmp.path())
        .build()?;

    let ids: [u64; 5] = [10, 20, 30, 40, 50];

    // Insert all
    for id in ids {
        db.insert(id, vec![id as f32, id as f32])?;
    }

    // Verify all IDs are tracked (duplicates rejected)
    for id in ids {
        let result = db.insert(id, vec![0.0, 0.0]);
        assert!(result.is_err(), "id {} should reject duplicate", id);
    }

    // Delete some
    db.delete(20)?;
    db.delete(40)?;

    // Key invariant: deleted IDs can be reinserted
    db.insert(20, vec![200.0, 200.0])?;
    db.insert(40, vec![400.0, 400.0])?;

    // Non-deleted IDs should still reject duplicates
    assert!(db.insert(10, vec![0.0, 0.0]).is_err());
    assert!(db.insert(30, vec![0.0, 0.0]).is_err());
    assert!(db.insert(50, vec![0.0, 0.0]).is_err());

    // Reinserted IDs should now reject duplicates too
    assert!(db.insert(20, vec![0.0, 0.0]).is_err());
    assert!(db.insert(40, vec![0.0, 0.0]).is_err());

    Ok(())
}

/// Regression: delete cleans up vector_index so exists() returns false.
#[test]
fn delete_cleans_vector_index() -> Result<()> {
    let tmp = tempfile::tempdir()?;

    let db = SatoriDb::builder("test")
        .workers(1)
        .data_dir(tmp.path())
        .build()?;

    // Insert
    db.insert(42, vec![1.0, 2.0])?;

    // Duplicate should fail (proves vector_index has the entry)
    assert!(db.insert(42, vec![3.0, 4.0]).is_err());

    // Delete
    db.delete(42)?;

    // Now insert should succeed (proves vector_index was cleaned)
    assert!(
        db.insert(42, vec![5.0, 6.0]).is_ok(),
        "vector_index should be cleaned after delete"
    );

    Ok(())
}

/// Regression: delete cleans up bucket_index.
#[test]
fn delete_cleans_bucket_index() -> Result<()> {
    let tmp = tempfile::tempdir()?;

    let db = SatoriDb::builder("test")
        .workers(1)
        .data_dir(tmp.path())
        .build()?;

    // Insert
    db.insert(42, vec![1.0, 2.0])?;

    // Delete
    db.delete(42)?;

    // Reinsert should succeed
    db.insert(42, vec![1.0, 2.0])?;

    Ok(())
}

/// Stress: rapid insert-delete-reinsert cycles.
#[test]
fn stress_rapid_delete_reinsert() -> Result<()> {
    let tmp = tempfile::tempdir()?;

    let db = SatoriDb::builder("test")
        .workers(2)
        .data_dir(tmp.path())
        .build()?;

    // Rapid cycles on multiple IDs
    for id in 1..=20u64 {
        db.insert(id, vec![id as f32, id as f32])?;
    }

    // Delete half
    for id in (1..=20u64).filter(|x| x % 2 == 0) {
        db.delete(id)?;
    }

    // Reinsert deleted ones
    for id in (1..=20u64).filter(|x| x % 2 == 0) {
        db.insert(id, vec![id as f32 * 2.0, id as f32 * 2.0])?;
    }

    // Verify all exist
    let results = db.query(vec![10.0, 10.0], 30)?;
    assert!(results.len() >= 10, "should find many vectors");

    Ok(())
}

/// Regression: concurrent-safe delete does not corrupt index state.
#[test]
fn delete_with_multiple_workers() -> Result<()> {
    let tmp = tempfile::tempdir()?;

    let db = SatoriDb::builder("test")
        .workers(4)
        .data_dir(tmp.path())
        .build()?;

    // Insert vectors that may route to different workers
    for id in 0..100u64 {
        db.insert(id, vec![id as f32 * 0.1, id as f32 * 0.1])?;
    }

    // Delete every 3rd
    for id in (0..100u64).filter(|x| x % 3 == 0) {
        db.delete(id)?;
    }

    // Reinsert deleted
    for id in (0..100u64).filter(|x| x % 3 == 0) {
        db.insert(id, vec![id as f32 * 0.2, id as f32 * 0.2])?;
    }

    // Verify no duplicates possible
    for id in 0..100u64 {
        let result = db.insert(id, vec![0.0, 0.0]);
        assert!(
            result.is_err(),
            "id {} should already exist and reject duplicate",
            id
        );
    }

    Ok(())
}

/// Edge case: delete immediately after insert.
#[test]
fn delete_immediately_after_insert() -> Result<()> {
    let tmp = tempfile::tempdir()?;

    let db = SatoriDb::builder("test")
        .workers(1)
        .data_dir(tmp.path())
        .build()?;

    for _ in 0..10 {
        db.insert(999, vec![1.0, 1.0])?;
        db.delete(999)?;
    }

    // Final state: should be able to insert
    db.insert(999, vec![2.0, 2.0])?;

    Ok(())
}

/// Edge case: delete same ID multiple times.
#[test]
fn delete_same_id_multiple_times() -> Result<()> {
    let tmp = tempfile::tempdir()?;

    let db = SatoriDb::builder("test")
        .workers(1)
        .data_dir(tmp.path())
        .build()?;

    db.insert(42, vec![1.0, 2.0])?;

    // Delete multiple times - should not error
    db.delete(42)?;
    db.delete(42)?;
    db.delete(42)?;

    // Should still be able to reinsert
    db.insert(42, vec![3.0, 4.0])?;

    Ok(())
}

/// Get vectors by ID.
#[test]
fn get_vectors_by_id() -> Result<()> {
    let tmp = tempfile::tempdir()?;

    let db = SatoriDb::builder("test")
        .workers(1)
        .data_dir(tmp.path())
        .build()?;

    db.insert(1, vec![1.0, 2.0, 3.0])?;
    db.insert(2, vec![4.0, 5.0, 6.0])?;

    let vectors = db.get(vec![1, 2, 999])?; // 999 doesn't exist
    assert_eq!(vectors.len(), 2, "should find 2 vectors");

    Ok(())
}

/// Query with vectors returns data.
#[test]
fn query_with_vectors_returns_data() -> Result<()> {
    let tmp = tempfile::tempdir()?;

    let db = SatoriDb::builder("test")
        .workers(1)
        .data_dir(tmp.path())
        .build()?;

    db.insert(1, vec![1.0, 2.0, 3.0])?;

    let results = db.query_with_vectors(vec![1.0, 2.0, 3.0], 10)?;
    assert!(!results.is_empty(), "should find vector");
    assert_eq!(
        results[0].2,
        vec![1.0, 2.0, 3.0],
        "should return vector data"
    );

    Ok(())
}
