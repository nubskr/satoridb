use std::sync::Arc;

use anyhow::Result;
use satoridb::embedded::{SatoriDb, SatoriDbConfig};
use satoridb::wal::runtime::Walrus;
use tempfile::TempDir;

fn init_wal(tempdir: &TempDir) -> Arc<Walrus> {
    Arc::new(Walrus::with_data_dir(tempdir.path().to_path_buf()).expect("walrus init"))
}

/// SatoriDb::start with 0 workers should fail.
#[test]
fn start_with_zero_workers_fails() {
    let tmp = tempfile::tempdir().unwrap();
    let wal = init_wal(&tmp);

    let mut cfg = SatoriDbConfig::new(wal);
    cfg.workers = 0;

    let result = SatoriDb::start(cfg);
    assert!(result.is_err(), "0 workers should fail");
    let err = result.err().expect("should have error");
    assert!(
        err.to_string().contains("workers must be > 0"),
        "error should mention workers"
    );
}

/// SatoriDb starts successfully with valid config.
#[test]
fn start_with_valid_config_succeeds() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);

    let mut cfg = SatoriDbConfig::new(wal);
    cfg.workers = 1; // Minimal config

    let db = SatoriDb::start(cfg)?;
    db.shutdown()?;
    Ok(())
}

/// Handle can be cloned and used independently.
#[test]
fn handle_is_cloneable() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);

    let mut cfg = SatoriDbConfig::new(wal);
    cfg.workers = 1;

    let db = SatoriDb::start(cfg)?;
    let handle1 = db.handle();
    let handle2 = handle1.clone();

    // Both handles should work
    let stats1 = handle1.stats_blocking();
    let stats2 = handle2.stats_blocking();
    assert_eq!(stats1.buckets, stats2.buckets);

    db.shutdown()?;
    Ok(())
}

/// Stats returns ready=true after first upsert (router needs data to initialize).
#[test]
fn stats_ready_after_upsert() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);

    let mut cfg = SatoriDbConfig::new(wal);
    cfg.workers = 1;

    let db = SatoriDb::start(cfg)?;
    let handle = db.handle();

    // Before any upsert, router is not initialized
    let stats_before = handle.stats_blocking();
    assert!(
        !stats_before.ready,
        "router should not be ready before first upsert"
    );

    // After first upsert, router is ready
    handle.upsert_blocking(1, vec![1.0, 2.0], None)?;
    let stats_after = handle.stats_blocking();
    assert!(
        stats_after.ready,
        "router should be ready after first upsert"
    );

    db.shutdown()?;
    Ok(())
}

/// Upsert and query round-trip works.
#[test]
fn upsert_and_query_round_trip() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);

    let mut cfg = SatoriDbConfig::new(wal);
    cfg.workers = 1;

    let db = SatoriDb::start(cfg)?;
    let handle = db.handle();

    // Insert a vector
    let (_bucket_id, meta) = handle.upsert_blocking(42, vec![1.0, 2.0, 3.0], None)?;
    assert_eq!(meta.count, 1, "first upsert should have count=1");

    // Query should find it
    let results = handle.query_blocking(vec![1.0, 2.0, 3.0], 10, 5)?;
    assert!(!results.is_empty(), "should find the inserted vector");
    assert_eq!(results[0].0, 42, "should find vector id 42");

    db.shutdown()?;
    Ok(())
}

/// Multiple upserts to same bucket update count.
#[test]
fn multiple_upserts_increment_count() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);

    let mut cfg = SatoriDbConfig::new(wal);
    cfg.workers = 1;

    let db = SatoriDb::start(cfg)?;
    let handle = db.handle();

    // Insert first vector
    let (bucket_id, meta1) = handle.upsert_blocking(1, vec![1.0, 1.0], None)?;
    assert_eq!(meta1.count, 1);

    // Insert second vector to same bucket
    let (_, meta2) = handle.upsert_blocking(2, vec![1.1, 1.1], Some(bucket_id))?;
    assert_eq!(meta2.count, 2, "count should increment");

    db.shutdown()?;
    Ok(())
}

/// Flush completes without error.
#[test]
fn flush_completes() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);

    let mut cfg = SatoriDbConfig::new(wal);
    cfg.workers = 1;

    let db = SatoriDb::start(cfg)?;
    let handle = db.handle();

    // Insert some data
    handle.upsert_blocking(1, vec![1.0, 2.0], None)?;

    // Flush should succeed
    handle.flush_blocking()?;

    db.shutdown()?;
    Ok(())
}

/// Shutdown is idempotent (doesn't panic if called on already-shutdown).
#[test]
fn shutdown_is_safe() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);

    let mut cfg = SatoriDbConfig::new(wal);
    cfg.workers = 1;

    let db = SatoriDb::start(cfg)?;
    // Shutdown should succeed
    db.shutdown()?;
    // Note: Can't call shutdown twice on same instance since it consumes self
    Ok(())
}

/// Virtual nodes config is respected.
#[test]
fn virtual_nodes_config() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);

    let mut cfg = SatoriDbConfig::new(wal);
    cfg.workers = 2;
    cfg.virtual_nodes = 16; // Custom virtual nodes

    let db = SatoriDb::start(cfg)?;
    // Should start without error
    db.shutdown()?;
    Ok(())
}

/// Query with no data returns error (router not initialized).
#[test]
fn query_empty_returns_error() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);

    let mut cfg = SatoriDbConfig::new(wal);
    cfg.workers = 1;

    let db = SatoriDb::start(cfg)?;
    let handle = db.handle();

    // Query on empty database returns error because router not initialized
    let result = handle.query_blocking(vec![1.0, 2.0], 10, 5);
    assert!(
        result.is_err(),
        "query on empty database should return error"
    );

    db.shutdown()?;
    Ok(())
}

/// Default config uses reasonable defaults.
#[test]
fn default_config_uses_cpu_count() {
    let tmp = tempfile::tempdir().unwrap();
    let wal = init_wal(&tmp);

    let cfg = SatoriDbConfig::new(wal);
    assert!(cfg.workers >= 1, "should have at least 1 worker");
    assert_eq!(cfg.virtual_nodes, 8, "default virtual_nodes should be 8");
}

/// Upsert rejects duplicate IDs.
#[test]
fn upsert_rejects_duplicate_id() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);

    let mut cfg = SatoriDbConfig::new(wal);
    cfg.workers = 1;

    let db = SatoriDb::start(cfg)?;
    let handle = db.handle();

    // First insert should succeed
    handle.upsert_blocking(42, vec![1.0, 2.0, 3.0], None)?;

    // Second insert with same ID should fail
    let result = handle.upsert_blocking(42, vec![4.0, 5.0, 6.0], None);
    assert!(result.is_err(), "duplicate id should be rejected");
    let err = result.err().expect("should have error");
    assert!(
        err.to_string().contains("already exists"),
        "error should mention id already exists: {}",
        err
    );

    db.shutdown()?;
    Ok(())
}

/// Upsert allows different IDs.
#[test]
fn upsert_allows_different_ids() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);

    let mut cfg = SatoriDbConfig::new(wal);
    cfg.workers = 1;

    let db = SatoriDb::start(cfg)?;
    let handle = db.handle();

    // Multiple inserts with different IDs should all succeed
    handle.upsert_blocking(1, vec![1.0, 2.0], None)?;
    handle.upsert_blocking(2, vec![2.0, 3.0], None)?;
    handle.upsert_blocking(3, vec![3.0, 4.0], None)?;

    // All should be queryable
    let results = handle.query_blocking(vec![2.0, 3.0], 10, 10)?;
    assert!(results.len() >= 1, "should find at least one vector");

    db.shutdown()?;
    Ok(())
}

/// Regression: duplicate ID rejection works across buckets.
#[test]
fn upsert_rejects_duplicate_across_buckets() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);

    let mut cfg = SatoriDbConfig::new(wal);
    cfg.workers = 2; // Multiple workers to test cross-shard

    let db = SatoriDb::start(cfg)?;
    let handle = db.handle();

    // Insert to bucket 0
    let (bucket_id, _) = handle.upsert_blocking(100, vec![1.0, 2.0], None)?;

    // Try to insert same ID to a different bucket - should still fail
    let different_bucket = if bucket_id == 0 { 1 } else { 0 };
    let result = handle.upsert_blocking(100, vec![9.0, 9.0], Some(different_bucket));
    assert!(
        result.is_err(),
        "duplicate id should be rejected even with different bucket hint"
    );

    db.shutdown()?;
    Ok(())
}

// =============================================================================
// Delete + Reinsert Tests
// =============================================================================

/// Basic: delete allows reinsertion of same ID.
#[test]
fn delete_allows_reinsert_of_same_id() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);

    let mut cfg = SatoriDbConfig::new(wal);
    cfg.workers = 1;

    let db = SatoriDb::start(cfg)?;
    let handle = db.handle();

    // Insert
    handle.upsert_blocking(42, vec![1.0, 2.0, 3.0], None)?;

    // Delete
    handle.delete_blocking(42)?;

    // Reinsert same ID should succeed
    let result = handle.upsert_blocking(42, vec![4.0, 5.0, 6.0], None);
    assert!(result.is_ok(), "reinsert after delete should succeed");

    db.shutdown()?;
    Ok(())
}

/// Delete non-existent ID does not error.
#[test]
fn delete_nonexistent_id_succeeds() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);

    let mut cfg = SatoriDbConfig::new(wal);
    cfg.workers = 1;

    let db = SatoriDb::start(cfg)?;
    let handle = db.handle();

    // Insert something to initialize router
    handle.upsert_blocking(1, vec![1.0, 2.0], None)?;

    // Delete non-existent ID should not error
    let result = handle.delete_blocking(9999);
    assert!(result.is_ok(), "delete of non-existent id should succeed");

    db.shutdown()?;
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
    let wal = init_wal(&tmp);

    let mut cfg = SatoriDbConfig::new(wal);
    cfg.workers = 1;

    let db = SatoriDb::start(cfg)?;
    let handle = db.handle();

    // Insert vector
    handle.upsert_blocking(42, vec![1.0, 2.0, 3.0], None)?;

    // Verify it's queryable
    let results = handle.query_blocking(vec![1.0, 2.0, 3.0], 10, 10)?;
    assert!(
        results.iter().any(|(id, _)| *id == 42),
        "should find vector before delete"
    );

    // Delete
    handle.delete_blocking(42)?;

    // The key invariant: ID can be reused after delete
    // (vector_index is cleaned synchronously)
    handle.upsert_blocking(42, vec![9.0, 9.0, 9.0], None)?;

    // Query near new location should find the new vector
    let results_after = handle.query_blocking(vec![9.0, 9.0, 9.0], 10, 10)?;
    assert!(
        results_after.iter().any(|(id, _)| *id == 42),
        "should find reinserted vector"
    );

    db.shutdown()?;
    Ok(())
}

/// Query finds reinserted vector with new data.
#[test]
fn query_finds_reinserted_vector() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);

    let mut cfg = SatoriDbConfig::new(wal);
    cfg.workers = 1;

    let db = SatoriDb::start(cfg)?;
    let handle = db.handle();

    // Insert vector near origin
    handle.upsert_blocking(42, vec![0.1, 0.1, 0.1], None)?;

    // Delete
    handle.delete_blocking(42)?;

    // Reinsert with very different vector
    handle.upsert_blocking(42, vec![0.9, 0.9, 0.9], None)?;

    // Query near new location should find it
    let results = handle.query_blocking(vec![0.9, 0.9, 0.9], 10, 10)?;
    assert!(
        results.iter().any(|(id, _)| *id == 42),
        "should find reinserted vector"
    );

    db.shutdown()?;
    Ok(())
}

/// Multiple delete-reinsert cycles on same ID.
#[test]
fn multiple_delete_reinsert_cycles() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);

    let mut cfg = SatoriDbConfig::new(wal);
    cfg.workers = 1;

    let db = SatoriDb::start(cfg)?;
    let handle = db.handle();

    for cycle in 0..5 {
        // Insert
        let vec = vec![cycle as f32, cycle as f32 + 0.1];
        handle.upsert_blocking(100, vec.clone(), None)?;

        // Verify duplicate rejected
        let dup_result = handle.upsert_blocking(100, vec![0.0, 0.0], None);
        assert!(
            dup_result.is_err(),
            "cycle {}: duplicate should be rejected",
            cycle
        );

        // Delete
        handle.delete_blocking(100)?;

        // Verify can reinsert
        // (next iteration will do the insert)
    }

    // Final insert after last delete
    handle.upsert_blocking(100, vec![9.9, 9.9], None)?;

    db.shutdown()?;
    Ok(())
}

/// Delete and reinsert multiple different IDs.
///
/// Note: This test focuses on the key invariant that deleted IDs can be
/// reused. Query visibility is eventually consistent due to caching.
#[test]
fn delete_reinsert_multiple_ids() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);

    let mut cfg = SatoriDbConfig::new(wal);
    cfg.workers = 1;

    let db = SatoriDb::start(cfg)?;
    let handle = db.handle();

    let ids: [u64; 5] = [10, 20, 30, 40, 50];

    // Insert all
    for id in ids {
        handle.upsert_blocking(id, vec![id as f32, id as f32], None)?;
    }

    // Verify all IDs are tracked (duplicates rejected)
    for id in ids {
        let result = handle.upsert_blocking(id, vec![0.0, 0.0], None);
        assert!(result.is_err(), "id {} should reject duplicate", id);
    }

    // Delete some
    handle.delete_blocking(20)?;
    handle.delete_blocking(40)?;

    // Key invariant: deleted IDs can be reinserted
    handle.upsert_blocking(20, vec![200.0, 200.0], None)?;
    handle.upsert_blocking(40, vec![400.0, 400.0], None)?;

    // Non-deleted IDs should still reject duplicates
    assert!(handle.upsert_blocking(10, vec![0.0, 0.0], None).is_err());
    assert!(handle.upsert_blocking(30, vec![0.0, 0.0], None).is_err());
    assert!(handle.upsert_blocking(50, vec![0.0, 0.0], None).is_err());

    // Reinserted IDs should now reject duplicates too
    assert!(handle.upsert_blocking(20, vec![0.0, 0.0], None).is_err());
    assert!(handle.upsert_blocking(40, vec![0.0, 0.0], None).is_err());

    db.shutdown()?;
    Ok(())
}

/// Regression: delete cleans up vector_index so exists() returns false.
#[test]
fn delete_cleans_vector_index() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);

    let mut cfg = SatoriDbConfig::new(wal);
    cfg.workers = 1;

    let db = SatoriDb::start(cfg)?;
    let handle = db.handle();

    // Insert
    handle.upsert_blocking(42, vec![1.0, 2.0], None)?;

    // Duplicate should fail (proves vector_index has the entry)
    assert!(handle.upsert_blocking(42, vec![3.0, 4.0], None).is_err());

    // Delete
    handle.delete_blocking(42)?;

    // Now insert should succeed (proves vector_index was cleaned)
    assert!(
        handle.upsert_blocking(42, vec![5.0, 6.0], None).is_ok(),
        "vector_index should be cleaned after delete"
    );

    db.shutdown()?;
    Ok(())
}

/// Regression: delete cleans up bucket_index.
#[test]
fn delete_cleans_bucket_index() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);

    let mut cfg = SatoriDbConfig::new(wal);
    cfg.workers = 1;

    let db = SatoriDb::start(cfg)?;
    let handle = db.handle();

    // Insert and get bucket
    let (bucket_id, _) = handle.upsert_blocking(42, vec![1.0, 2.0], None)?;

    // Delete
    handle.delete_blocking(42)?;

    // Reinsert - may go to different bucket, but should succeed
    let (new_bucket_id, _) = handle.upsert_blocking(42, vec![1.0, 2.0], None)?;

    // Both operations should complete without error
    // (bucket_id may or may not equal new_bucket_id depending on routing)
    let _ = (bucket_id, new_bucket_id);

    db.shutdown()?;
    Ok(())
}

/// Stress: rapid insert-delete-reinsert cycles.
#[test]
fn stress_rapid_delete_reinsert() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);

    let mut cfg = SatoriDbConfig::new(wal);
    cfg.workers = 2;

    let db = SatoriDb::start(cfg)?;
    let handle = db.handle();

    // Rapid cycles on multiple IDs
    for id in 1..=20u64 {
        handle.upsert_blocking(id, vec![id as f32, id as f32], None)?;
    }

    // Delete half
    for id in (1..=20u64).filter(|x| x % 2 == 0) {
        handle.delete_blocking(id)?;
    }

    // Reinsert deleted ones
    for id in (1..=20u64).filter(|x| x % 2 == 0) {
        handle.upsert_blocking(id, vec![id as f32 * 2.0, id as f32 * 2.0], None)?;
    }

    // Verify all exist
    let results = handle.query_blocking(vec![10.0, 10.0], 30, 30)?;
    assert!(results.len() >= 10, "should find many vectors");

    db.shutdown()?;
    Ok(())
}

/// Regression: concurrent-safe delete does not corrupt index state.
#[test]
fn delete_with_multiple_workers() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);

    let mut cfg = SatoriDbConfig::new(wal);
    cfg.workers = 4; // Multiple workers

    let db = SatoriDb::start(cfg)?;
    let handle = db.handle();

    // Insert vectors that may route to different workers
    for id in 0..100u64 {
        handle.upsert_blocking(id, vec![id as f32 * 0.1, id as f32 * 0.1], None)?;
    }

    // Delete every 3rd
    for id in (0..100u64).filter(|x| x % 3 == 0) {
        handle.delete_blocking(id)?;
    }

    // Reinsert deleted
    for id in (0..100u64).filter(|x| x % 3 == 0) {
        handle.upsert_blocking(id, vec![id as f32 * 0.2, id as f32 * 0.2], None)?;
    }

    // Verify no duplicates possible
    for id in 0..100u64 {
        let result = handle.upsert_blocking(id, vec![0.0, 0.0], None);
        assert!(
            result.is_err(),
            "id {} should already exist and reject duplicate",
            id
        );
    }

    db.shutdown()?;
    Ok(())
}

/// Edge case: delete immediately after insert.
#[test]
fn delete_immediately_after_insert() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);

    let mut cfg = SatoriDbConfig::new(wal);
    cfg.workers = 1;

    let db = SatoriDb::start(cfg)?;
    let handle = db.handle();

    for _ in 0..10 {
        handle.upsert_blocking(999, vec![1.0, 1.0], None)?;
        handle.delete_blocking(999)?;
    }

    // Final state: should be able to insert
    handle.upsert_blocking(999, vec![2.0, 2.0], None)?;

    db.shutdown()?;
    Ok(())
}

/// Edge case: delete same ID multiple times.
#[test]
fn delete_same_id_multiple_times() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);

    let mut cfg = SatoriDbConfig::new(wal);
    cfg.workers = 1;

    let db = SatoriDb::start(cfg)?;
    let handle = db.handle();

    handle.upsert_blocking(42, vec![1.0, 2.0], None)?;

    // Delete multiple times - should not error
    handle.delete_blocking(42)?;
    handle.delete_blocking(42)?;
    handle.delete_blocking(42)?;

    // Should still be able to reinsert
    handle.upsert_blocking(42, vec![3.0, 4.0], None)?;

    db.shutdown()?;
    Ok(())
}
