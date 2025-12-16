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
    assert!(!stats_before.ready, "router should not be ready before first upsert");

    // After first upsert, router is ready
    handle.upsert_blocking(1, vec![1.0, 2.0], None)?;
    let stats_after = handle.stats_blocking();
    assert!(stats_after.ready, "router should be ready after first upsert");

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
    assert!(result.is_err(), "query on empty database should return error");

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
