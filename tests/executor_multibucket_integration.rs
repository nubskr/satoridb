use std::sync::Arc;

use anyhow::Result;
use futures::executor::block_on;
use satoridb::executor::{Executor, WorkerCache};
use satoridb::storage::{Storage, Vector};
use satoridb::wal::runtime::Walrus;
use tempfile::TempDir;

fn init_wal(tempdir: &TempDir) -> Arc<Walrus> {
    Arc::new(Walrus::with_data_dir(tempdir.path().to_path_buf()).expect("walrus init"))
}

/// Ensure the executor merges results across buckets and returns the global top-k.
#[test]
fn executor_across_buckets_returns_global_topk() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let storage = Storage::new(wal);

    // Far bucket: vectors should never make it into the top-3.
    let far = vec![
        Vector::new(1, vec![100.0, 100.0]),
        Vector::new(2, vec![101.0, 101.0]),
    ];
    block_on(storage.put_chunk_raw(0, &far))?;

    // Near bucket: these should dominate the ranking.
    let near = vec![
        Vector::new(10, vec![1.0, 1.0]), // distance 0.0
        Vector::new(11, vec![2.0, 2.0]), // larger distance
        Vector::new(12, vec![0.5, 0.5]), // distance 0.5
    ];
    block_on(storage.put_chunk_raw(1, &near))?;

    let cache = WorkerCache::new(4, 64 * 1024, usize::MAX);
    let executor = Executor::new(storage.clone(), cache);

    let query = vec![1.0, 1.0];
    let results = block_on(executor.query(&query, &[0, 1], 3, 0, Arc::new(Vec::new()), false))?;
    let ids: Vec<u64> = results.iter().map(|(id, _, _)| *id).collect();

    assert_eq!(
        ids,
        vec![10, 12, 11],
        "executor should merge per-bucket heaps and return global top-k ordering"
    );

    Ok(())
}

/// Querying a non-existent bucket should be harmless and return no results.
#[test]
fn executor_handles_missing_bucket_gracefully() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let storage = Storage::new(wal);

    let cache = WorkerCache::new(2, 8 * 1024, usize::MAX);
    let executor = Executor::new(storage.clone(), cache);

    let query = vec![0.0f32, 0.0f32];
    let res = block_on(executor.query(&query, &[42], 5, 0, Arc::new(Vec::new()), false))?;
    assert!(
        res.is_empty(),
        "missing buckets should not cause errors and return empty results"
    );
    Ok(())
}
