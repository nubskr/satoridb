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

#[test]
fn oversized_bucket_is_not_cached() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let storage = Storage::new(wal);

    // Construct a bucket whose serialized size far exceeds the cache byte limit.
    // Each vector serializes to 8 (len prefix) + 16 (id+len) + 4*dim bytes.
    // With dim=1024, that's ~8 + 16 + 4096 = 4120 bytes per vector.
    // Two vectors already exceed a 2 KB limit.
    let mut big_vectors = Vec::new();
    for i in 0..2u64 {
        let data = vec![1.0f32; 1024];
        big_vectors.push(Vector::new(i, data));
    }
    block_on(storage.put_chunk_raw(0, &big_vectors))?;

    let cache = WorkerCache::new(4, 2 * 1024); // 2 KB max per bucket; bucket 0 should not cache.
    let executor = Executor::new(storage.clone(), cache);

    let q = vec![0.0f32; 1024];
    let first = block_on(executor.query(&q, &[0], 10, 0, Arc::new(Vec::new())))?;
    assert_eq!(first.len(), 2);

    // Append another vector to bucket 0. If the bucket were cached, the next query with the same
    // routing version would still see only the old contents. Since it's too large to cache, we
    // expect the new vector to appear even without a routing bump.
    let new_vec = Vector::new(99, vec![2.0f32; 1024]);
    block_on(storage.put_chunk_raw(0, &[new_vec]))?;
    let second = block_on(executor.query(&q, &[0], 10, 0, Arc::new(Vec::new())))?;
    let ids: Vec<u64> = second.iter().map(|(id, _)| *id).collect();
    assert!(
        ids.contains(&99),
        "bucket 0 should be reloaded on each query when over the cache byte limit"
    );

    Ok(())
}

#[test]
fn small_bucket_stays_cached_until_version_bump() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let storage = Storage::new(wal);

    let v0 = Vector::new(1, vec![0.0f32; 4]);
    block_on(storage.put_chunk_raw(1, &[v0]))?;

    let cache = WorkerCache::new(4, 16 * 1024);
    let executor = Executor::new(storage.clone(), cache);

    let q = vec![0.0f32; 4];
    let initial = block_on(executor.query(&q, &[1], 10, 0, Arc::new(Vec::new())))?;
    assert_eq!(initial.len(), 1);

    // Append another vector but do NOT bump routing. Because bucket 1 is small and cached,
    // the next query with the same routing version should still return only the cached content.
    let v1 = Vector::new(2, vec![1.0f32; 4]);
    block_on(storage.put_chunk_raw(1, &[v1]))?;
    let still_cached = block_on(executor.query(&q, &[1], 10, 0, Arc::new(Vec::new())))?;
    assert_eq!(
        still_cached.len(),
        1,
        "cache should serve stale contents when routing_version is unchanged"
    );

    // Now bump routing and mark bucket 1 changed; the executor should reload and see both vectors.
    let refreshed = block_on(executor.query(&q, &[1], 10, 1, Arc::new(vec![1])))?;
    assert_eq!(
        refreshed.len(),
        2,
        "after routing bump + changed bucket, executor should reload fresh data"
    );

    Ok(())
}
