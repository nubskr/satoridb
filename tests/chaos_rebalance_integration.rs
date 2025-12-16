use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use satoridb::executor::{
    clear_executor_fail_load_hook, set_executor_fail_load_hook, Executor, WorkerCache,
};
use satoridb::rebalancer::{
    clear_rebalance_fail_hook, set_rebalance_fail_hook, RebalanceTask, RebalanceTaskKind,
    RebalanceWorker,
};
use satoridb::router::RoutingTable;
use satoridb::storage::{Bucket, Storage, Vector};
use satoridb::wal::runtime::Walrus;
use tempfile::TempDir;

fn init_wal(tempdir: &TempDir) -> Arc<Walrus> {
    std::env::set_var("WALRUS_DATA_DIR", tempdir.path());
    Arc::new(Walrus::new().expect("walrus init"))
}

fn wait_until(timeout: Duration, mut f: impl FnMut() -> bool) -> bool {
    let start = Instant::now();
    while start.elapsed() < timeout {
        if f() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    false
}

#[test]
fn rebalance_survives_split_failures() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let storage = Storage::new(wal);
    let routing = Arc::new(RoutingTable::new());
    let worker = RebalanceWorker::spawn(storage.clone(), routing.clone(), None);

    // Seed one bucket with enough vectors to split.
    let mut bucket = Bucket::new(0, vec![0.0, 0.0]);
    for i in 0..32u64 {
        bucket.add_vector(Vector::new(i, vec![i as f32, i as f32 + 0.5]));
    }
    futures::executor::block_on(storage.put_chunk(&bucket))?;
    futures::executor::block_on(worker.prime_centroids(&[bucket.clone()]))?;

    // Fail every third split attempt to simulate flaky work.
    let counter = Arc::new(AtomicUsize::new(0));
    let c = counter.clone();
    set_rebalance_fail_hook(move |kind| {
        if kind != RebalanceTaskKind::Split {
            return false;
        }
        let n = c.fetch_add(1, Ordering::SeqCst);
        n.is_multiple_of(3)
    });

    for _ in 0..6 {
        worker
            .enqueue_blocking(RebalanceTask::Split(bucket.id))
            .expect("enqueue split");
    }

    let ok = wait_until(Duration::from_secs(4), || {
        worker.snapshot_sizes().len() >= 2 && routing.current_version() > 0
    });
    clear_rebalance_fail_hook();
    assert!(ok, "rebalance should progress despite injected failures");
    Ok(())
}

#[test]
fn rebalance_merges_with_flaky_tasks() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let storage = Storage::new(wal);
    let routing = Arc::new(RoutingTable::new());
    let worker = RebalanceWorker::spawn(storage.clone(), routing.clone(), None);

    // Two small buckets to merge.
    let mut a = Bucket::new(0, vec![0.0, 0.0]);
    a.add_vector(Vector::new(0, vec![0.0, 0.0]));
    a.add_vector(Vector::new(1, vec![0.1, 0.1]));
    let mut b = Bucket::new(1, vec![1.0, 1.0]);
    b.add_vector(Vector::new(2, vec![1.0, 1.0]));
    b.add_vector(Vector::new(3, vec![1.1, 1.2]));
    futures::executor::block_on(storage.put_chunk(&a))?;
    futures::executor::block_on(storage.put_chunk(&b))?;
    futures::executor::block_on(worker.prime_centroids(&[a.clone(), b.clone()]))?;

    let counter = Arc::new(AtomicUsize::new(0));
    let c = counter.clone();
    set_rebalance_fail_hook(move |kind| {
        if kind != RebalanceTaskKind::Merge {
            return false;
        }
        let n = c.fetch_add(1, Ordering::SeqCst);
        n.is_multiple_of(2)
    });

    for _ in 0..5 {
        worker
            .enqueue_blocking(RebalanceTask::Merge(a.id, b.id))
            .expect("enqueue merge");
    }

    let ok = wait_until(Duration::from_secs(4), || {
        let sizes = worker.snapshot_sizes();
        sizes.len() == 1 && routing.current_version() > 0
    });
    clear_rebalance_fail_hook();
    assert!(
        ok,
        "merge should eventually succeed even with intermittent failures"
    );
    Ok(())
}

#[test]
fn executor_cache_survives_version_churn() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let storage = Storage::new(wal);
    let cache = WorkerCache::new(32, 4 * 1024 * 1024);
    let executor = Arc::new(Executor::new(storage.clone(), cache));

    // Seed a few buckets.
    let mut buckets = Vec::new();
    for id in 0..4u64 {
        let mut b = Bucket::new(id, vec![id as f32, id as f32]);
        for j in 0..8u64 {
            b.add_vector(Vector::new(id * 100 + j, vec![j as f32, j as f32 + 1.0]));
        }
        futures::executor::block_on(storage.put_chunk(&b))?;
        buckets.push(b);
    }

    let version = Arc::new(AtomicUsize::new(1));
    let bucket_ids: Vec<u64> = buckets.iter().map(|b| b.id).collect();

    std::thread::scope(|s| {
        for t in 0..4 {
            let exec = executor.clone();
            let version = version.clone();
            let ids = bucket_ids.clone();
            s.spawn(move || {
                for i in 0..40 {
                    // Occasionally bump the router version and mark one bucket as changed.
                    let v = if i % 7 == 0 {
                        version.fetch_add(1, Ordering::SeqCst) + 1
                    } else {
                        version.load(Ordering::SeqCst)
                    };
                    let changed = if i % 5 == 0 {
                        Arc::new(vec![ids[(i + t) % ids.len()]])
                    } else {
                        Arc::new(Vec::new())
                    };
                    let q = vec![1.0, 2.0];
                    let res =
                        futures::executor::block_on(exec.query(&q, &ids, 4, v as u64, changed));
                    let res = res.expect("query succeeds");
                    assert!(
                        !res.is_empty(),
                        "query should return candidates even during churn"
                    );
                    for (_id, dist) in res {
                        assert!(dist.is_finite());
                    }
                }
            });
        }
    });

    // Final cache version should match the latest router version we tracked.
    let final_version = version.load(Ordering::SeqCst) as u64;
    assert_eq!(executor.cache_version(), final_version);
    Ok(())
}

#[test]
fn executor_load_failures_do_not_panic() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let storage = Storage::new(wal);
    let cache = WorkerCache::new(8, 2 * 1024 * 1024);
    let executor = Executor::new(storage.clone(), cache);

    // Seed one bucket.
    let mut b = Bucket::new(0, vec![0.0, 0.0]);
    b.add_vector(Vector::new(1, vec![1.0, 2.0]));
    futures::executor::block_on(storage.put_chunk(&b))?;

    // Fail the first two loads.
    let counter = Arc::new(AtomicUsize::new(0));
    let c = counter.clone();
    set_executor_fail_load_hook(move |_id| {
        let n = c.fetch_add(1, Ordering::SeqCst);
        n < 2
    });

    // Try a few times; first two loads are injected failures, the third should succeed.
    let mut last = Vec::new();
    for _ in 0..3 {
        let res = futures::executor::block_on(executor.query(
            &[1.0, 2.0],
            &[0],
            1,
            1,
            Arc::new(Vec::new()),
        ))?;
        if !res.is_empty() {
            last = res;
            break;
        }
    }
    clear_executor_fail_load_hook();
    // After failures, a subsequent load should succeed and return at least one candidate.
    assert!(!last.is_empty(), "eventual load should succeed");
    Ok(())
}

#[test]
fn rebalancer_handles_jittered_tasks() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let storage = Storage::new(wal);
    let routing = Arc::new(RoutingTable::new());
    let worker = RebalanceWorker::spawn(storage.clone(), routing.clone(), None);

    // Seed two buckets.
    let mut buckets = Vec::new();
    for id in 0..2u64 {
        let mut b = Bucket::new(id, vec![id as f32, id as f32]);
        for j in 0..16u64 {
            b.add_vector(Vector::new(id * 100 + j, vec![j as f32, j as f32 + 0.5]));
        }
        futures::executor::block_on(storage.put_chunk(&b))?;
        buckets.push(b);
    }
    futures::executor::block_on(worker.prime_centroids(&buckets))?;

    // Inject occasional failures.
    let counter = Arc::new(AtomicUsize::new(0));
    let c = counter.clone();
    set_rebalance_fail_hook(move |kind| {
        let n = c.fetch_add(1, Ordering::SeqCst);
        matches!(kind, RebalanceTaskKind::Split | RebalanceTaskKind::Merge) && n.is_multiple_of(5)
    });

    // Jittered enqueue of tasks.
    let mut rng = StdRng::seed_from_u64(42);
    for _ in 0..30 {
        if rng.gen_bool(0.6) {
            let _ = worker.enqueue_blocking(RebalanceTask::Split(0));
        } else {
            let _ = worker.enqueue_blocking(RebalanceTask::Merge(0, 1));
        }
        std::thread::sleep(Duration::from_millis(rng.gen_range(1..5)));
    }

    let progressed = wait_until(Duration::from_secs(5), || routing.current_version() > 0);
    clear_rebalance_fail_hook();
    assert!(progressed, "routing should advance despite jitter/failures");
    let sizes = worker.snapshot_sizes();
    assert!(
        !sizes.is_empty() && sizes.values().all(|s| *s > 0),
        "bucket sizes should remain positive"
    );
    Ok(())
}

#[test]
fn rebalancer_close_allows_shutdown() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let storage = Storage::new(wal);
    let routing = Arc::new(RoutingTable::new());
    let worker = RebalanceWorker::spawn(storage.clone(), routing.clone(), None);

    // Close channel and ensure further enqueue fails quickly.
    worker.close();
    let res = worker.enqueue_blocking(RebalanceTask::Split(0));
    assert!(res.is_err(), "enqueue after close should error");
    Ok(())
}

/// Enqueueing Split(X) multiple times should only create 2 buckets total (not 2 per split).
/// This tests idempotency: once a bucket is split and retired, subsequent splits are no-ops.
#[test]
fn duplicate_split_is_idempotent() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let storage = Storage::new(wal);
    let routing = Arc::new(RoutingTable::new());
    let worker = RebalanceWorker::spawn(storage.clone(), routing.clone(), None);

    // Create one bucket with enough vectors to split.
    let mut bucket = Bucket::new(0, vec![0.0, 0.0]);
    for i in 0..20u64 {
        bucket.add_vector(Vector::new(i, vec![i as f32, i as f32 + 0.5]));
    }
    futures::executor::block_on(storage.put_chunk(&bucket))?;
    futures::executor::block_on(worker.prime_centroids(&[bucket.clone()]))?;

    // Enqueue the same split 5 times rapidly.
    for _ in 0..5 {
        worker.enqueue_blocking(RebalanceTask::Split(0))?;
    }

    // Wait for all tasks to be processed.
    let done = wait_until(Duration::from_secs(4), || {
        worker.snapshot_sizes().len() >= 2 && routing.current_version() > 0
    });
    assert!(done, "split should complete");

    // Key assertion: we should have exactly 2 buckets (from one split), not 4, 6, 8, or 10.
    let sizes = worker.snapshot_sizes();
    assert_eq!(
        sizes.len(),
        2,
        "duplicate splits should be idempotent; expected 2 buckets, got {}",
        sizes.len()
    );

    Ok(())
}

/// Enqueueing Merge(A,B) multiple times should only create 1 merged bucket.
/// Once A and B are retired, subsequent merges are no-ops.
#[test]
fn duplicate_merge_is_idempotent() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let storage = Storage::new(wal);
    let routing = Arc::new(RoutingTable::new());
    let worker = RebalanceWorker::spawn(storage.clone(), routing.clone(), None);

    // Create two small buckets.
    let mut a = Bucket::new(0, vec![0.0, 0.0]);
    a.add_vector(Vector::new(0, vec![0.0, 0.0]));
    a.add_vector(Vector::new(1, vec![0.1, 0.1]));
    let mut b = Bucket::new(1, vec![1.0, 1.0]);
    b.add_vector(Vector::new(2, vec![1.0, 1.0]));
    b.add_vector(Vector::new(3, vec![1.1, 1.1]));
    futures::executor::block_on(storage.put_chunk(&a))?;
    futures::executor::block_on(storage.put_chunk(&b))?;
    futures::executor::block_on(worker.prime_centroids(&[a.clone(), b.clone()]))?;

    // Enqueue the same merge 5 times rapidly.
    for _ in 0..5 {
        worker.enqueue_blocking(RebalanceTask::Merge(0, 1))?;
    }

    // Wait for all tasks to be processed.
    let done = wait_until(Duration::from_secs(4), || {
        worker.snapshot_sizes().len() == 1 && routing.current_version() > 0
    });
    assert!(done, "merge should complete");

    // Key assertion: we should have exactly 1 bucket, not multiple merged copies.
    let sizes = worker.snapshot_sizes();
    assert_eq!(
        sizes.len(),
        1,
        "duplicate merges should be idempotent; expected 1 bucket, got {}",
        sizes.len()
    );

    // The merged bucket should have 4 vectors total.
    let total_vectors: usize = sizes.values().sum();
    assert_eq!(
        total_vectors, 4,
        "merged bucket should have 4 vectors, got {}",
        total_vectors
    );

    Ok(())
}

/// After a split completes and retires bucket X, enqueueing another Split(X) should be a no-op.
#[test]
fn split_on_retired_bucket_is_noop() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let storage = Storage::new(wal);
    let routing = Arc::new(RoutingTable::new());
    let worker = RebalanceWorker::spawn(storage.clone(), routing.clone(), None);

    // Create and split a bucket.
    let mut bucket = Bucket::new(0, vec![0.0, 0.0]);
    for i in 0..20u64 {
        bucket.add_vector(Vector::new(i, vec![i as f32, i as f32 + 0.5]));
    }
    futures::executor::block_on(storage.put_chunk(&bucket))?;
    futures::executor::block_on(worker.prime_centroids(&[bucket.clone()]))?;

    worker.enqueue_blocking(RebalanceTask::Split(0))?;

    // Wait for split to complete.
    let done = wait_until(Duration::from_secs(5), || worker.snapshot_sizes().len() == 2);
    assert!(done, "initial split should complete");

    let version_after_first_split = routing.current_version();
    let sizes_after_first_split = worker.snapshot_sizes();

    // Now enqueue another split on the retired bucket 0.
    worker.enqueue_blocking(RebalanceTask::Split(0))?;

    // Give it time to process.
    std::thread::sleep(Duration::from_millis(200));

    // Routing version and bucket count should remain unchanged.
    assert_eq!(
        routing.current_version(),
        version_after_first_split,
        "split on retired bucket should not bump routing version"
    );
    assert_eq!(
        worker.snapshot_sizes().len(),
        sizes_after_first_split.len(),
        "split on retired bucket should not create new buckets"
    );

    Ok(())
}

/// After a merge completes and retires buckets A and B, enqueueing Merge(A,B) again should be a no-op.
#[test]
fn merge_on_retired_bucket_is_noop() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let storage = Storage::new(wal);
    let routing = Arc::new(RoutingTable::new());
    let worker = RebalanceWorker::spawn(storage.clone(), routing.clone(), None);

    // Create two small buckets with enough vectors for merge to proceed.
    let mut a = Bucket::new(0, vec![0.0, 0.0]);
    a.add_vector(Vector::new(0, vec![0.0, 0.0]));
    a.add_vector(Vector::new(1, vec![0.1, 0.1]));
    let mut b = Bucket::new(1, vec![1.0, 1.0]);
    b.add_vector(Vector::new(2, vec![1.0, 1.0]));
    b.add_vector(Vector::new(3, vec![1.1, 1.1]));
    futures::executor::block_on(storage.put_chunk(&a))?;
    futures::executor::block_on(storage.put_chunk(&b))?;
    futures::executor::block_on(worker.prime_centroids(&[a.clone(), b.clone()]))?;

    worker.enqueue_blocking(RebalanceTask::Merge(0, 1))?;

    // Wait for merge to complete.
    let done = wait_until(Duration::from_secs(5), || {
        worker.snapshot_sizes().len() == 1 && routing.current_version() > 0
    });
    assert!(done, "initial merge should complete");

    let version_after_first_merge = routing.current_version();
    let sizes_after_first_merge = worker.snapshot_sizes();

    // Now enqueue another merge on the retired buckets 0 and 1.
    worker.enqueue_blocking(RebalanceTask::Merge(0, 1))?;

    // Give it time to process.
    std::thread::sleep(Duration::from_millis(200));

    // Routing version and bucket count should remain unchanged.
    assert_eq!(
        routing.current_version(),
        version_after_first_merge,
        "merge on retired buckets should not bump routing version"
    );
    assert_eq!(
        worker.snapshot_sizes().len(),
        sizes_after_first_merge.len(),
        "merge on retired buckets should not create new buckets"
    );

    Ok(())
}

/// Concurrent splits on the same bucket from multiple threads should not create duplicate buckets.
#[test]
fn concurrent_splits_same_bucket() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let storage = Storage::new(wal);
    let routing = Arc::new(RoutingTable::new());
    let worker = RebalanceWorker::spawn(storage.clone(), routing.clone(), None);

    // Create one bucket with enough vectors to split.
    let mut bucket = Bucket::new(0, vec![0.0, 0.0]);
    for i in 0..20u64 {
        bucket.add_vector(Vector::new(i, vec![i as f32, i as f32 + 0.5]));
    }
    futures::executor::block_on(storage.put_chunk(&bucket))?;
    futures::executor::block_on(worker.prime_centroids(&[bucket.clone()]))?;

    // Spawn multiple threads that all try to enqueue Split(0) at the same time.
    std::thread::scope(|s| {
        for _ in 0..4 {
            let w = &worker;
            s.spawn(move || {
                for _ in 0..3 {
                    let _ = w.enqueue_blocking(RebalanceTask::Split(0));
                    std::thread::sleep(Duration::from_millis(5));
                }
            });
        }
    });

    // Wait for processing to complete.
    let done = wait_until(Duration::from_secs(4), || {
        worker.snapshot_sizes().len() >= 2 && routing.current_version() > 0
    });
    assert!(done, "split should complete");

    // Key assertion: despite concurrent enqueues, we should have exactly 2 buckets.
    let sizes = worker.snapshot_sizes();
    assert_eq!(
        sizes.len(),
        2,
        "concurrent splits should be serialized; expected 2 buckets, got {}",
        sizes.len()
    );

    Ok(())
}
