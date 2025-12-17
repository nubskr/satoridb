//! Tests that must run serially (single-threaded) due to:
//! - Global fail hook interference between tests
//! - CPU-intensive operations that starve other tests
//!
//! These tests are marked `#[ignore]` so they don't run with `cargo test`.
//! Run with: `cargo test --test serial_tests -- --ignored --test-threads=1`

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use satoridb::rebalancer::{clear_rebalance_fail_hook, RebalanceTask, RebalanceWorker};
use satoridb::router::RoutingTable;
use satoridb::storage::{Bucket, Storage, Vector};
use satoridb::wal::runtime::Walrus;
use satoridb::quantizer::Quantizer;
use satoridb::router::Router;
use tempfile::TempDir;

fn init_wal(tempdir: &TempDir) -> Arc<Walrus> {
    Arc::new(Walrus::with_data_dir(tempdir.path().to_path_buf()).expect("walrus init"))
}

fn wait_until(timeout: Duration, mut f: impl FnMut() -> bool) -> bool {
    let start = Instant::now();
    while start.elapsed() < timeout {
        if f() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    false
}

// ============================================================================
// Rebalancer idempotency tests (affected by global fail hook)
// ============================================================================

/// After a split completes and retires bucket X, enqueueing another Split(X) should be a no-op.
#[test]
#[ignore]
fn split_on_retired_bucket_is_noop() -> Result<()> {
    clear_rebalance_fail_hook();

    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let storage = Storage::new(wal);
    let routing = Arc::new(RoutingTable::new());
    let worker = RebalanceWorker::spawn(storage.clone(), routing.clone(), None);

    let mut bucket = Bucket::new(0, vec![0.0, 0.0]);
    for i in 0..20u64 {
        bucket.add_vector(Vector::new(i, vec![i as f32, i as f32 + 0.5]));
    }
    futures::executor::block_on(storage.put_chunk(&bucket))?;
    futures::executor::block_on(worker.prime_centroids(&[bucket.clone()]))?;

    worker.enqueue_blocking(RebalanceTask::Split(0))?;

    let done = wait_until(Duration::from_secs(5), || worker.snapshot_sizes().len() == 2);
    assert!(done, "initial split should complete");

    let version_after_first_split = routing.current_version();
    let sizes_after_first_split = worker.snapshot_sizes();

    // Now enqueue another split on the retired bucket 0.
    worker.enqueue_blocking(RebalanceTask::Split(0))?;

    std::thread::sleep(Duration::from_millis(200));

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
#[ignore]
fn merge_on_retired_bucket_is_noop() -> Result<()> {
    clear_rebalance_fail_hook();

    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let storage = Storage::new(wal);
    let routing = Arc::new(RoutingTable::new());
    let worker = RebalanceWorker::spawn(storage.clone(), routing.clone(), None);

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

    let done = wait_until(Duration::from_secs(5), || {
        worker.snapshot_sizes().len() == 1 && routing.current_version() > 0
    });
    assert!(done, "initial merge should complete");

    let version_after_first_merge = routing.current_version();
    let sizes_after_first_merge = worker.snapshot_sizes();

    // Now enqueue another merge on the retired buckets 0 and 1.
    worker.enqueue_blocking(RebalanceTask::Merge(0, 1))?;

    std::thread::sleep(Duration::from_millis(200));

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

// ============================================================================
// Router load tests (CPU-intensive, 15k-30k centroids)
// ============================================================================

/// Large centroid set should still return the closest IDs when the graph path
/// and EF autotuning are used.
#[test]
#[ignore]
fn router_large_centroid_set_returns_nearest_ids() -> Result<()> {
    let mut router = Router::new(100_000, Quantizer::new(0.0, 100_000.0));

    let total = 15_000u64;
    for i in 0..total {
        let f = i as f32;
        router.add_centroid(i, &[f, f + 0.5]);
    }

    let q = [12_345.3_f32, 12_345.8_f32];
    let res = router.query(&q, 5)?;
    assert_eq!(res.len(), 5);

    let mut expected: Vec<(f32, u64)> = (0..total)
        .map(|i| {
            let f = i as f32;
            let dx = q[0] - f;
            let dy = q[1] - (f + 0.5);
            let dist = dx * dx + dy * dy;
            (dist, i)
        })
        .collect();
    expected.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let expected_ids: Vec<u64> = expected.iter().take(5).map(|(_, id)| *id).collect();
    let best_true = expected_ids[0] as i64;
    let best_got = res[0] as i64;
    assert!(
        (best_true - best_got).abs() <= 500,
        "router top-1 should stay reasonably close to true nearest (got {}, expected {})",
        best_got,
        best_true
    );

    for id in &res {
        assert!(
            *id < total,
            "router returned centroid id {} outside inserted range",
            id
        );
    }

    Ok(())
}

/// Very large centroid sets force the graph search path; ensure it returns
/// non-empty and reasonably close neighbors.
#[test]
#[ignore]
fn router_graph_path_returns_near_ids() -> Result<()> {
    let mut router = Router::new(200_000, Quantizer::new(0.0, 200_000.0));
    let total = 30_000u64;
    for i in 0..total {
        let f = i as f32;
        router.add_centroid(i, &[f, f + 1.0]);
    }

    let q = [25_432.7_f32, 25_433.7_f32];
    let res = router.query(&q, 10)?;
    assert_eq!(res.len(), 10, "graph path should return requested top-k");

    for id in &res {
        assert!(
            *id < total,
            "returned id {} should be within inserted range",
            id
        );
        let diff = (*id as i64 - 25_432).abs();
        assert!(
            diff < 2_000,
            "graph path returned a far centroid (id diff {} too large)",
            diff
        );
    }

    Ok(())
}
