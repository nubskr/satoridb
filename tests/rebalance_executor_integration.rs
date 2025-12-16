use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use futures::executor::block_on;
use satoridb::executor::{Executor, WorkerCache};
use satoridb::rebalancer::{RebalanceTask, RebalanceWorker};
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
        std::thread::sleep(Duration::from_millis(20));
    }
    false
}

#[test]
fn split_updates_routing_and_executor_results() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let storage = Storage::new(wal);
    let routing = Arc::new(RoutingTable::new());
    let worker = RebalanceWorker::spawn(storage.clone(), routing.clone(), None);

    // One bucket containing two clear clusters: near (0,0) and near (100,100).
    let mut bucket = Bucket::new(0, vec![0.0, 0.0]);
    for i in 0..4u64 {
        bucket.add_vector(Vector::new(i, vec![i as f32 * 0.01, 0.0])); // cluster A
    }
    for i in 0..4u64 {
        bucket.add_vector(Vector::new(100 + i, vec![100.0 + i as f32, 100.0])); // cluster B
    }
    block_on(storage.put_chunk(&bucket))?;
    block_on(worker.prime_centroids(&[bucket.clone()]))?;

    // Request a split of the original bucket.
    worker
        .enqueue_blocking(RebalanceTask::Split(bucket.id))
        .expect("enqueue split");

    // Wait for routing to advance and for sizes to show multiple buckets.
    let progressed = wait_until(Duration::from_secs(3), || {
        worker.snapshot_sizes().len() > 1 && routing.current_version() > 0
    });
    assert!(progressed, "split did not finish in time");

    // Snapshot routing after split.
    let snap = routing.snapshot().expect("routing installed");
    assert!(
        !snap.changed.is_empty(),
        "routing snapshot should list changed buckets"
    );

    // Use executor to query near cluster A and ensure we get the small ids.
    let executor = Executor::new(storage.clone(), WorkerCache::new(8, usize::MAX));
    let bucket_ids = snap.router.query(&[0.0, 0.0], 2)?;
    let sizes = worker.snapshot_sizes();
    let total_after: usize = sizes.values().sum();
    assert_eq!(total_after, 8, "split should preserve total vector count");

    let results = block_on(executor.query(
        &[0.0, 0.0],
        &bucket_ids,
        2,
        snap.version,
        snap.changed.clone(),
    ))?;
    assert!(
        results.iter().any(|(id, _)| *id < 50),
        "expected a vector from the low-id cluster, got {:?}",
        results
    );

    Ok(())
}

#[test]
fn merge_rebuilds_router_and_executor_reads_combined_bucket() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let storage = Storage::new(wal);
    let routing = Arc::new(RoutingTable::new());
    let worker = RebalanceWorker::spawn(storage.clone(), routing.clone(), None);

    // Two buckets that should merge.
    let mut b0 = Bucket::new(0, vec![0.0, 0.0]);
    let mut b1 = Bucket::new(1, vec![10.0, 10.0]);
    for i in 0..4u64 {
        b0.add_vector(Vector::new(i, vec![0.0 + i as f32, 0.0]));
        b1.add_vector(Vector::new(10 + i, vec![10.0 + i as f32, 10.0]));
    }
    block_on(storage.put_chunk(&b0))?;
    block_on(storage.put_chunk(&b1))?;
    block_on(worker.prime_centroids(&[b0.clone(), b1.clone()]))?;

    worker
        .enqueue_blocking(RebalanceTask::Merge(b0.id, b1.id))
        .expect("enqueue merge");

    let merged_ready = wait_until(Duration::from_secs(3), || {
        worker.snapshot_sizes().len() == 1 && routing.current_version() > 0
    });
    assert!(merged_ready, "merge did not finish in time");

    // Query using the rebuilt router; should see all 8 vectors from the merged bucket via executor.
    let snap = routing.snapshot().expect("router installed");
    let route_ids = snap.router.query(&[5.0, 5.0], 1)?;
    assert_eq!(route_ids.len(), 1, "expected a single merged bucket id");

    let executor = Executor::new(storage.clone(), WorkerCache::new(4, usize::MAX));
    let res = block_on(executor.query(
        &[5.0, 5.0],
        &route_ids,
        10,
        snap.version,
        snap.changed.clone(),
    ))?;
    assert_eq!(
        res.len(),
        8,
        "expected all vectors from both source buckets after merge"
    );

    let sizes = worker.snapshot_sizes();
    assert_eq!(sizes.len(), 1, "only one bucket should remain after merge");
    let total_after: usize = sizes.values().sum();
    assert_eq!(total_after, 8, "merge should preserve total vector count");

    Ok(())
}
