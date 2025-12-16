use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use futures::executor::block_on;
use satoridb::rebalancer::{RebalanceTask, RebalanceWorker};
use satoridb::router::RoutingTable;
use satoridb::storage::{Bucket, Storage, Vector};
use satoridb::wal::runtime::Walrus;
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
        thread::sleep(Duration::from_millis(20));
    }
    false
}

#[test]
fn rebalance_split_increases_buckets_and_router_version() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let storage = Storage::new(wal);
    let routing = Arc::new(RoutingTable::new());
    let worker = RebalanceWorker::spawn(storage.clone(), routing.clone(), None);

    // Persist a bucket with enough vectors to split.
    let mut bucket = Bucket::new(0, vec![0.0, 0.0]);
    for i in 0..16u64 {
        bucket.add_vector(Vector::new(i, vec![i as f32, 0.0]));
    }
    block_on(storage.put_chunk(&bucket))?;
    block_on(worker.prime_centroids(&[bucket]))?;

    let initial_version = routing.current_version();
    worker
        .enqueue_blocking(RebalanceTask::Split(0))
        .expect("enqueue split");

    let progressed = wait_until(Duration::from_secs(2), || {
        let sizes = worker.snapshot_sizes();
        sizes.len() > 1
    });
    assert!(progressed, "split did not produce new buckets");
    assert!(
        routing.current_version() > initial_version,
        "router version did not advance after split"
    );

    drop(worker);
    Ok(())
}

#[test]
fn rebalance_merge_retires_inputs_and_rebuilds_router() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let storage = Storage::new(wal);
    let routing = Arc::new(RoutingTable::new());
    let worker = RebalanceWorker::spawn(storage.clone(), routing.clone(), None);

    // Two small buckets that will be merged into a new one.
    let mut b0 = Bucket::new(0, vec![0.0, 0.0]);
    let mut b1 = Bucket::new(1, vec![1.0, 1.0]);
    for i in 0..4u64 {
        b0.add_vector(Vector::new(i, vec![i as f32, 0.0]));
        b1.add_vector(Vector::new(10 + i, vec![1.0 + i as f32, 1.0]));
    }
    block_on(storage.put_chunk(&b0))?;
    block_on(storage.put_chunk(&b1))?;
    block_on(worker.prime_centroids(&[b0, b1]))?;

    let initial_version = routing.current_version();
    worker
        .enqueue_blocking(RebalanceTask::Merge(0, 1))
        .expect("enqueue merge");

    let merged = wait_until(Duration::from_secs(2), || {
        let sizes = worker.snapshot_sizes();
        sizes.len() == 1 && sizes.values().next().copied().unwrap_or(0) >= 8
    });
    assert!(merged, "merge did not produce a single combined bucket");
    assert!(
        routing.current_version() > initial_version,
        "router version did not advance after merge"
    );

    drop(worker);
    Ok(())
}
