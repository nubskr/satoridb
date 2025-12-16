use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use satoridb::rebalancer::{RebalanceTask, RebalanceWorker};
use satoridb::router::RoutingTable;
use satoridb::storage::{Storage, Vector};
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
        std::thread::sleep(Duration::from_millis(10));
    }
    false
}

/// Enqueueing a split for a missing bucket should not crash and should keep routing unchanged.
#[test]
fn rebalance_handles_missing_bucket_task() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let storage = Storage::new(wal);
    let routing = Arc::new(RoutingTable::new());
    let worker = RebalanceWorker::spawn(storage.clone(), routing.clone(), None);

    // No primed centroids; enqueue split for a non-existent bucket id.
    worker
        .enqueue_blocking(RebalanceTask::Split(999))
        .expect("enqueue split");

    // Wait briefly to let the background worker process the task.
    let done = wait_until(Duration::from_secs(1), || {
        worker.snapshot_sizes().is_empty()
    });
    assert!(
        done,
        "missing-bucket split task should finish without hanging"
    );
    assert_eq!(
        routing.current_version(),
        0,
        "routing version should remain unchanged when split is skipped"
    );
    Ok(())
}

/// Fast-path split condition: when a bucket grows beyond target_size*2, ensure a split is enqueued and routing advances.
#[test]
fn rebalance_fast_path_split_triggers_on_oversized_bucket() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let storage = Storage::new(wal);
    let routing = Arc::new(RoutingTable::new());
    let worker = RebalanceWorker::spawn(storage.clone(), routing.clone(), None);

    // Create one large bucket and prime centroids.
    let mut big = satoridb::storage::Bucket::new(0, vec![0.0, 0.0]);
    for i in 0..50u64 {
        big.add_vector(Vector::new(i, vec![i as f32, 0.0]));
    }
    futures::executor::block_on(storage.put_chunk(&big))?;
    futures::executor::block_on(worker.prime_centroids(&[big.clone()]))?;

    // Manually enqueue split to simulate fast-path driver decision.
    worker
        .enqueue_blocking(RebalanceTask::Split(big.id))
        .expect("enqueue split");

    let progressed = wait_until(Duration::from_secs(3), || {
        worker.snapshot_sizes().len() > 1 && routing.current_version() > 0
    });
    assert!(
        progressed,
        "fast-path split should complete and advance routing"
    );
    Ok(())
}
