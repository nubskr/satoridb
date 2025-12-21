use std::sync::Arc;
use std::thread;
use std::time::Duration;

use anyhow::Result;
use futures::executor::block_on;
use satoridb::executor::{Executor, WorkerCache};
use satoridb::storage::{Storage, Vector};
use satoridb::wal::runtime::Walrus;

fn init_wal(tmp: &tempfile::TempDir) -> Arc<Walrus> {
    Arc::new(Walrus::with_data_dir(tmp.path().to_path_buf()).expect("walrus init"))
}

/// Stress that an Executor can be used from multiple threads concurrently while
/// routing versions change. This guards against regressions in Send/Sync safety
/// and cache invalidation under contention.
#[test]
fn executor_handles_concurrent_queries_and_version_bumps() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let storage = Storage::new(wal.clone());

    // Seed bucket 0 with an initial far vector.
    let cache = WorkerCache::new(16, 1024 * 1024, 1024 * 1024);
    let executor = Arc::new(Executor::new(storage.clone(), cache));
    let base_vec = Vector::new(1, vec![0.0, 0.0]);
    block_on(storage.put_chunk_raw(0, &[base_vec]))?;

    // Thread 1: repeatedly query with whatever routing version it is given.
    let exec_q = executor.clone();
    let query_handle = thread::spawn(move || {
        // Start with version 0; bump to 1 after a short delay.
        let mut version = 0u64;
        for i in 0..20 {
            if i == 10 {
                version = 1;
            }
            let res = block_on(exec_q.query(
                &[100.0, 100.0],
                &[0],
                1,
                version,
                Arc::new(if version == 1 { vec![0] } else { Vec::new() }),
                false,
            ))
            .expect("query ok");
            // Either the old or new vector id may win depending on version, but
            // the call must never panic or hang.
            assert!(!res.is_empty());
        }
    });

    // Thread 2: after a brief pause, append a closer vector and bump version.
    let storage_writer = storage.clone();
    let writer_handle = thread::spawn(move || {
        thread::sleep(Duration::from_millis(50));
        let closer = Vector::new(2, vec![100.0, 100.0]);
        block_on(storage_writer.put_chunk_raw(0, &[closer])).expect("put");
    });

    writer_handle.join().expect("writer joined");
    query_handle.join().expect("query joined");

    // Final query with a fresh version bump should return the closer id=2.
    let res = block_on(executor.query(&[100.0, 100.0], &[0], 1, 2, Arc::new(vec![0]), false))?;
    assert_eq!(res[0].0, 2);

    Ok(())
}
