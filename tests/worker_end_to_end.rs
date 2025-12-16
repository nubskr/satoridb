use std::sync::Arc;
use std::thread;

use async_channel::unbounded;
use futures::channel::oneshot;
use futures::executor::block_on;
use satoridb::ingest_counter;
use satoridb::storage::wal::runtime::Walrus;
use satoridb::storage::Vector;
use satoridb::wal::{FsyncSchedule, ReadConsistency};
use satoridb::worker::{run_worker, QueryRequest, WorkerMessage};

fn init_wal(tmp: &tempfile::TempDir) -> Arc<Walrus> {
    std::env::set_var("WALRUS_DATA_DIR", tmp.path());
    std::env::set_var("WALRUS_QUIET", "1");
    Arc::new(
        Walrus::with_consistency_and_schedule(
            ReadConsistency::StrictlyAtOnce,
            FsyncSchedule::NoFsync,
        )
        .expect("walrus init"),
    )
}

#[test]
fn worker_ingests_and_answers_queries() {
    let tmp = tempfile::tempdir().unwrap();
    let wal = init_wal(&tmp);

    let (tx, rx) = unbounded();
    let handle = thread::spawn(move || {
        let ex = glommio::LocalExecutorBuilder::new(glommio::Placement::Unbound)
            .name("test-worker")
            .make()
            .expect("make executor");
        ex.run(run_worker(0, rx, wal));
    });

    // Ingest a small batch into bucket 0.
    let before = ingest_counter::get();
    let vectors = vec![
        Vector::new(1, vec![0.0, 0.0]),
        Vector::new(2, vec![1.0, 0.0]),
    ];
    tx.send_blocking(WorkerMessage::Ingest {
        bucket_id: 0,
        vectors: vectors.clone(),
    })
    .expect("ingest send");

    // Flush to ensure the worker has drained ingest before querying.
    let (flush_tx, flush_rx) = oneshot::channel();
    tx.send_blocking(WorkerMessage::Flush {
        respond_to: flush_tx,
    })
    .expect("flush send");
    block_on(flush_rx).expect("flush ack");

    // Query the same bucket.
    let (resp_tx, resp_rx) = oneshot::channel();
    tx.send_blocking(WorkerMessage::Query(QueryRequest {
        query_vec: vec![0.0, 0.0],
        bucket_ids: vec![0],
        routing_version: 0,
        affected_buckets: Arc::new(Vec::new()),
        respond_to: resp_tx,
    }))
    .expect("query send");

    let results = block_on(resp_rx)
        .expect("query recv")
        .expect("query ok");
    assert_eq!(results.len(), vectors.len());
    assert!(results.iter().any(|(id, _)| *id == 1));
    assert!(results.iter().any(|(id, _)| *id == 2));

    // Ingest counter should have advanced by the batch size.
    let after = ingest_counter::get();
    assert!(
        after >= before + vectors.len() as u64,
        "ingest counter did not advance"
    );

    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    tx.send_blocking(WorkerMessage::Shutdown {
        respond_to: shutdown_tx,
    })
    .expect("shutdown send");
    block_on(shutdown_rx).expect("shutdown ack");
    handle.join().expect("worker thread joined");
}

#[test]
fn worker_flushes_and_shuts_down_cleanly() {
    let tmp = tempfile::tempdir().unwrap();
    let wal = init_wal(&tmp);

    let (tx, rx) = unbounded();
    let handle = thread::spawn(move || {
        let ex = glommio::LocalExecutorBuilder::new(glommio::Placement::Unbound)
            .name("test-worker2")
            .make()
            .expect("make executor");
        ex.run(run_worker(1, rx, wal));
    });

    let (flush_tx, flush_rx) = oneshot::channel();
    tx.send_blocking(WorkerMessage::Flush {
        respond_to: flush_tx,
    })
    .expect("flush send");
    block_on(flush_rx).expect("flush ack");

    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    tx.send_blocking(WorkerMessage::Shutdown {
        respond_to: shutdown_tx,
    })
    .expect("shutdown send");
    block_on(shutdown_rx).expect("shutdown ack");
    handle.join().expect("worker thread joined");
}
