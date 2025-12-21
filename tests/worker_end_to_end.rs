use std::sync::Arc;
use std::thread;

use async_channel::unbounded;
use futures::channel::oneshot;
use futures::executor::block_on;
use satoridb::ingest_counter;
use satoridb::storage::wal::runtime::Walrus;
use satoridb::storage::Vector;
use satoridb::vector_index::VectorIndex;
use satoridb::wal::{FsyncSchedule, ReadConsistency};
use satoridb::worker::{run_worker, QueryRequest, WorkerMessage};

fn init_wal(tmp: &tempfile::TempDir) -> Arc<Walrus> {
    std::env::set_var("WALRUS_QUIET", "1");
    Arc::new(
        Walrus::with_data_dir_and_options(
            tmp.path().to_path_buf(),
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
    let index = Arc::new(VectorIndex::open(tmp.path().join("vectors")).expect("index init"));
    let index_reader = index.clone();

    let (tx, rx) = unbounded();
    let handle = thread::spawn(move || {
        let ex = glommio::LocalExecutorBuilder::new(glommio::Placement::Unbound)
            .name("test-worker")
            .make()
            .expect("make executor");
        ex.run(run_worker(0, rx, wal, index));
    });

    // Ingest a small batch into bucket 0.
    let before = ingest_counter::get();
    let vectors = vec![
        Vector::new(1, vec![0.0, 0.0]),
        Vector::new(2, vec![1.0, 0.0]),
    ];
    let (ingest_tx, ingest_rx) = oneshot::channel();
    tx.send_blocking(WorkerMessage::Ingest {
        bucket_id: 0,
        vectors: vectors.clone(),
        respond_to: ingest_tx,
    })
    .expect("ingest send");
    block_on(ingest_rx).expect("ingest ack").expect("ingest ok");

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
        include_vectors: false,
        respond_to: resp_tx,
    }))
    .expect("query send");

    let results = block_on(resp_rx).expect("query recv").expect("query ok");
    assert_eq!(results.len(), vectors.len());
    assert!(results.iter().any(|(id, _, _)| *id == 1));
    assert!(results.iter().any(|(id, _, _)| *id == 2));

    // Ingest counter should have advanced by the batch size.
    let after = ingest_counter::get();
    assert!(
        after >= before + vectors.len() as u64,
        "ingest counter did not advance"
    );

    // Index should have stored both vectors.
    let indexed = index_reader
        .get_many(&[1, 2])
        .expect("index read")
        .into_iter()
        .map(|(id, v)| (id, v.data))
        .collect::<Vec<_>>();
    assert_eq!(indexed.len(), 2);

    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    tx.send_blocking(WorkerMessage::Shutdown {
        respond_to: shutdown_tx,
    })
    .expect("shutdown send");
    block_on(shutdown_rx).expect("shutdown ack");
    handle.join().expect("worker thread joined");
}

#[test]
fn worker_rejects_duplicate_ingest_ids() {
    let tmp = tempfile::tempdir().unwrap();
    let wal = init_wal(&tmp);
    let index = Arc::new(VectorIndex::open(tmp.path().join("vectors")).expect("index init"));

    let (tx, rx) = unbounded();
    let handle = thread::spawn(move || {
        let ex = glommio::LocalExecutorBuilder::new(glommio::Placement::Unbound)
            .name("test-worker-dup")
            .make()
            .expect("make executor");
        ex.run(run_worker(2, rx, wal, index));
    });

    let vecs = vec![Vector::new(42, vec![1.0, 2.0])];
    let (ingest_tx, ingest_rx) = oneshot::channel();
    tx.send_blocking(WorkerMessage::Ingest {
        bucket_id: 0,
        vectors: vecs.clone(),
        respond_to: ingest_tx,
    })
    .expect("first ingest send");
    block_on(ingest_rx)
        .expect("first ingest ack")
        .expect("first ingest ok");

    let (dup_tx, dup_rx) = oneshot::channel();
    tx.send_blocking(WorkerMessage::Ingest {
        bucket_id: 0,
        vectors: vecs,
        respond_to: dup_tx,
    })
    .expect("dup ingest send");
    let err = block_on(dup_rx)
        .expect("dup ingest ack")
        .expect_err("duplicate ingest should fail");
    assert!(
        err.to_string().contains("already exists") || err.to_string().contains("duplicate id"),
        "unexpected error: {:?}",
        err
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
    let index = Arc::new(VectorIndex::open(tmp.path().join("vectors")).expect("index init"));

    let (tx, rx) = unbounded();
    let handle = thread::spawn(move || {
        let ex = glommio::LocalExecutorBuilder::new(glommio::Placement::Unbound)
            .name("test-worker2")
            .make()
            .expect("make executor");
        ex.run(run_worker(1, rx, wal, index));
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
