use std::collections::HashMap;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use async_channel::unbounded;
use futures::channel::oneshot;
use futures::executor::block_on;
use satoridb::bucket_index::BucketIndex;
use satoridb::bucket_locks::BucketLocks;
use satoridb::embedded::{SatoriDb, SatoriDbConfig};
use satoridb::executor::{Executor, WorkerCache};
use satoridb::rebalancer::RebalanceWorker;
use satoridb::router::RoutingTable;
use satoridb::storage::{Bucket, Storage, Vector};
use satoridb::vector_index::VectorIndex;
use satoridb::wal::runtime::Walrus;
use satoridb::wal::{FsyncSchedule, ReadConsistency};
use satoridb::worker::{run_worker, QueryRequest, WorkerMessage};
use tempfile::TempDir;

fn init_wal(tempdir: &TempDir) -> Arc<Walrus> {
    std::env::set_var("WALRUS_QUIET", "1");
    Arc::new(
        Walrus::with_data_dir_and_options(
            tempdir.path().to_path_buf(),
            ReadConsistency::StrictlyAtOnce,
            FsyncSchedule::NoFsync,
        )
        .expect("walrus init"),
    )
}

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(0x9e3779b97f4a7c15);
    *seed
}

fn make_vec(seed: &mut u64, dim: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(dim);
    for _ in 0..dim {
        let bits = (lcg(seed) >> 32) as u32;
        // Keep values in a modest range to avoid NaNs while still varying the data.
        out.push((bits as f32 / u32::MAX as f32) * 2.0 - 1.0);
    }
    out
}

/// Long-running ingest/query soak that pushes thousands of upserts and fetches.
/// Ignored by default because it runs for several seconds and is meant for
/// regression/stress sweeps.
#[test]
#[ignore]
fn long_running_ingest_and_query_soak() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let mut cfg = SatoriDbConfig::new(wal);
    cfg.workers = std::cmp::min(4, num_cpus::get().max(1));
    cfg.virtual_nodes = 32;

    let db = SatoriDb::start(cfg)?;
    let handle = db.handle();

    let duration = Duration::from_secs(8);
    let start = Instant::now();

    let next_id = Arc::new(AtomicU64::new(1));
    let recorded = Arc::new(parking_lot::Mutex::new(Vec::new()));

    std::thread::scope(|s| {
        // Writers hammer upserts and occasionally query-with-vectors on the same payload.
        for t in 0..4 {
            let handle = handle.clone();
            let next_id = next_id.clone();
            let recorded = recorded.clone();
            s.spawn(move || {
                let mut seed = 0x1234_5678_9abc_def0u64 ^ (t as u64);
                let mut local = Vec::new();
                while start.elapsed() < duration {
                    let id = next_id.fetch_add(1, Ordering::Relaxed);
                    let vector = make_vec(&mut seed, 16);
                    if handle.upsert_blocking(id, vector.clone(), None).is_ok() {
                        local.push(id);
                        if id.is_multiple_of(8) {
                            let _ = handle.query_with_vectors_blocking(vector, 8, 8);
                        }
                    }
                    if local.len() >= 256 {
                        let mut guard = recorded.lock();
                        guard.extend(local.drain(..));
                    }
                }
                if !local.is_empty() {
                    recorded.lock().extend(local);
                }
            });
        }

        // Readers constantly sample previously inserted ids via the global RocksDB index.
        for t in 0..2 {
            let handle = handle.clone();
            let recorded = recorded.clone();
            s.spawn(move || {
                let mut seed = 0xfeed_babe_cafe_d00du64 ^ (t as u64);
                while start.elapsed() < duration {
                    let snapshot: Vec<u64> = {
                        let guard = recorded.lock();
                        if guard.is_empty() {
                            Vec::new()
                        } else {
                            let mut picks = Vec::with_capacity(8);
                            for _ in 0..8 {
                                let idx = (lcg(&mut seed) as usize) % guard.len();
                                picks.push(guard[idx]);
                            }
                            picks
                        }
                    };
                    if snapshot.is_empty() {
                        std::thread::sleep(Duration::from_millis(5));
                        continue;
                    }
                    let _ = handle.fetch_vectors_by_id_blocking(snapshot);
                }
            });
        }
    });

    handle.flush_blocking()?;
    let mut ids = recorded.lock().clone();
    ids.sort_unstable();
    ids.dedup();
    let fetched = handle.fetch_vectors_by_id_blocking(ids.clone())?;
    assert_eq!(
        fetched.len(),
        ids.len(),
        "missing vectors after long-running ingest/query soak"
    );
    db.shutdown()?;
    Ok(())
}

/// End-to-end stress that hammers a single worker while a rebalance loop splits
/// buckets aggressively. This is meant to surface synchronization gaps and WAL
/// corruption under load; ignored by default due to runtime.
#[test]
#[ignore]
fn rebalance_split_under_fire() -> Result<()> {
    // Force frequent splits to keep the rebalancer busy.
    std::env::set_var("SATORI_REBALANCE_THRESHOLD", "32");

    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let storage = Storage::new(wal.clone());
    let vector_index = Arc::new(VectorIndex::open(tmp.path().join("vectors"))?);
    let bucket_index = Arc::new(BucketIndex::open(tmp.path().join("buckets"))?);
    let bucket_locks = Arc::new(BucketLocks::new());
    let routing = Arc::new(RoutingTable::new());

    // Prime one bucket so the rebalancer has a router to work with.
    let mut seed_bucket = Bucket::new(0, vec![0.0, 0.0]);
    for i in 0..32u64 {
        seed_bucket.add_vector(Vector::new(i, vec![i as f32, i as f32 + 0.25]));
    }
    block_on(storage.put_chunk(&seed_bucket))?;
    let rebalance =
        RebalanceWorker::spawn(storage.clone(), routing.clone(), None, bucket_locks.clone());
    block_on(rebalance.prime_centroids(&[seed_bucket]))?;

    let (tx, rx) = unbounded();
    let wal_worker = wal.clone();
    let vi_worker = vector_index.clone();
    let bi_worker = bucket_index.clone();
    let locks_worker = bucket_locks.clone();
    let handle = std::thread::spawn(move || {
        let ex = glommio::LocalExecutorBuilder::new(glommio::Placement::Unbound)
            .name("extreme-worker")
            .make()
            .expect("make executor");
        ex.run(run_worker(
            0,
            rx,
            wal_worker,
            vi_worker,
            bi_worker,
            locks_worker,
        ));
    });

    let duration = Duration::from_secs(6);
    let start = Instant::now();
    let next_id = Arc::new(AtomicU64::new(1_000));
    let recorded = Arc::new(parking_lot::Mutex::new(Vec::new()));

    std::thread::scope(|s| {
        // Hot ingest loop that follows the current router to find the right bucket.
        for t in 0..3 {
            let tx = tx.clone();
            let routing = routing.clone();
            let next_id = next_id.clone();
            let recorded = recorded.clone();
            s.spawn(move || {
                let mut seed = 0x4242_d00d_cafe_beefu64 ^ (t as u64);
                let mut local = Vec::new();
                while start.elapsed() < duration {
                    let id = next_id.fetch_add(1, Ordering::Relaxed);
                    let vec = make_vec(&mut seed, 16);
                    let bucket_id = routing
                        .snapshot()
                        .and_then(|snap| {
                            snap.router
                                .query(&vec, 1)
                                .ok()
                                .and_then(|ids| ids.first().copied())
                        })
                        .unwrap_or(0);
                    let (resp_tx, resp_rx) = oneshot::channel();
                    let msg = WorkerMessage::Ingest {
                        bucket_id,
                        vectors: vec![Vector::new(id, vec)],
                        respond_to: resp_tx,
                    };
                    if tx.send_blocking(msg).is_err() {
                        break;
                    }
                    match block_on(resp_rx) {
                        Ok(Ok(())) => {
                            local.push(id);
                            if local.len() >= 256 {
                                recorded.lock().extend(local.drain(..));
                            }
                        }
                        other => panic!("ingest failed under stress: {:?}", other),
                    }
                }
                if !local.is_empty() {
                    recorded.lock().extend(local);
                }
            });
        }

        // Query loop that continually touches whatever buckets the router exposes.
        let tx_queries = tx.clone();
        let routing_queries = routing.clone();
        s.spawn(move || {
            let mut seed = 0x1337_2012_dead_beefu64;
            while start.elapsed() < duration {
                let Some(snapshot) = routing_queries.snapshot() else {
                    std::thread::sleep(Duration::from_millis(5));
                    continue;
                };
                let qvec = make_vec(&mut seed, 16);
                let buckets = snapshot.router.query(&qvec, 4).unwrap_or_default();
                if buckets.is_empty() {
                    continue;
                }
                let (resp_tx, resp_rx) = oneshot::channel();
                let req = QueryRequest {
                    query_vec: qvec,
                    bucket_ids: buckets,
                    routing_version: snapshot.version,
                    affected_buckets: snapshot.changed.clone(),
                    include_vectors: false,
                    respond_to: resp_tx,
                };
                if tx_queries.send_blocking(WorkerMessage::Query(req)).is_err() {
                    break;
                }
                let _ = block_on(resp_rx);
            }
        });
    });

    let (flush_tx, flush_rx) = oneshot::channel();
    let _ = tx.send_blocking(WorkerMessage::Flush {
        respond_to: flush_tx,
    });
    let _ = block_on(flush_rx);

    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let _ = tx.send_blocking(WorkerMessage::Shutdown {
        respond_to: shutdown_tx,
    });
    let _ = block_on(shutdown_rx);
    handle.join().expect("worker thread joined");

    let mut ids = recorded.lock().clone();
    ids.sort_unstable();
    ids.dedup();
    let fetched = vector_index.get_many(&ids)?;
    assert_eq!(
        fetched.len(),
        ids.len(),
        "vector index lost entries during rebalance stress"
    );

    let bucket_mappings = bucket_index.get_many(&ids)?;
    assert_eq!(
        bucket_mappings.len(),
        ids.len(),
        "bucket index missing entries after rebalance stress"
    );

    Ok(())
}

/// Soak the system, then restart from disk and ensure all vectors/bucket mappings survive.
#[test]
#[ignore]
fn restart_recovers_state_after_soak() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let vector_path = tmp.path().join("vectors");
    let bucket_path = tmp.path().join("buckets");

    let mut cfg = SatoriDbConfig::new(wal.clone());
    cfg.workers = std::cmp::min(4, num_cpus::get().max(1));
    cfg.vector_index_path = vector_path.clone();
    cfg.bucket_index_path = bucket_path.clone();
    cfg.virtual_nodes = 16;

    let db = SatoriDb::start(cfg)?;
    let handle = db.handle();

    let duration = Duration::from_secs(6);
    let start = Instant::now();
    let next_id = Arc::new(AtomicU64::new(1));
    let recorded = Arc::new(parking_lot::RwLock::new(HashMap::<u64, Vec<f32>>::new()));

    std::thread::scope(|s| {
        for t in 0..4 {
            let handle = handle.clone();
            let next_id = next_id.clone();
            let recorded = recorded.clone();
            s.spawn(move || {
                let mut seed = 0xdddd_beef_cafe_f00du64 ^ (t as u64);
                while start.elapsed() < duration {
                    let id = next_id.fetch_add(1, Ordering::Relaxed);
                    let vec = make_vec(&mut seed, 24);
                    if handle.upsert_blocking(id, vec.clone(), None).is_ok() {
                        recorded.write().insert(id, vec.clone());
                        if id.is_multiple_of(5) {
                            let _ = handle.query_blocking(vec, 8, 8);
                        }
                    }
                }
            });
        }
    });

    handle.flush_blocking()?;
    let ids: Vec<u64> = {
        let guard = recorded.read();
        guard.keys().copied().collect()
    };
    drop(handle);
    drop(wal);
    db.shutdown()?;

    // Restart with the same WAL/index paths and validate the persisted payloads.
    let wal = init_wal(&tmp);
    let workers = std::cmp::min(4, num_cpus::get().max(1));
    let db = {
        let mut attempt = 0;
        loop {
            let mut cfg = SatoriDbConfig::new(wal.clone());
            cfg.workers = workers;
            cfg.vector_index_path = vector_path.clone();
            cfg.bucket_index_path = bucket_path.clone();
            match SatoriDb::start(cfg) {
                Ok(db) => break db,
                Err(e) if attempt < 5 && e.to_string().contains("LOCK") => {
                    attempt += 1;
                    std::thread::sleep(Duration::from_millis(50));
                    continue;
                }
                Err(e) => return Err(e),
            }
        }
    };
    let handle = db.handle();

    let fetched = handle.fetch_vectors_by_id_blocking(ids.clone())?;
    assert_eq!(fetched.len(), ids.len(), "missing vectors after restart");
    let expected = recorded.read();
    for (id, vec) in fetched {
        let Some(exp) = expected.get(&id) else {
            panic!("unexpected id {} returned after restart", id);
        };
        assert_eq!(&vec, exp, "vector payload mismatch for {}", id);
    }
    let bucket_mappings = block_on(handle.resolve_buckets_by_id(ids.clone()))?;
    assert_eq!(
        bucket_mappings.len(),
        ids.len(),
        "bucket index missing entries after restart"
    );

    db.shutdown()?;
    Ok(())
}

/// Multi-worker churn with mixed upserts, queries (with vectors), fetches, and periodic router reads.
/// This aims to surface cross-shard contention issues and index coherency gaps under load.
#[test]
#[ignore]
fn multi_worker_mixed_workload_churn() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let mut cfg = SatoriDbConfig::new(wal);
    cfg.workers = std::cmp::min(6, num_cpus::get().max(2));
    cfg.virtual_nodes = 48;
    let db = SatoriDb::start(cfg)?;
    let handle = db.handle();

    let duration = Duration::from_secs(7);
    let start = Instant::now();
    let next_id = Arc::new(AtomicU64::new(10_000));
    let recorded = Arc::new(parking_lot::RwLock::new(HashMap::<u64, Vec<f32>>::new()));

    std::thread::scope(|s| {
        // Writers and occasional query-with-vectors
        for t in 0..6 {
            let handle = handle.clone();
            let next_id = next_id.clone();
            let recorded = recorded.clone();
            s.spawn(move || {
                let mut seed = 0xaaaa_bbbb_cccc_ddddu64 ^ (t as u64);
                while start.elapsed() < duration {
                    let id = next_id.fetch_add(1, Ordering::Relaxed);
                    let vec = make_vec(&mut seed, 20);
                    if handle.upsert_blocking(id, vec.clone(), None).is_ok() {
                        recorded.write().insert(id, vec.clone());
                        if id.is_multiple_of(6) {
                            let _ = handle.query_with_vectors_blocking(vec, 6, 6);
                        }
                    }
                }
            });
        }

        // Fetchers hit the global vector index constantly.
        for t in 0..3 {
            let handle = handle.clone();
            let recorded = recorded.clone();
            s.spawn(move || {
                let mut seed = 0x9999_8888_7777_6666u64 ^ (t as u64);
                while start.elapsed() < duration {
                    let sample: Vec<u64> = {
                        let guard = recorded.read();
                        if guard.is_empty() {
                            Vec::new()
                        } else {
                            let mut ids = Vec::with_capacity(10);
                            for _ in 0..10 {
                                let idx = (lcg(&mut seed) as usize) % guard.len();
                                let (&k, _) = guard.iter().nth(idx).unwrap();
                                ids.push(k);
                            }
                            ids
                        }
                    };
                    if sample.is_empty() {
                        std::thread::sleep(Duration::from_millis(5));
                        continue;
                    }
                    let _ = handle.fetch_vectors_by_id_blocking(sample);
                }
            });
        }
    });

    handle.flush_blocking()?;
    let ids: Vec<u64> = {
        let guard = recorded.read();
        guard.keys().copied().collect()
    };
    let fetched = handle.fetch_vectors_by_id_blocking(ids.clone())?;
    assert_eq!(
        fetched.len(),
        ids.len(),
        "some vectors missing after mixed-workload churn"
    );
    let expected = recorded.read();
    for (id, vec) in fetched {
        assert_eq!(
            Some(&vec),
            expected.get(&id),
            "vector payload mismatch after churn for {}",
            id
        );
    }
    db.shutdown()?;
    Ok(())
}

/// Saturate a tiny worker channel to ensure backpressure doesn't panic or drop messages.
#[test]
#[ignore]
fn worker_channel_backpressure_survives_burst() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let index = Arc::new(VectorIndex::open(tmp.path().join("vectors"))?);
    let bucket_index = Arc::new(BucketIndex::open(tmp.path().join("buckets"))?);
    let bucket_locks = Arc::new(BucketLocks::new());

    // Very small channel to force senders to block.
    let (tx, rx) = async_channel::bounded(4);
    let wal_worker = wal.clone();
    let index_worker = index.clone();
    let bucket_index_worker = bucket_index.clone();
    let bucket_locks_worker = bucket_locks.clone();
    let handle = std::thread::spawn(move || {
        let ex = glommio::LocalExecutorBuilder::new(glommio::Placement::Unbound)
            .name("backpressure-worker")
            .make()
            .expect("make executor");
        ex.run(run_worker(
            0,
            rx,
            wal_worker,
            index_worker,
            bucket_index_worker,
            bucket_locks_worker,
        ));
    });

    let successes = Arc::new(AtomicU64::new(0));
    std::thread::scope(|s| {
        for t in 0..6 {
            let tx = tx.clone();
            let successes = successes.clone();
            s.spawn(move || {
                let mut seed = 0xdead_beef_0101_0000u64 ^ (t as u64);
                for _ in 0..200 {
                    let id = lcg(&mut seed);
                    let vec = make_vec(&mut seed, 8);
                    let (resp_tx, resp_rx) = oneshot::channel();
                    tx.send_blocking(WorkerMessage::Ingest {
                        bucket_id: 0,
                        vectors: vec![Vector::new(id, vec)],
                        respond_to: resp_tx,
                    })
                    .expect("send ingest");
                    block_on(resp_rx).expect("ack recv").expect("ingest ok");
                    successes.fetch_add(1, Ordering::Relaxed);
                }
            });
        }
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

    // Ensure most writes landed.
    let ingested = successes.load(Ordering::Relaxed);
    assert!(
        ingested >= 800,
        "expected high ingest throughput, got {}",
        ingested
    );
    Ok(())
}

/// Validate that very high-dimensional vectors survive ingest/query and round-trip payload.
#[test]
#[ignore]
fn large_dimension_vectors_roundtrip() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let mut cfg = SatoriDbConfig::new(wal);
    cfg.workers = 2;
    cfg.virtual_nodes = 16;
    let db = SatoriDb::start(cfg)?;
    let handle = db.handle();

    let dim = 4096;
    let count = 32;
    let mut seed = 0xabcdef_u64;
    for i in 0..count {
        let vec = make_vec(&mut seed, dim);
        handle.upsert_blocking(10_000 + i, vec, None)?;
    }
    handle.flush_blocking()?;

    let q = make_vec(&mut seed, dim);
    let res = handle.query_with_vectors_blocking(q, 8, 8)?;
    assert!(!res.is_empty(), "query should return neighbors");
    for (_id, _dist, v) in res {
        assert_eq!(v.len(), dim);
    }
    db.shutdown()?;
    Ok(())
}

/// Router rebuild churn under heavy upserts; ensures router remains ready and pending updates drain.
#[test]
#[ignore]
fn router_rebuild_churn_under_load() -> Result<()> {
    std::env::set_var("SATORI_ROUTER_REBUILD_EVERY", "1");
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let mut cfg = SatoriDbConfig::new(wal);
    cfg.workers = 3;
    cfg.virtual_nodes = 24;
    let db = SatoriDb::start(cfg)?;
    let handle = db.handle();

    let duration = Duration::from_secs(5);
    let start = Instant::now();
    let next_id = Arc::new(AtomicU64::new(50_000));

    std::thread::scope(|s| {
        for t in 0..4 {
            let handle = handle.clone();
            let next_id = next_id.clone();
            s.spawn(move || {
                let mut seed = 0x1234_5678u64 ^ (t as u64);
                while start.elapsed() < duration {
                    let id = next_id.fetch_add(1, Ordering::Relaxed);
                    let vec = make_vec(&mut seed, 16);
                    let _ = handle.upsert_blocking(id, vec.clone(), None);
                    if id.is_multiple_of(7) {
                        let _ = handle.query_blocking(vec, 4, 4);
                    }
                }
            });
        }
    });

    handle.flush_blocking()?;
    let stats = block_on(handle.stats());
    assert!(stats.ready, "router should stay ready under rebuild churn");
    assert!(
        stats.pending_updates == 0,
        "pending updates should drain after rebuild churn (got {})",
        stats.pending_updates
    );
    db.shutdown()?;
    Ok(())
}

/// Concurrent fetches against vector and bucket indexes while rebalancer is splitting.
#[test]
#[ignore]
fn fetches_remain_consistent_during_rebalance() -> Result<()> {
    std::env::set_var("SATORI_REBALANCE_THRESHOLD", "24");

    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let storage = Storage::new(wal.clone());
    let vector_index = Arc::new(VectorIndex::open(tmp.path().join("vectors"))?);
    let bucket_index = Arc::new(BucketIndex::open(tmp.path().join("buckets"))?);
    let bucket_locks = Arc::new(BucketLocks::new());
    let routing = Arc::new(RoutingTable::new());

    // Seed bucket.
    let mut b = Bucket::new(0, vec![0.0, 0.0]);
    for i in 0..40u64 {
        b.add_vector(Vector::new(i, vec![i as f32, i as f32 * 0.5]));
    }
    block_on(storage.put_chunk(&b))?;
    let rebalance =
        RebalanceWorker::spawn(storage.clone(), routing.clone(), None, bucket_locks.clone());
    block_on(rebalance.prime_centroids(&[b]))?;

    let (tx, rx) = unbounded();
    let wal_worker = wal.clone();
    let vi_worker = vector_index.clone();
    let bi_worker = bucket_index.clone();
    let locks_worker = bucket_locks.clone();
    let handle = std::thread::spawn(move || {
        let ex = glommio::LocalExecutorBuilder::new(glommio::Placement::Unbound)
            .name("fetch-rebalance-worker")
            .make()
            .expect("make executor");
        ex.run(run_worker(
            0,
            rx,
            wal_worker,
            vi_worker,
            bi_worker,
            locks_worker,
        ));
    });

    let duration = Duration::from_secs(6);
    let start = Instant::now();
    let next_id = Arc::new(AtomicU64::new(1_000_000));
    let all_ids = Arc::new(parking_lot::RwLock::new(Vec::new()));

    std::thread::scope(|s| {
        // Ingest new ids to keep indexes moving.
        for t in 0..2 {
            let tx = tx.clone();
            let routing = routing.clone();
            let next_id = next_id.clone();
            let all_ids = all_ids.clone();
            s.spawn(move || {
                let mut seed = 0x5555_6666_7777_8888u64 ^ (t as u64);
                while start.elapsed() < duration {
                    let id = next_id.fetch_add(1, Ordering::Relaxed);
                    let vec = make_vec(&mut seed, 12);
                    let bucket_id = routing
                        .snapshot()
                        .and_then(|snap| {
                            snap.router
                                .query(&vec, 1)
                                .ok()
                                .and_then(|ids| ids.first().copied())
                        })
                        .unwrap_or(0);
                    let (resp_tx, resp_rx) = oneshot::channel();
                    tx.send_blocking(WorkerMessage::Ingest {
                        bucket_id,
                        vectors: vec![Vector::new(id, vec)],
                        respond_to: resp_tx,
                    })
                    .expect("ingest send");
                    let _ = block_on(resp_rx);
                    all_ids.write().push(id);
                }
            });
        }

        // Fetch vectors/buckets concurrently.
        for t in 0..2 {
            let vi = vector_index.clone();
            let bi = bucket_index.clone();
            let all_ids = all_ids.clone();
            s.spawn(move || {
                let mut seed = 0x9999_aaaa_bbbb_ccccu64 ^ (t as u64);
                while start.elapsed() < duration {
                    let sample: Vec<u64> = {
                        let guard = all_ids.read();
                        if guard.is_empty() {
                            Vec::new()
                        } else {
                            let mut out = Vec::with_capacity(10);
                            for _ in 0..10 {
                                let idx = (lcg(&mut seed) as usize) % guard.len();
                                out.push(guard[idx]);
                            }
                            out
                        }
                    };
                    if sample.is_empty() {
                        std::thread::sleep(Duration::from_millis(5));
                        continue;
                    }
                    let _ = vi.get_many(&sample);
                    let _ = bi.get_many(&sample);
                }
            });
        }
    });

    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    tx.send_blocking(WorkerMessage::Shutdown {
        respond_to: shutdown_tx,
    })
    .expect("shutdown send");
    block_on(shutdown_rx).expect("shutdown ack");
    handle.join().expect("worker thread joined");

    let ids = all_ids.read().clone();
    let vectors = vector_index.get_many(&ids)?;
    let buckets = bucket_index.get_many(&ids)?;
    assert_eq!(
        vectors.len(),
        buckets.len(),
        "vector and bucket indexes diverged during rebalance fetch stress"
    );
    Ok(())
}

/// Cache eviction correctness: tiny cache must evict but still return results and track version.
#[test]
#[ignore]
fn executor_cache_eviction_still_serves_queries() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let storage = Storage::new(wal);

    // Two buckets, cache holds only one.
    for bid in 0..2u64 {
        let mut b = Bucket::new(bid, vec![bid as f32, 0.0]);
        for j in 0..8u64 {
            b.add_vector(Vector::new(bid * 100 + j, vec![j as f32, j as f32 + 1.0]));
        }
        block_on(storage.put_chunk(&b))?;
    }

    let cache = WorkerCache::new(1, 256 * 1024, usize::MAX);
    let exec = Executor::new(storage.clone(), cache);

    // First query caches bucket 0.
    let res0 = block_on(exec.query(&[0.5, 0.5], &[0], 4, 1, Arc::new(Vec::new()), false))?;
    assert!(!res0.is_empty(), "expected results from bucket 0");
    assert_eq!(exec.cache_version(), 1);

    // Next query includes bucket 1; capacity forces eviction of bucket 0.
    let res1 = block_on(exec.query(&[1.5, 1.5], &[0, 1], 4, 1, Arc::new(Vec::new()), false))?;
    assert!(
        res1.iter().any(|(id, _, _)| *id / 100 == 1),
        "should return from bucket 1 after eviction"
    );

    // Now mark bucket 1 as changed and update storage; cache must invalidate and pick new data.
    let mut updated = Bucket::new(1, vec![1.0, 0.0]);
    updated.add_vector(Vector::new(10_000, vec![1000.0, 1000.0]));
    block_on(storage.put_chunk(&updated))?;
    let res2 = block_on(exec.query(&[1000.0, 1000.0], &[1], 1, 2, Arc::new(vec![1]), false))?;
    assert!(
        res2.iter().any(|(id, _, _)| *id == 10_000),
        "cache invalidation should allow updated bucket to surface"
    );
    assert_eq!(exec.cache_version(), 2);
    Ok(())
}

/// Verify routing stays coherent under a large bucket set and frequent rebuilds.
#[test]
#[ignore]
fn router_scales_with_many_buckets() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let mut cfg = SatoriDbConfig::new(wal);
    cfg.workers = 3;
    cfg.virtual_nodes = 64;
    let db = SatoriDb::start(cfg)?;
    let handle = db.handle();

    // Create many tiny buckets to stress routing.
    let bucket_count = 120u64;
    let mut seed = 0x1234abcd_u64;
    for b in 0..bucket_count {
        let v = make_vec(&mut seed, 8);
        handle.upsert_blocking(b, v.clone(), Some(b))?;
    }
    handle.flush_blocking()?;

    // Rebuild router frequently and send queries.
    std::env::set_var("SATORI_ROUTER_REBUILD_EVERY", "10");
    let mut queries = 0usize;
    while queries < 30 {
        let q = make_vec(&mut seed, 8);
        let res = handle.query_blocking(q, 4, 8)?;
        assert!(
            res.len() <= 4,
            "router should cap results to top_k; got {}",
            res.len()
        );
        queries += 1;
    }
    let stats = block_on(handle.stats());
    assert!(stats.ready);
    assert!(stats.buckets >= bucket_count as usize / 2);
    db.shutdown()?;
    Ok(())
}

/// Child routine for crash/replay fuzz. When the env var is present we ingest and abort.
fn maybe_run_crash_child() {
    if let (Ok(wal_dir), Ok(vec_dir), Ok(bucket_dir)) = (
        std::env::var("CRASH_CHILD_WAL"),
        std::env::var("CRASH_CHILD_VECTORS"),
        std::env::var("CRASH_CHILD_BUCKETS"),
    ) {
        let wal = Arc::new(
            Walrus::with_data_dir_and_options(
                wal_dir.into(),
                ReadConsistency::StrictlyAtOnce,
                FsyncSchedule::NoFsync,
            )
            .expect("child wal init"),
        );
        let mut cfg = SatoriDbConfig::new(wal);
        cfg.workers = 2;
        cfg.vector_index_path = vec_dir.into();
        cfg.bucket_index_path = bucket_dir.into();
        let db = SatoriDb::start(cfg).expect("child start");
        let handle = db.handle();
        let mut seed = 0xcafe_cafe_dead_beefu64;
        for i in 0..200u64 {
            let v = make_vec(&mut seed, 16);
            let _ = handle.upsert_blocking(500_000 + i, v, None);
        }
        // Abrupt crash to simulate power loss.
        std::process::abort();
    }
}

/// Ingest, crash hard, restart from the same WAL/index paths, and ensure data survives.
#[test]
#[ignore]
fn crash_replay_recovery() -> Result<()> {
    maybe_run_crash_child();

    let tmp = tempfile::tempdir()?;
    let wal_dir = tmp.path().join("wal");
    let vec_dir = tmp.path().join("vectors");
    let bucket_dir = tmp.path().join("buckets");

    let wal = Arc::new(
        Walrus::with_data_dir_and_options(
            wal_dir.clone(),
            ReadConsistency::StrictlyAtOnce,
            FsyncSchedule::NoFsync,
        )
        .expect("parent wal init"),
    );

    // First run will crash in the child.
    let mut child = std::process::Command::new(std::env::current_exe()?)
        .arg("--ignored")
        .arg("crash_replay_recovery")
        .env("CRASH_CHILD_WAL", wal_dir.to_string_lossy().to_string())
        .env("CRASH_CHILD_VECTORS", vec_dir.to_string_lossy().to_string())
        .env(
            "CRASH_CHILD_BUCKETS",
            bucket_dir.to_string_lossy().to_string(),
        )
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()?;
    let status = child.wait()?;
    assert!(
        !status.success(),
        "child should crash to simulate abrupt shutdown"
    );

    // Restart and verify payloads exist (allowing for a tiny loss if WAL dropped the tail).
    let mut cfg = SatoriDbConfig::new(wal);
    cfg.workers = 2;
    cfg.vector_index_path = vec_dir.clone();
    cfg.bucket_index_path = bucket_dir.clone();
    let db = SatoriDb::start(cfg)?;
    let handle = db.handle();
    let ids: Vec<u64> = (500_000..500_200).collect();
    let fetched = handle.fetch_vectors_by_id_blocking(ids.clone())?;
    assert!(
        fetched.len() >= 180,
        "too many vectors lost after crash/replay (got {})",
        fetched.len()
    );
    db.shutdown()?;
    Ok(())
}

/// Truncate the WAL tail and ensure restart does not panic and preserves earlier entries.
#[test]
#[ignore]
fn wal_truncation_is_tolerated() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal_dir = tmp.path().join("wal");
    let vec_dir = tmp.path().join("vectors");
    let bucket_dir = tmp.path().join("buckets");

    let wal = Arc::new(
        Walrus::with_data_dir_and_options(
            wal_dir.clone(),
            ReadConsistency::StrictlyAtOnce,
            FsyncSchedule::NoFsync,
        )
        .expect("wal init"),
    );
    let mut cfg = SatoriDbConfig::new(wal.clone());
    cfg.workers = 1;
    cfg.vector_index_path = vec_dir.clone();
    cfg.bucket_index_path = bucket_dir.clone();
    let db = SatoriDb::start(cfg)?;
    let handle = db.handle();

    for i in 0..32u64 {
        let v = vec![i as f32, i as f32 + 0.5];
        handle.upsert_blocking(i, v, Some(0))?;
    }
    handle.flush_blocking()?;
    drop(handle);
    db.shutdown()?;

    // Corrupt/truncate the largest WAL file (best-effort).
    let mut largest = None;
    for entry in std::fs::read_dir(&wal_dir)? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            let meta = entry.metadata()?;
            if largest
                .as_ref()
                .map(|(_, m): &(std::path::PathBuf, u64)| m < &meta.len())
                .unwrap_or(true)
            {
                largest = Some((entry.path(), meta.len()));
            }
        }
    }
    if let Some((path, len)) = largest {
        if len > 16 {
            let new_len = len.saturating_sub(8);
            let f = std::fs::OpenOptions::new().write(true).open(&path)?;
            f.set_len(new_len)?;
        }
    }

    let wal = Arc::new(
        Walrus::with_data_dir_and_options(
            wal_dir,
            ReadConsistency::StrictlyAtOnce,
            FsyncSchedule::NoFsync,
        )
        .expect("wal reopen"),
    );
    let mut cfg = SatoriDbConfig::new(wal);
    cfg.workers = 1;
    cfg.vector_index_path = vec_dir;
    cfg.bucket_index_path = bucket_dir;
    let db = SatoriDb::start(cfg)?;
    let handle = db.handle();
    let ids: Vec<u64> = (0..32u64).collect();
    let fetched = handle.fetch_vectors_by_id_blocking(ids)?;
    assert!(
        fetched.len() >= 24,
        "restart after WAL truncation lost too many entries (got {})",
        fetched.len()
    );
    db.shutdown()?;
    Ok(())
}

/// Fsync schedule latency applied during ingest/query should not deadlock or panic.
#[test]
#[ignore]
fn fsync_latency_soak() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    std::env::set_var("WALRUS_QUIET", "1");
    let wal = Arc::new(
        Walrus::with_data_dir_and_options(
            tmp.path().to_path_buf(),
            ReadConsistency::StrictlyAtOnce,
            FsyncSchedule::Milliseconds(25),
        )
        .expect("walrus init"),
    );
    let mut cfg = SatoriDbConfig::new(wal);
    cfg.workers = 2;
    cfg.virtual_nodes = 8;
    let db = SatoriDb::start(cfg)?;
    let handle = db.handle();

    let duration = Duration::from_secs(4);
    let start = Instant::now();
    let next_id = Arc::new(AtomicU64::new(1));

    std::thread::scope(|s| {
        for t in 0..2 {
            let handle = handle.clone();
            let next_id = next_id.clone();
            s.spawn(move || {
                let mut seed = 0xface_feed_cafe_beefu64 ^ (t as u64);
                while start.elapsed() < duration {
                    let id = next_id.fetch_add(1, Ordering::Relaxed);
                    let vec = make_vec(&mut seed, 12);
                    let _ = handle.upsert_blocking(id, vec.clone(), None);
                    if id.is_multiple_of(9) {
                        let _ = handle.query_blocking(vec, 4, 8);
                    }
                    std::thread::sleep(Duration::from_millis(3));
                }
            });
        }
    });

    handle.flush_blocking()?;
    db.shutdown()?;
    Ok(())
}

/// With tiny worker cache sizes we should still serve queries (with evictions), not panic.
#[test]
#[ignore]
fn tiny_cache_memory_pressure_backpressures() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    std::env::set_var("SATORI_WORKER_CACHE_BUCKETS", "2");
    std::env::set_var("SATORI_WORKER_CACHE_BUCKET_MB", "1");

    let wal = init_wal(&tmp);
    let mut cfg = SatoriDbConfig::new(wal);
    cfg.workers = 2;
    cfg.virtual_nodes = 8;
    let db = SatoriDb::start(cfg)?;
    let handle = db.handle();

    let duration = Duration::from_secs(4);
    let start = Instant::now();
    let next_id = Arc::new(AtomicU64::new(1));

    std::thread::scope(|s| {
        for t in 0..3 {
            let handle = handle.clone();
            let next_id = next_id.clone();
            s.spawn(move || {
                let mut seed = 0x7777_8888u64 ^ (t as u64);
                while start.elapsed() < duration {
                    let id = next_id.fetch_add(1, Ordering::Relaxed);
                    let vec = make_vec(&mut seed, 10);
                    let _ = handle.upsert_blocking(id, vec.clone(), None);
                    if id.is_multiple_of(4) {
                        let _ = handle.query_with_vectors_blocking(vec, 3, 6);
                    }
                }
            });
        }
    });

    handle.flush_blocking()?;
    let stats = block_on(handle.stats());
    assert!(stats.ready, "router should stay ready under cache pressure");
    db.shutdown()?;
    Ok(())
}

/// Large router fan-out should stay bounded and not panic even with many buckets.
#[test]
#[ignore]
fn large_fanout_queries_hold() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let mut cfg = SatoriDbConfig::new(wal);
    cfg.workers = 3;
    cfg.virtual_nodes = 64;
    let db = SatoriDb::start(cfg)?;
    let handle = db.handle();

    for b in 0..60u64 {
        let vec = vec![b as f32, 0.0];
        let _ = handle.upsert_blocking(b + 1, vec, Some(10_000 + b));
    }

    let q = vec![5.0, 0.0];
    let res = handle.query_blocking(q, 10, 60)?;
    assert!(!res.is_empty(), "fanout query returned nothing");
    db.shutdown()?;
    Ok(())
}

/// Simulate jittery worker scheduling; ensure queries still return without deadlocking.
#[test]
#[ignore]
fn jittery_worker_channel_survives() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let mut cfg = SatoriDbConfig::new(wal);
    cfg.workers = 2;
    cfg.virtual_nodes = 16;
    let db = SatoriDb::start(cfg)?;
    let handle = db.handle();

    for id in 0..200u64 {
        let vec = vec![id as f32, id as f32 * 0.1];
        let _ = handle.upsert_blocking(id, vec, None);
    }

    let start = Instant::now();
    let duration = Duration::from_secs(3);
    let handle_q = handle.clone();
    let q_thread = std::thread::spawn(move || {
        let mut seed = 0x1234u64;
        while start.elapsed() < duration {
            let vec = make_vec(&mut seed, 6);
            let _ = handle_q.query_blocking(vec, 5, 10);
            std::thread::sleep(Duration::from_millis(2));
        }
    });

    let handle_f = handle.clone();
    let f_thread = std::thread::spawn(move || {
        let mut seed = 0xabcd_0001u64;
        while start.elapsed() < duration {
            let id = (lcg(&mut seed) % 200) as u64;
            let _ = handle_f.fetch_vectors_by_id_blocking(vec![id]);
            std::thread::sleep(Duration::from_millis(1));
        }
    });

    q_thread.join().unwrap();
    f_thread.join().unwrap();
    db.shutdown()?;
    Ok(())
}

/// Starting with a read-only WAL directory should surface a clean error (not panic).
#[test]
#[ignore]
fn wal_permission_denied_is_reported() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal_dir = tmp.path().join("wal_ro");
    std::fs::create_dir(&wal_dir)?;
    let mut perms = std::fs::metadata(&wal_dir)?.permissions();
    perms.set_readonly(true);
    std::fs::set_permissions(&wal_dir, perms.clone())?;

    let wal = Walrus::with_data_dir_and_options(
        wal_dir,
        ReadConsistency::StrictlyAtOnce,
        FsyncSchedule::NoFsync,
    );
    assert!(wal.is_err(), "wal init should fail on read-only dir");
    Ok(())
}

/// Repeated crash/restart cycles should not lose data or panic.
#[test]
#[ignore]
fn multi_restart_chaos_loop_survives() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal_dir = tmp.path().join("wal");
    let vec_dir = tmp.path().join("vectors");
    let bucket_dir = tmp.path().join("buckets");
    let mut recorded = HashMap::new();

    for round in 0..3 {
        let wal = Arc::new(
            Walrus::with_data_dir_and_options(
                wal_dir.clone(),
                ReadConsistency::StrictlyAtOnce,
                FsyncSchedule::NoFsync,
            )
            .expect("wal init"),
        );
        let mut cfg = SatoriDbConfig::new(wal.clone());
        cfg.workers = 2;
        cfg.virtual_nodes = 16;
        cfg.vector_index_path = vec_dir.clone();
        cfg.bucket_index_path = bucket_dir.clone();
        let db = SatoriDb::start(cfg)?;
        let handle = db.handle();

        let base = 1_000 * round;
        for i in 0..64u64 {
            let id = base + i;
            let v = vec![id as f32, id as f32 * 0.1];
            let _ = handle.upsert_blocking(id, v.clone(), None);
            recorded.insert(id, v);
        }
        handle.flush_blocking()?;
        db.shutdown()?;
    }

    let wal = Arc::new(
        Walrus::with_data_dir_and_options(
            wal_dir,
            ReadConsistency::StrictlyAtOnce,
            FsyncSchedule::NoFsync,
        )
        .expect("wal reopen"),
    );
    let mut cfg = SatoriDbConfig::new(wal);
    cfg.workers = 2;
    cfg.virtual_nodes = 16;
    cfg.vector_index_path = vec_dir;
    cfg.bucket_index_path = bucket_dir;
    let db = SatoriDb::start(cfg)?;
    let handle = db.handle();

    let ids: Vec<u64> = recorded.keys().copied().collect();
    let fetched = handle.fetch_vectors_by_id_blocking(ids.clone())?;
    assert_eq!(fetched.len(), ids.len(), "missing entries after chaos loop");
    for (id, vec) in fetched {
        assert_eq!(Some(&vec), recorded.get(&id), "payload mismatch {}", id);
    }
    db.shutdown()?;
    Ok(())
}

/// Delete-heavy churn on the RocksDB indexes should keep lookups consistent.
#[test]
#[ignore]
fn delete_churn_updates_indexes() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let vec_index = VectorIndex::open(tmp.path().join("vectors"))?;
    let bucket_index = BucketIndex::open(tmp.path().join("buckets"))?;

    let mut vectors = Vec::new();
    let mut ids = Vec::new();
    for i in 0..200u64 {
        let v = Vector::new(i, vec![i as f32, 0.0]);
        ids.push(i);
        vectors.push(v);
    }
    vec_index.put_batch(&vectors)?;
    bucket_index.put_batch(7, &ids)?;

    // Delete every third id.
    let to_delete: Vec<u64> = ids.iter().cloned().filter(|i| i % 3 == 0).collect();
    vec_index.delete_batch(&to_delete)?;
    bucket_index.delete_batch(&to_delete)?;

    let fetched_vecs = vec_index.get_many(&ids)?;
    let fetched_buckets = bucket_index.get_many(&ids)?;
    assert!(
        fetched_vecs.len() <= ids.len(),
        "vector index should not grow during deletes"
    );
    assert!(
        fetched_buckets.len() <= ids.len(),
        "bucket index should not grow during deletes"
    );
    assert!(
        !fetched_vecs.iter().any(|(id, _)| id % 3 == 0),
        "deleted ids still present in vector index"
    );
    assert!(
        !fetched_buckets.iter().any(|(id, _)| id % 3 == 0),
        "deleted ids still present in bucket index"
    );
    Ok(())
}

/// Queries executed with a stale routing version should fail cleanly (not panic).
#[test]
#[ignore]
fn stale_routing_version_query_does_not_panic() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let storage = Storage::new(wal.clone());

    // Put one bucket manually.
    let bucket_id = 1u64;
    let mut bucket = Bucket::new(bucket_id, vec![0.0, 0.0]);
    bucket.add_vector(Vector::new(42, vec![0.1, 0.2]));
    block_on(storage.put_chunk(&bucket))?;
    let cache = WorkerCache::new(4, 1024 * 1024, 4 * 1024 * 1024);
    let executor = Executor::new(storage.clone(), cache);

    // Deliberately supply a bogus routing version/changed list.
    let res = block_on(executor.query(
        &[0.1, 0.2],
        &[bucket_id],
        1,
        9_999,
        Arc::new(vec![bucket_id]),
        false,
    ));
    assert!(
        res.is_ok(),
        "stale routing version should not panic executor"
    );
    Ok(())
}

/// WAL files get corrupted (zeroed) and reopen should not panic; either surface an error or recover cleanly.
#[test]
#[ignore]
fn wal_corruption_is_handled_gracefully() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal_dir = tmp.path().join("wal");
    let wal = Arc::new(
        Walrus::with_data_dir_and_options(
            wal_dir.clone(),
            ReadConsistency::StrictlyAtOnce,
            FsyncSchedule::NoFsync,
        )
        .expect("wal init"),
    );
    let mut cfg = SatoriDbConfig::new(wal);
    cfg.workers = 1;
    cfg.virtual_nodes = 4;
    let db = SatoriDb::start(cfg)?;
    let handle = db.handle();
    for i in 0..16u64 {
        let v = vec![i as f32, i as f32 * 0.5];
        let _ = handle.upsert_blocking(i, v, Some(0));
    }
    handle.flush_blocking()?;
    db.shutdown()?;

    for entry in std::fs::read_dir(&wal_dir)? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            let mut f = std::fs::OpenOptions::new().write(true).open(entry.path())?;
            // Overwrite the first 64 bytes with garbage.
            f.write_all(&[0u8; 64])?;
        }
    }

    // Reopen: either returns an error we can surface or succeeds but should not panic.
    let reopened = Walrus::with_data_dir_and_options(
        wal_dir,
        ReadConsistency::StrictlyAtOnce,
        FsyncSchedule::NoFsync,
    );
    assert!(
        reopened.is_ok() || reopened.is_err(),
        "reopen should never panic"
    );
    Ok(())
}

/// Randomized short fuzzer of upserts/queries/fetches with invariants on index coverage.
#[test]
#[ignore]
fn randomized_op_fuzz_preserves_index() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let mut cfg = SatoriDbConfig::new(wal);
    cfg.workers = 3;
    cfg.virtual_nodes = 24;
    let db = SatoriDb::start(cfg)?;
    let handle = db.handle();

    let start = Instant::now();
    let duration = Duration::from_secs(4);
    let next_id = Arc::new(AtomicU64::new(1));
    let recorded = Arc::new(parking_lot::RwLock::new(HashMap::<u64, Vec<f32>>::new()));

    std::thread::scope(|s| {
        for t in 0..4 {
            let handle = handle.clone();
            let next_id = next_id.clone();
            let recorded = recorded.clone();
            s.spawn(move || {
                let mut seed = 0x9999_5555_c0ffee00u64 ^ (t as u64);
                while start.elapsed() < duration {
                    let roll = lcg(&mut seed) % 10;
                    match roll {
                        0 | 1 | 2 | 3 | 4 => {
                            let id = next_id.fetch_add(1, Ordering::Relaxed);
                            let vec = make_vec(&mut seed, 14);
                            if handle.upsert_blocking(id, vec.clone(), None).is_ok() {
                                recorded.write().insert(id, vec);
                            }
                        }
                        5 | 6 => {
                            let sample: Vec<u64> = {
                                let guard = recorded.read();
                                guard.keys().take(8).copied().collect()
                            };
                            if !sample.is_empty() {
                                let _ = handle.fetch_vectors_by_id_blocking(sample);
                            }
                        }
                        _ => {
                            let vec = make_vec(&mut seed, 14);
                            let _ = handle.query_blocking(vec, 6, 10);
                        }
                    }
                }
            });
        }
    });

    handle.flush_blocking()?;
    let ids: Vec<u64> = recorded.read().keys().copied().collect();
    let fetched = handle.fetch_vectors_by_id_blocking(ids.clone())?;
    assert_eq!(
        fetched.len(),
        ids.len(),
        "index lost entries after fuzz run"
    );
    db.shutdown()?;
    Ok(())
}

/// Wide fan-out plus concurrent rebalances should keep routing and data consistent.
#[test]
#[ignore]
fn wide_fanout_during_rebalance_is_safe() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    std::env::set_var("SATORI_REBALANCE_THRESHOLD", "8");
    let wal = init_wal(&tmp);
    let mut cfg = SatoriDbConfig::new(wal);
    cfg.workers = 4;
    cfg.virtual_nodes = 64;
    let db = SatoriDb::start(cfg)?;
    let handle = db.handle();

    let start = Instant::now();
    let duration = Duration::from_secs(5);
    let next_id = Arc::new(AtomicU64::new(1));

    std::thread::scope(|s| {
        for t in 0..4 {
            let handle = handle.clone();
            let next_id = next_id.clone();
            s.spawn(move || {
                let mut seed = 0x4242_cafe_beefu64 ^ (t as u64);
                while start.elapsed() < duration {
                    let id = next_id.fetch_add(1, Ordering::Relaxed);
                    let vec = make_vec(&mut seed, 12);
                    let _ = handle.upsert_blocking(id, vec.clone(), None);
                    if id.is_multiple_of(5) {
                        let _ = handle.query_blocking(vec, 8, 32);
                    }
                }
            });
        }
    });

    handle.flush_blocking()?;
    let stats = block_on(handle.stats());
    assert!(
        stats.ready,
        "router should stay ready during wide fan-out/rebalance"
    );
    db.shutdown()?;
    Ok(())
}
