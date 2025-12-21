use std::collections::HashMap;
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
                        if id % 8 == 0 {
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
                        if id % 5 == 0 {
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
    let mut cfg = SatoriDbConfig::new(wal);
    cfg.workers = std::cmp::min(4, num_cpus::get().max(1));
    cfg.vector_index_path = vector_path;
    cfg.bucket_index_path = bucket_path;
    let db = SatoriDb::start(cfg)?;
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
                        if id % 6 == 0 {
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
                    if id % 7 == 0 {
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
