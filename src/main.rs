mod bvecs;
mod executor;
mod fvecs;
mod gnd;
mod indexer;
mod ingest_counter;
mod net;
mod quantizer;
mod router_hnsw;
mod rebalancer;
mod router;
mod storage;
mod wal;
mod worker;
mod flatbin;

use anyhow::anyhow;
use crossbeam_channel::{unbounded, Sender};
use flate2::read::GzDecoder;
use futures::channel::oneshot;
use futures::executor::block_on;
use futures::future::join_all;
use glommio::{LocalExecutorBuilder, Placement};
use log::info;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;
use tar::Archive;

use crate::bvecs::BvecsReader;
use crate::flatbin::FlatF32Reader;
use crate::fvecs::FvecsReader;
use crate::gnd::GndReader;
use crate::indexer::Indexer;
use crate::net::spawn_network_server;
use crate::rebalancer::RebalanceWorker;
use crate::router::RoutingTable;
use crate::storage::{Storage, Vector};
use crate::wal::runtime::Walrus;
use crate::wal::{FsyncSchedule, ReadConsistency};
use crate::worker::{run_worker, QueryRequest, WorkerMessage};

#[derive(Clone, Copy, Debug)]
enum DatasetKind {
    Bigann,
    Gist,
}

const DEFAULT_REBALANCE_INTERVAL_SECS: u64 = 1;

struct DatasetConfig {
    kind: DatasetKind,
    base_path: PathBuf,
    query_path: PathBuf,
    gnd_path: Option<PathBuf>,
    max_vectors: usize,
    base_is_prepared: bool,
}

pub struct RouterTask {
    pub query_vec: Vec<f32>,
    pub top_k: usize,
    pub respond_to: oneshot::Sender<anyhow::Result<RouterResult>>,
}

#[derive(Clone)]
pub struct RouterResult {
    pub bucket_ids: Vec<u64>,
    pub routing_version: u64,
    pub affected_buckets: std::sync::Arc<Vec<u64>>,
}

#[derive(Clone)]
pub struct ConsistentHashRing {
    ring: Vec<(u64, usize)>,
}

impl ConsistentHashRing {
    pub fn new(nodes: usize, virtual_nodes: usize) -> Self {
        let mut ring = Vec::new();
        for node in 0..nodes {
            for v in 0..virtual_nodes.max(1) {
                let mut hasher = DefaultHasher::new();
                (node as u64, v as u64).hash(&mut hasher);
                ring.push((hasher.finish(), node));
            }
        }
        ring.sort_by_key(|(h, _)| *h);
        Self { ring }
    }

    pub fn node_for(&self, key: u64) -> usize {
        if self.ring.is_empty() {
            return 0;
        }
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        let h = hasher.finish();
        for (hash, node) in &self.ring {
            if *hash >= h {
                return *node;
            }
        }
        self.ring[0].1
    }
}

#[cfg(test)]
mod tests {
    use super::ConsistentHashRing;

    #[test]
    fn hash_ring_is_deterministic() {
        let ring = ConsistentHashRing::new(4, 8);
        assert_eq!(ring.node_for(42), ring.node_for(42));
    }

    #[test]
    fn hash_ring_respects_node_bounds() {
        let nodes = 3;
        let ring = ConsistentHashRing::new(nodes, 4);
        for key in 0..100 {
            let node = ring.node_for(key);
            assert!(node < nodes);
        }
    }
}

fn ensure_gist_files(tar_path: &Path) -> anyhow::Result<()> {
    let base_path = Path::new("gist/gist_base.fvecs");
    let query_path = Path::new("gist/gist_query.fvecs");

    if base_path.exists() && query_path.exists() {
        return Ok(());
    }

    if !tar_path.exists() {
        return Err(anyhow!("gist.tar.gz not found at {}", tar_path.display()));
    }

    info!(
        "Extracting GIST1M base/query from archive {}...",
        tar_path.display()
    );
    fs::create_dir_all("gist")?;
    let file = File::open(tar_path)?;
    let decoder = GzDecoder::new(file);
    let mut archive = Archive::new(decoder);

    let mut extracted = 0;
    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?;
        if path == Path::new("gist/gist_base.fvecs") || path == Path::new("gist/gist_query.fvecs") {
            entry.unpack(".")?;
            extracted += 1;
        }
    }

    if base_path.exists() && query_path.exists() {
        Ok(())
    } else {
        Err(anyhow!(
            "Failed to extract gist_base.fvecs or gist_query.fvecs (extracted {} entries)",
            extracted
        ))
    }
}

fn trigger_indexing(router_shared: &Arc<RoutingTable>) -> anyhow::Result<()> {
    if router_shared.snapshot().is_some() {
        info!("Router already built; indexing trigger is a no-op.");
        Ok(())
    } else {
        Err(anyhow!(
            "Router not initialized; indexing step must run first"
        ))
    }
}

fn detect_dataset() -> anyhow::Result<Option<DatasetConfig>> {
    let gist_base = PathBuf::from("gist/gist_base.fvecs");
    let gist_query = PathBuf::from("gist/gist_query.fvecs");
    let gist_tar = PathBuf::from("gist.tar.gz");

    if (gist_base.exists() && gist_query.exists()) || gist_tar.exists() {
        if (!gist_base.exists() || !gist_query.exists()) && gist_tar.exists() {
            ensure_gist_files(&gist_tar)?;
        }
        let prepared_base = gist_base.with_extension("f32bin");
        let base_path = if prepared_base.exists() {
            prepared_base.clone()
        } else {
            gist_base.clone()
        };
        return Ok(Some(DatasetConfig {
            kind: DatasetKind::Gist,
            base_path,
            query_path: gist_query,
            gnd_path: if gist_tar.exists() {
                Some(gist_tar)
            } else {
                None
            },
            max_vectors: 1_000_000usize,
            base_is_prepared: prepared_base.exists(),
        }));
    }

    let bigann_base = PathBuf::from("bigann_base.bvecs.gz");
    let bigann_query = PathBuf::from("bigann_query.bvecs.gz");
    let bigann_gnd = PathBuf::from("bigann_gnd.tar.gz");
    let prepared_bigann = bigann_base.with_extension("f32bin");

    if bigann_base.exists() && bigann_query.exists() {
        return Ok(Some(DatasetConfig {
            kind: DatasetKind::Bigann,
            base_path: if prepared_bigann.exists() {
                prepared_bigann.clone()
            } else {
                bigann_base.clone()
            },
            query_path: bigann_query,
            gnd_path: if bigann_gnd.exists() {
                Some(bigann_gnd)
            } else {
                None
            },
            max_vectors: 100_000_000usize,
            base_is_prepared: prepared_bigann.exists(),
        }));
    }

    Ok(None)
}

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let satori_cores: usize = std::env::var("SATORI_CORES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| num_cpus::get().max(1));
    // Reserve two cores for background threads (rebalance driver/worker, logging, etc.).
    let usable_cores = satori_cores.saturating_sub(2).max(1);
    let total_cpus = num_cpus::get().max(1);
    let reserved_start = usable_cores.min(total_cpus);
    let mut reserved_cores: Vec<usize> = (reserved_start..total_cpus).collect();
    let rebalance_core = reserved_cores.pop();

    let router_pool = ((usable_cores as f32) * 0.2).round() as usize;
    let router_pool = router_pool.max(1);
    let executor_shards = (usable_cores.saturating_sub(router_pool)).max(1);
    info!(
        "SATORI_CORES={}, usable_cores={}, router_pool={}, executor_shards={}",
        satori_cores, usable_cores, router_pool, executor_shards
    );

    let wal = Arc::new(Walrus::with_consistency_and_schedule(
        ReadConsistency::StrictlyAtOnce,
        FsyncSchedule::NoFsync,
    )?);

    // Create Worker Channels (async for backpressure)
    let mut senders = Vec::new();
    let mut worker_handles = Vec::new();

    for i in 0..executor_shards {
        let (sender, receiver) = async_channel::bounded(1000);
        senders.push(sender);
        let wal_clone = wal.clone();
        let pin_cpu = i % usable_cores;
        let handle = thread::spawn(move || {
            let builder =
                LocalExecutorBuilder::new(Placement::Fixed(pin_cpu)).name(&format!("worker-{}", i));

            let result = builder
                .make()
                .expect("failed to create executor")
                .run(run_worker(i, receiver, wal_clone));

            result
        });
        worker_handles.push(handle);
    }

    let ring = ConsistentHashRing::new(executor_shards, 8);

    // Router pool using crossbeam MPMC; router is installed after clustering.
    let (router_tx, router_rx) = unbounded::<RouterTask>();
    let router_shared = Arc::new(RoutingTable::new());
    let rebalance_worker = RebalanceWorker::spawn(
        Storage::new(wal.clone()),
        router_shared.clone(),
        rebalance_core,
    );
    if let Some(core) = rebalance_core {
        info!("Rebalance worker pinned to reserved core {}", core);
    } else {
        info!("Rebalance worker running without dedicated core (only one CPU visible)");
    }

    // Optional periodic rebalance driver.
    let secs = std::env::var("SATORI_REBALANCE_INTERVAL_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(DEFAULT_REBALANCE_INTERVAL_SECS);
    let target_size = std::env::var("SATORI_BUCKET_TARGET_SIZE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(10_000);
    let worker = rebalance_worker.clone();
    let merge_interval_ticks = secs.max(1) * 10;
    std::thread::spawn(move || {
        let mut tick_ctr: u64 = 0;
        loop {
            let sizes = worker.snapshot_sizes();
            let mut sizes_vec: Vec<usize> = sizes.values().cloned().collect();
            sizes_vec.sort_unstable();
            log::debug!("rebalance: periodic tick sizes={:?}", sizes_vec);

            // Faster polling when any bucket is over the split threshold to avoid runaway growth.
            let fast_path = sizes.values().any(|sz| *sz > target_size * 2);
            let sleep_dur = if fast_path {
                std::time::Duration::from_millis(200)
            } else {
                std::time::Duration::from_secs(secs.max(1))
            };

            // Schedule a limited number of heavy tasks per tick to avoid queue explosion.
            // Split only the largest offenders first.
            let mut big: Vec<(u64, usize)> = sizes
                .iter()
                .filter(|(_, sz)| **sz > target_size * 2)
                .map(|(k, v)| (*k, *v))
                .collect();
            big.sort_by_key(|(_, sz)| std::cmp::Reverse(*sz));
            for (bid, _) in big.into_iter().take(1) {
                let _ = worker.enqueue_blocking(crate::rebalancer::RebalanceTask::Split(bid));
            }

            // Merge smallest two buckets if they are under target/2, but only every N ticks.
            if tick_ctr % merge_interval_ticks == 0 {
                let mut small: Vec<(u64, usize)> = sizes
                    .iter()
                    .filter(|(_, sz)| **sz < target_size / 2)
                    .map(|(k, v)| (*k, *v))
                    .collect();
                small.sort_by_key(|(_, sz)| *sz);
                if small.len() >= 2 {
                    let (a, _sa) = small[0];
                    let (b, _sb) = small[1];
                    let _ = worker.enqueue_blocking(crate::rebalancer::RebalanceTask::Merge(a, b));
                }
            }

            std::thread::sleep(sleep_dur);
            tick_ctr = tick_ctr.saturating_add(1);
        }
    });
    let mut router_handles = Vec::new();
    for i in 0..router_pool {
        let rx = router_rx.clone();
        let router_clone = router_shared.clone();
        let handle = thread::spawn(move || {
            let mut local_snapshot: Option<crate::router::RoutingSnapshot> = None;
            for task in rx.iter() {
                if local_snapshot
                    .as_ref()
                    .map(|s| s.version != router_clone.current_version())
                    .unwrap_or(true)
                {
                    local_snapshot = router_clone.snapshot();
                }

                let snapshot = match &local_snapshot {
                    Some(s) => s.clone(),
                    None => {
                        let _ = task.respond_to.send(Err(anyhow!("router not initialized")));
                        continue;
                    }
                };

                let res = snapshot
                    .router
                    .query(&task.query_vec, task.top_k)
                    .map(|bucket_ids| RouterResult {
                        bucket_ids,
                        routing_version: snapshot.version,
                        affected_buckets: snapshot.changed.clone(),
                    })
                    .map_err(|e| anyhow!("router {} query failed: {:?}", i, e));
                let _ = task.respond_to.send(res);
            }
            Ok::<(), anyhow::Error>(())
        });
        router_handles.push(handle);
    }

    let listen_addr = std::env::var("SATORI_LISTEN_ADDR").ok();

    let run_benchmark = std::env::var("SATORI_RUN_BENCH").is_ok();
    if run_benchmark {
        block_on(run_benchmark_mode(
            wal.clone(),
            router_tx.clone(),
            router_shared.clone(),
            rebalance_worker.clone(),
            ring.clone(),
            senders.clone(),
        ))?;
    } else {
        info!("Benchmark logic is disabled (set SATORI_RUN_BENCH=1 to enable).");
    }

    let net_handle = if let Some(addr_str) = listen_addr {
        match addr_str.parse::<SocketAddr>() {
            Ok(addr) => {
                match spawn_network_server(addr, router_tx.clone(), ring.clone(), senders.clone()) {
                    Ok(handle) => Some(handle),
                    Err(e) => {
                        log::error!("Failed to start network server on {}: {:?}", addr, e);
                        None
                    }
                }
            }
            Err(e) => {
                log::error!("Invalid SATORI_LISTEN_ADDR ({}): {:?}", addr_str, e);
                None
            }
        }
    } else {
        None
    };

    if let Some(handle) = net_handle {
        info!("Network server running; press Ctrl+C to exit.");
        let _ = handle.join();
        return Ok(());
    }

    // Close router channel and worker channels for clean shutdown
    drop(router_tx);
    for sender in senders {
        sender.close();
    }

    for h in worker_handles {
        let _ = h.join();
    }
    for h in router_handles {
        let _ = h.join();
    }
    Ok(())
}

async fn run_benchmark_mode(
    wal: Arc<Walrus>,
    router_tx: Sender<RouterTask>,
    router_shared: Arc<RoutingTable>,
    rebalance: RebalanceWorker,
    ring: ConsistentHashRing,
    senders: Vec<async_channel::Sender<WorkerMessage>>,
) -> anyhow::Result<()> {
    let dataset = match detect_dataset()? {
        Some(d) => d,
        None => {
            info!("No dataset detected (bigann_* or gist.tar.gz). Skipping benchmark.");
            return Ok(());
        }
    };
    info!(
        "Dataset: {:?}, base={}, query={}",
        dataset.kind,
        dataset.base_path.display(),
        dataset.query_path.display()
    );

    let storage = Storage::new(wal.clone());

    enum Reader {
        Bvecs(BvecsReader),
        Fvecs(FvecsReader),
        Flat(FlatF32Reader),
    }

    impl Reader {
        fn read_batch(&mut self, batch_size: usize) -> anyhow::Result<Vec<Vector>> {
            match self {
                Reader::Bvecs(r) => r.read_batch(batch_size),
                Reader::Fvecs(r) => r.read_batch(batch_size),
                Reader::Flat(r) => r.read_batch(batch_size),
            }
        }
    }

    // Dataset-specific knobs
    // Start with a larger number of buckets; background rebalance can adjust.
    let (initial_batch_size, stream_batch_size, k, router_top_k) = match dataset.kind {
        DatasetKind::Bigann => (100_000, 100_000, 100, 50),
        DatasetKind::Gist => (50_000, 100_000, 100, 400),
    };

    if dataset.base_path.exists() {
        let mut reader = match dataset.kind {
            DatasetKind::Bigann => {
                if dataset.base_is_prepared {
                    Reader::Flat(FlatF32Reader::new(&dataset.base_path)?)
                } else {
                    Reader::Bvecs(BvecsReader::new(&dataset.base_path)?)
                }
            }
            DatasetKind::Gist => {
                if dataset.base_is_prepared {
                    Reader::Flat(FlatF32Reader::new(&dataset.base_path)?)
                } else {
                    Reader::Fvecs(FvecsReader::new(&dataset.base_path)?)
                }
            }
        };

        info!("Reading initial batch of {} vectors...", initial_batch_size);
        let vectors = reader.read_batch(initial_batch_size)?;
        if vectors.is_empty() {
            return Err(anyhow!("Base file is empty"));
        }
        info!(
            "Clustering initial batch: {} vectors, k={}",
            vectors.len(),
            k
        );

        let total_initial = vectors.len();
        let buckets = Indexer::build_clusters(vectors, k);
        ingest_counter::add(total_initial as u64);
        info!("Indexer created {} buckets.", buckets.len());

        for bucket in &buckets {
            storage.put_chunk(bucket).await?;
        }
        rebalance.prime_centroids(&buckets).await?;
        info!(
            "Initial indexing complete (router version {}).",
            router_shared.current_version()
        );

        // Streaming Ingestion
        let mut total_processed = initial_batch_size;

        while total_processed < dataset.max_vectors {
            let batch = reader.read_batch(stream_batch_size)?;
            if batch.is_empty() {
                break;
            }

            info!("Ingesting batch (total: {})...", total_processed);

            let mut updates: HashMap<u64, Vec<Vector>> = HashMap::new();
            updates.reserve(router_top_k);

            if let Some(snapshot) = router_shared.snapshot() {
                for vec in batch {
                    if let Ok(ids) = snapshot.router.query(&vec.data, 1) {
                        if let Some(&id) = ids.first() {
                            updates
                                .entry(id)
                                .or_insert_with(|| Vec::with_capacity(64))
                                .push(vec);
                        }
                    }
                }
            } else {
                return Err(anyhow!("router missing"));
            }

            for (bucket_id, new_vectors) in updates {
                let shard = ring.node_for(bucket_id);
                let msg = WorkerMessage::Ingest {
                    bucket_id,
                    vectors: new_vectors,
                };
                if let Err(e) = senders[shard].send(msg).await {
                    log::error!("Failed to send ingest to shard {}: {:?}", shard, e);
                }
            }
            total_processed += stream_batch_size;
        }
        info!("Flushing all workers before queries...");
        let mut flush_waiters = Vec::new();
        for sender in &senders {
            let (tx, rx) = oneshot::channel();
            if let Err(e) = sender.send(WorkerMessage::Flush { respond_to: tx }).await {
                log::error!("Failed to send flush: {:?}", e);
            } else {
                flush_waiters.push(rx);
            }
        }
        // Wait for all workers to confirm flush so queries see every vector.
        for rx in flush_waiters {
            let _ = rx.await;
        }
        info!(
            "Ingestion complete. Total vectors ingested or scanned: {}",
            total_processed
        );
    } else {
        info!(
            "{} not found. Skipping Benchmark Ingestion.",
            dataset.base_path.display()
        );
    }

    if dataset.query_path.exists() {
        // Allow manual indexing trigger before queries.
        trigger_indexing(&router_shared)?;

        let mut reader = match dataset.kind {
            DatasetKind::Bigann => Reader::Bvecs(BvecsReader::new(&dataset.query_path)?),
            DatasetKind::Gist => Reader::Fvecs(FvecsReader::new(&dataset.query_path)?),
        };
        let queries = reader.read_batch(100)?;

        let ground_truth = match &dataset.gnd_path {
            Some(path) if path.exists() => {
                info!("Loading Ground Truth from {}...", path.display());
                Some(GndReader::new(path)?)
            }
            Some(path) => {
                info!(
                    "Ground Truth {} not found. Recall will be skipped.",
                    path.display()
                );
                None
            }
            None => {
                info!("Ground Truth path not provided. Recall will be skipped.");
                None
            }
        };

        let mut total_hits_at_10 = 0;
        let mut total_queries_checked = 0;
        let mut total_reachable_gt = 0;

        for (i, q) in queries.iter().enumerate() {
            let (tx, rx) = oneshot::channel();
            let task = RouterTask {
                query_vec: q.data.clone(),
                top_k: router_top_k,
                respond_to: tx,
            };
            router_tx
                .send(task)
                .map_err(|e| anyhow!("router send: {:?}", e))?;
            let router_result = rx.await.map_err(|e| anyhow!("router recv: {:?}", e))??;
            let RouterResult {
                bucket_ids,
                routing_version,
                affected_buckets,
            } = router_result;

            if i == 0 {
                info!(
                    "Debug query[0]: router returned {} bucket ids (top_k={})",
                    bucket_ids.len(),
                    router_top_k
                );
            }

            let mut pending = Vec::new();
            let mut requests: HashMap<usize, Vec<u64>> = HashMap::new();
            for &bid in &bucket_ids {
                let shard = ring.node_for(bid);
                requests.entry(shard).or_default().push(bid);
            }

            for (shard, bids) in requests {
                let (tx, rx) = oneshot::channel();
                let req = QueryRequest {
                    query_vec: q.data.clone(),
                    bucket_ids: bids,
                    routing_version,
                    affected_buckets: affected_buckets.clone(),
                    respond_to: tx,
                };
                let msg = WorkerMessage::Query(req);
                if let Ok(_) = senders[shard].send(msg).await {
                    pending.push(rx);
                }
            }

            let responses = join_all(pending).await;
            let mut all_results = Vec::new();
            for res in responses {
                if let Ok(Ok(candidates)) = res {
                    all_results.extend(candidates);
                }
            }
            if i == 0 {
                info!(
                    "Debug query[0]: collected {} candidates from workers",
                    all_results.len()
                );
            }
            all_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            if let Some(ref gt) = ground_truth {
                if i < gt.ground_truth.len() {
                    let true_neighbors = &gt.ground_truth[i];
                    let gt_top_10 = true_neighbors.iter().take(10).cloned().collect::<Vec<_>>();

                    let reachable = gt_top_10
                        .iter()
                        .filter(|&&id| (id as usize) < dataset.max_vectors)
                        .count();
                    total_reachable_gt += reachable;

                    let my_top_10: Vec<u32> =
                        all_results.iter().take(10).map(|r| r.0 as u32).collect();
                    if i == 0 {
                        info!("Debug query[0]: my_top_10={:?}", my_top_10);
                        info!("Debug query[0]: gt_top_10={:?}", gt_top_10);
                    }
                    let hits = my_top_10.iter().filter(|id| gt_top_10.contains(id)).count();
                    total_hits_at_10 += hits;
                    total_queries_checked += 1;
                }
            }

            if i % 10 == 0 {
                info!("Processed {} queries", i);
            }
        }

        if total_queries_checked > 0 {
            let adjusted_recall = if total_reachable_gt > 0 {
                total_hits_at_10 as f64 / total_reachable_gt as f64
            } else {
                0.0
            };
            info!(
                "Benchmark complete. Queries: {} Adjusted Recall@10: {:.2}%",
                total_queries_checked,
                adjusted_recall * 100.0
            );
        }
    }

    Ok(())
}
