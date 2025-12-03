mod bvecs;
mod executor;
mod fvecs;
mod gnd;
mod indexer;
mod quantizer;
mod router;
mod storage;
mod wal;
mod worker;

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
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;
use tar::Archive;

use crate::bvecs::BvecsReader;
use crate::fvecs::FvecsReader;
use crate::gnd::GndReader;
use crate::indexer::Indexer;
use crate::quantizer::Quantizer;
use crate::router::Router;
use crate::storage::{Storage, Vector};
use crate::wal::runtime::Walrus;
use crate::wal::{FsyncSchedule, ReadConsistency};
use crate::worker::{run_worker, QueryRequest, WorkerMessage};

#[derive(Clone, Copy, Debug)]
enum DatasetKind {
    Bigann,
    Gist,
}

struct DatasetConfig {
    kind: DatasetKind,
    base_path: PathBuf,
    query_path: PathBuf,
    gnd_path: Option<PathBuf>,
    max_vectors: usize,
}

struct RouterTask {
    query_vec: Vec<u8>,
    top_k: usize,
    respond_to: oneshot::Sender<anyhow::Result<Vec<u64>>>,
}

struct ConsistentHashRing {
    ring: Vec<(u64, usize)>,
}

impl ConsistentHashRing {
    fn new(nodes: usize, virtual_nodes: usize) -> Self {
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

    fn node_for(&self, key: u64) -> usize {
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

fn trigger_indexing(
    router_shared: &Arc<parking_lot::RwLock<Option<Router>>>,
) -> anyhow::Result<()> {
    if router_shared.read().is_some() {
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
        return Ok(Some(DatasetConfig {
            kind: DatasetKind::Gist,
            base_path: gist_base,
            query_path: gist_query,
            gnd_path: if gist_tar.exists() {
                Some(gist_tar)
            } else {
                None
            },
            max_vectors: 1_000_000usize,
        }));
    }

    let bigann_base = PathBuf::from("bigann_base.bvecs.gz");
    let bigann_query = PathBuf::from("bigann_query.bvecs.gz");
    let bigann_gnd = PathBuf::from("bigann_gnd.tar.gz");

    if bigann_base.exists() && bigann_query.exists() {
        return Ok(Some(DatasetConfig {
            kind: DatasetKind::Bigann,
            base_path: bigann_base,
            query_path: bigann_query,
            gnd_path: if bigann_gnd.exists() {
                Some(bigann_gnd)
            } else {
                None
            },
            max_vectors: 100_000_000usize,
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
    let router_pool = ((satori_cores as f32) * 0.1).round() as usize;
    let router_pool = router_pool.max(1);
    let executor_shards = (satori_cores.saturating_sub(router_pool)).max(1);
    info!(
        "SATORI_CORES={}, router_pool={}, executor_shards={}",
        satori_cores, router_pool, executor_shards
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
        let handle = thread::spawn(move || {
            let builder =
                LocalExecutorBuilder::new(Placement::Unbound).name(&format!("worker-{}", i));

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
    let router_shared: Arc<parking_lot::RwLock<Option<Router>>> =
        Arc::new(parking_lot::RwLock::new(None));
    let mut router_handles = Vec::new();
    for i in 0..router_pool {
        let rx = router_rx.clone();
        let router_clone = router_shared.clone();
        let handle = thread::spawn(move || {
            for task in rx.iter() {
                let res = router_clone
                    .read()
                    .as_ref()
                    .ok_or_else(|| anyhow!("router not initialized"))?
                    .query(&task.query_vec, task.top_k)
                    .map_err(|e| anyhow!("router {} query failed: {:?}", i, e));
                let _ = task.respond_to.send(res);
            }
            Ok::<(), anyhow::Error>(())
        });
        router_handles.push(handle);
    }

    let run_benchmark = std::env::var("SATORI_RUN_BENCH").is_ok();
    if run_benchmark {
        block_on(run_benchmark_mode(
            wal.clone(),
            router_tx.clone(),
            router_shared.clone(),
            ring,
            senders.clone(),
        ))?;
    } else {
        info!("Benchmark logic is disabled (set SATORI_RUN_BENCH=1 to enable).");
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
    router_shared: Arc<parking_lot::RwLock<Option<Router>>>,
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
    }

    impl Reader {
        fn read_batch(&mut self, batch_size: usize) -> anyhow::Result<Vec<Vector>> {
            match self {
                Reader::Bvecs(r) => r.read_batch(batch_size),
                Reader::Fvecs(r) => r.read_batch(batch_size),
            }
        }
    }

    // Dataset-specific knobs
    let (initial_batch_size, stream_batch_size, k, router_top_k) = match dataset.kind {
        DatasetKind::Bigann => (100_000, 100_000, 1000, 50),
        // Keep GIST tractable: smaller initial batch and moderate k.
        DatasetKind::Gist => (50_000, 100_000, 2_000, 400),
    };

    if dataset.base_path.exists() {
        let mut reader = match dataset.kind {
            DatasetKind::Bigann => Reader::Bvecs(BvecsReader::new(&dataset.base_path)?),
            DatasetKind::Gist => Reader::Fvecs(FvecsReader::new(&dataset.base_path)?),
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

        let buckets = Indexer::build_clusters(vectors, k);
        info!("Indexer created {} buckets.", buckets.len());

        let centroids: Vec<Vec<f32>> = buckets.iter().map(|b| b.centroid.clone()).collect();
        let (min, max) = Quantizer::compute_bounds(&centroids);
        let quantizer = Quantizer::new(min, max);

        let mut r = Router::new(100_000, quantizer.clone());
        for bucket in &buckets {
            storage.put_chunk(bucket).await?;
            r.add_centroid(bucket.id, &bucket.centroid);
        }
        *router_shared.write() = Some(r);
        info!("Initial indexing complete.");

        // Streaming Ingestion
        let mut total_processed = initial_batch_size;

        while total_processed < dataset.max_vectors {
            let batch = reader.read_batch(stream_batch_size)?;
            if batch.is_empty() {
                break;
            }

            info!("Ingesting batch (total: {})...", total_processed);

            let mut updates: HashMap<u64, Vec<Vector>> = HashMap::new();

            for vec in batch {
                let ids = {
                    let guard = router_shared.read();
                    let r_ref = guard.as_ref().ok_or_else(|| anyhow!("router missing"))?;
                    r_ref.query(&vec.data, 1)
                };
                if let Ok(ids) = ids {
                    if !ids.is_empty() {
                        updates.entry(ids[0]).or_default().push(vec);
                    }
                }
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
            let bucket_ids = rx.await.map_err(|e| anyhow!("router recv: {:?}", e))??;

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
