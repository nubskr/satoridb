use crate::bucket_index::BucketIndex;
use crate::bucket_locks::BucketLocks;
use crate::rebalancer::RebalanceWorker;
use crate::router::RoutingTable;
use crate::router_manager::{spawn_router_manager, RouterCommand, RouterShutdownRequest};
use crate::service::SatoriHandle;
use crate::storage::Storage;
use crate::tasks::ConsistentHashRing;
use crate::vector_index::VectorIndex;
use crate::wal::runtime::Walrus;
use crate::worker::{run_worker, WorkerMessage};
use anyhow::{anyhow, Result};
use crossbeam_channel::unbounded;
use futures::channel::oneshot;
use futures::executor::block_on;
use glommio::{LocalExecutorBuilder, Placement};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;

pub struct SatoriDbConfig {
    /// WAL implementation used for durable storage.
    pub wal: Arc<Walrus>,
    /// Number of worker shards (each shard runs on its own Glommio executor thread).
    pub workers: usize,
    /// Virtual nodes used by the consistent hash ring for bucket->shard mapping.
    pub virtual_nodes: usize,
    /// Path for the RocksDB-backed vector index (id -> serialized vector payload).
    pub vector_index_path: PathBuf,
    /// Path for the RocksDB-backed bucket index (id -> bucket mapping).
    pub bucket_index_path: PathBuf,
    /// Shared per-bucket locks to serialize writes vs. rebalances.
    pub bucket_locks: Arc<BucketLocks>,
}

impl SatoriDbConfig {
    /// Create a config with sane defaults.
    pub fn new(wal: Arc<Walrus>) -> Self {
        let vector_index_path = default_vector_index_path();
        let bucket_index_path = default_bucket_index_path();
        let bucket_locks = Arc::new(BucketLocks::new());
        Self {
            wal,
            workers: num_cpus::get().max(1),
            virtual_nodes: 8,
            vector_index_path,
            bucket_index_path,
            bucket_locks,
        }
    }
}

fn default_vector_index_path() -> PathBuf {
    if let Ok(p) = std::env::var("SATORI_VECTOR_INDEX_PATH") {
        return PathBuf::from(p);
    }

    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("vector_index_{}_{}", std::process::id(), n))
}

fn default_bucket_index_path() -> PathBuf {
    if let Ok(p) = std::env::var("SATORI_BUCKET_INDEX_PATH") {
        return PathBuf::from(p);
    }

    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("bucket_index_{}_{}", std::process::id(), n))
}

/// An embedded `satoridb` instance (router manager + worker shards).
pub struct SatoriDb {
    api: SatoriHandle,
    router_tx: crossbeam_channel::Sender<RouterCommand>,
    router_handle: thread::JoinHandle<()>,
    worker_senders: Vec<async_channel::Sender<WorkerMessage>>,
    worker_handles: Vec<thread::JoinHandle<()>>,
    #[allow(dead_code)]
    rebalance_worker: RebalanceWorker,
    rebalance_thread: Option<thread::JoinHandle<()>>,
}

impl SatoriDb {
    /// Start a new embedded database instance.
    ///
    /// This spawns:
    /// - 1 router manager thread
    /// - `cfg.workers` worker threads (each running a Glommio local executor)
    /// - 1 rebalancer thread
    pub fn start(cfg: SatoriDbConfig) -> Result<Self> {
        if cfg.workers == 0 {
            return Err(anyhow!("workers must be > 0"));
        }
        let workers = cfg.workers;
        let vector_index = Arc::new(VectorIndex::open(&cfg.vector_index_path)?);
        let bucket_index = Arc::new(BucketIndex::open(&cfg.bucket_index_path)?);
        let bucket_locks = cfg.bucket_locks.clone();

        let mut worker_senders = Vec::new();
        let mut worker_handles = Vec::new();
        for i in 0..workers {
            let (sender, receiver) = async_channel::bounded(1000);
            worker_senders.push(sender);

            let wal_clone = cfg.wal.clone();
            let index_clone = vector_index.clone();
            let bucket_index_clone = bucket_index.clone();
            let bucket_locks_clone = bucket_locks.clone();
            let pin_cpu = i % num_cpus::get().max(1);
            let handle = thread::spawn(move || {
                let builder = LocalExecutorBuilder::new(Placement::Fixed(pin_cpu))
                    .name(&format!("worker-{}", i));
                builder
                    .make()
                    .expect("failed to create executor")
                    .run(run_worker(
                        i,
                        receiver,
                        wal_clone,
                        index_clone,
                        bucket_index_clone,
                        bucket_locks_clone,
                    ));
            });
            worker_handles.push(handle);
        }

        let ring = ConsistentHashRing::new(workers, cfg.virtual_nodes);

        let (router_tx, router_rx) = unbounded::<RouterCommand>();
        let router_handle = spawn_router_manager(cfg.wal.clone(), router_rx);

        let routing_table = Arc::new(RoutingTable::new());

        let (rebalance_worker, rebalance_thread) = RebalanceWorker::spawn_delete_only(
            Storage::new(cfg.wal.clone()),
            vector_index.clone(),
            bucket_index.clone(),
            routing_table.clone(),
            bucket_locks.clone(),
        );

        let api = SatoriHandle::new(
            router_tx.clone(),
            ring,
            worker_senders.clone(),
            rebalance_worker.delete_tx.clone(),
            vector_index.clone(),
            bucket_index.clone(),
        );

        Ok(Self {
            api,
            router_tx,
            router_handle,
            worker_senders,
            worker_handles,
            rebalance_worker,
            rebalance_thread: Some(rebalance_thread),
        })
    }

    /// Get a cloneable handle for queries and upserts.
    pub fn handle(&self) -> SatoriHandle {
        self.api.clone()
    }

    /// Flush state (best-effort) and shut down router/workers.
    pub fn shutdown(mut self) -> Result<()> {
        let (flush_tx, flush_rx) = oneshot::channel();
        let _ = self.router_tx.send(RouterCommand::Flush(
            crate::router_manager::RouterFlushRequest {
                respond_to: flush_tx,
            },
        ));
        let _ = block_on(flush_rx);

        let (router_shutdown_tx, router_shutdown_rx) = oneshot::channel();
        let _ = self
            .router_tx
            .send(RouterCommand::Shutdown(RouterShutdownRequest {
                respond_to: router_shutdown_tx,
            }));

        let mut worker_shutdown_waiters = Vec::new();
        for sender in &self.worker_senders {
            let (tx, rx) = oneshot::channel();
            let _ = block_on(sender.send(WorkerMessage::Shutdown { respond_to: tx }));
            worker_shutdown_waiters.push(rx);
        }
        for rx in worker_shutdown_waiters {
            let _ = block_on(rx);
        }

        self.rebalance_worker.delete_tx.close();

        for sender in &self.worker_senders {
            sender.close();
        }

        drop(self.api);
        drop(self.router_tx);
        let _ = block_on(router_shutdown_rx);

        for h in self.worker_handles {
            let _ = h.join();
        }
        let _ = self.router_handle.join();

        if let Some(h) = self.rebalance_thread.take() {
            let _ = h.join();
        }

        Ok(())
    }
}
