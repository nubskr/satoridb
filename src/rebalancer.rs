use crate::indexer::Indexer;
use crate::quantizer::Quantizer;
use crate::router::{Router, RoutingTable};
use crate::storage::{Bucket, BucketMeta, BucketMetaStatus, Storage, Vector};
use async_channel::{Receiver, Sender};
use futures::executor::block_on;
use log::{error, info, warn};
use parking_lot::{Mutex, RwLock};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum RebalanceTask {
    Split(u64),
    Merge(u64, u64),
    Rebalance(u64),
}

struct RebalanceState {
    storage: Storage,
    routing: Arc<RoutingTable>,
    centroids: RwLock<HashMap<u64, Vec<f32>>>,
    bucket_sizes: RwLock<HashMap<u64, usize>>,
    retired: RwLock<HashSet<u64>>,
    next_bucket_id: AtomicU64,
    locks: Mutex<HashMap<u64, Arc<Mutex<()>>>>,
}

impl RebalanceState {
    fn new(storage: Storage, routing: Arc<RoutingTable>) -> Self {
        Self {
            storage,
            routing,
            centroids: RwLock::new(HashMap::new()),
            bucket_sizes: RwLock::new(HashMap::new()),
            retired: RwLock::new(HashSet::new()),
            next_bucket_id: AtomicU64::new(0),
            locks: Mutex::new(HashMap::new()),
        }
    }

    fn prime_centroids(&self, buckets: &[Bucket]) {
        let mut map = self.centroids.write();
        let mut sizes = self.bucket_sizes.write();
        let mut max_id = 0;
        for b in buckets {
            map.insert(b.id, b.centroid.clone());
            sizes.insert(b.id, b.vectors.len());
            if b.id > max_id {
                max_id = b.id;
            }
        }
        self.next_bucket_id.store(max_id + 1, Ordering::Release);
    }

    fn allocate_bucket_id(&self) -> u64 {
        self.next_bucket_id.fetch_add(1, Ordering::AcqRel)
    }

    fn lock_for(&self, bucket_id: u64) -> Arc<Mutex<()>> {
        let mut guard = self.locks.lock();
        guard
            .entry(bucket_id)
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    }

    fn load_bucket(&self, bucket_id: u64) -> Option<Bucket> {
        let chunks = block_on(self.storage.get_chunks(bucket_id)).ok()?;
        let mut latest: Option<Bucket> = None;
        for chunk in chunks {
            if chunk.len() < 8 {
                continue;
            }
            let mut len_bytes = [0u8; 8];
            len_bytes.copy_from_slice(&chunk[0..8]);
            let archive_len = u64::from_le_bytes(len_bytes) as usize;
            if 8 + archive_len > chunk.len() {
                continue;
            }
            let data_slice = &chunk[8..8 + archive_len];
            if let Ok(archived_bucket) = rkyv::check_archived_root::<Bucket>(data_slice) {
                let mut bucket = Bucket::new(bucket_id, archived_bucket.centroid.to_vec());
                bucket.vectors = archived_bucket
                    .vectors
                    .iter()
                    .map(|v| Vector {
                        id: v.id,
                        data: v.data.to_vec(),
                    })
                    .collect();
                latest = Some(bucket);
            }
        }
        latest
    }

    fn retire_bucket(&self, bucket_id: u64) {
        self.centroids.write().remove(&bucket_id);
        self.bucket_sizes.write().remove(&bucket_id);
        self.retired.write().insert(bucket_id);
        let _ = block_on(
            self.storage
                .put_bucket_meta(&BucketMeta { bucket_id, status: BucketMetaStatus::Retired }),
        );
    }

    fn rebuild_router(&self) {
        let centroids_map = self.centroids.read();
        if centroids_map.is_empty() {
            return;
        }
        let centroids: Vec<(u64, Vec<f32>)> = centroids_map
            .iter()
            .map(|(id, c)| (*id, c.clone()))
            .collect();
        drop(centroids_map);

        let changed: Vec<u64> = centroids.iter().map(|(id, _)| *id).collect();
        let only_centroids: Vec<Vec<f32>> = centroids.iter().map(|(_, c)| c.clone()).collect();
        let (min, max) = Quantizer::compute_bounds(&only_centroids);
        let quantizer = Quantizer::new(min, max);
        let mut router = Router::new(100_000, quantizer);
        for (id, centroid) in centroids {
            router.add_centroid(id, &centroid);
        }
        let version = self.routing.install(router, changed);
        info!("rebalance: published router version {}", version);
    }

    fn handle_split(&self, bucket_id: u64) {
        let lock = self.lock_for(bucket_id);
        let _guard = lock.lock();
        let bucket = match self.load_bucket(bucket_id) {
            Some(b) => b,
            None => {
                warn!("rebalance: split skipped, bucket {} not found", bucket_id);
                return;
            }
        };
        let splits = Indexer::split_bucket(bucket);
        if splits.is_empty() {
            return;
        }

        let mut new_entries = Vec::new();
        for split in splits {
            let new_id = self.allocate_bucket_id();
            let mut nb = Bucket::new(new_id, split.centroid.clone());
            nb.vectors = split.vectors;
            if let Err(e) = block_on(self.storage.put_chunk(&nb)) {
                error!(
                    "rebalance: failed to persist split bucket {} -> {}: {:?}",
                    bucket_id, new_id, e
                );
                continue;
            }
            let _ = block_on(
                self.storage
                    .put_bucket_meta(&BucketMeta { bucket_id: new_id, status: BucketMetaStatus::Active }),
            );
            new_entries.push((new_id, nb.centroid.clone(), nb.vectors.len()));
        }

        if new_entries.is_empty() {
            return;
        }

        self.retire_bucket(bucket_id);
        let mut centroids = self.centroids.write();
        let mut sizes = self.bucket_sizes.write();
        for (id, centroid, size) in new_entries {
            centroids.insert(id, centroid);
            sizes.insert(id, size);
        }
        drop(sizes);
        drop(centroids);
        self.rebuild_router();
    }

    fn handle_merge(&self, a: u64, b: u64) {
        let (first, second) = if a <= b { (a, b) } else { (b, a) };
        let lock_first = self.lock_for(first);
        let lock_second = self.lock_for(second);
        let _g1 = lock_first.lock();
        let _g2 = lock_second.lock();

        let ba = match self.load_bucket(first) {
            Some(bk) => bk,
            None => {
                warn!("rebalance: merge skipped, bucket {} not found", first);
                return;
            }
        };
        let bb = match self.load_bucket(second) {
            Some(bk) => bk,
            None => {
                warn!("rebalance: merge skipped, bucket {} not found", second);
                return;
            }
        };

        let mut combined = Vec::new();
        combined.extend(ba.vectors);
        combined.extend(bb.vectors);
        let centroid = compute_centroid(&combined);
        let new_id = self.allocate_bucket_id();
        let mut merged = Bucket::new(new_id, centroid.clone());
        merged.vectors = combined;
        let merged_size = merged.vectors.len();
        if let Err(e) = block_on(self.storage.put_chunk(&merged)) {
            error!(
                "rebalance: failed to persist merged bucket {}+{} -> {}: {:?}",
                first, second, new_id, e
            );
            return;
        }
        let _ = block_on(
            self.storage
                .put_bucket_meta(&BucketMeta { bucket_id: new_id, status: BucketMetaStatus::Active }),
        );

        self.retire_bucket(first);
        self.retire_bucket(second);
        let mut centroids = self.centroids.write();
        let mut sizes = self.bucket_sizes.write();
        centroids.insert(new_id, centroid);
        sizes.insert(new_id, merged_size);
        drop(sizes);
        drop(centroids);
        self.rebuild_router();
    }

    fn handle_rebalance(&self, bucket_id: u64) {
        // For now, just recompute centroid from storage and republish routing.
        let lock = self.lock_for(bucket_id);
        let _guard = lock.lock();
        let bucket = match self.load_bucket(bucket_id) {
            Some(b) => b,
            None => {
                warn!("rebalance: rebalance skipped, bucket {} not found", bucket_id);
                return;
            }
        };
        let centroid = compute_centroid(&bucket.vectors);
        let mut centroids = self.centroids.write();
        let mut sizes = self.bucket_sizes.write();
        centroids.insert(bucket_id, centroid);
        sizes.insert(bucket_id, bucket.vectors.len());
        drop(sizes);
        drop(centroids);
        self.rebuild_router();
    }
}

#[derive(Clone)]
pub struct RebalanceWorker {
    #[allow(dead_code)]
    tx: Sender<RebalanceTask>,
    state: Arc<RebalanceState>,
}

impl RebalanceWorker {
    pub fn spawn(storage: Storage, routing: Arc<RoutingTable>) -> Self {
        let state = Arc::new(RebalanceState::new(storage, routing));
        let (tx, rx) = async_channel::unbounded();
        let state_clone = state.clone();
        thread::Builder::new()
            .name("rebalance-worker".to_string())
            .spawn(move || worker_loop(state_clone, rx))
            .expect("rebalance worker");
        Self { tx, state }
    }

    pub fn prime_centroids(&self, buckets: &[Bucket]) {
        self.state.prime_centroids(buckets);
        self.state.rebuild_router();
        for b in buckets {
            let _ = block_on(self.state.storage.put_bucket_meta(&BucketMeta {
                bucket_id: b.id,
                status: BucketMetaStatus::Active,
            }));
        }
    }

    #[allow(dead_code)]
    pub async fn enqueue(&self, task: RebalanceTask) -> anyhow::Result<()> {
        self.tx
            .send(task)
            .await
            .map_err(|e| anyhow::anyhow!("rebalance enqueue failed: {e:?}"))
    }

    pub fn enqueue_blocking(&self, task: RebalanceTask) -> anyhow::Result<()> {
        self.tx
            .send_blocking(task)
            .map_err(|e| anyhow::anyhow!("rebalance enqueue failed: {e:?}"))
    }

    pub fn snapshot_sizes(&self) -> HashMap<u64, usize> {
        self.state.bucket_sizes.read().clone()
    }
}

fn worker_loop(state: Arc<RebalanceState>, rx: Receiver<RebalanceTask>) {
    while let Ok(task) = rx.recv_blocking() {
        match task {
            RebalanceTask::Split(bucket) => state.handle_split(bucket),
            RebalanceTask::Merge(a, b) => state.handle_merge(a, b),
            RebalanceTask::Rebalance(bucket) => state.handle_rebalance(bucket),
        }
    }
}

pub(crate) fn compute_centroid(vectors: &[Vector]) -> Vec<f32> {
    if vectors.is_empty() {
        return Vec::new();
    }
    let dim = vectors[0].data.len();
    let mut sums = vec![0.0f32; dim];
    for v in vectors {
        for (i, val) in v.data.iter().enumerate() {
            sums[i] += *val;
        }
    }
    let count = vectors.len() as f32;
    for s in sums.iter_mut() {
        *s /= count;
    }
    sums
}
