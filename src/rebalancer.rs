use crate::indexer::Indexer;
use crate::ingest_counter;
use crate::quantizer::Quantizer;
use crate::router::{Router, RoutingTable};
use crate::storage::{Bucket, BucketMeta, BucketMetaStatus, Storage, Vector};
use crate::wal::runtime::Walrus;
use anyhow::Result;
use futures::executor::block_on;
use log::{debug, error, warn};
use parking_lot::{Mutex, RwLock};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::thread;
use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RebalanceTaskKind {
    Split,
    Merge,
    Rebalance,
}

type RebalanceFailHook = Arc<dyn Fn(RebalanceTaskKind) -> bool + Send + Sync>;

static FAIL_HOOK: StdMutex<Option<RebalanceFailHook>> = StdMutex::new(None);

pub fn set_rebalance_fail_hook<F>(hook: F)
where
    F: Fn(RebalanceTaskKind) -> bool + Send + Sync + 'static,
{
    *FAIL_HOOK.lock().expect("fail hook poisoned") = Some(Arc::new(hook));
}

pub fn clear_rebalance_fail_hook() {
    *FAIL_HOOK.lock().expect("fail hook poisoned") = None;
}

fn should_fail(kind: RebalanceTaskKind) -> bool {
    if let Some(h) = FAIL_HOOK.lock().expect("fail hook poisoned").as_ref() {
        return h(kind);
    }
    let _ = kind;
    false
}

pub(crate) struct RebalanceState {
    storage: Storage,
    wal: Arc<Walrus>,
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
            wal: storage.wal.clone(),
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

    fn refresh_sizes(&self) -> HashMap<u64, usize> {
        let ids: Vec<u64> = self.centroids.read().keys().cloned().collect();
        let prev_sizes = self.bucket_sizes.read().clone();
        let mut fresh = HashMap::new();
        for id in ids {
            let topic = crate::storage::Storage::topic_for(id);
            let count = self.wal.get_topic_entry_count(&topic) as usize;
            // Make counts monotonic per bucket to avoid temporary regressions when WAL lags.
            let stabilized = count.max(prev_sizes.get(&id).cloned().unwrap_or(0));
            fresh.insert(id, stabilized);
        }
        let mut sizes = self.bucket_sizes.write();
        sizes.clear();
        sizes.extend(fresh.iter().map(|(k, v)| (*k, *v)));
        let sum: u64 = fresh.values().map(|v| *v as u64).sum();
        let inserted = ingest_counter::get();

        if sum < inserted {
            debug!(
                "rebalance: wal counts sum {} is less than total inserted {}; proceeding with monotonic sizes",
                sum, inserted
            );
        }
        sizes.clone()
    }

    fn lock_for(&self, bucket_id: u64) -> Arc<Mutex<()>> {
        let mut guard = self.locks.lock();
        guard
            .entry(bucket_id)
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    }

    pub(crate) fn load_bucket_vectors(&self, bucket_id: u64) -> Option<Vec<Vector>> {
        let chunks = block_on(self.storage.get_chunks(bucket_id)).ok()?;
        let mut vectors: Vec<Vector> = Vec::new();
        for chunk in chunks {
            if chunk.len() < 8 {
                continue;
            }
            let mut len_bytes = [0u8; 8];
            len_bytes.copy_from_slice(&chunk[0..8]);
            let archive_len = u64::from_le_bytes(len_bytes) as usize;
            if 8 + archive_len > chunk.len() || archive_len < 16 {
                continue;
            }
            // Current WAL format: len prefix, then id (u64), dim (u64), then dim*f32 bytes.
            let mut off = 8;
            let mut id_bytes = [0u8; 8];
            id_bytes.copy_from_slice(&chunk[off..off + 8]);
            off += 8;
            let mut dim_bytes = [0u8; 8];
            dim_bytes.copy_from_slice(&chunk[off..off + 8]);
            off += 8;
            let dim = u64::from_le_bytes(dim_bytes) as usize;
            let Some(expected_bytes) = dim.checked_mul(4) else {
                continue;
            };
            if off + expected_bytes > chunk.len() {
                continue;
            }
            let mut data = Vec::with_capacity(dim);
            let data_bytes = &chunk[off..off + expected_bytes];
            for chunked in data_bytes.chunks_exact(4) {
                let mut fb = [0u8; 4];
                fb.copy_from_slice(chunked);
                data.push(f32::from_bits(u32::from_le_bytes(fb)));
            }
            let id = u64::from_le_bytes(id_bytes);
            vectors.push(Vector { id, data });
        }
        (!vectors.is_empty()).then_some(vectors)
    }

    fn retire_bucket_local(&self, bucket_id: u64) {
        self.centroids.write().remove(&bucket_id);
        self.bucket_sizes.write().remove(&bucket_id);
        self.retired.write().insert(bucket_id);
        // Keep the lock around - don't remove it. This ensures any concurrent
        // operations on this bucket_id serialize properly and can check the
        // retired set while holding the same lock.
    }

    fn retire_bucket_io(&self, bucket_id: u64) {
        let _ = block_on(self.storage.put_bucket_meta(&BucketMeta {
            bucket_id,
            status: BucketMetaStatus::Retired,
        }));
    }

    fn mark_bucket_checkpointed(&self, bucket_id: u64) {
        let topic = crate::storage::Storage::topic_for(bucket_id);
        // Drain the topic with checkpoint=true to mark all blocks checkpointed.
        let max_bytes = 16 * 1024 * 1024; // read in chunks
        loop {
            match self.wal.batch_read_for_topic(&topic, max_bytes, true, None) {
                Ok(entries) => {
                    if entries.is_empty() {
                        break;
                    }
                }
                Err(e) => {
                    warn!("rebalance: checkpoint drain failed for {}: {:?}", topic, e);
                    break;
                }
            }
        }
    }

    fn rebuild_router(&self, changed_buckets: Vec<u64>) {
        let centroids_map = self.centroids.read();
        let bucket_count = centroids_map.len();
        if centroids_map.is_empty() {
            return;
        }
        let mut centroids: Vec<(u64, Vec<f32>)> = Vec::with_capacity(bucket_count);
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for (id, c) in centroids_map.iter() {
            for &val in c {
                if val < min {
                    min = val;
                }
                if val > max {
                    max = val;
                }
            }
            centroids.push((*id, c.clone()));
        }
        drop(centroids_map);

        let Some((min, max)) = Quantizer::compute_bounds_from_minmax(min, max) else {
            return;
        };
        let quantizer = Quantizer::new(min, max);
        let mut router = Router::new(100_000, quantizer);
        for (id, centroid) in &centroids {
            router.add_centroid(*id, centroid);
        }
        let version = self.routing.install(router, changed_buckets);
        if log::log_enabled!(log::Level::Debug) {
            let sizes_map = self.bucket_sizes.read();
            let mut sizes: Vec<usize> = sizes_map.values().copied().collect();
            sizes.sort_unstable();
            debug!(
                "rebalance: published router version {} (buckets={}, sizes={:?})",
                version, bucket_count, sizes
            );
        }
    }

    fn handle_split_sync(&self, bucket_id: u64) {
        if should_fail(RebalanceTaskKind::Split) {
            debug!(
                "rebalance: injected failure for split on bucket {}",
                bucket_id
            );
            return;
        }
        let lock = self.lock_for(bucket_id);
        let _guard = lock.lock();

        // Check if bucket was already retired (e.g., by a prior split/merge).
        if self.retired.read().contains(&bucket_id) {
            debug!(
                "rebalance: split skipped, bucket {} already retired",
                bucket_id
            );
            return;
        }

        // 1. Load Vectors
        let vectors = match self.load_bucket_vectors(bucket_id) {
            Some(v) => v,
            None => {
                log::debug!("rebalance: split skipped, bucket {} not found", bucket_id);
                return;
            }
        };

        // 2. K-Means Split
        let mut bucket = Bucket::new(bucket_id, Vec::new());
        bucket.vectors = vectors;
        let splits = Indexer::split_bucket_once(bucket);
        if splits.is_empty() {
            return;
        }

        // 3. Persist New Buckets
        let mut new_entries = Vec::new();
        for mut split in splits {
            let new_id = self.allocate_bucket_id();
            split.id = new_id;
            if let Err(e) = block_on(self.storage.put_chunk(&split)) {
                error!(
                    "rebalance: failed to persist split bucket {} -> {}: {:?}",
                    bucket_id, new_id, e
                );
                continue;
            }
            let _ = block_on(self.storage.put_bucket_meta(&BucketMeta {
                bucket_id: new_id,
                status: BucketMetaStatus::Active,
            }));
            new_entries.push((new_id, split.centroid, split.vectors.len()));
        }

        if new_entries.is_empty() {
            return;
        }

        // 4. Retire Old Bucket & Update Router
        self.retire_bucket_local(bucket_id);
        self.retire_bucket_io(bucket_id);

        let mut centroids = self.centroids.write();
        let mut sizes = self.bucket_sizes.write();
        let new_count = new_entries.len();
        let mut changed = Vec::with_capacity(1 + new_count);
        changed.push(bucket_id);
        for (id, centroid, size) in new_entries {
            centroids.insert(id, centroid);
            sizes.insert(id, size);
            changed.push(id);
        }
        drop(sizes);
        drop(centroids);
        self.rebuild_router(changed);

        // 5. Checkpoint (Cleanup) - Inline
        self.mark_bucket_checkpointed(bucket_id);

        log::info!(
            "rebalance: split {} into {} buckets (new_total={})",
            bucket_id,
            new_count,
            self.centroids.read().len()
        );
    }
}

#[derive(Clone)]
pub struct RebalanceWorker {
    state: Arc<RebalanceState>,
}

impl RebalanceWorker {
    pub fn spawn(storage: Storage, routing: Arc<RoutingTable>, pin_cpu: Option<usize>) -> Self {
        let state = Arc::new(RebalanceState::new(storage, routing));
        let state_clone = state.clone();
        let name = "rebalance-loop".to_string();

        match pin_cpu {
            Some(cpu) => {
                // Run on a dedicated Glommio executor pinned to a reserved core.
                let builder =
                    glommio::LocalExecutorBuilder::new(glommio::Placement::Fixed(cpu)).name(&name);
                std::thread::spawn(move || {
                    builder
                        .make()
                        .expect("failed to create rebalance executor")
                        .run(run_autonomous_loop(state_clone));
                });
            }
            None => {
                thread::Builder::new()
                    .name(name.clone())
                    .spawn(move || {
                        // FIX: Run inside a Glommio executor to support spawn_blocking and timers
                        glommio::LocalExecutorBuilder::default()
                            .name(&name)
                            .make()
                            .expect("failed to create default rebalance executor")
                            .run(run_autonomous_loop(state_clone));
                    })
                    .expect("rebalance worker");
            }
        }
        Self { state }
    }

    pub async fn prime_centroids(&self, buckets: &[Bucket]) -> Result<()> {
        self.state.prime_centroids(buckets);
        self.state.rebuild_router(Vec::new());
        for b in buckets {
            self.state
                .storage
                .put_bucket_meta(&BucketMeta {
                    bucket_id: b.id,
                    status: BucketMetaStatus::Active,
                })
                .await?;
        }
        Ok(())
    }

    pub fn snapshot_sizes(&self) -> HashMap<u64, usize> {
        self.state.refresh_sizes()
    }

    // No-op for compatibility, loop shuts down with process for now
    pub fn close(&self) {}
}

async fn run_autonomous_loop(state: Arc<RebalanceState>) {
    // Configurable threshold: split if bucket has more than N vectors.
    let threshold: usize = std::env::var("SATORI_REBALANCE_THRESHOLD")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2000);

    loop {
        // 1. Refresh sizes
        let sizes = state.refresh_sizes();

        // 2. Find biggest bucket
        let mut max_id = 0;
        let mut max_size = 0;
        for (id, size) in sizes {
            if size > max_size {
                max_size = size;
                max_id = id;
            }
        }

        // 3. Decide
        if max_size > threshold {
            // 4. Execute Split (Synchronously)
            // Note: Since this function is async but handle_split_sync is blocking (uses block_on internally),
            // we wrap it in spawn_blocking if we were in a strictly async context.
            // But here we are the only thing running on this thread/executor.
            // Using spawn_blocking is safer for Glommio integration.
            let state_ref = state.clone();
            let _ = glommio::executor()
                .spawn_blocking(move || {
                    state_ref.handle_split_sync(max_id);
                })
                .await;
        } else {
            // Nothing urgent, sleep a bit
            glommio::timer::Timer::new(Duration::from_millis(500)).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn centroid_of_vectors_is_mean() {
        let vectors = vec![
            Vector::new(0, vec![1.0, 3.0]),
            Vector::new(1, vec![3.0, 5.0]),
        ];
        let centroid = compute_centroid(&vectors);
        assert_eq!(centroid, vec![2.0, 4.0]);
    }

    #[test]
    fn centroid_of_empty_is_empty() {
        let centroid = compute_centroid(&[]);
        assert!(centroid.is_empty());
    }

    /// Centroid of single vector is the vector itself.
    #[test]
    fn centroid_of_single_is_identity() {
        let vectors = vec![Vector::new(0, vec![5.0, 10.0, 15.0])];
        let centroid = compute_centroid(&vectors);
        assert_eq!(centroid, vec![5.0, 10.0, 15.0]);
    }

    /// Centroid with many vectors maintains numerical stability.
    #[test]
    fn centroid_many_vectors() {
        let n = 1000;
        let vectors: Vec<Vector> = (0..n)
            .map(|i| Vector::new(i as u64, vec![i as f32, (i * 2) as f32]))
            .collect();
        let centroid = compute_centroid(&vectors);
        // Mean of 0..999 = 499.5, mean of 0..1998 (step 2) = 999
        let expected_x = (0..n).sum::<usize>() as f32 / n as f32;
        let expected_y = (0..n).map(|i| (i * 2) as f32).sum::<f32>() / n as f32;
        assert!((centroid[0] - expected_x).abs() < 0.01);
        assert!((centroid[1] - expected_y).abs() < 0.01);
    }

    /// Centroid with negative values.
    #[test]
    fn centroid_negative_values() {
        let vectors = vec![
            Vector::new(0, vec![-10.0, -20.0]),
            Vector::new(1, vec![10.0, 20.0]),
        ];
        let centroid = compute_centroid(&vectors);
        assert_eq!(centroid, vec![0.0, 0.0]);
    }

    /// Centroid of zero-dimension vectors is zero-dimension.
    #[test]
    fn centroid_zero_dimension() {
        let vectors = vec![Vector::new(0, vec![]), Vector::new(1, vec![])];
        let centroid = compute_centroid(&vectors);
        assert!(centroid.is_empty());
    }
}
