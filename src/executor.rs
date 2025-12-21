use crate::storage::{Storage, Vector};
use anyhow::Result;
use parking_lot::Mutex;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::Mutex as StdMutex;

pub struct Executor {
    storage: Storage,
    cache: Mutex<WorkerCache>,
    cache_version: AtomicU64,
    last_changed: Mutex<Arc<Vec<u64>>>,
}

type FailLoadHook = Arc<dyn Fn(u64) -> bool + Send + Sync>;

static FAIL_LOAD_HOOK: StdMutex<Option<FailLoadHook>> = StdMutex::new(None);

pub fn set_executor_fail_load_hook<F>(hook: F)
where
    F: Fn(u64) -> bool + Send + Sync + 'static,
{
    *FAIL_LOAD_HOOK.lock().expect("fail hook poisoned") = Some(Arc::new(hook));
}

pub fn clear_executor_fail_load_hook() {
    *FAIL_LOAD_HOOK.lock().expect("fail hook poisoned") = None;
}

#[derive(Clone, Copy, Debug)]
struct Link {
    prev: Option<usize>,
    next: Option<usize>,
}

#[derive(Debug)]
struct Slot {
    key: u64,
    occupied: bool,
    link: Link,
    off: usize,
    len: usize,
}

pub struct WorkerCache {
    slots: Vec<Slot>,
    map: HashMap<u64, usize>,
    head: Option<usize>,
    tail: Option<usize>,
    free: Vec<usize>,
    arena: Vec<u8>,
    bucket_max_bytes: usize,
}

impl WorkerCache {
    pub fn new(max_buckets: usize, bucket_max_bytes: usize, _total_max_bytes: usize) -> Self {
        let max_buckets = max_buckets.max(1);
        let bucket_max_bytes = bucket_max_bytes.max(1);
        let arena_bytes = max_buckets
            .checked_mul(bucket_max_bytes)
            .expect("arena too large");
        let arena = vec![0u8; arena_bytes];

        let mut slots = Vec::with_capacity(max_buckets);
        for i in 0..max_buckets {
            slots.push(Slot {
                key: 0,
                occupied: false,
                link: Link {
                    prev: None,
                    next: None,
                },
                off: i * bucket_max_bytes,
                len: 0,
            });
        }

        let mut free = Vec::with_capacity(max_buckets);
        for i in (0..max_buckets).rev() {
            free.push(i);
        }

        let map = HashMap::with_capacity((max_buckets * 2).max(16));

        Self {
            slots,
            map,
            head: None,
            tail: None,
            free,
            arena,
            bucket_max_bytes,
        }
    }

    fn get(&mut self, bucket_id: u64) -> Option<&[u8]> {
        let idx = *self.map.get(&bucket_id)?;
        self.touch(idx);
        let slot = &self.slots[idx];
        Some(&self.arena[slot.off..slot.off + slot.len])
    }

    #[cfg(test)]
    fn put_bytes(&mut self, bucket_id: u64, data: &[u8]) -> Option<&[u8]> {
        if data.is_empty() || data.len() > self.bucket_max_bytes {
            return None;
        }
        let idx = self.prepare_slot(bucket_id);
        let slot = &mut self.slots[idx];
        let start = slot.off;
        let end = start + data.len();
        self.arena[start..end].copy_from_slice(data);
        slot.len = data.len();
        slot.key = bucket_id;
        slot.occupied = true;
        Some(&self.arena[self.slots[idx].off..self.slots[idx].off + self.slots[idx].len])
    }

    fn put_from_chunks(&mut self, bucket_id: u64, chunks: &[Vec<u8>]) -> Option<&[u8]> {
        let total: usize = chunks.iter().map(|c| c.len()).sum();
        if total == 0 || total > self.bucket_max_bytes {
            return None;
        }

        let idx = self.prepare_slot(bucket_id);
        let mut write_off = self.slots[idx].off;
        for chunk in chunks {
            let end = write_off + chunk.len();
            self.arena[write_off..end].copy_from_slice(chunk);
            write_off = end;
        }
        self.slots[idx].len = total;
        Some(&self.arena[self.slots[idx].off..self.slots[idx].off + self.slots[idx].len])
    }

    fn clear(&mut self) {
        self.map.clear();
        self.head = None;
        self.tail = None;
        self.free.clear();

        for (i, slot) in self.slots.iter_mut().enumerate() {
            slot.occupied = false;
            slot.key = 0;
            slot.len = 0;
            slot.link = Link {
                prev: None,
                next: None,
            };
            self.free.push(i);
        }
    }

    fn invalidate_many(&mut self, ids: &[u64]) {
        for id in ids {
            if let Some(idx) = self.map.remove(id) {
                self.detach(idx);
                let slot = &mut self.slots[idx];
                slot.occupied = false;
                slot.key = 0;
                slot.len = 0;
                self.free.push(idx);
            }
        }
    }

    fn prepare_slot(&mut self, bucket_id: u64) -> usize {
        if let Some(&idx) = self.map.get(&bucket_id) {
            self.touch(idx);
            return idx;
        }

        let idx = if let Some(idx) = self.free.pop() {
            idx
        } else {
            let victim = self.tail.expect("cache has no tail on eviction");
            self.evict_slot(victim);
            victim
        };

        self.slots[idx].occupied = true;
        self.slots[idx].key = bucket_id;
        self.slots[idx].len = 0;
        self.attach_head(idx);
        self.map.insert(bucket_id, idx);
        idx
    }

    fn evict_slot(&mut self, idx: usize) {
        if self.slots[idx].occupied {
            self.map.remove(&self.slots[idx].key);
        }
        self.detach(idx);
        self.slots[idx].occupied = false;
        self.slots[idx].key = 0;
        self.slots[idx].len = 0;
    }

    fn touch(&mut self, idx: usize) {
        if self.head == Some(idx) {
            return;
        }
        self.detach(idx);
        self.attach_head(idx);
    }

    fn attach_head(&mut self, idx: usize) {
        self.slots[idx].link.prev = None;
        self.slots[idx].link.next = self.head;

        if let Some(h) = self.head {
            self.slots[h].link.prev = Some(idx);
        } else {
            self.tail = Some(idx);
        }

        self.head = Some(idx);
    }

    fn detach(&mut self, idx: usize) {
        let (p, n) = {
            let l = self.slots[idx].link;
            (l.prev, l.next)
        };

        if let Some(prev) = p {
            self.slots[prev].link.next = n;
        } else if self.head == Some(idx) {
            self.head = n;
        }

        if let Some(next) = n {
            self.slots[next].link.prev = p;
        } else if self.tail == Some(idx) {
            self.tail = p;
        }

        self.slots[idx].link.prev = None;
        self.slots[idx].link.next = None;
    }
}

impl Executor {
    pub fn new(storage: Storage, cache: WorkerCache) -> Self {
        Self {
            storage,
            cache: Mutex::new(cache),
            cache_version: AtomicU64::new(0),
            last_changed: Mutex::new(Arc::new(Vec::new())),
        }
    }

    pub fn cache_version(&self) -> u64 {
        self.cache_version.load(Ordering::Relaxed)
    }

    /// Return the vectors for the given ids within a bucket, using the worker cache when possible.
    pub(crate) async fn fetch_vectors(&self, bucket_id: u64, ids: &[u64]) -> Result<Vec<Vector>> {
        let wanted: HashSet<u64> = ids.iter().copied().collect();
        if wanted.is_empty() {
            return Ok(Vec::new());
        }

        // Try cache first
        if let Some(hits) = {
            let mut cache = self.cache.lock();
            cache
                .get(bucket_id)
                .map(|data| collect_vectors_from_slice(data, &wanted))
        } {
            return Ok(hits);
        }

        // Load and populate cache
        let chunks = self.load_bucket_chunks(bucket_id).await?;
        let hits = collect_vectors_from_chunks(&chunks, &wanted);

        if !chunks.is_empty() {
            let mut cache = self.cache.lock();
            let _ = cache.put_from_chunks(bucket_id, &chunks);
        }

        Ok(hits)
    }

    pub(crate) async fn load_bucket_chunks(&self, bucket_id: u64) -> Result<Vec<Vec<u8>>> {
        if let Some(h) = FAIL_LOAD_HOOK.lock().expect("fail hook poisoned").as_ref() {
            if h(bucket_id) {
                anyhow::bail!("executor load hook injected failure");
            }
        }
        Ok(self.storage.get_chunks(bucket_id).await.unwrap_or_default())
    }

    pub async fn query(
        &self,
        query_vec: &[f32],
        bucket_ids: &[u64],
        top_k: usize,
        routing_version: u64,
        changed_buckets: Arc<Vec<u64>>,
        include_vectors: bool,
    ) -> Result<Vec<(u64, f32, Option<Vec<f32>>)>> {
        if self.cache_version.load(Ordering::Relaxed) != routing_version {
            let mut cache = self.cache.lock();
            if changed_buckets.is_empty() {
                cache.clear();
            } else {
                cache.invalidate_many(&changed_buckets);
            }
            self.cache_version.store(routing_version, Ordering::Relaxed);
            *self.last_changed.lock() = changed_buckets;
        }

        let mut candidates: Vec<(u64, f32, Option<Vec<f32>>)> = Vec::new();

        for &id in bucket_ids {
            // First try cache under a short lock.
            let mut handled = false;
            {
                let mut cache = self.cache.lock();
                if let Some(data) = cache.get(id) {
                    scan_bucket_slice(data, query_vec, include_vectors, &mut candidates);
                    handled = true;
                }
            }

            if handled {
                continue;
            }

            // Cache miss: load from storage without holding the lock.
            match self.load_bucket_chunks(id).await {
                Ok(chunks) => {
                    if chunks.is_empty() {
                        continue;
                    }

                    let mut cached = false;
                    {
                        let mut cache = self.cache.lock();
                        if let Some(data) = cache.put_from_chunks(id, &chunks) {
                            scan_bucket_slice(data, query_vec, include_vectors, &mut candidates);
                            cached = true;
                        }
                    }

                    if !cached {
                        scan_bucket_chunks(&chunks, query_vec, include_vectors, &mut candidates);
                    }
                }
                Err(e) => {
                    log::debug!("executor: load failed for bucket {}: {:?}", id, e);
                    continue;
                }
            }
        }

        // Sort by distance (ascending)
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top_k
        if candidates.len() > top_k {
            candidates.truncate(top_k);
        }

        Ok(candidates)
    }
}

fn parse_vector_payload(payload: &[u8]) -> Option<(u64, &[u8])> {
    if payload.len() < 16 {
        return None;
    }
    let mut id_bytes = [0u8; 8];
    id_bytes.copy_from_slice(&payload[0..8]);
    let mut dim_bytes = [0u8; 8];
    dim_bytes.copy_from_slice(&payload[8..16]);
    let dim = u64::from_le_bytes(dim_bytes) as usize;
    let data_len = dim.checked_mul(4)?;
    let data_end = 16usize.checked_add(data_len)?;
    let data = payload.get(16..data_end)?;
    Some((u64::from_le_bytes(id_bytes), data))
}

fn walk_bucket_slice(data: &[u8], mut f: impl FnMut(u64, &[u8])) {
    let mut offset = 0usize;
    while offset + 8 <= data.len() {
        let mut len_bytes = [0u8; 8];
        len_bytes.copy_from_slice(&data[offset..offset + 8]);
        let payload_len = u64::from_le_bytes(len_bytes) as usize;
        if payload_len < 16 {
            break;
        }
        let end = offset + 8 + payload_len;
        if end > data.len() {
            break;
        }
        if let Some((id, vec_bytes)) = parse_vector_payload(&data[offset + 8..end]) {
            f(id, vec_bytes);
        }
        offset = end;
    }
}

fn walk_bucket_chunks(chunks: &[Vec<u8>], mut f: impl FnMut(u64, &[u8])) {
    // Walk from newest to oldest and ignore stale duplicates so rewrites (e.g., deletes)
    // take effect without truncating the WAL topic.
    let mut seen = HashSet::new();
    for chunk in chunks.iter().rev() {
        if chunk.len() < 8 {
            continue;
        }
        let mut len_bytes = [0u8; 8];
        len_bytes.copy_from_slice(&chunk[0..8]);
        let payload_len = u64::from_le_bytes(len_bytes) as usize;
        if payload_len < 16 || 8 + payload_len > chunk.len() {
            continue;
        }
        if let Some((id, vec_bytes)) = parse_vector_payload(&chunk[8..8 + payload_len]) {
            if seen.insert(id) {
                f(id, vec_bytes);
            }
        }
    }
}

fn decode_vector_bytes(raw: &[u8]) -> Option<Vec<f32>> {
    if !raw.len().is_multiple_of(4) {
        return None;
    }
    let mut data = Vec::with_capacity(raw.len() / 4);
    for chunk in raw.chunks_exact(4) {
        data.push(f32::from_bits(u32::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3],
        ])));
    }
    Some(data)
}

fn l2_distance_bytes(raw: &[u8], query: &[f32]) -> Option<f32> {
    if !raw.len().is_multiple_of(4) {
        return None;
    }
    let dim = raw.len() / 4;
    if dim != query.len() {
        return None;
    }

    let mut sum_sq = 0f32;
    for (chunk, q) in raw.chunks_exact(4).zip(query.iter()) {
        let val = f32::from_bits(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        let diff = val - q;
        sum_sq += diff * diff;
    }
    Some(sum_sq.sqrt())
}

fn collect_vectors_from_slice(data: &[u8], wanted: &HashSet<u64>) -> Vec<Vector> {
    let mut hits = Vec::new();
    walk_bucket_slice(data, |id, vec_bytes| {
        if wanted.contains(&id) {
            if let Some(data) = decode_vector_bytes(vec_bytes) {
                hits.push(Vector { id, data });
            }
        }
    });
    hits
}

fn collect_vectors_from_chunks(chunks: &[Vec<u8>], wanted: &HashSet<u64>) -> Vec<Vector> {
    let mut hits = Vec::new();
    walk_bucket_chunks(chunks, |id, vec_bytes| {
        if wanted.contains(&id) {
            if let Some(data) = decode_vector_bytes(vec_bytes) {
                hits.push(Vector { id, data });
            }
        }
    });
    hits
}

fn scan_bucket_slice(
    data: &[u8],
    query_vec: &[f32],
    include_vectors: bool,
    candidates: &mut Vec<(u64, f32, Option<Vec<f32>>)>,
) {
    walk_bucket_slice(data, |id, vec_bytes| {
        if let Some(dist) = l2_distance_bytes(vec_bytes, query_vec) {
            let payload = if include_vectors {
                decode_vector_bytes(vec_bytes)
            } else {
                None
            };
            candidates.push((id, dist, payload));
        }
    });
}

fn scan_bucket_chunks(
    chunks: &[Vec<u8>],
    query_vec: &[f32],
    include_vectors: bool,
    candidates: &mut Vec<(u64, f32, Option<Vec<f32>>)>,
) {
    walk_bucket_chunks(chunks, |id, vec_bytes| {
        if let Some(dist) = l2_distance_bytes(vec_bytes, query_vec) {
            let payload = if include_vectors {
                decode_vector_bytes(vec_bytes)
            } else {
                None
            };
            candidates.push((id, dist, payload));
        }
    });
}

#[cfg(test)]
fn l2_distance_f32(a: &[f32], b: &[f32]) -> f32 {
    if cfg!(target_arch = "x86_64") && std::arch::is_x86_feature_detected!("avx2") {
        // Prefer the FMA path when available; it gives a small bump on AVX2 hardware.
        if std::arch::is_x86_feature_detected!("fma") {
            unsafe { l2_distance_f32_avx2_fma(a, b) }
        } else {
            unsafe { l2_distance_f32_avx2(a, b) }
        }
    } else {
        let sum_sq: f32 = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let diff = x - y;
                diff * diff
            })
            .sum();
        sum_sq.sqrt()
    }
}

#[cfg(test)]
#[target_feature(enable = "avx2")]
unsafe fn l2_distance_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len().min(b.len());
    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut i = 0;

    // Unroll by 2x to reduce loop overhead.
    while i + 16 <= len {
        let va0 = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb0 = _mm256_loadu_ps(b.as_ptr().add(i));
        let diff0 = _mm256_sub_ps(va0, vb0);
        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(diff0, diff0));

        let va1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
        let vb1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
        let diff1 = _mm256_sub_ps(va1, vb1);
        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(diff1, diff1));

        i += 16;
    }

    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let diff = _mm256_sub_ps(va, vb);
        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(diff, diff));
        i += 8;
    }

    let mut tmp0 = [0f32; 8];
    _mm256_storeu_ps(tmp0.as_mut_ptr(), sum0);
    let mut total = tmp0.iter().sum::<f32>();

    let mut tmp1 = [0f32; 8];
    _mm256_storeu_ps(tmp1.as_mut_ptr(), sum1);
    total += tmp1.iter().sum::<f32>();

    for j in i..len {
        let d = a[j] - b[j];
        total += d * d;
    }

    total.sqrt()
}

#[cfg(test)]
#[target_feature(enable = "avx2,fma")]
unsafe fn l2_distance_f32_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len().min(b.len());
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);
        i += 8;
    }

    let mut tmp = [0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), sum);
    let mut total = tmp.iter().sum::<f32>();

    for j in i..len {
        let d = a[j] - b[j];
        total += d * d;
    }

    total.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_distance_matches_scalar() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![1.0f32, 1.0, 1.0, 1.0];
        let expected: f32 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt();
        let dist = l2_distance_f32(&a, &b);
        assert!((dist - expected).abs() < 1e-5);
    }

    #[test]
    fn l2_distance_handles_mismatched_lengths() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![1.0f32, 2.0];
        let dist = l2_distance_f32(&a, &b);
        // Should only consider the min length (2 elements)
        let expected = ((0.0f32).powi(2) + (0.0f32).powi(2)).sqrt();
        assert!((dist - expected).abs() < 1e-5);
    }

    /// L2 distance of identical vectors is zero.
    #[test]
    fn l2_distance_identical_is_zero() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let dist = l2_distance_f32(&a, &a);
        assert!(dist.abs() < 1e-10, "distance to self should be 0");
    }

    /// L2 distance with empty vectors is zero.
    #[test]
    fn l2_distance_empty_vectors() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        let dist = l2_distance_f32(&a, &b);
        assert_eq!(dist, 0.0);
    }

    /// L2 distance with single-element vectors.
    #[test]
    fn l2_distance_single_element() {
        let a = vec![0.0f32];
        let b = vec![3.0f32];
        let dist = l2_distance_f32(&a, &b);
        assert!((dist - 3.0).abs() < 1e-5);
    }

    /// L2 distance should handle large vectors (SIMD path).
    #[test]
    fn l2_distance_large_vectors() {
        let n = 1024;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
        // Each element differs by 1, so sum of squares = n, sqrt = sqrt(n)
        let expected = (n as f32).sqrt();
        let dist = l2_distance_f32(&a, &b);
        assert!(
            (dist - expected).abs() < 1e-3,
            "expected {}, got {}",
            expected,
            dist
        );
    }

    /// L2 distance with negative values.
    #[test]
    fn l2_distance_negative_values() {
        let a = vec![-1.0f32, -2.0, -3.0];
        let b = vec![1.0f32, 2.0, 3.0];
        // Differences: -2, -4, -6 → squares: 4, 16, 36 → sum: 56 → sqrt ≈ 7.483
        let expected = 56.0f32.sqrt();
        let dist = l2_distance_f32(&a, &b);
        assert!((dist - expected).abs() < 1e-4);
    }

    /// WorkerCache respects max_buckets limit.
    #[test]
    fn worker_cache_evicts_on_limit() {
        let mut cache = WorkerCache::new(2, 16, usize::MAX);
        cache.put_bytes(0, &[0u8; 8]);
        cache.put_bytes(1, &[1u8; 8]);
        cache.put_bytes(2, &[2u8; 8]);
        // LRU should have evicted bucket 0
        assert!(cache.get(0).is_none(), "bucket 0 should be evicted");
        assert!(cache.get(1).is_some(), "bucket 1 should still be cached");
        assert!(cache.get(2).is_some(), "bucket 2 should still be cached");
    }

    /// WorkerCache skips buckets exceeding byte limit.
    #[test]
    fn worker_cache_rejects_oversized() {
        let mut cache = WorkerCache::new(10, 100, usize::MAX); // max 100 bytes per bucket
        cache.put_bytes(0, &[0u8; 50]);
        cache.put_bytes(1, &[0u8; 200]); // too big
        assert!(cache.get(0).is_some(), "small bucket should be cached");
        assert!(
            cache.get(1).is_none(),
            "oversized bucket should not be cached"
        );
    }

    /// WorkerCache clear removes all entries.
    #[test]
    fn worker_cache_clear() {
        let mut cache = WorkerCache::new(10, 32, usize::MAX);
        cache.put_bytes(0, &[0u8; 8]);
        cache.put_bytes(1, &[1u8; 8]);
        cache.clear();
        assert!(cache.get(0).is_none());
        assert!(cache.get(1).is_none());
    }

    /// WorkerCache invalidate_many removes specific entries.
    #[test]
    fn worker_cache_invalidate_many() {
        let mut cache = WorkerCache::new(10, 32, usize::MAX);
        cache.put_bytes(0, &[0u8; 8]);
        cache.put_bytes(1, &[1u8; 8]);
        cache.put_bytes(2, &[2u8; 8]);
        cache.invalidate_many(&[0, 2]);
        assert!(cache.get(0).is_none(), "bucket 0 should be invalidated");
        assert!(cache.get(1).is_some(), "bucket 1 should remain");
        assert!(cache.get(2).is_none(), "bucket 2 should be invalidated");
    }
}
