use crate::storage::{Storage, Vector};
use anyhow::Result;
use lru::LruCache;
use parking_lot::Mutex;
use std::convert::TryInto;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::Mutex as StdMutex;

pub struct Executor {
    storage: Storage,
    cache: Mutex<WorkerCache>,
    cache_version: AtomicU64,
    last_changed: Mutex<Arc<Vec<u64>>>,
}

struct CachedBucket {
    vectors: Vec<Vector>,
    byte_size: usize,
}

type FailLoadHook = Arc<dyn Fn(u64) -> bool + Send + Sync>;

static FAIL_LOAD_HOOK: StdMutex<Option<FailLoadHook>> =
    StdMutex::new(None);

pub fn set_executor_fail_load_hook<F>(hook: F)
where
    F: Fn(u64) -> bool + Send + Sync + 'static,
{
    *FAIL_LOAD_HOOK.lock().expect("fail hook poisoned") = Some(Arc::new(hook));
}

pub fn clear_executor_fail_load_hook() {
    *FAIL_LOAD_HOOK.lock().expect("fail hook poisoned") = None;
}

pub struct WorkerCache {
    buckets: LruCache<u64, CachedBucket>,
    bucket_max_bytes: usize,
}

impl WorkerCache {
    pub fn new(max_buckets: usize, bucket_max_bytes: usize) -> Self {
        let cap = NonZeroUsize::new(max_buckets.max(1)).unwrap();
        Self {
            buckets: LruCache::new(cap),
            bucket_max_bytes,
        }
    }

    fn get(&mut self, bucket_id: u64) -> Option<&CachedBucket> {
        self.buckets.get(&bucket_id)
    }

    fn put(&mut self, bucket_id: u64, bucket: CachedBucket) {
        if bucket.byte_size <= self.bucket_max_bytes {
            self.buckets.put(bucket_id, bucket);
        }
    }

    fn clear(&mut self) {
        self.buckets.clear();
    }

    fn invalidate_many(&mut self, ids: &[u64]) {
        for id in ids {
            self.buckets.pop(id);
        }
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

    async fn load_bucket(&self, bucket_id: u64) -> Result<CachedBucket> {
        if let Some(h) = FAIL_LOAD_HOOK.lock().expect("fail hook poisoned").as_ref() {
            if h(bucket_id) {
                anyhow::bail!("executor load hook injected failure");
            }
        }
        let chunks = self.storage.get_chunks(bucket_id).await.unwrap_or_default();
        let mut vectors = Vec::new();
        let mut total_bytes = 0usize;

        for chunk in chunks {
            total_bytes += chunk.len();
            if chunk.len() < 8 {
                continue;
            }
            let len_bytes: [u8; 8] = chunk[0..8].try_into().unwrap_or([0; 8]);
            let archive_len = u64::from_le_bytes(len_bytes) as usize;

            if 8 + archive_len > chunk.len() || archive_len < 16 {
                continue;
            }
            let data_slice = &chunk[8..8 + archive_len];
            // Format: [id: u64][len: u64][data: len * f32]
            if data_slice.len() < 16 {
                continue;
            }
            let id = u64::from_le_bytes(data_slice[0..8].try_into().unwrap_or([0; 8]));
            let count = u64::from_le_bytes(data_slice[8..16].try_into().unwrap_or([0; 8])) as usize;
            if 16 + count * 4 > data_slice.len() {
                continue;
            }
            let mut data = Vec::with_capacity(count);
            let mut offset = 16;
            for _ in 0..count {
                let bytes: [u8; 4] = data_slice[offset..offset + 4].try_into().unwrap_or([0; 4]);
                data.push(f32::from_bits(u32::from_le_bytes(bytes)));
                offset += 4;
            }
            vectors.push(Vector { id, data });
        }

        Ok(CachedBucket {
            vectors,
            byte_size: total_bytes,
        })
    }

    pub async fn query(
        &self,
        query_vec: &[f32],
        bucket_ids: &[u64],
        top_k: usize,
        routing_version: u64,
        changed_buckets: Arc<Vec<u64>>,
    ) -> Result<Vec<(u64, f32)>> {
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

        let mut candidates = Vec::new();

        for &id in bucket_ids {
            // First try cache under a short lock.
            let cached_vectors = {
                let mut cache = self.cache.lock();
                cache.get(id).map(|c| c.vectors.clone())
            };

            if let Some(vectors) = cached_vectors {
                for vector in &vectors {
                    if vector.data.len() != query_vec.len() {
                        continue;
                    }
                    let dist = l2_distance_f32(&vector.data, query_vec);
                    candidates.push((vector.id, dist));
                }
                continue;
            }

            // Cache miss: load from storage without holding the lock.
            match self.load_bucket(id).await {
                Ok(loaded) => {
                    for vector in &loaded.vectors {
                        if vector.data.len() != query_vec.len() {
                            continue;
                        }
                        let dist = l2_distance_f32(&vector.data, query_vec);
                        candidates.push((vector.id, dist));
                    }
                    // Insert into cache after use.
                    let mut cache = self.cache.lock();
                    cache.put(id, loaded);
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
        assert!((dist - expected).abs() < 1e-3, "expected {}, got {}", expected, dist);
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
        let mut cache = WorkerCache::new(2, usize::MAX);
        cache.put(0, CachedBucket { vectors: vec![], byte_size: 10 });
        cache.put(1, CachedBucket { vectors: vec![], byte_size: 10 });
        cache.put(2, CachedBucket { vectors: vec![], byte_size: 10 });
        // LRU should have evicted bucket 0
        assert!(cache.get(0).is_none(), "bucket 0 should be evicted");
        assert!(cache.get(1).is_some(), "bucket 1 should still be cached");
        assert!(cache.get(2).is_some(), "bucket 2 should still be cached");
    }

    /// WorkerCache skips buckets exceeding byte limit.
    #[test]
    fn worker_cache_rejects_oversized() {
        let mut cache = WorkerCache::new(10, 100); // max 100 bytes per bucket
        cache.put(0, CachedBucket { vectors: vec![], byte_size: 50 });
        cache.put(1, CachedBucket { vectors: vec![], byte_size: 200 }); // too big
        assert!(cache.get(0).is_some(), "small bucket should be cached");
        assert!(cache.get(1).is_none(), "oversized bucket should not be cached");
    }

    /// WorkerCache clear removes all entries.
    #[test]
    fn worker_cache_clear() {
        let mut cache = WorkerCache::new(10, usize::MAX);
        cache.put(0, CachedBucket { vectors: vec![], byte_size: 10 });
        cache.put(1, CachedBucket { vectors: vec![], byte_size: 10 });
        cache.clear();
        assert!(cache.get(0).is_none());
        assert!(cache.get(1).is_none());
    }

    /// WorkerCache invalidate_many removes specific entries.
    #[test]
    fn worker_cache_invalidate_many() {
        let mut cache = WorkerCache::new(10, usize::MAX);
        cache.put(0, CachedBucket { vectors: vec![], byte_size: 10 });
        cache.put(1, CachedBucket { vectors: vec![], byte_size: 10 });
        cache.put(2, CachedBucket { vectors: vec![], byte_size: 10 });
        cache.invalidate_many(&[0, 2]);
        assert!(cache.get(0).is_none(), "bucket 0 should be invalidated");
        assert!(cache.get(1).is_some(), "bucket 1 should remain");
        assert!(cache.get(2).is_none(), "bucket 2 should be invalidated");
    }
}
