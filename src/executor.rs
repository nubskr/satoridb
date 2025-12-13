use crate::storage::{Storage, Vector};
use anyhow::Result;
use lru::LruCache;
use std::cell::{Cell, RefCell};
use std::convert::TryInto;
use std::num::NonZeroUsize;

pub struct Executor {
    storage: Storage,
    cache: RefCell<WorkerCache>,
    cache_version: Cell<u64>,
    last_changed: RefCell<std::sync::Arc<Vec<u64>>>,
}

struct CachedBucket {
    vectors: Vec<Vector>,
    byte_size: usize,
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
            cache: RefCell::new(cache),
            cache_version: Cell::new(0),
            last_changed: RefCell::new(std::sync::Arc::new(Vec::new())),
        }
    }

    async fn load_bucket(&self, bucket_id: u64) -> Result<CachedBucket> {
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
        changed_buckets: std::sync::Arc<Vec<u64>>,
    ) -> Result<Vec<(u64, f32)>> {
        if self.cache_version.get() != routing_version {
            let mut cache = self.cache.borrow_mut();
            if changed_buckets.is_empty() {
                cache.clear();
            } else {
                cache.invalidate_many(&changed_buckets);
            }
            self.cache_version.set(routing_version);
            *self.last_changed.borrow_mut() = changed_buckets;
        }

        let mut candidates = Vec::new();
        let mut cache = self.cache.borrow_mut();

        for &id in bucket_ids {
            match cache.get(id) {
                Some(cached) => {
                    for vector in &cached.vectors {
                        if vector.data.len() != query_vec.len() {
                            continue;
                        }
                        let dist = l2_distance_f32(&vector.data, query_vec);
                        candidates.push((vector.id, dist));
                    }
                }
                None => {
                    let loaded = self.load_bucket(id).await?;
                    for vector in &loaded.vectors {
                        if vector.data.len() != query_vec.len() {
                            continue;
                        }
                        let dist = l2_distance_f32(&vector.data, query_vec);
                        candidates.push((vector.id, dist));
                    }
                    cache.put(id, loaded);
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
}
