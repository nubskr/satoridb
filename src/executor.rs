use crate::storage::{Bucket, Storage, Vector};
use anyhow::Result;
use lru::LruCache;
use std::cell::RefCell;
use std::convert::TryInto;
use std::num::NonZeroUsize;

pub struct Executor {
    storage: Storage,
    cache: RefCell<WorkerCache>,
}

#[derive(Clone)]
struct CachedBucket {
    vectors: Vec<Vector>,
    byte_size: usize,
}

#[derive(Clone)]
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

    fn get(&mut self, bucket_id: u64) -> Option<CachedBucket> {
        self.buckets.get(&bucket_id).cloned()
    }

    fn put(&mut self, bucket_id: u64, bucket: CachedBucket) {
        if bucket.byte_size <= self.bucket_max_bytes {
            self.buckets.put(bucket_id, bucket);
        }
    }
}

impl Executor {
    pub fn new(storage: Storage, cache: WorkerCache) -> Self {
        Self {
            storage,
            cache: RefCell::new(cache),
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

            if 8 + archive_len > chunk.len() {
                continue;
            }
            let data_slice = &chunk[8..8 + archive_len];

            if let Ok(archived_bucket) = rkyv::check_archived_root::<Bucket>(data_slice) {
                for vector in archived_bucket.vectors.iter() {
                    vectors.push(Vector {
                        id: vector.id,
                        data: vector.data.to_vec(),
                    });
                }
            }
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
    ) -> Result<Vec<(u64, f32)>> {
        let mut candidates = Vec::new();
        let mut cache = self.cache.borrow_mut();

        for &id in bucket_ids {
            if let Some(cached) = cache.get(id) {
                for vector in &cached.vectors {
                    if vector.data.len() != query_vec.len() {
                        continue;
                    }
                    let dist = l2_distance_f32(&vector.data, query_vec);
                    candidates.push((vector.id, dist));
                }
                continue;
            }

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
        unsafe { l2_distance_f32_avx2(a, b) }
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
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let diff = _mm256_sub_ps(va, vb);
        let sq = _mm256_mul_ps(diff, diff);
        sum = _mm256_add_ps(sum, sq);
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
