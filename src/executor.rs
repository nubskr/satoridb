use crate::storage::{Storage, Bucket};
use anyhow::Result;
use lru::LruCache;
use std::cell::RefCell;
use glommio::io::ReadResult;
use std::convert::TryInto;
use std::rc::Rc;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;

pub struct Executor {
    storage: Storage,
    cache: RefCell<LruCache<u64, Rc<ReadResult>>>,
    current_cache_bytes: RefCell<usize>,
}

const CACHE_LIMIT_BYTES: usize = 200 * 1024 * 1024; // 200 MB

impl Executor {
    pub fn new(storage: Storage) -> Self {
        Self { 
            storage,
            cache: RefCell::new(LruCache::unbounded()),
            current_cache_bytes: RefCell::new(0),
        }
    }

    pub async fn query(&self, query_vec: &[f32], bucket_ids: &[u64], top_k: usize) -> Result<Vec<(u64, f32)>> {
        let mut candidates = Vec::new();

        for &id in bucket_ids {
            let buf_opt = {
                let mut cache = self.cache.borrow_mut();
                cache.get(&id).cloned()
            };

            let buf = match buf_opt {
                Some(b) => b,
                None => {
                    // Cache Miss
                    match self.storage.fetch_raw(id).await {
                        Ok(b) => {
                            let b_rc = Rc::new(b);
                            // Eviction Logic
                            let mut cache = self.cache.borrow_mut();
                            let mut current_bytes = self.current_cache_bytes.borrow_mut();
                            
                            while *current_bytes + b_rc.len() > CACHE_LIMIT_BYTES && cache.len() > 0 {
                                if let Some((_, evicted)) = cache.pop_lru() {
                                    *current_bytes -= evicted.len();
                                }
                            }
                            
                            cache.put(id, b_rc.clone());
                            *current_bytes += b_rc.len();
                            b_rc
                        }
                        Err(_) => continue, // Bucket not found
                    }
                }
            };

            // Access Archived Data (Zero-Copy)
            // Double deref: Rc -> ReadResult -> [u8]
            let bytes = &**buf;
            
            if bytes.len() < 8 { continue; }
            let len_bytes: [u8; 8] = bytes[0..8].try_into().unwrap_or([0; 8]);
            let archive_len = u64::from_le_bytes(len_bytes) as usize;
            
            if 8 + archive_len > bytes.len() { continue; }
            let data_slice = &bytes[8..8+archive_len];

            // Access bucket using rkyv
            if let Ok(archived_bucket) = rkyv::check_archived_root::<Bucket>(data_slice) {
                 for vector in archived_bucket.vectors.iter() {
                     // Verify dimensions
                     if vector.data.len() != query_vec.len() { continue; }
                     
                     let dist = l2_distance(&vector.data, query_vec);
                     candidates.push((vector.id, dist));
                 }
            }
        }

        // Sort by distance (ascending for L2)
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Take top_k
        if candidates.len() > top_k {
            candidates.truncate(top_k);
        }

        Ok(candidates)
    }
}

fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { l2_distance_avx2(a, b) };
        }
    }
    l2_distance_scalar(a, b)
}

fn l2_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
     .zip(b.iter())
     .map(|(x, y)| (x - y).powi(2))
     .sum::<f32>()
     .sqrt()
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn l2_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    let mut i = 0;
    let len = a.len();
    let mut sum256 = _mm256_setzero_ps();

    while i + 8 <= len {
        let a_ptr = a.as_ptr().add(i);
        let b_ptr = b.as_ptr().add(i);

        // Use unaligned load as Vec<f32> is not guaranteed to be 32-byte aligned
        let va = _mm256_loadu_ps(a_ptr);
        let vb = _mm256_loadu_ps(b_ptr);

        let diff = _mm256_sub_ps(va, vb);
        let diff_sq = _mm256_mul_ps(diff, diff);
        
        sum256 = _mm256_add_ps(sum256, diff_sq);
        i += 8;
    }

    // Horizontal sum of 8 floats in YMM register
    // 1. Extract low 128 bits
    let sum128_lo = _mm256_castps256_ps128(sum256);
    // 2. Extract high 128 bits
    let sum128_hi = _mm256_extractf128_ps(sum256, 1);
    // 3. Add them
    let sum128 = _mm_add_ps(sum128_lo, sum128_hi);
    // 4. Horizontal add the 4 floats in XMM
    let sum128 = _mm_hadd_ps(sum128, sum128);
    let sum128 = _mm_hadd_ps(sum128, sum128);
    
    let mut sum = _mm_cvtss_f32(sum128);

    // Handle remaining elements
    while i < len {
        let diff = a[i] - b[i];
        sum += diff * diff;
        i += 1;
    }

    sum.sqrt()
}