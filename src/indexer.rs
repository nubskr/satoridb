use crate::storage::{Bucket, Vector};
use log::info;
use rand::prelude::*;
use rand::seq::SliceRandom;

pub struct Indexer;

impl Indexer {
    /// basic k-means implementation
    /// consumes vectors and returns buckets
    pub fn build_clusters(vectors: Vec<Vector>, k: usize) -> Vec<Bucket> {
        if vectors.is_empty() || k == 0 {
            return vec![];
        }

        let n = vectors.len();
        let dim = vectors[0].data.len();
        if dim == 0 {
            return vec![];
        }

        // Optional: if your data can be ragged, this prevents UB in the AVX paths.
        // If it's always fixed-dim, remove for speed.
        if vectors.iter().any(|v| v.data.len() != dim) {
            return vec![];
        }

        let mut rng = rand::thread_rng();

        // Runtime feature detection ONCE (do not do this per distance call).
        let (use_avx2_fma, use_avx2) = detect_simd();

        // 1) Init centroids: contiguous (k * dim)
        let mut centroids = vec![0.0f32; k * dim];
        {
            // pick k random vectors (with replacement-ish fallback)
            let chosen: Vec<&Vector> = vectors.choose_multiple(&mut rng, k).collect();
            let mut c = 0usize;

            for v in chosen {
                centroids[c * dim..(c + 1) * dim].copy_from_slice(&v.data);
                c += 1;
            }

            while c < k {
                // fallback: random centroid
                let base = c * dim;
                for d in 0..dim {
                    centroids[base + d] = rng.gen::<f32>();
                }
                c += 1;
            }
        }

        let max_iters = 20;
        let mut assignments = vec![0usize; n];

        // Reused buffers to avoid per-iter allocations.
        let mut sums = vec![0.0f32; k * dim];
        let mut counts = vec![0u32; k];

        // Only needed for the 8-centroid-at-a-time kernel.
        let mut centroids_t = Vec::<f32>::new();

        for iter in 0..max_iters {
            info!("kmeans iter {}/{}", iter + 1, max_iters);

            // reset accumulators
            sums.fill(0.0);
            counts.fill(0);

            // build transposed centroid matrix if we can benefit
            let use_block8 = use_avx2_fma && k >= 8;
            if use_block8 {
                transpose_centroids_aos_to_soa(&centroids, k, dim, &mut centroids_t);
            }

            let mut changed = false;

            // Assign + accumulate
            for (i, v) in vectors.iter().enumerate() {
                let vec_data = &v.data;

                let best_cluster = if use_block8 {
                    // 8 centroids at a time: fast path
                    unsafe { nearest_centroid_l2_sq_avx2_fma(vec_data, &centroids_t, k, dim) }
                } else if use_avx2 {
                    // fallback: per-centroid AVX2 kernel (still faster than scalar for decent dim)
                    unsafe { nearest_centroid_l2_sq_avx2_pairwise(vec_data, &centroids, k, dim, use_avx2_fma) }
                } else {
                    nearest_centroid_l2_sq_scalar(vec_data, &centroids, k, dim)
                };

                if assignments[i] != best_cluster {
                    assignments[i] = best_cluster;
                    changed = true;
                }

                // accumulate into sums
                let base = best_cluster * dim;
                for d in 0..dim {
                    sums[base + d] += vec_data[d];
                }
                counts[best_cluster] += 1;
            }

            if !changed {
                break;
            }

            // Update centroids
            for j in 0..k {
                if counts[j] > 0 {
                    let inv = 1.0 / counts[j] as f32;
                    let base = j * dim;
                    for d in 0..dim {
                        centroids[base + d] = sums[base + d] * inv;
                    }
                } else {
                    // re-seed empty cluster with a random existing vector
                    if let Some(v) = vectors.choose(&mut rng) {
                        centroids[j * dim..(j + 1) * dim].copy_from_slice(&v.data);
                    }
                }
            }
        }

        // Build Buckets (reserve using final counts to avoid realloc churn)
        let mut final_counts = vec![0usize; k];
        for &a in &assignments {
            final_counts[a] += 1;
        }

        let mut buckets_data: Vec<Vec<Vector>> = (0..k).map(|j| Vec::with_capacity(final_counts[j])).collect();
        for (i, vec) in vectors.into_iter().enumerate() {
            let cluster_idx = assignments[i];
            buckets_data[cluster_idx].push(vec);
        }

        let mut buckets = Vec::new();
        for (i, b_vectors) in buckets_data.into_iter().enumerate() {
            if !b_vectors.is_empty() {
                let centroid_vec = centroids[i * dim..(i + 1) * dim].to_vec();
                let mut b = Bucket::new(i as u64, centroid_vec);
                b.vectors = b_vectors;
                buckets.push(b);
            }
        }

        let balanced = Self::rebalance_buckets(buckets);

        // Reindex buckets to ensure unique, dense IDs after rebalancing/splitting.
        let mut reindexed = Vec::with_capacity(balanced.len());
        for (i, b) in balanced.into_iter().enumerate() {
            let mut nb = Bucket::new(i as u64, b.centroid);
            nb.vectors = b.vectors;
            reindexed.push(nb);
        }

        reindexed
    }

    pub fn split_bucket(bucket: Bucket) -> Vec<Bucket> {
        Self::build_clusters(bucket.vectors, 2)
    }

    fn rebalance_buckets(buckets: Vec<Bucket>) -> Vec<Bucket> {
        if buckets.is_empty() {
            return buckets;
        }
        let total: usize = buckets.iter().map(|b| b.vectors.len()).sum();
        let avg = (total as f32 / buckets.len() as f32).max(1.0);
        let mut balanced = Vec::new();

        for bucket in buckets {
            if bucket.vectors.len() as f32 > avg * 1.2 && bucket.vectors.len() > 2 {
                let mut splits = Self::split_bucket(bucket);
                balanced.append(&mut splits);
            } else {
                balanced.push(bucket);
            }
        }

        balanced
    }
}

fn detect_simd() -> (bool, bool) {
    #[cfg(target_arch = "x86_64")]
    {
        let avx2 = std::arch::is_x86_feature_detected!("avx2");
        let fma = std::arch::is_x86_feature_detected!("fma");
        (avx2 && fma, avx2)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        (false, false)
    }
}

fn transpose_centroids_aos_to_soa(centroids_aos: &[f32], k: usize, dim: usize, out: &mut Vec<f32>) {
    out.clear();
    out.resize(dim * k, 0.0);
    for d in 0..dim {
        let row = d * k;
        for c in 0..k {
            out[row + c] = centroids_aos[c * dim + d];
        }
    }
}

fn nearest_centroid_l2_sq_scalar(point: &[f32], centroids: &[f32], k: usize, dim: usize) -> usize {
    let mut best = 0usize;
    let mut best_dist = f32::INFINITY;
    for c in 0..k {
        let base = c * dim;
        let mut acc = 0f32;
        for d in 0..dim {
            let diff = point[d] - centroids[base + d];
            acc += diff * diff;
        }
        if acc < best_dist {
            best_dist = acc;
            best = c;
        }
    }
    best
}

#[cfg(target_arch = "x86_64")]
unsafe fn nearest_centroid_l2_sq_avx2_pairwise(
    point: &[f32],
    centroids: &[f32],
    k: usize,
    dim: usize,
    use_fma: bool,
) -> usize {
    let mut best = 0usize;
    let mut best_dist = f32::INFINITY;

    for c in 0..k {
        let base = c * dim;
        let dist = if use_fma {
            l2_dist_sq_avx2_fma_dim(point.as_ptr(), centroids.as_ptr().add(base), dim)
        } else {
            l2_dist_sq_avx2_dim(point.as_ptr(), centroids.as_ptr().add(base), dim)
        };
        if dist < best_dist {
            best_dist = dist;
            best = c;
        }
    }
    best
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l2_dist_sq_avx2_dim(a: *const f32, b: *const f32, dim: usize) -> f32 {
    use std::arch::x86_64::*;

    let mut sum = _mm256_setzero_ps();
    let mut i = 0usize;

    while i + 8 <= dim {
        let va = _mm256_loadu_ps(a.add(i));
        let vb = _mm256_loadu_ps(b.add(i));
        let diff = _mm256_sub_ps(va, vb);
        let sq = _mm256_mul_ps(diff, diff);
        sum = _mm256_add_ps(sum, sq);
        i += 8;
    }

    let mut tmp = [0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), sum);
    let mut total = tmp.iter().sum::<f32>();

    while i < dim {
        let d = *a.add(i) - *b.add(i);
        total += d * d;
        i += 1;
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn l2_dist_sq_avx2_fma_dim(a: *const f32, b: *const f32, dim: usize) -> f32 {
    use std::arch::x86_64::*;

    let mut sum = _mm256_setzero_ps();
    let mut i = 0usize;

    while i + 8 <= dim {
        let va = _mm256_loadu_ps(a.add(i));
        let vb = _mm256_loadu_ps(b.add(i));
        let diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);
        i += 8;
    }

    let mut tmp = [0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), sum);
    let mut total = tmp.iter().sum::<f32>();

    while i < dim {
        let d = *a.add(i) - *b.add(i);
        total += d * d;
        i += 1;
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn nearest_centroid_l2_sq_avx2_fma(point: &[f32], centroids_t: &[f32], k: usize, dim: usize) -> usize {
    use std::arch::x86_64::*;

    let mut best_c = 0usize;
    let mut best_dist = f32::INFINITY;

    let mut c0 = 0usize;
    let mut tmp = [0f32; 8];

    while c0 + 8 <= k {
        let base = c0;
        let mut acc = _mm256_setzero_ps();

        for d in 0..dim {
            let p = _mm256_set1_ps(*point.get_unchecked(d));
            let cvals = _mm256_loadu_ps(centroids_t.as_ptr().add(d * k + base));
            let diff = _mm256_sub_ps(p, cvals);
            acc = _mm256_fmadd_ps(diff, diff, acc);
        }

        _mm256_storeu_ps(tmp.as_mut_ptr(), acc);

        for lane in 0..8 {
            let dist = tmp[lane];
            if dist < best_dist {
                best_dist = dist;
                best_c = base + lane;
            }
        }

        c0 += 8;
    }

    // tail centroids
    while c0 < k {
        let mut accs = 0f32;
        for d in 0..dim {
            let cd = *centroids_t.get_unchecked(d * k + c0);
            let diff = *point.get_unchecked(d) - cd;
            accs += diff * diff;
        }
        if accs < best_dist {
            best_dist = accs;
            best_c = c0;
        }
        c0 += 1;
    }

    best_c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::Vector;

    #[test]
    fn builds_reasonable_clusters() {
        let vectors = vec![
            Vector::new(0, vec![0.0, 0.0]),
            Vector::new(1, vec![0.0, 1.0]),
            Vector::new(2, vec![10.0, 10.0]),
            Vector::new(3, vec![10.0, 11.0]),
        ];
        let buckets = Indexer::build_clusters(vectors, 2);
        assert_eq!(buckets.len(), 2);
        let total_vectors: usize = buckets.iter().map(|b| b.vectors.len()).sum();
        assert_eq!(total_vectors, 4);
    }

    #[test]
    fn handles_empty() {
        let buckets = Indexer::build_clusters(Vec::new(), 2);
        assert!(buckets.is_empty());
    }
}

