use crate::storage::{Vector, Bucket};
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

        // Convert to f32 for K-Means math
        let float_data: Vec<Vec<f32>> = vectors.iter()
            .map(|v| v.data.iter().map(|&x| x as f32).collect())
            .collect();

        let dim = float_data[0].len();
        let mut rng = rand::thread_rng();

        // 1. Init centroids
        let mut centroids: Vec<Vec<f32>> = float_data
            .choose_multiple(&mut rng, k)
            .cloned()
            .collect();

        while centroids.len() < k {
             let mut random_vec = vec![0.0; dim];
             for x in random_vec.iter_mut() { *x = rng.gen(); }
             centroids.push(random_vec);
        }

        let max_iters = 20;
        let mut assignments = vec![0; vectors.len()];

        for _iter in 0..max_iters {
            let mut changed = false;
            let mut sums = vec![vec![0.0; dim]; k];
            let mut counts = vec![0; k];

            // Assign
            for (i, vec_data) in float_data.iter().enumerate() {
                let mut min_dist = f32::MAX;
                let mut best_cluster = 0;

                for (c_idx, centroid) in centroids.iter().enumerate() {
                    let dist = l2_dist_sq(vec_data, centroid);
                    if dist < min_dist {
                        min_dist = dist;
                        best_cluster = c_idx;
                    }
                }

                if assignments[i] != best_cluster {
                    assignments[i] = best_cluster;
                    changed = true;
                }

                // Accumulate
                for (d, &val) in vec_data.iter().enumerate() {
                    sums[best_cluster][d] += val;
                }
                counts[best_cluster] += 1;
            }

            if !changed {
                break;
            }

            // Update
            for j in 0..k {
                if counts[j] > 0 {
                    for d in 0..dim {
                        centroids[j][d] = sums[j][d] / counts[j] as f32;
                    }
                } else {
                    if let Some(v) = float_data.choose(&mut rng) {
                        centroids[j] = v.clone();
                    }
                }
            }
        }

        // Build Buckets
        let mut buckets_data: Vec<Vec<Vector>> = (0..k).map(|_| Vec::new()).collect();
        
        for (i, vec) in vectors.into_iter().enumerate() {
            let cluster_idx = assignments[i];
            buckets_data[cluster_idx].push(vec);
        }

        let mut buckets = Vec::new();
        for (i, b_vectors) in buckets_data.into_iter().enumerate() {
            if !b_vectors.is_empty() {
                let mut b = Bucket::new(i as u64, centroids[i].clone());
                b.vectors = b_vectors;
                buckets.push(b);
            }
        }

        buckets
    }

    pub fn split_bucket(bucket: Bucket) -> Vec<Bucket> {
        Self::build_clusters(bucket.vectors, 2)
    }
}

fn l2_dist_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter())
     .map(|(x, y)| (x - y).powi(2))
     .sum()
}
