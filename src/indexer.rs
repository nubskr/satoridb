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

        let dim = vectors[0].data.len();
        let mut rng = rand::thread_rng();

        // 1. Init centroids (Forgy method: choose k random observations)
        let mut centroids: Vec<Vec<f32>> = vectors
            .choose_multiple(&mut rng, k)
            .map(|v| v.data.clone())
            .collect();

        // If we didn't get enough unique vectors (e.g. dataset < k), handle it
        while centroids.len() < k {
             // fill with random noise or just duplicate
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
            for (i, vec) in vectors.iter().enumerate() {
                let mut min_dist = f32::MAX;
                let mut best_cluster = 0;

                for (c_idx, centroid) in centroids.iter().enumerate() {
                    let dist = l2_dist_sq(&vec.data, centroid);
                    if dist < min_dist {
                        min_dist = dist;
                        best_cluster = c_idx;
                    }
                }

                if assignments[i] != best_cluster {
                    assignments[i] = best_cluster;
                    changed = true;
                }

                // Accumulate for update
                for (d, val) in vec.data.iter().enumerate() {
                    sums[best_cluster][d] += val;
                }
                counts[best_cluster] += 1;
            }

            if !changed {
                break;
            }

            // Update centroids
            for j in 0..k {
                if counts[j] > 0 {
                    for d in 0..dim {
                        centroids[j][d] = sums[j][d] / counts[j] as f32;
                    }
                } else {
                    // Re-init empty cluster to a random vector's position (or keep same)
                    // Simple strategy: leave it or reset to a random point. 
                    // Leaving it often results in it staying empty.
                    // Resetting to a random vector from the dataset:
                    if let Some(v) = vectors.choose(&mut rng) {
                        centroids[j] = v.data.clone();
                    }
                }
            }
        }

        // Build Buckets
        // We need to group vectors by assignment.
        // Since we consumed 'vectors', we can move them.
        
        // Prepare buckets
        let mut buckets_data: Vec<Vec<Vector>> = (0..k).map(|_| Vec::new()).collect();
        
        for (i, vec) in vectors.into_iter().enumerate() {
            let cluster_idx = assignments[i];
            buckets_data[cluster_idx].push(vec);
        }

        let mut buckets = Vec::new();
        for (i, b_vectors) in buckets_data.into_iter().enumerate() {
            // Only create bucket if it has vectors (or we could keep empty ones)
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
