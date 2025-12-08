use std::collections::{HashMap, HashSet, BinaryHeap};
use std::cmp::Reverse;

const SPLIT_THRESHOLD: usize = 1000;
const MERGE_THRESHOLD: usize = 100;
const NUM_CLUSTERS_TO_SEARCH: usize = 10;
const REBALANCE_NEIGHBORS: usize = 64;

pub struct SpFresh {
    clusters: Vec<Cluster>,
    assignments: HashMap<u64, u32>,
    versions: HashMap<u64, u64>,
}

pub struct Cluster {
    id: u32,
    centroid: Vec<f32>,
    vectors: Vec<Vector>,
    deleted: HashSet<u64>,
}

pub struct Vector {
    id: u64,
    data: Vec<f32>,
}

impl SpFresh {
    // whatever
    pub fn new() -> Self {
        Self {
            clusters: Vec::new(),
            assignments: HashMap::new(),
            versions: HashMap::new(),
        }
    }

    // just a stupid helper
    fn distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
    }

    // just a helper
    fn find_merge_candidate(&self, cluster_id: u32) -> Option<u32> {
        let cluster = self.clusters.iter().find(|c| c.id == cluster_id)?;
        
        self.clusters
            .iter()
            .filter(|c| c.id != cluster_id)
            .min_by(|a, b| {
                Self::distance(&a.centroid, &cluster.centroid)
                    .partial_cmp(&Self::distance(&b.centroid, &cluster.centroid))
                    .unwrap()
            })
            .map(|c| c.id)
    }

    // helper
    fn compute_centroid(vectors: &[Vector]) -> Vec<f32> {
        if vectors.is_empty() {
            return Vec::new();
        }
        
        let dim = vectors[0].data.len();
        let mut centroid = vec![0.0; dim];
        
        for v in vectors {
            for (i, val) in v.data.iter().enumerate() {
                centroid[i] += val;
            }
        }
        
        let n = vectors.len() as f32;
        centroid.iter_mut().for_each(|x| *x /= n);
        centroid
    }

    // stupid helper
    fn find_nearest_cluster(&self, vector: &[f32]) -> Option<u32> {
        self.clusters
            .iter()
            .min_by(|a, b| {
                Self::distance(vector, &a.centroid)
                    .partial_cmp(&Self::distance(vector, &b.centroid))
                    .unwrap()
            })
            .map(|c| c.id)
    }

    // this is clowny
    pub fn update(&mut self, id: u64, vector: &[f32]) {
        self.delete(id);
        self.insert(id, vector);
    }

    // --------------------stupid part ends-------------------------------







    // !!
    pub fn insert(&mut self, id: u64, vector: &[f32]) {
        if self.clusters.is_empty() {
            self.clusters.push(Cluster {
                id: 0,
                centroid: vector.to_vec(),
                vectors: Vec::new(),
                deleted: HashSet::new(),
            });
        }

        let cluster_id = self.find_nearest_cluster(vector).unwrap();
        let cluster = self.clusters.iter_mut().find(|c| c.id == cluster_id).unwrap();
        
        cluster.vectors.push(Vector {
            id,
            data: vector.to_vec(),
        });

        self.assignments.insert(id, cluster_id);
        let version = self.versions.get(&id).unwrap_or(&0) + 1;
        self.versions.insert(id, version);

        // below shit is syncronous, we don't want that
        if cluster.vectors.len() > SPLIT_THRESHOLD {
            self.split(cluster_id);
        }
    }

    // !!
    pub fn delete(&mut self, id: u64) {
        let cluster_id = match self.assignments.get(&id) {
            Some(&cid) => cid,
            None => return,
        };

        if let Some(cluster) = self.clusters.iter_mut().find(|c| c.id == cluster_id) {
            cluster.deleted.insert(id);

            // auto merge if too sparse
            let live_count = cluster.vectors.len() - cluster.deleted.len();
            if live_count < MERGE_THRESHOLD && self.clusters.len() > 1 {
                if let Some(merge_target) = self.find_merge_candidate(cluster_id) {
                    self.merge(cluster_id, merge_target);
                }
            }
        }

        self.assignments.remove(&id);
    }

    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        let mut cluster_dists: Vec<_> = self.clusters
            .iter()
            .map(|c| (c.id, Self::distance(query, &c.centroid)))
            .collect();
        cluster_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut heap: BinaryHeap<Reverse<(ordered_float::NotNan<f32>, u64)>> = BinaryHeap::new();

        for (cluster_id, _) in cluster_dists.iter().take(NUM_CLUSTERS_TO_SEARCH) {
            let cluster = self.clusters.iter().find(|c| c.id == *cluster_id).unwrap();
            
            for vec in &cluster.vectors {
                if cluster.deleted.contains(&vec.id) {
                    continue;
                }
                let dist = Self::distance(query, &vec.data);
                if let Ok(d) = ordered_float::NotNan::new(dist) {
                    heap.push(Reverse((d, vec.id)));
                }
            }
        }

        heap.into_iter()
            .take(k)
            .map(|Reverse((d, id))| (id, d.into_inner()))
            .collect()
    }

    // need to be done in bg 
    pub fn compact(&mut self) {
        for cluster in &mut self.clusters {
            cluster.vectors.retain(|v| !cluster.deleted.contains(&v.id));
            cluster.deleted.clear();
        }
    }

    // need to be done in bg
    pub fn split(&mut self, cluster_id: u32) {
        let cluster = match self.clusters.iter_mut().find(|c| c.id == cluster_id) {
            Some(c) => c,
            None => return,
        };

        cluster.vectors.retain(|v| !cluster.deleted.contains(&v.id));
        cluster.deleted.clear();

        if cluster.vectors.len() < 2 {
            return;
        }

        // two farthest points as seeds
        let c1 = cluster.vectors[0].data.clone();
        let mut c2 = cluster.vectors[1].data.clone();
        let mut max_dist = 0.0f32;
        
        for v in &cluster.vectors {
            let d = Self::distance(&c1, &v.data);
            if d > max_dist {
                max_dist = d;
                c2 = v.data.clone();
            }
        }

        let mut group1 = Vec::new();
        let mut group2 = Vec::new();

        for v in cluster.vectors.drain(..) {
            if Self::distance(&v.data, &c1) < Self::distance(&v.data, &c2) {
                group1.push(v);
            } else {
                group2.push(v);
            }
        }

        cluster.centroid = Self::compute_centroid(&group1);
        cluster.vectors = group1;

        let new_id = self.clusters.iter().map(|c| c.id).max().unwrap_or(0) + 1;
        
        for v in &group2 {
            self.assignments.insert(v.id, new_id);
        }

        self.clusters.push(Cluster {
            id: new_id,
            centroid: Self::compute_centroid(&group2),
            vectors: group2,
            deleted: HashSet::new(),
        });

        // rebalance neighbors
        self.rebalance_after_split(cluster_id);
        self.rebalance_after_split(new_id);
    }

    // MUST be done in bg
    fn rebalance_after_split(&mut self, cluster_id: u32) {
        let centroid = match self.clusters.iter().find(|c| c.id == cluster_id) {
            Some(c) => c.centroid.clone(),
            None => return,
        };

        let mut cluster_dists: Vec<_> = self.clusters
            .iter()
            .filter(|c| c.id != cluster_id)
            .map(|c| (c.id, Self::distance(&c.centroid, &centroid)))
            .collect();
        cluster_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let neighbor_ids: Vec<u32> = cluster_dists
            .iter()
            .take(REBALANCE_NEIGHBORS)
            .map(|(id, _)| *id)
            .collect();

        for neighbor_id in neighbor_ids {
            self.maybe_migrate_vectors(neighbor_id, cluster_id, &centroid);
        }
    }

    // MUST be done in bg
    fn maybe_migrate_vectors(&mut self, from_id: u32, to_id: u32, to_centroid: &[f32]) {
        let from_cluster = match self.clusters.iter_mut().find(|c| c.id == from_id) {
            Some(c) => c,
            None => return,
        };

        let from_centroid = from_cluster.centroid.clone();
        let mut to_migrate = Vec::new();

        for v in &from_cluster.vectors {
            if from_cluster.deleted.contains(&v.id) {
                continue;
            }
            if Self::distance(&v.data, to_centroid) < Self::distance(&v.data, &from_centroid) {
                to_migrate.push(v.id);
            }
        }

        // extract vectors to move
        let mut moving = Vec::new();
        for id in &to_migrate {
            if let Some(pos) = from_cluster.vectors.iter().position(|v| v.id == *id) {
                moving.push(from_cluster.vectors.remove(pos));
            }
        }

        // update centroid
        from_cluster.centroid = Self::compute_centroid(&from_cluster.vectors);

        // move to target cluster
        let to_cluster = match self.clusters.iter_mut().find(|c| c.id == to_id) {
            Some(c) => c,
            None => return,
        };

        for v in moving {
            self.assignments.insert(v.id, to_id);
            to_cluster.vectors.push(v);
        }
        to_cluster.centroid = Self::compute_centroid(&to_cluster.vectors);
    }

    // MUST be done in bg
    pub fn merge(&mut self, cluster_a: u32, cluster_b: u32) {
        let b_idx = match self.clusters.iter().position(|c| c.id == cluster_b) {
            Some(idx) => idx,
            None => return,
        };
        let cluster_b_data = self.clusters.remove(b_idx);

        let cluster_a = match self.clusters.iter_mut().find(|c| c.id == cluster_a) {
            Some(c) => c,
            None => return,
        };

        for v in cluster_b_data.vectors {
            if !cluster_b_data.deleted.contains(&v.id) {
                self.assignments.insert(v.id, cluster_a.id);
                cluster_a.vectors.push(v);
            }
        }

        cluster_a.centroid = Self::compute_centroid(&cluster_a.vectors);
    }

}


/*
important parts are:

- splits
- merges
- rebalancings
- updating the routing HNSW indexes


doing all these in background while juggling with stateless executors

holy fuck :O
*/
