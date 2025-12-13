use futures::channel::oneshot;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[derive(Clone)]
pub struct RouterResult {
    pub bucket_ids: Vec<u64>,
    pub routing_version: u64,
    pub affected_buckets: std::sync::Arc<Vec<u64>>,
}

pub struct RouterTask {
    pub query_vec: Vec<f32>,
    pub top_k: usize,
    pub respond_to: oneshot::Sender<anyhow::Result<RouterResult>>,
}

#[derive(Clone)]
pub struct ConsistentHashRing {
    ring: Vec<(u64, usize)>,
}

impl ConsistentHashRing {
    pub fn new(nodes: usize, virtual_nodes: usize) -> Self {
        let mut ring = Vec::new();
        for node in 0..nodes {
            for v in 0..virtual_nodes.max(1) {
                let mut hasher = DefaultHasher::new();
                (node as u64, v as u64).hash(&mut hasher);
                ring.push((hasher.finish(), node));
            }
        }
        ring.sort_by_key(|(h, _)| *h);
        Self { ring }
    }

    pub fn node_for(&self, key: u64) -> usize {
        if self.ring.is_empty() {
            return 0;
        }
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        let h = hasher.finish();
        for (hash, node) in &self.ring {
            if *hash >= h {
                return *node;
            }
        }
        self.ring[0].1
    }
}

#[cfg(test)]
mod tests {
    use super::ConsistentHashRing;

    #[test]
    fn hash_ring_is_deterministic() {
        let ring = ConsistentHashRing::new(4, 8);
        assert_eq!(ring.node_for(42), ring.node_for(42));
    }

    #[test]
    fn hash_ring_handles_single_node() {
        let ring = ConsistentHashRing::new(1, 1);
        assert_eq!(ring.node_for(99), 0);
    }
}
