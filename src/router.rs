use crate::quantizer::Quantizer;
use crate::router_hnsw::HnswIndex;
use anyhow::Result;
use parking_lot::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

pub struct Router {
    index: HnswIndex,
    quantizer: Quantizer,
}

impl Router {
    pub fn new(_max_elements: usize, quantizer: Quantizer) -> Self {
        // M=16, ef_construction=100 are standard defaults
        let index = HnswIndex::new(16, 100);
        Self { index, quantizer }
    }

    pub fn add_centroid(&mut self, id: u64, vector: &[f32]) {
        let q_vec = self.quantizer.quantize(vector);
        self.index.insert(id as usize, q_vec);
    }

    pub fn query(&self, vector: &[f32], top_k: usize) -> Result<Vec<u64>> {
        // Quantize the incoming float vector to match centroid space.
        let q_vec = self.quantizer.quantize(vector);
        let ef_search = std::cmp::max(top_k * 20, 200);
        let neighbors = self.index.search(&q_vec, top_k, ef_search);

        let ids = neighbors.into_iter().map(|id| id as u64).collect();

        Ok(ids)
    }
}

/// Atomically published routing state so worker threads can refresh cheaply.
#[derive(Clone)]
pub struct RoutingTable {
    version: Arc<AtomicU64>,
    router: Arc<RwLock<Option<RoutingData>>>,
}

impl RoutingTable {
    pub fn new() -> Self {
        Self {
            version: Arc::new(AtomicU64::new(0)),
            router: Arc::new(RwLock::new(None)),
        }
    }

    /// Install a new router and bump the version counter.
    pub fn install(&self, router: Router, changed_buckets: Vec<u64>) -> u64 {
        let next = self.version.fetch_add(1, Ordering::AcqRel) + 1;
        *self.router.write() = Some(RoutingData {
            router: Arc::new(router),
            changed: Arc::new(changed_buckets),
        });
        next
    }

    /// Returns the current router and version, if one has been installed.
    pub fn snapshot(&self) -> Option<RoutingSnapshot> {
        let router_guard = self.router.read();
        router_guard.as_ref().cloned().map(|data| RoutingSnapshot {
            router: data.router.clone(),
            version: self.version.load(Ordering::Acquire),
            changed: data.changed.clone(),
        })
    }

    pub fn current_version(&self) -> u64 {
        self.version.load(Ordering::Acquire)
    }
}

#[derive(Clone)]
pub struct RoutingSnapshot {
    pub router: Arc<Router>,
    pub version: u64,
    pub changed: Arc<Vec<u64>>,
}

#[derive(Clone)]
struct RoutingData {
    router: Arc<Router>,
    changed: Arc<Vec<u64>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantizer::Quantizer;

    #[test]
    fn routes_to_nearest_centroid() {
        let quantizer = Quantizer::new(0.0, 1.0);
        let mut router = Router::new(10, quantizer);
        router.add_centroid(42, &[0.0, 0.0]);
        router.add_centroid(7, &[1.0, 1.0]);

        let ids = router.query(&[0.1, 0.0], 1).unwrap();
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0], 42);
    }
}
