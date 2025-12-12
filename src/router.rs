use crate::quantizer::Quantizer;
use crate::router_hnsw::HnswIndex;
use anyhow::Result;
use parking_lot::RwLock;
use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use thread_local::ThreadLocal;

pub struct Router {
    index: HnswIndex,
    scratch: parking_lot::Mutex<crate::router_hnsw::SearchScratch>,
    flat_scratch: parking_lot::Mutex<Vec<(f32, usize)>>,
    // Per-thread quant buffers to avoid contention on the hot path.
    quant_bufs: ThreadLocal<RefCell<Vec<u8>>>,
    quant_min: f32,
    quant_scale: f32,
}

impl Router {
    pub fn new(_max_elements: usize, quantizer: Quantizer) -> Self {
        // M=16, ef_construction=100 are standard defaults
        let index = HnswIndex::new(16, 100);
        let range = quantizer.max - quantizer.min;
        let quant_scale = if range.abs() < f32::EPSILON {
            0.0
        } else {
            255.0 / range
        };
        Self {
            index,
            quant_min: quantizer.min,
            quant_scale,
            scratch: parking_lot::Mutex::new(crate::router_hnsw::SearchScratch::new()),
            flat_scratch: parking_lot::Mutex::new(Vec::new()),
            quant_bufs: thread_local::ThreadLocal::new(),
        }
    }

    pub fn add_centroid(&mut self, id: u64, vector: &[f32]) {
        let mut buf = Vec::with_capacity(vector.len());
        self.quantize_into(vector, &mut buf);
        self.index.insert(id as usize, buf);
    }

    pub fn query(&self, vector: &[f32], top_k: usize) -> Result<Vec<u64>> {
        let neighbors = {
            // Get a per-thread buffer to avoid locking.
            let q_local = self
                .quant_bufs
                .get_or(|| std::cell::RefCell::new(Vec::new()));
            let mut q_buf = q_local.borrow_mut();
            self.quantize_into(vector, &mut q_buf);

            // For small centroid sets, flat scan is cheaper than graph traversal.
            if self.index.len() <= 512 {
                let mut buf = self.flat_scratch.lock();
                self.index.flat_search_with_scratch(&q_buf, top_k, &mut buf)
            } else {
                let ef_search = std::cmp::max(top_k * 10, 150);
                // Reuse scratch to avoid per-query allocations.
                let mut scratch = self.scratch.lock();
                self.index
                    .search_with_scratch(&q_buf, top_k, ef_search, &mut scratch)
            }
        };

        let ids = neighbors.into_iter().map(|id| id as u64).collect();

        Ok(ids)
    }

    #[inline(always)]
    fn quantize_into(&self, src: &[f32], dst: &mut Vec<u8>) {
        dst.clear();
        dst.resize(src.len(), 0);
        let min = self.quant_min;
        let scale = self.quant_scale;
        // Manually unroll in chunks of 4 to help auto-vectorization.
        let len = src.len();
        let mut i = 0;
        while i + 4 <= len {
            let v0 = (src[i] - min) * scale;
            let v1 = (src[i + 1] - min) * scale;
            let v2 = (src[i + 2] - min) * scale;
            let v3 = (src[i + 3] - min) * scale;
            dst[i] = v0.clamp(0.0, 255.0) as u8;
            dst[i + 1] = v1.clamp(0.0, 255.0) as u8;
            dst[i + 2] = v2.clamp(0.0, 255.0) as u8;
            dst[i + 3] = v3.clamp(0.0, 255.0) as u8;
            i += 4;
        }
        while i < len {
            let v = (src[i] - min) * scale;
            dst[i] = v.clamp(0.0, 255.0) as u8;
            i += 1;
        }
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
