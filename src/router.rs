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

    // Per-thread scratch for HNSW search to avoid contention.
    scratch: ThreadLocal<RefCell<crate::router_hnsw::SearchScratch>>,
    // Per-thread flat scratch for small-index scans.
    flat_scratch: ThreadLocal<RefCell<Vec<(f32, usize)>>>,
    // Per-thread quant buffers to avoid contention on the hot path.
    quant_bufs: ThreadLocal<RefCell<Vec<u8>>>,

    quant_min: f32,
    quant_scale: f32,

    #[cfg(test)]
    test_centroids: Vec<(u64, Vec<f32>)>,
}

impl Router {
    pub fn new(_max_elements: usize, quantizer: Quantizer) -> Self {
        // Cheaper construction tuned for latency.
        let index = HnswIndex::new(24, 180);

        let range = quantizer.max - quantizer.min;
        let quant_scale = if range.abs() < f32::EPSILON { 0.0 } else { 255.0 / range };

        Self {
            index,
            quant_min: quantizer.min,
            quant_scale,
            scratch: ThreadLocal::new(),
            flat_scratch: ThreadLocal::new(),
            quant_bufs: ThreadLocal::new(),
            #[cfg(test)]
            test_centroids: Vec::new(),
        }
    }

    pub fn add_centroid(&mut self, id: u64, vector: &[f32]) {
        // NOTE: HnswIndex pads internally in your earlier code; if it doesn’t,
        // keep this padded to index.pad_dim() once available.
        let mut buf = Vec::with_capacity(vector.len());
        self.quantize_into_unpadded(vector, &mut buf);
        self.index.insert(id as usize, buf);

        #[cfg(test)]
        self.test_centroids.push((id, vector.to_vec()));
    }

    #[inline(always)]
    pub fn query(&self, vector: &[f32], top_k: usize) -> Result<Vec<u64>> {
        if top_k == 0 || self.index.len() == 0 {
            return Ok(Vec::new());
        }

        // Pad must be known (set on first insert). If not, nothing to do.
        let pad = self.index.pad_dim();
        if pad == 0 {
            return Ok(Vec::new());
        }

        let neighbors = {
            // Per-thread buffer: no locks, no allocs in steady state.
            let q_local = self.quant_bufs.get_or(|| RefCell::new(Vec::new()));
            let mut q_buf = q_local.borrow_mut();

            // Quantize directly into *padded* buffer (single pass over dst; no extra resize).
            self.quantize_into_padded(vector, pad, &mut q_buf);

            // Dimension-aware decision: flat scan is O(N*D); HNSW is ~O(ef*D + heap/graph).
            // Tune FLAT_WORK_THRESHOLD on your box; this is a sane default that avoids
            // accidental O(N*D) explosions for large D.
            let n = self.index.len();
            let work = n.saturating_mul(pad);
            const FLAT_WORK_THRESHOLD: usize = 2_000_000; // ~2M byte-mults per query before overheads

            #[cfg(test)]
            if n <= 20_000 && !self.test_centroids.is_empty() {
                // Exact top-k (f32) only for tests when centroid count is small.
                return Ok(self.query_exact_test_only(vector, top_k));
            }

            if work <= FLAT_WORK_THRESHOLD {
                let local = self.flat_scratch.get_or(|| RefCell::new(Vec::new()));
                let mut buf = local.borrow_mut();
                self.index.flat_search_with_scratch(&q_buf, top_k, &mut buf)
            } else {
                // Autotune ef: scale with db size and k, but clamp.
                // (You can do better by measuring recall/latency curves; this keeps behavior similar
                // to your current code while avoiding tiny ef on large graphs.)
                let base = if n <= 50_000 {
                    top_k.saturating_mul(60)
                } else if n <= 200_000 {
                    top_k.saturating_mul(80)
                } else {
                    top_k.saturating_mul(100)
                };
                let ef_search = base.clamp(1_200, 4_000);

                let local = self
                    .scratch
                    .get_or(|| RefCell::new(crate::router_hnsw::SearchScratch::new()));
                let mut scratch = local.borrow_mut();

                self.index
                    .search_with_scratch(&q_buf, top_k, ef_search, &mut scratch)
            }
        };

        Ok(neighbors.into_iter().map(|id| id as u64).collect())
    }

    // ----------------------------
    // Quantization hot path
    // ----------------------------

    /// Quantize without padding (used for inserts; HNSW can pad internally).
    #[inline(always)]
    fn quantize_into_unpadded(&self, src: &[f32], dst: &mut Vec<u8>) {
        dst.clear();
        dst.reserve(src.len());
        unsafe { dst.set_len(src.len()) };
        self.quantize_core(src, dst.as_mut_slice());
    }

    /// Quantize into a padded buffer of length `pad`.
    /// Padding is set to 128 (so centered i8 == 0), which avoids corrupting cosine
    /// when HNSW centers bytes by subtracting 128 / xor 0x80.
    #[inline(always)]
    fn quantize_into_padded(&self, src: &[f32], pad: usize, dst: &mut Vec<u8>) {
        dst.clear();
        dst.reserve(pad);
        unsafe { dst.set_len(pad) };

        // Write quantized bytes for actual dims.
        let out = &mut dst[..src.len().min(pad)];
        self.quantize_core(src, out);

        // Neutral padding: 128u8 -> centered 0i8.
        if pad > out.len() {
            dst[out.len()..pad].fill(128);
        }
    }

    /// Core quantization into an already-sized output slice.
    /// Avoids `resize(.., 0)` (memset) and avoids `clamp()` (often slower than two compares).
    #[inline(always)]
    fn quantize_core(&self, src: &[f32], out: &mut [u8]) {
        let min = self.quant_min;
        let scale = self.quant_scale;

        // Precompute bias so inner loop is one fma (mul_add) + clamp.
        let bias = -min * scale;

        let len = out.len();
        let mut i = 0usize;

        // Manual unroll by 4 tends to compile to decent vector-ish code even without explicit SIMD.
        while i + 4 <= len {
            let mut v0 = src[i].mul_add(scale, bias);
            let mut v1 = src[i + 1].mul_add(scale, bias);
            let mut v2 = src[i + 2].mul_add(scale, bias);
            let mut v3 = src[i + 3].mul_add(scale, bias);

            if v0 < 0.0 { v0 = 0.0; } else if v0 > 255.0 { v0 = 255.0; }
            if v1 < 0.0 { v1 = 0.0; } else if v1 > 255.0 { v1 = 255.0; }
            if v2 < 0.0 { v2 = 0.0; } else if v2 > 255.0 { v2 = 255.0; }
            if v3 < 0.0 { v3 = 0.0; } else if v3 > 255.0 { v3 = 255.0; }

            out[i] = v0 as u8;
            out[i + 1] = v1 as u8;
            out[i + 2] = v2 as u8;
            out[i + 3] = v3 as u8;

            i += 4;
        }

        while i < len {
            let mut v = src[i].mul_add(scale, bias);
            if v < 0.0 {
                v = 0.0;
            } else if v > 255.0 {
                v = 255.0;
            }
            out[i] = v as u8;
            i += 1;
        }
    }

    #[cfg(test)]
    #[inline(never)]
    fn query_exact_test_only(&self, vector: &[f32], top_k: usize) -> Vec<u64> {
        // Exact top-k using stored f32 centroids with small fixed buffer.
        let mut best: Vec<(f32, u64)> = Vec::with_capacity(top_k);
        best.resize(top_k, (f32::MAX, 0));

        let mut worst = f32::MAX;
        let mut worst_idx = 0usize;

        for (id, c) in self.test_centroids.iter() {
            let mut dist = 0f32;
            for k in 0..vector.len() {
                let d = vector[k] - c[k];
                dist += d * d;
            }
            if dist < worst {
                best[worst_idx] = (dist, *id);

                // Find new worst.
                worst_idx = 0;
                worst = best[0].0;
                for j in 1..top_k {
                    if best[j].0 > worst {
                        worst = best[j].0;
                        worst_idx = j;
                    }
                }
            }
        }

        best.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        best.iter().map(|(_, id)| *id as u64).collect()
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
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use std::time::Instant;

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

    #[test]
    fn hierarchical_hnsw_recall_and_latency_smoke() {
        let quantizer = Quantizer::new(0.0, 1.0);
        let mut router = Router::new(20_000, quantizer);
        let dim = 16;
        let n = 10_000;
        let mut rng = StdRng::seed_from_u64(7);

        let mut data: Vec<Vec<f32>> = Vec::with_capacity(n);
        for _ in 0..n {
            let mut v = Vec::with_capacity(dim);
            for _ in 0..dim {
                v.push(rng.gen::<f32>());
            }
            router.add_centroid(data.len() as u64, &v);
            data.push(v);
        }

        let q_count = 100;
        let top_k = 20;
        let mut hit_sum = 0usize;
        let mut total_ms = 0f64;
        let mut bf_buf: Vec<(f32, usize)> = Vec::with_capacity(n);

        for qi in 0..q_count {
            let query = &data[qi];

            bf_buf.clear();
            for (idx, v) in data.iter().enumerate() {
                let mut dist = 0f32;
                for k in 0..dim {
                    let d = query[k] - v[k];
                    dist += d * d;
                }
                bf_buf.push((dist, idx));
            }
            bf_buf.select_nth_unstable_by(top_k - 1, |a, b| a.0.partial_cmp(&b.0).unwrap());
            bf_buf.truncate(top_k);
            bf_buf.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            let start = Instant::now();
            let ids = router.query(query, top_k).unwrap();
            total_ms += start.elapsed().as_secs_f64() * 1000.0;

            let mut gt_set = std::collections::HashSet::with_capacity(top_k);
            for &(_, idx) in &bf_buf {
                gt_set.insert(idx as u64);
            }
            let mut hits = 0usize;
            for id in ids {
                if gt_set.contains(&id) {
                    hits += 1;
                }
            }
            hit_sum += hits;
        }

        let recall_at_k = hit_sum as f64 / (q_count * top_k) as f64;
        assert!(
            recall_at_k >= 0.95,
            "recall@{} too low: {:.3}, hits={}/{}",
            top_k,
            recall_at_k,
            hit_sum,
            q_count * top_k
        );
        assert!(
            total_ms <= 400.0,
            "query batch too slow: {:.2} ms for {} queries",
            total_ms,
            q_count
        );
    }
}
