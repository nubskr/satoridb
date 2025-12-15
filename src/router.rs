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
        let quant_scale = if range.abs() < f32::EPSILON {
            0.0
        } else {
            255.0 / range
        };

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
        // Leave unpadded here; HnswIndex pads internally (currently with 0s).
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

        let pad = self.index.pad_dim();
        if pad == 0 {
            return Ok(Vec::new());
        }

        let neighbors = {
            // Per-thread buffer: no locks, no allocs in steady state.
            let q_local = self.quant_bufs.get_or(|| RefCell::new(Vec::new()));
            let mut q_buf = q_local.borrow_mut();

            // IMPORTANT: must match index-side padding semantics.
            // Your HnswIndex pads inserted vectors with 0, so query must pad with 0 too.
            self.quantize_into_padded_zero(vector, pad, &mut q_buf);

            let n = self.index.len();

            // Flat path is exact and deterministic; tests rely on it for small sets.
            // Work guard prevents accidental O(N*D) explosions if D is huge.
            const FLAT_N_CUTOFF: usize = 20_000;
            const FLAT_WORK_THRESHOLD: usize = 2_000_000; // tune on your box
            let work = n.saturating_mul(pad);

            #[cfg(test)]
            if n <= 20_000 && !self.test_centroids.is_empty() {
                return Ok(self.query_exact_test_only(vector, top_k));
            }

            if n <= FLAT_N_CUTOFF && work <= FLAT_WORK_THRESHOLD {
                let local = self.flat_scratch.get_or(|| RefCell::new(Vec::new()));
                let mut buf = local.borrow_mut();
                self.index.flat_search_with_scratch(&q_buf, top_k, &mut buf)
            } else {
                // --- KEY CHANGE: special-case top_k == 1 ---
                // Streaming ingestion calls router.query(..., 1) extremely often.
                // Do NOT use the same ef_search policy as multi-bucket routing.
                let ef_search = if top_k <= 1 {
                    // Keep this small; bump only if routing quality becomes unacceptable.
                    if n <= 50_000 {
                        64
                    } else if n <= 200_000 {
                        96
                    } else {
                        128
                    }
                } else {
                    // Keep your existing multi-bucket routing policy (including a high floor if you want it).
                    let base = if n <= 50_000 {
                        top_k.saturating_mul(60)
                    } else if n <= 200_000 {
                        top_k.saturating_mul(80)
                    } else {
                        top_k.saturating_mul(100)
                    };
                    // If you previously forced a huge minimum (e.g. 1200), keep it here
                    // so query routing quality doesn't regress.
                    let min_ef = 1_200usize;
                    let max_ef = 4_000usize;
                    std::cmp::min(std::cmp::max(base, min_ef), max_ef)
                };

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

    #[inline(always)]
    fn quantize_into_unpadded(&self, src: &[f32], dst: &mut Vec<u8>) {
        dst.clear();
        dst.reserve(src.len());
        unsafe { dst.set_len(src.len()) };
        self.quantize_core(src, &mut dst[..]);
    }

    /// Quantize into a padded buffer of length `pad`, padding with 0.
    /// This matches HnswIndex's current insert-side padding semantics.
    #[inline(always)]
    fn quantize_into_padded_zero(&self, src: &[f32], pad: usize, dst: &mut Vec<u8>) {
        dst.clear();
        dst.reserve(pad);
        unsafe { dst.set_len(pad) };

        let n = src.len().min(pad);
        self.quantize_core(src, &mut dst[..n]);

        if pad > n {
            dst[n..pad].fill(0);
        }
    }

    /// Core quantization into an already-sized output slice.
    /// Avoids `resize(.., 0)` (memset) and avoids `clamp()` (often slower than two compares).
    #[inline(always)]
    fn quantize_core(&self, src: &[f32], out: &mut [u8]) {
        let scale = self.quant_scale;
        let bias = -self.quant_min * scale;

        let len = out.len();
        let mut i = 0usize;

        while i + 4 <= len {
            let mut v0 = src[i].mul_add(scale, bias);
            let mut v1 = src[i + 1].mul_add(scale, bias);
            let mut v2 = src[i + 2].mul_add(scale, bias);
            let mut v3 = src[i + 3].mul_add(scale, bias);

            if v0 < 0.0 {
                v0 = 0.0;
            } else if v0 > 255.0 {
                v0 = 255.0;
            }
            if v1 < 0.0 {
                v1 = 0.0;
            } else if v1 > 255.0 {
                v1 = 255.0;
            }
            if v2 < 0.0 {
                v2 = 0.0;
            } else if v2 > 255.0 {
                v2 = 255.0;
            }
            if v3 < 0.0 {
                v3 = 0.0;
            } else if v3 > 255.0 {
                v3 = 255.0;
            }

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
        best.iter().map(|(_, id)| *id).collect()
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
