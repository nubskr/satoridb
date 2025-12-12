use smallvec::SmallVec;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Minimal single-layer HNSW-style graph for routing centroids.
/// Stores byte-quantized vectors and keeps up to `m` neighbors per node.
pub struct HnswIndex {
    m: usize,
    ef_construction: usize,
    vectors: Vec<Vec<u8>>,
    neighbors: Vec<SmallVec<[usize; 16]>>,
}

impl HnswIndex {
    pub fn new(m: usize, ef_construction: usize) -> Self {
        Self {
            m: m.max(1),
            ef_construction: ef_construction.max(1),
            vectors: Vec::new(),
            neighbors: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Flat linear scan over all stored vectors. Fast for small centroid counts.
    #[inline(always)]
    pub fn flat_search_with_scratch(
        &self,
        query: &[u8],
        top_k: usize,
        scratch: &mut Vec<(f32, usize)>,
    ) -> Vec<usize> {
        scratch.clear();
        if scratch.capacity() < self.vectors.len() {
            scratch.reserve(self.vectors.len() - scratch.capacity());
        }
        if self.vectors.is_empty() {
            return Vec::new();
        }

        for (id, v) in self.vectors.iter().enumerate() {
            if v.is_empty() || v.len() != query.len() {
                continue;
            }
            scratch.push((l2_bytes(query, v), id));
        }

        let cmp = |a: &(f32, usize), b: &(f32, usize)| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal);
        if scratch.len() <= top_k {
            scratch.sort_by(cmp);
        } else {
            let kth = top_k.min(scratch.len());
            scratch.select_nth_unstable_by(kth - 1, cmp);
            scratch[..kth].sort_by(cmp);
            scratch.truncate(kth);
        }

        scratch.iter().take(top_k).map(|(_, id)| *id).collect()
    }

    /// Search using reusable scratch buffers to cut per-query allocations.
    #[inline(always)]
    pub fn search_with_scratch(
        &self,
        query: &[u8],
        top_k: usize,
        ef_search: usize,
        scratch: &mut SearchScratch,
    ) -> Vec<usize> {
        scratch.reset(self.vectors.len(), ef_search);
        if self.vectors.is_empty() {
            return Vec::new();
        }

        let entry = 0usize;
        let visited = &mut scratch.visited;
        let epoch = scratch.epoch;
        let candidates = &mut scratch.candidates;
        let results = &mut scratch.results;

        let dist0 = l2_bytes(query, &self.vectors[entry]);
        candidates.push(HeapItem::new(entry, dist0));
        results.push(HeapItem::new(entry, dist0));
        visited[entry] = epoch;

        while let Some(cur) = candidates.pop() {
            let worst = results.peek().map(|h| h.dist).unwrap_or(f32::MAX);
            if cur.dist > worst && results.len() >= ef_search {
                break;
            }

            for &nbr in &self.neighbors[cur.id] {
                if visited.get(nbr).copied().unwrap_or(0) == epoch {
                    continue;
                }
                visited[nbr] = epoch;
                let d = l2_bytes(query, &self.vectors[nbr]);
                let item = HeapItem::new(nbr, d);
                if results.len() < ef_search {
                    results.push(item.clone());
                    candidates.push(item);
                } else if d < worst {
                    results.pop();
                    results.push(item.clone());
                    candidates.push(item);
                }
            }
        }

        let mut res: Vec<_> = results.drain().collect();
        res.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
        res.truncate(top_k);
        res.into_iter().map(|h| h.id).collect()
    }

    pub fn insert(&mut self, id: usize, vector: Vec<u8>) {
        // Ensure capacity for provided id.
        if id >= self.vectors.len() {
            let to_add = id + 1 - self.vectors.len();
            self.vectors.reserve(to_add);
            self.neighbors.reserve(to_add);
            while self.vectors.len() <= id {
                self.vectors.push(Vec::new());
                self.neighbors.push(SmallVec::new());
            }
        }

        self.vectors[id] = vector;

        // First element short-circuits (entrypoint).
        if id == 0 {
            return;
        }

        // Find candidates among existing nodes.
        let mut dists: Vec<(f32, usize)> = self
            .vectors
            .iter()
            .enumerate()
            .filter(|(i, v)| *i != id && !v.is_empty())
            .map(|(i, v)| (l2_bytes(v, &self.vectors[id]), i))
            .collect();

        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        dists.truncate(self.ef_construction);

        // Connect to up to m closest.
        for &(_, nid) in dists.iter().take(self.m) {
            self.neighbors[nid].push(id);
            self.neighbors[id].push(nid);
        }

        self.trim_neighbors(id);
        for &(_, nid) in dists.iter().take(self.m) {
            self.trim_neighbors(nid);
        }
    }

    pub fn search(&self, query: &[u8], top_k: usize, ef_search: usize) -> Vec<usize> {
        // Fallback that allocates fresh scratch for callers that don't reuse buffers.
        let mut scratch = SearchScratch::new();
        self.search_with_scratch(query, top_k, ef_search, &mut scratch)
    }

    fn trim_neighbors(&mut self, id: usize) {
        if self.neighbors[id].len() <= self.m {
            return;
        }
        let mut neigh = std::mem::take(&mut self.neighbors[id]);
        neigh.sort_by(|&a, &b| {
            let da = l2_bytes(&self.vectors[id], &self.vectors[a]);
            let db = l2_bytes(&self.vectors[id], &self.vectors[b]);
            da.partial_cmp(&db).unwrap_or(Ordering::Equal)
        });
        neigh.truncate(self.m);
        self.neighbors[id] = neigh;
    }
}

#[derive(Clone)]
struct HeapItem {
    id: usize,
    dist: f32,
}

impl HeapItem {
    fn new(id: usize, dist: f32) -> Self {
        Self { id, dist }
    }
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}
impl Eq for HeapItem {}

// Reverse ordering for min-heap behavior using BinaryHeap.
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.dist.partial_cmp(&self.dist)
    }
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

#[inline(always)]
fn l2_bytes(a: &[u8], b: &[u8]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        // Prefer AVX-512 when available, then AVX2, else scalar.
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512dq")
        {
            unsafe { l2_bytes_avx512(a, b) }
        } else if std::arch::is_x86_feature_detected!("avx2") {
            unsafe { l2_bytes_avx2(a, b) }
        } else {
            l2_bytes_scalar(a, b)
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        l2_bytes_scalar(a, b)
    }
}

#[inline(always)]
fn l2_bytes_scalar(a: &[u8], b: &[u8]) -> f32 {
    let len = a.len().min(b.len());
    let mut sum = 0f32;
    let mut i = 0;
    while i + 4 <= len {
        let d0 = a[i] as f32 - b[i] as f32;
        let d1 = a[i + 1] as f32 - b[i + 1] as f32;
        let d2 = a[i + 2] as f32 - b[i + 2] as f32;
        let d3 = a[i + 3] as f32 - b[i + 3] as f32;
        sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
        i += 4;
    }
    while i < len {
        let d = a[i] as f32 - b[i] as f32;
        sum += d * d;
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn l2_bytes_avx2(a: &[u8], b: &[u8]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len().min(b.len());
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    // Process 32 bytes per iteration.
    while i + 32 <= len {
        let va = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
        let vb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);

        // Unpack to 16-bit lanes and process all bytes.
        let va16_lo = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<0>(va));
        let vb16_lo = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<0>(vb));
        let va16_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(va));
        let vb16_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(vb));

        // Lower 8 lanes of lower half.
        let va_lo_lo = _mm256_castsi256_si128(va16_lo);
        let vb_lo_lo = _mm256_castsi256_si128(vb16_lo);
        let va_lo_lo_ps = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(va_lo_lo));
        let vb_lo_lo_ps = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(vb_lo_lo));
        let diff_lo_lo = _mm256_sub_ps(va_lo_lo_ps, vb_lo_lo_ps);
        acc = _mm256_fmadd_ps(diff_lo_lo, diff_lo_lo, acc);

        // Upper 8 lanes of lower half.
        let va_lo_hi = _mm256_extracti128_si256::<1>(va16_lo);
        let vb_lo_hi = _mm256_extracti128_si256::<1>(vb16_lo);
        let va_lo_hi_ps = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(va_lo_hi));
        let vb_lo_hi_ps = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(vb_lo_hi));
        let diff_lo_hi = _mm256_sub_ps(va_lo_hi_ps, vb_lo_hi_ps);
        acc = _mm256_fmadd_ps(diff_lo_hi, diff_lo_hi, acc);

        // Lower 8 lanes of upper half.
        let va_hi_lo = _mm256_castsi256_si128(va16_hi);
        let vb_hi_lo = _mm256_castsi256_si128(vb16_hi);
        let va_hi_lo_ps = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(va_hi_lo));
        let vb_hi_lo_ps = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(vb_hi_lo));
        let diff_hi_lo = _mm256_sub_ps(va_hi_lo_ps, vb_hi_lo_ps);
        acc = _mm256_fmadd_ps(diff_hi_lo, diff_hi_lo, acc);

        // Upper 8 lanes of upper half.
        let va_hi_hi = _mm256_extracti128_si256::<1>(va16_hi);
        let vb_hi_hi = _mm256_extracti128_si256::<1>(vb16_hi);
        let va_hi_hi_ps = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(va_hi_hi));
        let vb_hi_hi_ps = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(vb_hi_hi));
        let diff_hi_hi = _mm256_sub_ps(va_hi_hi_ps, vb_hi_hi_ps);
        acc = _mm256_fmadd_ps(diff_hi_hi, diff_hi_hi, acc);

        i += 32;
    }

    let mut tmp = [0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
    let mut total = tmp.iter().sum::<f32>();

    // Tail process scalars.
    while i < len {
        let d = a[i] as f32 - b[i] as f32;
        total += d * d;
        i += 1;
    }

    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512dq", enable = "fma")]
unsafe fn l2_bytes_avx512(a: &[u8], b: &[u8]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len().min(b.len());
    let mut acc = _mm512_setzero_ps();
    let mut i = 0;
    // Process 64 bytes per iteration.
    while i + 64 <= len {
        let va = _mm512_loadu_si512(a.as_ptr().add(i) as *const __m512i);
        let vb = _mm512_loadu_si512(b.as_ptr().add(i) as *const __m512i);

        // Lower 32 bytes.
        let va_lo_bytes = _mm512_castsi512_si256(va);
        let vb_lo_bytes = _mm512_castsi512_si256(vb);
        let va_lo_16 = _mm512_cvtepu8_epi16(va_lo_bytes);
        let vb_lo_16 = _mm512_cvtepu8_epi16(vb_lo_bytes);

        // lower 16 lanes of lower half
        let va_lo16_lo = _mm512_castsi512_si256(va_lo_16);
        let vb_lo16_lo = _mm512_castsi512_si256(vb_lo_16);
        let va_lo_lo_ps = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(va_lo16_lo));
        let vb_lo_lo_ps = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vb_lo16_lo));
        let diff_lo_lo = _mm512_sub_ps(va_lo_lo_ps, vb_lo_lo_ps);
        acc = _mm512_fmadd_ps(diff_lo_lo, diff_lo_lo, acc);

        // upper 16 lanes of lower half
        let va_lo16_hi = _mm512_extracti64x4_epi64::<1>(va_lo_16);
        let vb_lo16_hi = _mm512_extracti64x4_epi64::<1>(vb_lo_16);
        let va_lo_hi_ps = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(va_lo16_hi));
        let vb_lo_hi_ps = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vb_lo16_hi));
        let diff_lo_hi = _mm512_sub_ps(va_lo_hi_ps, vb_lo_hi_ps);
        acc = _mm512_fmadd_ps(diff_lo_hi, diff_lo_hi, acc);

        // Upper 32 bytes.
        let va_hi_bytes = _mm512_extracti64x4_epi64::<1>(va);
        let vb_hi_bytes = _mm512_extracti64x4_epi64::<1>(vb);
        let va_hi_16 = _mm512_cvtepu8_epi16(va_hi_bytes);
        let vb_hi_16 = _mm512_cvtepu8_epi16(vb_hi_bytes);

        let va_hi16_lo = _mm512_castsi512_si256(va_hi_16);
        let vb_hi16_lo = _mm512_castsi512_si256(vb_hi_16);
        let va_hi_lo_ps = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(va_hi16_lo));
        let vb_hi_lo_ps = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vb_hi16_lo));
        let diff_hi_lo = _mm512_sub_ps(va_hi_lo_ps, vb_hi_lo_ps);
        acc = _mm512_fmadd_ps(diff_hi_lo, diff_hi_lo, acc);

        let va_hi16_hi = _mm512_extracti64x4_epi64::<1>(va_hi_16);
        let vb_hi16_hi = _mm512_extracti64x4_epi64::<1>(vb_hi_16);
        let va_hi_hi_ps = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(va_hi16_hi));
        let vb_hi_hi_ps = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vb_hi16_hi));
        let diff_hi_hi = _mm512_sub_ps(va_hi_hi_ps, vb_hi_hi_ps);
        acc = _mm512_fmadd_ps(diff_hi_hi, diff_hi_hi, acc);

        i += 64;
    }

    let mut tmp = [0f32; 16];
    _mm512_storeu_ps(tmp.as_mut_ptr(), acc);
    let mut total: f32 = tmp.iter().sum();

    // Tail
    while i < len {
        let d = a[i] as f32 - b[i] as f32;
        total += d * d;
        i += 1;
    }

    total
}

pub struct SearchScratch {
    visited: Vec<u32>,
    epoch: u32,
    candidates: BinaryHeap<HeapItem>,
    results: BinaryHeap<HeapItem>,
}

impl SearchScratch {
    pub fn new() -> Self {
        Self {
            visited: Vec::new(),
            epoch: 1,
            candidates: BinaryHeap::new(),
            results: BinaryHeap::new(),
        }
    }

    fn reset(&mut self, n: usize, reserve: usize) {
        // Bump epoch instead of clearing the whole visited array.
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            // Wrapped; reset the whole array.
            self.epoch = 1;
            self.visited.iter_mut().for_each(|v| *v = 0);
        }
        if self.visited.len() < n {
            self.visited.resize(n, 0);
        }
        // Pre-reserve heaps to avoid per-query growth.
        if self.candidates.capacity() < reserve {
            self.candidates
                .reserve(reserve.saturating_sub(self.candidates.capacity()));
        }
        if self.results.capacity() < reserve {
            self.results
                .reserve(reserve.saturating_sub(self.results.capacity()));
        }
        self.candidates.clear();
        self.results.clear();
    }
}
