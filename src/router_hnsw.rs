#[cfg(feature = "packed_simd_2")]
use packed_simd_2::f32x16;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Minimal hierarchical HNSW for routing centroids.
/// Stores byte-quantized vectors and keeps up to `m0` neighbors on level 0 and
/// `m` neighbors on upper layers. Distances are cosine over quantized bytes;
/// we cache 1/norm per vector to avoid repeated normalization in the hot path.
pub struct HnswIndex {
    m: usize,
    m0: usize,
    ef_construction: usize,
    dim: usize,
    pad_dim: usize,
    vectors: Vec<Vec<u8>>, // original quantized bytes (kept for compatibility)
    traversal: Vec<Vec<i8>>, // centered i8 copy for fast dot/cosine
    inv_norms: Vec<f32>,   // inv-norm for traversal (i8)
    level_of: Vec<i32>,
    neighbors: Vec<Vec<SmallVec<[usize; 16]>>>,
    entry: Option<usize>,
    max_level: i32,
    rng: StdRng,
    insert_scratch: SearchScratch,
}

impl HnswIndex {
    pub fn new(m: usize, ef_construction: usize) -> Self {
        Self {
            m: m.max(1),
            m0: (m * 2).max(2),
            ef_construction: ef_construction.max(1),
            dim: 0,
            pad_dim: 0,
            vectors: Vec::new(),
            traversal: Vec::new(),
            inv_norms: Vec::new(),
            level_of: Vec::new(),
            neighbors: Vec::new(),
            entry: None,
            max_level: -1,
            rng: StdRng::seed_from_u64(0xBAD5_EED),
            insert_scratch: SearchScratch::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn pad_dim(&self) -> usize {
        self.pad_dim
    }

    /// Flat linear scan over all stored vectors. Fast for small centroid counts.
    #[inline(always)]
    pub fn flat_search_with_scratch(
        &self,
        query: &[u8],
        top_k: usize,
        scratch: &mut Vec<(f32, usize)>,
    ) -> Vec<usize> {
        if self.dim == 0 || query.len() != self.pad_dim {
            return Vec::new();
        }
        let mut q_i8 = Vec::with_capacity(query.len());
        center_bytes_to_i8(query, &mut q_i8);
        let q_inv = inv_norm_i8(&q_i8);
        scratch.clear();
        if scratch.capacity() < self.vectors.len() {
            scratch.reserve(self.vectors.len() - scratch.capacity());
        }
        if self.vectors.is_empty() {
            return Vec::new();
        }

        for (id, v) in self.traversal.iter().enumerate() {
            if v.is_empty() || v.len() != q_i8.len() {
                continue;
            }
            let d = cosine_i8(&q_i8, q_inv, v, self.inv_norms[id]);
            scratch.push((d, id));
        }

        let cmp =
            |a: &(f32, usize), b: &(f32, usize)| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal);
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
        if self.entry.is_none() || self.dim == 0 || query.len() != self.pad_dim {
            return Vec::new();
        }
        let mut entry = self.entry.unwrap();
        let max_level = self.max_level;
        let mut q_buf = std::mem::take(&mut scratch.q_buf);
        center_bytes_to_i8(query, &mut q_buf);
        let q_inv = inv_norm_i8(&q_buf);
        let q_slice: &[i8] = &q_buf;

        // Greedy descent on upper layers.
        if max_level >= 1 {
            for level in (1..=max_level as usize).rev() {
                entry = self.greedy_search_layer(q_slice, q_inv, entry, level as i32);
            }
        }

        // Layer 0: best-first search with a wider ef_search beam.
        let results = self.layer_search(q_slice, q_inv, entry, 0, ef_search, scratch);
        let mut res: Vec<_> = results.into_iter().take(top_k).collect();
        res.truncate(top_k);
        // Return q_buf to scratch for reuse.
        scratch.q_buf = q_buf;
        res.into_iter().map(|h| h.id).collect()
    }

    pub fn search(&self, query: &[u8], top_k: usize, ef_search: usize) -> Vec<usize> {
        // Fallback that allocates fresh scratch for callers that don't reuse buffers.
        let mut scratch = SearchScratch::new();
        self.search_with_scratch(query, top_k, ef_search, &mut scratch)
    }

    pub fn insert(&mut self, id: usize, vector: Vec<u8>) {
        if self.dim == 0 {
            self.dim = vector.len();
            self.pad_dim = round_up_32(self.dim);
        } else if vector.len() != self.dim {
            // Mismatched dimensions; skip insert.
            return;
        }
        let mut padded = vector;
        if padded.len() < self.pad_dim {
            padded.resize(self.pad_dim, 0);
        }
        let level = self.sample_level();
        self.ensure_node(id, level);
        self.vectors[id] = padded.clone();
        // Centered i8 traversal copy.
        let mut trav: Vec<i8> = Vec::with_capacity(self.pad_dim);
        trav.resize(self.pad_dim, 0);
        for i in 0..self.pad_dim {
            let b = padded[i] as i16 - 128;
            trav[i] = b.clamp(-128, 127) as i8;
        }
        let inv = inv_norm_i8(&trav);
        self.traversal[id] = trav;
        self.inv_norms[id] = inv;

        // First element short-circuits.
        if self.entry.is_none() {
            self.entry = Some(id);
            self.max_level = level;
            return;
        }

        let mut entry = self.entry.unwrap();
        let prev_max = self.max_level;
        if level > self.max_level {
            self.max_level = level;
            self.entry = Some(id);
        }

        // Greedy descent until one level above the node's level.
        if prev_max > level {
            for l in ((level + 1) as usize..=prev_max as usize).rev() {
                entry = self.greedy_search_layer(&self.traversal[id], inv, entry, l as i32);
            }
        }

        // Connect on all levels down to 0.
        for l in (0..=level as usize).rev() {
            let level_i32 = l as i32;
            // Slightly wider beam on level 0; keep upper layers modest to save CPU.
            let beam = if level_i32 == 0 {
                self.m_for_level(level_i32) * 3
            } else {
                self.m_for_level(level_i32) * 2
            };
            let ef = self.ef_construction.max(beam);
            // Reuse a shared scratch for inserts; move it out to satisfy the borrow checker.
            let mut scratch = std::mem::take(&mut self.insert_scratch);
            let mut candidates =
                self.layer_search(&self.traversal[id], inv, entry, level_i32, ef, &mut scratch);
            self.insert_scratch = scratch;
            let mut selected =
                self.select_neighbors(&mut candidates, self.m_for_level(level_i32), true);
            // Level 0 aggressive fallback: if too sparse, retry without diversity to fill degree.
            if level_i32 == 0 && selected.len() < self.m_for_level(level_i32) / 2 {
                selected =
                    self.select_neighbors(&mut candidates, self.m_for_level(level_i32), false);
            }
            for &nbr in &selected {
                self.connect_nodes(id, nbr as usize, level_i32);
            }
            if let Some(&closest) = selected.first() {
                entry = closest as usize;
            }
        }
    }

    fn greedy_search_layer(&self, query: &[i8], q_inv: f32, entry: usize, level: i32) -> usize {
        let mut best = entry;
        let mut best_dist = cosine_i8(query, q_inv, &self.traversal[best], self.inv_norms[best]);
        loop {
            let mut improved = false;
            for &nbr in self.neighbors_at(best, level) {
                let nbr_usize = nbr as usize;
                prefetch_vector_bytes(&self.vectors[nbr_usize]);
                let d = cosine_i8(
                    query,
                    q_inv,
                    &self.traversal[nbr_usize],
                    self.inv_norms[nbr_usize],
                );
                if d < best_dist {
                    best_dist = d;
                    best = nbr_usize;
                    improved = true;
                }
            }
            if !improved {
                break;
            }
        }
        best
    }

    fn layer_search(
        &self,
        query: &[i8],
        q_inv: f32,
        entry: usize,
        level: i32,
        ef: usize,
        scratch: &mut SearchScratch,
    ) -> Vec<HeapItem> {
        scratch.reset(self.vectors.len(), ef);
        let visited = &mut scratch.visited;
        let epoch = scratch.epoch;
        let candidates = &mut scratch.candidates;
        let results = &mut scratch.results;

        let dist0 = cosine_i8(query, q_inv, &self.traversal[entry], self.inv_norms[entry]);
        candidates.push(HeapItem::new(entry, dist0));
        results.push(HeapItem::new(entry, dist0));
        visited[entry] = epoch;

        while let Some(cur) = candidates.pop() {
            let worst = results.peek().map(|h| h.dist).unwrap_or(f32::MAX);
            if cur.dist > worst && results.len() >= ef {
                break;
            }

            for &nbr in self.neighbors_at(cur.id, level) {
                if visited.get(nbr).copied().unwrap_or(0) == epoch {
                    continue;
                }
                visited[nbr] = epoch;
                prefetch_vector_bytes(&self.vectors[nbr]);
                let d = cosine_i8(query, q_inv, &self.traversal[nbr], self.inv_norms[nbr]);
                let item = HeapItem::new(nbr, d);
                if results.len() < ef {
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
        res
    }

    fn select_neighbors(
        &self,
        candidates: &mut [HeapItem],
        m: usize,
        diversity: bool,
    ) -> Vec<usize> {
        // Redis-inspired heuristic: keep a diverse set; if we can't fill, fall back to closest.
        candidates.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
        let mut accepted: Vec<usize> = Vec::with_capacity(m);

        for cand in candidates.iter() {
            if accepted.len() >= m {
                break;
            }
            let mut keep = true;
            if diversity {
                // Diversity check: accept if no already-accepted node is closer to cand than new node is.
                for &acc in accepted.iter() {
                    let d_acc = cosine_i8(
                        &self.traversal[cand.id],
                        self.inv_norms[cand.id],
                        &self.traversal[acc],
                        self.inv_norms[acc],
                    );
                    if d_acc < cand.dist {
                        keep = false;
                        break;
                    }
                }
            }
            if keep {
                accepted.push(cand.id);
            }
        }

        // If we didn't fill, top up with closest remaining.
        if accepted.len() < m {
            for cand in candidates.iter() {
                if accepted.len() >= m {
                    break;
                }
                // Linear contains is fine here; candidate list is small (ef scale).
                if !accepted.contains(&cand.id) {
                    accepted.push(cand.id);
                }
            }
        }

        accepted
    }

    fn connect_nodes(&mut self, a: usize, b: usize, level: i32) {
        let inserted_ab = self.add_edge(a, b, level);
        let inserted_ba = self.add_edge(b, a, level);
        // Keep symmetry: if only one side succeeded, drop the other side.
        if inserted_ab && !inserted_ba {
            self.remove_edge(a, b, level);
        } else if inserted_ba && !inserted_ab {
            self.remove_edge(b, a, level);
        }
    }

    fn add_edge(&mut self, src: usize, dst: usize, level: i32) -> bool {
        self.ensure_level(src, level);
        let cap = self.m_for_level(level);
        let dist_new = cosine_i8(
            &self.traversal[src],
            self.inv_norms[src],
            &self.traversal[dst],
            self.inv_norms[dst],
        );
        // Immutable view to check duplicates and compute worst.
        {
            let list = &self.neighbors[src][level as usize];
            if list.contains(&dst) {
                return true;
            }
            if list.len() < cap {
                // will push below
            } else {
                // Determine worst without mutating.
                let mut worst_idx = 0usize;
                let mut worst_dist = dist_new;
                for (i, &nid) in list.iter().enumerate() {
                    let d = cosine_i8(
                        &self.traversal[src],
                        self.inv_norms[src],
                        &self.traversal[nid],
                        self.inv_norms[nid],
                    );
                    if i == 0 || d > worst_dist {
                        worst_idx = i;
                        worst_dist = d;
                    }
                }
                if dist_new >= worst_dist {
                    return false;
                }
                let victim = list[worst_idx];
                let victim_deg = self
                    .neighbors
                    .get(victim)
                    .and_then(|levels| levels.get(level as usize))
                    .map(|v| v.len())
                    .unwrap_or(0);
                if victim_deg <= (cap / 4).max(1) {
                    return false;
                }
                let list_mut = &mut self.neighbors[src][level as usize];
                list_mut[worst_idx] = dst;
                let _ = list_mut;
                self.remove_edge(victim, src, level);
                return true;
            }
        }
        // push path
        let list = &mut self.neighbors[src][level as usize];
        list.push(dst);
        true
    }

    fn remove_edge(&mut self, src: usize, dst: usize, level: i32) {
        if let Some(levels) = self.neighbors.get_mut(src) {
            if let Some(list) = levels.get_mut(level as usize) {
                if let Some(pos) = list.iter().position(|&x| x == dst) {
                    list.swap_remove(pos);
                }
            }
        }
    }

    fn trim_neighbors(&mut self, id: usize, level: i32) {
        let cap = self.m_for_level(level);
        let list = &mut self.neighbors[id][level as usize];
        if list.len() <= cap {
            return;
        }
        let vec = std::mem::take(list);
        let mut vec: SmallVec<[usize; 16]> = vec;
        vec.sort_by(|&a, &b| {
            let da = cosine_i8(
                &self.traversal[id],
                self.inv_norms[id],
                &self.traversal[a],
                self.inv_norms[a],
            );
            let db = cosine_i8(
                &self.traversal[id],
                self.inv_norms[id],
                &self.traversal[b],
                self.inv_norms[b],
            );
            da.partial_cmp(&db).unwrap_or(Ordering::Equal)
        });
        vec.truncate(cap);
        *list = vec;
    }

    fn neighbors_at(&self, id: usize, level: i32) -> &[usize] {
        self.neighbors
            .get(id)
            .and_then(|levels| levels.get(level as usize))
            .map(|s| s.as_slice())
            .unwrap_or(&[])
    }

    fn ensure_node(&mut self, id: usize, level: i32) {
        if id >= self.vectors.len() {
            let missing = id + 1 - self.vectors.len();
            self.vectors.reserve(missing);
            self.traversal.reserve(missing);
            self.inv_norms.reserve(missing);
            self.level_of.reserve(missing);
            self.neighbors.reserve(missing);
            while self.vectors.len() <= id {
                self.vectors.push(Vec::new());
                self.traversal.push(Vec::new());
                self.inv_norms.push(0.0);
                self.level_of.push(-1);
                self.neighbors.push(Vec::new());
            }
        }
        self.level_of[id] = level;
        self.ensure_level(id, level);
    }

    fn ensure_level(&mut self, id: usize, level: i32) {
        let target = (level + 1) as usize;
        if self.neighbors[id].len() < target {
            self.neighbors[id].resize_with(target, SmallVec::new);
        }
    }

    fn m_for_level(&self, level: i32) -> usize {
        if level == 0 {
            self.m0
        } else {
            self.m
        }
    }

    fn sample_level(&mut self) -> i32 {
        // Redis-like: geometric with p=0.25 and capped height to keep search shallow.
        const P: f32 = 0.25;
        // Lower cap to reduce traversal work while keeping hierarchy depth reasonable.
        const MAX_LEVEL: i32 = 12;
        let mut level = 0;
        while self.rng.gen::<f32>() < P && level < MAX_LEVEL {
            level += 1;
        }
        level
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
fn cosine_i8(a: &[i8], a_inv_norm: f32, b: &[i8], b_inv_norm: f32) -> f32 {
    if a_inv_norm == 0.0 || b_inv_norm == 0.0 {
        return 1.0;
    }
    #[cfg(target_arch = "x86_64")]
    {
        // Prefer AVX-512 when available, then AVX2, else scalar.
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512dq")
        {
            let dot = unsafe { dot_i8_avx512(a, b) };
            return 1.0 - dot * a_inv_norm * b_inv_norm;
        } else if std::arch::is_x86_feature_detected!("avx2") {
            let dot = unsafe { dot_i8_avx2(a, b) };
            return 1.0 - dot * a_inv_norm * b_inv_norm;
        } else {
            let dot = dot_i8_scalar(a, b);
            return 1.0 - dot * a_inv_norm * b_inv_norm;
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let dot = dot_i8_scalar(a, b);
        return 1.0 - dot * a_inv_norm * b_inv_norm;
    }
}

#[inline(always)]
fn dot_i8_scalar(a: &[i8], b: &[i8]) -> f32 {
    let len = a.len().min(b.len());
    let mut sum0 = 0f32;
    let mut sum1 = 0f32;
    let mut i = 0;
    while i + 8 <= len {
        sum0 += (a[i] as f32) * (b[i] as f32);
        sum0 += (a[i + 1] as f32) * (b[i + 1] as f32);
        sum0 += (a[i + 2] as f32) * (b[i + 2] as f32);
        sum0 += (a[i + 3] as f32) * (b[i + 3] as f32);
        sum1 += (a[i + 4] as f32) * (b[i + 4] as f32);
        sum1 += (a[i + 5] as f32) * (b[i + 5] as f32);
        sum1 += (a[i + 6] as f32) * (b[i + 6] as f32);
        sum1 += (a[i + 7] as f32) * (b[i + 7] as f32);
        i += 8;
    }
    while i < len {
        sum0 += (a[i] as f32) * (b[i] as f32);
        i += 1;
    }
    sum0 + sum1
}

#[inline(always)]
fn inv_norm_i8(v: &[i8]) -> f32 {
    let mut sum = 0f32;
    let mut i = 0;
    let len = v.len();
    while i + 8 <= len {
        sum += (v[i] as f32) * (v[i] as f32);
        sum += (v[i + 1] as f32) * (v[i + 1] as f32);
        sum += (v[i + 2] as f32) * (v[i + 2] as f32);
        sum += (v[i + 3] as f32) * (v[i + 3] as f32);
        sum += (v[i + 4] as f32) * (v[i + 4] as f32);
        sum += (v[i + 5] as f32) * (v[i + 5] as f32);
        sum += (v[i + 6] as f32) * (v[i + 6] as f32);
        sum += (v[i + 7] as f32) * (v[i + 7] as f32);
        i += 8;
    }
    while i < len {
        sum += (v[i] as f32) * (v[i] as f32);
        i += 1;
    }
    if sum <= 0.0 {
        0.0
    } else {
        1.0 / sum.sqrt()
    }
}

#[inline(always)]
fn round_up_32(v: usize) -> usize {
    (v + 31) & !31
}

#[inline(always)]
fn prefetch_vector_bytes(v: &[u8]) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::_mm_prefetch;
        const HINT: i32 = std::arch::x86_64::_MM_HINT_T0;
        let ptr = v.as_ptr();
        // Prefetch first cache line; vectors are small-ish (centroids).
        _mm_prefetch(ptr as *const i8, HINT);
        if v.len() > 64 {
            _mm_prefetch(ptr.add(64) as *const i8, HINT);
        }
    }
}

#[inline(always)]
fn center_bytes_to_i8(src: &[u8], dst: &mut Vec<i8>) {
    dst.clear();
    dst.resize(src.len(), 0);
    for i in 0..src.len() {
        let v = src[i] as i16 - 128;
        dst[i] = v.clamp(-128, 127) as i8;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_i8_avx2(a: &[i8], b: &[i8]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len().min(b.len());
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    // Process 32 bytes per iteration.
    while i + 32 <= len {
        let va = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
        let vb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);

        // Unpack to 16-bit lanes and process all bytes.
        let va16_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<0>(va));
        let vb16_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<0>(vb));
        let va16_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(va));
        let vb16_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(vb));

        // Lower 8 lanes of lower half.
        let va_lo_lo = _mm256_castsi256_si128(va16_lo);
        let vb_lo_lo = _mm256_castsi256_si128(vb16_lo);
        let va_lo_lo_ps = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(va_lo_lo));
        let vb_lo_lo_ps = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(vb_lo_lo));
        acc = _mm256_fmadd_ps(va_lo_lo_ps, vb_lo_lo_ps, acc);

        // Upper 8 lanes of lower half.
        let va_lo_hi = _mm256_extracti128_si256::<1>(va16_lo);
        let vb_lo_hi = _mm256_extracti128_si256::<1>(vb16_lo);
        let va_lo_hi_ps = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(va_lo_hi));
        let vb_lo_hi_ps = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(vb_lo_hi));
        acc = _mm256_fmadd_ps(va_lo_hi_ps, vb_lo_hi_ps, acc);

        // Lower 8 lanes of upper half.
        let va_hi_lo = _mm256_castsi256_si128(va16_hi);
        let vb_hi_lo = _mm256_castsi256_si128(vb16_hi);
        let va_hi_lo_ps = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(va_hi_lo));
        let vb_hi_lo_ps = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(vb_hi_lo));
        acc = _mm256_fmadd_ps(va_hi_lo_ps, vb_hi_lo_ps, acc);

        // Upper 8 lanes of upper half.
        let va_hi_hi = _mm256_extracti128_si256::<1>(va16_hi);
        let vb_hi_hi = _mm256_extracti128_si256::<1>(vb16_hi);
        let va_hi_hi_ps = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(va_hi_hi));
        let vb_hi_hi_ps = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(vb_hi_hi));
        acc = _mm256_fmadd_ps(va_hi_hi_ps, vb_hi_hi_ps, acc);

        i += 32;
    }

    let mut tmp = [0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
    let mut total = tmp.iter().sum::<f32>();

    // Tail process scalars.
    while i < len {
        total += (a[i] as f32) * (b[i] as f32);
        i += 1;
    }

    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(
    enable = "avx512f",
    enable = "avx512bw",
    enable = "avx512dq",
    enable = "fma"
)]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn dot_i8_avx512(a: &[i8], b: &[i8]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len().min(b.len());
    let mut i = 0usize;

    // 16 lanes of i32
    let mut acc32 = _mm512_setzero_si512();

    while i + 64 <= len {
        // load 2x32 bytes (because cvtepi8_epi16 takes __m256i)
        let a0 = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
        let b0 = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
        let a1 = _mm256_loadu_si256(a.as_ptr().add(i + 32) as *const __m256i);
        let b1 = _mm256_loadu_si256(b.as_ptr().add(i + 32) as *const __m256i);

        let a0_16 = _mm512_cvtepi8_epi16(a0);
        let b0_16 = _mm512_cvtepi8_epi16(b0);
        let a1_16 = _mm512_cvtepi8_epi16(a1);
        let b1_16 = _mm512_cvtepi8_epi16(b1);

        // (a0*b0)[0]+(a0*b0)[1] etc -> 16x i32 partial sums
        let p0 = _mm512_madd_epi16(a0_16, b0_16);
        let p1 = _mm512_madd_epi16(a1_16, b1_16);

        acc32 = _mm512_add_epi32(acc32, p0);
        acc32 = _mm512_add_epi32(acc32, p1);

        i += 64;
    }

    // horizontal reduce i32
    let mut tmp = [0i32; 16];
    _mm512_storeu_si512(tmp.as_mut_ptr() as *mut __m512i, acc32);
    let mut total: i32 = tmp.iter().sum();

    while i < len {
        total += (a[i] as i32) * (b[i] as i32);
        i += 1;
    }

    total as f32
}


#[derive(Default)]
pub struct SearchScratch {
    visited: Vec<u32>,
    epoch: u32,
    candidates: BinaryHeap<HeapItem>,
    results: BinaryHeap<HeapItem>,
    q_buf: Vec<i8>,
}

impl SearchScratch {
    pub fn new() -> Self {
        Self {
            visited: Vec::new(),
            epoch: 1,
            candidates: BinaryHeap::new(),
            results: BinaryHeap::new(),
            q_buf: Vec::new(),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn pad_query(prefix: &[u8], len: usize) -> Vec<u8> {
        let mut q = vec![0u8; len];
        q[..prefix.len()].copy_from_slice(prefix);
        q
    }

    #[test]
    fn round_up_and_pad_dim() {
        assert_eq!(round_up_32(1), 32);
        assert_eq!(round_up_32(33), 64);
    }

    #[test]
    fn centers_bytes_to_i8() {
        let src = vec![0u8, 128, 255];
        let mut dst = Vec::new();
        center_bytes_to_i8(&src, &mut dst);
        assert_eq!(dst, vec![-128, 0, 127]);
    }

    #[test]
    fn search_prefers_closer_vector() {
        let mut hnsw = HnswIndex::new(8, 32);
        let a = vec![10u8, 10, 10, 10];
        let b = vec![200u8, 200, 200, 200];
        hnsw.insert(0, a.clone());
        hnsw.insert(1, b.clone());
        assert_eq!(hnsw.len(), 2);
        assert_eq!(hnsw.pad_dim(), 32);

        let query = pad_query(&b, hnsw.pad_dim());
        let res = hnsw.search(&query, 1, 64);
        assert_eq!(res, vec![1], "expected the closer centroid to win");

        let mut scratch = Vec::new();
        let flat = hnsw.flat_search_with_scratch(&query, 2, &mut scratch);
        assert_eq!(flat[0], 1);
    }

    #[test]
    fn search_returns_empty_on_dim_mismatch() {
        let mut hnsw = HnswIndex::new(4, 8);
        hnsw.insert(0, vec![1u8, 2, 3, 4]);
        // Query is too short, so search should bail out.
        let res = hnsw.search(&[0u8; 4], 1, 8);
        assert!(res.is_empty());
    }
}
