use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

type DotFn = unsafe fn(&[i8], &[i8]) -> i32;
type CenterFn = unsafe fn(src: *const u8, dst: *mut i8, len: usize);

/// Minimal hierarchical HNSW for routing centroids.
/// Stores centered i8 vectors in a contiguous slab and caches 1/norm per vector.
/// Distances are cosine over centered i8: d = 1 - dot(a,b) * invnorm(a) * invnorm(b).
pub struct HnswIndex {
    m: usize,
    m0: usize,
    ef_construction: usize,
    dim: usize,
    pad_dim: usize,

    // Contiguous storage: traversal[id * pad_dim .. (id+1)*pad_dim]
    traversal: Vec<i8>,
    inv_norms: Vec<f32>,
    level_of: Vec<i32>,
    neighbors: Vec<Vec<SmallVec<[usize; 16]>>>,

    entry: Option<usize>,
    max_level: i32,
    rng: StdRng,

    // Dispatch chosen once per index to avoid per-call CPUID branching.
    dot_fn: DotFn,
    center_fn: CenterFn,

    insert_scratch: SearchScratch,
}

impl HnswIndex {
    pub fn new(m: usize, ef_construction: usize) -> Self {
        let (dot_fn, center_fn) = pick_kernels();
        Self {
            m: m.max(1),
            m0: (m * 2).max(2),
            ef_construction: ef_construction.max(1),
            dim: 0,
            pad_dim: 0,
            traversal: Vec::new(),
            inv_norms: Vec::new(),
            level_of: Vec::new(),
            neighbors: Vec::new(),
            entry: None,
            max_level: -1,
            rng: StdRng::seed_from_u64(0xBAD5_EED),
            dot_fn,
            center_fn,
            insert_scratch: SearchScratch::new(),
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.inv_norms.len()
    }

    #[inline(always)]
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
        if self.len() == 0 {
            return Vec::new();
        }

        let mut q_i8 = Vec::<i8>::new();
        self.center_bytes_to_i8_vec(query, &mut q_i8);
        let q_inv = self.inv_norm_i8(&q_i8);

        scratch.clear();
        if scratch.capacity() < self.len() {
            scratch.reserve(self.len() - scratch.capacity());
        }

        for id in 0..self.len() {
            let v = self.trav(id);
            if v.len() != q_i8.len() {
                continue;
            }
            let d = self.cosine_i8(&q_i8, q_inv, v, self.inv_norms[id]);
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
        self.center_bytes_to_i8_vec(query, &mut q_buf);
        let q_inv = self.inv_norm_i8(&q_buf);
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

        scratch.q_buf = q_buf;
        res.into_iter().map(|h| h.id).collect()
    }

    pub fn search(&self, query: &[u8], top_k: usize, ef_search: usize) -> Vec<usize> {
        let mut scratch = SearchScratch::new();
        self.search_with_scratch(query, top_k, ef_search, &mut scratch)
    }

    pub fn insert(&mut self, id: usize, vector: Vec<u8>) {
        if self.dim == 0 {
            self.dim = vector.len();
            self.pad_dim = round_up_32(self.dim);
        } else if vector.len() != self.dim {
            return;
        }

        let mut padded = vector;
        if padded.len() < self.pad_dim {
            padded.resize(self.pad_dim, 0);
        }

        let level = self.sample_level();
        self.ensure_node(id, level);

        // Center into traversal slab: (u8 -> i8) is just xor 0x80.
        {
            let dst = self.trav_mut(id);
            unsafe { (self.center_fn)(padded.as_ptr(), dst.as_mut_ptr(), self.pad_dim) };
        }

        let inv = self.inv_norm_i8(self.trav(id));
        self.inv_norms[id] = inv;

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
            let v_id = self.trav(id);
            for l in ((level + 1) as usize..=prev_max as usize).rev() {
                entry = self.greedy_search_layer(v_id, inv, entry, l as i32);
            }
        }

        // Connect on all levels down to 0.
        let v_id = self.trav(id);
        for l in (0..=level as usize).rev() {
            let level_i32 = l as i32;
            let beam = if level_i32 == 0 {
                self.m_for_level(level_i32) * 3
            } else {
                self.m_for_level(level_i32) * 2
            };
            let ef = self.ef_construction.max(beam);

            let mut scratch = std::mem::take(&mut self.insert_scratch);
            let mut candidates = self.layer_search(v_id, inv, entry, level_i32, ef, &mut scratch);
            self.insert_scratch = scratch;

            let mut selected =
                self.select_neighbors(&mut candidates, self.m_for_level(level_i32), true);
            if level_i32 == 0 && selected.len() < self.m_for_level(level_i32) / 2 {
                selected =
                    self.select_neighbors(&mut candidates, self.m_for_level(level_i32), false);
            }
            for &nbr in &selected {
                self.connect_nodes(id, nbr, level_i32);
            }
            if let Some(&closest) = selected.first() {
                entry = closest;
            }
        }
    }

    #[inline(always)]
    fn trav(&self, id: usize) -> &[i8] {
        let off = id * self.pad_dim;
        &self.traversal[off..off + self.pad_dim]
    }

    #[inline(always)]
    fn trav_mut(&mut self, id: usize) -> &mut [i8] {
        let off = id * self.pad_dim;
        &mut self.traversal[off..off + self.pad_dim]
    }

    #[inline(always)]
    fn dot_i8(&self, a: &[i8], b: &[i8]) -> i32 {
        unsafe { (self.dot_fn)(a, b) }
    }

    #[inline(always)]
    fn inv_norm_i8(&self, v: &[i8]) -> f32 {
        let sum = self.dot_i8(v, v) as f32;
        if sum <= 0.0 {
            0.0
        } else {
            1.0 / sum.sqrt()
        }
    }

    #[inline(always)]
    fn cosine_i8(&self, a: &[i8], a_inv_norm: f32, b: &[i8], b_inv_norm: f32) -> f32 {
        if a_inv_norm == 0.0 || b_inv_norm == 0.0 {
            return 1.0;
        }
        let dot = self.dot_i8(a, b) as f32;
        1.0 - dot * a_inv_norm * b_inv_norm
    }

    #[inline(always)]
    fn center_bytes_to_i8_vec(&self, src: &[u8], dst: &mut Vec<i8>) {
        dst.resize(src.len(), 0);
        unsafe { (self.center_fn)(src.as_ptr(), dst.as_mut_ptr(), src.len()) };
    }

    #[inline(always)]
    fn prefetch_traversal(&self, id: usize) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::_mm_prefetch;
            const HINT: i32 = std::arch::x86_64::_MM_HINT_T0;
            let ptr = self.traversal.as_ptr().add(id * self.pad_dim);
            _mm_prefetch(ptr as *const i8, HINT);
            if self.pad_dim > 64 {
                _mm_prefetch(ptr.add(64) as *const i8, HINT);
            }
        }
    }

    fn greedy_search_layer(&self, query: &[i8], q_inv: f32, entry: usize, level: i32) -> usize {
        let mut best = entry;
        let mut best_dist = self.cosine_i8(query, q_inv, self.trav(best), self.inv_norms[best]);
        loop {
            let mut improved = false;
            for &nbr in self.neighbors_at(best, level) {
                self.prefetch_traversal(nbr);
                let d = self.cosine_i8(query, q_inv, self.trav(nbr), self.inv_norms[nbr]);
                if d < best_dist {
                    best_dist = d;
                    best = nbr;
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
        scratch.reset(self.len(), ef);
        let visited = &mut scratch.visited;
        let epoch = scratch.epoch;
        let candidates = &mut scratch.candidates;
        let results = &mut scratch.results;

        let dist0 = self.cosine_i8(query, q_inv, self.trav(entry), self.inv_norms[entry]);
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

                self.prefetch_traversal(nbr);
                let d = self.cosine_i8(query, q_inv, self.trav(nbr), self.inv_norms[nbr]);

                if results.len() < ef {
                    results.push(HeapItem::new(nbr, d));
                    candidates.push(HeapItem::new(nbr, d));
                } else if d < worst {
                    results.pop();
                    results.push(HeapItem::new(nbr, d));
                    candidates.push(HeapItem::new(nbr, d));
                }
            }
        }

        let mut res: Vec<_> = results.drain().collect();
        res.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
        res
    }

    fn select_neighbors(&self, candidates: &mut [HeapItem], m: usize, diversity: bool) -> Vec<usize> {
        candidates.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));

        // Usually small (m ~ 8..64). Use a smallvec to avoid heap in the hot path.
        let mut accepted: SmallVec<[usize; 32]> = SmallVec::new();
        accepted.reserve(m);

        for cand in candidates.iter() {
            if accepted.len() >= m {
                break;
            }
            let mut keep = true;
            if diversity {
                let cand_v = self.trav(cand.id);
                let cand_inv = self.inv_norms[cand.id];
                for &acc in accepted.iter() {
                    let d_acc = self.cosine_i8(cand_v, cand_inv, self.trav(acc), self.inv_norms[acc]);
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

        if accepted.len() < m {
            for cand in candidates.iter() {
                if accepted.len() >= m {
                    break;
                }
                if !accepted.iter().any(|&x| x == cand.id) {
                    accepted.push(cand.id);
                }
            }
        }

        accepted.into_vec()
    }

    fn connect_nodes(&mut self, a: usize, b: usize, level: i32) {
        let inserted_ab = self.add_edge(a, b, level);
        let inserted_ba = self.add_edge(b, a, level);
        if inserted_ab && !inserted_ba {
            self.remove_edge(a, b, level);
        } else if inserted_ba && !inserted_ab {
            self.remove_edge(b, a, level);
        }
    }

    fn add_edge(&mut self, src: usize, dst: usize, level: i32) -> bool {
        self.ensure_level(src, level);
        let cap = self.m_for_level(level);

        let dist_new = self.cosine_i8(
            self.trav(src),
            self.inv_norms[src],
            self.trav(dst),
            self.inv_norms[dst],
        );

        {
            let list = &self.neighbors[src][level as usize];
            if list.contains(&dst) {
                return true;
            }
            if list.len() >= cap {
                let mut worst_idx = 0usize;
                let mut worst_dist = dist_new;

                for (i, &nid) in list.iter().enumerate() {
                    let d = self.cosine_i8(
                        self.trav(src),
                        self.inv_norms[src],
                        self.trav(nid),
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
                self.remove_edge(victim, src, level);
                return true;
            }
        }

        self.neighbors[src][level as usize].push(dst);
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
        let mut vec: SmallVec<[usize; 16]> = std::mem::take(list);
        vec.sort_by(|&a, &b| {
            let da = self.cosine_i8(self.trav(id), self.inv_norms[id], self.trav(a), self.inv_norms[a]);
            let db = self.cosine_i8(self.trav(id), self.inv_norms[id], self.trav(b), self.inv_norms[b]);
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
        if id >= self.len() {
            let new_n = id + 1;
            self.inv_norms.resize(new_n, 0.0);
            self.level_of.resize(new_n, -1);
            self.neighbors.resize_with(new_n, Vec::new);
            // traversal slab grows to new_n * pad_dim
            self.traversal.resize(new_n * self.pad_dim, 0);
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
        if level == 0 { self.m0 } else { self.m }
    }

    fn sample_level(&mut self) -> i32 {
        const P: f32 = 0.25;
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
    #[inline(always)]
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
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.epoch = 1;
            self.visited.iter_mut().for_each(|v| *v = 0);
        }
        if self.visited.len() < n {
            self.visited.resize(n, 0);
        }
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

#[inline(always)]
fn round_up_32(v: usize) -> usize {
    (v + 31) & !31
}

fn pick_kernels() -> (DotFn, CenterFn) {
    // Defaults
    let mut dot: DotFn = dot_i8_scalar_i32;
    let mut center: CenterFn = center_bytes_scalar_xor_0x80;

    #[cfg(target_arch = "x86_64")]
    unsafe {
        // Note: do NOT require avx512dq/fma; kernels below only need bw/f for the chosen path.
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
        {
            dot = dot_i8_avx512_i32;
            // centering via AVX-512 is fine but AVX2 is also fine; prefer AVX-512 when present.
            center = center_bytes_avx512_xor_0x80;
            return (dot, center);
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            dot = dot_i8_avx2_i32;
            center = center_bytes_avx2_xor_0x80;
            return (dot, center);
        }
        if std::arch::is_x86_feature_detected!("sse2") {
            center = center_bytes_sse2_xor_0x80;
        }
    }

    (dot, center)
}

#[inline(always)]
unsafe fn dot_i8_scalar_i32(a: &[i8], b: &[i8]) -> i32 {
    let len = a.len().min(b.len());
    let mut sum: i32 = 0;
    let mut i = 0usize;
    while i + 8 <= len {
        sum += (a[i] as i32) * (b[i] as i32);
        sum += (a[i + 1] as i32) * (b[i + 1] as i32);
        sum += (a[i + 2] as i32) * (b[i + 2] as i32);
        sum += (a[i + 3] as i32) * (b[i + 3] as i32);
        sum += (a[i + 4] as i32) * (b[i + 4] as i32);
        sum += (a[i + 5] as i32) * (b[i + 5] as i32);
        sum += (a[i + 6] as i32) * (b[i + 6] as i32);
        sum += (a[i + 7] as i32) * (b[i + 7] as i32);
        i += 8;
    }
    while i < len {
        sum += (a[i] as i32) * (b[i] as i32);
        i += 1;
    }
    sum
}

#[inline(always)]
unsafe fn center_bytes_scalar_xor_0x80(src: *const u8, dst: *mut i8, len: usize) {
    let s = std::slice::from_raw_parts(src, len);
    let d = std::slice::from_raw_parts_mut(dst, len);
    for i in 0..len {
        d[i] = (s[i] ^ 0x80) as i8;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i8_avx2_i32(a: &[i8], b: &[i8]) -> i32 {
    use std::arch::x86_64::*;

    let len = a.len().min(b.len());
    let mut i = 0usize;

    let mut acc32 = _mm256_setzero_si256();

    while i + 32 <= len {
        let va = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
        let vb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);

        let va_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va));
        let vb_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb));
        let va_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(va));
        let vb_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(vb));

        let p0 = _mm256_madd_epi16(va_lo, vb_lo);
        let p1 = _mm256_madd_epi16(va_hi, vb_hi);

        acc32 = _mm256_add_epi32(acc32, p0);
        acc32 = _mm256_add_epi32(acc32, p1);

        i += 32;
    }

    let mut tmp = [0i32; 8];
    _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, acc32);
    let mut total: i32 = tmp.iter().sum();

    while i < len {
        total += (a[i] as i32) * (b[i] as i32);
        i += 1;
    }

    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn dot_i8_avx512_i32(a: &[i8], b: &[i8]) -> i32 {
    use std::arch::x86_64::*;

    let len = a.len().min(b.len());
    let mut i = 0usize;

    let mut acc32 = _mm512_setzero_si512();

    while i + 64 <= len {
        // cvtepi8_epi16 takes __m256i, so do 2x32B.
        let a0 = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
        let b0 = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
        let a1 = _mm256_loadu_si256(a.as_ptr().add(i + 32) as *const __m256i);
        let b1 = _mm256_loadu_si256(b.as_ptr().add(i + 32) as *const __m256i);

        let a0_16 = _mm512_cvtepi8_epi16(a0);
        let b0_16 = _mm512_cvtepi8_epi16(b0);
        let a1_16 = _mm512_cvtepi8_epi16(a1);
        let b1_16 = _mm512_cvtepi8_epi16(b1);

        let p0 = _mm512_madd_epi16(a0_16, b0_16);
        let p1 = _mm512_madd_epi16(a1_16, b1_16);

        acc32 = _mm512_add_epi32(acc32, p0);
        acc32 = _mm512_add_epi32(acc32, p1);

        i += 64;
    }

    let mut tmp = [0i32; 16];
    _mm512_storeu_si512(tmp.as_mut_ptr() as *mut __m512i, acc32);
    let mut total: i32 = tmp.iter().sum();

    while i < len {
        total += (a[i] as i32) * (b[i] as i32);
        i += 1;
    }

    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn center_bytes_avx2_xor_0x80(src: *const u8, dst: *mut i8, len: usize) {
    use std::arch::x86_64::*;
    let mut i = 0usize;
    let mask = _mm256_set1_epi8(0x80u8 as i8);
    while i + 32 <= len {
        let v = _mm256_loadu_si256(src.add(i) as *const __m256i);
        let x = _mm256_xor_si256(v, mask);
        _mm256_storeu_si256(dst.add(i) as *mut __m256i, x);
        i += 32;
    }
    while i < len {
        *dst.add(i) = (*src.add(i) ^ 0x80) as i8;
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn center_bytes_avx512_xor_0x80(src: *const u8, dst: *mut i8, len: usize) {
    use std::arch::x86_64::*;
    let mut i = 0usize;
    let mask = _mm512_set1_epi8(0x80u8 as i8);
    while i + 64 <= len {
        let v = _mm512_loadu_si512(src.add(i) as *const _);
        let x = _mm512_xor_si512(v, mask);
        _mm512_storeu_si512(dst.add(i) as *mut _, x);
        i += 64;
    }
    while i < len {
        *dst.add(i) = (*src.add(i) ^ 0x80) as i8;
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn center_bytes_sse2_xor_0x80(src: *const u8, dst: *mut i8, len: usize) {
    use std::arch::x86_64::*;
    let mut i = 0usize;
    let mask = _mm_set1_epi8(0x80u8 as i8);
    while i + 16 <= len {
        let v = _mm_loadu_si128(src.add(i) as *const __m128i);
        let x = _mm_xor_si128(v, mask);
        _mm_storeu_si128(dst.add(i) as *mut __m128i, x);
        i += 16;
    }
    while i < len {
        *dst.add(i) = (*src.add(i) ^ 0x80) as i8;
        i += 1;
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
    fn centers_bytes_to_i8_xor() {
        let src = vec![0u8, 128, 255];
        let (.., center_fn) = super::pick_kernels();
        let mut dst = vec![0i8; src.len()];
        unsafe { (center_fn)(src.as_ptr(), dst.as_mut_ptr(), src.len()) };
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
        let res = hnsw.search(&[0u8; 4], 1, 8);
        assert!(res.is_empty());
    }
}
