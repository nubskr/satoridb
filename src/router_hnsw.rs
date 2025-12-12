use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Minimal single-layer HNSW-style graph for routing centroids.
/// Stores byte-quantized vectors and keeps up to `m` neighbors per node.
pub struct HnswIndex {
    m: usize,
    ef_construction: usize,
    vectors: Vec<Vec<u8>>,
    neighbors: Vec<Vec<usize>>,
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

    pub fn insert(&mut self, id: usize, vector: Vec<u8>) {
        // Ensure capacity for provided id.
        if id >= self.vectors.len() {
            let to_add = id + 1 - self.vectors.len();
            self.vectors.reserve(to_add);
            self.neighbors.reserve(to_add);
            while self.vectors.len() <= id {
                self.vectors.push(Vec::new());
                self.neighbors.push(Vec::new());
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
        if self.vectors.is_empty() {
            return Vec::new();
        }

        let entry = 0usize;
        let mut visited = vec![false; self.vectors.len()];
        let mut candidates: BinaryHeap<HeapItem> = BinaryHeap::new();
        let mut results: BinaryHeap<HeapItem> = BinaryHeap::new();

        let dist0 = l2_bytes(query, &self.vectors[entry]);
        candidates.push(HeapItem::new(entry, dist0));
        results.push(HeapItem::new(entry, dist0));
        visited[entry] = true;

        while let Some(cur) = candidates.pop() {
            let worst = results.peek().map(|h| h.dist).unwrap_or(f32::MAX);
            if cur.dist > worst && results.len() >= ef_search {
                break;
            }

            for &nbr in &self.neighbors[cur.id] {
                if visited.get(nbr).copied().unwrap_or(false) {
                    continue;
                }
                visited[nbr] = true;
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

        // Extract closest top_k.
        let mut res: Vec<_> = results.into_iter().collect();
        res.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
        res.truncate(top_k);
        res.into_iter().map(|h| h.id).collect()
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

fn l2_bytes(a: &[u8], b: &[u8]) -> f32 {
    let len = a.len().min(b.len());
    let mut sum = 0f32;
    let mut i = 0;
    while i + 8 <= len {
        // process 8 at a time to help auto-vectorization
        let chunk_a = &a[i..i + 8];
        let chunk_b = &b[i..i + 8];
        sum += (chunk_a[0] as f32 - chunk_b[0] as f32).powi(2);
        sum += (chunk_a[1] as f32 - chunk_b[1] as f32).powi(2);
        sum += (chunk_a[2] as f32 - chunk_b[2] as f32).powi(2);
        sum += (chunk_a[3] as f32 - chunk_b[3] as f32).powi(2);
        sum += (chunk_a[4] as f32 - chunk_b[4] as f32).powi(2);
        sum += (chunk_a[5] as f32 - chunk_b[5] as f32).powi(2);
        sum += (chunk_a[6] as f32 - chunk_b[6] as f32).powi(2);
        sum += (chunk_a[7] as f32 - chunk_b[7] as f32).powi(2);
        i += 8;
    }
    while i < len {
        let d = a[i] as f32 - b[i] as f32;
        sum += d * d;
        i += 1;
    }
    sum.sqrt()
}
