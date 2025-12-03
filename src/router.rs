use crate::quantizer::Quantizer;
use anyhow::Result;
use hnsw_rs::prelude::*;

pub struct Router {
    index: Hnsw<'static, u8, DistL2>,
    quantizer: Quantizer,
}

impl Router {
    pub fn new(_max_elements: usize, quantizer: Quantizer) -> Self {
        // M=16, ef_construction=100 are standard defaults
        let index = Hnsw::new(16, 10000, 16, 100, DistL2);
        Self { index, quantizer }
    }

    pub fn add_centroid(&mut self, id: u64, vector: &[f32]) {
        let q_vec = self.quantizer.quantize(vector);
        self.index.insert((&q_vec, id as usize));
    }

    pub fn query(&self, vector: &[f32], top_k: usize) -> Result<Vec<u64>> {
        // Quantize the incoming float vector to match centroid space.
        let q_vec = self.quantizer.quantize(vector);
        let ef_search = std::cmp::max(top_k * 20, 200);
        let neighbors = self.index.search(&q_vec, top_k, ef_search);

        let ids = neighbors.iter().map(|n| n.d_id as u64).collect();

        Ok(ids)
    }
}
