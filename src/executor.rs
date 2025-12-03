use crate::storage::{Storage, Bucket};
use anyhow::Result;
use glommio::io::ReadResult;
use std::convert::TryInto;

pub struct Executor {
    storage: Storage,
}

impl Executor {
    pub fn new(storage: Storage) -> Self {
        Self { storage }
    }

    pub async fn query(&self, query_vec: &[u8], bucket_ids: &[u64], top_k: usize) -> Result<Vec<(u64, f32)>> {
        let mut candidates = Vec::new();

        for &id in bucket_ids {
            // Fetch all chunks for this bucket (Append-Only Storage)
            let chunks = self.storage.get_chunks(id).await.unwrap_or_default();

            for chunk in chunks {
                // Access Archived Data (Zero-Copy)
                let bytes = &*chunk;
                
                if bytes.len() < 8 { continue; }
                let len_bytes: [u8; 8] = bytes[0..8].try_into().unwrap_or([0; 8]);
                let archive_len = u64::from_le_bytes(len_bytes) as usize;
                
                if 8 + archive_len > bytes.len() { continue; }
                let data_slice = &bytes[8..8+archive_len];

                // Access bucket using rkyv
                if let Ok(archived_bucket) = rkyv::check_archived_root::<Bucket>(data_slice) {
                     for vector in archived_bucket.vectors.iter() {
                         if vector.data.len() != query_vec.len() { continue; }
                         let dist = l2_distance_u8(&vector.data, query_vec);
                         candidates.push((vector.id, dist));
                     }
                }
            }
        }

        // Sort by distance (ascending)
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Take top_k
        if candidates.len() > top_k {
            candidates.truncate(top_k);
        }

        Ok(candidates)
    }
}

fn l2_distance_u8(a: &[u8], b: &[u8]) -> f32 {
    let sum_sq: u32 = a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let diff = x as i32 - y as i32;
            (diff * diff) as u32
        })
        .sum();
    (sum_sq as f32).sqrt()
}
