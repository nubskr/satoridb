use crate::storage::Vector;
use anyhow::{anyhow, Context, Result};
use memmap2::{Mmap, MmapOptions};
use std::fs::File;
use std::path::Path;

/// Reader for flattened f32 blobs produced by prepare_dataset:
/// [u32 dim][u64 count][f32 data...]
pub struct FlatF32Reader {
    mmap: Mmap,
    dim: usize,
    count: usize,
    index: usize,
}

impl FlatF32Reader {
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let path_ref = path.as_ref();
        let file = File::open(path_ref).with_context(|| format!("open {}", path_ref.display()))?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        if mmap.len() < 12 {
            return Err(anyhow!("file too small to contain header: {}", path_ref.display()));
        }
        let dim = u32::from_le_bytes(mmap[0..4].try_into().unwrap()) as usize;
        let count = u64::from_le_bytes(mmap[4..12].try_into().unwrap()) as usize;
        let expected_bytes = count
            .checked_mul(dim)
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| anyhow!("overflow computing payload size"))?;
        let payload = mmap.len().saturating_sub(12);
        if payload < expected_bytes {
            return Err(anyhow!(
                "payload truncated: have {} bytes, need {} (count={}, dim={})",
                payload,
                expected_bytes,
                count,
                dim
            ));
        }
        Ok(Self {
            mmap,
            dim,
            count,
            index: 0,
        })
    }

    pub fn read_batch(&mut self, batch_size: usize) -> Result<Vec<Vector>> {
        if self.index >= self.count {
            return Ok(Vec::new());
        }
        let remaining = self.count - self.index;
        let take = remaining.min(batch_size);
        let start_f32 = self.index * self.dim;
        let end_f32 = start_f32 + take * self.dim;
        let start_byte = 12 + start_f32 * 4;
        let end_byte = 12 + end_f32 * 4;
        let slice = &self.mmap[start_byte..end_byte];

        let mut vectors = Vec::with_capacity(take);
        let mut offset = 0;
        for _ in 0..take {
            let mut data = Vec::with_capacity(self.dim);
            for _ in 0..self.dim {
                let bytes: [u8; 4] = slice[offset..offset + 4].try_into().unwrap();
                data.push(f32::from_le_bytes(bytes));
                offset += 4;
            }
            vectors.push(Vector::new(self.index as u64, data));
            self.index += 1;
        }
        Ok(vectors)
    }
}
