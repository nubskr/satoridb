use crate::storage::Vector;
use anyhow::{anyhow, Context, Result};
use memmap2::{Mmap, MmapOptions};
use std::fs::File;
use std::path::Path;

/// Reader for flattened f32 blobs produced by prepare_dataset:
/// [u32 dim][u64 count][f32 data...]
#[derive(Debug)]
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
            return Err(anyhow!(
                "file too small to contain header: {}",
                path_ref.display()
            ));
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn reads_batches_from_flat_file() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&2u32.to_le_bytes()).unwrap();
        file.write_all(&3u64.to_le_bytes()).unwrap();
        for f in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            file.write_all(&f.to_le_bytes()).unwrap();
        }
        file.flush().unwrap();

        let mut reader = FlatF32Reader::new(file.path()).unwrap();
        let first = reader.read_batch(2).unwrap();
        assert_eq!(first.len(), 2);
        assert_eq!(first[0].data, vec![1.0, 2.0]);
        assert_eq!(first[1].data, vec![3.0, 4.0]);

        let second = reader.read_batch(2).unwrap();
        assert_eq!(second.len(), 1);
        assert_eq!(second[0].data, vec![5.0, 6.0]);

        assert!(reader.read_batch(2).unwrap().is_empty());
    }

    #[test]
    fn rejects_truncated_payload() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&2u32.to_le_bytes()).unwrap();
        file.write_all(&2u64.to_le_bytes()).unwrap();
        for f in [1.0f32, 2.0] {
            file.write_all(&f.to_le_bytes()).unwrap();
        }
        file.flush().unwrap();

        let err = FlatF32Reader::new(file.path()).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("payload truncated"), "unexpected error: {msg}");
    }
}
