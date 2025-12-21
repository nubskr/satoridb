use crate::storage::Vector;
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

/// Reader for `.fvecs` files: dim (u32 LE) + dim f32 values.
pub struct FvecsReader {
    reader: BufReader<File>,
    current_id: u64,
}

impl FvecsReader {
    pub fn new(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        Ok(Self {
            reader,
            current_id: 0,
        })
    }

    /// Reads a batch of vectors; returns empty Vec on EOF.
    pub fn read_batch(&mut self, batch_size: usize) -> anyhow::Result<Vec<Vector>> {
        let mut vectors = Vec::with_capacity(batch_size);
        let mut dim_buf = [0u8; 4];

        for _ in 0..batch_size {
            if let Err(e) = self.reader.read_exact(&mut dim_buf) {
                if e.kind() == io::ErrorKind::UnexpectedEof {
                    break;
                }
                return Err(e.into());
            }
            let dim = u32::from_le_bytes(dim_buf) as usize;
            let mut data_buf = vec![0u8; dim * 4];
            self.reader.read_exact(&mut data_buf)?;

            let mut data = Vec::with_capacity(dim);
            for chunk in data_buf.chunks_exact(4) {
                let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                data.push(val);
            }

            vectors.push(Vector::new(self.current_id, data));
            self.current_id += 1;
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
    fn reads_fvecs_records() {
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&2u32.to_le_bytes()).unwrap();
        for f in [1.5f32, 2.5f32] {
            tmp.write_all(&f.to_le_bytes()).unwrap();
        }
        tmp.write_all(&2u32.to_le_bytes()).unwrap();
        for f in [3.5f32, 4.5f32] {
            tmp.write_all(&f.to_le_bytes()).unwrap();
        }
        tmp.flush().unwrap();

        let mut reader = FvecsReader::new(tmp.path()).unwrap();
        let batch = reader.read_batch(10).unwrap();
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0].data, vec![1.5, 2.5]);
        assert_eq!(batch[1].data, vec![3.5, 4.5]);
    }

    #[test]
    fn empty_on_eof() {
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&1u32.to_le_bytes()).unwrap();
        tmp.write_all(&0.0f32.to_le_bytes()).unwrap();
        tmp.flush().unwrap();

        let mut reader = FvecsReader::new(tmp.path()).unwrap();
        assert_eq!(reader.read_batch(1).unwrap().len(), 1);
        assert!(reader.read_batch(1).unwrap().is_empty());
    }
}
