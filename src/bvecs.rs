use crate::storage::Vector;
use flate2::read::GzDecoder;
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

pub struct BvecsReader {
    reader: BufReader<GzDecoder<File>>,
    current_id: u64,
}

impl BvecsReader {
    pub fn new(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let file = File::open(path)?;
        let decoder = GzDecoder::new(file);
        let reader = BufReader::new(decoder);
        Ok(Self {
            reader,
            current_id: 0,
        })
    }

    /// Reads a batch of vectors.
    /// Returns empty Vec if EOF.
    pub fn read_batch(&mut self, batch_size: usize) -> anyhow::Result<Vec<Vector>> {
        let mut vectors = Vec::with_capacity(batch_size);
        let mut buf = [0u8; 4]; // For dimension

        for _ in 0..batch_size {
            // Read dimension (4 bytes, little endian)
            if let Err(e) = self.reader.read_exact(&mut buf) {
                if e.kind() == io::ErrorKind::UnexpectedEof {
                    break;
                }
                return Err(e.into());
            }
            let dim = u32::from_le_bytes(buf) as usize;

            // Validate dim (SIFT is 128)
            if dim != 128 {
                // In generic bvecs, dim can vary? No, usually fixed.
                // If EOF happens exactly between vectors?
                // read_exact handles partial read as UnexpectedEof.
                // If dim is weird, maybe corrupt.
                // SIFT1B is 128.
            }

            // Read vector data (dim * 1 byte)
            let mut data_buf = vec![0u8; dim];
            self.reader.read_exact(&mut data_buf)?;

            // Upcast to f32 for distance calculations.
            let data: Vec<f32> = data_buf.iter().map(|&v| v as f32).collect();
            vectors.push(Vector::new(self.current_id, data));
            self.current_id += 1;
        }

        Ok(vectors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn reads_multiple_vectors() {
        let mut tmp = NamedTempFile::new().unwrap();
        {
            let mut encoder = GzEncoder::new(&mut tmp, Compression::default());
            // Two vectors, dim = 3
            for vec_vals in [[1u8, 2, 3], [4u8, 5, 6]] {
                encoder
                    .write_all(&(vec_vals.len() as u32).to_le_bytes())
                    .unwrap();
                encoder.write_all(&vec_vals).unwrap();
            }
            encoder.finish().unwrap();
        }

        let mut reader = BvecsReader::new(tmp.path()).unwrap();
        let batch = reader.read_batch(10).unwrap();
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0].id, 0);
        assert_eq!(batch[0].data, vec![1.0, 2.0, 3.0]);
        assert_eq!(batch[1].id, 1);
        assert_eq!(batch[1].data, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn stops_on_eof() {
        let mut tmp = NamedTempFile::new().unwrap();
        {
            let mut encoder = GzEncoder::new(&mut tmp, Compression::default());
            encoder.write_all(&1u32.to_le_bytes()).unwrap(); // dim = 1
            encoder.write_all(&[9u8]).unwrap();
            encoder.finish().unwrap();
        }

        let mut reader = BvecsReader::new(tmp.path()).unwrap();
        let first = reader.read_batch(1).unwrap();
        let second = reader.read_batch(1).unwrap();
        assert_eq!(first.len(), 1);
        assert!(second.is_empty());
    }
}
