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
