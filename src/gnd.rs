use flate2::read::GzDecoder;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;
use tar::Archive;

pub struct GndReader {
    // We load all into memory because random access or parallel check is needed,
    // and GT files are usually small (10k queries * 100 ints = 4MB).
    pub ground_truth: Vec<Vec<u32>>,
}

impl GndReader {
    pub fn new(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let file = File::open(path)?;
        let decoder = GzDecoder::new(file);
        let mut archive = Archive::new(decoder);

        // Find the .ivecs file inside tar
        let mut entries = archive.entries()?;
        let mut ivecs_data = Vec::new();

        while let Some(entry) = entries.next() {
            let mut entry = entry?;
            let path = entry.path()?;
            if let Some(ext) = path.extension() {
                if ext == "ivecs" {
                    // Found it
                    entry.read_to_end(&mut ivecs_data)?;
                    break;
                }
            }
        }

        if ivecs_data.is_empty() {
            return Err(anyhow::anyhow!(
                "No .ivecs file found in ground truth archive"
            ));
        }

        // Parse ivecs
        let mut reader = io::Cursor::new(ivecs_data);
        let mut ground_truth = Vec::new();
        let mut buf = [0u8; 4];

        loop {
            // Read dim (K)
            if let Err(_) = reader.read_exact(&mut buf) {
                break; // EOF
            }
            let k = u32::from_le_bytes(buf) as usize;

            // Read K integers
            let mut ids = Vec::with_capacity(k);
            for _ in 0..k {
                reader.read_exact(&mut buf)?;
                ids.push(u32::from_le_bytes(buf));
            }
            ground_truth.push(ids);
        }

        Ok(Self { ground_truth })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;
    use tar::Builder;
    use tempfile::NamedTempFile;

    #[test]
    fn parses_ivecs_from_tar_gz() {
        let mut tmp = NamedTempFile::new().unwrap();
        {
            let gz = GzEncoder::new(&mut tmp, Compression::default());
            let mut tar_builder = Builder::new(gz);
            let data = {
                let mut buf = Vec::new();
                buf.extend_from_slice(&2u32.to_le_bytes()); // dim=2
                buf.extend_from_slice(&10u32.to_le_bytes());
                buf.extend_from_slice(&20u32.to_le_bytes());
                buf
            };
            let mut header = tar::Header::new_gnu();
            header.set_size(data.len() as u64);
            header.set_cksum();
            tar_builder
                .append_data(&mut header, "gt.ivecs", data.as_slice())
                .unwrap();
            tar_builder.finish().unwrap();
        }

        let reader = GndReader::new(tmp.path()).unwrap();
        assert_eq!(reader.ground_truth.len(), 1);
        assert_eq!(reader.ground_truth[0], vec![10u32, 20u32]);
    }
}
