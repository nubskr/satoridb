use rkyv::{Archive, Deserialize, Serialize};
use std::path::PathBuf;
use anyhow::Result;
use glommio::io::{DmaBuffer, DmaFile, rename, ReadResult};
use std::convert::TryInto;
use uuid::Uuid;
use std::fs;

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[archive(check_bytes)]
pub struct Vector {
    pub id: u64,
    pub data: Vec<u8>,
}

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[archive(check_bytes)]
pub struct Bucket {
    pub id: u64,
    pub centroid: Vec<f32>,
    pub vectors: Vec<Vector>,
}

impl Vector {
    pub fn new(id: u64, data: Vec<u8>) -> Self {
        Self { id, data }
    }
}

impl Bucket {
    pub fn new(id: u64, centroid: Vec<f32>) -> Self {
        Self {
            id,
            centroid,
            vectors: Vec::new(),
        }
    }

    pub fn add_vector(&mut self, vector: Vector) {
        self.vectors.push(vector);
    }
}

#[derive(Clone)]
pub struct Storage {
    root_path: PathBuf,
}

impl Storage {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        let root_path = path.into();
        std::fs::create_dir_all(&root_path).expect("Failed to create storage directory");
        Self {
            root_path,
        }
    }

    /// Fetches raw content of a specific file path.
    pub async fn fetch_file(&self, path: PathBuf) -> Result<ReadResult> {
        let file = DmaFile::open(&path).await
            .map_err(|e| anyhow::anyhow!("Failed to open file {:?}: {:?}", path, e))?;
            
        let size = file.file_size().await
            .map_err(|e| anyhow::anyhow!("Failed to get file size for {:?}: {:?}", path, e))?;
            
        let buffer = file.read_at(0, size as usize).await
            .map_err(|e| anyhow::anyhow!("Failed to read file {:?}: {:?}", path, e))?;
            
        file.close().await
            .map_err(|e| anyhow::anyhow!("Failed to close file {:?}: {:?}", path, e))?;
            
        Ok(buffer)
    }

    /// Persists a bucket as a new chunk (immutable file).
    pub async fn put_chunk(&self, bucket: &Bucket) -> Result<()> {
        let uuid = Uuid::new_v4();
        let file_name = format!("bucket_{}_{}.rkyv", bucket.id, uuid);
        let final_path = self.root_path.join(&file_name);
        
        // Serialize the bucket
        let bytes = rkyv::to_bytes::<_, 1024>(bucket)
            .map_err(|e| anyhow::anyhow!("Failed to serialize bucket: {}", e))?;

        // Create file
        let file = DmaFile::create(&final_path).await
            .map_err(|e| anyhow::anyhow!("Failed to create file {:?}: {:?}", final_path, e))?;

        // Prepare buffer with length prefix
        let archive_len = bytes.len();
        let total_len = 8 + archive_len;
        let aligned_len = (total_len + 511) & !511;

        // Allocate aligned buffer
        let mut dma_buf = file.alloc_dma_buffer(aligned_len);
        
        // Write length prefix
        dma_buf.as_bytes_mut()[0..8].copy_from_slice(&archive_len.to_le_bytes());
        
        // Copy archive data
        dma_buf.as_bytes_mut()[8..8+archive_len].copy_from_slice(bytes.as_slice());
        
        // Zero padding
        if 8 + archive_len < aligned_len {
            dma_buf.as_bytes_mut()[8+archive_len..].fill(0);
        }
            
        // Write at offset 0
        file.write_at(dma_buf, 0).await
            .map_err(|e| anyhow::anyhow!("Failed to write to file {:?}: {:?}", final_path, e))?;
            
        file.close().await
            .map_err(|e| anyhow::anyhow!("Failed to close file {:?}: {:?}", final_path, e))?;

        Ok(())
    }

    /// Retrieves all chunks for a bucket.
    pub async fn get_chunks(&self, bucket_id: u64) -> Result<Vec<ReadResult>> {
        let prefix = format!("bucket_{}_", bucket_id);
        let mut chunks = Vec::new();
        
        // Scan directory (Blocking, but fast enough for metadata)
        let entries = fs::read_dir(&self.root_path)?;
        
        for entry in entries {
            let entry = entry?;
            let fname = entry.file_name().into_string().unwrap_or_default();
            if fname.starts_with(&prefix) && fname.ends_with(".rkyv") {
                let path = entry.path();
                // Async read
                let buffer = self.fetch_file(path).await?;
                chunks.push(buffer);
            }
        }
        
        Ok(chunks)
    }
}