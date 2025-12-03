use rkyv::{Archive, Deserialize, Serialize};
use std::path::PathBuf;
use anyhow::Result;
use glommio::io::{DmaBuffer, DmaFile, rename, ReadResult};
use std::convert::TryInto;

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[archive(check_bytes)]
pub struct Vector {
    pub id: u64,
    pub data: Vec<f32>,
}

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[archive(check_bytes)]
pub struct Bucket {
    pub id: u64,
    pub centroid: Vec<f32>,
    pub vectors: Vec<Vector>,
}

impl Vector {
    pub fn new(id: u64, data: Vec<f32>) -> Self {
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

    /// Fetches the raw file content (including length prefix) for a bucket.
    pub async fn fetch_raw(&self, bucket_id: u64) -> Result<ReadResult> {
        let file_path = self.root_path.join(format!("bucket_{}.rkyv", bucket_id));
        
        let file = DmaFile::open(&file_path).await
            .map_err(|e| anyhow::anyhow!("Failed to open file {:?}: {:?}", file_path, e))?;
            
        let size = file.file_size().await
            .map_err(|e| anyhow::anyhow!("Failed to get file size for {:?}: {:?}", file_path, e))?;
            
        let buffer = file.read_at(0, size as usize).await
            .map_err(|e| anyhow::anyhow!("Failed to read file {:?}: {:?}", file_path, e))?;
            
        file.close().await
            .map_err(|e| anyhow::anyhow!("Failed to close file {:?}: {:?}", file_path, e))?;
            
        Ok(buffer)
    }

    /// Persists a bucket to disk atomically.
    pub async fn put(&self, bucket: &Bucket) -> Result<()> {
        let file_name = format!("bucket_{}.rkyv", bucket.id);
        let final_path = self.root_path.join(&file_name);
        let temp_path = self.root_path.join(format!("{}.tmp", file_name));
        
        // Serialize the bucket
        let bytes = rkyv::to_bytes::<_, 1024>(bucket)
            .map_err(|e| anyhow::anyhow!("Failed to serialize bucket: {}", e))?;

        // Create temp file
        let file = DmaFile::create(&temp_path).await
            .map_err(|e| anyhow::anyhow!("Failed to create file {:?}: {:?}", temp_path, e))?;

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
            .map_err(|e| anyhow::anyhow!("Failed to write to file {:?}: {:?}", temp_path, e))?;
            
        file.close().await
            .map_err(|e| anyhow::anyhow!("Failed to close file {:?}: {:?}", temp_path, e))?;

        // Atomic Rename
        rename(&temp_path, &final_path).await
            .map_err(|e| anyhow::anyhow!("Failed to rename {:?} to {:?}: {:?}", temp_path, final_path, e))?;

        Ok(())
    }

    /// Retrieves buckets from disk by their IDs.
    pub async fn get(&self, bucket_ids: &[u64]) -> Result<Vec<Bucket>> {
        let mut buckets = Vec::with_capacity(bucket_ids.len());
        
        for id in bucket_ids {
            let buffer = self.fetch_raw(*id).await?;
            
            // Use as_ref() (or deref) to access bytes
            let bytes = &*buffer;
            
            if bytes.len() < 8 {
                return Err(anyhow::anyhow!("File too small (corrupted header) for bucket {}", id));
            }

            let len_bytes: [u8; 8] = bytes[0..8].try_into()
                .map_err(|e| anyhow::anyhow!("Failed to read length prefix: {:?}", e))?;
            let archive_len = u64::from_le_bytes(len_bytes) as usize;

            if 8 + archive_len > bytes.len() {
                return Err(anyhow::anyhow!("Archive length {} exceeds buffer size {}", archive_len, bytes.len()));
            }
                
            let archive_slice = &bytes[8..8+archive_len];
                
            let bucket: Bucket = rkyv::from_bytes(archive_slice)
                .map_err(|e| anyhow::anyhow!("Failed to deserialize bucket {}: {}", id, e))?;
                
            buckets.push(bucket);
        }
        
        Ok(buckets)
    }
}