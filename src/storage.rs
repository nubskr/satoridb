use crate::storage::wal::block::Entry;
use crate::storage::wal::runtime::Walrus;
use anyhow::Result;
use rkyv::{Archive, Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

pub mod wal;

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
    pub(crate) wal: Arc<Walrus>,
}

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[archive(check_bytes)]
pub enum BucketMetaStatus {
    Active,
    Retired,
}

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[archive(check_bytes)]
pub struct BucketMeta {
    pub bucket_id: u64,
    pub status: BucketMetaStatus,
}

impl Storage {
    pub fn new(wal: Arc<Walrus>) -> Self {
        Self { wal }
    }

    pub(crate) fn topic_for(bucket_id: u64) -> String {
        format!("bucket_{}", bucket_id)
    }

    /// Persists a bucket as an append-only wal entry.
    pub async fn put_chunk(&self, bucket: &Bucket) -> Result<()> {
        // One entry per vector to keep WAL entries granular.
        let mut buffers: Vec<Vec<u8>> = Vec::with_capacity(bucket.vectors.len().max(1));
        for vector in &bucket.vectors {
            let mut single = Bucket::new(bucket.id, bucket.centroid.clone());
            single.vectors.push(vector.clone());

            let uuid = Uuid::new_v4();
            let file_name = format!("bucket_{}_{}.rkyv", single.id, uuid);

            let bytes = rkyv::to_bytes::<_, 1024>(&single)
                .map_err(|e| anyhow::anyhow!("Failed to serialize bucket: {}", e))?;

            let mut payload = Vec::with_capacity(8 + bytes.len());
            payload.extend_from_slice(&bytes.len().to_le_bytes());
            payload.extend_from_slice(bytes.as_slice());
            buffers.push(payload);

            // If serialization fails mid-loop, ensure we surface the file name.
            let _ = file_name;
        }

        if buffers.is_empty() {
            return Ok(());
        }

        let batch_refs: Vec<&[u8]> = buffers.iter().map(|b| b.as_slice()).collect();
        // Walrus enforces a max batch of 2000 entries; chunk if needed.
        for chunk in batch_refs.chunks(2000) {
            self.wal
                .batch_append_for_topic(&Self::topic_for(bucket.id), chunk)
                .map_err(|e| anyhow::anyhow!("Walrus batch append failed for bucket {}: {:?}", bucket.id, e))?;
        }
        Ok(())
    }

    /// Emits a bucket metadata record (active/retired) for housekeeping.
    pub async fn put_bucket_meta(&self, meta: &BucketMeta) -> Result<()> {
        let bytes = rkyv::to_bytes::<_, 256>(meta)
            .map_err(|e| anyhow::anyhow!("Failed to serialize bucket meta: {}", e))?;
        let mut payload = Vec::with_capacity(8 + bytes.len());
        payload.extend_from_slice(&bytes.len().to_le_bytes());
        payload.extend_from_slice(bytes.as_slice());
        self.wal
            .append_for_topic("bucket_meta", &payload)
            .map_err(|e| anyhow::anyhow!("Walrus append failed (bucket_meta): {:?}", e))
    }

    /// Retrieves all chunks for a bucket.
    pub async fn get_chunks(&self, bucket_id: u64) -> Result<Vec<Vec<u8>>> {
        let topic = Self::topic_for(bucket_id);
        let topic_size = self.wal.get_topic_size(&topic) as usize;
        if topic_size == 0 {
            return Ok(Vec::new());
        }

        // Use stateful read (start_offset=None) so the active writer block is also included.
        // Stateless reads with start_offset=Some(0) can miss in-flight data still in the writer.
        let entries: Vec<Entry> = self
            .wal
            .batch_read_for_topic(&topic, topic_size + 1024, false, None)
            .map_err(|e| anyhow::anyhow!("Walrus read failed for {}: {:?}", topic, e))?;

        Ok(entries.into_iter().map(|e| e.data).collect())
    }
}
