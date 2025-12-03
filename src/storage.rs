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
    wal: Arc<Walrus>,
}

impl Storage {
    pub fn new(wal: Arc<Walrus>) -> Self {
        Self { wal }
    }

    fn topic_for(bucket_id: u64) -> String {
        format!("bucket_{}", bucket_id)
    }

    /// Persists a bucket as an append-only wal entry.
    pub async fn put_chunk(&self, bucket: &Bucket) -> Result<()> {
        let uuid = Uuid::new_v4();
        let file_name = format!("bucket_{}_{}.rkyv", bucket.id, uuid);

        let bytes = rkyv::to_bytes::<_, 1024>(bucket)
            .map_err(|e| anyhow::anyhow!("Failed to serialize bucket: {}", e))?;

        // Keep the legacy len prefix to stay compatible with existing consumers.
        let mut payload = Vec::with_capacity(8 + bytes.len());
        payload.extend_from_slice(&bytes.len().to_le_bytes());
        payload.extend_from_slice(bytes.as_slice());

        self.wal
            .append_for_topic(&Self::topic_for(bucket.id), &payload)
            .map_err(|e| anyhow::anyhow!("Walrus append failed ({}): {:?}", file_name, e))
    }

    /// Retrieves all chunks for a bucket.
    pub async fn get_chunks(&self, bucket_id: u64) -> Result<Vec<Vec<u8>>> {
        let topic = Self::topic_for(bucket_id);
        let topic_size = self.wal.get_topic_size(&topic) as usize;
        if topic_size == 0 {
            return Ok(Vec::new());
        }

        let entries: Vec<Entry> = self
            .wal
            .batch_read_for_topic(&topic, topic_size + 1024, false, Some(0))
            .map_err(|e| anyhow::anyhow!("Walrus read failed for {}: {:?}", topic, e))?;

        Ok(entries.into_iter().map(|e| e.data).collect())
    }
}
