use crate::storage::wal::block::Entry;
use crate::storage::wal::runtime::Walrus;
use anyhow::Result;
use rkyv::{Archive, Deserialize, Serialize};
use std::cell::RefCell;
use std::sync::Arc;

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

thread_local! {
    static TL_BACKING: RefCell<Vec<u8>> = RefCell::new(Vec::new());
    static TL_RANGES: RefCell<Vec<(usize, usize)>> = RefCell::new(Vec::new());
    static TL_SLICES: RefCell<Vec<*const [u8]>> = RefCell::new(Vec::new());
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
        // One entry per vector to keep WAL entries granular and allow per-vector recovery.
        // Serialize into a single backing buffer to reduce per-entry allocations.
        if bucket.vectors.is_empty() {
            return Ok(());
        }

        let topic = Self::topic_for(bucket.id);
        self.put_chunk_raw_with_topic(bucket.id, &topic, &bucket.vectors)
            .await
    }

    /// Fast path when only vectors are provided; avoids centroid clone.
    pub async fn put_chunk_raw(&self, bucket_id: u64, vectors: &[Vector]) -> Result<()> {
        let topic = Self::topic_for(bucket_id);
        self.put_chunk_raw_with_topic(bucket_id, &topic, vectors).await
    }

    /// Same as `put_chunk_raw` but uses a precomputed topic string to avoid per-call formatting.
    pub async fn put_chunk_raw_with_topic(
        &self,
        bucket_id: u64,
        topic: &str,
        vectors: &[Vector],
    ) -> Result<()> {
        if vectors.is_empty() {
            return Ok(());
        }
        TL_BACKING.with(|backing_cell| {
            TL_RANGES.with(|ranges_cell| {
                TL_SLICES.with(|slices_cell| -> Result<()> {
                    let mut backing = backing_cell.borrow_mut();
                    backing.clear();
                    // Each entry: len prefix + id + count + data bytes.
                    let est: usize = vectors.iter().map(|v| v.data.len() * 4 + 24).sum();
                    backing.reserve(est);

                    let mut ranges = ranges_cell.borrow_mut();
                    ranges.clear();
                    ranges.reserve(vectors.len());

                    for v in vectors {
                        let start = backing.len();
                        // payload_len excludes the outer length prefix itself.
                        let payload_len = 16 + v.data.len() * 4;
                        backing.extend_from_slice(&(payload_len as u64).to_le_bytes());
                        backing.extend_from_slice(&v.id.to_le_bytes());
                        backing.extend_from_slice(&(v.data.len() as u64).to_le_bytes());
                        let data_bytes = unsafe {
                            std::slice::from_raw_parts(
                                v.data.as_ptr() as *const u8,
                                v.data.len() * 4,
                            )
                        };
                        backing.extend_from_slice(data_bytes);
                        let end = backing.len();
                        ranges.push((start, end));
                    }

                    let mut slices = slices_cell.borrow_mut();
                    let need = ranges.len().min(2000);
                    let cap = slices.capacity();
                    if cap < need {
                        slices.reserve(need - cap);
                    }
                    for chunk in ranges.chunks(2000) {
                        slices.clear();
                        slices.extend(chunk.iter().map(|(s, e)| {
                            // Store raw pointer to avoid lifetime issues; convert immediately.
                            &backing[*s..*e] as *const [u8]
                        }));
                        // SAFETY: pointers come from `backing` which lives for this scope.
                        let slice_refs: Vec<&[u8]> = slices
                            .iter()
                            .map(|ptr| unsafe { &**ptr })
                            .collect();
                        self.wal
                            .batch_append_for_topic(&topic, &slice_refs)
                            .map_err(|e| {
                                anyhow::anyhow!(
                                    "Walrus batch append failed for bucket {}: {:?}",
                                    bucket_id,
                                    e
                                )
                            })?;
                    }
                    Ok(())
                })
            })
        })
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
