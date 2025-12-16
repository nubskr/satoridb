use std::sync::Arc;

use anyhow::Result;
use futures::executor::block_on;
use satoridb::storage::{Bucket, Storage, Vector};
use satoridb::wal::runtime::Walrus;
use tempfile::TempDir;

fn init_wal(tempdir: &TempDir) -> Arc<Walrus> {
    std::env::set_var("WALRUS_DATA_DIR", tempdir.path());
    Arc::new(Walrus::new().expect("walrus init"))
}

/// Empty vector list should not write anything to WAL.
#[test]
fn empty_bucket_does_not_write() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let storage = Storage::new(wal.clone());

    let bucket = Bucket::new(0, vec![0.0, 0.0]);
    // Empty bucket (no vectors)
    block_on(storage.put_chunk(&bucket))?;

    // Reading should return empty
    let chunks = block_on(storage.get_chunks(0))?;
    assert!(chunks.is_empty(), "empty bucket should not create WAL entries");

    Ok(())
}

/// Single vector round-trip: write and read back.
#[test]
fn single_vector_round_trip() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let storage = Storage::new(wal);

    let mut bucket = Bucket::new(0, vec![1.0, 2.0]);
    bucket.add_vector(Vector::new(42, vec![1.0, 2.0, 3.0, 4.0]));
    block_on(storage.put_chunk(&bucket))?;

    let chunks = block_on(storage.get_chunks(0))?;
    assert!(!chunks.is_empty(), "should have WAL entries");

    // Parse the chunk to verify format
    let chunk = &chunks[0];
    assert!(chunk.len() >= 8, "chunk should have length prefix");

    Ok(())
}

/// Multiple vectors in same bucket.
#[test]
fn multiple_vectors_same_bucket() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let storage = Storage::new(wal);

    let mut bucket = Bucket::new(0, vec![0.0, 0.0]);
    for i in 0..10u64 {
        bucket.add_vector(Vector::new(i, vec![i as f32, (i * 2) as f32]));
    }
    block_on(storage.put_chunk(&bucket))?;

    let chunks = block_on(storage.get_chunks(0))?;
    assert_eq!(chunks.len(), 10, "should have 10 WAL entries (one per vector)");

    Ok(())
}

/// Vectors with different dimensions can coexist in same bucket (no validation at storage layer).
#[test]
fn mixed_dimensions_stored() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let storage = Storage::new(wal);

    let mut bucket = Bucket::new(0, vec![0.0]);
    bucket.add_vector(Vector::new(0, vec![1.0]));
    bucket.add_vector(Vector::new(1, vec![1.0, 2.0]));
    bucket.add_vector(Vector::new(2, vec![1.0, 2.0, 3.0]));
    block_on(storage.put_chunk(&bucket))?;

    let chunks = block_on(storage.get_chunks(0))?;
    assert_eq!(chunks.len(), 3, "all vectors should be stored");

    Ok(())
}

/// Reading non-existent bucket returns empty.
#[test]
fn nonexistent_bucket_returns_empty() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let storage = Storage::new(wal);

    let chunks = block_on(storage.get_chunks(999))?;
    assert!(chunks.is_empty(), "non-existent bucket should return empty");

    Ok(())
}

/// Multiple buckets are independent.
#[test]
fn multiple_buckets_independent() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let storage = Storage::new(wal);

    let mut b0 = Bucket::new(0, vec![0.0]);
    b0.add_vector(Vector::new(0, vec![0.0]));
    b0.add_vector(Vector::new(1, vec![1.0]));

    let mut b1 = Bucket::new(1, vec![10.0]);
    b1.add_vector(Vector::new(100, vec![10.0]));

    block_on(storage.put_chunk(&b0))?;
    block_on(storage.put_chunk(&b1))?;

    let chunks0 = block_on(storage.get_chunks(0))?;
    let chunks1 = block_on(storage.get_chunks(1))?;

    assert_eq!(chunks0.len(), 2, "bucket 0 should have 2 vectors");
    assert_eq!(chunks1.len(), 1, "bucket 1 should have 1 vector");

    Ok(())
}

/// Large batch of vectors (tests chunking at 2000 entries).
#[test]
fn large_batch_vectors() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let storage = Storage::new(wal);

    let mut bucket = Bucket::new(0, vec![0.0, 0.0]);
    for i in 0..3000u64 {
        bucket.add_vector(Vector::new(i, vec![i as f32, (i * 2) as f32]));
    }
    block_on(storage.put_chunk(&bucket))?;

    let chunks = block_on(storage.get_chunks(0))?;
    assert_eq!(chunks.len(), 3000, "all 3000 vectors should be stored");

    Ok(())
}

/// put_chunk_raw works same as put_chunk.
#[test]
fn put_chunk_raw_equivalent() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let storage = Storage::new(wal);

    let vectors = vec![
        Vector::new(0, vec![1.0, 2.0]),
        Vector::new(1, vec![3.0, 4.0]),
    ];
    block_on(storage.put_chunk_raw(0, &vectors))?;

    let chunks = block_on(storage.get_chunks(0))?;
    assert_eq!(chunks.len(), 2);

    Ok(())
}

/// Zero-dimension vector is stored (no validation at storage layer).
#[test]
fn zero_dimension_vector_stored() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let storage = Storage::new(wal);

    let mut bucket = Bucket::new(0, vec![]);
    bucket.add_vector(Vector::new(0, vec![]));
    block_on(storage.put_chunk(&bucket))?;

    let chunks = block_on(storage.get_chunks(0))?;
    assert_eq!(chunks.len(), 1, "zero-dim vector should be stored");

    Ok(())
}

/// High-dimension vector.
#[test]
fn high_dimension_vector() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let storage = Storage::new(wal);

    let dim = 1536; // OpenAI embedding dimension
    let data: Vec<f32> = (0..dim).map(|i| i as f32 / dim as f32).collect();
    let mut bucket = Bucket::new(0, data.clone());
    bucket.add_vector(Vector::new(0, data));
    block_on(storage.put_chunk(&bucket))?;

    let chunks = block_on(storage.get_chunks(0))?;
    assert_eq!(chunks.len(), 1);

    Ok(())
}
