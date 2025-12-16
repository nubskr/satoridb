use anyhow::Result;
use flate2::write::GzEncoder;
use flate2::Compression;
use futures::executor::block_on;
use satoridb::bvecs::BvecsReader;
use satoridb::executor::{Executor, WorkerCache};
use satoridb::flatbin::FlatF32Reader;
use satoridb::fvecs::FvecsReader;
use satoridb::storage::Storage;
use satoridb::wal::runtime::Walrus;
use std::io::Write;
use std::sync::Arc;
use tempfile::{tempdir, NamedTempFile, TempDir};

fn init_wal(tempdir: &TempDir) -> Arc<Walrus> {
    Arc::new(Walrus::with_data_dir(tempdir.path().to_path_buf()).expect("walrus init"))
}

#[test]
fn flat_reader_round_trips_through_storage_and_executor() -> Result<()> {
    // Build a small flat f32 file: dim=3, count=3.
    let mut file = NamedTempFile::new()?;
    file.write_all(&3u32.to_le_bytes())?; // dim
    file.write_all(&3u64.to_le_bytes())?; // count
                                          // Three vectors.
    for vals in [[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]] {
        for v in vals {
            file.write_all(&v.to_le_bytes())?;
        }
    }
    file.flush()?;

    let mut reader = FlatF32Reader::new(file.path())?;
    let vectors = reader.read_batch(10)?;
    assert_eq!(vectors.len(), 3, "reader should return all vectors");
    assert_eq!(vectors[0].id, 0);
    assert_eq!(vectors[1].id, 1);
    assert_eq!(vectors[2].id, 2);

    let wal_tmp = tempdir()?;
    let wal = init_wal(&wal_tmp);
    let storage = Storage::new(wal);

    // Persist into bucket 0.
    block_on(storage.put_chunk_raw(0, &vectors))?;

    // Query through executor to ensure decode path returns the same IDs.
    let executor = Executor::new(storage.clone(), WorkerCache::new(4, usize::MAX));
    let res = block_on(executor.query(&[0.0f32; 3], &[0], 10, 0, Arc::new(Vec::new())))?;
    let ids: Vec<u64> = res.iter().map(|(id, _)| *id).collect();
    assert_eq!(ids.len(), vectors.len());
    for v in &vectors {
        assert!(
            ids.contains(&v.id),
            "missing id {} after round-trip through storage/executor",
            v.id
        );
    }

    Ok(())
}

#[test]
fn bvecs_reader_round_trip_pipeline() -> Result<()> {
    // Build a gzipped bvecs file with two vectors of dim=2.
    let mut tmp = NamedTempFile::new()?;
    {
        let mut encoder = GzEncoder::new(&mut tmp, Compression::default());
        for vals in [[1u8, 2u8], [10u8, 20u8]] {
            encoder.write_all(&(vals.len() as u32).to_le_bytes())?;
            encoder.write_all(&vals)?;
        }
        encoder.finish()?;
    }

    let mut reader = BvecsReader::new(tmp.path())?;
    let vectors = reader.read_batch(10)?;
    assert_eq!(vectors.len(), 2);
    assert_eq!(vectors[0].data, vec![1.0, 2.0]);
    assert_eq!(vectors[1].data, vec![10.0, 20.0]);

    let wal_tmp = tempdir()?;
    let wal = init_wal(&wal_tmp);
    let storage = Storage::new(wal);
    block_on(storage.put_chunk_raw(0, &vectors))?;

    let executor = Executor::new(storage.clone(), WorkerCache::new(4, usize::MAX));
    let res = block_on(executor.query(&[0.0, 0.0], &[0], 10, 0, Arc::new(Vec::new())))?;
    let ids: Vec<u64> = res.iter().map(|(id, _)| *id).collect();
    assert_eq!(ids.len(), 2);
    assert!(ids.contains(&0));
    assert!(ids.contains(&1));
    Ok(())
}

#[test]
fn fvecs_reader_round_trip_pipeline() -> Result<()> {
    // Build a small fvecs file: two vectors of dim=3.
    let mut tmp = NamedTempFile::new()?;
    // First vector
    tmp.write_all(&3u32.to_le_bytes())?;
    for f in [1.0f32, 2.0, 3.0] {
        tmp.write_all(&f.to_le_bytes())?;
    }
    // Second vector
    tmp.write_all(&3u32.to_le_bytes())?;
    for f in [4.0f32, 5.0, 6.0] {
        tmp.write_all(&f.to_le_bytes())?;
    }
    tmp.flush()?;

    let mut reader = FvecsReader::new(tmp.path())?;
    let vectors = reader.read_batch(10)?;
    assert_eq!(vectors.len(), 2);
    assert_eq!(vectors[0].data, vec![1.0, 2.0, 3.0]);
    assert_eq!(vectors[1].data, vec![4.0, 5.0, 6.0]);

    let wal_tmp = tempdir()?;
    let wal = init_wal(&wal_tmp);
    let storage = Storage::new(wal);
    block_on(storage.put_chunk_raw(0, &vectors))?;

    let executor = Executor::new(storage.clone(), WorkerCache::new(4, usize::MAX));
    let res = block_on(executor.query(&[0.0, 0.0, 0.0], &[0], 10, 0, Arc::new(Vec::new())))?;
    let ids: Vec<u64> = res.iter().map(|(id, _)| *id).collect();
    assert_eq!(ids.len(), 2);
    assert!(ids.contains(&0));
    assert!(ids.contains(&1));
    Ok(())
}
