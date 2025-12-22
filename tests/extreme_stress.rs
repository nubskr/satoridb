use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use satoridb::SatoriDb;

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(0x9e3779b97f4a7c15);
    *seed
}

fn make_vec(seed: &mut u64, dim: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(dim);
    for _ in 0..dim {
        let bits = (lcg(seed) >> 32) as u32;
        out.push((bits as f32 / u32::MAX as f32) * 2.0 - 1.0);
    }
    out
}

/// Long-running ingest/query soak that pushes thousands of inserts and queries.
/// Ignored by default because it runs for several seconds.
#[test]
#[ignore]
fn long_running_ingest_and_query_soak() -> Result<()> {
    let tmp = tempfile::tempdir()?;

    let db = SatoriDb::builder("stress")
        .workers(std::cmp::min(4, num_cpus::get().max(1)))
        .virtual_nodes(32)
        .data_dir(tmp.path())
        .build()?;

    let duration = Duration::from_secs(8);
    let start = Instant::now();

    let next_id = Arc::new(AtomicU64::new(1));
    let recorded = Arc::new(parking_lot::Mutex::new(Vec::new()));

    std::thread::scope(|s| {
        // Writers hammer inserts and occasionally query
        for t in 0..4 {
            let db = &db;
            let next_id = next_id.clone();
            let recorded = recorded.clone();
            s.spawn(move || {
                let mut seed = 0x1234_5678_9abc_def0u64 ^ (t as u64);
                let mut local = Vec::new();
                while start.elapsed() < duration {
                    let id = next_id.fetch_add(1, Ordering::Relaxed);
                    let vector = make_vec(&mut seed, 16);
                    if db.insert(id, vector.clone()).is_ok() {
                        local.push(id);
                        if id.is_multiple_of(8) {
                            let _ = db.query_with_vectors(vector, 8);
                        }
                    }
                    if local.len() >= 256 {
                        let mut guard = recorded.lock();
                        guard.extend(local.drain(..));
                    }
                }
                if !local.is_empty() {
                    recorded.lock().extend(local);
                }
            });
        }

        // Readers do queries
        for t in 0..2 {
            let db = &db;
            s.spawn(move || {
                let mut seed = 0xdead_beef_cafe_babeu64 ^ (t as u64);
                while start.elapsed() < duration {
                    let vector = make_vec(&mut seed, 16);
                    let _ = db.query(vector, 10);
                }
            });
        }
    });

    let ids = recorded.lock().clone();
    println!("Inserted {} vectors in {:?}", ids.len(), start.elapsed());

    // Verify some are queryable
    let sample = &ids[..std::cmp::min(100, ids.len())];
    let fetched = db.get(sample.to_vec())?;
    assert!(!fetched.is_empty(), "should fetch at least some vectors");

    Ok(())
}

/// Stress test: many inserts then queries
#[test]
#[ignore]
fn stress_many_inserts_then_query() -> Result<()> {
    let tmp = tempfile::tempdir()?;

    let db = SatoriDb::builder("stress")
        .workers(4)
        .data_dir(tmp.path())
        .build()?;

    let count = 10_000;
    let mut seed = 0x12345678u64;

    for id in 0..count {
        let vec = make_vec(&mut seed, 32);
        db.insert(id, vec)?;
    }

    // Query should work
    seed = 0x12345678u64;
    for _ in 0..100 {
        let vec = make_vec(&mut seed, 32);
        let results = db.query(vec, 10)?;
        assert!(!results.is_empty());
    }

    Ok(())
}

/// Stress test: concurrent inserts and deletes
#[test]
#[ignore]
fn stress_concurrent_insert_delete() -> Result<()> {
    let tmp = tempfile::tempdir()?;

    let db = Arc::new(
        SatoriDb::builder("stress")
            .workers(4)
            .data_dir(tmp.path())
            .build()?,
    );

    let duration = Duration::from_secs(5);
    let start = Instant::now();
    let next_id = Arc::new(AtomicU64::new(1));

    std::thread::scope(|s| {
        // Inserters
        for _ in 0..2 {
            let db = db.clone();
            let next_id = next_id.clone();
            s.spawn(move || {
                let mut seed = 0xaabbccddu64;
                while start.elapsed() < duration {
                    let id = next_id.fetch_add(1, Ordering::Relaxed);
                    let vec = make_vec(&mut seed, 16);
                    let _ = db.insert(id, vec);
                }
            });
        }

        // Deleters (delete old IDs)
        for _ in 0..1 {
            let db = db.clone();
            s.spawn(move || {
                let mut delete_id = 1u64;
                while start.elapsed() < duration {
                    let _ = db.delete(delete_id);
                    delete_id += 1;
                    std::thread::sleep(Duration::from_micros(100));
                }
            });
        }

        // Queriers
        for _ in 0..1 {
            let db = db.clone();
            s.spawn(move || {
                let mut seed = 0xdeadbeefu64;
                while start.elapsed() < duration {
                    let vec = make_vec(&mut seed, 16);
                    let _ = db.query(vec, 5);
                }
            });
        }
    });

    println!("Final ID counter: {}", next_id.load(Ordering::Relaxed));

    Ok(())
}

/// Basic stress: rapid insert-query cycles
#[test]
fn rapid_insert_query_cycles() -> Result<()> {
    let tmp = tempfile::tempdir()?;

    let db = SatoriDb::builder("stress")
        .workers(2)
        .data_dir(tmp.path())
        .build()?;

    let mut seed = 0x1234u64;

    for cycle in 0..100 {
        let id = cycle as u64;
        let vec = make_vec(&mut seed, 8);
        db.insert(id, vec.clone())?;

        let results = db.query(vec, 5)?;
        assert!(!results.is_empty(), "cycle {} should find results", cycle);
    }

    Ok(())
}

/// Stress: get after many inserts
#[test]
fn stress_get_after_inserts() -> Result<()> {
    let tmp = tempfile::tempdir()?;

    let db = SatoriDb::builder("stress")
        .workers(2)
        .data_dir(tmp.path())
        .build()?;

    let count = 500;
    let mut seed = 0xabcdu64;

    for id in 0..count {
        let vec = make_vec(&mut seed, 16);
        db.insert(id, vec)?;
    }

    // Get all at once
    let ids: Vec<u64> = (0..count).collect();
    let fetched = db.get(ids)?;
    assert_eq!(fetched.len(), count as usize, "should fetch all vectors");

    Ok(())
}
