use std::sync::Arc;

use anyhow::Result;
use futures::executor::block_on;
use satoridb::executor::{Executor, WorkerCache};
use satoridb::indexer::Indexer;
use satoridb::quantizer::Quantizer;
use satoridb::router::{Router, RoutingTable};
use satoridb::storage::{Bucket, Storage, Vector};
use satoridb::wal::runtime::Walrus;
use tempfile::TempDir;

fn init_wal(tempdir: &TempDir) -> Arc<Walrus> {
    Arc::new(Walrus::with_data_dir(tempdir.path().to_path_buf()).expect("walrus init"))
}

#[test]
fn storage_executor_pipeline_finds_nearest_vectors() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let storage = Storage::new(wal);

    // Two clearly separated clusters; ids encoded by cluster.
    let vectors = vec![
        Vector::new(1, vec![0.0, 0.0]),
        Vector::new(2, vec![0.2, 0.1]),
        Vector::new(101, vec![10.0, 10.0]),
        Vector::new(102, vec![9.8, 10.1]),
    ];

    // Let the indexer build buckets and persist them.
    let mut buckets = Indexer::build_clusters(vectors, 2);
    buckets.sort_by_key(|b| b.id);
    for b in &buckets {
        block_on(storage.put_chunk(b))?;
    }

    // Build a router over the bucket centroids.
    let quantizer = Quantizer::new(0.0, 10.0);
    let mut router = Router::new(buckets.len(), quantizer);
    for b in &buckets {
        router.add_centroid(b.id, &b.centroid);
    }
    let routing = RoutingTable::new();
    let version = routing.install(router, buckets.iter().map(|b| b.id).collect());
    let snap = routing.snapshot().expect("router installed");

    // Executor should route each query to the correct cluster and fetch from WAL.
    let executor = Executor::new(
        storage.clone(),
        WorkerCache::new(8, 1024 * 1024, 1024 * 1024),
    );

    let res_a = block_on(executor.query(
        &[0.05, 0.05],
        &snap.router.query(&[0.05, 0.05], 1)?,
        1,
        version,
        snap.changed.clone(),
        false,
    ))?;
    assert_eq!(res_a.len(), 1);
    assert!(
        res_a[0].0 == 1 || res_a[0].0 == 2,
        "expected nearest from small cluster, got {:?}",
        res_a
    );

    let res_b = block_on(executor.query(
        &[9.9, 9.9],
        &snap.router.query(&[9.9, 9.9], 1)?,
        1,
        version,
        snap.changed.clone(),
        false,
    ))?;
    assert_eq!(res_b.len(), 1);
    assert!(
        res_b[0].0 == 101 || res_b[0].0 == 102,
        "expected nearest from large cluster, got {:?}",
        res_b
    );

    Ok(())
}

#[test]
fn storage_put_and_read_keeps_all_vectors() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let storage = Storage::new(wal);

    let mut bucket = Bucket::new(7, vec![0.0, 0.0]);
    for i in 0..5 {
        let f = i as f32;
        bucket.add_vector(Vector::new(1_000 + i, vec![f, -f]));
    }
    block_on(storage.put_chunk(&bucket))?;

    // Re-read via executor, which exercises the decoding path used in production.
    let executor = Executor::new(
        storage.clone(),
        WorkerCache::new(4, 1024 * 1024, 1024 * 1024),
    );
    let res = block_on(executor.query(
        &[0.0, 0.0],
        &[bucket.id],
        10,
        0,
        Arc::new(Vec::new()),
        false,
    ))?;
    let ids: Vec<u64> = res.iter().map(|(id, _, _)| *id).collect();
    assert_eq!(
        ids.len(),
        bucket.vectors.len(),
        "executor should read every stored vector"
    );
    for v in &bucket.vectors {
        assert!(
            ids.contains(&v.id),
            "missing vector id {} after round-trip",
            v.id
        );
    }

    Ok(())
}
