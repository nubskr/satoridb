use std::sync::Arc;

use anyhow::Result;
use futures::executor::block_on;
use satoridb::executor::{Executor, WorkerCache};
use satoridb::router::{Router, RoutingTable};
use satoridb::storage::{Storage, Vector};
use satoridb::wal::runtime::Walrus;
use tempfile::TempDir;

fn init_wal(tempdir: &TempDir) -> Arc<Walrus> {
    Arc::new(Walrus::with_data_dir(tempdir.path().to_path_buf()).expect("walrus init"))
}

#[test]
fn executor_respects_routing_versions_and_cache_invalidation() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let storage = Storage::new(wal);

    // Seed bucket 0 with a single far-away vector and warm the executor cache.
    let cache = WorkerCache::new(8, usize::MAX);
    let executor = Executor::new(storage.clone(), cache);
    let v1 = Vector::new(1, vec![0.0, 0.0]);
    block_on(storage.put_chunk_raw(0, &[v1]))?;

    let query = vec![100.0, 100.0];
    let res = block_on(executor.query(&query, &[0], 1, 0, Arc::new(Vec::new()), false))?;
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].0, 1, "cached bucket should return original vector");

    // Append a closer vector to the same bucket. Without a routing version bump,
    // the executor should continue serving from the cached content.
    let v2 = Vector::new(2, vec![100.0, 100.0]);
    block_on(storage.put_chunk_raw(0, &[v2]))?;
    let res_stale = block_on(executor.query(&query, &[0], 1, 0, Arc::new(Vec::new()), false))?;
    assert_eq!(
        res_stale[0].0, 1,
        "without routing bump, cache should still serve old contents"
    );

    // Bump the routing version and include the changed bucket id: the cache
    // must be invalidated and the fresher vector should now win.
    let res_refreshed = block_on(executor.query(&query, &[0], 1, 1, Arc::new(vec![0]), false))?;
    assert_eq!(
        res_refreshed[0].0, 2,
        "after routing bump, executor should reload bucket 0"
    );

    Ok(())
}

#[test]
fn routing_table_installs_router_and_bumps_version() {
    let routing = RoutingTable::new();
    assert_eq!(routing.current_version(), 0);

    // Minimal router with one centroid.
    let quantizer = satoridb::quantizer::Quantizer::new(0.0, 1.0);
    let mut router = Router::new(1, quantizer);
    router.add_centroid(42, &[0.0, 0.0]);

    let changed = vec![123, 456];
    let new_version = routing.install(router, changed.clone());
    assert_eq!(new_version, 1);
    assert_eq!(routing.current_version(), 1);

    let snap = routing.snapshot().expect("snapshot present");
    assert_eq!(snap.version, 1);
    assert_eq!(&*snap.changed, &changed);
}
