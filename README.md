# satoridb

Embedded vector database for approximate nearest neighbor (ANN) search.

This repo currently builds `satoridb` as a library crate (embedded-only, no TCP server).

## Add As A Dependency

```bash
cargo add satoridb
```

## Build

```bash
cargo build --release
```

## Embedded (in-process) mode

Run the full Satori stack (router manager + worker shards) inside your process and call it directly:

```rust
use std::sync::Arc;

use satoridb::wal::runtime::Walrus;
use satoridb::wal::{FsyncSchedule, ReadConsistency};
use satoridb::{SatoriDb, SatoriDbConfig};

fn main() -> anyhow::Result<()> {
    // By default Walrus writes under `wal_files/<key>` (relative to the current working dir).
    // Set `WALRUS_DATA_DIR=/some/dir` to control the parent directory.
    let wal = Arc::new(Walrus::with_consistency_and_schedule_for_key(
        "my_app",
        ReadConsistency::StrictlyAtOnce,
        FsyncSchedule::Milliseconds(200),
    )?);

    let mut cfg = SatoriDbConfig::new(wal);
    cfg.workers = 4;
    let db = SatoriDb::start(cfg)?;
    let api = db.handle();

    api.upsert_blocking(42, vec![1.0, 2.0, 3.0], None)?;
    let results = api.query_blocking(vec![1.0, 2.0, 3.0], 10, 200)?;
    println!("results={:?}", results);

    db.shutdown()?;
    Ok(())
}
```

You can also run the example:

```bash
cargo run --example embedded_basic
```

### Returning vectors inline

By default queries return `(id, distance)` to keep payloads small. If you want the stored vector data back on the hot path (e.g., to avoid an extra fetch), use the opt-in helper:

```rust
let hits = api.query_with_vectors_blocking(vec![1.0, 2.0, 3.0], 5, 50)?;
for (id, distance, vector) in hits {
    println!("id={id} dist={distance} vec={vector:?}");
}
```

Trade-offs: enabling this clones and ships the full vectors—expect higher CPU and larger responses. Leave it off for latency-sensitive or high-QPS paths.

## Configuration via environment variables

- `SATORI_CORES`: total cores to reserve for the process (defaults to `num_cpus`). Router/executor threads are derived from this.
- `SATORI_WORKER_CACHE_BUCKETS`: max cached buckets per worker executor (default `64`; cache arena is preallocated at startup).
- `SATORI_WORKER_CACHE_BUCKET_MB`: max size (MB) per cached bucket (default `64`; buckets larger than this are served uncached, and total preallocation is `buckets * MB`).
- `SATORI_ROUTER_REBUILD_EVERY`: how many router updates before rebuilding the in-memory router (default `1000`).
- `SATORI_REBALANCE_THRESHOLD`: vectors per bucket before triggering a split (default `4`).

### Durability

- Restart durability (clean restart): preserved as long as the process had time to write to the WAL.
- Crash durability: configure `Walrus` with a fsync schedule like `FsyncSchedule::Milliseconds(200)` or `FsyncSchedule::SyncEach` instead of `NoFsync`.

## Test

```bash
# Run all parallel-safe tests
cargo test

# Run serial tests (CPU-intensive or require single-threaded execution)
cargo test --test serial_tests -- --test-threads=1

# Run everything
cargo test && cargo test --test serial_tests -- --test-threads=1
```

### Serial Tests

Some tests are separated into `tests/serial_tests.rs` and must run single-threaded:

- **Router load tests** (`router_large_centroid_set_returns_nearest_ids`, `router_graph_path_returns_near_ids`): Build 15k-30k centroids, CPU-intensive (~16s each)
- **Rebalancer idempotency tests** (`split_on_retired_bucket_is_noop`, `merge_on_retired_bucket_is_noop`): Affected by global fail hooks used by other chaos tests
