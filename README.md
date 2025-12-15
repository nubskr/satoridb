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

### Durability

- Restart durability (clean restart): preserved as long as the process had time to write to the WAL.
- Crash durability: configure `Walrus` with a fsync schedule like `FsyncSchedule::Milliseconds(200)` or `FsyncSchedule::SyncEach` instead of `NoFsync`.

## Test

```bash
cargo test
```
