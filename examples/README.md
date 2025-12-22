# Examples

- `embedded_basic.rs`: minimal setup - open, insert, query
- `embedded_async.rs`: async API with custom configuration
- `api_tour.rs`: comprehensive tour of all public APIs

## Quick Start

```rust
use satoridb::SatoriDb;

fn main() -> anyhow::Result<()> {
    let db = SatoriDb::open("my_app")?;

    db.insert(1, vec![0.1, 0.2, 0.3])?;
    db.insert(2, vec![0.2, 0.3, 0.4])?;

    let results = db.query(vec![0.15, 0.25, 0.35], 10)?;
    for (id, distance) in results {
        println!("id={id} distance={distance}");
    }

    Ok(())
}
```

## Configuration

```rust
let db = SatoriDb::builder("my_app")
    .workers(4)              // Worker threads (default: num_cpus)
    .fsync_ms(100)           // Fsync interval (default: 200ms)
    .data_dir("/custom/path") // Custom data directory
    .virtual_nodes(16)       // Hash ring granularity (default: 8)
    .build()?;
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SATORI_REBALANCE_THRESHOLD` | `2000` | Split bucket when vector count exceeds this |
| `SATORI_ROUTER_REBUILD_EVERY` | `1000` | Rebuild HNSW index after N inserts |
| `SATORI_WORKER_CACHE_BUCKETS` | `64` | Max buckets cached per worker |
| `SATORI_WORKER_CACHE_BUCKET_MB` | `64` | Max MB per cached bucket |
