<div align="center">
  <img src="https://avatars.githubusercontent.com/u/241193657?s=200&v=4"
       alt="octopii"
       width="8%">
  <div>SatoriDB: Billion scale embedded vector database</div>
  <br>

</div>

# Architecture

![architecture](./assets/architecture.png)

SatoriDB runs entirely in-process on a single node. It uses a **two-tier search** architecture:

1. **Routing (HNSW)**: A quantized HNSW index over bucket centroids finds the most relevant clusters in O(log N)
2. **Scanning (Workers)**: CPU-pinned Glommio executors scan selected buckets in parallel using SIMD-accelerated L2 distance

We use a variant of [SPFresh](https://arxiv.org/pdf/2410.14452), Vectors are organized into **buckets** (clusters of similar vectors). A background rebalancer automatically splits buckets via k-means when they exceed a threshold, keeping search efficient as data grows. All data is persisted to walrus high performance storage engine and we use RocksDB indexes for point lookups.

See [docs/architecture.md](docs/architecture.md) for detailed documentation including:
- System overview and component diagrams
- Two-tier search architecture
- Storage layer (Walrus + RocksDB)
- Rebalancer and clustering algorithms
- Data flow diagrams

```
SatoriHandle ──▶ Router Manager ──▶ HNSW Index (centroids)
      │                                   │
      │         ┌─────────────────────────┘
      │         ▼
      │    Bucket IDs ──▶ Consistent Hash Ring
      │                          │
      ▼                          ▼
  Workers ◀──────────────── bucket_id → shard
      │
      ▼
  Walrus (storage) + RocksDB (indexes)
```

## Features

- **Embedded**: runs entirely in-process, no external services
- **Two-tier search**: HNSW routing + parallel bucket scanning
- **Automatic clustering**: vectors grouped by similarity, splits when buckets grow
- **CPU-pinned workers**: Glommio executors with io_uring
- **SIMD acceleration**: AVX2/AVX-512 for distance computation
- **Configurable durability**: fsync schedules from "every write" to "no sync"
- **Persistent storage**: Walrus (topic-based append storage) + RocksDB indexes

Linux only (requires io_uring, kernel 5.8+)

## Install

```bash
cargo add satoridb
```

## Quick Start

```rust
use satoridb::SatoriDb;

fn main() -> anyhow::Result<()> {
    let db = SatoriDb::open("my_app")?;

    db.insert(1, vec![0.1, 0.2, 0.3])?;
    db.insert(2, vec![0.2, 0.3, 0.4])?;
    db.insert(3, vec![0.9, 0.8, 0.7])?;

    let results = db.query(vec![0.15, 0.25, 0.35], 10)?;
    for (id, distance) in results {
        println!("id={id} distance={distance}");
    }

    Ok(()) // auto-shutdown on drop
}
```

## API

### Core Operations

```rust
// Insert a vector (rejects duplicates)
db.insert(id, vector)?;

// Delete a vector
db.delete(id)?;

// Query nearest neighbors: returns Vec<(id, distance)>
let results = db.query(query_vector, top_k)?;

// Query with vectors inline: returns Vec<(id, distance, vector)>
let results = db.query_with_vectors(query_vector, top_k)?;

// Fetch vectors by ID
let vectors = db.get(vec![1, 2, 3])?;

// Get stats
let stats = db.stats();
println!("buckets={} vectors={}", stats.buckets, stats.vectors);
```

### Async API

```rust
db.insert_async(id, vector).await?;
db.delete_async(id).await?;
let results = db.query_async(query_vector, top_k).await?;
let vectors = db.get_async(ids).await?;
```

## Configuration

```rust
let db = SatoriDb::builder("my_app")
    .workers(4)              // Worker threads (default: num_cpus)
    .fsync_ms(100)           // Fsync interval (default: 200ms)
    .data_dir("/custom/path") // Data directory
    .build()?;
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SATORI_REBALANCE_THRESHOLD` | `2000` | Split bucket when vector count exceeds this |
| `SATORI_ROUTER_REBUILD_EVERY` | `1000` | Rebuild HNSW index after N inserts |
| `SATORI_WORKER_CACHE_BUCKETS` | `64` | Max buckets cached per worker |
| `SATORI_WORKER_CACHE_BUCKET_MB` | `64` | Max MB per cached bucket |

## Build

```bash
cargo build --release
```

## Test

## Benchmark (BigANN)

- Requires significant disk (~1TB+ download + converted). See `Makefile` targets.
- Run `make benchmark` to download BigANN base/query/ground-truth, convert the base set via `prepare_dataset`, and execute the benchmark (`SATORI_RUN_BENCH=1 cargo run --release --bin satoridb`).
- Default ingest ceiling is 1B vectors (BigANN); uses streaming ingestion and queries via `src/bin/satoridb.rs`.
- On 1B+ (bigger-than-RAM) workloads, the benchmark reports 95%+ recall using the default settings.

## License

See [LICENSE](LICENSE).

> **Note**: SatoriDB is in early development (v0.1.0). APIs may change between versions. See [CHANGELOG.md](CHANGELOG.md) for release notes.
