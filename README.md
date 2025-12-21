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
use std::sync::Arc;
use satoridb::wal::runtime::Walrus;
use satoridb::wal::{FsyncSchedule, ReadConsistency};
use satoridb::{SatoriDb, SatoriDbConfig};

fn main() -> anyhow::Result<()> {
    // Initialize storage (writes to wal_files/my_app/)
    let wal = Arc::new(Walrus::with_consistency_and_schedule_for_key(
        "my_app",
        ReadConsistency::StrictlyAtOnce,
        FsyncSchedule::Milliseconds(200),
    )?);

    // Start database with 4 worker threads
    let mut cfg = SatoriDbConfig::new(wal);
    cfg.workers = 4;
    let db = SatoriDb::start(cfg)?;
    let api = db.handle();

    // Upsert vectors
    api.upsert_blocking(1, vec![0.1, 0.2, 0.3], None)?;
    api.upsert_blocking(2, vec![0.2, 0.3, 0.4], None)?;
    api.upsert_blocking(3, vec![0.9, 0.8, 0.7], None)?;

    // Query: find 10 nearest neighbors, probe 200 buckets
    let results = api.query_blocking(vec![0.15, 0.25, 0.35], 10, 200)?;
    for (id, distance) in results {
        println!("id={id} distance={distance}");
    }

    db.shutdown()?;
    Ok(())
}
```

```bash
cargo run --example embedded_basic
```

## API

### Core Operations

```rust
// Insert a vector (id, data, optional bucket_hint)
api.upsert_blocking(id, vector, None)?;

// Query: returns Vec<(id, distance)>
let results = api.query_blocking(query_vector, top_k, router_top_k)?;

// Query with vectors inline: returns Vec<(id, distance, vector)>
let results = api.query_with_vectors_blocking(query_vector, top_k, router_top_k)?;

// Fetch vectors by ID (via RocksDB index)
let vectors = api.fetch_vectors_by_id_blocking(vec![1, 2, 3])?;
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `top_k` | Number of results to return |
| `router_top_k` | Number of buckets to probe (higher = better recall, slower) |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SATORI_REBALANCE_THRESHOLD` | `2000` | Split bucket when vector count exceeds this |
| `SATORI_ROUTER_REBUILD_EVERY` | `1000` | Rebuild HNSW index after N upserts |
| `SATORI_WORKER_CACHE_BUCKETS` | `64` | Max buckets cached per worker |
| `SATORI_WORKER_CACHE_BUCKET_MB` | `64` | Max MB per cached bucket |
| `SATORI_VECTOR_INDEX_PATH` | `vector_index` | RocksDB path for id→vector index |
| `SATORI_BUCKET_INDEX_PATH` | `bucket_index` | RocksDB path for id→bucket index |
| `WALRUS_DATA_DIR` | `./wal_files` | Storage directory |

### Durability

Configure via `FsyncSchedule` when creating Walrus:

```rust
// Fsync every 200ms (default), balances durability and throughput
FsyncSchedule::Milliseconds(200)

// Fsync every write, maximum durability
FsyncSchedule::SyncEach

// No fsync, maximum throughput, data loss on crash
FsyncSchedule::NoFsync
```

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
