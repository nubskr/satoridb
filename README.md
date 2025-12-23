<div align="center">
  <img src="https://avatars.githubusercontent.com/u/241193657?s=200&v=4"
       alt="octopii"
       width="8%">
  <div>SatoriDB: Billion scale embedded vector database</div>
  <br>

</div>

# Architecture

![architecture](./assets/architecture.png)

SatoriDB is a two-tier search system: a small "hot" index in RAM routes queries to "cold" vector data on disk. This lets us handle billion-scale datasets without holding everything in memory.

**Routing (Hot Tier)**

Quantized HNSW index over bucket centroids. Centroids are scalar-quantized (f32 → u8) so the whole routing index fits in RAM even at 500k+ buckets. When a query comes in, HNSW finds the top-K most relevant buckets in O(log N). We only search those, not the entire dataset.

**Scanning (Cold Tier)**

CPU-pinned Glommio workers scan selected buckets in parallel. Shared-nothing: each worker has its own io_uring ring, LRU cache, and pre-allocated heap. No cross-core synchronization on the query path. SIMD everywhere: L2 distance, dot products, quantization, k-means assignments all have AVX2/AVX-512 paths. Cache misses stream from disk without blocking.

**Clustering & Rebalancing**

Vectors are grouped into buckets (clusters) via k-means. A background rebalancer automatically splits buckets when they exceed ~2000 vectors, keeping bucket sizes predictable. Predictable sizes = predictable query latency. Inspired by [SPFresh](https://arxiv.org/pdf/2410.14452).

**Storage**

Walrus handles bulk vector storage (append-only, io_uring, topic-per-bucket). RocksDB indexes handle point lookups (fetch-by-id, duplicate detection). See [docs/architecture.md](docs/architecture.md) for the full deep-dive.

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
    let db = SatoriDb::builder("my_app")
        .workers(4)              // Worker threads (default: num_cpus)
        .fsync_ms(100)           // Fsync interval (default: 200ms)
        .data_dir("/tmp/mydb")   // Data directory
        .build()?;

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

> **Note**: SatoriDB is in early development (v0.1.2). APIs may change between versions. See [CHANGELOG.md](CHANGELOG.md) for release notes.
