# Changelog

All notable changes to SatoriDB will be documented in this file.

## [0.1.1] - 2025-12-23

### Breaking Changes

Complete public API redesign. The old `SatoriHandle` + `SatoriDbConfig` API is gone.

### New API

**Simple open:**
```rust
let db = SatoriDb::open("my_app")?;
```

**Builder for configuration:**
```rust
let db = SatoriDb::builder("my_app")
    .workers(4)
    .fsync_ms(100)
    .data_dir("/custom/path")
    .virtual_nodes(8)
    .build()?;
```

**Core operations:**
- `db.insert(id, vector)` — insert (rejects duplicates)
- `db.delete(id)` — delete by ID
- `db.query(vector, top_k)` — nearest neighbor search
- `db.query_with_probes(vector, top_k, probes)` — query with custom probe count
- `db.query_with_vectors(vector, top_k)` — query returning stored vectors
- `db.get(ids)` — fetch vectors by ID
- `db.stats()` — database statistics
- `db.flush()` — flush pending writes

**Async variants:** `insert_async`, `delete_async`, `query_async`, `get_async`

### Improvements

- **Auto-shutdown on Drop** — no manual shutdown required
- **Duplicate ID rejection** — bloom filter optimization for fast duplicate detection
- **Walrus logs suppressed by default** — set `WALRUS_QUIET=0` to enable

### Removed

- `SatoriDbConfig` struct
- `SatoriHandle` direct access
- `upsert_blocking`, `query_blocking` naming

## [0.1.0] - 2025-12-21

Initial release.

### Core Architecture

- **Two-tier search**: HNSW index over bucket centroids for O(log N) routing, followed by parallel linear scan within selected buckets
- **Bucket-based clustering**: Vectors automatically grouped by similarity using k-means
- **Automatic rebalancing**: Buckets split when exceeding threshold (default: 2000 vectors)
- **CPU-pinned workers**: Glommio executors with io_uring for async I/O (Linux)
- **Consistent hash ring**: Deterministic bucket-to-worker assignment

### Storage

- **Walrus**: Topic-based append-only storage engine
  - 10MB blocks, 1GB files
  - FNV-1a checksums for integrity
  - Configurable fsync schedules (per-write, periodic, none)
  - FD backend with io_uring batch I/O (Linux) or mmap fallback
- **RocksDB indexes**:
  - VectorIndex: id → vector lookup
  - BucketIndex: id → bucket_id mapping

### HNSW Router

- Quantized centroids (f32 → u8) for memory efficiency
- Cosine similarity over centered i8 vectors
- Parameters: M=24, M0=48, ef_construction=180
- Adaptive ef_search scaling (64-4000 based on query requirements)

### Performance

- **SIMD acceleration**: AVX2/AVX-512 kernels for:
  - L2 distance computation (k-means, bucket scan)
  - Dot product (HNSW cosine similarity)
  - Byte centering (u8 → i8 conversion)
- **Worker cache**: LRU cache for hot bucket data (configurable size)
- **Batch I/O**: io_uring submission for Walrus reads/writes

### API

**SatoriHandle (main interface):**
- `upsert_blocking(id, vector, bucket_hint)` — insert vector (fails if id exists)
- `query_blocking(vector, top_k, router_top_k)` — ANN search returning (id, distance)
- `query_with_vectors_blocking(...)` — search with vectors inline
- `fetch_vectors_by_id_blocking(ids)` — retrieve vectors by ID

**RebalanceWorker (delete operations):**
- `delete(id, bucket_hint)` — async delete
- `delete_inline_blocking(id, bucket_hint)` — sync delete

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SATORI_REBALANCE_THRESHOLD` | 2000 | Bucket split threshold |
| `SATORI_ROUTER_REBUILD_EVERY` | 1000 | HNSW rebuild frequency |
| `SATORI_WORKER_CACHE_BUCKETS` | 64 | Max cached buckets per worker |
| `SATORI_WORKER_CACHE_BUCKET_MB` | 64 | Max MB per cached bucket |
| `WALRUS_DATA_DIR` | `./wal_files` | Storage directory |

### Platform

- **Linux only** — Glommio requires io_uring (Linux 5.8+)

### Known Limitations

- Embedded-only (no TCP server)
- Single-node only (no distributed mode)
