# Changelog

All notable changes to SatoriDB will be documented in this file.

## [0.1.0] - 2024-12-21

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
