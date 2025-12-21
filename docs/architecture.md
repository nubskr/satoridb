# SatoriDB Architecture

SatoriDB is an embedded vector database for approximate nearest neighbor (ANN) search.
It runs entirely in-process and persists vector data to Walrus, a topic-based storage engine.

---

## System Overview

```
                                 ┌─────────────────────────────────────────────────────────────┐
                                 │                        SatoriDB                             │
                                 └─────────────────────────────────────────────────────────────┘
                                                            │
                         ┌──────────────────────────────────┼──────────────────────────────────┐
                         │                                  │                                  │
                         ▼                                  ▼                                  ▼
              ┌────────────────────┐             ┌────────────────────┐             ┌────────────────────┐
              │   SatoriHandle     │             │   Router Manager   │             │     Rebalancer     │
              │      (API)         │             │     (Thread)       │             │     (Thread)       │
              │                    │             │                    │             │                    │
              │  • query()         │────────────▶│  • HNSW Router     │             │  • Monitor sizes   │
              │  • insert()        │             │  • Bucket metadata │             │  • Split buckets   │
              │  • flush()         │             │  • Centroid mgmt   │             │  • K-means (k=2)   │
              └────────────────────┘             └────────────────────┘             └────────────────────┘
                         │
                         │
                         ▼
              ┌─────────────────────────────────────────────────────────────────────────────────────────┐
              │                              Consistent Hash Ring                                       │
              │                            bucket_id  ───▶  worker_shard                                │
              └─────────────────────────────────────────────────────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┬───────────────┬───────────────┐
         │               │               │               │               │
         ▼               ▼               ▼               ▼               ▼
   ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐
   │  Worker 0 │   │  Worker 1 │   │  Worker 2 │   │  Worker 3 │   │  Worker N │
   │  (CPU 0)  │   │  (CPU 1)  │   │  (CPU 2)  │   │  (CPU 3)  │   │  (CPU N)  │
   │           │   │           │   │           │   │           │   │           │
   │  Glommio  │   │  Glommio  │   │  Glommio  │   │  Glommio  │   │  Glommio  │
   │  Executor │   │  Executor │   │  Executor │   │  Executor │   │  Executor │
   └─────┬─────┘   └─────┬─────┘   └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
         │               │               │               │               │
         └───────────────┴───────────────┴───────┬───────┴───────────────┘
                                                 │
                                                 ▼
              ┌─────────────────────────────────────────────────────────────────────────────────────────┐
              │                                                                                         │
              │                                   Storage Layer                                         │
              │                                                                                         │
              │    ┌─────────────────────────────────────────────────────────────────────────────┐     │
              │    │                              Walrus                                          │     │
              │    │                     (Core Storage Engine)                                    │     │
              │    │                                                                              │     │
              │    │   Topic-based append-only storage for bucket/cluster data                   │     │
              │    │   • io_uring batch I/O (Linux)                                              │     │
              │    │   • 10MB blocks, 1GB files                                                  │     │
              │    │   • FNV-1a checksums                                                        │     │
              │    └─────────────────────────────────────────────────────────────────────────────┘     │
              │                                                                                         │
              │    ┌────────────────────────────┐       ┌────────────────────────────┐                 │
              │    │       VectorIndex          │       │       BucketIndex          │                 │
              │    │        (RocksDB)           │       │        (RocksDB)           │                 │
              │    │     id ──▶ vector          │       │     id ──▶ bucket_id       │                 │
              │    └────────────────────────────┘       └────────────────────────────┘                 │
              │                                                                                         │
              └─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Concepts

### Buckets (Clusters)

Vectors are organized into **buckets** - clusters of similar vectors. Each bucket has:

```
┌──────────────────────────────────────────────────────────────────────┐
│                           Bucket                                     │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ID:        42                                                      │
│   Centroid:  [0.12, 0.45, 0.78, 0.33, ...]   ◄── mean of all vectors│
│   Count:     1,847                                                   │
│                                                                      │
│   Vectors (stored in Walrus topic "bucket_42"):                      │
│   ┌────────────────────────────────────────────────────────────┐    │
│   │  id: 1001  │  [0.11, 0.44, 0.79, 0.32, ...]               │    │
│   ├────────────────────────────────────────────────────────────┤    │
│   │  id: 1002  │  [0.13, 0.46, 0.77, 0.34, ...]               │    │
│   ├────────────────────────────────────────────────────────────┤    │
│   │  id: 1003  │  [0.12, 0.45, 0.78, 0.33, ...]               │    │
│   ├────────────────────────────────────────────────────────────┤    │
│   │    ...     │              ...                              │    │
│   └────────────────────────────────────────────────────────────┘    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Two-Tier Search Architecture

```
                              ┌───────────────────┐
                              │    Query Vector   │
                              │  [0.1, 0.2, ...]  │
                              └─────────┬─────────┘
                                        │
                    ════════════════════╪════════════════════
                         TIER 1: ROUTING (Router Manager)
                    ════════════════════╪════════════════════
                                        │
                                        ▼
                    ┌───────────────────────────────────────┐
                    │             HNSW Index                │
                    │       (over bucket centroids)         │
                    │                                       │
                    │   Quantized to u8, cosine distance    │
                    │   M=24, ef_construction=180           │
                    │                                       │
                    │   Input:  query vector, top_k=200     │
                    │   Output: [bucket_12, bucket_47, ...] │
                    └───────────────────────────────────────┘
                                        │
                                        ▼
                    ┌───────────────────────────────────────┐
                    │      Selected Bucket IDs              │
                    │   [12, 47, 89, 103, 156, 201, ...]    │
                    └───────────────────────────────────────┘
                                        │
                    ════════════════════╪════════════════════
                        TIER 2: SCANNING (Workers)
                    ════════════════════╪════════════════════
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              │                         │                         │
              ▼                         ▼                         ▼
     ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
     │    Worker 0     │       │    Worker 1     │       │    Worker 2     │
     │                 │       │                 │       │                 │
     │  Buckets:       │       │  Buckets:       │       │  Buckets:       │
     │  [12, 89, 201]  │       │  [47, 156]      │       │  [103]          │
     │                 │       │                 │       │                 │
     │  For each:      │       │  For each:      │       │  For each:      │
     │  1. Load from   │       │  1. Load from   │       │  1. Load from   │
     │     Walrus      │       │     Walrus      │       │     Walrus      │
     │  2. L2 scan     │       │  2. L2 scan     │       │  2. L2 scan     │
     │  3. Return      │       │  3. Return      │       │  3. Return      │
     │     top-100     │       │     top-100     │       │     top-100     │
     └────────┬────────┘       └────────┬────────┘       └────────┬────────┘
              │                         │                         │
              └─────────────────────────┼─────────────────────────┘
                                        │
                                        ▼
                    ┌───────────────────────────────────────┐
                    │         Merge & Sort Results          │
                    │                                       │
                    │    Sort all candidates by distance    │
                    │    Return final top_k to caller       │
                    └───────────────────────────────────────┘
```

---

## Component Details

### Router Manager

Single-threaded component that manages bucket metadata and the HNSW routing index:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Router Manager                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────┐      ┌─────────────────────────────────┐  │
│  │      Bucket Metadata        │      │          HNSW Router            │  │
│  │                             │      │                                 │  │
│  │  HashMap<bucket_id,         │      │  • Stores quantized centroids  │  │
│  │    BucketMeta {             │─────▶│  • Hierarchical graph layers   │  │
│  │      id: u64,               │      │  • Cosine similarity search    │  │
│  │      centroid: Vec<f32>,    │      │                                 │  │
│  │      count: u64,            │      │  Query: O(log N) bucket lookup │  │
│  │    }                        │      │                                 │  │
│  │  >                          │      └─────────────────────────────────┘  │
│  └─────────────────────────────┘                                            │
│                                                                             │
│  Centroid update formula (running mean):                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  new_centroid[i] = (old_centroid[i] * count + vector[i]) / (count+1)│   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Persisted to Walrus topics:                                                │
│    • "router_snapshot" ─── full bucket state                               │
│    • "router_updates"  ─── incremental centroid updates                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Workers

Each worker runs a Glommio executor pinned to a CPU:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Worker Architecture                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Glommio Executor                             │   │
│  │                    (io_uring async runtime)                          │   │
│  │                                                                      │   │
│  │   • CPU-pinned for cache locality                                   │   │
│  │   • Bounded message channel (capacity: 1000)                        │   │
│  │   • Max 32 concurrent query tasks                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                           LRU Cache                                  │   │
│  │                                                                      │   │
│  │   Caches hot bucket data to avoid repeated Walrus reads             │   │
│  │                                                                      │   │
│  │   Default config:                                                   │   │
│  │     • max_buckets: 64                                               │   │
│  │     • max_bytes_per_bucket: 64 MB                                   │   │
│  │     • total: ~4 GB arena                                            │   │
│  │                                                                      │   │
│  │   Eviction: LRU with routing-version invalidation                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Message Types:                                                             │
│    ├── Query   ──▶ search buckets, return (id, distance) pairs            │
│  │    ├── Upsert  ──▶ persist single vector                                 │
│    ├── Ingest ──▶ batch persist (with duplicate detection)                │
│    ├── Flush  ──▶ drain pending writes                                    │
│    └── FetchVectors ──▶ retrieve by ID                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Consistent Hash Ring

Maps bucket IDs to worker shards:

```
                          Consistent Hash Ring
            ┌──────────────────────────────────────────────┐
            │                                              │
            │                  ┌─────┐                     │
            │             ┌────┤ W0  ├────┐                │
            │        ┌────┘    └─────┘    └────┐           │
            │   ┌────┘                        └────┐      │
            │ ┌─────┐                            ┌─────┐  │
            │ │ W3  │                            │ W1  │  │
            │ └─────┘                            └─────┘  │
            │   └────┐                        ┌────┘      │
            │        └────┐    ┌─────┐    ┌────┘           │
            │             └────┤ W2  ├────┘                │
            │                  └─────┘                     │
            │                                              │
            │   hash(bucket_id) lands on ring              │
            │   → assigned to next clockwise worker        │
            │                                              │
            │   Virtual nodes per worker: 8 (default)      │
            │                                              │
            └──────────────────────────────────────────────┘

    Code: src/tasks.rs:23-50
```

---

## Storage Layer

### Walrus - Core Storage Engine

Walrus is a topic-based append-only storage engine that stores all bucket/cluster data:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                  Walrus                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Topics (one per bucket + system topics):                                   │
│                                                                             │
│    bucket_0          ──▶  [entry][entry][entry][entry]...                  │
│    bucket_1          ──▶  [entry][entry]...                                │
│    bucket_42         ──▶  [entry][entry][entry][entry][entry]...           │
│    bucket_103        ──▶  [entry][entry][entry]...                         │
│    router_snapshot   ──▶  [snap][snap]...                                  │
│    router_updates    ──▶  [update][update][update]...                      │
│    bucket_meta       ──▶  [active:0][active:1][retired:5]...               │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Entry Format (per vector):                                                 │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ PREFIX_META (256B)  │  len (8B)  │  id (8B)  │  dim (8B)  │  f32[]   │  │
│  │                     │            │           │            │          │  │
│  │  • meta length      │  payload   │  vector   │  vector    │  vector  │  │
│  │  • owned_by topic   │  size      │  id       │  dimension │  data    │  │
│  │  • next_block_start │            │           │            │          │  │
│  │  • FNV-1a checksum  │            │           │            │          │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Physical Layout:                                                           │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         File (1 GB)                                  │   │
│  │  ┌─────────┬─────────┬─────────┬─────────┬─────────────────────────┐│   │
│  │  │ Block 0 │ Block 1 │ Block 2 │   ...   │      Block 99           ││   │
│  │  │  10 MB  │  10 MB  │  10 MB  │         │       10 MB             ││   │
│  │  └─────────┴─────────┴─────────┴─────────┴─────────────────────────┘│   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Constants (src/storage/wal/config.rs):                                     │
│    • DEFAULT_BLOCK_SIZE:  10 MB                                            │
│    • BLOCKS_PER_FILE:     100                                              │
│    • PREFIX_META_SIZE:    256 bytes                                        │
│    • MAX_BATCH_ENTRIES:   2,000,000                                        │
│    • MAX_BATCH_BYTES:     10 GB                                            │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  I/O Backends:                                                              │
│    • FD Backend (default): pread/pwrite, io_uring for batches             │
│    • Mmap Backend:         memory-mapped files (fallback)                  │
│                                                                             │
│  Fsync Schedule:                                                            │
│    • Milliseconds(200)  ─── default, fsync every 200ms                     │
│    • SyncEach           ─── O_SYNC on every write                          │
│    • NoFsync            ─── maximum throughput, no durability              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

    Code: src/storage/wal/
```

### RocksDB Indexes

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             RocksDB Indexes                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────┐   ┌─────────────────────────────────┐ │
│  │         VectorIndex             │   │         BucketIndex             │ │
│  │                                 │   │                                 │ │
│  │  Key:   vector_id (u64, LE)     │   │  Key:   vector_id (u64, LE)     │ │
│  │  Value: rkyv-serialized Vector  │   │  Value: bucket_id (u64, LE)     │ │
│  │                                 │   │                                 │ │
│  │  Purpose:                       │   │  Purpose:                       │ │
│  │  • Fetch vector by ID           │   │  • Find which bucket contains   │ │
│  │  • Duplicate detection on       │   │    a given vector               │ │
│  │    ingest (first_existing)      │   │  • Used for delete operations   │ │
│  │                                 │   │                                 │ │
│  │  Memory: ~96 MB                 │   │  Memory: ~96 MB                 │ │
│  │    (64MB block cache +          │   │    (64MB block cache +          │ │
│  │     2×16MB write buffers)       │   │     2×16MB write buffers)       │ │
│  └─────────────────────────────────┘   └─────────────────────────────────┘ │
│                                                                             │
│  Code: src/vector_index.rs, src/bucket_index.rs                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Insert Path

```
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                              INSERT FLOW                                 │
   └─────────────────────────────────────────────────────────────────────────┘

   Client
      │
      │  api.upsert(id=42, vector=[0.1, 0.2, ...])
      │
      ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │  1. ROUTE TO BUCKET                                                      │
   │                                                                          │
   │     RouterManager.RouteOrInit(vector)                                    │
   │       │                                                                  │
   │       ├── If no buckets exist: create bucket_0 with this vector as      │
   │       │   centroid                                                       │
   │       │                                                                  │
   │       └── Else: HNSW.query(vector, top_k=1) → nearest bucket_id         │
   │                                                                          │
   └──────────────────────────────────┬──────────────────────────────────────┘
                                      │
                                      ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │  2. DISPATCH TO WORKER                                                   │
   │                                                                          │
   │     shard = hash_ring.node_for(bucket_id)                               │
   │     worker_senders[shard].send(Upsert { bucket_id, vector })            │
   │                                                                          │
   └──────────────────────────────────┬──────────────────────────────────────┘
                                      │
                                      ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │  3. WORKER PERSISTS                                                      │
   │                                                                          │
   │     bucket_lock = bucket_locks.lock_for(bucket_id)                      │
   │     _guard = bucket_lock.lock()                                          │
   │                                                                          │
   │     Storage.put_chunk(bucket_id, vector)     ────▶  Walrus topic        │
   │     VectorIndex.put(id, vector)              ────▶  RocksDB             │
   │     BucketIndex.put(id, bucket_id)           ────▶  RocksDB             │
   │                                                                          │
   └──────────────────────────────────┬──────────────────────────────────────┘
                                      │
                                      ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │  4. UPDATE ROUTER STATE                                                  │
   │                                                                          │
   │     RouterManager.ApplyUpsert(bucket_id, vector)                        │
   │       │                                                                  │
   │       ├── Update running centroid mean                                  │
   │       ├── pending_updates++                                              │
   │       ├── persist_update() → Walrus "router_updates" topic              │
   │       │                                                                  │
   │       └── If pending_updates >= 1000:                                   │
   │             rebuild_router_and_persist()                                 │
   │             persist_snapshot() → Walrus "router_snapshot" topic         │
   │                                                                          │
   └─────────────────────────────────────────────────────────────────────────┘

   Code: src/service.rs:77-110, src/worker.rs:116-141, src/router_manager.rs:186-200
```

### Query Path

```
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                               QUERY FLOW                                 │
   └─────────────────────────────────────────────────────────────────────────┘

   Client
      │
      │  api.query(vector=[0.1, 0.2, ...], top_k=10, router_top_k=200)
      │
      ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │  1. ROUTER QUERY                                                         │
   │                                                                          │
   │     RouterManager.Query(vector, top_k=200)                              │
   │                                                                          │
   │     HNSW search:                                                         │
   │       • Quantize query vector (f32 → u8)                                │
   │       • Greedy descent through layers                                   │
   │       • Beam search at layer 0 with ef_search=1200-4000                 │
   │       • Return top-200 bucket IDs                                        │
   │                                                                          │
   │     → [bucket_12, bucket_47, bucket_89, bucket_103, ...]                │
   │                                                                          │
   └──────────────────────────────────┬──────────────────────────────────────┘
                                      │
                                      ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │  2. GROUP BY WORKER                                                      │
   │                                                                          │
   │     For each bucket_id:                                                  │
   │       shard = hash_ring.node_for(bucket_id)                             │
   │       requests[shard].push(bucket_id)                                   │
   │                                                                          │
   │     Worker 0: [12, 89, 201]                                             │
   │     Worker 1: [47, 156]                                                 │
   │     Worker 2: [103]                                                     │
   │                                                                          │
   └──────────────────────────────────┬──────────────────────────────────────┘
                                      │
                                      ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │  3. PARALLEL WORKER QUERIES                                              │
   │                                                                          │
   │     Send QueryRequest to each shard (parallel via join_all)             │
   │                                                                          │
   │     Each worker:                                                         │
   │     ┌───────────────────────────────────────────────────────────────┐   │
   │     │  for bucket_id in bucket_ids:                                  │   │
   │     │      if cache.get(bucket_id):                                  │   │
   │     │          scan_bucket_slice(cached_data, query_vec)             │   │
   │     │      else:                                                     │   │
   │     │          chunks = Storage.get_chunks(bucket_id)  ◄── Walrus   │   │
   │     │          cache.put_from_chunks(bucket_id, chunks)              │   │
   │     │          scan_bucket_chunks(chunks, query_vec)                 │   │
   │     │                                                                │   │
   │     │      // L2 distance computation (SIMD accelerated)            │   │
   │     │      for each vector in bucket:                                │   │
   │     │          dist = sqrt(sum((v[i] - q[i])^2))                    │   │
   │     │          candidates.push((id, dist))                           │   │
   │     │                                                                │   │
   │     │  candidates.sort_by_distance()                                 │   │
   │     │  return candidates.truncate(100)                               │   │
   │     └───────────────────────────────────────────────────────────────┘   │
   │                                                                          │
   └──────────────────────────────────┬──────────────────────────────────────┘
                                      │
                                      ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │  4. MERGE RESULTS                                                        │
   │                                                                          │
   │     all_results = flatten(worker_responses)                             │
   │     all_results.sort_by(|a, b| a.distance.cmp(&b.distance))             │
   │     return all_results.truncate(top_k)                                  │
   │                                                                          │
   │     → [(id: 1001, dist: 0.023), (id: 4521, dist: 0.031), ...]           │
   │                                                                          │
   └─────────────────────────────────────────────────────────────────────────┘

   Code: src/service.rs:44-56, 244-300
```

---

## Rebalancer

Autonomous background worker that splits oversized buckets:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              REBALANCER LOOP                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│     Every 500ms:                                                            │
│                                                                             │
│     1. refresh_sizes()                                                      │
│        └── For each bucket: count = wal.get_topic_entry_count(topic)       │
│                                                                             │
│     2. Find largest bucket                                                  │
│        └── max_bucket = argmax(bucket_sizes)                               │
│                                                                             │
│     3. If max_size > threshold (default: 2000):                            │
│                                                                             │
│        ┌─────────────────────────────────────────────────────────────┐     │
│        │                    SPLIT PROCESS                             │     │
│        ├─────────────────────────────────────────────────────────────┤     │
│        │                                                              │     │
│        │  a) Acquire bucket lock (serialize with workers)            │     │
│        │                                                              │     │
│        │  b) Load vectors from Walrus                                 │     │
│        │     chunks = Storage.get_chunks(bucket_id)                  │     │
│        │     vectors = decode_all_chunks(chunks)                     │     │
│        │                                                              │     │
│        │  c) K-means split (k=2)                                      │     │
│        │     ┌──────────────────────────────────────────────────┐    │     │
│        │     │  Init: farthest-pair centroids                   │    │     │
│        │     │  Iterate: max 8 iterations                       │    │     │
│        │     │    • Assign vectors to nearest centroid          │    │     │
│        │     │    • Recompute centroids                          │    │     │
│        │     │  Output: 2 new buckets with vectors               │    │     │
│        │     └──────────────────────────────────────────────────┘    │     │
│        │                                                              │     │
│        │  d) Persist new buckets                                      │     │
│        │     new_id_1 = allocate_bucket_id()                         │     │
│        │     new_id_2 = allocate_bucket_id()                         │     │
│        │     Storage.put_chunk(new_id_1, vectors_1)                  │     │
│        │     Storage.put_chunk(new_id_2, vectors_2)                  │     │
│        │                                                              │     │
│        │  e) Update BucketIndex for all moved vectors                │     │
│        │     bucket_index.put_batch(new_id_1, vector_ids_1)          │     │
│        │     bucket_index.put_batch(new_id_2, vector_ids_2)          │     │
│        │                                                              │     │
│        │  f) Retire old bucket                                        │     │
│        │     centroids.remove(old_bucket_id)                         │     │
│        │     Storage.put_bucket_meta(Retired)                        │     │
│        │                                                              │     │
│        │  g) Rebuild router with new centroids                       │     │
│        │     rebuild_router([old_id, new_id_1, new_id_2])            │     │
│        │                                                              │     │
│        │  h) Checkpoint old bucket (mark blocks as reclaimable)      │     │
│        │                                                              │     │
│        └─────────────────────────────────────────────────────────────┘     │
│                                                                             │
│     Else: sleep 500ms                                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

   Threshold: SATORI_REBALANCE_THRESHOLD env var (default: 2000)

   Code: src/rebalancer.rs:497-545 (loop), 259-359 (split)
```

---

## Algorithms

### HNSW Router Index

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                               HNSW Structure                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Layer 3:          ●───────────────────────────────●                       │
│                     │                               │                        │
│   Layer 2:          ●───────────●───────────────────●                       │
│                     │           │                   │                        │
│   Layer 1:          ●───●───────●───────●───────────●───●                   │
│                     │   │       │       │           │   │                    │
│   Layer 0:    ●─────●───●───●───●───●───●───●───●───●───●───●───●           │
│               ↑                                                             │
│             entry                                                           │
│             point                                                           │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Parameters (src/router.rs:30):                                            │
│     M = 24                  (max neighbors per node, layers > 0)           │
│     M0 = 48                 (max neighbors at layer 0)                      │
│     ef_construction = 180   (beam width during insertion)                   │
│                                                                             │
│   Query Parameters (src/router.rs:102-124):                                 │
│     ef_search scales with index size and top_k:                            │
│       • top_k=1:   64-128 (fast path for insert routing)                   │
│       • top_k>1:   1200-4000 (quality-focused for queries)                 │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Quantization (src/quantizer.rs):                                          │
│                                                                             │
│     f32 → u8 scalar quantization:                                          │
│       quantized = clamp((value - min) * 255 / (max - min), 0, 255)         │
│                                                                             │
│     Bounds computed with 0.1% padding to handle edge values                 │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Distance (src/router_hnsw.rs):                                            │
│                                                                             │
│     Cosine similarity over centered i8 vectors:                            │
│       centered[i] = u8[i] ^ 0x80   (convert u8 to signed)                  │
│       dist = 1 - dot(a, b) / (norm(a) * norm(b))                           │
│                                                                             │
│     SIMD: AVX2, AVX-512 kernels (src/router_hnsw.rs:722-871)               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### K-Means Clustering (Indexer)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            K-MEANS CLUSTERING                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Used for:                                                                  │
│     • Initial cluster formation: build_clusters(vectors, k)                │
│     • Bucket splitting: split_bucket_once(bucket) → 2 buckets              │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Initialization (for k=2 splits):                                          │
│                                                                             │
│     Farthest-pair method:                                                   │
│       1. c1 = vectors[0]                                                   │
│       2. c2 = argmax_v( distance(v, c1) )                                  │
│       3. c1 = argmax_v( distance(v, c2) )     (refine)                     │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Iteration (max 8 for splits, max 20 for initial build):                  │
│                                                                             │
│     for iter in 0..max_iters:                                               │
│       1. Assign each vector to nearest centroid                            │
│       2. Recompute centroids as mean of assigned vectors                   │
│       3. If no assignments changed → converged, break                      │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   SIMD Acceleration (src/indexer.rs:392-586):                              │
│                                                                             │
│     • k=2 special kernel: compute both distances in single pass            │
│     • k>=8 block kernel: process 8 centroids at once (SoA layout)          │
│     • AVX2 + FMA for L2 distance computation                               │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Fallback (src/indexer.rs:217-243):                                        │
│                                                                             │
│     If k=2 and one cluster ends up empty → split vectors in half           │
│     (guarantees split always produces 2 non-empty buckets)                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Recovery

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              STARTUP RECOVERY                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   1. Router Manager Recovery (src/router_manager.rs:364-391):              │
│                                                                             │
│      ┌────────────────────────────────────────────────────────────────┐    │
│      │  a) Load latest router_snapshot from Walrus                    │    │
│      │     → Recovers: bucket metadata (id, centroid, count)         │    │
│      │     → Also stores: updates_offset (how far updates were read) │    │
│      │                                                                │    │
│      │  b) Replay router_updates from updates_offset                  │    │
│      │     → Applies incremental centroid changes since snapshot     │    │
│      │                                                                │    │
│      │  c) Rebuild HNSW router from recovered centroids               │    │
│      └────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│   2. Workers: Start with empty caches                                       │
│      → Bucket data loaded on-demand from Walrus topics                     │
│                                                                             │
│   3. RocksDB indexes: Just reopen (already durable)                        │
│      → VectorIndex, BucketIndex                                            │
│                                                                             │
│   4. Rebalancer: Refresh sizes from Walrus topic entry counts              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Thread Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              THREAD ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  Main Thread                                                         │  │
│   │    • Creates SatoriDb                                                │  │
│   │    • Holds SatoriHandle for API calls                               │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  Router Manager Thread ("router-mgr")                               │  │
│   │    • Single thread, blocking channel receive                        │  │
│   │    • Handles: Query, RouteOrInit, ApplyUpsert, Flush, Stats        │  │
│   │    • Channel: crossbeam unbounded                                   │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  Worker Threads ("worker-0", "worker-1", ..., "worker-N")          │  │
│   │    • N = num_cpus (configurable via SatoriDbConfig.workers)        │  │
│   │    • Each runs Glommio LocalExecutor, CPU-pinned                   │  │
│   │    • Channel: async_channel bounded(1000)                          │  │
│   │    • Concurrency limit: 32 parallel query tasks per worker         │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  Rebalancer Thread ("rebalance-loop")                               │  │
│   │    • Glommio LocalExecutor (optional CPU pin)                      │  │
│   │    • Autonomous monitoring loop                                     │  │
│   │    • Delete command channel: async_channel bounded(1024)           │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Synchronization:                                                          │
│     • BucketLocks: per-bucket async Mutex (workers ↔ rebalancer)          │
│     • RoutingTable: Arc<RwLock> for sharing router snapshots              │
│     • Channels: message passing for all commands                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SATORI_REBALANCE_THRESHOLD` | 2000 | Split bucket when vector count exceeds |
| `SATORI_ROUTER_REBUILD_EVERY` | 1000 | Rebuild HNSW after N updates |
| `SATORI_WORKER_CACHE_BUCKETS` | 64 | Max buckets in worker LRU cache |
| `SATORI_WORKER_CACHE_BUCKET_MB` | 64 | Max MB per cached bucket |
| `SATORI_VECTOR_INDEX_PATH` | temp dir | Path for VectorIndex RocksDB |
| `SATORI_BUCKET_INDEX_PATH` | temp dir | Path for BucketIndex RocksDB |
| `WALRUS_DATA_DIR` | `./wal_files` | Walrus storage directory |
| `WALRUS_QUIET` | unset | Suppress Walrus debug output |
