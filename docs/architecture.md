# SatoriDB Architecture Documentation

## Overview

SatoriDB is a high-performance distributed vector database designed for approximate nearest neighbor (ANN) search at scale. It implements a **two-tier indexing strategy** combining K-means clustering with HNSW (Hierarchical Navigable Small World) graphs, backed by a custom write-ahead log (Walrus) for persistence.

**Key Design Principles:**
- **Sharded worker architecture** for horizontal scalability
- **Two-level ANN search**: Coarse-grained routing via HNSW on centroids, fine-grained search within buckets
- **SIMD-accelerated distance calculations** (AVX2 on x86_64)
- **Async I/O with io_uring** (Linux) for high-throughput persistence
- **LRU caching** at worker level to minimize disk access

---

## System Architecture

```
                                    USER QUERY
                            [query_vec: Vec<f32>, top_k: usize]
                                         |
                                         v
                        +=====================================+
                        |         ROUTER POOL                 |
                        |         (10% of cores)              |
                        |  +-------------------------------+  |
                        |  | crossbeam MPMC channel        |  |
                        |  | (unbounded queue)             |  |
                        |  +-------------------------------+  |
                        |                                     |
                        |  Arc<RwLock<Option<Router>>>        |
                        |  +-------------------------------+  |
                        |  | HNSW Index                    |  |
                        |  | - Index: u8 vectors (quantized)|  |
                        |  | - M=16, ef_construction=100   |  |
                        |  | - ef_search=max(k*20, 200)    |  |
                        |  | - Distance: L2 on u8          |  |
                        |  +-------------------------------+  |
                        |         Returns: Vec<u64>           |
                        |         (top-K bucket IDs)          |
                        +=====================================+
                                         |
                                         | bucket_ids [17, 42, 93, ...]
                                         v
                        +=====================================+
                        |     CONSISTENT HASH RING            |
                        |  +-------------------------------+  |
                        |  | Virtual nodes: 8 per shard    |  |
                        |  | Hash: DefaultHasher           |  |
                        |  | Sorted ring: Vec<(hash, node)>|  |
                        |  +-------------------------------+  |
                        |   bucket_id -> hash -> worker_id    |
                        +=====================================+
                                         |
                    +--------------------+--------------------+
                    |                    |                    |
                    v                    v                    v
        +-----------------+  +-----------------+  +-----------------+
        | WORKER SHARD 0  |  | WORKER SHARD 1  |  | WORKER SHARD N  |
        | (glommio)       |  | (glommio)       |  | (glommio)       |
        | Placement:      |  | Placement:      |  | Placement:      |
        | UNBOUND         |  | UNBOUND         |  | UNBOUND         |
        |                 |  |                 |  |                 |
        | async_channel   |  | async_channel   |  | async_channel   |
        | (bounded: 1000) |  | (bounded: 1000) |  | (bounded: 1000) |
        |                 |  |                 |  |                 |
        | +-------------+ |  | +-------------+ |  | +-------------+ |
        | | Executor    | |  | | Executor    | |  | | Executor    | |
        | | +---------+ | |  | | +---------+ | |  | | +---------+ | |
        | | |LRU Cache| | |  | | |LRU Cache| | |  | | |LRU Cache| | |
        | | |64 bkts  | | |  | | |64 bkts  | |  | | |64 bkts  | | |
        | | |8MB max  | | |  | | |8MB max  | |  | | |8MB max  | | |
        | | +----|----+ | |  | | +----|----+ | |  | | +----|----+ | |
        | |      |hit   | |  | |      |hit   | |  | |      |hit   | |
        | |      v      | |  | |      v      | |  | |      v      | |
        | |  [vectors]  | |  | |  [vectors]  | |  | |  [vectors]  | |
        | |      |      | |  | |      |      | |  | |      |      | |
        | |      |miss  | |  | |      |miss  | |  | |      |miss  | |
        | +------|------+ |  | +------|------+ |  | +------|------+ |
        |        |        |  |        |        |  |        |        |
        | +------|------+ |  | +------|------+ |  | +------|------+ |
        | | Storage     | |  | | Storage     | |  | | Storage     | |
        | | (topic API) | |  | | (topic API) | |  | | (topic API) | |
        | +------|------+ |  | +------|------+ |  | +------|------+ |
        +-----------------+  +-----------------+  +-----------------+
                    |                    |                    |
                    +--------------------+--------------------+
                                         |
                                         v
                        +=====================================+
                        |         WALRUS WAL                  |
                        |  +-------------------------------+  |
                        |  | Backend: FD (io_uring/pwrite) |  |
                        |  |     or mmap                   |  |
                        |  | Consistency: StrictlyAtOnce   |  |
                        |  | FsyncSchedule: NoFsync        |  |
                        |  +-------------------------------+  |
                        |                                     |
                        |  Topic-based storage:               |
                        |  - topic = "bucket_{id}"            |
                        |  - 10MB blocks                      |
                        |  - 100 blocks per file (1GB)        |
                        |  - rkyv serialization               |
                        |  - 8-byte length prefix             |
                        +=====================================+
                                         |
                                         v
                                   FILESYSTEM
                        wal_files/default/topic_bucket_*/
```

---

## Data Flow: Ingestion Pipeline

```
+---------------------+
|   Dataset Files     |
| gist_base.fvecs     |
| bigann_base.bvecs.gz|
+---------+-----------+
          |
          | read_batch(initial_batch_size)
          | BigANN: 100k vectors
          | GIST: 50k vectors
          v
+---------+-----------+
| FvecsReader/        |
| BvecsReader         |
| - Sequential IDs    |
| - f32 conversion    |
+---------+-----------+
          |
          | Vec<Vector>
          | Vector { id: u64, data: Vec<f32> }
          v
+=========================================+
|           INDEXER (K-means)             |
|  +-----------------------------------+  |
|  | Lloyd's algorithm (max 20 iters)  |  |
|  | k = 1000 (BigANN) / 2000 (GIST)   |  |
|  |                                   |  |
|  | Distance: l2_dist_sq_avx2()       |  |
|  | - AVX2: 8 floats/iteration        |  |
|  | - Fallback: scalar loop           |  |
|  |                                   |  |
|  | Rebalancing:                      |  |
|  | - Split if size > avg * 1.2       |  |
|  | - Recursive split (k=2)           |  |
|  | - Reindex IDs (0..N)              |  |
|  +-----------------------------------+  |
+=========================================+
          |
          | Vec<Bucket>
          | Bucket { id: u64, centroid: Vec<f32>, vectors: Vec<Vector> }
          |
          +----------+----------+
          |                     |
          v                     v
+-------------------+   +-------------------+
|   ROUTER BUILD    |   |  STORAGE PERSIST  |
| +---------------+ |   | +---------------+ |
| | Quantizer     | |   | | Serialize     | |
| | compute_bounds| |   | | rkyv::to_bytes| |
| | min/max ±0.01 | |   | |               | |
| +-------+-------+ |   | | Length prefix | |
|         |         |   | | (8 bytes)     | |
|         v         |   | +-------+-------+ |
| +-------+-------+ |   |         |         |
| | Quantize      | |   |         v         |
| | f32 -> u8     | |   | +-------+-------+ |
| | 255 levels    | |   | | WAL append    | |
| +-------+-------+ |   | | topic:        | |
|         |         |   | | bucket_{id}   | |
|         v         |   | +---------------+ |
| +-------+-------+ |   +-------------------+
| | HNSW insert   | |
| | M=16          | |
| | ef_const=100  | |
| +---------------+ |
+-------------------+
          |
          | Router ready
          v
+=========================================+
|       STREAMING INGESTION               |
|  (remaining vectors up to max_vectors)  |
|                                         |
|  Loop:                                  |
|    read_batch(stream_batch_size)        |
|    BigANN/GIST: 100k vectors            |
|                                         |
|    For each vector:                     |
|      bucket_id = Router.query(vec, 1)   |
|      shard = HashRing.node_for(bucket)  |
|      send WorkerMessage::Ingest         |
+=========================================+
          |
          v
+=========================================+
|          WORKER INGEST BUFFER           |
|  +-----------------------------------+  |
|  | HashMap<u64, Vec<Vector>>         |  |
|  | - Key: bucket_id                  |  |
|  | - Value: buffered vectors         |  |
|  |                                   |  |
|  | FLUSH_THRESHOLD = 10,000 vectors  |  |
|  |                                   |  |
|  | On threshold:                     |  |
|  |   Bucket::new(id, vec![])         |  |
|  |   bucket.vectors = mem::take()    |  |
|  |   Storage.put_chunk(&bucket)      |  |
|  +-----------------------------------+  |
+=========================================+
          |
          v
+-------------------+
|  WorkerMessage::  |
|  Flush            |
| (before queries)  |
+-------------------+
```

---

## Data Flow: Query Pipeline

```
+---------------------+
|   Query Vector      |
| Vec<f32>            |
| top_k: usize        |
+---------+-----------+
          |
          | crossbeam::send(RouterTask)
          v
+=========================================+
|         ROUTER POOL THREAD              |
|  (one of N threads, ~10% cores)         |
|                                         |
|  Router.query(query_vec, router_top_k): |
|    router_top_k = 50 (BigANN)           |
|                  400 (GIST)             |
|                                         |
|  Steps:                                 |
|  1. Quantizer.quantize(query_vec)       |
|     -> Vec<u8>                          |
|                                         |
|  2. HNSW.search(&q_vec, top_k, ef)      |
|     ef_search = max(top_k * 20, 200)    |
|     -> Vec<Neighbor>                    |
|                                         |
|  3. Extract bucket IDs                  |
|     -> Vec<u64>                         |
+=========================================+
          |
          | oneshot::send(bucket_ids)
          v
+---------+-----------+
| Main thread         |
| aggregates response |
+---------+-----------+
          |
          | For each bucket_id:
          |   shard = HashRing.node_for(bucket_id)
          |   requests[shard].push(bucket_id)
          |
          | HashMap<shard_id, Vec<bucket_id>>
          v
+=========================================+
|      PARALLEL WORKER DISPATCH           |
|                                         |
|  For each (shard, bucket_ids):          |
|    QueryRequest {                       |
|      query_vec,                         |
|      bucket_ids,                        |
|      respond_to: oneshot                |
|    }                                    |
|    send to worker[shard]                |
+=========================================+
          |
          +----------+----------+----------+
          |          |          |          |
          v          v          v          v
    +--------+  +--------+  +--------+  +--------+
    |Worker 0|  |Worker 1|  |Worker 2|  |Worker N|
    +--------+  +--------+  +--------+  +--------+
          |          |          |          |
          v          v          v          v
+=========================================+
|       EXECUTOR.query()                  |
|  (per worker, async)                    |
|                                         |
|  For each bucket_id in request:         |
|                                         |
|  +-----------------------------------+  |
|  | 1. Cache lookup                   |  |
|  |    LRU.get(bucket_id)             |  |
|  |    if hit -> CachedBucket         |  |
|  +-----------------------------------+  |
|           |                             |
|           | miss                        |
|           v                             |
|  +-----------------------------------+  |
|  | 2. Load from WAL                  |  |
|  |    Storage.get_chunks(bucket_id)  |  |
|  |      topic = "bucket_{id}"        |  |
|  |      Walrus.batch_read_for_topic  |  |
|  |        -> Vec<Entry>              |  |
|  |                                   |  |
|  |    For each Entry:                |  |
|  |      read 8-byte length           |  |
|  |      rkyv::check_archived_root    |  |
|  |      deserialize vectors          |  |
|  |                                   |  |
|  |    CachedBucket {                 |  |
|  |      vectors,                     |  |
|  |      byte_size                    |  |
|  |    }                              |  |
|  +-----------------------------------+  |
|           |                             |
|           v                             |
|  +-----------------------------------+  |
|  | 3. Cache update (if size <= 8MB)  |  |
|  |    LRU.put(bucket_id, cached)     |  |
|  +-----------------------------------+  |
|           |                             |
|           v                             |
|  +-----------------------------------+  |
|  | 4. Compute distances              |  |
|  |    For each vector in bucket:     |  |
|  |      dist = l2_distance_f32(      |  |
|  |        &vector.data,              |  |
|  |        query_vec                  |  |
|  |      )                            |  |
|  |      candidates.push((id, dist))  |  |
|  +-----------------------------------+  |
|                                         |
|  Return: Vec<(u64, f32)>                |
+=========================================+
          |
          | oneshot::send(candidates)
          v
+---------+-----------+
| Main thread         |
| futures::join_all() |
+---------+-----------+
          |
          | Collect all worker responses
          v
+=========================================+
|       AGGREGATE & SORT                  |
|                                         |
|  all_results = Vec::new()               |
|  for response in responses:             |
|    all_results.extend(candidates)       |
|                                         |
|  all_results.sort_by(|a,b|              |
|    a.1.partial_cmp(&b.1)                |
|  )                                      |
|                                         |
|  all_results.truncate(top_k)            |
+=========================================+
          |
          v
+---------+-----------+
| Vec<(u64, f32)>     |
| (vector_id, dist)   |
+---------------------+
```

---

## Threading Model

```
+==============================================================================+
|                        PROCESS: satoridb                                    |
+==============================================================================+

    MAIN THREAD
    -----------
    - Dataset detection
    - Initial indexing (Indexer::build_clusters)
    - Router construction (HNSW build)
    - Thread spawning
    - Benchmark orchestration (async via block_on)


    ROUTER POOL (10% of SATORI_CORES, min 1)
    ------------
    Thread 0:  |  Thread 1:  |  ...  |  Thread N:
    OS thread  |  OS thread  |       |  OS thread
    BLOCKING   |  BLOCKING   |       |  BLOCKING
               |             |       |
    loop {
      task = router_rx.recv()  <--- crossbeam MPMC (unbounded)
      router.read().query()    <--- Arc<RwLock<Option<Router>>>
      task.respond_to.send()
    }


    WORKER SHARDS (90% of SATORI_CORES, min 1)
    --------------
    Worker 0:      |  Worker 1:      |  ...  |  Worker N:
    OS thread      |  OS thread      |       |  OS thread
    glommio        |  glommio        |       |  glommio
    Placement:     |  Placement:     |       |  Placement:
    UNBOUND        |  UNBOUND        |       |  UNBOUND
                   |                 |       |
    LocalExecutor::run(async {
      run_worker(id, receiver, wal)
        loop {
          msg = receiver.recv().await  <--- async_channel (bounded: 1000)
          match msg {
            Query => executor.query().await
            Ingest => buffer, maybe flush
            Flush => force flush
          }
        }
    })


    SYNCHRONIZATION POINTS
    ----------------------
    1. Router build:
       main thread builds Router
       workers spawned BEFORE router is ready
       router_shared initially None, set after indexing

    2. Flush barrier:
       main sends WorkerMessage::Flush to all workers
       waits on Vec<oneshot::Receiver<()>>
       uses futures::join_all (block_on)

    3. Query aggregation:
       main sends QueryRequest to workers
       collects Vec<oneshot::Receiver<Result>>
       uses futures::join_all (block_on)

    4. Shutdown:
       drop(router_tx)           -> router threads exit
       for s in senders: s.close() -> workers exit
       join all handles
```

---

## Memory Layout

```
+==============================================================================+
|                              HEAP MEMORY                                    |
+==============================================================================+

ARC-SHARED ACROSS THREADS:
--------------------------
Arc<Walrus>
  └─> Walrus {
        topics: HashMap<String, TopicState>
        consistency: ReadConsistency
        fsync_schedule: FsyncSchedule
        backend: FdBackend or MmapBackend
      }

Arc<RwLock<Option<Router>>>
  └─> Router {
        index: Hnsw<'static, u8, DistL2>
          └─> ~(M * num_centroids * 8) bytes for graph
          └─> ~(dim * num_centroids) bytes for u8 vectors
        quantizer: Quantizer { min: f32, max: f32 }
      }


PER-ROUTER-THREAD (10% cores):
-------------------------------
Stack:
  - RouterTask (transient)
  - Read guard on Arc<RwLock<Router>>

Heap:
  - None (stateless worker)


PER-WORKER-THREAD (90% cores):
-------------------------------
Stack:
  - glommio executor stack

Heap:
  Storage (clone of Arc<Walrus>)

  WorkerCache {
    LruCache<u64, CachedBucket> {
      max_buckets: 64
      bucket_max_bytes: 8MB
      total_memory: ~512MB max per worker
    }
  }

  ingest_buffers: HashMap<u64, Vec<Vector>>
    └─> per-bucket buffers, flushed at 10k vectors
    └─> transient: ~(10k * sizeof(Vector)) per bucket

  Executor {
    storage: Storage
    cache: RefCell<WorkerCache>
  }


DISK (WAL):
-----------
wal_files/default/
  topic_bucket_0/
    000000.wal (1GB = 100 blocks * 10MB)
      [block 0: 10MB]
        [entry 0: 8 bytes length + rkyv Bucket]
        [entry 1: ...]
      [block 1: 10MB]
      ...
    offsets.meta (read checkpoint)
  topic_bucket_1/
  ...

Bucket serialization (rkyv):
  - id: u64 (8 bytes)
  - centroid: Vec<f32> (4 * dim bytes + overhead)
  - vectors: Vec<Vector>
    - per vector: 8 bytes (id) + 4*dim bytes (data) + overhead
  - Total: ~(num_vectors * (8 + 4*dim)) bytes per bucket
```

---

## Module Descriptions

### 1. main.rs - System Orchestrator
**Lines of Code:** 556
**Location:** `src/main.rs`

**Purpose:** Entry point that bootstraps the entire system and runs benchmarks.

**Key Components:**
- `ConsistentHashRing` (lines 60-92): Maps bucket IDs to worker shards
  - Virtual nodes: 8 per shard (line 68)
  - Hash function: `DefaultHasher` from std
  - Lookup: Binary search for closest hash (lines 85-90)

- `RouterTask` (lines 54-58): Query request envelope
  - `query_vec`: The query vector
  - `top_k`: Number of bucket IDs to return
  - `respond_to`: oneshot channel for response

- `DatasetConfig` (lines 46-52): Dataset metadata
  - `kind`: Bigann or Gist
  - `base_path`, `query_path`, `gnd_path`: File paths
  - `max_vectors`: Ingestion limit

**Data Flow (run_benchmark_mode, lines 286-555):**
1. Detect dataset (lines 293-299)
2. Read initial batch (lines 330-340)
3. Cluster with Indexer (line 347)
4. Build Router with quantized centroids (lines 350-360)
5. Stream remaining batches (lines 365-399)
   - Route each vector to bucket via Router.query
   - Hash bucket_id to worker shard
   - Send WorkerMessage::Ingest
6. Flush all workers (lines 400-413)
7. Execute queries (lines 425-551)
   - Route query to buckets via Router
   - Dispatch to workers via hash ring
   - Aggregate and sort results
   - Calculate recall@10 if ground truth exists

**Configuration:**
- `SATORI_CORES`: Total cores (default: num_cpus)
- Router pool: 10% (line 198)
- Worker shards: 90% (line 200)
- Router channel: unbounded MPMC (line 236)
- Worker channel: bounded 1000 (line 216)

---

### 2. indexer.rs - K-Means Clustering Engine
**Lines of Code:** 182
**Location:** `src/indexer.rs`

**Purpose:** Partitions vectors into balanced buckets using K-means clustering.

**Algorithm (build_clusters, lines 12-118):**
- Lloyd's algorithm with random initialization
- Max iterations: 20 (line 38)
- Assignment: Find nearest centroid by L2 distance (lines 47-63)
- Update: Compute new centroids as mean of assigned vectors (lines 77-87)
- Early stopping: If no assignments change (lines 72-74)

**SIMD Acceleration (l2_dist_sq_avx2, lines 155-181):**
- AVX2: 8 floats per iteration (lines 160-168)
- Horizontal sum of AVX register (lines 170-173)
- Remainder handled by scalar loop (lines 176-179)
- Speedup: ~4x on aligned data

**Rebalancing (lines 124-142):**
- Threshold: avg * 1.2 (line 133)
- Split strategy: Recursive K-means with k=2 (lines 120-122)
- Reindexing: Assign sequential IDs 0..N after rebalancing (lines 109-117)

**Edge Cases:**
- Empty centroids: Re-initialize from random vector (lines 82-86)
- Fewer than k vectors: Sample with replacement (lines 30-36)

**Output:** `Vec<Bucket>` with dense IDs and balanced sizes

---

### 3. router.rs - Centroid-Based Query Router
**Lines of Code:** 33
**Location:** `src/router.rs`

**Purpose:** Quickly identifies candidate buckets for a query using HNSW on centroids.

**Implementation:**
- **Index:** `Hnsw<'static, u8, DistL2>` from hnsw_rs crate
- **Parameters (line 13):**
  - M=16: Connections per layer
  - max_elements=10,000: Capacity hint
  - max_layer=16: Max levels in hierarchy
  - ef_construction=100: Search quality during build

**Key Functions:**
- `add_centroid(id, vector)` (lines 17-20):
  - Quantize f32 vector to u8
  - Insert into HNSW with bucket ID as label

- `query(vector, top_k)` (lines 22-31):
  - Quantize query vector
  - ef_search = max(top_k * 20, 200) (line 25)
  - HNSW search returns `Vec<Neighbor>`
  - Extract bucket IDs (line 28)

**Why Quantization:**
- Memory: 4x reduction (f32 -> u8)
- Speed: Smaller vectors, faster distance
- Recall: <5% loss for coarse routing (fine search in executor compensates)

**Routing Strategy:**
- Two-level ANN: Coarse (HNSW on centroids) + Fine (L2 in buckets)
- Router top_k >> query top_k to ensure coverage

---

### 4. worker.rs - Async Message Processor
**Lines of Code:** 100
**Location:** `src/worker.rs`

**Purpose:** Per-shard async event loop handling queries and ingestion.

**Message Types (lines 11-26):**
- `Query(QueryRequest)`: Contains query_vec, bucket_ids, respond_to channel
- `Ingest{bucket_id, vectors}`: Batch of vectors for a bucket
- `Flush{respond_to}`: Force-flush signal (used before queries)

**Ingestion Strategy (lines 47-62):**
- Buffer: `HashMap<u64, Vec<Vector>>` (line 34)
- Flush threshold: 10,000 vectors (line 37)
- On threshold:
  - Create `Bucket::new(bucket_id, vec![])`
  - Move vectors with `mem::take(buffer)`
  - Persist via `Storage.put_chunk(&bucket)`

**Query Strategy (lines 41-45):**
- Delegate to `Executor.query(query_vec, bucket_ids, 100)`
- Executor handles cache + WAL reads
- Return candidates via oneshot channel

**Graceful Shutdown (lines 84-96):**
- Channel close (receiver drops) exits loop
- Flush all remaining buffers to WAL
- Log any errors (no panic)

**Concurrency:**
- Runtime: glommio LocalExecutor (line 223 in main.rs)
- Placement: Unbound (no core pinning, line 221 in main.rs)
- Backpressure: async_channel with 1000 capacity

---

### 5. executor.rs - Query Executor with LRU Cache
**Lines of Code:** 142
**Location:** `src/executor.rs`

**Purpose:** Executes queries against buckets with local caching.

**Components:**
- `WorkerCache` (lines 20-43):
  - LRU: `LruCache<u64, CachedBucket>`
  - max_buckets: 64 (configurable)
  - bucket_max_bytes: 8MB (configurable)
  - Eviction: LRU policy, rejects buckets >8MB

- `CachedBucket` (lines 13-17):
  - `vectors`: Deserialized `Vec<Vector>`
  - `byte_size`: Total bytes (for eviction decision)

**Query Flow (lines 87-128):**
1. For each bucket_id:
   - Check cache (lines 97-105)
   - On miss: `load_bucket()` from WAL (lines 108-116)
   - Compute L2 distance for all vectors (lines 99-104, 110-115)
   - Cache loaded bucket if size <= 8MB (line 116)
2. Sort candidates by distance (line 120)
3. Truncate to top_k (lines 123-125)

**load_bucket() (lines 53-85):**
- Fetch all chunks for bucket from WAL (line 54)
- For each chunk:
  - Read 8-byte length prefix (lines 63-64)
  - Validate with `rkyv::check_archived_root` (line 71)
  - Deserialize vectors (lines 72-77)
- Track total byte size for cache decision (line 59)

**Distance Calculation (lines 131-141):**
- Scalar L2 distance (not SIMD-accelerated)
- sqrt of sum of squared differences
- Dimension mismatch check (lines 99-101, 110-112)

---

### 6. storage.rs - WAL Abstraction Layer
**Lines of Code:** 93
**Location:** `src/storage.rs`

**Purpose:** Provides topic-based persistence interface over Walrus WAL.

**Key Concepts:**
- **Topic:** `"bucket_{id}"` (line 56)
- **Chunk:** Single serialized bucket append
- **Format:** 8-byte length + rkyv-serialized Bucket (lines 68-70)

**API:**
- `put_chunk(bucket)` (lines 60-75):
  - Serialize with `rkyv::to_bytes::<_, 1024>` (line 64)
  - Add length prefix (8 bytes LE)
  - Append to topic via `Walrus.append_for_topic`

- `get_chunks(bucket_id)` (lines 78-92):
  - Query topic size via `Walrus.get_topic_size`
  - Batch read all entries (line 85-88)
  - Return raw `Vec<Vec<u8>>`

**Data Types (lines 10-43):**
- `Vector`: `{ id: u64, data: Vec<f32> }` with rkyv traits
- `Bucket`: `{ id: u64, centroid: Vec<f32>, vectors: Vec<Vector> }` with rkyv traits

**Serialization:**
- rkyv: Zero-copy (in theory), but executor copies on deserialization
- Buffer hint: 1024 bytes (line 64)
- Unique filename: UUID-based (line 62)

---

### 7. quantizer.rs - Float32 to Uint8 Compression
**Lines of Code:** 47
**Location:** `src/quantizer.rs`

**Purpose:** Linearly quantizes f32 vectors to u8 for HNSW router.

**Algorithm (quantize, lines 31-45):**
```
range = max - min
scale = 255.0 / range  (or 0 if range ~= 0)
normalized = (value - min) * scale
quantized = clamp(normalized, 0, 255) as u8
```

**compute_bounds() (lines 12-29):**
- Scan all vectors for global min/max (lines 16-25)
- Add padding: min - 0.01, max + 0.01 (line 28)
- Purpose: Avoid boundary clipping

**Edge Cases:**
- Degenerate range (min ≈ max): scale=0, all values map to 0 (lines 33-37)
- NaN/Inf: Not handled (undefined behavior)

**Use Case:**
- Router operates entirely in u8 space
- 4x memory savings (128 dims * 1 byte vs 128 dims * 4 bytes)
- Faster L2 distance (smaller vectors fit in cache)

**Trade-offs:**
- Quantization error: ~0.4% per dimension (255 levels)
- Acceptable for coarse routing (recall loss compensated by router_top_k >> query_top_k)

---

### 8. Walrus WAL (storage/wal/) - Write-Ahead Log
**Lines of Code:** ~8000 (including runtime)
**Location:** `src/storage/lib.rs`, `src/storage/wal/`

**Purpose:** High-performance persistent log with topic-based organization.

**Architecture:**
- **Topics:** Independent append-only streams (namespace per bucket)
- **Blocks:** 10MB fixed-size chunks
- **Files:** 100 blocks per file (1GB files)
- **Backends:** FD (pread/pwrite, io_uring on Linux) or mmap

**Key Components (from lib.rs):**
- `Walrus`: Main handle, constructed with consistency/fsync settings
- `Entry`: `{ data: Vec<u8> }` - opaque payload
- `ReadConsistency`:
  - `StrictlyAtOnce`: Every checkpoint persisted
  - `AtLeastOnce { persist_every: N }`: Checkpoint every N reads
- `FsyncSchedule`:
  - `NoFsync`: No durability (benchmark mode)
  - `SyncEach`: O_SYNC flag (every write is durable)
  - `Milliseconds(N)`: Background fsync thread every N ms

**Operations:**
- `append_for_topic(topic, data)`: Single append
- `batch_append_for_topic(topic, entries)`: Atomic batch (up to 2000 entries)
- `read_next(topic, checkpoint)`: Sequential read (checkpoint=true advances offset)
- `batch_read_for_topic(topic, max_bytes, checkpoint)`: Batch read with byte limit

**Performance Features:**
- io_uring: Batch I/O on Linux (FD backend only)
- mmap: Direct memory access (all platforms)
- O_SYNC: Synchronous writes when FsyncSchedule::SyncEach
- Persistent offsets: Read positions survive restarts

**Storage Layout:**
```
wal_files/
  default/                  # Instance key (WALRUS_INSTANCE_KEY)
    topic_bucket_0/
      000000.wal            # 1GB file (100 * 10MB blocks)
      offsets.meta          # Read checkpoint
    topic_bucket_1/
    ...
```

**Configuration in main.rs (lines 206-209):**
- `ReadConsistency::StrictlyAtOnce`
- `FsyncSchedule::NoFsync`

---

### 9. Dataset Readers (fvecs.rs, bvecs.rs, gnd.rs)

**fvecs.rs - GIST1M Reader:**
**Lines of Code:** 51
**Location:** `src/fvecs.rs`

- Format: `[dim:u32le][f32 * dim]` repeated
- BufReader for I/O efficiency (line 15)
- Sequential ID assignment (lines 9, 44-45)
- Batch reading with EOF handling (lines 28-32)

**bvecs.rs - BigANN Reader:**
**Lines of Code:** 61
**Location:** `src/bvecs.rs`

- Format: `[dim:u32le][u8 * dim]` repeated (gzip compressed)
- GzDecoder wrapper (line 15)
- Upcast u8 -> f32 (line 53)
- Validates dim=128 for SIFT (lines 40-46)

**gnd.rs - Ground Truth Reader:**
**Lines of Code:** (not read in detail, but exists)
**Location:** `src/gnd.rs`

- Reads tar.gz with true neighbors per query
- Used for recall@K calculation in benchmarks
- Filters by max_vectors to handle truncated datasets

---

## Known Limitations

### Architectural
1. **No Dynamic Rebalancing:** Bucket splits only during initial indexing
2. **Static Shard Count:** Cannot add/remove workers at runtime
3. **No Replication:** Single WAL, no HA
4. **Cache Invalidation:** Workers don't know when other workers update buckets
5. **No Backpressure on Router:** Unbounded MPMC channel can OOM under load

### Durability
6. **NoFsync in Benchmark:** Data loss on crash (acceptable for benchmarks)
7. **No WAL Compaction:** Topics grow unbounded (deleted vectors not reclaimed)
8. **No Checkpointing:** WAL reads scan entire topic (inefficient for large buckets)

### Performance
9. **Executor Distance:** Scalar L2 (no SIMD, unlike indexer)
10. **Router Quantization:** Fixed 255 levels (no adaptive quantization)
11. **Cache Size:** Fixed 64 buckets, 8MB limit (no auto-tuning)
12. **Over-fetching:** Router returns top_k buckets, but may need fewer

### Operational
13. **No Schema Evolution:** Changing vector dimensions requires full rebuild
14. **No Monitoring:** No metrics, logs, or observability
15. **No API:** Benchmark-only, no HTTP/gRPC interface
16. **No Authentication:** No security layer

---

## Configuration Reference

### Environment Variables
| Variable | Default | Purpose |
|----------|---------|---------|
| `SATORI_CORES` | `num_cpus::get()` | Total cores for router + workers |
| `SATORI_RUN_BENCH` | unset (disabled) | Enable benchmark mode |
| `WALRUS_DATA_DIR` | `./wal_files` | WAL storage location |
| `WALRUS_INSTANCE_KEY` | `default` | WAL namespace |
| `WALRUS_QUIET` | unset | Set to `1` to suppress WAL logs |

### Code Constants

**main.rs:**
- Line 198: `router_pool = (satori_cores * 0.1).max(1)`
- Line 200: `executor_shards = (satori_cores - router_pool).max(1)`
- Line 233: `virtual_nodes = 8` (consistent hash ring)
- Line 236: `router_tx = unbounded()` (MPMC channel)
- Line 216: `async_channel::bounded(1000)` (worker backpressure)


---

## Glossary

| Term | Definition |
|------|------------|
| **Bucket** | Cluster of vectors with shared centroid (output of K-means) |
| **Centroid** | Mean vector of all vectors in a bucket |
| **Topic** | WAL namespace (format: `bucket_{id}`) |
| **Chunk** | Serialized bucket appended to topic |
| **Router** | HNSW index over centroids for coarse-grained search |
| **Worker** | Async shard handling queries and ingestion for subset of buckets |
| **Executor** | Query engine with LRU cache (one per worker) |
| **Quantization** | f32 to u8 linear compression for HNSW router |
| **Virtual Node** | Hash ring entry for consistent hashing (8 per worker) |
| **ef_search** | HNSW parameter controlling search quality (higher = more accurate) |
| **M** | HNSW parameter for graph connectivity (higher = more memory, faster search) |
| **LRU Cache** | Least Recently Used cache eviction policy |
| **HNSW** | Hierarchical Navigable Small World graph (ANN index) |
| **AVX2** | x86_64 SIMD instruction set (8 floats per operation) |
| **io_uring** | Linux async I/O interface (batch submissions) |
| **rkyv** | Zero-copy serialization library |
| **glommio** | Thread-per-core async runtime |

---

## Component Interaction Matrix

| Component | Depends On | Provides To | Communication |
|-----------|------------|-------------|---------------|
| **main.rs** | All modules | - | Thread spawn, async channels |
| **indexer.rs** | storage.rs (Vector, Bucket) | main.rs | Synchronous function call |
| **router.rs** | quantizer.rs | main.rs, router pool | Shared via Arc<RwLock> |
| **quantizer.rs** | - | router.rs | Direct struct usage |
| **worker.rs** | executor.rs, storage.rs | main.rs | async_channel (1000 cap) |
| **executor.rs** | storage.rs | worker.rs | Direct async function call |
| **storage.rs** | wal/ (Walrus) | executor.rs, worker.rs, main.rs | Async API |
| **wal/** | OS (io_uring, mmap) | storage.rs | Blocking/async API |
| **fvecs.rs** | storage.rs (Vector) | main.rs | Synchronous iterator |
| **bvecs.rs** | storage.rs (Vector) | main.rs | Synchronous iterator |
| **gnd.rs** | tar, flate2 | main.rs | Synchronous reader |

---
