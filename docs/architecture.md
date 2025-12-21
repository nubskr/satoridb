# SatoriDB Architecture

SatoriDB is an embedded vector database for approximate nearest neighbor (ANN) search. It runs entirely in-process—no external services required.

## Core Idea: Two-Level Search

Instead of scanning all vectors, SatoriDB groups vectors into **buckets** and searches in two stages:

```
                              ┌───────────────┐
                              │  Query Vector │
                              └───────┬───────┘
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │           ROUTER              │
                      │  ┌─────────────────────────┐  │
                      │  │  HNSW index over bucket │  │
                      │  │       centroids         │  │
                      │  └─────────────────────────┘  │
                      │       O(log B) lookup         │
                      └───────────────┬───────────────┘
                                      │
                      ┌───────────────┼───────────────┐
                      │               │               │
                      ▼               ▼               ▼
               ┌───────────┐   ┌───────────┐   ┌───────────┐
               │  Bucket   │   │  Bucket   │   │  Bucket   │
               │    #42    │   │   #107    │   │   #891    │
               │ ┌───────┐ │   │ ┌───────┐ │   │ ┌───────┐ │
               │ │vectors│ │   │ │vectors│ │   │ │vectors│ │
               │ └───────┘ │   │ └───────┘ │   │ └───────┘ │
               └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
                     │               │               │
                     │   Exhaustive L2 scan          │
                     └───────────────┼───────────────┘
                                     │
                                     ▼
                         ┌───────────────────┐
                         │   Merge & Rank    │
                         │   by distance     │
                         └─────────┬─────────┘
                                   │
                                   ▼
                           ┌─────────────┐
                           │ Top-K Results│
                           └─────────────┘
```

Each bucket has a **centroid** (mean of its vectors). The router maintains an HNSW index over centroids. Queries first find nearby centroids (fast), then scan those buckets (accurate).

---

## System Overview

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                              SATORIDB PROCESS                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ┌────────────────────────────────────────────────────────────────────────┐  ║
║  │                           SatoriHandle                                 │  ║
║  │                 query() │ upsert() │ flush() │ stats()                 │  ║
║  └──────────────────────────────────┬─────────────────────────────────────┘  ║
║                                     │                                        ║
║            ┌────────────────────────┴────────────────────────┐               ║
║            │                                                 │               ║
║            │ crossbeam channel                async_channel  │               ║
║            ▼                                                 ▼               ║
║  ┌──────────────────────┐                 ┌───────────────────────────────┐  ║
║  │    ROUTER MANAGER    │                 │           WORKERS             │  ║
║  │     (1 thread)       │                 │      (N = num_cpus)           │  ║
║  │                      │                 │                               │  ║
║  │  ┌────────────────┐  │                 │  ┌─────────┐  ┌─────────┐     │  ║
║  │  │ BucketMeta Map │  │                 │  │Worker 0 │  │Worker 1 │ ... │  ║
║  │  │ id → centroid  │  │                 │  │─────────│  │─────────│     │  ║
║  │  └────────────────┘  │                 │  │Executor │  │Executor │     │  ║
║  │  ┌────────────────┐  │                 │  │ Cache   │  │ Cache   │     │  ║
║  │  │ Router (HNSW)  │  │                 │  │Storage  │  │Storage  │     │  ║
║  │  └────────────────┘  │                 │  └─────────┘  └─────────┘     │  ║
║  │  ┌────────────────┐  │                 │                               │  ║
║  │  │   Quantizer    │  │                 │  ┌─────────────────────────┐  │  ║
║  │  │   f32 → u8     │  │                 │  │  ConsistentHashRing     │  │  ║
║  │  └────────────────┘  │                 │  │  bucket_id → worker     │  │  ║
║  └──────────────────────┘                 │  └─────────────────────────┘  │  ║
║            │                              └───────────────┬───────────────┘  ║
║            │                                              │                  ║
║            └──────────────────────┬───────────────────────┘                  ║
║                                   │                                          ║
║                                   ▼                                          ║
║  ┌────────────────────────────────────────────────────────────────────────┐  ║
║  │                            WALRUS (WAL)                                │  ║
║  │                                                                        │  ║
║  │  ┌──────────────┐ ┌──────────────┐ ┌──────────┐ ┌──────────┐          │  ║
║  │  │   router_    │ │   router_    │ │ bucket_0 │ │ bucket_1 │  ...     │  ║
║  │  │   snapshot   │ │   updates    │ │          │ │          │          │  ║
║  │  └──────────────┘ └──────────────┘ └──────────┘ └──────────┘          │  ║
║  └────────────────────────────────────────────────────────────────────────┘  ║
║                                   │                                          ║
║                                   ▼                                          ║
║                            ┌────────────┐                                    ║
║                            │    DISK    │                                    ║
║                            └────────────┘                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

**Source references:**
- `SatoriDb` struct: `embedded.rs:35-41`
- `SatoriHandle`: `service.rs:14-18`
- Router manager spawn: `embedded.rs:77`
- Worker spawn loop: `embedded.rs:57-72`

---

## Components

### SatoriDb & SatoriHandle

`SatoriDb::start(config)` spawns the system:
- 1 router manager thread
- N worker threads (default: `num_cpus::get()`)
- Each worker runs a Glommio async executor pinned to a CPU

`SatoriHandle` is the cloneable API for queries and upserts.

```rust
// embedded.rs:49-100
pub fn start(cfg: SatoriDbConfig) -> Result<Self> {
    // Spawn N workers with Glommio executors
    for i in 0..workers {
        let handle = thread::spawn(move || {
            LocalExecutorBuilder::new(Placement::Fixed(pin_cpu))
                .make()
                .run(run_worker(
                    i,
                    receiver,
                    wal_clone,
                    vector_index.clone(),
                    bucket_index.clone(),
                    bucket_locks.clone(),
                ));
        });
    }
    // Spawn router manager
    let router_handle = spawn_router_manager(cfg.wal.clone(), router_rx);
    // Create consistent hash ring for bucket→worker mapping
    let ring = ConsistentHashRing::new(workers, cfg.virtual_nodes);
    ...
}
```

### Router Manager

Single-threaded, owns the routing state. Handles commands via crossbeam channel.

**State** (`router_manager.rs:135-142`):
```rust
struct RouterManager {
    wal: Arc<Walrus>,
    buckets: HashMap<u64, BucketMeta>,  // id → (centroid, count)
    router: Option<Router>,              // HNSW index over centroids
    quantizer: Option<Quantizer>,        // f32 → u8 conversion
    pending_updates: u64,
    rebuild_every: u64,                  // default: 1000
}
```

**Centroid update** (`router_manager.rs:271-286`):
```rust
// Running mean: new_centroid = (old * count + vector) / (count + 1)
let new_count_f = new_count as f32;
for (c, &x) in entry.centroid.iter_mut().zip(vector.iter()) {
    *c = (*c * prev_count_f + x) / new_count_f;
}
```

**Persistence:**
- `router_snapshot`: Full state (quantizer bounds + all bucket metadata)
- `router_updates`: Incremental centroid changes between snapshots

### Router (HNSW Index)

The `Router` wraps an HNSW index with quantization (`router.rs:10-49`):

```rust
pub struct Router {
    index: HnswIndex,
    scratch: ThreadLocal<RefCell<SearchScratch>>,    // per-thread
    quant_bufs: ThreadLocal<RefCell<Vec<u8>>>,       // per-thread
    quant_min: f32,
    quant_scale: f32,  // 255.0 / (max - min)
}
```

**HNSW parameters** (`router.rs:30`):
- `m = 24` (connections per node)
- `ef_construction = 180`

**Query routing** (`router.rs:62-138`):
- If ≤20,000 centroids: flat linear scan (exact, deterministic)
- Otherwise: HNSW search with adaptive `ef_search`:
  - `top_k = 1` (ingestion): ef = 64-128 (fast)
  - `top_k > 1` (query): ef = 1200-4000 (high recall)

### Workers

Each worker runs on a dedicated thread with a Glommio async executor (`worker.rs:39-144`).

```rust
pub async fn run_worker(
    id: usize,
    receiver: Receiver<WorkerMessage>,
    wal: Arc<Walrus>,
    vector_index: Arc<VectorIndex>,
    bucket_index: Arc<BucketIndex>,
    bucket_locks: Arc<BucketLocks>,
) {
    let storage = Storage::new(wal.clone());
    let cache = WorkerCache::new(64, 64 * 1024 * 1024, 64 * 64 * 1024 * 1024);  // prealloc 64 buckets × 64MB each
    let executor = Rc::new(Executor::new(storage, cache));

    // Semaphore for concurrency control (max 32 concurrent ops)
    let (limit_tx, limit_rx) = async_channel::bounded(32);

    while let Ok(msg) = receiver.recv().await {
        match msg {
            WorkerMessage::Query(req) => { /* dispatch to executor */ }
            WorkerMessage::Upsert { .. } => { /* append to WAL + update vector/bucket indexes */ }
            WorkerMessage::Ingest { respond_to, .. } => { /* batch append + update vector/bucket indexes, ack via respond_to */ }
            ...
        }
    }
}
```

### Executor

Executes queries on bucket data with LRU caching (`executor.rs:11-16`):

```rust
pub struct Executor {
    storage: Storage,
    cache: Mutex<WorkerCache>,       // LRU: 512 buckets, 64MB each
    cache_version: AtomicU64,        // tracks routing version
    last_changed: Mutex<Arc<Vec<u64>>>,
}
```

**Query execution** (`executor.rs:134-203`):
1. Check cache version, invalidate if routing changed
2. For each bucket: load from cache or storage
3. Compute L2 distance to query (SIMD-optimized)
4. Return top-k candidates

**L2 distance** (`executor.rs:206-300`): AVX2+FMA vectorized, 8 elements per iteration.

### Consistent Hash Ring

Maps bucket IDs to worker shards (`tasks.rs:18-51`):

```
                        CONSISTENT HASH RING
            ┌─────────────────────────────────────────┐
            │                                         │
            │      Worker 0          Worker 1         │
            │    ┌─────────┐       ┌─────────┐        │
            │    │ v0  v1  │       │ v0  v1  │        │
            │    │ v2  ... │       │ v2  ... │        │
            │    └─────────┘       └─────────┘        │
            │         │                 │             │
            │         └────────┬────────┘             │
            │                  ▼                      │
            │     hash(bucket_id) → ring position     │
            │              → worker index             │
            │                                         │
            └─────────────────────────────────────────┘

            Default: 8 virtual nodes per worker
```

---

## Data Flow

### Upsert

```
    ┌─────────────────────────────────────────────────────────────────────┐
    │                         UPSERT FLOW                                 │
    └─────────────────────────────────────────────────────────────────────┘

         upsert(id=42, vector=[0.1, 0.2, ...])
                          │
                          ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  SatoriHandle.upsert()                          service.rs:94-127  │
    └────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  RouterCommand::RouteOrInit                                         │
    │  ├─ If router exists: query HNSW for nearest bucket                 │
    │  └─ If first vector: create bucket #0                               │
    └────────────────────────────────┬────────────────────────────────────┘
                                     │ bucket_id
                                     ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  ConsistentHashRing.node_for(bucket_id)                             │
    │  └─ Determine which worker owns this bucket                         │
    └────────────────────────────────┬────────────────────────────────────┘
                                     │ worker_idx
                                     ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  Worker[worker_idx]                                                 │
    │  └─ WorkerMessage::Upsert                                           │
    │     └─ Storage.put_chunk() → WAL topic "bucket_{id}"                │
    └────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  RouterCommand::ApplyUpsert                                         │
    │  ├─ Update centroid: c = (c * n + v) / (n + 1)                      │
    │  ├─ Persist to router_updates topic                                 │
    │  └─ If pending_updates >= 1000: rebuild HNSW + snapshot             │
    └─────────────────────────────────────────────────────────────────────┘
```

### Query

```
    ┌─────────────────────────────────────────────────────────────────────┐
    │                          QUERY FLOW                                 │
    └─────────────────────────────────────────────────────────────────────┘

         query(vector, top_k=10, router_top_k=200)
                          │
                          ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  RouterCommand::Query                           router_manager.rs   │
    │  └─ Router.query(top_k=200) → [bucket_ids...]                       │
    └────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  Group buckets by worker via ConsistentHashRing    service.rs:46-56 │
    └────────────────────────────────┬────────────────────────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
              ▼                      ▼                      ▼
     ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
     │    Worker 0     │   │    Worker 1     │   │    Worker 2     │
     │                 │   │                 │   │                 │
     │ Executor.query()│   │ Executor.query()│   │ Executor.query()│
     │ ┌─────────────┐ │   │ ┌─────────────┐ │   │ ┌─────────────┐ │
     │ │ Load bucket │ │   │ │ Load bucket │ │   │ │ Load bucket │ │
     │ │ from cache  │ │   │ │ from cache  │ │   │ │ from cache  │ │
     │ │ or WAL      │ │   │ │ or WAL      │ │   │ │ or WAL      │ │
     │ ├─────────────┤ │   │ ├─────────────┤ │   │ ├─────────────┤ │
     │ │ L2 distance │ │   │ │ L2 distance │ │   │ │ L2 distance │ │
     │ │ (AVX2+FMA)  │ │   │ │ (AVX2+FMA)  │ │   │ │ (AVX2+FMA)  │ │
     │ ├─────────────┤ │   │ ├─────────────┤ │   │ ├─────────────┤ │
     │ │ Local top-k │ │   │ │ Local top-k │ │   │ │ Local top-k │ │
     │ └─────────────┘ │   │ └─────────────┘ │   │ └─────────────┘ │
     └────────┬────────┘   └────────┬────────┘   └────────┬────────┘
              │                     │                     │
              │      PARALLEL       │                     │
              └─────────────────────┼─────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  Merge results from all workers                    service.rs:79-91 │
    │  ├─ Concatenate candidates                                          │
    │  ├─ Sort by distance (ascending)                                    │
    │  └─ Truncate to top_k                                               │
    └────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │  [(id, dist), ...]  │
                          │     Top-10 results  │
                          └─────────────────────┘
```

---

## Storage

### WAL Topics

Each bucket is a separate WAL topic (`storage.rs:112-114`):
```rust
fn topic_for(bucket_id: u64) -> String {
    format!("bucket_{}", bucket_id)
}
```

### Vector Format

Each vector stored as (`storage.rs:182-194`):
```
┌────────────────────────────────────────────────────────────────────────────┐
│                           VECTOR RECORD                                    │
├──────────────┬──────────────┬──────────────┬───────────────────────────────┤
│ payload_len  │  vector_id   │  dimension   │         f32 data...           │
│   8 bytes    │   8 bytes    │   8 bytes    │      dim × 4 bytes            │
├──────────────┼──────────────┼──────────────┼───────────────────────────────┤
│   u64 LE     │   u64 LE     │   u64 LE     │   [f32 LE, f32 LE, ...]       │
└──────────────┴──────────────┴──────────────┴───────────────────────────────┘
```

### Storage Modes

```rust
pub enum StorageExecMode {
    Direct,   // Synchronous WAL operations
    Offload,  // Offload to Glommio spawn_blocking
}
```

Workers use `Offload` mode to avoid blocking the async executor.

---

## Quantization

The router quantizes centroids from f32 to u8 for HNSW (`quantizer.rs:1-87`):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           QUANTIZATION                                      │
│                                                                             │
│    Input: f32 vector                    Output: u8 vector                   │
│    [0.12, -0.34, 0.89, ...]     →      [128, 42, 241, ...]                  │
│                                                                             │
│    Formula:  u8_val = clamp((f32_val - min) × scale, 0, 255)                │
│              where scale = 255 / (max - min)                                │
│                                                                             │
│    Bounds computed from all centroids with 0.1% padding                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

```rust
pub struct Quantizer {
    pub min: f32,
    pub max: f32,
    scale: f32,  // 255.0 / (max - min)
}

fn quant_one(x: f32, min: f32, scale: f32) -> u8 {
    let v = x.mul_add(scale, -min * scale);
    v.clamp(0.0, 255.0) as u8
}
```

---

## HNSW Implementation

Custom HNSW in `router_hnsw.rs`:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              HNSW INDEX                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    Level 2    ○───────────────────────────────○                             │
│               │                               │                             │
│    Level 1    ○───────○───────────○───────────○───────○                     │
│               │       │           │           │       │                     │
│    Level 0    ○─○─○─○─○─○─○─○─○─○─○─○─○─○─○─○─○─○─○─○─○                     │
│                                                                             │
│    Parameters:                                                              │
│    • m = 24           (connections per node)                                │
│    • m0 = 48          (connections at level 0)                              │
│    • ef_construction = 180                                                  │
│                                                                             │
│    Distance: Cosine over centered i8                                        │
│    d = 1 - dot(a,b) × invnorm(a) × invnorm(b)                               │
│                                                                             │
│    SIMD: AVX-512 → AVX2 → SSE2 → scalar (auto-detected)                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

```rust
// router_hnsw.rs:13-38
pub struct HnswIndex {
    m: usize,              // connections per node (24)
    m0: usize,             // connections at level 0 (48)
    ef_construction: usize, // beam width during insert (180)

    traversal: Vec<i8>,    // contiguous centered vectors
    inv_norms: Vec<f32>,   // cached 1/norm per vector
    neighbors: Vec<Vec<SmallVec<[usize; 16]>>>,  // adjacency lists per level

    entry: Option<usize>,  // entry point
    max_level: i32,

    dot_fn: DotFn,         // SIMD dispatch (picked once at construction)
    center_fn: CenterFn,
}
```

---

## Rebalancer

Background thread that splits oversized buckets (`rebalancer.rs:396-434`):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          REBALANCER LOOP                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    ┌──────────────────────────────────────────────────────────────┐         │
│    │                    Every 500ms                               │         │
│    └──────────────────────────────┬───────────────────────────────┘         │
│                                   │                                         │
│                                   ▼                                         │
│                      ┌────────────────────────┐                             │
│                      │  Refresh bucket sizes  │                             │
│                      │    from WAL topics     │                             │
│                      └────────────┬───────────┘                             │
│                                   │                                         │
│                                   ▼                                         │
│                      ┌────────────────────────┐                             │
│                      │ Find largest bucket    │                             │
│                      └────────────┬───────────┘                             │
│                                   │                                         │
│                    ┌──────────────┴──────────────┐                          │
│                    │                             │                          │
│              size > 2000?                   size ≤ 2000                     │
│                    │                             │                          │
│                    ▼                             ▼                          │
│    ┌─────────────────────────────┐         ┌──────────┐                     │
│    │         SPLIT               │         │  Sleep   │                     │
│    │  ┌───────────────────────┐  │         │  500ms   │                     │
│    │  │ Load vectors from WAL │  │         └──────────┘                     │
│    │  └───────────┬───────────┘  │                                          │
│    │              ▼              │                                          │
│    │  ┌───────────────────────┐  │                                          │
│    │  │   K-means (k=2)       │  │                                          │
│    │  │   split_bucket_once() │  │                                          │
│    │  └───────────┬───────────┘  │                                          │
│    │              ▼              │                                          │
│    │  ┌───────────────────────┐  │                                          │
│    │  │ Persist new buckets   │  │                                          │
│    │  │ Retire old bucket     │  │                                          │
│    │  │ Rebuild router        │  │                                          │
│    │  └───────────────────────┘  │                                          │
│    └─────────────────────────────┘                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Indexer (K-Means)

SIMD-optimized k-means clustering (`indexer.rs:24-313`):

```rust
pub fn build_clusters(vectors: Vec<Vector>, k: usize) -> Vec<Bucket>
pub fn split_bucket_once(bucket: Bucket) -> Vec<Bucket>  // optimized k=2
```

**SIMD kernels:**
- `k = 2`: Compute both distances in one pass (`nearest_centroid_k2_avx2_fma`)
- `k ≥ 8`: Process 8 centroids at once with transposed layout (`nearest_centroid_block8_avx2_fma`)

---

## Recovery

On startup (`router_manager.rs:364-391`):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RECOVERY FLOW                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    ┌─────────────────────────────────────────────────────────────────┐      │
│    │ 1. Load last router_snapshot                                    │      │
│    │    ├─ Quantizer bounds (min, max)                               │      │
│    │    ├─ All bucket metadata (id, centroid, count)                 │      │
│    │    └─ updates_offset (checkpoint position)                      │      │
│    └─────────────────────────────────────────────────────────────────┘      │
│                                   │                                         │
│                                   ▼                                         │
│    ┌─────────────────────────────────────────────────────────────────┐      │
│    │ 2. Replay router_updates from updates_offset                    │      │
│    │    └─ Apply incremental centroid changes since snapshot         │      │
│    └─────────────────────────────────────────────────────────────────┘      │
│                                   │                                         │
│                                   ▼                                         │
│    ┌─────────────────────────────────────────────────────────────────┐      │
│    │ 3. Rebuild HNSW index from centroids                            │      │
│    └─────────────────────────────────────────────────────────────────┘      │
│                                   │                                         │
│                                   ▼                                         │
│    ┌─────────────────────────────────────────────────────────────────┐      │
│    │ 4. Workers lazy-load bucket vectors on first query              │      │
│    └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Thread Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            THREAD MODEL                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    Main Thread                                                              │
│    └── Creates SatoriDb, holds SatoriHandle                                 │
│                                                                             │
│    Router Manager Thread (1)                                                │
│    ├── Single-threaded command loop                                         │
│    └── Owns: buckets map, router, quantizer                                 │
│                                                                             │
│    Worker Threads (N = num_cpus)                                            │
│    ├── Each runs Glommio LocalExecutor                                      │
│    ├── CPU-pinned via Placement::Fixed(i % cpus)                            │
│    └── Handles subset of buckets via consistent hashing                     │
│                                                                             │
│    Rebalancer Thread (1)                                                    │
│    ├── Runs Glommio LocalExecutor                                           │
│    └── Autonomous bucket split loop                                         │
│                                                                             │
│    ═══════════════════════════════════════════════════════════════════      │
│    Communication: All via channels. No shared mutable state.                │
│    ═══════════════════════════════════════════════════════════════════      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `SATORI_ROUTER_REBUILD_EVERY` | 1000 | Rebuild HNSW after N upserts |
| `SATORI_REBALANCE_THRESHOLD` | 2000 | Split buckets larger than this |
| `SATORI_VECTOR_INDEX_PATH` | `vector_index` | RocksDB-backed `id -> vector` index location |
| `SATORI_BUCKET_INDEX_PATH` | `bucket_index` | RocksDB-backed `id -> bucket` index location |
| `WALRUS_DATA_DIR` | `./wal_files` | WAL storage directory |

---

## Performance Notes

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PERFORMANCE OPTIMIZATIONS                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  MEMORY                                                                     │
│  ├── Quantized routing: 4× reduction (f32 → u8)                             │
│  ├── Cached 1/norm per vector: avoids sqrt in hot path                      │
│  └── LRU bucket cache: 512 buckets × 64MB per worker                        │
│                                                                             │
│  COMPUTE                                                                    │
│  ├── SIMD L2 distance: AVX2+FMA, 8 elements/iteration                       │
│  ├── SIMD k-means: specialized k=2 and k≥8 kernels                          │
│  ├── SIMD HNSW dot: AVX-512/AVX2/SSE2 auto-dispatch                         │
│  └── Per-thread scratch buffers: zero allocation in steady state            │
│                                                                             │
│  I/O                                                                        │
│  ├── io_uring backend: async disk I/O on Linux                              │
│  ├── Batch WAL appends: up to 2000 entries per syscall                      │
│  └── Topic-per-bucket: isolated append streams                              │
│                                                                             │
│  ROUTING                                                                    │
│  ├── Flat search cutoff: ≤20k centroids uses exact scan                     │
│  ├── Adaptive ef_search: 64-128 for ingestion, 1200-4000 for queries        │
│  └── Consistent hashing: uniform bucket distribution across workers         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Design Patterns

Recurring choices scattered throughout the codebase:

### 1. SIMD with Fallback Chains, Dispatch Once

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   router_hnsw.rs    AVX-512 → AVX2 → SSE2 → scalar   (pick_kernels())       │
│   executor.rs       AVX2+FMA → scalar                (L2 distance)          │
│   indexer.rs        AVX2+FMA → scalar                (k-means)              │
│                     + specialized k=2 and k≥8 kernels                       │
│                                                                             │
│   Pattern: Detect CPU features ONCE at construction, store function         │
│            pointer, never branch per-call in hot path.                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2. Thread-Local Scratch Buffers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   router.rs         ThreadLocal<RefCell<SearchScratch>>                     │
│   router.rs         ThreadLocal<RefCell<Vec<u8>>>       (quant buffers)     │
│   storage.rs        TL_BACKING, TL_RANGES                                   │
│   router_hnsw.rs    insert_scratch field                                    │
│   executor.rs       reusable buffers in query path                          │
│                                                                             │
│   Pattern: Pre-allocate per-thread, clear-and-reuse, never allocate         │
│            in steady state.                                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3. Single-Threaded Actors with Channels

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   router_manager    1 thread    crossbeam receiver                          │
│   workers           1 thread    async_channel receiver                      │
│   rebalancer        1 thread    autonomous loop                             │
│                                                                             │
│   Pattern: No locks on hot data, no shared mutable state. Each component    │
│            is owned by one thread, accessed only via message passing.       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4. Lossy Where It Doesn't Hurt, Precise Where It Does

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   APPROXIMATE (speed/memory)          PRECISE (accuracy)                    │
│   ─────────────────────────           ──────────────────                    │
│   Router centroids: f32 → u8          Bucket storage: f32                   │
│   HNSW vectors: u8 → i8 centered      L2 final ranking: f32                 │
│   Routing lookup: quantized cosine    Distance computation: exact L2        │
│                                                                             │
│   Pattern: Routing/indexing can be approximate, final distance              │
│            computation must be precise.                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5. Adaptive Algorithms by Scale

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   router.rs:62-70     ≤20k centroids → flat scan                            │
│                       >20k centroids → HNSW                                 │
│                                                                             │
│   router.rs:80-100    top_k = 1  → ef_search = 64-128   (fast)              │
│                       top_k > 1  → ef_search = 1200-4000 (thorough)         │
│                                                                             │
│   indexer.rs          k = 2  → specialized paired kernel                    │
│                       k ≥ 8  → block-of-8 transposed kernel                 │
│                                                                             │
│   Pattern: Don't use the complex algorithm when the simple one is           │
│            faster. Scale-aware branching.                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6. Incremental Updates + Periodic Rebuilds

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   Centroids          Running mean: c = (c * n + v) / (n + 1)                │
│   HNSW index         Rebuild every 1000 updates                             │
│   Router snapshot    Periodic full snapshot to WAL                          │
│   Router updates     Incremental changes between snapshots                  │
│                                                                             │
│   Pattern: Amortize expensive operations. Don't rebuild on every            │
│            change, batch them up.                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7. Pre-Compute Derived Values

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   quantizer.rs       scale = 255.0 / (max - min)    stored once             │
│   router_hnsw.rs     inv_norms[] = 1.0 / sqrt(dot(v,v))   per vector        │
│   router.rs          quant_scale, quant_min         stored once             │
│                                                                             │
│   Pattern: If you'll use a derived value repeatedly, compute it once        │
│            and cache it. Avoid sqrt/division in hot paths.                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Philosophy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                    ALLOCATION IS THE ENEMY                                  │
│                    BRANCHING IS THE ENEMY                                   │
│                    PRECISION IS A SPECTRUM                                  │
│                                                                             │
│   The entire codebase is tuned for predictable, cache-friendly,             │
│   zero-allocation steady-state performance.                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
