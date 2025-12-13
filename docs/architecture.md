# SatoriDB Architecture

High-performance vector database for approximate nearest neighbor search.

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│                                    SATORI                                          │
│                                                                                    │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌───────────────────────┐  │
│  │  Client  │───▶│   Network    │───▶│   Router    │───▶│   Executor Workers    │  │
│  │  (TCP)   │◀───│   Server     │◀───│   (HNSW)    │◀───│   (L2 + Cache)        │  │
│  └──────────┘    └──────────────┘    └──────┬──────┘    └───────────┬───────────┘  │
│                                             │                       │              │
│                                             │ rebuild               │ read/write   │
│                                             ▼                       ▼              │
│                                      ┌─────────────┐         ┌───────────┐         │
│                                      │ Rebalancer  │◀───────▶│  Storage  │         │
│                                      │ Split/Merge │         │  (Walrus) │         │
│                                      └─────────────┘         └───────────┘         │
│                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Model

```
┌─────────────────────────────────────────────────────────────────┐
│ Vector                           │ Bucket                       │
│ ┌─────────────────────────────┐  │ ┌─────────────────────────┐  │
│ │ id:   u64                   │  │ │ id:       u64           │  │
│ │ data: Vec<f32>  [D dims]    │  │ │ centroid: Vec<f32>      │  │
│ └─────────────────────────────┘  │ │ vectors:  Vec<Vector>   │  │
│                                  │ └─────────────────────────┘  │
│ Example: id=42, data=[0.1, ...]  │ Cluster of similar vectors   │
└─────────────────────────────────────────────────────────────────┘
                                        │
                                        │ stored as
                                        ▼
                              ┌─────────────────────┐
                              │ Walrus Topic        │
                              │ "bucket_{id}"       │
                              │                     │
                              │ [len][id][dim][f32s]│
                              │ [len][id][dim][f32s]│
                              │ ...                 │
                              └─────────────────────┘
```

---

## Query Flow

```
                          ┌─────────────────────────────────────────┐
                          │ {"vector": [...], "top_k": 10}          │
                          └───────────────────┬─────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ NETWORK (net.rs)                                                                    │
│                                                                                     │
│   Parse JSON ──▶ Extract vector, top_k, router_top_k                                │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ ROUTER (router.rs)                                                                  │
│                                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                            HNSW Index                                       │   │
│   │                                                                             │   │
│   │    Query ──▶ Quantize f32→u8 ──▶ Search centroids ──▶ top 200 bucket IDs    │   │
│   │                                                                             │   │
│   │    Layers:  L2 ──▶ L1 ──▶ L0 (greedy descent, then beam search)             │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│   Output: [bucket_42, bucket_17, bucket_99, ...]  (200 IDs)                         │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ CONSISTENT HASH RING (tasks.rs)                                                     │
│                                                                                     │
│   bucket_id ──▶ hash(id) ──▶ ring lookup ──▶ shard index                            │
│                                                                                     │
│   ┌─────┬─────┬─────┬─────┬─────┐                                                   │
│   │ S0  │ S1  │ S2  │ S3  │ S0  │ ...  (virtual nodes on ring)                      │
│   └─────┴─────┴─────┴─────┴─────┘                                                   │
│                                                                                     │
│   Group: {shard_0: [b42, b99], shard_1: [b17], ...}                                 │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                          ┌───────────────────┼───────────────────┐
                          ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ EXECUTOR WORKERS (worker.rs + executor.rs)                  [parallel per shard]    │
│                                                                                     │
│   ┌───────────────────────────────────────────────────────────────────────────┐     │
│   │ Worker Shard 0                                                            │     │
│   │                                                                           │     │
│   │   for bucket_id in [b42, b99]:                                            │     │
│   │       vectors = cache.get(bucket_id) or load_from_walrus(bucket_id)       │     │
│   │       for v in vectors:                                                   │     │
│   │           dist = L2(query, v.data)    ◀── SIMD AVX2/FMA                   │     │
│   │           candidates.push((v.id, dist))                                   │     │
│   │                                                                           │     │
│   │   return top 100 by distance                                              │     │
│   └───────────────────────────────────────────────────────────────────────────┘     │
│                                                                                     │
│   Cache: LRU, 512 buckets, invalidated on routing version change                    │
└─────────────────────────────────────────────────────────────────────────────────────┘
                          │                   │                   │
                          └───────────────────┼───────────────────┘
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ MERGE & RETURN                                                                      │
│                                                                                     │
│   Collect all shard results ──▶ Sort by distance ──▶ Take top_k (10)                │
│                                                                                     │
│   Return: [{"id": 1234, "distance": 0.05}, {"id": 5678, "distance": 0.08}, ...]     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Ingestion Flow

### Initial Load (Batch Clustering)

```
┌──────────────────┐
│ Raw Dataset      │
│ (fvecs/bvecs)    │
└────────┬─────────┘
         │ read 100k vectors
         ▼
┌──────────────────────────────────────────────────────────────────┐
│ INDEXER (indexer.rs)                                             │
│                                                                  │
│   K-Means Clustering (k=100)                                     │
│                                                                  │
│   ┌────────────────────────────────────────────────────────────┐ │
│   │  1. Pick k random centroids                                │ │
│   │  2. Assign each vector to nearest centroid (SIMD)          │ │
│   │  3. Recompute centroids as cluster means                   │ │
│   │  4. Repeat until convergence (max 20 iters)                │ │
│   │  5. Rebalance: split oversized clusters recursively        │ │
│   └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│   Output: Vec<Bucket> with id, centroid, vectors                 │
└──────────────────────────────────────────────────────────────────┘
         │
         │ for each bucket
         ▼
┌──────────────────────────────────────────────────────────────────┐
│ STORAGE (storage.rs)                                             │
│                                                                  │
│   storage.put_chunk(bucket)                                      │
│       │                                                          │
│       ▼                                                          │
│   Walrus WAL: append vectors to topic "bucket_{id}"              │
└──────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────┐
│ REBALANCER (rebalancer.rs)                                       │
│                                                                  │
│   prime_centroids(buckets)                                       │
│       │                                                          │
│       ├── Store centroid map: {bucket_id → centroid}             │
│       ├── Build Quantizer from min/max values                    │
│       ├── Create Router with HNSW index                          │
│       └── Install in RoutingTable (version 1)                    │
└──────────────────────────────────────────────────────────────────┘
```

### Streaming Ingestion

```
┌──────────────────┐
│ New Vectors      │
│ (batch of 100k)  │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────┐
│ ROUTER LOOKUP                                                    │
│                                                                  │
│   for vec in batch:                                              │
│       bucket_id = router.query(vec, top_k=1)[0]   ◀── nearest    │
│       shard = ring.node_for(bucket_id)                           │
│       batches[shard][bucket_id].push(vec)                        │
└──────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────┐
│ WORKERS                                                          │
│                                                                  │
│   WorkerMessage::Ingest { bucket_id, vectors }                   │
│       │                                                          │
│       ▼                                                          │
│   storage.put_chunk_raw(bucket_id, vectors)                      │
│       │                                                          │
│       └── Appends to existing WAL topic (no re-clustering)       │
└──────────────────────────────────────────────────────────────────┘
```

---

## Rebalancing

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ PERIODIC DRIVER (main.rs)                                        runs every 1s     │
│                                                                                     │
│   sizes = rebalancer.snapshot_sizes()   ◀── reads WAL entry counts                 │
│                                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │ SPLIT DECISION                                                              │   │
│   │                                                                             │   │
│   │   if any bucket.size > target * 2:    (target = 10,000)                     │   │
│   │       enqueue Split(largest_bucket_id)                                      │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │ MERGE DECISION (every N ticks)                                              │   │
│   │                                                                             │   │
│   │   if two smallest buckets < target / 2:                                     │   │
│   │       enqueue Merge(smallest, second_smallest)                              │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ REBALANCE WORKER                                                                    │
│                                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │ SPLIT(bucket_id)                                                            │   │
│   │                                                                             │   │
│   │   1. Lock bucket                                                            │   │
│   │   2. Load vectors from WAL                                                  │   │
│   │   3. Indexer::split_bucket_once() ──▶ k-means with k=2                      │   │
│   │   4. Allocate 2 new bucket IDs                                              │   │
│   │   5. Persist new buckets to WAL                                             │   │
│   │   6. Retire old bucket (mark + checkpoint)                                  │   │
│   │   7. Update centroid map                                                    │   │
│   │   8. Rebuild router ──▶ install new version                                 │   │
│   │                                                                             │   │
│   │   bucket_42 (25k vecs) ──▶ bucket_100 (12k) + bucket_101 (13k)              │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │ MERGE(bucket_a, bucket_b)                                                   │   │
│   │                                                                             │   │
│   │   1. Lock both buckets (ordered by ID to prevent deadlock)                  │   │
│   │   2. Load vectors from both                                                 │   │
│   │   3. Combine + compute new centroid                                         │   │
│   │   4. Allocate new bucket ID, persist                                        │   │
│   │   5. Retire both old buckets                                                │   │
│   │   6. Rebuild router                                                         │   │
│   │                                                                             │   │
│   │   bucket_5 (2k) + bucket_8 (3k) ──▶ bucket_102 (5k)                         │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ ROUTER REBUILD                                                                      │
│                                                                                     │
│   1. Read all centroids from map                                                    │
│   2. Compute global min/max ──▶ Quantizer(min, max)                                 │
│   3. Create new Router(HNSW)                                                        │
│   4. for (id, centroid) in centroids:                                               │
│          router.add_centroid(id, quantize(centroid))                                │
│   5. routing_table.install(router, changed_buckets=[old, new1, new2])               │
│          └── bumps version, executors invalidate cache for changed buckets         │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Threading Model

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ CPU LAYOUT (8-core example)                                                         │
│                                                                                     │
│   SATORI_CORES=8, usable=6, router_pool=1, executor_shards=5                        │
│                                                                                     │
│   ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐   │
│   │ CPU 0 │ │ CPU 1 │ │ CPU 2 │ │ CPU 3 │ │ CPU 4 │ │ CPU 5 │ │ CPU 6 │ │ CPU 7 │   │
│   ├───────┤ ├───────┤ ├───────┤ ├───────┤ ├───────┤ ├───────┤ ├───────┤ ├───────┤   │
│   │Worker │ │Worker │ │Worker │ │Worker │ │Worker │ │Router │ │Rebal- │ │ Net   │   │
│   │  0    │ │  1    │ │  2    │ │  3    │ │  4    │ │ Pool  │ │ancer  │ │ I/O   │   │
│   └───────┘ └───────┘ └───────┘ └───────┘ └───────┘ └───────┘ └───────┘ └───────┘   │
│   glommio   glommio   glommio   glommio   glommio   std      glommio   glommio      │
│   Fixed(0)  Fixed(1)  Fixed(2)  Fixed(3)  Fixed(4)  thread   Fixed(6)  Unbound      │
│                                                                                     │
│   └──────────────────────────────────────┘                                          │
│              executor shards                                                        │
│              (handle Query + Ingest)                                                │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│ CHANNEL TOPOLOGY                                                                    │
│                                                                                     │
│                     ┌───────────────────┐                                           │
│                     │   router_tx/rx    │  crossbeam MPMC                           │
│                     │   (RouterTask)    │                                           │
│                     └─────────┬─────────┘                                           │
│                               │                                                     │
│            ┌──────────────────┼──────────────────┐                                  │
│            ▼                  ▼                  ▼                                  │
│      ┌──────────┐       ┌──────────┐       ┌──────────┐                             │
│      │ Router 0 │       │ Router 1 │       │    ...   │   (thread pool)             │
│      └──────────┘       └──────────┘       └──────────┘                             │
│                                                                                     │
│                                                                                     │
│      ┌──────────┐       ┌──────────┐       ┌──────────┐                             │
│      │senders[0]│       │senders[1]│       │senders[N]│   async_channel (bounded)   │
│      └────┬─────┘       └────┬─────┘       └────┬─────┘                             │
│           ▼                  ▼                  ▼                                   │
│      ┌──────────┐       ┌──────────┐       ┌──────────┐                             │
│      │ Worker 0 │       │ Worker 1 │       │ Worker N │   (WorkerMessage)           │
│      └──────────┘       └──────────┘       └──────────┘                             │
│                                                                                     │
│                                                                                     │
│                     ┌───────────────────┐                                           │
│                     │  rebalance_tx/rx  │  async_channel (unbounded)                │
│                     │  (RebalanceTask)  │                                           │
│                     └─────────┬─────────┘                                           │
│                               ▼                                                     │
│                     ┌───────────────────┐                                           │
│                     │ Rebalance Worker  │                                           │
│                     └───────────────────┘                                           │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### Router (router.rs, router_hnsw.rs)

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ HNSW INDEX STRUCTURE                                                                │
│                                                                                     │
│   Layer 2:    [C5]─────────────────────[C42]                    (sparse)            │
│                │                         │                                          │
│   Layer 1:    [C5]────[C12]────[C23]────[C42]────[C67]          (medium)            │
│                │       │        │        │        │                                 │
│   Layer 0:    [C5][C8][C12][C17][C23][C31][C42][C55][C67][C89]  (dense, all nodes)  │
│                                                                                     │
│   C = centroid, edges = M neighbors per node                                        │
│   m=24 (upper layers), m0=48 (layer 0), ef_construction=180                         │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│ SEARCH                                                                              │
│                                                                                     │
│   query(vec, top_k):                                                                │
│       │                                                                             │
│       ├── Quantize: f32[D] ──▶ u8[D] (scale to 0-255)                               │
│       │                                                                             │
│       ├── if num_centroids ≤ 20k:                                                   │
│       │       flat_search() ──▶ exact scan, O(n)                                    │
│       │                                                                             │
│       └── else:                                                                     │
│               greedy_descent(L2 → L1) ──▶ find entry point                          │
│               beam_search(L0, ef=1200-4000) ──▶ explore neighbors                   │
│               return top_k bucket IDs                                               │
│                                                                                     │
│   Distance: cosine over centered i8 vectors                                         │
│             d = 1 - dot(a,b) * inv_norm(a) * inv_norm(b)                            │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│ ROUTING TABLE (thread-safe versioned state)                                         │
│                                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐                   │
│   │ RoutingTable                                                │                   │
│   │                                                             │                   │
│   │   version: AtomicU64 ──────────────────────┐                │                   │
│   │                                            │                │                   │
│   │   router: RwLock<Option<RoutingData>>      │                │                   │
│   │              │                             │                │                   │
│   │              ▼                             │                │                   │
│   │   ┌─────────────────────────────┐          │                │                   │
│   │   │ RoutingData                 │          │                │                   │
│   │   │   router: Arc<Router>       │          │                │                   │
│   │   │   changed: Arc<Vec<u64>>    │◀─────────┘                │                   │
│   │   └─────────────────────────────┘   (invalidation list)     │                   │
│   └─────────────────────────────────────────────────────────────┘                   │
│                                                                                     │
│   install(router, changed_buckets):                                                 │
│       version.fetch_add(1)                                                          │
│       *router.write() = Some(RoutingData { router, changed })                       │
│                                                                                     │
│   snapshot():                                                                       │
│       return RoutingSnapshot { router, version, changed }                           │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### Executor (executor.rs)

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ EXECUTOR                                                                            │
│                                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │ LRU Cache                                                                   │   │
│   │                                                                             │   │
│   │   capacity: 512 buckets                                                     │   │
│   │   max_bucket_bytes: 64 MB                                                   │   │
│   │                                                                             │   │
│   │   ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐                               │   │
│   │   │ B:42   │ │ B:17   │ │ B:99   │ │ B:8    │ ... (most recently used)      │   │
│   │   │ 5k vec │ │ 3k vec │ │ 8k vec │ │ 2k vec │                               │   │
│   │   └────────┘ └────────┘ └────────┘ └────────┘                               │   │
│   │                                                             ──▶ evict LRU   │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│   cache_version: AtomicU64                                                          │
│                                                                                     │
│   query(vec, bucket_ids, routing_version, changed_buckets):                         │
│       │                                                                             │
│       ├── if cache_version != routing_version:                                      │
│       │       invalidate(changed_buckets)  // or clear all if empty                 │
│       │       cache_version = routing_version                                       │
│       │                                                                             │
│       ├── for bucket_id in bucket_ids:                                              │
│       │       vectors = cache.get(bucket_id)                                        │
│       │                   or load_bucket(bucket_id)  ──▶ Walrus                     │
│       │                                                                             │
│       │       for v in vectors:                                                     │
│       │           dist = l2_distance(vec, v.data)                                   │
│       │           candidates.push((v.id, dist))                                     │
│       │                                                                             │
│       └── return top 100 sorted by distance                                         │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│ L2 DISTANCE (SIMD)                                                                  │
│                                                                                     │
│   l2_distance_f32_avx2_fma(a, b):                                                   │
│                                                                                     │
│       sum = _mm256_setzero_ps()                                                     │
│       for i in 0..len step 8:                                                       │
│           va = _mm256_loadu_ps(a + i)                                               │
│           vb = _mm256_loadu_ps(b + i)                                               │
│           diff = _mm256_sub_ps(va, vb)                                              │
│           sum = _mm256_fmadd_ps(diff, diff, sum)   // sum += diff²                  │
│       return sqrt(horizontal_sum(sum))                                              │
│                                                                                     │
│   Throughput: ~8 floats per cycle with FMA                                          │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### Indexer (indexer.rs)

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ K-MEANS CLUSTERING                                                                  │
│                                                                                     │
│   build_clusters(vectors, k):                                                       │
│                                                                                     │
│       1. INIT CENTROIDS                                                             │
│          ├── Random: pick k random vectors                                          │
│          └── FarthestPairK2: for k=2, pick two most distant vectors                 │
│                                                                                     │
│       2. ITERATE (max 20)                                                           │
│          │                                                                          │
│          │   ┌─────────────────────────────────────────────────────────────────┐    │
│          │   │ ASSIGN (SIMD optimized)                                         │    │
│          │   │                                                                 │    │
│          │   │   k=2:  nearest_centroid_k2_avx2_fma                            │    │
│          │   │         (both centroids in one pass over point)                 │    │
│          │   │                                                                 │    │
│          │   │   k≥8:  nearest_centroid_block8_avx2_fma                        │    │
│          │   │         (8 centroids at once, SoA transposed layout)            │    │
│          │   │                                                                 │    │
│          │   │   else: pairwise AVX2 distance                                  │    │
│          │   └─────────────────────────────────────────────────────────────────┘    │
│          │                                                                          │
│          │   ┌─────────────────────────────────────────────────────────────────┐    │
│          │   │ UPDATE CENTROIDS                                                │    │
│          │   │                                                                 │    │
│          │   │   centroid[j] = mean(vectors where assignment == j)             │    │
│          │   │   if cluster empty: reseed with random vector                   │    │
│          │   └─────────────────────────────────────────────────────────────────┘    │
│          │                                                                          │
│          └── break if no assignments changed                                        │
│                                                                                     │
│       3. REBALANCE                                                                  │
│          if any bucket.len() > avg * 1.2:                                           │
│              recursively split_bucket()                                             │
│                                                                                     │
│       4. REINDEX                                                                    │
│          assign sequential IDs: 0, 1, 2, ...                                        │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│ split_bucket_once(bucket)  ──▶  for rebalancer, k=2 only                            │
│                                                                                     │
│   • Uses FarthestPairK2 init (deterministic, good separation)                       │
│   • max 8 iterations (fast)                                                         │
│   • force_two_buckets fallback: if degenerate, split in half                        │
│   • no recursive rebalancing                                                        │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Storage

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ STORAGE LAYER (storage.rs)                                                          │
│                                                                                     │
│   Storage { wal: Arc<Walrus>, mode: Direct|Offload }                                │
│                                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │ WRITE                                                                       │   │
│   │                                                                             │   │
│   │   put_chunk(bucket) / put_chunk_raw(bucket_id, vectors):                    │   │
│   │                                                                             │   │
│   │       topic = "bucket_{id}"                                                 │   │
│   │                                                                             │   │
│   │       for vec in vectors:                                                   │   │
│   │           ┌────────┬────────┬────────┬─────────────────┐                    │   │
│   │           │len: u64│id: u64 │dim: u64│ data: dim × f32 │                    │   │
│   │           └────────┴────────┴────────┴─────────────────┘                    │   │
│   │                                                                             │   │
│   │       wal.batch_append_for_topic(topic, serialized_entries)                 │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │ READ                                                                        │   │
│   │                                                                             │   │
│   │   get_chunks(bucket_id) ──▶ Vec<Vec<u8>>:                                   │   │
│   │                                                                             │   │
│   │       topic = "bucket_{id}"                                                 │   │
│   │       entries = wal.batch_read_for_topic(topic, max_bytes, checkpoint=false)│   │
│   │       return entries.map(|e| e.data)                                        │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│   Modes:                                                                            │
│     Direct:  blocking call (used by rebalancer)                                     │
│     Offload: glommio::spawn_blocking (used by workers)                              │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### Walrus (Black Box)

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ WALRUS WAL                                                                          │
│                                                                                     │
│   Topic-based append-only log with crash recovery.                                  │
│                                                                                     │
│   Layout:                                                                           │
│     wal_files/                                                                      │
│       ├── 1700000000           # 1GB file (100 × 10MB blocks)                       │
│       ├── 1700001000           # next file                                          │
│       └── read_offset_idx.db   # persisted read cursors                             │
│                                                                                     │
│   APIs:                                                                             │
│     append_for_topic(topic, data)                                                   │
│     batch_append_for_topic(topic, entries[])  ──▶ io_uring on Linux                 │
│     batch_read_for_topic(topic, max_bytes, checkpoint)                              │
│                                                                                     │
│   Config:                                                                           │
│     ReadConsistency: StrictlyAtOnce | AtLeastOnce                                   │
│     FsyncSchedule:   NoFsync | Milliseconds(n) | SyncEach                           │
│                                                                                     │
│   See: src/storage/docs_compressed.txt                                              │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SATORI_CORES` | `num_cpus` | Total cores for Satori |
| `SATORI_LISTEN_ADDR` | (none) | TCP bind address, e.g. `127.0.0.1:9000` |
| `SATORI_RUN_BENCH` | (unset) | Run benchmark mode on startup |
| `SATORI_REBALANCE_INTERVAL_SECS` | `1` | Rebalance check frequency |
| `SATORI_BUCKET_TARGET_SIZE` | `10000` | Target vectors per bucket |
