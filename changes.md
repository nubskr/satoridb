todo:

- use walrus for storage stuff (docs at @src/storage/docs_compressed.txt)

- use router pool with crossbeam mpmc

- use consistent hashing to map chunks to sharded workers (the shards are core pinned, so most 
complexity is not needed anyway)

- the darn sharded worker dedicated DRAM cache is clownly right now, make it better, something like:
instead of considering the counts, so each bucket is just a 2D vector, so we can kinda just configure that, each bucket would have atmax a certain size, after that, we just don't put more shit in that bucket, and the cache eviction policy should be simple, like each bucket shouldnt be bigger than X mbs, and in the cache for each core pinned worker, we only keep Y buckets and evict based on LRU, so each worker has atmost X*Y memory (which we allocate statically at the beginning)

- the bucketing algo is not balanced right now, we need to balance it so that each bucket has roughly the same size of elements (we want the 'work per bucket' to be the same for each bucket for workers)