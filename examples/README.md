# Examples and Tuning

- `embedded_basic.rs`: minimal blocking setup, two upserts, one query, clean shutdown.
- `embedded_async.rs`: async API, custom WAL durability (read consistency and fsync schedule), tuned worker/virtual-node counts, router_top_k, bucket-hinted upsert, flush + stats.
- `api_tour.rs`: comprehensive tour of all APIs including query variants, global vs. bucket-local fetching, bucket resolution, and stats.

## Tuning Cheatsheet

- **WAL location**: set `WALRUS_DATA_DIR=/path` to move WAL files; `*_for_key` helpers namespace files under that root.
- **Durability**: pick `ReadConsistency::StrictlyAtOnce` (default) or `AtLeastOnce { persist_every }`; choose `FsyncSchedule::{NoFsync, Milliseconds(u64), SyncEach}` to trade durability vs throughput.
- **Topology**: adjust `SatoriDbConfig.workers` (threads) and `virtual_nodes` (hash ring granularity).
- **Routing**: pass a higher `router_top_k` to `query` for better recall; use `bucket_hint` in `upsert` to force placement when you already know the destination bucket.
- **Maintenance**: call `flush` to persist worker state and router snapshots; `stats` reports bucket count, total vectors, pending router updates, and readiness.
