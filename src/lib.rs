//! An embedded vector database for approximate nearest neighbor search.
//!
//! `satoridb` runs a router manager plus a set of worker shards inside your process. Writes and
//! routing state are persisted to the bundled WAL implementation (“Walrus”), so a clean restart can
//! recover routing and keep queries accurate. Queries are approximate (HNSW-based) and meant for a
//! single process; there is no distributed mode.
//!
//! # Quickstart
//! ```no_run
//! use std::sync::Arc;
//!
//! use satoridb::wal::runtime::Walrus;
//! use satoridb::wal::{FsyncSchedule, ReadConsistency};
//! use satoridb::{SatoriDb, SatoriDbConfig};
//!
//! fn main() -> anyhow::Result<()> {
//!     // By default Walrus writes under `wal_files/<key>` (relative to the current working dir).
//!     // Set `WALRUS_DATA_DIR=/some/dir` to control the parent directory.
//!     let wal = Arc::new(Walrus::with_consistency_and_schedule_for_key(
//!         "example",
//!         ReadConsistency::StrictlyAtOnce,
//!         FsyncSchedule::Milliseconds(200),
//!     )?);
//!
//!     let mut cfg = SatoriDbConfig::new(wal);
//!     cfg.workers = 4;
//!     let db = SatoriDb::start(cfg)?;
//!     let api = db.handle();
//!
//!     api.upsert_blocking(42, vec![1.0, 2.0, 3.0], None)?;
//!     let results = api.query_blocking(vec![1.0, 2.0, 3.0], 10, 200)?;
//!     println!("results={:?}", results);
//!
//!     db.shutdown()?;
//!     Ok(())
//! }
//! ```
//!
//! ## Async API
//! ```no_run
//! # use std::sync::Arc;
//! # use futures::executor::block_on;
//! # use satoridb::wal::runtime::Walrus;
//! # use satoridb::wal::{FsyncSchedule, ReadConsistency};
//! # use satoridb::{SatoriDb, SatoriDbConfig};
//! # fn main() -> anyhow::Result<()> {
//! let wal = Arc::new(Walrus::with_consistency_and_schedule_for_key(
//!     "async_quickstart",
//!     ReadConsistency::AtLeastOnce { persist_every: 1 },
//!     FsyncSchedule::NoFsync,
//! )?);
//! let db = SatoriDb::start(SatoriDbConfig::new(wal))?;
//! let api = db.handle();
//! block_on(async {
//!     api.upsert(1, vec![0.1, 0.2], None).await?;
//!     let hits = api.query(vec![0.1, 0.2], 5, 16).await?;
//!     println!("{hits:?}");
//!     api.flush().await?;
//!     Ok::<(), anyhow::Error>(())
//! })?;
//! db.shutdown()?;
//! # Ok(()) }
//! ```
//!
//! # Configuration
//! - `SatoriDbConfig.workers`: worker shard threads (defaults to logical cores).
//! - `SatoriDbConfig.virtual_nodes`: consistent-hash ring granularity for bucket placement.
//! - Per-query `router_top_k`: probe more buckets for higher recall at the cost of more work.
//! - WAL durability: choose `ReadConsistency` (`StrictlyAtOnce` or `AtLeastOnce { persist_every }`)
//!   and `FsyncSchedule` (`NoFsync`, `Milliseconds(u64)`, `SyncEach`) via `Walrus::with_*` helpers.
//! - Paths: override RocksDB locations with `SATORI_VECTOR_INDEX_PATH` and
//!   `SATORI_BUCKET_INDEX_PATH`; set `WALRUS_DATA_DIR` to choose the WAL parent directory.
//!
//! # Platform & Stability
//! - Linux-only for now (Glommio/io_uring).
//! - Pre-`1.0`: expect breaking API changes.
//! - Single-process embedded database; approximate recall only.
//!
//! # Examples
//! - `examples/embedded_basic.rs`: minimal blocking setup and query.
//! - `examples/embedded_async.rs`: async API, durability tuning, and topology knobs.
//! - `examples/api_tour.rs`: comprehensive end-to-end tour of queries, fetches, deletes, flush, and stats.

// Public library exports so integration tests and external tooling can use the same modules.
#[doc(hidden)]
pub mod bucket_index;
#[doc(hidden)]
pub mod bucket_locks;
#[doc(hidden)]
pub mod bvecs;
#[doc(hidden)]
pub mod executor;
#[doc(hidden)]
pub mod flatbin;
#[doc(hidden)]
pub mod fvecs;
#[doc(hidden)]
pub mod gnd;
#[doc(hidden)]
pub mod indexer;
#[doc(hidden)]
pub mod ingest_counter;
#[doc(hidden)]
pub mod quantizer;
#[doc(hidden)]
pub mod rebalancer;
#[doc(hidden)]
pub mod rng;
#[doc(hidden)]
pub mod router;
#[doc(hidden)]
pub mod router_hnsw;
#[doc(hidden)]
pub mod router_manager;
#[doc(hidden)]
pub mod service;
#[doc(hidden)]
pub mod storage;
#[doc(hidden)]
pub mod tasks;
#[doc(hidden)]
pub mod vector_index;
pub mod wal;
#[doc(hidden)]
pub mod worker;

pub mod embedded;

pub use embedded::{SatoriDb, SatoriDbConfig};
pub use service::SatoriHandle;
