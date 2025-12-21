//! An embedded vector database for approximate nearest neighbor search.
//!
//! `satoridb` runs a router manager plus a set of worker shards inside your process. Writes and
//! routing state are persisted to the bundled WAL implementation (“Walrus”), so a clean restart can
//! recover routing and keep queries accurate.
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
//! # Notes
//! - This crate is currently Linux-focused due to the Glommio/io_uring stack.
//! - The API is still evolving; expect breaking changes until `1.0`.

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
