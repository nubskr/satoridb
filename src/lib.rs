//! An embedded vector database for approximate nearest neighbor search.
//!
//! # Quick Start
//!
//! ```no_run
//! use satoridb::SatoriDb;
//!
//! fn main() -> anyhow::Result<()> {
//!     let db = SatoriDb::open("my_app")?;
//!
//!     db.insert(1, vec![0.1, 0.2, 0.3])?;
//!     db.insert(2, vec![0.2, 0.3, 0.4])?;
//!     db.insert(3, vec![0.9, 0.8, 0.7])?;
//!
//!     let results = db.query(vec![0.15, 0.25, 0.35], 10)?;
//!     for (id, distance) in results {
//!         println!("id={id} distance={distance}");
//!     }
//!
//!     Ok(()) // auto-shutdown on drop
//! }
//! ```
//!
//! # Configuration
//!
//! For custom configuration, use the builder:
//!
//! ```no_run
//! use satoridb::SatoriDb;
//!
//! fn main() -> anyhow::Result<()> {
//!     let db = SatoriDb::builder("my_app")
//!         .workers(4)
//!         .fsync_ms(100)
//!         .data_dir("/custom/path")
//!         .build()?;
//!
//!     db.insert(1, vec![0.1, 0.2, 0.3])?;
//!     // ...
//!
//!     Ok(())
//! }
//! ```
//!
//! # Async API
//!
//! ```no_run
//! use satoridb::SatoriDb;
//!
//! async fn example() -> anyhow::Result<()> {
//!     let db = SatoriDb::open("my_app")?;
//!
//!     db.insert_async(1, vec![0.1, 0.2, 0.3]).await?;
//!     let results = db.query_async(vec![0.1, 0.2, 0.3], 10).await?;
//!
//!     Ok(())
//! }
//! ```
//!
//! # Platform & Stability
//!
//! - Linux-only (Glommio/io_uring)
//! - Pre-`1.0`: expect breaking API changes
//! - Single-process embedded database

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

pub use embedded::{SatoriDb, SatoriDbBuilder, Stats};
