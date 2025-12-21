//! Write-ahead log (Walrus) configuration and runtime handles.
//!
//! Walrus persists router snapshots and worker operations so a restart can recover without losing
//! routing accuracy. Choose durability by combining a read consistency model with an fsync schedule:
//! - `ReadConsistency::StrictlyAtOnce`: every write is immediately durable (higher latency).
//! - `ReadConsistency::AtLeastOnce { persist_every }`: batched durability; accepts brief replay on crash.
//! - `FsyncSchedule`: `NoFsync`, `Milliseconds(u64)`, or `SyncEach` to trade throughput vs. flush rate.
//!
//! By default, Walrus writes under `wal_files/<key>` in the current directory. Override the parent
//! directory with `WALRUS_DATA_DIR=/path`. The `*_for_key` constructors namespace files under that root.
//!
//! ```no_run
//! use std::sync::Arc;
//! use satoridb::wal::runtime::Walrus;
//! use satoridb::wal::{FsyncSchedule, ReadConsistency};
//!
//! # fn main() -> std::io::Result<()> {
//! let wal: Arc<Walrus> = Arc::new(Walrus::with_consistency_and_schedule_for_key(
//!     "my_db",
//!     ReadConsistency::StrictlyAtOnce,
//!     FsyncSchedule::Milliseconds(200),
//! )?);
//! # Ok(())
//! # }
//! ```
pub use crate::storage::wal::*;

pub mod runtime {
    pub use crate::storage::wal::runtime::*;
}
pub mod config {
    pub use crate::storage::wal::config::*;
}
pub mod block {
    pub use crate::storage::wal::block::*;
}
