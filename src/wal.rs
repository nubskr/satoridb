pub use crate::storage::wal::*;

// Re-export internal wal modules so existing paths like `crate::wal::runtime::Walrus`
// continue to work even though the canonical definitions live under storage::wal.
pub mod runtime {
    pub use crate::storage::wal::runtime::*;
}
pub mod config {
    pub use crate::storage::wal::config::*;
}
pub mod block {
    pub use crate::storage::wal::block::*;
}
