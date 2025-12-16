use async_trait::async_trait;
use std::io::Result;

/// Abstract interface for file I/O operations.
/// This allows swapping the backend between standard synchronous I/O (for testing/tools)
/// and Glommio-native asynchronous I/O (for the runtime).
#[async_trait(?Send)]
pub trait WalrusFile: std::fmt::Debug {
    /// Read `len` bytes from the file at the specified `offset`.
    async fn read_at(&self, offset: u64, len: usize) -> Result<Vec<u8>>;

    /// Write the provided buffer to the file at the specified `offset`.
    async fn write_at(&self, offset: u64, buf: &[u8]) -> Result<usize>;

    /// Sync all changes to disk (fdatasync/fsync).
    async fn sync_all(&self) -> Result<()>;

    /// Get the current file length.
    async fn len(&self) -> Result<u64>;

    async fn is_empty(&self) -> Result<bool> {
        Ok(self.len().await? == 0)
    }
}
