use async_trait::async_trait;
use futures::executor::block_on;
use std::io::Result;

/// Abstract interface for file I/O operations.
/// This allows swapping the backend between standard synchronous I/O (for testing/tools)
/// and Glommio-native asynchronous I/O (for the runtime).
#[async_trait(?Send)]
pub trait WalrusFile: std::fmt::Debug {
    /// Read `len` bytes from the file at the specified `offset`.
    async fn read_at(&self, offset: u64, len: usize) -> Result<Vec<u8>>;

    fn read_at_sync(&self, offset: u64, len: usize) -> Result<Vec<u8>> {
        block_on(self.read_at(offset, len))
    }

    /// Write the provided buffer to the file at the specified `offset`.
    async fn write_at(&self, offset: u64, buf: &[u8]) -> Result<usize>;

    fn write_at_sync(&self, offset: u64, buf: &[u8]) -> Result<usize> {
        block_on(self.write_at(offset, buf))
    }

    /// Sync all changes to disk (fdatasync/fsync).
    async fn sync_all(&self) -> Result<()>;

    fn sync_all_sync(&self) -> Result<()> {
        block_on(self.sync_all())
    }

    /// Get the current file length.
    async fn len(&self) -> Result<u64>;

    fn len_sync(&self) -> Result<u64> {
        block_on(self.len())
    }

    async fn is_empty(&self) -> Result<bool> {
        Ok(self.len().await? == 0)
    }
}
