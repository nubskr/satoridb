use crate::storage::wal::config::FsyncSchedule;
use crate::storage::wal::io::WalrusFile;
use async_trait::async_trait;
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, RwLock};
use std::time::SystemTime;

#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Debug)]
pub(crate) struct FdBackend {
    file: std::fs::File,
    len: usize,
}

impl FdBackend {
    fn new(path: &str, use_o_sync: bool) -> std::io::Result<Self> {
        let mut opts = OpenOptions::new();
        opts.read(true).write(true);

        #[cfg(unix)]
        if use_o_sync {
            opts.custom_flags(libc::O_SYNC);
        }

        let file = opts.open(path)?;
        let metadata = file.metadata()?;
        let len = metadata.len() as usize;

        Ok(Self { file, len })
    }

    pub(crate) fn write(&self, offset: usize, data: &[u8]) -> std::io::Result<usize> {
        use std::os::unix::fs::FileExt;
        self.file.write_at(data, offset as u64)
    }

    pub(crate) fn read(&self, offset: usize, dest: &mut [u8]) -> std::io::Result<usize> {
        use std::os::unix::fs::FileExt;
        self.file.read_at(dest, offset as u64)
    }

    pub(crate) fn flush(&self) -> std::io::Result<()> {
        self.file.sync_all()
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn file(&self) -> &std::fs::File {
        &self.file
    }
}

thread_local! {
    static DMA_FILE_CACHE: RefCell<HashMap<String, Rc<glommio::io::DmaFile>>> = RefCell::new(HashMap::new());
}

async fn get_or_open_dma_file(path: &str) -> std::io::Result<Rc<glommio::io::DmaFile>> {
    let cached = DMA_FILE_CACHE.with(|cache| {
        cache.borrow().get(path).cloned()
    });
    
    if let Some(file) = cached {
        return Ok(file);
    }

    let file = glommio::io::DmaFile::open(path)
        .await
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    let rc_file = Rc::new(file);

    DMA_FILE_CACHE.with(|cache| {
        cache.borrow_mut().insert(path.to_string(), rc_file.clone());
    });

    Ok(rc_file)
}

#[derive(Debug)]
pub(crate) struct StorageImpl {
    path: String,
    sync_backend: FdBackend,
}

#[async_trait(?Send)]
impl WalrusFile for StorageImpl {
    async fn read_at(&self, offset: u64, len: usize) -> std::io::Result<Vec<u8>> {
        if glommio::executor().id() != 0 {
            let dma_file = get_or_open_dma_file(&self.path).await?;
            let res = dma_file
                .read_at(offset, len)
                .await
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
            Ok(res.to_vec())
        } else {
            self.read_at_sync(offset, len)
        }
    }

    fn read_at_sync(&self, offset: u64, len: usize) -> std::io::Result<Vec<u8>> {
        let mut buf = vec![0u8; len];
        self.sync_backend.read(offset as usize, &mut buf)?;
        Ok(buf)
    }

    async fn write_at(&self, offset: u64, buf: &[u8]) -> std::io::Result<usize> {
        if glommio::executor().id() != 0 {
            let dma_file = get_or_open_dma_file(&self.path).await?;
            // Glommio requires DmaBuffer for write_at. Copy from &[u8].
            let mut dma_buf = dma_file.alloc_dma_buffer(buf.len());
            dma_buf.as_bytes_mut().copy_from_slice(buf);
            
            let n = dma_file
                .write_at(dma_buf, offset)
                .await
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
            Ok(n)
        } else {
            self.write_at_sync(offset, buf)
        }
    }

    fn write_at_sync(&self, offset: u64, buf: &[u8]) -> std::io::Result<usize> {
        self.sync_backend.write(offset as usize, buf)
    }

    async fn sync_all(&self) -> std::io::Result<()> {
        if glommio::executor().id() != 0 {
            let dma_file = get_or_open_dma_file(&self.path).await?;
            dma_file
                .fdatasync()
                .await
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
            Ok(())
        } else {
            self.sync_all_sync()
        }
    }

    fn sync_all_sync(&self) -> std::io::Result<()> {
        self.sync_backend.flush()
    }

    async fn len(&self) -> std::io::Result<u64> {
        if glommio::executor().id() != 0 {
            let dma_file = get_or_open_dma_file(&self.path).await?;
            let len = dma_file
                .file_size()
                .await
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
            Ok(len)
        } else {
            self.len_sync()
        }
    }

    fn len_sync(&self) -> std::io::Result<u64> {
        Ok(self.sync_backend.len() as u64)
    }
}

impl StorageImpl {
    #[allow(dead_code)]
    pub(crate) fn write(&self, offset: usize, data: &[u8]) {
        let _ = self.write_at_sync(offset as u64, data);
    }

    pub(crate) fn read(&self, offset: usize, dest: &mut [u8]) {
        let _ = self.sync_backend.read(offset, dest);
    }

    pub(crate) fn flush(&self) -> std::io::Result<()> {
        self.sync_all_sync()
    }

    pub(crate) fn len(&self) -> usize {
        self.len_sync().unwrap_or(0) as usize
    }

    #[allow(dead_code)]
    pub(crate) fn as_fd(&self) -> Option<&FdBackend> {
        Some(&self.sync_backend)
    }
}

static GLOBAL_FSYNC_SCHEDULE: OnceLock<FsyncSchedule> = OnceLock::new();

fn should_use_o_sync() -> bool {
    GLOBAL_FSYNC_SCHEDULE
        .get()
        .map(|s| matches!(s, FsyncSchedule::SyncEach))
        .unwrap_or(false)
}

fn create_storage_impl(path: &str) -> std::io::Result<StorageImpl> {
    // Always create FdBackend for sync fallback and compatibility
    let use_o_sync = should_use_o_sync();
    let sync_backend = FdBackend::new(path, use_o_sync)?;
    
    // We don't eagerly create Glommio backend here because we might not be in a reactor.
    // It is lazy-loaded in read_at/write_at.
    
    Ok(StorageImpl {
        path: path.to_string(),
        sync_backend,
    })
}

#[derive(Debug)]
pub(crate) struct SharedMmap {
    storage: StorageImpl,
    last_touched_at: AtomicU64,
}

// SAFETY: `SharedMmap` provides interior mutability only via methods that
// enforce bounds and perform atomic timestamp updates; the underlying
// storage supports concurrent reads and explicit flushes.
unsafe impl Sync for SharedMmap {}
// SAFETY: The struct holds storage that is safe to move between threads;
// timestamps are atomics, so sending is sound.
unsafe impl Send for SharedMmap {}

#[async_trait(?Send)]
impl WalrusFile for SharedMmap {
    async fn read_at(&self, offset: u64, len: usize) -> std::io::Result<Vec<u8>> {
        self.read_at_sync(offset, len)
    }

    fn read_at_sync(&self, offset: u64, len: usize) -> std::io::Result<Vec<u8>> {
        self.storage.read_at_sync(offset, len)
    }

    async fn write_at(&self, offset: u64, buf: &[u8]) -> std::io::Result<usize> {
        self.write_at_sync(offset, buf)
    }

    fn write_at_sync(&self, offset: u64, buf: &[u8]) -> std::io::Result<usize> {
        let res = self.storage.write_at_sync(offset, buf);
        if res.is_ok() {
            let now_ms = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_else(|_| std::time::Duration::from_secs(0))
                .as_millis() as u64;
            self.last_touched_at.store(now_ms, Ordering::Relaxed);
        }
        res
    }

    async fn sync_all(&self) -> std::io::Result<()> {
        self.sync_all_sync()
    }

    fn sync_all_sync(&self) -> std::io::Result<()> {
        self.storage.sync_all_sync()
    }

    async fn len(&self) -> std::io::Result<u64> {
        self.len_sync()
    }

    fn len_sync(&self) -> std::io::Result<u64> {
        WalrusFile::len_sync(&self.storage)
    }
}

impl SharedMmap {
    pub(crate) fn new(path: &str) -> std::io::Result<Arc<Self>> {
        let storage = create_storage_impl(path)?;

        let now_ms = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_else(|_| std::time::Duration::from_secs(0))
            .as_millis() as u64;
        Ok(Arc::new(Self {
            storage,
            last_touched_at: AtomicU64::new(now_ms),
        }))
    }

    #[allow(dead_code)]
    pub(crate) fn write(&self, offset: usize, data: &[u8]) {
        // Bounds check before raw copy to maintain memory safety
        debug_assert!(offset <= self.storage.len());
        debug_assert!(self.storage.len() - offset >= data.len());

        self.storage.write(offset, data);

        let now_ms = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_else(|_| std::time::Duration::from_secs(0))
            .as_millis() as u64;
        self.last_touched_at.store(now_ms, Ordering::Relaxed);
    }

    pub(crate) fn read(&self, offset: usize, dest: &mut [u8]) {
        debug_assert!(offset + dest.len() <= self.storage.len());
        self.storage.read(offset, dest);
    }

    #[allow(dead_code)]
    pub(crate) fn len(&self) -> usize {
        self.storage.len()
    }

    #[allow(dead_code)]
    pub(crate) fn flush(&self) -> std::io::Result<()> {
        self.storage.flush()
    }

    #[allow(dead_code)]
    pub(crate) fn storage(&self) -> &StorageImpl {
        &self.storage
    }
}

pub(crate) struct SharedMmapKeeper {
    data: HashMap<String, Arc<SharedMmap>>,
}

impl SharedMmapKeeper {
    fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    // Fast path: many readers concurrently
    fn get_mmap_arc_read(path: &str) -> Option<Arc<SharedMmap>> {
        static MMAP_KEEPER: OnceLock<RwLock<SharedMmapKeeper>> = OnceLock::new();
        let keeper_lock = MMAP_KEEPER.get_or_init(|| RwLock::new(SharedMmapKeeper::new()));
        let keeper = keeper_lock.read().ok()?;
        keeper.data.get(path).cloned()
    }

    // Read-mostly accessor that escalates to write lock only on miss
    pub(crate) fn get_mmap_arc(path: &str) -> std::io::Result<Arc<SharedMmap>> {
        if let Some(existing) = Self::get_mmap_arc_read(path) {
            return Ok(existing);
        }

        static MMAP_KEEPER: OnceLock<RwLock<SharedMmapKeeper>> = OnceLock::new();
        let keeper_lock = MMAP_KEEPER.get_or_init(|| RwLock::new(SharedMmapKeeper::new()));

        // Double-check with a fresh read lock to avoid unnecessary write lock
        {
            let keeper = keeper_lock
                .read()
                .map_err(|_| std::io::Error::other("mmap keeper read lock poisoned"))?;
            if let Some(existing) = keeper.data.get(path) {
                return Ok(existing.clone());
            }
        }

        let mut keeper = keeper_lock
            .write()
            .map_err(|_| std::io::Error::other("mmap keeper write lock poisoned"))?;
        if let Some(existing) = keeper.data.get(path) {
            return Ok(existing.clone());
        }

        let arc = SharedMmap::new(path)?;
        keeper.data.insert(path.to_string(), arc.clone());
        Ok(arc)
    }
}

pub(crate) fn set_fsync_schedule(schedule: FsyncSchedule) {
    let _ = GLOBAL_FSYNC_SCHEDULE.set(schedule);
}

#[allow(dead_code)]
pub(crate) fn fsync_schedule() -> Option<FsyncSchedule> {
    GLOBAL_FSYNC_SCHEDULE.get().copied()
}

pub(crate) fn open_storage_for_path(path: &str) -> std::io::Result<StorageImpl> {
    create_storage_impl(path)
}
