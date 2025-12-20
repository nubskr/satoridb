use crate::storage::wal::config::{FsyncSchedule, USE_FD_BACKEND};
use crate::storage::wal::io::WalrusFile;
use async_trait::async_trait;
use memmap2::MmapMut;
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, RwLock};
use std::time::SystemTime;

#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;

#[derive(Debug)]
pub(crate) struct GlommioBackend {
    file: glommio::io::DmaFile,
    len: u64,
}

impl GlommioBackend {
    pub(crate) async fn new(path: &str) -> std::io::Result<Self> {
        let file = glommio::io::DmaFile::open(path)
            .await
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let len = file
            .file_size()
            .await
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        Ok(Self { file, len })
    }
}

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
        // pwrite doesn't move the file cursor
        self.file.write_at(data, offset as u64)
    }

    pub(crate) fn read(&self, offset: usize, dest: &mut [u8]) -> std::io::Result<usize> {
        use std::os::unix::fs::FileExt;
        // pread doesn't move the file cursor
        self.file.read_at(dest, offset as u64)
    }

    pub(crate) fn flush(&self) -> std::io::Result<()> {
        self.file.sync_all()
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    #[allow(dead_code)]
    pub(crate) fn file(&self) -> &std::fs::File {
        &self.file
    }
}

#[derive(Debug)]
pub(crate) enum StorageImpl {
    Mmap(MmapMut),
    Fd(FdBackend),
    Glommio(GlommioBackend),
}

#[async_trait(?Send)]
impl WalrusFile for StorageImpl {
    async fn read_at(&self, offset: u64, len: usize) -> std::io::Result<Vec<u8>> {
        match self {
            StorageImpl::Mmap(mmap) => {
                let mut buf = vec![0u8; len];
                let off = offset as usize;
                if off + len > mmap.len() {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "read beyond mmap bound",
                    ));
                }
                buf.copy_from_slice(&mmap[off..off + len]);
                Ok(buf)
            }
            StorageImpl::Fd(fd) => {
                let mut buf = vec![0u8; len];
                fd.read(offset as usize, &mut buf)?;
                Ok(buf)
            }
            StorageImpl::Glommio(gf) => {
                let res = gf
                    .file
                    .read_at(offset, len)
                    .await
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
                Ok(res.to_vec())
            }
        }
    }

    fn read_at_sync(&self, offset: u64, len: usize) -> std::io::Result<Vec<u8>> {
        match self {
            StorageImpl::Glommio(_) => Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Synchronous read not supported on Glommio backend",
            )),
            _ => {
                let mut buf = vec![0u8; len];
                match self {
                    StorageImpl::Mmap(mmap) => {
                        let off = offset as usize;
                        if off + len > mmap.len() {
                            return Err(std::io::Error::new(
                                std::io::ErrorKind::UnexpectedEof,
                                "read beyond mmap bound",
                            ));
                        }
                        buf.copy_from_slice(&mmap[off..off + len]);
                        Ok(buf)
                    }
                    StorageImpl::Fd(fd) => {
                        fd.read(offset as usize, &mut buf)?;
                        Ok(buf)
                    }
                    _ => unreachable!(),
                }
            }
        }
    }

    async fn write_at(&self, offset: u64, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            StorageImpl::Mmap(mmap) => {
                let off = offset as usize;
                let len = buf.len();
                if off + len > mmap.len() {
                    return Err(std::io::Error::other("write beyond mmap bound"));
                }
                unsafe {
                    let ptr = mmap.as_ptr() as *mut u8;
                    std::ptr::copy_nonoverlapping(buf.as_ptr(), ptr.add(off), len);
                }
                Ok(len)
            }
            StorageImpl::Fd(fd) => fd.write(offset as usize, buf),
            StorageImpl::Glommio(gf) => {
                let n = gf
                    .file
                    .write_at(buf, offset)
                    .await
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
                Ok(n)
            }
        }
    }

    fn write_at_sync(&self, offset: u64, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            StorageImpl::Glommio(_) => Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Synchronous write not supported on Glommio backend",
            )),
            _ => match self {
                StorageImpl::Mmap(mmap) => {
                    let off = offset as usize;
                    let len = buf.len();
                    if off + len > mmap.len() {
                        return Err(std::io::Error::other("write beyond mmap bound"));
                    }
                    unsafe {
                        let ptr = mmap.as_ptr() as *mut u8;
                        std::ptr::copy_nonoverlapping(buf.as_ptr(), ptr.add(off), len);
                    }
                    Ok(len)
                }
                StorageImpl::Fd(fd) => fd.write(offset as usize, buf),
                _ => unreachable!(),
            },
        }
    }

    async fn sync_all(&self) -> std::io::Result<()> {
        match self {
            StorageImpl::Mmap(mmap) => mmap.flush(),
            StorageImpl::Fd(fd) => fd.flush(),
            StorageImpl::Glommio(gf) => gf
                .file
                .fdatasync()
                .await
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e)),
        }
    }

    fn sync_all_sync(&self) -> std::io::Result<()> {
        match self {
            StorageImpl::Glommio(_) => Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Synchronous sync_all not supported on Glommio backend",
            )),
            StorageImpl::Mmap(mmap) => mmap.flush(),
            StorageImpl::Fd(fd) => fd.flush(),
            _ => unreachable!(),
        }
    }

    async fn len(&self) -> std::io::Result<u64> {
        match self {
            StorageImpl::Mmap(mmap) => Ok(mmap.len() as u64),
            StorageImpl::Fd(fd) => Ok(fd.len() as u64),
            StorageImpl::Glommio(gf) => Ok(gf.len),
        }
    }

    fn len_sync(&self) -> std::io::Result<u64> {
        match self {
            StorageImpl::Glommio(_) => Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Synchronous len not supported on Glommio backend",
            )),
            StorageImpl::Mmap(mmap) => Ok(mmap.len() as u64),
            StorageImpl::Fd(fd) => Ok(fd.len() as u64),
            _ => unreachable!(),
        }
    }
}

impl StorageImpl {
    #[allow(dead_code)]
    pub(crate) fn write(&self, offset: usize, data: &[u8]) {
        let _ = self.write_at_sync(offset as u64, data);
    }

    pub(crate) fn read(&self, offset: usize, dest: &mut [u8]) {
        match self {
            StorageImpl::Mmap(mmap) => {
                debug_assert!(offset + dest.len() <= mmap.len());
                let src = &mmap[offset..offset + dest.len()];
                dest.copy_from_slice(src);
            }
            StorageImpl::Fd(fd) => {
                let _ = fd.read(offset, dest);
            }
            StorageImpl::Glommio(_) => {
                panic!("StorageImpl::read called on Glommio backend");
            }
        }
    }

    pub(crate) fn flush(&self) -> std::io::Result<()> {
        self.sync_all_sync()
    }

    pub(crate) fn len(&self) -> usize {
        self.len_sync().unwrap_or(0) as usize
    }

    #[allow(dead_code)]
    pub(crate) fn as_fd(&self) -> Option<&FdBackend> {
        if let StorageImpl::Fd(fd) = self {
            Some(fd)
        } else {
            None
        }
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
    if USE_FD_BACKEND.load(Ordering::Relaxed) {
        let use_o_sync = should_use_o_sync();
        Ok(StorageImpl::Fd(FdBackend::new(path, use_o_sync)?))
    } else {
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        // SAFETY: `file` is opened read/write and lives for the duration of this
        // mapping; `memmap2` upholds aliasing invariants for `MmapMut`.
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        Ok(StorageImpl::Mmap(mmap))
    }
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
