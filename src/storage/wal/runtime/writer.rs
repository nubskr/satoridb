use super::allocator::{BlockAllocator, FileStateTracker};
use super::reader::Reader;
use crate::wal::block::Block;
use crate::wal::config::{
    debug_print, FsyncSchedule, DEFAULT_BLOCK_SIZE, MAX_BATCH_BYTES, MAX_BATCH_ENTRIES,
    PREFIX_META_SIZE,
};
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

pub(super) struct Writer {
    allocator: Arc<BlockAllocator>,
    current_block: Mutex<Block>,
    reader: Arc<Reader>,
    col: String,
    publisher: Arc<mpsc::Sender<String>>,
    current_offset: Mutex<u64>,
    fsync_schedule: FsyncSchedule,
    is_batch_writing: AtomicBool,
}

impl Writer {
    pub(super) fn new(
        allocator: Arc<BlockAllocator>,
        current_block: Block,
        reader: Arc<Reader>,
        col: String,
        publisher: Arc<mpsc::Sender<String>>,
        fsync_schedule: FsyncSchedule,
    ) -> Self {
        Writer {
            allocator,
            current_block: Mutex::new(current_block),
            reader,
            col: col.clone(),
            publisher,
            current_offset: Mutex::new(0),
            fsync_schedule,
            is_batch_writing: AtomicBool::new(false),
        }
    }

    pub(super) fn write(&self, data: &[u8]) -> std::io::Result<()> {
        // Check if batch write is in progress
        if self.is_batch_writing.load(Ordering::Acquire) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::WouldBlock,
                "batch write in progress for this topic",
            ));
        }

        let mut block = self
            .current_block
            .lock()
            .map_err(|_| std::io::Error::other("current_block lock poisoned"))?;
        let mut cur = self
            .current_offset
            .lock()
            .map_err(|_| std::io::Error::other("current_offset lock poisoned"))?;

        let need = (PREFIX_META_SIZE as u64) + (data.len() as u64);
        if *cur + need > block.limit {
            debug_print!(
                "[writer] sealing: col={}, block_id={}, used={}, need={}, limit={}",
                self.col,
                block.id,
                *cur,
                need,
                block.limit
            );
            FileStateTracker::set_block_unlocked(block.id as usize);
            let mut sealed = block.clone();
            sealed.used = *cur;
            sealed.file.sync_all_sync()?;
            let _ = self.reader.append_block_to_chain(&self.col, sealed);
            debug_print!("[writer] appended sealed block to chain: col={}", self.col);
            // switch to new block
            // SAFETY: We hold `current_block` and `current_offset` mutexes, so
            // this writer has exclusive ownership of the active block. The
            // allocator's internal lock ensures unique block handout.
            let new_block = unsafe { self.allocator.alloc_block(need) }?;
            debug_print!(
                "[writer] switched to new block: col={}, new_block_id={}",
                self.col,
                new_block.id
            );
            *block = new_block;
            *cur = 0;
        }
        let next_block_start = block.offset + block.limit; // simplistic for now
        block.write_sync(*cur, data, &self.col, next_block_start)?;
        debug_print!(
            "[writer] wrote: col={}, block_id={}, offset_before={}, bytes={}, offset_after={}",
            self.col,
            block.id,
            *cur,
            need,
            *cur + need
        );
        *cur += need;

        // Handle fsync based on schedule
        match self.fsync_schedule {
            FsyncSchedule::SyncEach => {
                // Immediate mmap flush, skip background flusher
                block.file.sync_all_sync()?;
                debug_print!(
                    "[writer] immediate fsync: col={}, block_id={}",
                    self.col,
                    block.id
                );
            }
            FsyncSchedule::Milliseconds(_) => {
                // Send to background flusher
                let _ = self.publisher.send(block.file_path.clone());
            }
            FsyncSchedule::NoFsync => {
                // No fsyncing at all - maximum throughput, no durability guarantees
                debug_print!("[writer] no fsync: col={}, block_id={}", self.col, block.id);
            }
        }

        Ok(())
    }

    pub(super) fn batch_write(&self, batch: &[&[u8]]) -> std::io::Result<()> {
        // RAII guard to ensure batch flag is released
        struct BatchGuard<'a> {
            flag: &'a AtomicBool,
        }
        impl<'a> Drop for BatchGuard<'a> {
            fn drop(&mut self) {
                self.flag.store(false, Ordering::Release);
                debug_print!("[batch] released batch_writing flag");
            }
        }

        // Phase 0: Validate batch size
        if batch.len() > MAX_BATCH_ENTRIES {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("batch exceeds {} entry limit", MAX_BATCH_ENTRIES),
            ));
        }

        let total_bytes: u64 = batch
            .iter()
            .map(|data| (PREFIX_META_SIZE as u64) + (data.len() as u64))
            .sum();

        if total_bytes > MAX_BATCH_BYTES {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "batch exceeds 10GB limit",
            ));
        }

        if batch.is_empty() {
            return Ok(());
        }

        // Try to acquire batch write flag
        if self
            .is_batch_writing
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::WouldBlock,
                "another batch write already in progress",
            ));
        }

        // Ensure we release the flag even if we panic
        let _guard = BatchGuard {
            flag: &self.is_batch_writing,
        };

        debug_print!(
            "[batch] START: col={}, entries={}, total_bytes={}",
            self.col,
            batch.len(),
            total_bytes
        );

        // Phase 1: Pre-allocation & Planning
        let mut block = self
            .current_block
            .lock()
            .map_err(|_| std::io::Error::other("current_block lock poisoned"))?;
        let mut cur_offset = self
            .current_offset
            .lock()
            .map_err(|_| std::io::Error::other("current_offset lock poisoned"))?;

        let mut revert_info = BatchRevertInfo {
            original_offset: *cur_offset,
            allocated_block_ids: Vec::new(),
        };

        // Build write plan: (Block, in_block_offset, batch_index)
        let mut write_plan: Vec<(Block, u64, usize)> = Vec::new();
        let mut batch_idx = 0;

        // Use a LOCAL offset for planning, don't update the writer's offset yet
        let mut planning_offset = *cur_offset;

        while batch_idx < batch.len() {
            let data = batch[batch_idx];
            let need = (PREFIX_META_SIZE as u64) + (data.len() as u64);
            let available = block.limit - planning_offset;

            if available >= need {
                // Fits in current block
                write_plan.push((block.clone(), planning_offset, batch_idx));
                planning_offset += need;
                batch_idx += 1;
            } else {
                // Need to seal and allocate new block
                debug_print!(
                    "[batch] sealing block_id={}, used={}, need={}, limit={}",
                    block.id,
                    planning_offset,
                    need,
                    block.limit
                );
                FileStateTracker::set_block_unlocked(block.id as usize);
                let mut sealed = block.clone();
                sealed.used = planning_offset;
                sealed.file.sync_all_sync()?;
                let _ = self.reader.append_block_to_chain(&self.col, sealed);

                // Allocate new block
                // SAFETY: We hold locks, so this writer has exclusive ownership
                let new_block =
                    unsafe { self.allocator.alloc_block(need.max(DEFAULT_BLOCK_SIZE))? };
                debug_print!("[batch] allocated new block_id={}", new_block.id);

                revert_info.allocated_block_ids.push(new_block.id);
                *block = new_block;
                planning_offset = 0;
            }
        }

        debug_print!(
            "[batch] planning complete: {} write operations across {} blocks",
            write_plan.len(),
            revert_info.allocated_block_ids.len() + 1
        );

        // Fallback: use regular block.write() in a loop (mmap backend or non-Linux builds)
        for (blk, offset, data_idx) in write_plan.iter() {
            let data = batch[*data_idx];
            let next_block_start = blk.offset + blk.limit;

            if let Err(e) = blk.write_sync(*offset, data, &self.col, next_block_start) {
                // Clean up any partially written headers up to and including the failed index
                for (w_blk, w_off, _) in write_plan[0..=(*data_idx)].iter() {
                    let _ = w_blk.zero_range_sync(*w_off, PREFIX_META_SIZE as u64);
                }

                // Flush zeros and rollback
                let mut fsynced = HashSet::new();
                for (w_blk, _, _) in write_plan[0..=(*data_idx)].iter() {
                    if fsynced.insert(w_blk.file_path.clone()) {
                        let _ = w_blk.file.sync_all_sync();
                    }
                }

                *cur_offset = revert_info.original_offset;
                for block_id in revert_info.allocated_block_ids {
                    FileStateTracker::set_block_unlocked(block_id as usize);
                }
                return Err(e);
            }
        }

        // Success - fsync touched files
        let mut fsynced = HashSet::new();
        for (blk, _, _) in write_plan.iter() {
            if !fsynced.contains(&blk.file_path) {
                blk.file.sync_all_sync()?;
                fsynced.insert(blk.file_path.clone());
            }
        }

        // NOW update the writer's offset to make data visible to readers
        *cur_offset = planning_offset;

        debug_print!(
            "[batch] SUCCESS (mmap): wrote {} entries, {} bytes to topic={}",
            batch.len(),
            total_bytes,
            self.col
        );
        Ok(())
    }
}

struct BatchRevertInfo {
    original_offset: u64,
    allocated_block_ids: Vec<u64>,
}

impl Writer {
    pub(super) fn snapshot_block(&self) -> std::io::Result<(Block, u64)> {
        let block = self
            .current_block
            .lock()
            .map_err(|_| std::io::Error::other("current_block lock poisoned"))?;
        let offset = self
            .current_offset
            .lock()
            .map_err(|_| std::io::Error::other("current_offset lock poisoned"))?;
        Ok((block.clone(), *offset))
    }
}
