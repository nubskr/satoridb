use std::sync::atomic::{AtomicU64, Ordering};

// Tracks total vectors inserted by executors / initial ingest for sanity checks.
static TOTAL_VECTORS_INSERTED: AtomicU64 = AtomicU64::new(0);

pub fn add(count: u64) {
    if count == 0 {
        return;
    }
    TOTAL_VECTORS_INSERTED.fetch_add(count, Ordering::Relaxed);
}

pub fn get() -> u64 {
    TOTAL_VECTORS_INSERTED.load(Ordering::Relaxed)
}
