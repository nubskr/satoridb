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

#[cfg(test)]
pub(crate) fn reset() {
    TOTAL_VECTORS_INSERTED.store(0, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulates_counts() {
        reset();
        assert_eq!(get(), 0);
        add(5);
        add(7);
        assert_eq!(get(), 12);
    }

    #[test]
    fn ignores_zero_adds() {
        reset();
        add(0);
        assert_eq!(get(), 0);
    }
}
