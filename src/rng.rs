//! Tiny, deterministic PRNG helpers to avoid pulling in the `rand` crate.
//! Uses a SplitMix64 generator for good statistical properties at low cost.
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug)]
pub struct SplitMix64 {
    state: AtomicU64,
}

impl SplitMix64 {
    pub fn with_seed(seed: u64) -> Self {
        Self {
            state: AtomicU64::new(seed),
        }
    }

    pub fn seeded_from_time() -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        // Mix in address of self to reduce collision chance across threads.
        let addr = (&nanos as *const u64 as usize) as u64;
        Self::with_seed(nanos ^ addr)
    }

    #[inline(always)]
    pub fn next_u64(&self) -> u64 {
        let mut x = self
            .state
            .fetch_add(0x9E3779B97F4A7C15, Ordering::Relaxed)
            .wrapping_add(0x9E3779B97F4A7C15);
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
        x ^ (x >> 31)
    }

    #[inline(always)]
    pub fn next_f32(&self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32; // take high 24 bits
        bits as f32 / (1u32 << 24) as f32
    }

    #[inline(always)]
    pub fn gen_range(&self, upper: usize) -> usize {
        if upper == 0 {
            return 0;
        }
        (self.next_u64() as usize) % upper
    }
}

/// In-place Fisher–Yates shuffle.
pub fn shuffle<T>(rng: &SplitMix64, slice: &mut [T]) {
    let len = slice.len();
    if len <= 1 {
        return;
    }
    for i in (1..len).rev() {
        let j = rng.gen_range(i + 1);
        slice.swap(i, j);
    }
}

/// Reservoir sample k distinct indices in [0, n).
pub fn sample_k_indices(rng: &SplitMix64, n: usize, k: usize) -> Vec<usize> {
    if k == 0 || n == 0 {
        return Vec::new();
    }
    if k >= n {
        return (0..n).collect();
    }
    let mut res: Vec<usize> = (0..k).collect();
    for i in k..n {
        let j = rng.gen_range(i + 1);
        if j < k {
            res[j] = i;
        }
    }
    res
}
