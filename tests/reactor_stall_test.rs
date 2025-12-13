use glommio::timer::Timer;
use glommio::{LocalExecutorBuilder, Placement};
use satoridb::storage::wal::runtime::Walrus;
use satoridb::storage::{Storage, StorageExecMode, Vector};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tempfile::tempdir;

#[test]
fn test_reactor_does_not_stall_during_offload_io() {
    let dir = tempdir().unwrap();
    let dir_path = dir.path().to_str().unwrap().to_string();
    std::env::set_var("WALRUS_DATA_DIR", &dir_path);

    let builder = LocalExecutorBuilder::new(Placement::Unbound)
        .name("stall-test")
        .spawn(move || async move {
            let wal = Arc::new(Walrus::new().expect("failed to init wal"));
            // IMPORTANT: Use Offload mode to prevent blocking the reactor
            let storage = Storage::new(wal).with_mode(StorageExecMode::Offload);

            let heartbeat_count = Arc::new(AtomicUsize::new(0));
            let heartbeat_running = Arc::new(AtomicUsize::new(1)); // 1 = running, 0 = stop

            // 1. Heartbeat Task: Ticks every 10ms
            // If the reactor stalls on I/O, this task won't get scheduled.
            let hb_count = heartbeat_count.clone();
            let hb_run = heartbeat_running.clone();
            let heartbeat_handle = glommio::spawn_local(async move {
                while hb_run.load(Ordering::Relaxed) == 1 {
                    Timer::new(Duration::from_millis(10)).await;
                    hb_count.fetch_add(1, Ordering::Relaxed);
                }
            })
            .detach();

            // 2. Heavy IO Task: Write a lot of data
            // Create a large batch of vectors (e.g., 5MB of data)
            let mut vectors = Vec::new();
            for i in 0..10_000 {
                let data = vec![i as f32; 128]; // 128 dims * 4 bytes = 512 bytes
                vectors.push(Vector::new(i, data));
            }

            // Perform the write. In Offload mode, this should happen on a background thread,
            // allowing the heartbeat timer to fire in the meantime.
            let start = std::time::Instant::now();
            storage
                .put_chunk_raw(1, &vectors)
                .await
                .expect("write failed");
            let duration = start.elapsed();

            // Stop heartbeat
            heartbeat_running.store(0, Ordering::Relaxed);
            // Give the heartbeat loop a moment to exit
            Timer::new(Duration::from_millis(20)).await;

            let ticks = heartbeat_count.load(Ordering::Relaxed);
            println!(
                "IO took {:?}, Heartbeat ticks: {} (expected > 0)",
                duration, ticks
            );

            // Assertions
            // If blocking I/O happened on this thread, the reactor would be blocked for the full `duration`.
            // Since we are writing ~5MB+overhead, it takes non-trivial time (e.g. >10ms).
            // We expect at least some ticks. If ticks == 0, it means we stalled completely.
            // (Relaxed assertion to avoid flakiness on slow CI, but >0 is the critical check).
            assert!(
                ticks > 0,
                "Reactor stalled! Heartbeat counter is 0 after {:?} of IO",
                duration
            );
        });

    builder.expect("failed to spawn executor").join().unwrap();
}
