//! Tests that must run serially (single-threaded) due to:
//! - Global fail hook interference between tests
//! - CPU-intensive operations that starve other tests
//!
//! These tests are marked `#[ignore]` so they don't run with `cargo test`.
//! Run with: `cargo test --test serial_tests -- --ignored --test-threads=1`

use anyhow::Result;
use satoridb::quantizer::Quantizer;
use satoridb::router::Router;

// ============================================================================
// Router load tests (CPU-intensive, 15k-30k centroids)
// ============================================================================

/// Large centroid set should still return the closest IDs when the graph path
/// and EF autotuning are used.
#[test]
#[ignore]
fn router_large_centroid_set_returns_nearest_ids() -> Result<()> {
    let mut router = Router::new(100_000, Quantizer::new(0.0, 100_000.0));

    let total = 15_000u64;
    for i in 0..total {
        let f = i as f32;
        router.add_centroid(i, &[f, f + 0.5]);
    }

    let q = [12_345.3_f32, 12_345.8_f32];
    let res = router.query(&q, 5)?;
    assert_eq!(res.len(), 5);

    let mut expected: Vec<(f32, u64)> = (0..total)
        .map(|i| {
            let f = i as f32;
            let dx = q[0] - f;
            let dy = q[1] - (f + 0.5);
            let dist = dx * dx + dy * dy;
            (dist, i)
        })
        .collect();
    expected.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let expected_ids: Vec<u64> = expected.iter().take(5).map(|(_, id)| *id).collect();
    let best_true = expected_ids[0] as i64;
    let best_got = res[0] as i64;
    assert!(
        (best_true - best_got).abs() <= 500,
        "router top-1 should stay reasonably close to true nearest (got {}, expected {})",
        best_got,
        best_true
    );

    for id in &res {
        assert!(
            *id < total,
            "router returned centroid id {} outside inserted range",
            id
        );
    }

    Ok(())
}

/// Very large centroid sets force the graph search path; ensure it returns
/// non-empty and reasonably close neighbors.
#[test]
#[ignore]
fn router_graph_path_returns_near_ids() -> Result<()> {
    let mut router = Router::new(200_000, Quantizer::new(0.0, 200_000.0));
    let total = 30_000u64;
    for i in 0..total {
        let f = i as f32;
        router.add_centroid(i, &[f, f + 1.0]);
    }

    let q = [25_432.7_f32, 25_433.7_f32];
    let res = router.query(&q, 10)?;
    assert_eq!(res.len(), 10, "graph path should return requested top-k");

    for id in &res {
        assert!(
            *id < total,
            "returned id {} should be within inserted range",
            id
        );
        let diff = (*id as i64 - 25_432).abs();
        assert!(
            diff < 2_000,
            "graph path returned a far centroid (id diff {} too large)",
            diff
        );
    }

    Ok(())
}
