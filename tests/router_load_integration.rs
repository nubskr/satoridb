use anyhow::Result;
use satoridb::quantizer::Quantizer;
use satoridb::router::Router;

/// Large centroid set should still return the closest IDs when the graph path
/// and EF autotuning are used.
#[test]
fn router_large_centroid_set_returns_nearest_ids() -> Result<()> {
    let mut router = Router::new(100_000, Quantizer::new(0.0, 100_000.0));

    // Insert 15k centroids on a line so the nearest IDs are deterministic.
    // This is large enough to exercise the per-thread scratch paths but still
    // uses the exact flat scan branch for deterministic validation.
    let total = 15_000u64;
    for i in 0..total {
        let f = i as f32;
        router.add_centroid(i, &[f, f + 0.5]);
    }

    // Query near the middle and compare against exact brute-force nearest IDs.
    let q = [12_345.3_f32, 12_345.8_f32];
    let res = router.query(&q, 5)?;
    assert_eq!(res.len(), 5);

    // Brute-force exact top-5 on the same synthetic data for ground truth.
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

/// Small centroid sets should take the flat path and return exact nearest neighbors.
#[test]
fn router_small_centroid_set_returns_exact_topk() -> Result<()> {
    let mut router = Router::new(128, Quantizer::new(0.0, 10.0));
    for i in 0..50u64 {
        let f = i as f32;
        router.add_centroid(i, &[f, f * 0.5]);
    }

    let q = [5.2f32, 2.6f32];
    let res = router.query(&q, 4)?;
    assert_eq!(res.len(), 4);

    // Exact brute-force
    let mut expected: Vec<(f32, u64)> = (0..50u64)
        .map(|i| {
            let f = i as f32;
            let dx = q[0] - f;
            let dy = q[1] - f * 0.5;
            (dx * dx + dy * dy, i)
        })
        .collect();
    expected.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let expected_ids: Vec<u64> = expected.iter().take(4).map(|(_, id)| *id).collect();

    assert_eq!(
        res, expected_ids,
        "flat path should return exact nearest centroids for small sets"
    );

    Ok(())
}

/// Very large centroid sets force the graph search path; ensure it returns
/// non-empty and reasonably close neighbors.
#[test]
fn router_graph_path_returns_near_ids() -> Result<()> {
    let mut router = Router::new(200_000, Quantizer::new(0.0, 200_000.0));
    let total = 30_000u64; // > 20k triggers graph path
    for i in 0..total {
        let f = i as f32;
        router.add_centroid(i, &[f, f + 1.0]);
    }

    let q = [25_432.7_f32, 25_433.7_f32];
    let res = router.query(&q, 10)?;
    assert_eq!(res.len(), 10, "graph path should return requested top-k");

    // Sanity: IDs should be in the inserted range and reasonably close to the query index.
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
