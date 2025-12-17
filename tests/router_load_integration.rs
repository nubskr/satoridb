use anyhow::Result;
use satoridb::quantizer::Quantizer;
use satoridb::router::Router;

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
