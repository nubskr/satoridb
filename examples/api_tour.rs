use satoridb::SatoriDb;

fn main() -> anyhow::Result<()> {
    println!("--- SatoriDB API Tour ---");

    // 1. Open database
    let db = SatoriDb::builder("api_tour").workers(2).build()?;
    println!("> Database started.");

    // 2. Insert vectors
    let vectors = vec![
        (1u64, vec![1.0, 0.0, 0.0]),
        (2u64, vec![0.0, 1.0, 0.0]),
        (3u64, vec![0.0, 0.0, 1.0]),
    ];

    for (id, vec) in &vectors {
        db.insert(*id, vec.clone())?;
    }
    println!("> Inserted {} vectors.", vectors.len());

    // 3. Query (Basic)
    let query_vec = vec![1.0, 0.1, 0.1];
    let results = db.query(query_vec.clone(), 2)?;
    println!("> Query results (top_k=2):");
    for (id, dist) in &results {
        println!("  - ID: {}, Distance: {:.4}", id, dist);
    }

    // 4. Query with Vectors
    let results_with_data = db.query_with_vectors(query_vec, 1)?;
    println!("> Query with vectors (top_k=1):");
    for (id, dist, vec) in &results_with_data {
        println!("  - ID: {}, Distance: {:.4}, Vector: {:?}", id, dist, vec);
    }

    // 5. Get Vectors by ID
    let fetched = db.get(vec![2, 3])?;
    println!("> Get by ID results: {} found", fetched.len());
    for (id, data) in &fetched {
        println!("  - ID: {}, Vector: {:?}", id, data);
    }

    // 6. Delete
    println!("> Deleting vector ID 2...");
    db.delete(2)?;

    // Verify delete by trying to reinsert (should succeed)
    db.insert(2, vec![0.5, 0.5, 0.0])?;
    println!("> Verified: ID 2 can be reused after delete.");

    // 7. Stats
    let stats = db.stats();
    println!(
        "> Stats: buckets={}, vectors={}",
        stats.buckets, stats.vectors
    );

    println!("> Done!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use satoridb::SatoriDb;
    use tempfile::tempdir;

    #[test]
    fn example_runs_and_validates_outputs() {
        let dir = tempdir().unwrap();
        let db = SatoriDb::builder("test")
            .workers(2)
            .data_dir(dir.path())
            .build()
            .expect("db start");

        // Insert
        db.insert(1, vec![1.0, 0.0, 0.0]).unwrap();
        db.insert(2, vec![0.0, 1.0, 0.0]).unwrap();
        db.insert(3, vec![0.0, 0.0, 1.0]).unwrap();

        // Query
        let results = db.query(vec![1.0, 0.1, 0.1], 2).unwrap();
        let ids: std::collections::HashSet<u64> = results.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&1));

        // Query with vectors
        let results_with_data = db.query_with_vectors(vec![1.0, 0.1, 0.1], 1).unwrap();
        assert_eq!(results_with_data.len(), 1);
        assert_eq!(results_with_data[0].0, 1);
        assert_eq!(results_with_data[0].2, vec![1.0, 0.0, 0.0]);

        // Get
        let fetched = db.get(vec![2, 3]).unwrap();
        let fetched_ids: std::collections::HashSet<u64> =
            fetched.iter().map(|(id, _)| *id).collect();
        assert!(fetched_ids.contains(&2) && fetched_ids.contains(&3));

        // Delete and reinsert
        db.delete(2).unwrap();
        db.insert(2, vec![0.5, 0.5, 0.0]).unwrap();

        // Stats
        let stats = db.stats();
        assert!(stats.buckets >= 1);
    }
}
