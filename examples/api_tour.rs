use futures::executor::block_on;
use satoridb::SatoriDb;
use tempfile::tempdir;

fn main() -> anyhow::Result<()> {
    println!("--- SatoriDB API Tour ---\n");

    // Use temp dir for clean demo
    let dir = tempdir()?;

    // =========================================================================
    // 1. Opening the Database
    // =========================================================================
    println!("## Opening the Database\n");

    // Simple one-liner (uses defaults)
    // let db = SatoriDb::open("my_app")?;

    // Builder for full configuration
    let db = SatoriDb::builder("api_tour")
        .workers(2) // Worker threads (default: num_cpus)
        .fsync_ms(100) // Fsync interval in ms (default: 200)
        .data_dir(dir.path()) // Custom data directory
        .virtual_nodes(8) // Hash ring granularity (default: 8)
        .build()?;

    println!("> Database opened with custom configuration.\n");

    // =========================================================================
    // 2. Insert
    // =========================================================================
    println!("## Insert\n");

    let vectors = vec![
        (1u64, vec![1.0, 0.0, 0.0]),
        (2u64, vec![0.0, 1.0, 0.0]),
        (3u64, vec![0.0, 0.0, 1.0]),
        (4u64, vec![0.5, 0.5, 0.0]),
    ];

    for (id, vec) in &vectors {
        db.insert(*id, vec.clone())?;
    }
    println!("> Inserted {} vectors.\n", vectors.len());

    // Duplicate rejection
    match db.insert(1, vec![9.0, 9.0, 9.0]) {
        Err(e) => println!("> Duplicate ID rejected as expected: {}\n", e),
        Ok(_) => println!("> Warning: duplicate was not rejected!\n"),
    }

    // =========================================================================
    // 3. Query (Basic)
    // =========================================================================
    println!("## Query (Basic)\n");

    let query_vec = vec![1.0, 0.1, 0.1];
    let results = db.query(query_vec.clone(), 2)?;

    println!("> Query for {:?}, top_k=2:", query_vec);
    for (id, dist) in &results {
        println!("  - ID: {}, Distance: {:.4}", id, dist);
    }
    println!();

    // =========================================================================
    // 4. Query with Custom Probes (Advanced)
    // =========================================================================
    println!("## Query with Custom Probes (Advanced)\n");

    // Higher probe_buckets = better recall, higher latency
    let results = db.query_with_probes(vec![0.5, 0.5, 0.0], 3, 100)?;

    println!("> Query with probe_buckets=100:");
    for (id, dist) in &results {
        println!("  - ID: {}, Distance: {:.4}", id, dist);
    }
    println!();

    // =========================================================================
    // 5. Query with Vectors
    // =========================================================================
    println!("## Query with Vectors\n");

    let results = db.query_with_vectors(vec![1.0, 0.0, 0.0], 1)?;

    println!("> Query returning stored vectors:");
    for (id, dist, vec) in &results {
        println!("  - ID: {}, Distance: {:.4}, Vector: {:?}", id, dist, vec);
    }
    println!();

    // =========================================================================
    // 6. Get by ID
    // =========================================================================
    println!("## Get by ID\n");

    let fetched = db.get(vec![2, 3, 999])?; // 999 doesn't exist, will be skipped

    println!("> Get IDs [2, 3, 999] (999 missing, skipped):");
    for (id, data) in &fetched {
        println!("  - ID: {}, Vector: {:?}", id, data);
    }
    println!();

    // =========================================================================
    // 7. Delete
    // =========================================================================
    println!("## Delete\n");

    db.delete(2)?;
    println!("> Deleted ID 2.");

    // ID can be reused immediately after delete
    db.insert(2, vec![0.9, 0.9, 0.9])?;
    println!("> Reused ID 2 with new vector.\n");

    // =========================================================================
    // 8. Flush
    // =========================================================================
    println!("## Flush\n");

    db.flush()?;
    println!("> Flushed all pending writes to disk.\n");

    // =========================================================================
    // 9. Stats
    // =========================================================================
    println!("## Stats\n");

    let stats = db.stats();
    println!(
        "> ready={}, buckets={}, vectors={}\n",
        stats.ready, stats.buckets, stats.vectors
    );

    // =========================================================================
    // 10. Async API
    // =========================================================================
    println!("## Async API\n");

    block_on(async {
        // Async insert
        db.insert_async(100, vec![0.1, 0.2, 0.3]).await?;
        println!("> Async insert: ID 100");

        // Async query
        let results = db.query_async(vec![0.1, 0.2, 0.3], 1).await?;
        println!("> Async query: {:?}", results);

        // Async get
        let fetched = db.get_async(vec![100]).await?;
        println!("> Async get: {:?}", fetched);

        // Async delete
        db.delete_async(100).await?;
        println!("> Async delete: ID 100");

        Ok::<_, anyhow::Error>(())
    })?;

    println!();

    // =========================================================================
    // 11. Shutdown
    // =========================================================================
    println!("## Shutdown\n");
    println!("> Drop handles shutdown automatically.");
    println!("> Or call db.shutdown() explicitly for error handling.\n");

    // Explicit shutdown (optional - Drop handles this)
    db.shutdown()?;

    println!("--- API Tour Complete ---");
    Ok(())
}

#[cfg(test)]
mod tests {
    use futures::executor::block_on;
    use satoridb::SatoriDb;
    use tempfile::tempdir;

    #[test]
    fn api_tour_comprehensive() {
        let dir = tempdir().unwrap();
        let db = SatoriDb::builder("test")
            .workers(2)
            .fsync_ms(50)
            .data_dir(dir.path())
            .virtual_nodes(4)
            .build()
            .expect("db start");

        // Insert
        db.insert(1, vec![1.0, 0.0, 0.0]).unwrap();
        db.insert(2, vec![0.0, 1.0, 0.0]).unwrap();
        db.insert(3, vec![0.0, 0.0, 1.0]).unwrap();

        // Duplicate rejection
        assert!(db.insert(1, vec![9.0, 9.0, 9.0]).is_err());

        // Query
        let results = db.query(vec![1.0, 0.1, 0.1], 2).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].0, 1); // Closest should be ID 1

        // Query with probes
        let results = db.query_with_probes(vec![1.0, 0.0, 0.0], 1, 50).unwrap();
        assert_eq!(results[0].0, 1);

        // Query with vectors
        let results = db.query_with_vectors(vec![1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results[0].0, 1);
        assert_eq!(results[0].2, vec![1.0, 0.0, 0.0]);

        // Get by ID
        let fetched = db.get(vec![2, 3, 999]).unwrap();
        assert_eq!(fetched.len(), 2); // 999 is missing
        let ids: std::collections::HashSet<u64> = fetched.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&2) && ids.contains(&3));

        // Delete and reuse
        db.delete(2).unwrap();
        db.insert(2, vec![0.5, 0.5, 0.0]).unwrap();

        // Flush
        db.flush().unwrap();

        // Stats
        let stats = db.stats();
        assert!(stats.buckets >= 1);

        // Async operations
        block_on(async {
            db.insert_async(100, vec![0.1, 0.2, 0.3]).await.unwrap();

            let results = db.query_async(vec![0.1, 0.2, 0.3], 1).await.unwrap();
            assert!(!results.is_empty());

            let fetched = db.get_async(vec![100]).await.unwrap();
            assert_eq!(fetched.len(), 1);

            db.delete_async(100).await.unwrap();
        });
    }
}
