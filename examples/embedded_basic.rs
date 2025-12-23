use satoridb::SatoriDb;

fn main() -> anyhow::Result<()> {
    let db = SatoriDb::open("my_app")?;

    db.insert(1, vec![0.0, 0.0, 0.0])?;
    db.insert(2, vec![1.0, 1.0, 1.0])?;

    let results = db.query(vec![0.1, 0.1, 0.1], 10)?;
    for (id, distance) in &results {
        println!("id={id} distance={distance}");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use satoridb::SatoriDb;
    use tempfile::tempdir;

    #[test]
    fn example_runs_and_queries_both_vectors() {
        let dir = tempdir().unwrap();
        let db = SatoriDb::builder("test")
            .workers(2)
            .data_dir(dir.path())
            .build()
            .expect("db start");

        db.insert(1, vec![0.0, 0.0, 0.0]).expect("insert 1");
        db.insert(2, vec![1.0, 1.0, 1.0]).expect("insert 2");

        let results = db.query(vec![0.1, 0.1, 0.1], 10).expect("query");
        let ids: std::collections::HashSet<u64> = results.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&1) && ids.contains(&2));
    }
}
