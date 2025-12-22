use futures::executor::block_on;
use satoridb::SatoriDb;

fn main() -> anyhow::Result<()> {
    let db = SatoriDb::builder("my_app")
        .workers(2)
        .fsync_ms(100)
        .build()?;

    block_on(async {
        db.insert_async(10, vec![0.2, 0.2, 0.2]).await?;
        db.insert_async(11, vec![0.9, 0.9, 0.9]).await?;

        let results = db.query_async(vec![0.1, 0.1, 0.1], 5).await?;
        for (id, distance) in &results {
            println!("id={id} distance={distance}");
        }

        Ok::<_, anyhow::Error>(())
    })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use futures::executor::block_on;
    use satoridb::SatoriDb;
    use tempfile::tempdir;

    #[test]
    fn example_runs_and_returns_inserted_ids() {
        let dir = tempdir().unwrap();
        let db = SatoriDb::builder("test")
            .workers(2)
            .data_dir(dir.path())
            .build()
            .expect("db start");

        block_on(async {
            db.insert_async(10, vec![0.2, 0.2, 0.2]).await.unwrap();
            db.insert_async(11, vec![0.9, 0.9, 0.9]).await.unwrap();

            let results = db.query_async(vec![0.1, 0.1, 0.1], 5).await.unwrap();
            let ids: std::collections::HashSet<u64> = results.iter().map(|(id, _)| *id).collect();
            assert!(ids.contains(&10) && ids.contains(&11));
        });
    }
}
