use crate::storage::Vector;
use anyhow::{Context, Result};
use async_channel::Sender;
use aws_config::meta::region::RegionProviderChain;
use aws_config::BehaviorVersion;
use aws_sdk_s3::{config::Credentials, config::Region, Client};
use std::thread;
use tokio::io::AsyncReadExt;

pub fn spawn_s3_reader(
    endpoint: String,
    bucket: String,
    key: String,
    access_key: String,
    secret_key: String,
    batch_size: usize,
) -> async_channel::Receiver<Result<Vec<Vector>>> {
    let (tx, rx) = async_channel::bounded(10);

    thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("failed to build tokio runtime");

        rt.block_on(async move {
            if let Err(e) = run_s3_stream(
                tx.clone(),
                endpoint,
                bucket,
                key,
                access_key,
                secret_key,
                batch_size,
            )
            .await
            {
                let _ = tx.send(Err(e)).await;
            }
        });
    });

    rx
}

async fn run_s3_stream(
    tx: Sender<Result<Vec<Vector>>>,
    endpoint: String,
    bucket: String,
    key: String,
    access_key: String,
    secret_key: String,
    batch_size: usize,
) -> Result<()> {
    let credentials = Credentials::new(access_key, secret_key, None, None, "static");
    let region_provider = RegionProviderChain::default_provider().or_else(Region::new("auto"));
    let config = aws_config::defaults(BehaviorVersion::latest())
        .region(region_provider)
        .endpoint_url(endpoint)
        .credentials_provider(credentials)
        .load()
        .await;

    let client = Client::new(&config);

    log::info!("Connecting to S3: bucket={}, key={}", bucket, key);

    let object = client
        .get_object()
        .bucket(bucket)
        .key(key)
        .send()
        .await
        .context("failed to get object from s3")?;

    let mut body = object.body.into_async_read();

    // Read Header
    // dim: u32 (4 bytes)
    // count: u64 (8 bytes)
    let mut header = [0u8; 12];
    body.read_exact(&mut header)
        .await
        .context("failed to read header")?;

    let dim = u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
    let count = u64::from_le_bytes(header[4..12].try_into().unwrap()) as usize;

    log::info!("S3 Stream: dim={}, count={}", dim, count);

    let vector_size = dim * 4;
    let mut buffer = vec![0u8; batch_size * vector_size];
    let mut current_idx = 0;

    while current_idx < count {
        let remaining = count - current_idx;
        let this_batch = remaining.min(batch_size);
        let bytes_needed = this_batch * vector_size;

        body.read_exact(&mut buffer[..bytes_needed])
            .await
            .context("failed to read batch")?;

        let mut vectors = Vec::with_capacity(this_batch);
        let mut offset = 0;
        for _ in 0..this_batch {
            let mut data = Vec::with_capacity(dim);
            for _ in 0..dim {
                let bytes: [u8; 4] = buffer[offset..offset + 4].try_into().unwrap();
                data.push(f32::from_le_bytes(bytes));
                offset += 4;
            }
            vectors.push(Vector::new(current_idx as u64, data));
            current_idx += 1;
        }

        if tx.send(Ok(vectors)).await.is_err() {
            log::warn!("S3 stream receiver dropped, stopping download.");
            break;
        }
        current_idx += this_batch;
    }

    log::info!("S3 Stream finished. Read {} vectors.", current_idx);

    Ok(())
}
