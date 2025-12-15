use crate::tasks::ConsistentHashRing;
use crate::router_manager::{BucketMeta, RouterCommand, RouterStats};
use crate::worker::{QueryRequest, WorkerMessage};
use anyhow::{anyhow, Result};
use async_channel::Sender as AsyncSender;
use crossbeam_channel::Sender as CrossbeamSender;
use futures::channel::oneshot;
use futures::executor::block_on;
use futures::future::join_all;
use std::sync::Arc;

#[derive(Clone)]
/// A cloneable, in-process client for an embedded [`crate::SatoriDb`] instance.
pub struct SatoriHandle {
    router_tx: CrossbeamSender<RouterCommand>,
    ring: ConsistentHashRing,
    worker_senders: Vec<AsyncSender<WorkerMessage>>,
}

impl SatoriHandle {
    pub fn new(
        router_tx: CrossbeamSender<RouterCommand>,
        ring: ConsistentHashRing,
        worker_senders: Vec<AsyncSender<WorkerMessage>>,
    ) -> Self {
        Self {
            router_tx,
            ring,
            worker_senders,
        }
    }

    /// Query the database for nearest neighbors.
    ///
    /// - `top_k`: number of final results to return.
    /// - `router_top_k`: number of buckets to probe via the router (higher can improve recall).
    pub async fn query(
        &self,
        vector: Vec<f32>,
        top_k: usize,
        router_top_k: usize,
    ) -> Result<Vec<(u64, f32)>> {
        let bucket_ids = self.route_query_buckets(&vector, router_top_k).await?;

        let mut pending = Vec::new();
        let mut requests: Vec<Vec<u64>> = Vec::new();
        for &bid in &bucket_ids {
            let shard = self.ring.node_for(bid);
            if shard >= self.worker_senders.len() {
                continue;
            }
            if requests.len() <= shard {
                requests.resize_with(shard + 1, Vec::new);
            }
            requests[shard].push(bid);
        }

        for (shard, bids) in requests.into_iter().enumerate() {
            if bids.is_empty() {
                continue;
            }
            let (tx, rx) = oneshot::channel();
            let req = QueryRequest {
                query_vec: vector.clone(),
                bucket_ids: bids,
                routing_version: 0,
                affected_buckets: Arc::new(Vec::new()),
                respond_to: tx,
            };
            if self.worker_senders[shard]
                .send(WorkerMessage::Query(req))
                .await
                .is_ok()
            {
                pending.push(rx);
            }
        }

        let responses = join_all(pending).await;
        let mut all_results = Vec::new();
        for res in responses {
            if let Ok(Ok(candidates)) = res {
                all_results.extend(candidates);
            }
        }
        all_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        if all_results.len() > top_k {
            all_results.truncate(top_k);
        }
        Ok(all_results)
    }

    /// Insert or update a vector.
    pub async fn upsert(
        &self,
        id: u64,
        vector: Vec<f32>,
        bucket_hint: Option<u64>,
    ) -> Result<(u64, BucketMeta)> {
        let bucket_id = if let Some(b) = bucket_hint {
            b
        } else {
            self.route_or_init_bucket(&vector).await?
        };

        let shard = self.ring.node_for(bucket_id);
        if shard >= self.worker_senders.len() {
            return Err(anyhow!("invalid shard for bucket {}", bucket_id));
        }

        let (tx, rx) = oneshot::channel();
        let msg = WorkerMessage::Upsert {
            bucket_id,
            vector: crate::storage::Vector::new(id, vector.clone()),
            respond_to: tx,
        };
        self.worker_senders[shard]
            .send(msg)
            .await
            .map_err(|_| anyhow!("worker channel closed"))?;
        rx.await
            .map_err(|e| anyhow!("upsert canceled: {:?}", e))?
            .map_err(|e| anyhow!("upsert persist failed: {:?}", e))?;

        let meta = self.apply_upsert_to_router(bucket_id, &vector).await?;
        Ok((bucket_id, meta))
    }

    /// Flush worker state and write a router snapshot.
    pub async fn flush(&self) -> Result<()> {
        let mut flush_waiters = Vec::new();
        for sender in &self.worker_senders {
            let (tx, rx) = oneshot::channel();
            if sender.send(WorkerMessage::Flush { respond_to: tx }).await.is_ok() {
                flush_waiters.push(rx);
            }
        }
        for rx in flush_waiters {
            let _ = rx.await;
        }
        self.flush_router_snapshot().await?;
        Ok(())
    }

    /// Return basic router statistics.
    pub async fn stats(&self) -> RouterStats {
        let (tx, rx) = oneshot::channel();
        if self
            .router_tx
            .send(RouterCommand::Stats(crate::router_manager::RouterStatsRequest {
                respond_to: tx,
            }))
            .is_err()
        {
            return RouterStats::default();
        }
        rx.await.unwrap_or_default()
    }

    /// Blocking wrapper around [`Self::query`].
    pub fn query_blocking(&self, vector: Vec<f32>, top_k: usize, router_top_k: usize) -> Result<Vec<(u64, f32)>> {
        block_on(self.query(vector, top_k, router_top_k))
    }

    /// Blocking wrapper around [`Self::upsert`].
    pub fn upsert_blocking(
        &self,
        id: u64,
        vector: Vec<f32>,
        bucket_hint: Option<u64>,
    ) -> Result<(u64, BucketMeta)> {
        block_on(self.upsert(id, vector, bucket_hint))
    }

    /// Blocking wrapper around [`Self::flush`].
    pub fn flush_blocking(&self) -> Result<()> {
        block_on(self.flush())
    }

    /// Blocking wrapper around [`Self::stats`].
    pub fn stats_blocking(&self) -> RouterStats {
        block_on(self.stats())
    }

    async fn route_query_buckets(&self, vector: &[f32], top_k: usize) -> Result<Vec<u64>> {
        let (tx, rx) = oneshot::channel();
        let task = RouterCommand::Query(crate::router_manager::RouterQuery {
            query_vec: vector.to_vec(),
            top_k,
            respond_to: tx,
        });
        self.router_tx
            .send(task)
            .map_err(|e| anyhow!("router channel closed: {:?}", e))?;
        match rx.await {
            Ok(Ok(ids)) => Ok(ids),
            Ok(Err(e)) => Err(anyhow!("router error: {:?}", e)),
            Err(e) => Err(anyhow!("router canceled: {:?}", e)),
        }
    }

    async fn route_or_init_bucket(&self, vector: &[f32]) -> Result<u64> {
        let (tx, rx) = oneshot::channel();
        let task = RouterCommand::RouteOrInit(crate::router_manager::RouterRouteOrInit {
            vector: vector.to_vec(),
            respond_to: tx,
        });
        self.router_tx
            .send(task)
            .map_err(|e| anyhow!("router channel closed: {:?}", e))?;
        match rx.await {
            Ok(Ok(id)) => Ok(id),
            Ok(Err(e)) => Err(anyhow!("router error: {:?}", e)),
            Err(e) => Err(anyhow!("router canceled: {:?}", e)),
        }
    }

    async fn apply_upsert_to_router(&self, bucket_id: u64, vector: &[f32]) -> Result<BucketMeta> {
        let (tx, rx) = oneshot::channel();
        let task = RouterCommand::ApplyUpsert(crate::router_manager::RouterApplyUpsert {
            bucket_id,
            vector: vector.to_vec(),
            respond_to: tx,
        });
        self.router_tx
            .send(task)
            .map_err(|e| anyhow!("router channel closed: {:?}", e))?;
        match rx.await {
            Ok(Ok(meta)) => Ok(meta),
            Ok(Err(e)) => Err(anyhow!("router error: {:?}", e)),
            Err(e) => Err(anyhow!("router canceled: {:?}", e)),
        }
    }

    async fn flush_router_snapshot(&self) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        self.router_tx
            .send(RouterCommand::Flush(crate::router_manager::RouterFlushRequest {
                respond_to: tx,
            }))
            .map_err(|_| anyhow!("router channel closed"))?;
        rx.await.unwrap_or_else(|e| Err(e.into()))
    }
}
