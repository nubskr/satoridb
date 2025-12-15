use crate::quantizer::Quantizer;
use crate::router::Router;
use crate::wal::runtime::Walrus;
use anyhow::{anyhow, Result};
use crossbeam_channel::Receiver;
use futures::channel::oneshot;
use log::{error, info};
use rkyv::{Archive, Deserialize, Serialize};
use rkyv::AlignedVec;
use std::collections::HashMap;
use std::sync::Arc;

const ROUTER_SNAPSHOT_TOPIC: &str = "router_snapshot";
const ROUTER_UPDATES_TOPIC: &str = "router_updates";

pub struct RouterQuery {
    pub query_vec: Vec<f32>,
    pub top_k: usize,
    pub respond_to: oneshot::Sender<anyhow::Result<Vec<u64>>>,
}

pub struct RouterRouteOrInit {
    pub vector: Vec<f32>,
    pub respond_to: oneshot::Sender<anyhow::Result<u64>>,
}

pub struct RouterApplyUpsert {
    pub bucket_id: u64,
    pub vector: Vec<f32>,
    pub respond_to: oneshot::Sender<anyhow::Result<BucketMeta>>,
}

pub struct RouterInstallSnapshot {
    pub buckets: Vec<BucketMeta>,
    pub respond_to: oneshot::Sender<anyhow::Result<()>>,
}

pub struct RouterStatsRequest {
    pub respond_to: oneshot::Sender<RouterStats>,
}

pub struct RouterFlushRequest {
    pub respond_to: oneshot::Sender<anyhow::Result<()>>,
}

pub struct RouterShutdownRequest {
    pub respond_to: oneshot::Sender<()>,
}

pub enum RouterCommand {
    Query(RouterQuery),
    RouteOrInit(RouterRouteOrInit),
    ApplyUpsert(RouterApplyUpsert),
    InstallSnapshot(RouterInstallSnapshot),
    Stats(RouterStatsRequest),
    Flush(RouterFlushRequest),
    Shutdown(RouterShutdownRequest),
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct BucketMeta {
    pub id: u64,
    pub centroid: Vec<f32>,
    pub count: u64,
}

#[derive(Clone, Debug, Default)]
pub struct RouterStats {
    pub buckets: usize,
    pub total_vectors: u64,
    pub ready: bool,
    pub pending_updates: u64,
}

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[archive(check_bytes)]
struct BucketMetaRecord {
    id: u64,
    centroid: Vec<f32>,
    count: u64,
}

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[archive(check_bytes)]
struct RouterSnapshotV1 {
    min: f32,
    max: f32,
    buckets: Vec<BucketMetaRecord>,
}

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[archive(check_bytes)]
struct RouterSnapshotV2 {
    min: f32,
    max: f32,
    updates_offset: u64,
    buckets: Vec<BucketMetaRecord>,
}

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[archive(check_bytes)]
struct RouterUpdateRecord {
    id: u64,
    centroid: Vec<f32>,
    count: u64,
}

pub fn spawn_router_manager(
    wal: Arc<Walrus>,
    rx: Receiver<RouterCommand>,
) -> std::thread::JoinHandle<()> {
    std::thread::Builder::new()
        .name("router-mgr".to_string())
        .spawn(move || {
            let mut mgr = RouterManager::new(wal);
            mgr.try_load_state();

            for cmd in rx.iter() {
                match cmd {
                    RouterCommand::Shutdown(s) => {
                        let _ = s.respond_to.send(());
                        break;
                    }
                    other => {
                        if let Err(e) = mgr.handle(other) {
                            error!("Router manager error: {:?}", e);
                        }
                    }
                }
            }
        })
        .expect("failed to spawn router-mgr thread")
}

struct RouterManager {
    wal: Arc<Walrus>,
    buckets: HashMap<u64, BucketMeta>,
    router: Option<Router>,
    quantizer: Option<Quantizer>,
    pending_updates: u64,
    rebuild_every: u64,
}

impl RouterManager {
    fn new(wal: Arc<Walrus>) -> Self {
        let rebuild_every = std::env::var("SATORI_ROUTER_REBUILD_EVERY")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(1_000);
        Self {
            wal,
            buckets: HashMap::new(),
            router: None,
            quantizer: None,
            pending_updates: 0,
            rebuild_every,
        }
    }

    fn handle(&mut self, cmd: RouterCommand) -> Result<()> {
        match cmd {
            RouterCommand::Query(q) => {
                let res = match self.router.as_ref() {
                    Some(router) => router
                        .query(&q.query_vec, q.top_k)
                        .map_err(|e| anyhow!("router query failed: {:?}", e)),
                    None => Err(anyhow!("router not initialized")),
                };
                let _ = q.respond_to.send(res);
                Ok(())
            }
            RouterCommand::RouteOrInit(r) => {
                let res = match self.router.as_ref() {
                    Some(router) => router
                        .query(&r.vector, 1)
                        .map(|ids| ids.first().copied().unwrap_or(0))
                        .map_err(|e| anyhow!("router query failed: {:?}", e)),
                    None => {
                        self.init_single_bucket(&r.vector)?;
                        Ok(0)
                    }
                };
                let _ = r.respond_to.send(res);
                Ok(())
            }
            RouterCommand::ApplyUpsert(u) => {
                let res = self
                    .apply_upsert(u.bucket_id, &u.vector)
                    .and_then(|(meta, created_new_bucket)| {
                        self.persist_update(&meta)?;
                        self.pending_updates = self.pending_updates.saturating_add(1);
                        if self.router.is_none()
                            || created_new_bucket
                            || self.pending_updates >= self.rebuild_every
                        {
                            self.rebuild_router_and_persist()?;
                            self.pending_updates = 0;
                        }
                        Ok(meta)
                    });
                let _ = u.respond_to.send(res);
                Ok(())
            }
            RouterCommand::InstallSnapshot(s) => {
                let res = self
                    .install_snapshot(s.buckets)
                    .and_then(|_| self.rebuild_router_and_persist());
                let _ = s.respond_to.send(res);
                Ok(())
            }
            RouterCommand::Stats(s) => {
                let stats = RouterStats {
                    buckets: self.buckets.len(),
                    total_vectors: self.buckets.values().map(|b| b.count).sum(),
                    ready: self.router.is_some(),
                    pending_updates: self.pending_updates,
                };
                let _ = s.respond_to.send(stats);
                Ok(())
            }
            RouterCommand::Flush(f) => {
                let res = self.rebuild_router_and_persist().map(|_| {
                    self.pending_updates = 0;
                });
                let _ = f.respond_to.send(res);
                Ok(())
            }
            RouterCommand::Shutdown(_s) => Ok(()),
        }
    }

    fn init_single_bucket(&mut self, centroid: &[f32]) -> Result<()> {
        self.buckets.insert(
            0,
            BucketMeta {
                id: 0,
                centroid: centroid.to_vec(),
                count: 0,
            },
        );
        self.rebuild_router_and_persist()?;
        self.pending_updates = 0;
        Ok(())
    }

    fn install_snapshot(&mut self, buckets: Vec<BucketMeta>) -> Result<()> {
        self.buckets.clear();
        for b in buckets {
            self.buckets.insert(b.id, b);
        }
        Ok(())
    }

    fn apply_upsert(&mut self, bucket_id: u64, vector: &[f32]) -> Result<(BucketMeta, bool)> {
        let created_new_bucket = !self.buckets.contains_key(&bucket_id);
        let entry = self.buckets.entry(bucket_id).or_insert_with(|| BucketMeta {
            id: bucket_id,
            centroid: vector.to_vec(),
            count: 0,
        });

        if entry.centroid.len() != vector.len() {
            return Err(anyhow!(
                "vector dim mismatch for bucket {}: centroid_dim={} vector_dim={}",
                bucket_id,
                entry.centroid.len(),
                vector.len()
            ));
        }

        let prev_count = entry.count;
        let new_count = prev_count.saturating_add(1);
        if prev_count == 0 {
            entry.centroid.clone_from_slice(vector);
            entry.count = 1;
            return Ok((entry.clone(), created_new_bucket));
        }

        let prev_count_f = prev_count as f32;
        let new_count_f = new_count as f32;
        for (c, &x) in entry.centroid.iter_mut().zip(vector.iter()) {
            *c = (*c * prev_count_f + x) / new_count_f;
        }
        entry.count = new_count;
        Ok((entry.clone(), created_new_bucket))
    }

    fn rebuild_router_in_memory(&mut self) -> Result<()> {
        if self.buckets.is_empty() {
            self.router = None;
            self.quantizer = None;
            return Ok(());
        }

        let centroids: Vec<Vec<f32>> = self.buckets.values().map(|b| b.centroid.clone()).collect();
        let (min, max) = Quantizer::compute_bounds(&centroids).unwrap_or((0.0, 1.0));
        let quantizer = Quantizer::new(min, max);
        let mut router = Router::new(centroids.len().max(1), quantizer.clone());
        for b in self.buckets.values() {
            router.add_centroid(b.id, &b.centroid);
        }
        self.quantizer = Some(quantizer);
        self.router = Some(router);
        Ok(())
    }

    fn rebuild_router_and_persist(&mut self) -> Result<()> {
        self.rebuild_router_in_memory()?;
        self.persist_snapshot()
    }

    fn persist_snapshot(&self) -> Result<()> {
        if self.quantizer.is_none() {
            return Ok(());
        }
        let q = self.quantizer.as_ref().unwrap();
        let mut buckets: Vec<BucketMetaRecord> = self
            .buckets
            .values()
            .map(|b| BucketMetaRecord {
                id: b.id,
                centroid: b.centroid.clone(),
                count: b.count,
            })
            .collect();
        buckets.sort_by_key(|b| b.id);
        let updates_offset = self.wal.get_topic_size(ROUTER_UPDATES_TOPIC);
        let snapshot = RouterSnapshotV2 {
            min: q.min,
            max: q.max,
            updates_offset,
            buckets,
        };

        let bytes = rkyv::to_bytes::<_, 1024>(&snapshot)
            .map_err(|e| anyhow!("Failed to serialize router snapshot: {}", e))?;
        let mut payload = Vec::with_capacity(8 + bytes.len());
        payload.extend_from_slice(&bytes.len().to_le_bytes());
        payload.extend_from_slice(bytes.as_slice());

        self.wal
            .append_for_topic(ROUTER_SNAPSHOT_TOPIC, &payload)
            .map_err(|e| anyhow!("Walrus append failed (router snapshot): {:?}", e))?;
        Ok(())
    }

    fn persist_update(&self, meta: &BucketMeta) -> Result<()> {
        let record = RouterUpdateRecord {
            id: meta.id,
            centroid: meta.centroid.clone(),
            count: meta.count,
        };
        let bytes = rkyv::to_bytes::<_, 1024>(&record)
            .map_err(|e| anyhow!("Failed to serialize router update: {}", e))?;
        let mut payload = Vec::with_capacity(8 + bytes.len());
        payload.extend_from_slice(&bytes.len().to_le_bytes());
        payload.extend_from_slice(bytes.as_slice());
        self.wal
            .append_for_topic(ROUTER_UPDATES_TOPIC, &payload)
            .map_err(|e| anyhow!("Walrus append failed (router update): {:?}", e))?;
        Ok(())
    }

    fn try_load_state(&mut self) {
        let updates_offset = match self.try_load_snapshot_and_get_updates_offset() {
            Ok(off) => off,
            Err(e) => {
                error!("Router snapshot load failed; replaying updates from 0: {:?}", e);
                self.buckets.clear();
                self.router = None;
                self.quantizer = None;
                0
            }
        };

        if let Err(e) = self.apply_updates_from_offset(updates_offset) {
            error!(
                "Router updates replay failed (from {}): {:?}",
                updates_offset, e
            );
        }
        if let Err(e) = self.rebuild_router_in_memory() {
            error!("Router rebuild after replay failed: {:?}", e);
            self.router = None;
            self.quantizer = None;
        }
        self.pending_updates = 0;
    }

    fn try_load_snapshot_and_get_updates_offset(&mut self) -> Result<u64> {
        let topic_size = self.wal.get_topic_size(ROUTER_SNAPSHOT_TOPIC) as usize;
        if topic_size == 0 {
            info!("No router snapshot found; router starts empty.");
            self.buckets.clear();
            self.router = None;
            self.quantizer = None;
            return Ok(0);
        }

        let entries = self
            .wal
            .batch_read_for_topic(ROUTER_SNAPSHOT_TOPIC, topic_size + 1024, false, Some(0))
            .map_err(|e| anyhow!("Walrus read failed for router snapshot: {:?}", e))?;
        let last = match entries.last() {
            Some(e) => &e.data,
            None => return Ok(0),
        };
        let snapshot = decode_snapshot_v2_or_v1(last)?
            .ok_or_else(|| anyhow!("router snapshot decode failed"))?;

        self.buckets.clear();
        for b in snapshot.buckets {
            self.buckets.insert(
                b.id,
                BucketMeta {
                    id: b.id,
                    centroid: b.centroid,
                    count: b.count,
                },
            );
        }
        info!(
            "Loaded router snapshot with {} buckets.",
            self.buckets.len()
        );
        Ok(snapshot.updates_offset)
    }

    fn apply_updates_from_offset(&mut self, start_offset: u64) -> Result<()> {
        let topic_size = self.wal.get_topic_size(ROUTER_UPDATES_TOPIC);
        if topic_size == 0 || start_offset >= topic_size {
            return Ok(());
        }
        let to_read = (topic_size - start_offset) as usize + 1024;
        let entries = self
            .wal
            .batch_read_for_topic(ROUTER_UPDATES_TOPIC, to_read, false, Some(start_offset))
            .map_err(|e| anyhow!("Walrus read failed for router updates: {:?}", e))?;
        for entry in entries {
            if let Some(update) = decode_update(&entry.data)? {
                self.buckets.insert(
                    update.id,
                    BucketMeta {
                        id: update.id,
                        centroid: update.centroid,
                        count: update.count,
                    },
                );
            }
        }
        Ok(())
    }
}

struct DecodedSnapshot {
    updates_offset: u64,
    buckets: Vec<BucketMetaRecord>,
}

fn decode_snapshot_v2_or_v1(data: &[u8]) -> Result<Option<DecodedSnapshot>> {
    if data.len() < 8 {
        return Ok(None);
    }
    let len = u64::from_le_bytes(data[0..8].try_into().unwrap_or([0; 8])) as usize;
    if 8 + len > data.len() {
        return Ok(None);
    }
    let slice = &data[8..8 + len];
    let mut aligned = AlignedVec::with_capacity(slice.len());
    aligned.extend_from_slice(slice);

    if let Ok(archived) = rkyv::check_archived_root::<RouterSnapshotV2>(&aligned) {
        let mut buckets = Vec::with_capacity(archived.buckets.len());
        for b in archived.buckets.iter() {
            buckets.push(BucketMetaRecord {
                id: b.id,
                centroid: b.centroid.to_vec(),
                count: b.count,
            });
        }
        return Ok(Some(DecodedSnapshot {
            updates_offset: archived.updates_offset,
            buckets,
        }));
    }

    let archived = match rkyv::check_archived_root::<RouterSnapshotV1>(&aligned) {
        Ok(v) => v,
        Err(_) => return Ok(None),
    };
    let mut buckets = Vec::with_capacity(archived.buckets.len());
    for b in archived.buckets.iter() {
        buckets.push(BucketMetaRecord {
            id: b.id,
            centroid: b.centroid.to_vec(),
            count: b.count,
        });
    }
    Ok(Some(DecodedSnapshot {
        updates_offset: 0,
        buckets,
    }))
}

fn decode_update(data: &[u8]) -> Result<Option<RouterUpdateRecord>> {
    if data.len() < 8 {
        return Ok(None);
    }
    let len = u64::from_le_bytes(data[0..8].try_into().unwrap_or([0; 8])) as usize;
    if 8 + len > data.len() {
        return Ok(None);
    }
    let slice = &data[8..8 + len];
    let mut aligned = AlignedVec::with_capacity(slice.len());
    aligned.extend_from_slice(slice);
    let archived = match rkyv::check_archived_root::<RouterUpdateRecord>(&aligned) {
        Ok(v) => v,
        Err(_) => return Ok(None),
    };
    Ok(Some(RouterUpdateRecord {
        id: archived.id,
        centroid: archived.centroid.to_vec(),
        count: archived.count,
    }))
}
