// Public library exports so integration tests and external tooling can use the
// same modules as the binary.
pub mod bvecs;
pub mod executor;
pub mod flatbin;
pub mod fvecs;
pub mod gnd;
pub mod indexer;
pub mod ingest_counter;
pub mod quantizer;
pub mod rebalancer;
pub mod router;
pub mod router_hnsw;
pub mod router_manager;
pub mod storage;
pub mod service;
pub mod tasks;
pub mod wal;
pub mod worker;

pub mod embedded;

pub use embedded::{SatoriDb, SatoriDbConfig};
pub use service::SatoriHandle;
