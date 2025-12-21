#[cfg(test)]
mod tests {
    use super::RebalanceState;
    use crate::router::RoutingTable;
    use crate::storage::Storage;
    use crate::wal::runtime::Walrus;
    use std::sync::Arc;
    use tempfile::TempDir;

    fn init_wal(tempdir: &TempDir) -> Arc<Walrus> {
        std::env::set_var("WALRUS_DATA_DIR", tempdir.path());
        Arc::new(Walrus::new().expect("walrus init"))
    }

    #[test]
    fn load_bucket_skips_corrupted_chunk() {
        let tmp = tempfile::tempdir().unwrap();
        let wal = init_wal(&tmp);
        let storage = Storage::new(wal.clone());
        let routing = Arc::new(RoutingTable::new());
        let state = RebalanceState::new(storage.clone(), routing.clone());

        let topic = crate::storage::Storage::topic_for(123);
        wal.append_batch(&topic, &[vec![1u8, 2, 3]]).unwrap();

        let loaded = state.load_bucket(123);
        assert!(loaded.is_none(), "corrupted chunk should not decode into a bucket");
    }
}
