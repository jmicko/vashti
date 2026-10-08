//! Actual imported allocator, host-only numeric storage; no GPU or user data.
use cubecl_ir::MemoryDeviceProperties;
use cubecl_runtime::{
    logging::ServerLogger,
    memory_management::{
        MemoryAllocationMode, MemoryConfiguration, MemoryManagement, MemoryManagementOptions,
        MemoryPoolOptions, PoolType,
    },
    server::IoError,
    storage::{ComputeStorage, StorageHandle, StorageId, StorageUtilization},
};
use std::{collections::HashMap, sync::Arc};

#[derive(Default)]
struct NumericStorage {
    live: HashMap<StorageId, u64>,
    deallocated: usize,
}
impl ComputeStorage for NumericStorage {
    type Resource = StorageId;
    fn alignment(&self) -> usize {
        32
    }
    fn get(&mut self, h: &StorageHandle) -> StorageId {
        assert!(self.live.contains_key(&h.id));
        h.id
    }
    fn alloc(&mut self, size: u64) -> Result<StorageHandle, IoError> {
        let id = StorageId::new();
        self.live.insert(id, size);
        Ok(StorageHandle::new(
            id,
            StorageUtilization { offset: 0, size },
        ))
    }
    fn dealloc(&mut self, id: StorageId) {
        assert!(self.live.remove(&id).is_some());
        self.deallocated += 1;
    }
    fn flush(&mut self) {}
}
fn pool() -> MemoryManagement<NumericStorage> {
    MemoryManagement::from_configuration(
        NumericStorage::default(),
        &MemoryDeviceProperties {
            alignment: 32,
            max_page_size: 1024,
        },
        MemoryConfiguration::Custom {
            pool_options: vec![MemoryPoolOptions {
                pool_type: PoolType::ExclusivePages {
                    max_alloc_size: 1024,
                },
                dealloc_period: Some(1),
            }],
        },
        Arc::new(ServerLogger::default()),
        MemoryManagementOptions::new("public-host-allocator-control")
            .mode(MemoryAllocationMode::Auto),
    )
}
#[test]
fn retained_pages_keep_real_storage_identity_across_repeated_cleanup() {
    let mut p = pool();
    let handles: Vec<_> = (0..128).map(|_| p.reserve(1024).unwrap()).collect();
    let ids: Vec<_> = handles
        .iter()
        .map(|h| p.get_storage(h.clone().binding()).unwrap().id)
        .collect();
    for _ in 0..64 {
        p.cleanup(false);
        for (h, id) in handles.iter().zip(&ids) {
            assert_eq!(
                p.get_resource(h.clone().binding(), None, None).unwrap(),
                *id
            );
        }
    }
    assert_eq!(p.storage().deallocated, 0);
    assert_eq!(p.memory_usage().number_allocs, 128);
    drop(handles);
    p.cleanup(true);
    assert_eq!(p.storage().live.len(), 0);
    assert_eq!(p.storage().deallocated, 128);
}
#[test]
fn removing_multiple_holes_updates_retained_handle_and_binding_locations() {
    let mut p = pool();
    let a = p.reserve(1024).unwrap();
    let b = p.reserve(1024).unwrap();
    let c = p.reserve(1024).unwrap();
    let d = p.reserve(1024).unwrap();
    let e = p.reserve(1024).unwrap();
    let cid = p.get_storage(c.clone().binding()).unwrap().id;
    let eid = p.get_storage(e.clone().binding()).unwrap().id;
    let cb = c.clone().binding();
    let eb = e.clone().binding();
    drop(b);
    drop(d);
    p.cleanup(false);
    assert_eq!(p.storage().deallocated, 2);
    assert_eq!(p.get_resource(cb.clone(), None, None).unwrap(), cid);
    assert_eq!(p.get_resource(eb.clone(), None, None).unwrap(), eid);
    drop(a);
    drop(c);
    drop(e);
    p.cleanup(true);
    assert_eq!(p.storage().live.len(), 2);
    drop(cb);
    drop(eb);
    p.cleanup(true);
    assert_eq!(p.storage().live.len(), 0);
}
#[test]
fn reused_free_pages_keep_original_aging_and_explicit_release_policy() {
    let mut p = pool();
    let h = p.reserve(1024).unwrap();
    let id = p.get_storage(h.clone().binding()).unwrap().id;
    drop(h);
    let h = p.reserve(1024).unwrap();
    assert_eq!(p.get_storage(h.clone().binding()).unwrap().id, id);
    drop(h);
    p.cleanup(false);
    assert_eq!(p.storage().deallocated, 0);
    p.cleanup(false);
    assert_eq!(p.storage().deallocated, 1);
    let h = p.reserve(1024).unwrap();
    drop(h);
    p.cleanup(true);
    assert_eq!(p.storage().deallocated, 2);
    assert_eq!(p.memory_usage().bytes_reserved, 0);
}
