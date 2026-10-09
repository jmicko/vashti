//! Device-free PUBLIC ownership/compiler fixture; never creates a GPU device.
extern crate alloc;
use cubecl::prelude::*;
use cubecl_common::stream_id::StreamId;
use cubecl_runtime::{
    id::KernelId,
    kernel::CompiledKernel,
    public_external::{PublicExternalBudget, PublicExternalHostSource, PublicExternalPool},
    server::{Handle, KernelArguments},
};
pub use cubecl_wgpu::{AutoCompiler, AutoRepresentation};
mod storage {
    pub use cubecl_wgpu::WgpuResource;
}
#[allow(dead_code)]
#[path = "../../../vendor/cubecl-wgpu/src/compute/public_external.rs"]
mod subject;
const STREAM: StreamId = StreamId { value: 912 };
fn budget(n: usize, backing: u64) -> PublicExternalBudget {
    PublicExternalBudget {
        backing_bytes: backing,
        buffers: n,
        metadata_bytes: 1024 * 1024,
    }
}
fn source(id: u32, bytes: u64, backing: u64) -> PublicExternalHostSource<u32> {
    // SAFETY: numeric fake public resource, no actual device memory.
    unsafe {
        PublicExternalHostSource::from_initialized_public_host(id, bytes, backing, 2, 0, [1; 32])
            .unwrap()
    }
}
#[test]
fn envelope_rejection_precedes_registration() {
    for (bytes, backing, flags, heap, hash) in [
        (0, 32, 2, 0, [1; 32]),
        (31, 32, 2, 0, [1; 32]),
        (34, 36, 2, 0, [1; 32]),
        (134217732, 134217732, 2, 0, [1; 32]),
        (32, 31, 2, 0, [1; 32]),
        (32, u64::MAX, 2, 0, [1; 32]),
        (32, 32, 0, 0, [1; 32]),
        (32, 32, 3, 0, [1; 32]),
        (32, 32, 18, 0, [1; 32]),
        (32, 32, 34, 0, [1; 32]),
        (32, 32, 2, 1, [1; 32]),
        (32, 32, 2, 0, [0; 32]),
    ] {
        assert!(
            unsafe {
                PublicExternalHostSource::from_initialized_public_host(
                    0u32, bytes, backing, flags, heap, hash,
                )
            }
            .is_err()
        );
    }
    for b in [
        budget(0, 32),
        budget(1025, 32),
        budget(1, 0),
        budget(1, u64::MAX),
        PublicExternalBudget {
            metadata_bytes: 1,
            ..budget(1, 32)
        },
    ] {
        assert!(PublicExternalPool::<u32>::new(b, STREAM).is_err());
    }
    assert!(
        unsafe {
            PublicExternalHostSource::from_initialized_public_host(0u32, 32, 64, 6, 0, [2; 32])
        }
        .is_ok()
    );
}
#[test]
fn actual_typed_capacity_and_failed_insert_nonmutation() {
    let mut p = PublicExternalPool::new(budget(2, 96), STREAM).unwrap();
    assert!(p.usage().metadata_owned_bytes > 65536 && p.usage().metadata_owned_bytes <= 1048576);
    let a = p.register(source(17, 32, 64)).unwrap();
    assert!(!a.handle().can_mut());
    let before = p.usage();
    assert_eq!(
        (
            before.registered_backing_bytes,
            before.total_registered_bytes
        ),
        (64, 64)
    );
    assert!(p.register(source(19, 32, 64)).is_err());
    assert_eq!(p.usage(), before);
    assert_eq!(p.resolve(&a.handle().binding()).unwrap(), 17);
    let _b = p.register(source(21, 32, 32)).unwrap();
    let before = p.usage();
    assert!(p.register(source(23, 32, 32)).is_err());
    assert_eq!(p.usage(), before);
    assert_eq!(p.resources().copied().collect::<Vec<_>>(), [17, 21]);
}
#[test]
fn real_descriptor_identity_extent_and_stream_not_just_slot_index() {
    let mut p = PublicExternalPool::new(budget(2, 128), STREAM).unwrap();
    let a = p.register(source(17, 64, 64)).unwrap();
    let b = p.register(source(19, 64, 64)).unwrap();
    let mut other = PublicExternalPool::new(budget(1, 64), STREAM).unwrap();
    let c = other.register(source(23, 64, 64)).unwrap();
    assert_eq!(p.resolve(&a.handle().binding()).unwrap(), 17);
    assert_eq!(p.resolve(&b.handle().binding()).unwrap(), 19);
    assert!(p.resolve(&c.handle().binding()).is_err());
    assert!(p.resolve(&Handle::new(STREAM, 64).binding()).is_err());
    assert_eq!(
        p.resolve(&a.handle().offset_start(4).offset_end(4).binding())
            .unwrap(),
        17
    );
    for h in [
        a.handle().offset_start(1),
        a.handle().offset_end(1),
        a.handle().offset_start(64),
        a.handle().offset_start(u64::MAX),
        a.handle().offset_start(32).offset_end(32),
    ] {
        assert!(p.resolve(&h.binding()).is_err());
    }
    let mut h = a.handle();
    h.stream.value += 1;
    assert!(p.resolve(&h.binding()).is_err());
    let altered = Handle::from_memory(a.handle().memory, STREAM, 32);
    assert!(p.resolve(&altered.binding()).is_err());
}
#[test]
fn detach_revokes_and_failed_join_retains_accounting_and_never_reuses_slot() {
    let mut p = PublicExternalPool::new(budget(2, 128), STREAM).unwrap();
    let a = p.register(source(17, 32, 64)).unwrap();
    let old = a.handle().binding();
    let before = p.usage();
    assert_eq!(p.begin_detach(&a).unwrap(), 17);
    assert!(p.resolve(&old).is_err());
    assert!(p.begin_detach(&a).is_err());
    assert_eq!(p.usage(), before);
    assert_eq!(p.resources().count(), 1);
    // SAFETY: fake numeric resource representing successful destroy and two joins.
    let r = unsafe { p.finish_detach(a) }.unwrap();
    assert_eq!((r.buffer_bytes, r.backing_bytes), (32, 64));
    assert_eq!(*r.resource(), 17);
    assert!(r.queue_joined && r.source_destroyed && r.matches_source(&old));
    assert_eq!(
        (
            p.usage().registered_backing_bytes,
            p.usage().total_detached_bytes
        ),
        (0, 64)
    );
    assert!(p.resolve(&old).is_err());
    let b = p.register(source(19, 32, 64)).unwrap();
    assert!(!r.matches_source(&b.handle().binding()));
    assert!(!r.matches_source(&Handle::new(STREAM, 32).binding()));
    let mut sliced = old.clone();
    sliced.offset_start = Some(0);
    assert!(!r.matches_source(&sliced));
    let before = p.usage();
    assert_eq!(before.registration_count, 2);
    assert!(p.register(source(21, 32, 32)).is_err());
    assert_eq!(p.usage(), before);
}
#[test]
fn unbegun_finish_cannot_issue_release_credit() {
    let mut p = PublicExternalPool::new(budget(1, 32), STREAM).unwrap();
    let a = p.register(source(17, 32, 32)).unwrap();
    let saved = a.handle().binding();
    let before = p.usage();
    // Invalid transaction preflight on purely fake public resource.
    assert!(unsafe { p.finish_detach(a) }.is_err());
    assert_eq!(p.usage(), before);
    assert_eq!(p.resolve(&saved).unwrap(), 17);
}
#[test]
fn private_clear_rejects_external_but_readonly_metadata_remains_legal() {
    let mut p = PublicExternalPool::new(budget(1, 32), STREAM).unwrap();
    let a = p.register(source(17, 32, 32)).unwrap();
    let bs = [a.handle().binding()];
    assert!(cubecl_runtime::transient::retained_metadata_bindings(&bs, STREAM).is_ok());
    assert!(cubecl_runtime::transient::clear_owner_bindings(&bs, STREAM).is_err());
    assert!(
        cubecl_runtime::transient::clear_owner_bindings(
            &[Handle::new(STREAM, 32).binding()],
            STREAM
        )
        .is_ok()
    );
}
#[test]
fn external_mask_and_launch_stream_restriction_keep_ordinary_behavior() {
    let mut p = PublicExternalPool::new(budget(1, 32), STREAM).unwrap();
    let a = p.register(source(17, 32, 32)).unwrap();
    let mut args = KernelArguments {
        buffers: vec![
            Handle::new(StreamId { value: 999 }, 32).binding(),
            a.handle().binding(),
        ],
        ..Default::default()
    };
    assert_eq!(subject::external_mask(&args).unwrap(), 2);
    assert!(subject::validate_stream(&args, STREAM).is_ok());
    assert!(subject::validate_stream(&args, StreamId { value: 999 }).is_err());
    args.buffers.remove(1);
    assert!(subject::validate_stream(&args, STREAM).is_ok());
    args.buffers = (0..33).map(|_| Handle::new(STREAM, 32).binding()).collect();
    assert_eq!(subject::external_mask(&args).unwrap(), 0);
    args.buffers[32] = a.handle().binding();
    assert!(subject::external_mask(&args).is_err());
}
#[test]
fn kernel_cache_identity_separates_mask_and_preserves_original_comptime() {
    use std::hash::{Hash, Hasher};
    fn hash(id: &KernelId) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        id.hash(&mut h);
        h.finish()
    }
    let ordinary = KernelId::new::<u32>().info((42u32, 64u32));
    let zero = ordinary.clone().public_external_read_mask(0);
    assert_eq!(ordinary, zero);
    assert_eq!(ordinary.stable_format(), zero.stable_format());
    assert_eq!(ordinary.stable_hash(), zero.stable_hash());
    let a = ordinary.clone().public_external_read_mask(1);
    let b = ordinary.clone().public_external_read_mask(2);
    let c = KernelId::new::<u32>()
        .info((43u32, 64u32))
        .public_external_read_mask(1);
    for other in [&a, &b, &c] {
        assert_ne!(&ordinary, other);
        assert_ne!(hash(&ordinary), hash(other));
        assert_ne!(ordinary.stable_hash(), other.stable_hash());
    }
    assert_ne!(a, b);
    assert_ne!(a, c);
    assert_ne!(hash(&a), hash(&b));
}
#[cube]
fn numeric(a: &Array<f32>, b: &Array<f32>, out: &mut Array<f32>) {
    let x = a[UNIT_POS as usize] * 2.0;
    let y = x + b[UNIT_POS as usize];
    out[UNIT_POS as usize] = y;
}
fn generated(atomic: bool) -> CompiledKernel<AutoCompiler> {
    use cubecl::ir::{ElemType, FloatKind, StorageType, Type, UIntKind};
    use cubecl::{Compiler, KernelSettings, WgpuCompilationOptions, compute::KernelBuilder};
    let uint = StorageType::Scalar(ElemType::UInt(UIntKind::U32));
    let float = StorageType::Scalar(ElemType::Float(FloatKind::F32));
    let mut b = KernelBuilder::default();
    b.scope.register_type::<usize>(uint);
    b.scope.register_type::<u32>(uint);
    b.scope.register_type::<f32>(float);
    b.scope
        .register_type::<bool>(StorageType::Scalar(ElemType::Bool));
    if atomic {
        let _a = b.input_array(Type::new(StorageType::Atomic(ElemType::UInt(
            UIntKind::U32,
        ))));
    } else {
        let a = b.input_array(Type::new(float)).into();
        let c = b.input_array(Type::new(float)).into();
        let out = b.output_array(Type::new(float)).into();
        numeric::expand(&mut b.scope, a, c, out);
    }
    let repr = cubecl_wgpu::WgslCompiler::default()
        .compile(
            b.build(
                KernelSettings::default()
                    .cube_dim(CubeDim::new_1d(64))
                    .kernel_name("public_external_numeric"),
            ),
            &WgpuCompilationOptions::default(),
            ExecutionMode::Checked,
            uint,
        )
        .unwrap();
    CompiledKernel {
        source: repr.to_string(),
        repr: Some(repr.into()),
        cube_dim: CubeDim::new_1d(64),
        entrypoint_name: "public_external_numeric".into(),
        debug_name: None,
        debug_info: None,
    }
}
#[test]
fn selected_original_read_args_become_load_without_numeric_source_change() {
    let mut c = generated(false);
    let old = c.source.clone();
    assert!(old.contains("var<storage, read_write>"));
    subject::lower(&mut c, 0).unwrap();
    assert_eq!(c.source, old);
    subject::lower(&mut c, 3).unwrap();
    subject::check_readonly_source(&c.source, 3).unwrap();
    assert!(subject::check_readonly_source(&c.source, 4).is_err());
    assert_eq!(
        &old[old.find("@compute").unwrap()..],
        &c.source[c.source.find("@compute").unwrap()..]
    );
    assert_eq!(
        c.source.replace(
            "var<storage, read> buffer_",
            "var<storage, read_write> buffer_"
        ),
        old
    );
}
#[test]
fn writable_atomic_and_out_of_inventory_masks_reject_without_source_mutation() {
    let mut c = generated(false);
    let old = c.source.clone();
    for mask in [4, 8, u32::MAX] {
        assert!(subject::lower(&mut c, mask).is_err());
        assert_eq!(c.source, old);
    }
    let mut c = generated(true);
    let old = c.source.clone();
    assert!(subject::lower(&mut c, 1).is_err());
    assert_eq!(c.source, old);
}
#[test]
fn custom_wgsl_must_already_be_load_only_nonatomic_and_complete() {
    let ro = "@group(0) @binding(0) var<storage,read> a:array<f32>; @group(0) @binding(1) var<storage,read_write> out:array<f32>; @compute @workgroup_size(1) fn main(){out[0]=a[0]*2.0;}";
    let mut c = generated(false);
    c.repr = None;
    c.source = ro.into();
    subject::lower(&mut c, 1).unwrap();
    assert_eq!(c.source, ro);
    assert!(subject::lower(&mut c, 2).is_err());
    assert!(subject::lower(&mut c, 4).is_err());
    c.source = ro.replace("storage,read>", "storage,read_write>");
    assert!(subject::lower(&mut c, 1).is_err());
    c.source="@group(0) @binding(0) var<storage,read> a:array<atomic<u32>>; @compute @workgroup_size(1) fn main(){let x=atomicLoad(&a[0]);}".into();
    assert!(subject::lower(&mut c, 1).is_err());
    c.source = "broken".into();
    assert!(subject::lower(&mut c, 1).is_err());
}

#[derive(Default)]
struct PublicNumericStorage {
    allocs: usize,
}
impl cubecl_runtime::storage::ComputeStorage for PublicNumericStorage {
    type Resource = cubecl_runtime::storage::StorageId;
    fn alignment(&self) -> usize {
        32
    }
    fn get(&mut self, h: &cubecl_runtime::storage::StorageHandle) -> Self::Resource {
        h.id
    }
    fn alloc(
        &mut self,
        size: u64,
    ) -> Result<cubecl_runtime::storage::StorageHandle, cubecl_runtime::server::IoError> {
        self.allocs += 1;
        Ok(cubecl_runtime::storage::StorageHandle::new(
            cubecl_runtime::storage::StorageId::new(),
            cubecl_runtime::storage::StorageUtilization { offset: 0, size },
        ))
    }
    fn dealloc(&mut self, _: Self::Resource) {}
    fn flush(&mut self) {}
}
#[test]
fn external_identity_never_resolves_or_rebinds_through_real_ordinary_allocator() {
    use cubecl_runtime::{
        logging::ServerLogger,
        memory_management::{
            MemoryAllocationMode, MemoryConfiguration, MemoryManagement, MemoryManagementOptions,
        },
    };
    let mut ordinary = MemoryManagement::from_configuration(
        PublicNumericStorage::default(),
        &cubecl_ir::MemoryDeviceProperties {
            alignment: 32,
            max_page_size: 1024,
        },
        MemoryConfiguration::ExclusivePages,
        std::sync::Arc::new(ServerLogger::default()),
        MemoryManagementOptions::new("numeric-public-test").mode(MemoryAllocationMode::Auto),
    );
    let mut p = PublicExternalPool::new(budget(1, 64), STREAM).unwrap();
    let a = p.register(source(17, 32, 64)).unwrap();
    assert_eq!(ordinary.storage().allocs, 0);
    assert_eq!(ordinary.memory_usage().number_allocs, 0);
    assert!(ordinary.get_storage(a.handle().binding().memory).is_err());
    let standard = ordinary.reserve(32).unwrap();
    let before = ordinary.memory_usage();
    assert!(
        ordinary
            .bind(standard.clone(), a.handle().memory, 0)
            .is_err()
    );
    assert!(
        ordinary
            .bind(a.handle().memory, standard.clone(), 0)
            .is_err()
    );
    assert_eq!(ordinary.memory_usage().number_allocs, before.number_allocs);
    assert_eq!(
        ordinary.memory_usage().bytes_reserved,
        before.bytes_reserved
    );
    assert!(
        ordinary
            .get_resource(standard.binding(), None, None)
            .is_ok()
    );
}
#[test]
fn custom_selected_nested_atomic_rejects_but_unselected_fault_atomic_is_allowed() {
    let ro = "@group(0) @binding(0) var<storage,read> a:array<f32>; @group(0) @binding(1) var<storage,read_write> fault:array<atomic<u32>>; @compute @workgroup_size(1) fn main(){let x=a[0]; atomicOr(&fault[0], bitcast<u32>(x));}";
    assert!(subject::check_readonly_source(ro, 1).is_ok());
    let nested = "struct A{v:atomic<u32>} @group(0) @binding(0) var<storage,read> a:array<A>; @compute @workgroup_size(1) fn main(){let x=atomicLoad(&a[0].v);}";
    assert!(subject::check_readonly_source(nested, 1).is_err());
}
