//! Compiled PUBLIC Q2_0 weights, read directly from explicit non-device-local
//! Vulkan host backing through the ordinary Rust-authored CubeCL decoder.
//! Caller supplies ONE actual adapter/device/queue/client setup. No new device,
//! model/prompt input, production import, full mirror, or speed claim is made.
//! Source detach authority precedes the separate caller-owned backing close;
//! every failure before that authority deliberately retains backing+device.
use crate::{
    main_admission::{self, MainAdmission},
    private_arena::{BufferId, PrivateArena},
    public_host_weight_pool as pool,
};
use cubecl::{prelude::*, server::Handle, wgpu::WgpuRuntime};
use cubecl_runtime::public_external::{PublicExternalBudget, PublicExternalHostSource};
use std::{
    ffi::OsString,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
use zeroize::{Zeroize, Zeroizing};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const WORDS: usize = 8192;
const PRIVATE_BYTES: u64 = (WORDS * 4) as u64;
const INPUT_FIRST: usize = 3;
const OUTPUT_FIRST: usize = 8;
const DOT_FIRST: usize = OUTPUT_FIRST + pool::ROWS * pool::COLUMNS;
const CASES: usize = 7;
const NORMAL_CASES: usize = 2;
const MAIN_CAP: u64 = 1024 * 1024;
const IO_CAP: u64 = 16 * 1024 * 1024;
const HOST_CAP: u64 = 8 * 1024 * 1024;
const REGISTRY_METADATA: usize = 1024 * 1024;

pub(crate) fn registration(
    args: &[OsString],
    env: impl IntoIterator<Item = (OsString, OsString)>,
    allowed_build: bool,
) -> std::result::Result<PathBuf, &'static str> {
    if !allowed_build
        || args.len() != 3
        || !matches!(args[1].to_str(), Some("intel" | "nvidia"))
        || args[2] != "public-host-weight-cubecl-probe"
    {
        return Err("public host CubeCL fixed card/build required");
    }
    let (mut gate, mut receipt) = (false, None);
    for (key, value) in env {
        if ![b"KUNO_LAB_".as_slice(), b"KUNO_TEST_", b"KUNO_PUBLIC_"]
            .iter()
            .any(|prefix| key.as_encoded_bytes().starts_with(prefix))
        {
            continue;
        }
        if key == "KUNO_LAB_PUBLIC_HOST_CUBECL_ONLY" && !gate && value == "1" {
            gate = true;
        } else if key == "KUNO_PUBLIC_RECEIPT" && receipt.is_none() && !value.is_empty() {
            receipt = Some(PathBuf::from(value));
        } else {
            return Err("public host CubeCL inherited/duplicate option rejected");
        }
    }
    if !gate {
        return Err("public host CubeCL explicit public gate required");
    }
    receipt.ok_or("public host CubeCL new numeric receipt required")
}
pub(crate) fn validate_limits(limits: &wgpu::Limits) -> std::result::Result<(), &'static str> {
    if limits.max_compute_invocations_per_workgroup < 64
        || limits.max_compute_workgroup_size_x < 64
        || limits.max_compute_workgroup_storage_size < 512
        || limits.max_storage_buffers_per_shader_stage < 4
        || limits.max_bindings_per_bind_group < 5
        || limits.max_bind_groups < 1
        || limits.max_storage_buffer_binding_size < pool::BUFFER_BYTES
        || limits.max_buffer_size < pool::BUFFER_BYTES
        || limits.max_uniform_buffer_binding_size < 2048
        || limits.max_compute_workgroups_per_dimension < 1
    {
        return Err("public host CubeCL actual device limits rejected");
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fault {
    None,
    SourceTail,
    InputNan,
    InputTail,
    OutputDirty,
    Sticky,
}
impl Fault {
    fn bits(self) -> u32 {
        match self {
            Self::None => 0,
            Self::SourceTail => 1,
            Self::InputNan => 2,
            Self::InputTail => 4,
            Self::OutputDirty => 8,
            Self::Sticky => 256 | 32,
        }
    }
}

/// Complete owner checks precede every publication. The reduction barriers are
/// unconditional, including invalid inputs, so a fault never changes liveness.
/// No source stores, activation quantization or fused multiply-add are used.
// CubeCL supports explicit comparisons rather than Range::contains/partial_cmp;
// !(abs<=MAX) deliberately rejects NaN as well as infinity.
#[allow(clippy::manual_range_contains, clippy::neg_cmp_op_on_partial_ord)]
#[cube(launch)]
fn public_host_q2(
    source: &Array<u32>,
    input: &Array<f32>,
    output: &mut Array<f32>,
    fault: &mut Array<Atomic<u32>>,
    #[comptime] prefix: usize,
) {
    let lane = UNIT_POS as usize;
    let mut bad = 0u32;
    let mut word = lane;
    while word < pool::BUFFER_BYTES as usize / 4 {
        let value = source[word];
        for byte in 0usize..4usize {
            let index = word * 4 + byte;
            if (index < prefix || index >= prefix + pool::ROWS * pool::BLOCK_BYTES)
                && ((value >> (byte as u32 * 8)) & 255) != 0
            {
                bad |= 1;
            }
        }
        word += 64;
    }
    let mut index = lane;
    while index < WORDS {
        if index >= INPUT_FIRST && index < INPUT_FIRST + pool::COLUMNS {
            if !(input[index].abs() <= f32::MAX) {
                bad |= 2;
            }
        } else if u32::reinterpret(input[index]) != 0 {
            bad |= 4;
        }
        if u32::reinterpret(output[index]) != 0 {
            bad |= 8;
        }
        if fault[index].load() != 0 {
            bad |= 32;
        }
        index += 64;
    }
    let mut flags = SharedMemory::<u32>::new(64usize);
    flags[lane] = bad;
    sync_cube();
    let mut width = 32usize;
    while width > 0 {
        if lane < width {
            flags[lane] |= flags[lane + width];
        }
        sync_cube();
        width /= 2;
    }
    if lane == 0 {
        fault[0].fetch_or(flags[0]);
    }
    let mut sums = SharedMemory::<f32>::new(64usize);
    for row in 0usize..pool::ROWS {
        let weight = crate::quant_matvec::packed_value(
            source,
            prefix + row * pool::BLOCK_BYTES,
            lane,
            crate::quant_matvec::Q2_0,
        );
        sums[lane] = weight * input[INPUT_FIRST + lane];
        sync_cube();
        // EXACT original low+high, then 16/8/4/2/1 column%64 tree.
        if lane < 32 {
            sums[lane] += sums[lane + 32];
        }
        sync_cube();
        let mut stride = 16usize;
        while stride > 0 {
            if lane < stride {
                sums[lane] += sums[lane + stride];
            }
            sync_cube();
            stride /= 2;
        }
        if flags[0] == 0 {
            output[OUTPUT_FIRST + row * pool::COLUMNS + lane] = weight;
            if lane == 0 {
                output[DOT_FIRST + row] = sums[0];
            }
        }
        sync_cube();
    }
}

fn input() -> Zeroizing<Vec<u32>> {
    let mut words = Zeroizing::new(vec![0; WORDS]);
    for column in 0..pool::COLUMNS {
        words[INPUT_FIRST + column] = (((column * 13 % 17) as f32 - 8.) / 8.).to_bits();
    }
    words
}
fn expected(fixture: pool::Fixture, input: &[u32]) -> Result<Zeroizing<Vec<u32>>> {
    if input.len() != WORDS {
        return Err("public host CubeCL CPU input extent".into());
    }
    let mut weights = Zeroizing::new(vec![0.; pool::ROWS * pool::COLUMNS]);
    kuno_text::TensorType::from_gguf_id(42)?.decode(&fixture.packed(), &mut weights)?;
    let mut output = Zeroizing::new(vec![0; WORDS]);
    for row in 0..pool::ROWS {
        let mut lanes = [0f32; 64];
        for column in 0..64 {
            let weight = weights[row * 64 + column];
            output[OUTPUT_FIRST + row * 64 + column] = weight.to_bits();
            lanes[column] = weight * f32::from_bits(input[INPUT_FIRST + column]);
        }
        for lane in 0..32 {
            lanes[lane] += lanes[lane + 32];
        }
        for step in [16, 8, 4, 2, 1] {
            for lane in 0..step {
                lanes[lane] += lanes[lane + step];
            }
        }
        output[DOT_FIRST + row] = lanes[0].to_bits();
        lanes.zeroize();
    }
    Ok(output)
}
fn fixture_sha256(fixture: pool::Fixture) -> [u8; 32] {
    // SHA256 of every byte of the existing immutable 64KiB fixture, including
    // prefix/gaps/tail. Active bytes are independently decoded by kuno-text.
    // These are provenance attestations for this compiled source, not a digest
    // supplied by a user or backend authentication of arbitrary model bytes.
    match fixture {
        pool::Fixture::Aligned => [
            0xa9, 0x8a, 0x4e, 0x84, 0x6b, 0xed, 0x3f, 0x02, 0x5c, 0x9d, 0x5a, 0x58, 0x51, 0xbd,
            0x3f, 0xef, 0x6f, 0xb9, 0x01, 0x1e, 0xbf, 0xd8, 0xb5, 0x7b, 0x22, 0x94, 0x32, 0x57,
            0x0a, 0x4a, 0x8a, 0xeb,
        ],
        pool::Fixture::Prefix3 => [
            0x6c, 0x34, 0x1a, 0x06, 0x26, 0x64, 0x57, 0x3c, 0x56, 0xd1, 0xf2, 0x5c, 0xc9, 0x13,
            0x84, 0x72, 0x8a, 0x86, 0xcb, 0xc7, 0x0b, 0x8a, 0xcb, 0x45, 0xbd, 0x14, 0xac, 0x35,
            0x16, 0x27, 0x1f, 0x0e,
        ],
        pool::Fixture::BadPublicTail => [
            0xc3, 0xe0, 0xc1, 0x4b, 0xee, 0xad, 0xfa, 0xc7, 0xfc, 0x09, 0x48, 0x02, 0x03, 0xbf,
            0x63, 0x3f, 0x98, 0x8d, 0x8d, 0x70, 0x2e, 0x83, 0x93, 0x1e, 0x8d, 0x4c, 0x59, 0x5d,
            0x07, 0x1c, 0x91, 0x4d,
        ],
    }
}

struct Upload(Vec<u8>);
impl Drop for Upload {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}
struct Read(cubecl::bytes::Bytes);
impl Drop for Read {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}
fn write(arena: &PrivateArena, id: BufferId, words: &[u32], retired: &AtomicBool) -> Result<()> {
    if words.len() != arena.layout(id)?.owned_words {
        return Err("public host CubeCL complete write extent".into());
    }
    let mut bytes = Upload(words.iter().flat_map(|v| v.to_le_bytes()).collect());
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        arena
            .client()?
            .write_private(arena.handle(id)?.clone(), std::mem::take(&mut bytes.0))
            .map_err(|_| "public host CubeCL guarded write failed")?;
        Ok::<_, Box<dyn std::error::Error>>(())
    }));
    if !matches!(result, Ok(Ok(()))) {
        retired.store(true, Ordering::Release);
        return Err("public host CubeCL write failed; context retired".into());
    }
    Ok(())
}
fn read(arena: &PrivateArena, id: BufferId, retired: &AtomicBool) -> Result<Zeroizing<Vec<u32>>> {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let data = Read(
            arena
                .client()?
                .read_one(arena.handle(id)?.clone())
                .map_err(|_| "public host CubeCL completed read failed")?,
        );
        if data.0.len() != arena.layout(id)?.owned_bytes {
            return Err("public host CubeCL read full physical extent".into());
        }
        Ok::<_, Box<dyn std::error::Error>>(Zeroizing::new(
            data.0
                .chunks_exact(4)
                .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
                .collect(),
        ))
    }));
    match result {
        Ok(Ok(words)) => Ok(words),
        _ => {
            retired.store(true, Ordering::Release);
            Err("public host CubeCL read failed; context retired".into())
        }
    }
}
fn audit(client: &ComputeClient<WgpuRuntime>, retired: &AtomicBool) -> Result<()> {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cubecl::future::block_on(client.sanitize_transient_io_audited())
    }));
    let r = match result {
        Ok(Ok(r)) => r,
        _ => {
            retired.store(true, Ordering::Release);
            return Err("public host CubeCL checked I/O cleanup failed".into());
        }
    };
    let total = r
        .upload_bytes
        .checked_add(r.uniform_bytes)
        .and_then(|n| n.checked_add(r.readback_bytes));
    if total.is_none_or(|n| n == 0 || n > IO_CAP || n != r.inspected_bytes)
        || r.inspected_buffers == 0
        || retired.load(Ordering::Acquire)
    {
        retired.store(true, Ordering::Release);
        return Err("public host CubeCL malformed full I/O audit".into());
    }
    Ok(())
}

/// The existing pool's raw-wgpu Drop is appropriate before exposure; after the
/// unsafe source declaration ANY error must retain it until checked detach.
/// A partial channel/register failure cannot be treated as nonregistration.
struct RetainOnFailure(Option<pool::PublicHostWeightPool>);
impl Drop for RetainOnFailure {
    fn drop(&mut self) {
        if let Some(source) = self.0.take() {
            std::mem::forget(source);
        }
    }
}
fn physical(
    client: &ComputeClient<WgpuRuntime>,
    arena: &PrivateArena,
    ids: &[BufferId; 3],
    source: &Handle,
    actual: &wgpu::Buffer,
) -> Result<()> {
    if source.size_in_used() != pool::BUFFER_BYTES || source.can_mut() {
        return Err("public host CubeCL immutable source extent/class".into());
    }
    let mut buffers = Vec::with_capacity(4);
    for (handle, bytes) in ids
        .iter()
        .map(|&id| Ok((arena.handle(id)?.clone(), PRIVATE_BYTES)))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .chain(std::iter::once((source.clone(), pool::BUFFER_BYTES)))
    {
        if handle.stream != source.stream || handle.size_in_used() != bytes {
            return Err("public host CubeCL complete owner stream/extent".into());
        }
        let resource = client
            .get_resource(handle)
            .map_err(|_| "public host CubeCL complete owner resource")?;
        let resource = resource.resource();
        if resource.offset != 0
            || resource.size != bytes
            || resource.buffer.size() != bytes
            || buffers.contains(&resource.buffer)
        {
            return Err("public host CubeCL physical disjoint full owners".into());
        }
        buffers.push(resource.buffer.clone());
    }
    if buffers[3] != *actual || actual.usage() != wgpu::BufferUsages::STORAGE {
        return Err("public host CubeCL actual imported buffer identity/usage".into());
    }
    Ok(())
}
fn validate_readback(
    actual: &[Zeroizing<Vec<u32>>; 3],
    fixture: pool::Fixture,
    kind: Fault,
    initial: &[Zeroizing<Vec<u32>>; 3],
) -> Result<()> {
    if actual
        .iter()
        .chain(initial)
        .any(|words| words.len() != WORDS)
        || actual[0] != initial[0]
    {
        return Err("public host CubeCL full readback/immutable input".into());
    }
    if kind == Fault::None {
        let oracle = expected(fixture, &initial[0])?;
        if actual[1] != oracle || actual[2].iter().any(|&word| word != 0) {
            return Err("public host CubeCL independent original packed bits/full owner".into());
        }
    } else if actual[1] != initial[1]
        || actual[2][0] != kind.bits()
        || actual[2][1..] != initial[2][1..]
        || actual[2][0] & initial[2][0] != initial[2][0]
    {
        return Err("public host CubeCL fault full-owner no publication".into());
    }
    Ok(())
}
struct Case {
    placement: pool::Placement,
    allocation: u64,
    zero_bytes: u64,
    zero_buffers: u64,
}
fn case(
    device: &wgpu::Device,
    client: &ComputeClient<WgpuRuntime>,
    fixture: pool::Fixture,
    kind: Fault,
    retired: Arc<AtomicBool>,
    ledger: &mut MainAdmission,
) -> Result<Case> {
    let mut source = RetainOnFailure(Some(pool::PublicHostWeightPool::new(device, fixture)?));
    let backing = source.0.as_ref().unwrap();
    let placement = backing.placement();
    let allocation = backing.allocated_bytes();
    let before = client
        .memory_usage()
        .map_err(|_| "public host CubeCL main sample")?;
    // SAFETY: source is the existing completely initialized immutable compiled
    // fixture on this exact supplied device. Actual placement flags are obtained
    // from its Vulkan allocation. No other registry/raw consumer is created.
    // RetainOnFailure keeps backing/device alive on ANY later channel/queue error.
    let declared = unsafe {
        PublicExternalHostSource::from_initialized_public_host(
            cubecl::wgpu::WgpuResource::new(backing.source().clone(), 0, pool::BUFFER_BYTES),
            pool::BUFFER_BYTES,
            allocation,
            placement.property_flags,
            placement.heap_flags,
            fixture_sha256(fixture),
        )
    }?;
    let registration = client
        .register_public_external_host(declared)
        .map_err(|_| "public host CubeCL registration failed; retain backing")?;
    let handle = registration.handle();
    let binding = handle.clone().binding();
    let after = client
        .memory_usage()
        .map_err(|_| "public host CubeCL main sample")?;
    if before.number_allocs != after.number_allocs
        || before.bytes_in_use != after.bytes_in_use
        || before.bytes_padding != after.bytes_padding
        || before.bytes_reserved != after.bytes_reserved
    {
        return Err("public host CubeCL import must not allocate managed main memory".into());
    }
    ledger.observe_before(after.bytes_reserved)?;
    ledger.precharge(&[PRIVATE_BYTES; 3])?;
    let mut arena = PrivateArena::new_transient_on_configured(
        client.clone(),
        MAIN_CAP as usize,
        retired.clone(),
    )?;
    arena.enable_public_owned_zero_audit()?;
    let ids = [
        arena.allocate_zeroed(WORDS)?,
        arena.allocate_zeroed(WORDS)?,
        arena.allocate_zeroed(WORDS)?,
    ];
    physical(client, &arena, &ids, &handle, backing.source())?;
    ledger.observe_after(
        client
            .memory_usage()
            .map_err(|_| "public host CubeCL main sample")?
            .bytes_reserved,
    )?;
    let mut initial = [
        input(),
        Zeroizing::new(vec![0; WORDS]),
        Zeroizing::new(vec![0; WORDS]),
    ];
    match kind {
        Fault::InputNan => initial[0][INPUT_FIRST + 7] = f32::NAN.to_bits(),
        Fault::InputTail => initial[0][WORDS - 1] = 1,
        Fault::OutputDirty => initial[1][WORDS - 1] = 0xd00d,
        Fault::Sticky => initial[2][0] = 256,
        Fault::None | Fault::SourceTail => (),
    }
    for (&id, words) in ids.iter().zip(&initial) {
        write(&arena, id, words, &retired)?;
    }
    let launched = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: exact complete same-stream and physically disjoint owners,
        // immutable source class/extent and fixed geometry were proved above.
        public_host_q2::launch::<WgpuRuntime>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(64),
            unsafe { ArrayArg::from_raw_parts(handle.clone(), pool::BUFFER_BYTES as usize / 4) },
            unsafe { ArrayArg::from_raw_parts(arena.handle(ids[0])?.clone(), WORDS) },
            unsafe { ArrayArg::from_raw_parts(arena.handle(ids[1])?.clone(), WORDS) },
            unsafe { ArrayArg::from_raw_parts(arena.handle(ids[2])?.clone(), WORDS) },
            fixture.prefix(),
        );
        Ok::<_, Box<dyn std::error::Error>>(())
    }));
    if !matches!(launched, Ok(Ok(()))) {
        retired.store(true, Ordering::Release);
        return Err("public host CubeCL launch failed; retain backing".into());
    }
    let actual = [
        read(&arena, ids[0], &retired)?,
        read(&arena, ids[1], &retired)?,
        read(&arena, ids[2], &retired)?,
    ];
    audit(client, &retired)?;
    validate_readback(&actual, fixture, kind, &initial)?;
    ledger.completed_fence(
        client
            .memory_usage()
            .map_err(|_| "public host CubeCL main sample")?
            .bytes_reserved,
    )?;
    drop((actual, initial));
    arena.close()?;
    let zero = arena
        .owned_zero_audit_receipt()
        .ok_or("public host CubeCL full private zero receipt")?;
    if zero.bytes != PRIVATE_BYTES * 3 || zero.buffers != 3 || retired.load(Ordering::Acquire) {
        return Err("public host CubeCL private zero extent/context".into());
    }
    drop(arena);
    let detached = client
        .detach_public_external_host(registration)
        .map_err(|_| "public host CubeCL checked detach failed; retain backing")?;
    if !detached.matches_source(&binding)
        || detached.resource().buffer != *backing.source()
        || detached.resource().offset != 0
        || detached.resource().size != pool::BUFFER_BYTES
        || detached.buffer_bytes != pool::BUFFER_BYTES
        || detached.backing_bytes != allocation
        || !detached.queue_joined
        || !detached.source_destroyed
    {
        return Err("public host CubeCL typed detach source identity/extent".into());
    }
    drop((handle, binding, detached));
    // This is the ONLY handoff that disarms the leak-on-failure guard. The
    // separate pool now performs actual CPU full-allocation verify/zero/free.
    let close = source.0.take().unwrap().close()?;
    if close.allocation_bytes != allocation
        || !close.source_unchanged
        || !close.full_allocation_zero
        || !close.ledger_closed
    {
        return Err("public host CubeCL caller backing full close".into());
    }
    cubecl::future::block_on(client.sync()).map_err(|_| "public host CubeCL checked case join")?;
    client.memory_cleanup();
    cubecl::future::block_on(client.sync())
        .map_err(|_| "public host CubeCL checked main cleanup")?;
    ledger.completed_fence(
        client
            .memory_usage()
            .map_err(|_| "public host CubeCL main sample")?
            .bytes_reserved,
    )?;
    Ok(Case {
        placement,
        allocation,
        zero_bytes: zero.bytes,
        zero_buffers: zero.buffers,
    })
}

/// Caller preserves the actual WgpuSetup device/queue/adapter before init_device
/// and supplies its resulting CubeCL client. Registration validates the actual
/// source against that client too; equal Vulkan identities alone are not used
/// as proof of CubeCL ownership. The fresh serialized stream is used exclusively.
pub(crate) fn run(
    adapter: &wgpu::Adapter,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    client: &ComputeClient<WgpuRuntime>,
    limits: &wgpu::Limits,
) -> Result<serde_json::Value> {
    validate_limits(limits)?;
    validate_limits(&device.limits())?;
    // SAFETY: borrow actual identity only, never submit/destroy via raw handles.
    let a = unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan>() }
        .ok_or("public host CubeCL Vulkan adapter")?;
    let d = unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() }
        .ok_or("public host CubeCL Vulkan device")?;
    let q = unsafe { queue.as_hal::<wgpu::hal::api::Vulkan>() }
        .ok_or("public host CubeCL Vulkan queue")?;
    if a.raw_physical_device() != d.raw_physical_device()
        || a.shared_instance().raw_instance().handle()
            != d.shared_instance().raw_instance().handle()
        || q.raw_device().handle() != d.raw_device().handle()
        || q.as_raw() != d.raw_queue()
        || adapter.get_info().device_type != wgpu::DeviceType::DiscreteGpu
    {
        return Err("public host CubeCL mixed setup/non-discrete device".into());
    }
    drop((a, d, q));
    main_admission::check_capacity(MAIN_CAP, &[PRIVATE_BYTES; 3])?;
    // Registry full typed ceiling, physical source backing, all initial/current
    // full snapshots, raw read leases/uploads/reference and bounded metadata.
    main_admission::check_capacity(
        HOST_CAP,
        &[
            pool::HOST_CAP,
            REGISTRY_METADATA as u64,
            PRIVATE_BYTES * 16,
            65536,
        ],
    )?;
    let baseline = client
        .memory_usage()
        .map_err(|_| "public host CubeCL initial main telemetry")?;
    if baseline.number_allocs != 0
        || baseline.bytes_in_use != 0
        || baseline.bytes_padding != 0
        || baseline.bytes_reserved != 0
    {
        return Err("public host CubeCL fresh exclusive main context required".into());
    }
    client
        .configure_transient_io(IO_CAP)
        .map_err(|_| "public host CubeCL initial I/O configuration")?;
    client
        .configure_public_external_host(PublicExternalBudget {
            backing_bytes: pool::HOST_CAP,
            buffers: CASES,
            metadata_bytes: REGISTRY_METADATA,
        })
        .map_err(|_| "public host CubeCL initial external metadata configuration")?;
    let retired = Arc::new(AtomicBool::new(false));
    let mut ledger = MainAdmission::new(MAIN_CAP, 0)?;
    let mut cases = Vec::with_capacity(CASES);
    let mut total = 0u64;
    for (fixture, kind) in [
        (pool::Fixture::Aligned, Fault::None),
        (pool::Fixture::Prefix3, Fault::None),
        (pool::Fixture::BadPublicTail, Fault::SourceTail),
        (pool::Fixture::Prefix3, Fault::InputNan),
        (pool::Fixture::Prefix3, Fault::InputTail),
        (pool::Fixture::Prefix3, Fault::OutputDirty),
        (pool::Fixture::Prefix3, Fault::Sticky),
    ] {
        cases.push(case(
            device,
            client,
            fixture,
            kind,
            retired.clone(),
            &mut ledger,
        )?);
        total = total
            .checked_add(cases.last().unwrap().allocation)
            .ok_or("public host CubeCL backing sum overflow")?;
        let usage = client
            .public_external_host_usage()
            .map_err(|_| "public host CubeCL external ownership telemetry")?;
        if usage.registered_backing_bytes != 0
            || usage.total_registered_bytes != total
            || usage.total_detached_bytes != total
            || usage.registration_count != cases.len()
            || usage.metadata_owned_bytes > REGISTRY_METADATA
        {
            return Err("public host CubeCL separate backing ledger/retired slot proof".into());
        }
    }
    let final_main = client
        .memory_usage()
        .map_err(|_| "public host CubeCL final main telemetry")?;
    if final_main.number_allocs != 0
        || final_main.bytes_in_use != 0
        || final_main.bytes_padding != 0
        || final_main.bytes_reserved != 0
        || retired.load(Ordering::Acquire)
    {
        return Err("public host CubeCL final managed cleanup/context".into());
    }
    let usage = client
        .public_external_host_usage()
        .map_err(|_| "public host CubeCL final host telemetry")?;
    Ok(serde_json::json!({
        "scope":"compiled_public_nonlocal_host_q2_cubecl_operator_only",
        "normal_cases":NORMAL_CASES,"fault_cases":CASES-NORMAL_CASES,
        "original_dtype":42,"rows":pool::ROWS,"columns":pool::COLUMNS,
        "exact_decoded_values":NORMAL_CASES*pool::ROWS*pool::COLUMNS,
        "exact_original64_dots":NORMAL_CASES*pool::ROWS,
        "source_load_only_reflection_required":true,"source_full_buffer_bytes":pool::BUFFER_BYTES,
        "memory_type_indices":cases.iter().map(|case|case.placement.memory_type).collect::<Vec<_>>(),
        "heap_indices":cases.iter().map(|case|case.placement.heap).collect::<Vec<_>>(),
        "memory_property_flags":cases.iter().map(|case|case.placement.property_flags).collect::<Vec<_>>(),
        "memory_heap_flags":cases.iter().map(|case|case.placement.heap_flags).collect::<Vec<_>>(),
        "physical_backing_bytes":cases.iter().map(|case|case.allocation).collect::<Vec<_>>(),
        "host_backing_registered_bytes":usage.total_registered_bytes,
        "host_backing_detached_bytes":usage.total_detached_bytes,
        "host_backing_live_bytes":usage.registered_backing_bytes,
        "registry_metadata_owned_bytes":usage.metadata_owned_bytes,
        "registration_count":usage.registration_count,
        "source_import_managed_allocations":0,"weight_upload_bytes":0,
        "full_private_zero_bytes":cases.iter().map(|case|case.zero_bytes).sum::<u64>(),
        "full_private_zero_buffers":cases.iter().map(|case|case.zero_buffers).sum::<u64>(),
        "full_source_backing_zero_bytes":total,"typed_detach_and_cpu_free":true,
        "fault_full_owner_no_publication":true,"managed_cleanup_complete":true,
        "main_cap_bytes":MAIN_CAP,"io_cap_bytes":IO_CAP,"host_cap_bytes":HOST_CAP,
        "registry_metadata_cap_bytes":REGISTRY_METADATA,
        "private_request_accepted":false,"whole_model_accepted":false,
        "whole_owning_erasure_accepted":false,"performance_accepted":false
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn registration_rejects_every_inherited_or_duplicate_lab_control() {
        let args = [
            "lab".into(),
            "intel".into(),
            "public-host-weight-cubecl-probe".into(),
        ];
        let env = [
            ("KUNO_LAB_PUBLIC_HOST_CUBECL_ONLY".into(), "1".into()),
            (
                "KUNO_PUBLIC_RECEIPT".into(),
                "/tmp/fresh-public-receipt".into(),
            ),
        ];
        assert!(registration(&args, env.clone(), true).is_ok());
        assert!(registration(&args, env.clone(), false).is_err());
        assert!(registration(&args[..2], env.clone(), true).is_err());
        for name in [
            "KUNO_LAB_OTHER",
            "KUNO_TEST_OTHER",
            "KUNO_PUBLIC_OTHER",
            "KUNO_LAB_PUBLIC_HOST_CUBECL_ONLY",
            "KUNO_PUBLIC_RECEIPT",
        ] {
            let mut wrong = env.to_vec();
            wrong.push((name.into(), "0".into()));
            assert!(registration(&args, wrong, true).is_err());
        }
    }
    #[test]
    fn complete_physical_limits_and_separate_reservations_are_required() {
        let mut limits = wgpu::Limits::default();
        validate_limits(&limits).unwrap();
        limits.max_compute_workgroup_storage_size = 511;
        assert!(validate_limits(&limits).is_err());
        limits = wgpu::Limits::default();
        limits.max_storage_buffer_binding_size = pool::BUFFER_BYTES - 1;
        assert!(validate_limits(&limits).is_err());
        limits = wgpu::Limits::default();
        limits.max_bindings_per_bind_group = 4;
        assert!(validate_limits(&limits).is_err());
        limits = wgpu::Limits::default();
        limits.max_uniform_buffer_binding_size = 2047;
        assert!(validate_limits(&limits).is_err());
        assert_eq!(
            main_admission::check_capacity(MAIN_CAP, &[PRIVATE_BYTES; 3]).unwrap(),
            98_304
        );
        assert!(
            main_admission::check_capacity(
                HOST_CAP,
                &[
                    pool::HOST_CAP,
                    REGISTRY_METADATA as u64,
                    16 * PRIVATE_BYTES,
                    65536
                ]
            )
            .is_ok()
        );
    }
    #[test]
    fn exact_oracle_and_every_complete_fault_owner_are_checked() {
        let fixture = pool::Fixture::Prefix3;
        let base = [
            input(),
            Zeroizing::new(vec![0; WORDS]),
            Zeroizing::new(vec![0; WORDS]),
        ];
        let mut actual = [
            base[0].clone(),
            expected(fixture, &base[0]).unwrap(),
            base[2].clone(),
        ];
        validate_readback(&actual, fixture, Fault::None, &base).unwrap();
        for owner in 0..3 {
            actual[owner][WORDS - 1] ^= 1;
            assert!(validate_readback(&actual, fixture, Fault::None, &base).is_err());
            actual[owner][WORDS - 1] ^= 1;
        }
        for kind in [
            Fault::SourceTail,
            Fault::InputNan,
            Fault::InputTail,
            Fault::OutputDirty,
            Fault::Sticky,
        ] {
            let mut before = base.clone();
            if kind == Fault::Sticky {
                before[2][0] = 256;
            }
            let mut after = before.clone();
            after[2][0] = kind.bits();
            validate_readback(&after, fixture, kind, &before).unwrap();
            after[1][OUTPUT_FIRST] ^= 1;
            assert!(validate_readback(&after, fixture, kind, &before).is_err());
            after[1][OUTPUT_FIRST] ^= 1;
            after[2][WORDS - 1] ^= 1;
            assert!(validate_readback(&after, fixture, kind, &before).is_err());
        }
        assert_eq!(
            expected(pool::Fixture::Aligned, &base[0]).unwrap(),
            expected(fixture, &base[0]).unwrap()
        );
    }
    fn shader(prefix: usize) -> String {
        use cubecl::compute::KernelBuilder;
        use cubecl::ir::{ElemType, FloatKind, IntKind, StorageType, Type, UIntKind};
        use cubecl::{Compiler, KernelSettings, WgpuCompilationOptions, wgpu::WgslCompiler};
        let uint = StorageType::Scalar(ElemType::UInt(UIntKind::U32));
        let float = StorageType::Scalar(ElemType::Float(FloatKind::F32));
        let mut b = KernelBuilder::default();
        b.scope.register_type::<u32>(uint);
        b.scope.register_type::<usize>(uint);
        b.scope.register_type::<f32>(float);
        b.scope
            .register_type::<i32>(StorageType::Scalar(ElemType::Int(IntKind::I32)));
        b.scope
            .register_type::<bool>(StorageType::Scalar(ElemType::Bool));
        let source = b.input_array(Type::new(uint)).into();
        let input = b.input_array(Type::new(float)).into();
        let output = b.output_array(Type::new(float)).into();
        let fault = b
            .output_array(Type::new(StorageType::Atomic(ElemType::UInt(
                UIntKind::U32,
            ))))
            .into();
        public_host_q2::expand(&mut b.scope, source, input, output, fault, prefix);
        let mut compiled = WgslCompiler::default()
            .compile(
                b.build(
                    KernelSettings::default()
                        .cube_dim(CubeDim::new_1d(64))
                        .kernel_name("fixed_public_host_q2_cubecl"),
                ),
                &WgpuCompilationOptions::default(),
                cubecl::server::ExecutionMode::Checked,
                uint,
            )
            .unwrap();
        compiled.set_public_external_readonly_mask(1).unwrap();
        compiled.to_string()
    }
    #[test]
    fn actual_generated_source_is_load_only_and_original_math_validates() {
        use wgpu::naga::{AddressSpace, StorageAccess};
        for prefix in [0, 3] {
            let source = shader(prefix);
            assert!(!source.contains("fma("));
            let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
            let info = wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::empty(),
            )
            .validate(&module)
            .unwrap();
            let globals = module
                .global_variables
                .iter()
                .filter(|(_, v)| v.binding.is_some())
                .collect::<Vec<_>>();
            let packed = globals
                .iter()
                .find(|(_, v)| v.binding.as_ref().unwrap().binding == 0)
                .unwrap();
            assert_eq!(
                packed.1.space,
                AddressSpace::Storage {
                    access: StorageAccess::LOAD
                }
            );
            assert_eq!(module.entry_points[0].workgroup_size, [64, 1, 1]);
            assert!(info.get_entry_point(0)[packed.0].contains(wgpu::naga::valid::GlobalUse::READ));
            assert!(
                !info.get_entry_point(0)[packed.0].contains(wgpu::naga::valid::GlobalUse::WRITE)
            );
        }
    }
}
