//! Rust Vulkan execution for the coherent Intel Strata port. MIT upstream
//! fb58e0dbc8399662c0e47c76578c6e878b14f6cf records a whole verification graph.
//! This backend owns native compute pipelines/descriptor sets/command buffers;
//! Wgpu retains all buffer allocation, upload, readback and privacy ledgers.
//! Only serialized fixed-public laboratory streams may select this backend.
use super::{WgpuResource, strata_schedule, transient_io::fault};
use ash::{vk, vk::Handle as _};
use cubecl_core::server::ServerError;
use std::{
    collections::{HashMap, HashSet},
    ffi::CString,
    sync::{Arc, Mutex, OnceLock},
};
use wgpu::naga;
#[path = "strata_atomic_or.rs"]
mod atomic_or;
#[path = "strata_binding_owners.rs"]
mod binding_owners;
#[path = "strata_capture_buffers.rs"]
mod capture_buffers;
#[path = "strata_capture_scratch.rs"]
mod capture_scratch;
#[path = "strata_compute_barriers.rs"]
mod compute_barriers;
#[path = "strata_descriptors.rs"]
mod descriptors;
#[path = "strata_push_descriptors.rs"]
mod push_descriptors;
#[path = "strata_transfer_barriers.rs"]
mod transfer_barriers;

static PROFILE_ACTIVE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub fn set_native_strata_profile_active(active: bool) {
    PROFILE_ACTIVE.store(active, std::sync::atomic::Ordering::Release);
}
fn profile_active() -> bool {
    enabled()
        && PROFILE_ACTIVE.load(std::sync::atomic::Ordering::Acquire)
        && matches!(
            std::env::var("KUNO_STRATA_NATIVE_PROFILE").as_deref(),
            Ok("round-8" | "warm-round-64" | "prefill-layer")
        )
        && std::env::var("KUNO_LAB_LANGUAGE_PUBLIC_ONLY").as_deref() == Ok("1")
}
type OperatorTotals = std::collections::BTreeMap<&'static str, (u64, u64)>;
fn profile_totals() -> &'static Mutex<OperatorTotals> {
    static TOTALS: OnceLock<Mutex<OperatorTotals>> = OnceLock::new();
    TOTALS.get_or_init(Mutex::default)
}
pub fn native_strata_operator_profile() -> Vec<(String, u64, u64)> {
    profile_totals()
        .lock()
        .expect("public operator totals")
        .iter()
        .map(|(name, (calls, ns))| ((*name).to_owned(), *calls, *ns))
        .collect()
}

/// Query values contain only performance counters for one fixed public round.
/// Recording owns this pool through the same completion fence as all buffers.
struct OperatorProfile {
    device: ash::Device,
    pool: vk::QueryPool,
    names: Vec<&'static str>,
    period: f64,
    mask: u64,
}
impl OperatorProfile {
    fn collect(&self) -> Result<(), ServerError> {
        let mut ticks = vec![0u64; self.names.len() * 2];
        unsafe {
            self.device.get_query_pool_results(
                self.pool,
                0,
                &mut ticks,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )
        }
        .map_err(|_| fault("native public operator timestamps"))?;
        let mut totals = profile_totals()
            .lock()
            .map_err(|_| fault("public operator totals"))?;
        for (name, pair) in self.names.iter().zip(ticks.chunks_exact(2)) {
            let ns = (pair[1].wrapping_sub(pair[0]) & self.mask) as f64 * self.period;
            if !ns.is_finite() || ns > 120_000_000_000.0 {
                return Err(fault("native public timestamp interval"));
            }
            let entry = totals.entry(name).or_default();
            entry.0 += 1;
            entry.1 = entry.1.saturating_add(ns.round() as u64);
        }
        Ok(())
    }
    fn stamp(&self, command: vk::CommandBuffer, index: usize, end: bool) {
        // These timestamp boundaries serialize the sampled nodes. Diagnostic
        // operator costs only; this run is not uninstrumented throughput proof.
        unsafe {
            self.device.cmd_write_timestamp(
                command,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.pool,
                (2 * index + usize::from(end)) as u32,
            );
        }
    }
}
impl Drop for OperatorProfile {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_query_pool(self.pool, None);
        }
    }
}

static SHAPES: [std::sync::atomic::AtomicU64; 5] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 5];
static REPLAY_SHAPES: [std::sync::atomic::AtomicU64; 5] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 5];
static CAPTURE_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static WAIT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
// Anonymous encoded COPY-node/barrier totals. Only the existing fixed-public
// profiling collector publishes deltas, after its real original sync.
static COPY_BARRIERS: [std::sync::atomic::AtomicU64; 2] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 2];
static COMPUTE_BARRIERS: [std::sync::atomic::AtomicU64; 10] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 10];
pub fn native_strata_graph_compute_barrier_stats() -> [u64; 10] {
    std::array::from_fn(|i| COMPUTE_BARRIERS[i].load(std::sync::atomic::Ordering::Relaxed))
}
pub fn native_strata_graph_copy_barrier_stats() -> [u64; 2] {
    std::array::from_fn(|i| COPY_BARRIERS[i].load(std::sync::atomic::Ordering::Relaxed))
}
pub fn native_strata_graph_execution_stats() -> ([u64; 5], [u64; 5], u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (
        std::array::from_fn(|i| SHAPES[i].load(Relaxed)),
        std::array::from_fn(|i| REPLAY_SHAPES[i].load(Relaxed)),
        CAPTURE_NS.load(Relaxed),
        WAIT_NS.load(Relaxed),
    )
}
// Anonymous process totals only, like CAPTURE_NS. No request identity,
// route, shape or content; only the compiled-public bridge publishes deltas.
static ROLLING_TOTALS: [std::sync::atomic::AtomicU64; 3] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 3];
/// Accepted original recordings, nonempty submitted windows and original
/// submitted segments. Counts imply actual host submission, not completion.
pub fn native_strata_graph_rolling_stats() -> [u64; 3] {
    std::array::from_fn(|i| ROLLING_TOTALS[i].load(std::sync::atomic::Ordering::Relaxed))
}
/// Explicit compiled-public terminal-worker diagnostic only. No private
/// inference setting or wire field selects this one-shot negative control.
pub fn arm_native_strata_public_rolling_failure() -> Result<(), &'static str> {
    strata_schedule::arm_public_failure()
}
/// Fired controls, actual successfully waited recovery queues, failed queues.
/// Anonymous aggregates are reported only by the explicit public diagnostic.
pub fn native_strata_public_rolling_failure_stats() -> [u64; 3] {
    strata_schedule::public_failure_stats()
}
static SUBMISSIONS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DISPATCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static REPLAYS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub fn native_strata_graph_stats() -> (u64, u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (
        SUBMISSIONS.load(Relaxed),
        DISPATCHES.load(Relaxed),
        REPLAYS.load(Relaxed),
    )
}
fn credit(nodes: usize, replay: bool) {
    use std::sync::atomic::Ordering::Relaxed;
    let bucket = match nodes {
        0..32 => 0,
        32..128 => 1,
        128..512 => 2,
        512..2048 => 3,
        _ => 4,
    };
    SHAPES[bucket].fetch_add(1, Relaxed);
    if replay {
        REPLAY_SHAPES[bucket].fetch_add(1, Relaxed);
    }
    SUBMISSIONS.fetch_add(1, Relaxed);
    DISPATCHES.fetch_add(nodes as u64, Relaxed);
    if replay {
        REPLAYS.fetch_add(1, Relaxed);
    }
}

pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("KUNO_STRATA_INTEL_CORE").as_deref(),
            Ok("public" | "protected-worker")
        )
    })
}
// NATIVE_PROGRAM_DEVICE_LIMITS_BEGIN
/// Actual physical-device limits for one original native Graph. This is not a
/// software preference or the HAL selector's bounded supported-shape inventory.
#[derive(Clone, Copy, Debug)]
struct ProgramDeviceLimits {
    min_subgroup: u32,
    max_subgroup: u32,
    max_subgroups: u32,
    max_invocations: u32,
    max_size: [u32; 3],
    max_shared_bytes: u64,
}
impl ProgramDeviceLimits {
    fn validate(
        self,
        words: &[u32],
        entry: &[u8],
        required_subgroup: Option<u32>,
        shared_bytes: Option<u64>,
    ) -> Result<(), &'static str> {
        let size = literal_program_workgroup(words, entry)?;
        if size
            .iter()
            .zip(self.max_size)
            .any(|(&n, max)| n == 0 || n > max)
        {
            return Err("native program workgroup dimension limit");
        }
        let invocations = size
            .into_iter()
            .try_fold(1_u32, u32::checked_mul)
            .ok_or("native program workgroup invocation overflow")?;
        if invocations > self.max_invocations {
            return Err("native program workgroup invocation limit");
        }
        if let Some(width) = required_subgroup {
            if !width.is_power_of_two() || width < self.min_subgroup || width > self.max_subgroup {
                return Err("native program required subgroup size limit");
            }
            // Ceiling matters for legal partial subgroups, including the
            // existing workgroup-of-one fixed32 operators. Never use a floor.
            if invocations.div_ceil(width) > self.max_subgroups {
                return Err("native program compute subgroup count limit");
            }
        }
        // With no required width the Vulkan implementation chooses the width;
        // do not invent a minimum-width choice and reject legal automatic32.
        // Rust/SPIR-V supplies its actual compiler-owned shared footprint.
        // WGSL retains the original Wgpu/Naga workgroup-memory admission.
        if shared_bytes.is_some_and(|bytes| bytes > self.max_shared_bytes) {
            return Err("native program shared memory limit");
        }
        Ok(())
    }
}

/// Read only the selected compute entry's original literal LocalSize. The
/// bounded native path has no specialization-size contract: reject LocalSizeId
/// or a WorkgroupSize builtin which could override these literal dimensions.
/// The original module validators still own full SPIR-V semantic validation.
fn literal_program_workgroup(words: &[u32], name: &[u8]) -> Result<[u32; 3], &'static str> {
    if words.len() < 5 || words[0] != 0x07230203 || words[3] == 0 || name.is_empty() {
        return Err("native program workgroup SPIRV header");
    }
    let mut entry = None;
    let mut at = 5;
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        let end = at
            .checked_add(count)
            .filter(|&end| count != 0 && end <= words.len())
            .ok_or("native program workgroup SPIRV extent")?;
        let i = &words[at..end];
        if i[0] & 65535 == 15 && count >= 4 && i[1] == 5 {
            let mut bytes = i[3..].iter().flat_map(|word| word.to_le_bytes());
            let matches =
                name.iter().all(|&byte| bytes.next() == Some(byte)) && bytes.next() == Some(0);
            if matches {
                if i[2] == 0 || i[2] >= words[3] || entry.replace(i[2]).is_some() {
                    return Err("native program workgroup compute entry");
                }
            }
        }
        at = end;
    }
    let entry = entry.ok_or("native program workgroup compute entry")?;
    let mut size = None;
    at = 5;
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        // Every instruction extent was checked in the first pass.
        let i = &words[at..at + count];
        let op = i[0] & 65535;
        if (op == 16 || op == 331) && count >= 3 && i[1] == entry {
            if i[2] == 38 {
                return Err("native program specialized workgroup unsupported");
            }
            if i[2] == 17 {
                if op != 16 || count != 6 || size.replace([i[3], i[4], i[5]]).is_some() {
                    return Err("native program literal workgroup mode");
                }
            }
        }
        if op == 71 && count >= 4 && i[2] == 11 && i[3] == 25 {
            return Err("native program overriding workgroup builtin unsupported");
        }
        at += count;
    }
    size.ok_or("native program literal workgroup missing")
}
// NATIVE_PROGRAM_DEVICE_LIMITS_END
#[derive(Debug)]
pub(crate) struct Program {
    name: &'static str,
    words: Vec<u32>,
    bindings: Vec<vk::DescriptorType>,
    accesses: Vec<vk::AccessFlags>,
    atomic_or: u32,
    entry: CString,
    required_subgroup: Option<u32>,
    shared_bytes: Option<u64>,
    physical_reads: bool,
    release_after: bool,
}
#[derive(Default)]
struct Registry {
    modules: HashMap<wgpu::ShaderModule, Arc<Program>>,
    pipelines: HashMap<wgpu::ComputePipeline, Arc<Program>>,
}
fn registry() -> &'static Mutex<Registry> {
    static R: OnceLock<Mutex<Registry>> = OnceLock::new();
    R.get_or_init(Mutex::default)
}
pub(crate) fn register_module(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    source: &str,
    entry: &str,
) -> Result<(), ServerError> {
    if !enabled() {
        return Ok(());
    }
    let m = naga::front::wgsl::parse_str(source).map_err(|_| fault("native graph WGSL parse"))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&m)
    .map_err(|_| fault("native graph WGSL validation"))?;
    let mut bindings = Vec::new();
    let entry_index = m
        .entry_points
        .iter()
        .position(|e| e.name == entry && e.stage == naga::ShaderStage::Compute)
        .ok_or_else(|| fault("native graph compute entrypoint"))?;
    let usage = info.get_entry_point(entry_index);
    for (handle, g) in m.global_variables.iter() {
        if let Some(b) = &g.binding {
            if b.group != 0 {
                return Err(fault("native graph descriptor group"));
            }
            let t = match g.space {
                naga::AddressSpace::Uniform => vk::DescriptorType::UNIFORM_BUFFER,
                naga::AddressSpace::Storage { .. } => vk::DescriptorType::STORAGE_BUFFER,
                _ => return Err(fault("native graph descriptor kind")),
            };
            let used = usage[handle];
            let mut access = vk::AccessFlags::empty();
            if used.contains(naga::valid::GlobalUse::READ) {
                access |= if t == vk::DescriptorType::UNIFORM_BUFFER {
                    vk::AccessFlags::UNIFORM_READ
                } else {
                    vk::AccessFlags::SHADER_READ
                };
            }
            if used.contains(naga::valid::GlobalUse::WRITE) {
                access |= vk::AccessFlags::SHADER_WRITE;
            }
            if source.contains("STRATA_PHYSICAL_EXPERTS")
                && g.name
                    .as_ref()
                    .is_some_and(|name| name.starts_with("strata_staged_"))
            {
                // Physical expert reads alias these genuine private staging
                // roots. The logical shader only queries their owning extents.
                access |= vk::AccessFlags::SHADER_READ;
            }
            bindings.push((b.binding, t, access));
        }
    }
    bindings.sort_by_key(|x| x.0);
    if bindings.iter().enumerate().any(|(i, b)| b.0 as usize != i) {
        return Err(fault("native graph descriptor order"));
    }
    let options = naga::back::spv::Options {
        lang_version: (1, 3),
        zero_initialize_workgroup_memory: naga::back::spv::ZeroInitializeWorkgroupMemoryMode::None,
        bounds_check_policies: naga::proc::BoundsCheckPolicies {
            index: naga::proc::BoundsCheckPolicy::Unchecked,
            buffer: naga::proc::BoundsCheckPolicy::Unchecked,
            image_load: naga::proc::BoundsCheckPolicy::Unchecked,
            binding_array: naga::proc::BoundsCheckPolicy::Unchecked,
        },
        ..Default::default()
    };
    let mut words = naga::back::spv::write_vec(
        &m,
        &info,
        &options,
        Some(&naga::back::spv::PipelineOptions {
            shader_stage: naga::ShaderStage::Compute,
            entry_point: entry.into(),
        }),
    )
    .map_err(|_| fault("native graph SPIRV lowering"))?;
    words = super::strata_physical::canonical_workgroup_layout(&words)?;
    if source.contains("STRATA_PHYSICAL_EXPERTS") {
        words = super::strata_physical::native_expert_shader_words(source)?;
    }
    if source.contains("STRATA_ROUND_PRODUCTS") {
        words = super::strata_physical::preserve_native_product_rounding(&words)?;
    }
    // Retained replay consumes the SAME requirement as the original ordinary
    // Vulkan device/factory. Reject an unmarked device rather than infer it from
    // a shader name/comment or permit varying-size execution of fixed32 math.
    let required_subgroup = unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() }
        .and_then(|d| d.kuno_wgsl_subgroup_size());
    if required_subgroup != Some(32) {
        return Err(fault("native WGSL subgroup owner requirement"));
    }
    let proven_or = atomic_or::mask(
        &words,
        bindings.iter().enumerate().fold(0u32, |mask, (index, b)| {
            mask | if index < 32
                && b.1 == vk::DescriptorType::STORAGE_BUFFER
                && b.2 == (vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
            {
                1u32 << index
            } else {
                0
            }
        }),
    );
    let p = Arc::new(Program {
        name: "UnlabelledNativeOperator",
        words,
        required_subgroup,
        shared_bytes: None,
        physical_reads: source.contains("STRATA_PHYSICAL_EXPERTS"),
        release_after: false,
        accesses: bindings.iter().map(|b| b.2).collect(),
        atomic_or: proven_or,
        bindings: bindings.into_iter().map(|b| b.1).collect(),
        entry: CString::new(entry).map_err(|_| fault("native graph entrypoint"))?,
    });
    registry()
        .lock()
        .map_err(|_| fault("native graph program registry"))?
        .modules
        .insert(module.clone(), p);
    Ok(())
}
/// Register the actual already-compiled Rust/SPIR-V graph, without pretending
/// it is WGSL or recompiling it through Naga. The compiler owns the complete
/// ordered descriptor inventory; its runtime INFO binding is supplied by the
/// same KernelArguments that create the Wgpu pipeline.
#[cfg(feature = "spirv")]
pub(crate) fn register_spirv_module(
    module: &wgpu::ShaderModule,
    kernel: &cubecl_spirv::SpirvKernel,
    entry: &str,
    metadata: bool,
) -> Result<(), ServerError> {
    if !enabled() {
        return Ok(());
    }
    if kernel.assembled_module.len() < 5
        || kernel.assembled_module[0] != 0x07230203
        || kernel.bindings.is_empty()
        || kernel.bindings.len() + usize::from(metadata) > 32
    {
        return Err(fault("native SPIRV descriptor inventory"));
    }
    let mut bindings = Vec::with_capacity(kernel.bindings.len() + usize::from(metadata));
    let mut accesses = Vec::with_capacity(bindings.capacity());
    for visibility in &kernel.bindings {
        bindings.push(vk::DescriptorType::STORAGE_BUFFER);
        let mut access = vk::AccessFlags::SHADER_READ;
        if matches!(visibility, cubecl_runtime::kernel::Visibility::ReadWrite) {
            access |= vk::AccessFlags::SHADER_WRITE;
        }
        accesses.push(access);
    }
    if metadata {
        bindings.push(if kernel.uniform_info {
            vk::DescriptorType::UNIFORM_BUFFER
        } else {
            vk::DescriptorType::STORAGE_BUFFER
        });
        accesses.push(if kernel.uniform_info {
            vk::AccessFlags::UNIFORM_READ
        } else {
            vk::AccessFlags::SHADER_READ
        });
    }
    // Every Intel DPAS tile uses one16-lane subgroup; a tiled workgroup may
    // contain several such subgroups. The shared selector checks the literal
    // block shape and known subgroup-scoped matrix types before selecting16.
    let words = super::strata_physical::canonical_workgroup_layout(&kernel.assembled_module)?;
    // create_module first creates the HAL ShaderModule, which verifies the
    // device's FloatControls properties. Native replay creates its own Vulkan
    // module from Program.words, so it must apply the identical FP16 controls.
    let words = wgpu::hal::vulkan::kuno_cooperative_fp16_preserve(&words)
        .map_err(|_| fault("native SPIRV FP16 preservation"))?
        .unwrap_or(words);
    let required_subgroup = wgpu::hal::vulkan::kuno_cooperative_subgroup(&words);
    let mut at = 5;
    let mut cooperative = false;
    while at < words.len() {
        let first = words[at];
        let count = (first >> 16) as usize;
        if count == 0 || at.checked_add(count).is_none_or(|end| end > words.len()) {
            return Err(fault("native SPIRV instruction extent"));
        }
        if first & 65535 == 4456 {
            cooperative = true;
        }
        at += count;
    }
    if cooperative && required_subgroup != Some(16) {
        return Err(fault("native SPIRV cooperative subgroup geometry"));
    }
    let proven_or = atomic_or::mask(
        &words,
        bindings.iter().enumerate().fold(0u32, |mask, (index, b)| {
            mask | if index < 32
                && *b == vk::DescriptorType::STORAGE_BUFFER
                && accesses[index] == (vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
            {
                1u32 << index
            } else {
                0
            }
        }),
    );
    let program = Arc::new(Program {
        name: "UnlabelledNativeOperator",
        words,
        bindings,
        accesses,
        atomic_or: proven_or,
        entry: CString::new(entry).map_err(|_| fault("native SPIRV entrypoint"))?,
        required_subgroup,
        shared_bytes: Some(
            u64::try_from(kernel.shared_size)
                .map_err(|_| fault("native SPIRV shared memory extent"))?,
        ),
        physical_reads: false,
        release_after: false,
    });
    registry()
        .lock()
        .map_err(|_| fault("native SPIRV program registry"))?
        .modules
        .insert(module.clone(), program);
    Ok(())
}
pub(crate) fn label_module(
    module: &wgpu::ShaderModule,
    name: &'static str,
    release_after: bool,
) -> Result<(), ServerError> {
    if !enabled() {
        return Ok(());
    }
    let mut r = registry()
        .lock()
        .map_err(|_| fault("native operator module label"))?;
    let p = r
        .modules
        .get_mut(module)
        .and_then(Arc::get_mut)
        .ok_or_else(|| fault("native operator module identity"))?;
    p.name = name;
    p.release_after = release_after;
    Ok(())
}
pub(crate) fn register_pipeline(pipeline: &wgpu::ComputePipeline, module: &wgpu::ShaderModule) {
    if !enabled() {
        return;
    }
    let mut r = registry().lock().expect("native public program registry");
    if let Some(p) = r.modules.remove(module) {
        r.pipelines.insert(pipeline.clone(), p);
    }
}
#[derive(Debug)]
struct Pipeline {
    shader: vk::ShaderModule,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    program: Arc<Program>,
    push_descriptors: bool,
}
#[derive(Debug)]
struct Node {
    shared: bool,
    release_after: bool,
    pipeline: Option<wgpu::ComputePipeline>,
    resources: Vec<WgpuResource>,
    read_dependencies: Vec<WgpuResource>,
    dispatch: Dispatch,
    owners: Option<binding_owners::Bundle>,
}
#[derive(Debug)]
pub(crate) enum Dispatch {
    Direct(u32, u32, u32),
    Indirect(WgpuResource),
    Copy,
    Fill,
}
#[derive(Clone, Copy)]
struct ComputeQueue {
    family: u32,
    queue: vk::Queue,
    timestamp_bits: u32,
}
pub(crate) struct Graph {
    identity: u64,
    program_limits: ProgramDeviceLimits,
    push_descriptors: Option<(ash::khr::push_descriptor::Device, u32)>,
    timestamp_period: f64,
    timestamp_bits: u32,
    compute_queues: [ComputeQueue; 2],
    compute_lane: usize,
    shared_request: Option<u64>,
    handoff: super::strata_handoff::Handoff,
    info_uploads: [super::strata_info_transfer::Uploads; 2],
    device: ash::Device,
    family: u32,
    queue: vk::Queue,
    copy_family: u32,
    copy_queue: wgpu::hal::vulkan::KunoTransferQueue,
    copy_timestamp_bits: u32,
    pool: vk::CommandPool,
    descriptors: vk::DescriptorPool,
    command: vk::CommandBuffer,
    fences: [vk::Fence; 2],
    pipelines: HashMap<wgpu::ComputePipeline, Pipeline>,
    nodes: Vec<Node>,
    flights: super::strata_flights::Book,
    draft_scope: super::strata_draft_scope::Book,
    // Keep allocator binding as well as VkBuffer alive: a resource clone alone
    // does not prevent allocator reuse of the original verified-R allocation.
    draft_source: Option<cubecl_runtime::storage::ManagedResource<WgpuResource>>,
    draft_readiness: Option<super::strata_handoff::Handoff>,
    next_recording: u64,
    pub submissions: u64,
    pub dispatches: u64,
    recordings: Vec<Recording>,
    pub replays: u64,
    initialized: HashSet<wgpu::Buffer>,
}
// No completion credit is minted by submission. The original recording and
// complete node inventory remain retained until the actual final queue fence.
struct Recording {
    id: u64,
    family: u32,
    key: Option<super::strata_parameters::RecordingKey>,
    ready: bool,
    compute_count: usize,
    work: strata_schedule::Commands,
    profile: Option<OperatorProfile>,
    device: ash::Device,
    pool: vk::CommandPool,
    descriptors: vk::DescriptorPool,
    command: vk::CommandBuffer,
    signature: Vec<u64>,
    _resources: Vec<WgpuResource>,
    _owners: binding_owners::Owners,
}
impl Recording {
    fn metadata_bytes(&self) -> Option<usize> {
        self.work
            .metadata_bytes()
            .checked_add(self.signature.capacity().checked_mul(size_of::<u64>())?)?
            .checked_add(
                self._resources
                    .capacity()
                    .checked_mul(size_of::<WgpuResource>())?,
            )?
            .checked_add(self._owners.metadata_bytes())
    }
}
impl Drop for Recording {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_descriptor_pool(self.descriptors, None);
            self.device.destroy_command_pool(self.pool, None);
        }
    }
}
fn raw_buffer(r: &WgpuResource) -> Result<vk::Buffer, ServerError> {
    let h = unsafe { r.buffer.as_hal::<wgpu::hal::api::Vulkan>() }
        .ok_or_else(|| fault("native graph non-Vulkan buffer"))?;
    Ok(unsafe { h.raw_buffer_handle() })
}
// NATIVE_COMMAND_BOUNDARIES_BEGIN
/// Fixed stack metadata for the actual recording's command-buffer segments.
/// Retain external availability/visibility at both ends of EACH physical queue;
/// only interior segment boundaries rely on the unchanged actual-buffer hazard
/// barriers and original inter-queue timeline memory dependencies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CommandBoundaries {
    first: [usize; 3],
    last: [usize; 3],
    segments: usize,
}
impl CommandBoundaries {
    fn from_queues(queues: impl Iterator<Item = usize>) -> Result<Self, &'static str> {
        let mut result = Self {
            first: [usize::MAX; 3],
            last: [usize::MAX; 3],
            segments: 0,
        };
        for (segment, queue) in queues.enumerate() {
            if segment >= 32769 || queue >= 3 || segment == 0 && queue != 0 {
                return Err("native command boundary segment inventory");
            }
            if result.first[queue] == usize::MAX {
                result.first[queue] = segment;
            }
            result.last[queue] = segment;
            result.segments += 1;
        }
        if result.segments == 0 {
            return Err("native command boundary empty inventory");
        }
        Ok(result)
    }
    fn before(&self, segment: usize) -> bool {
        segment < self.segments && self.first.contains(&segment)
    }
    fn after(&self, segment: usize) -> bool {
        segment < self.segments && self.last.contains(&segment)
    }
}

#[cfg(test)]
mod command_boundary_tests {
    use super::CommandBoundaries;

    #[test]
    fn single_and_empty_queue_cases_keep_external_boundaries() {
        let only = CommandBoundaries::from_queues([0].into_iter()).unwrap();
        assert!(only.before(0) && only.after(0));
        let main = CommandBoundaries::from_queues([0, 0, 0].into_iter()).unwrap();
        assert_eq!(
            (0..3).map(|i| main.before(i)).collect::<Vec<_>>(),
            [true, false, false]
        );
        assert_eq!(
            (0..3).map(|i| main.after(i)).collect::<Vec<_>>(),
            [false, false, true]
        );
        assert!(!main.before(usize::MAX) && !main.after(usize::MAX));
        assert!(!main.before(3) && !main.after(3));
    }

    #[test]
    fn alternating_release_segments_keep_both_ends_of_each_original_queue() {
        let queues = [0, 1, 0, 1, 0, 2, 0, 2, 1, 0];
        let p = CommandBoundaries::from_queues(queues.into_iter()).unwrap();
        assert_eq!(p.first, [0, 1, 5]);
        assert_eq!(p.last, [9, 8, 7]);
        for segment in 0..queues.len() {
            assert_eq!(p.before(segment), [0, 1, 5].contains(&segment));
            assert_eq!(p.after(segment), [9, 8, 7].contains(&segment));
        }
        // A retained command buffer replays the same already-encoded policy;
        // deriving it again from its immutable plan cannot change its boundaries.
        assert_eq!(
            p,
            CommandBoundaries::from_queues(queues.into_iter()).unwrap()
        );
    }

    #[test]
    fn full_capacity_inventory_has_exactly_two_boundaries_per_used_queue() {
        let mut state = 23u32;
        let queues = (0..32769)
            .map(|i| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                if i == 0 { 0 } else { (state % 3) as usize }
            })
            .collect::<Vec<_>>();
        let p = CommandBoundaries::from_queues(queues.iter().copied()).unwrap();
        for queue in 0..3 {
            let segments = queues
                .iter()
                .enumerate()
                .filter(|(_, q)| **q == queue)
                .map(|(i, _)| i)
                .collect::<Vec<_>>();
            assert_eq!(segments.iter().filter(|&&i| p.before(i)).count(), 1);
            assert_eq!(segments.iter().filter(|&&i| p.after(i)).count(), 1);
            assert!(p.before(segments[0]));
            assert!(p.after(*segments.last().unwrap()));
        }
        assert_eq!((0..queues.len()).filter(|&i| p.before(i)).count(), 3);
        assert_eq!((0..queues.len()).filter(|&i| p.after(i)).count(), 3);
    }

    #[test]
    fn invalid_or_unbounded_segment_inventory_is_rejected() {
        assert!(CommandBoundaries::from_queues(std::iter::empty()).is_err());
        assert!(CommandBoundaries::from_queues([1].into_iter()).is_err());
        assert!(CommandBoundaries::from_queues([0, 3].into_iter()).is_err());
        assert!(CommandBoundaries::from_queues([0, usize::MAX].into_iter()).is_err());
        assert!(CommandBoundaries::from_queues(std::iter::repeat_n(0, 32770)).is_err());
    }
}
// NATIVE_COMMAND_BOUNDARIES_END
// NATIVE_ROLLING_SELECTION_BEGIN
fn rolling_prefill_selected(
    lane: usize,
    named: bool,
    profiling: bool,
    nodes: usize,
    compute: usize,
    expert_image: bool,
) -> bool {
    lane == 0
        && !named
        && !profiling
        && expert_image
        && (1024..=32768).contains(&compute)
        && (compute..=32768).contains(&nodes)
}
#[cfg(test)]
mod rolling_selection_tests {
    use super::*;
    #[test]
    fn only_large_unnamed_unprofiled_main_expert_recordings_can_stream() {
        for lane in 0..3 {
            for named in [false, true] {
                for profiling in [false, true] {
                    for image in [false, true] {
                        for compute in [0, 1023, 1024, 2048, 32768, 32769] {
                            assert_eq!(
                                rolling_prefill_selected(
                                    lane, named, profiling, compute, compute, image
                                ),
                                lane == 0
                                    && !named
                                    && !profiling
                                    && image
                                    && (1024..=32768).contains(&compute)
                            );
                        }
                    }
                }
            }
        }
        assert!(!rolling_prefill_selected(0, false, false, 1023, 1024, true));
        assert!(!rolling_prefill_selected(
            0, false, false, 32769, 1024, true
        ));
    }
}
// NATIVE_ROLLING_SELECTION_END
fn begin_native_command(
    device: &ash::Device,
    command: vk::CommandBuffer,
    external: bool,
) -> Result<(), ServerError> {
    unsafe { device.begin_command_buffer(command, &vk::CommandBufferBeginInfo::default()) }
        .map_err(|_| fault("native queue begin command"))?;
    if external {
        native_memory_boundary(device, command);
    }
    Ok(())
}
fn native_memory_boundary(device: &ash::Device, command: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
        .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
    unsafe {
        device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );
    }
}
/// Explicit device-to-host availability for original owned readback spans.
/// A host fence alone is not the buffer memory dependency. Native readback
/// never relies on Wgpu to infer/insert this barrier or initialize the owner.
fn native_host_read_boundary(
    device: &ash::Device,
    command: vk::CommandBuffer,
    target: &WgpuResource,
    native: &capture_buffers::Guarded<'_>,
) -> Result<(), ServerError> {
    if target.buffer.usage().contains(wgpu::BufferUsages::MAP_READ) {
        let barrier = vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(native.get(target)?)
            .offset(target.offset)
            .size(target.size);
        unsafe {
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[],
                &[barrier],
                &[],
            );
        }
    }
    Ok(())
}
fn end_native_command(
    device: &ash::Device,
    command: vk::CommandBuffer,
    external: bool,
) -> Result<(), ServerError> {
    if external {
        native_memory_boundary(device, command);
    }
    unsafe { device.end_command_buffer(command) }.map_err(|_| fault("native queue end command"))
}
impl Graph {
    pub(crate) fn new(device: &wgpu::Device) -> Result<Self, ServerError> {
        // SAFETY: Vulkan-only borrowed device; the stream owns its lifetime and
        // serializes all native/Wgpu submissions. No device or queue is destroyed.
        let h = unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() }
            .ok_or_else(|| fault("native graph requires Vulkan"))?;
        if !h
            .enabled_device_extensions()
            .contains(&ash::khr::buffer_device_address::NAME)
        {
            return Err(fault("native expert pointer extension missing"));
        }
        let mut subgroup = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
        let mut props = vk::PhysicalDeviceProperties2::default().push_next(&mut subgroup);
        unsafe {
            h.shared_instance()
                .raw_instance()
                .get_physical_device_properties2(h.raw_physical_device(), &mut props);
        }
        let push_descriptors = if h
            .enabled_device_extensions()
            .contains(&ash::khr::push_descriptor::NAME)
            && std::env::var("KUNO_STRATA_PUSH_DESCRIPTORS").as_deref() == Ok("command-owned")
        {
            let mut limits = vk::PhysicalDevicePushDescriptorPropertiesKHR::default();
            let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut limits);
            unsafe {
                h.shared_instance()
                    .raw_instance()
                    .get_physical_device_properties2(h.raw_physical_device(), &mut properties);
            }
            (limits.max_push_descriptors > 0).then(|| {
                (
                    ash::khr::push_descriptor::Device::new(
                        h.shared_instance().raw_instance(),
                        h.raw_device(),
                    ),
                    limits.max_push_descriptors,
                )
            })
        } else {
            None
        };
        let timestamp_period = props.properties.limits.timestamp_period as f64;
        let physical_limits = props.properties.limits;
        let timestamp_bits = unsafe {
            h.shared_instance()
                .raw_instance()
                .get_physical_device_queue_family_properties(h.raw_physical_device())
        }
        .get(h.queue_family_index() as usize)
        .map_or(0, |q| q.timestamp_valid_bits);
        if subgroup.min_subgroup_size > 16
            || subgroup.max_subgroup_size < 32
            || !subgroup
                .required_subgroup_size_stages
                .contains(vk::ShaderStageFlags::COMPUTE)
        {
            return Err(fault(
                "native Strata requires Intel matrix16 and expert32 subgroups",
            ));
        }
        let program_limits = ProgramDeviceLimits {
            min_subgroup: subgroup.min_subgroup_size,
            max_subgroup: subgroup.max_subgroup_size,
            max_subgroups: subgroup.max_compute_workgroup_subgroups,
            max_invocations: physical_limits.max_compute_work_group_invocations,
            max_size: physical_limits.max_compute_work_group_size,
            max_shared_bytes: u64::from(physical_limits.max_compute_shared_memory_size),
        };
        let copy_queue = h
            .kuno_transfer_queue_authority()
            .ok_or_else(|| fault("native actual transfer queue missing"))?;
        let copy_family = copy_queue.family_index();
        let (draft_family, draft_queue) = h
            .kuno_compute_queue()
            .ok_or_else(|| fault("native graph needs the actually adopted draft queue"))?;
        if draft_family == h.queue_family_index() || draft_family == copy_family {
            return Err(fault("native draft queue family alias"));
        }
        let draft_timestamp_bits = unsafe {
            h.shared_instance()
                .raw_instance()
                .get_physical_device_queue_family_properties(h.raw_physical_device())
        }
        .get(draft_family as usize)
        .map_or(0, |q| q.timestamp_valid_bits);
        let copy_timestamp_bits = unsafe {
            h.shared_instance()
                .raw_instance()
                .get_physical_device_queue_family_properties(h.raw_physical_device())
        }
        .get(copy_family as usize)
        .map_or(0, |q| q.timestamp_valid_bits);
        static NEXT_GRAPH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let identity = NEXT_GRAPH
            .fetch_update(
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
                |n| n.checked_add(1),
            )
            .map_err(|_| fault("native graph identity exhausted"))?;
        let d = h.raw_device().clone();
        let queue = h.raw_queue();
        let pool = unsafe {
            d.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                    .queue_family_index(h.queue_family_index()),
                None,
            )
        }
        .map_err(|_| fault("native graph command pool"))?;
        // With the full original 32-binding capacity covered, no native set
        // allocation is needed. Keep a tiny valid pool for Recording ownership.
        let all_push = push_descriptors.as_ref().is_some_and(|(_, n)| *n >= 32);
        let sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: if all_push { 1 } else { 262144 },
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: if all_push { 1 } else { 32768 },
            },
        ];
        let descriptors = match unsafe {
            d.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(if all_push { 1 } else { 32768 })
                    .pool_sizes(&sizes),
                None,
            )
        } {
            Ok(v) => v,
            Err(_) => {
                unsafe { d.destroy_command_pool(pool, None) };
                return Err(fault("native graph descriptor pool"));
            }
        };
        let command = match unsafe {
            d.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        } {
            Ok(v) => v[0],
            Err(_) => {
                unsafe {
                    d.destroy_descriptor_pool(descriptors, None);
                    d.destroy_command_pool(pool, None)
                };
                return Err(fault("native graph command buffer"));
            }
        };
        let fence = match unsafe { d.create_fence(&vk::FenceCreateInfo::default(), None) } {
            Ok(v) => v,
            Err(_) => {
                unsafe {
                    d.destroy_descriptor_pool(descriptors, None);
                    d.destroy_command_pool(pool, None)
                };
                return Err(fault("native graph fence"));
            }
        };
        let draft_fence = match unsafe { d.create_fence(&vk::FenceCreateInfo::default(), None) } {
            Ok(v) => v,
            Err(_) => {
                unsafe {
                    d.destroy_fence(fence, None);
                    d.destroy_descriptor_pool(descriptors, None);
                    d.destroy_command_pool(pool, None);
                }
                return Err(fault("native draft flight fence"));
            }
        };
        let handoff = match super::strata_handoff::Handoff::new(&d) {
            Ok(v) => v,
            Err(e) => {
                unsafe {
                    d.destroy_fence(fence, None);
                    d.destroy_fence(draft_fence, None);
                    d.destroy_descriptor_pool(descriptors, None);
                    d.destroy_command_pool(pool, None);
                }
                return Err(fault(e));
            }
        };
        let info_uploads = match (|| {
            Ok::<_, &'static str>([
                super::strata_info_transfer::Uploads::new(&d, h.queue_family_index())?,
                super::strata_info_transfer::Uploads::new(&d, draft_family)?,
            ])
        })() {
            Ok(v) => v,
            Err(e) => {
                unsafe {
                    d.destroy_fence(fence, None);
                    d.destroy_fence(draft_fence, None);
                    d.destroy_descriptor_pool(descriptors, None);
                    d.destroy_command_pool(pool, None);
                }
                return Err(fault(e));
            }
        };
        Ok(Self {
            identity,
            program_limits,
            push_descriptors,
            timestamp_period,
            timestamp_bits,
            compute_queues: [
                ComputeQueue {
                    family: h.queue_family_index(),
                    queue,
                    timestamp_bits,
                },
                ComputeQueue {
                    family: draft_family,
                    queue: draft_queue,
                    timestamp_bits: draft_timestamp_bits,
                },
            ],
            compute_lane: 0,
            shared_request: None,
            handoff,
            info_uploads,
            device: d,
            family: h.queue_family_index(),
            queue,
            copy_family,
            copy_queue,
            copy_timestamp_bits,
            pool,
            descriptors,
            command,
            fences: [fence, draft_fence],
            pipelines: HashMap::new(),
            nodes: Vec::new(),
            flights: super::strata_flights::Book::default(),
            draft_scope: super::strata_draft_scope::Book::default(),
            draft_source: None,
            draft_readiness: None,
            next_recording: 0,
            submissions: 0,
            dispatches: 0,
            recordings: Vec::new(),
            replays: 0,
            initialized: HashSet::new(),
        })
    }
    pub(crate) fn initialize_resources(&mut self, resources: &[WgpuResource]) -> bool {
        let mut fresh = false;
        for r in resources {
            fresh |= self.initialized.insert(r.buffer.clone());
        }
        fresh
    }
    pub(crate) fn push(
        &mut self,
        pipeline: &wgpu::ComputePipeline,
        resources: Vec<WgpuResource>,
        read_dependencies: Vec<WgpuResource>,
        dispatch: Dispatch,
    ) -> Result<(), ServerError> {
        self.push_owned(pipeline, resources, read_dependencies, dispatch, None)
    }
    pub(crate) fn push_owned(
        &mut self,
        pipeline: &wgpu::ComputePipeline,
        resources: Vec<WgpuResource>,
        read_dependencies: Vec<WgpuResource>,
        dispatch: Dispatch,
        owners: Option<Arc<Vec<cubecl_runtime::server::Binding>>>,
    ) -> Result<(), ServerError> {
        if self.nodes.len() >= 32768 {
            return Err(fault("native graph node capacity"));
        }
        if !self.pipelines.contains_key(pipeline) {
            let p = registry()
                .lock()
                .map_err(|_| fault("native graph program registry"))?
                .pipelines
                .get(pipeline)
                .cloned()
                .ok_or_else(|| fault("native graph program missing"))?;
            self.program_limits
                .validate(
                    &p.words,
                    p.entry.as_bytes(),
                    p.required_subgroup,
                    p.shared_bytes,
                )
                .map_err(fault)?;
            let shader = unsafe {
                self.device.create_shader_module(
                    &vk::ShaderModuleCreateInfo::default().code(&p.words),
                    None,
                )
            }
            .map_err(|_| fault("native graph shader"))?;
            let bindings = p
                .bindings
                .iter()
                .enumerate()
                .map(|(i, t)| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i as u32)
                        .descriptor_type(*t)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect::<Vec<_>>();
            let push = self
                .push_descriptors
                .as_ref()
                .is_some_and(|(_, max)| push_descriptors::selected(true, *max, p.bindings.len()));
            let set_layout = match unsafe {
                self.device.create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default()
                        .bindings(&bindings)
                        .flags(if push {
                            vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR
                        } else {
                            vk::DescriptorSetLayoutCreateFlags::empty()
                        }),
                    None,
                )
            } {
                Ok(v) => v,
                Err(_) => {
                    unsafe { self.device.destroy_shader_module(shader, None) };
                    return Err(fault("native graph set layout"));
                }
            };
            let layouts = [set_layout];
            let layout = match unsafe {
                self.device.create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts),
                    None,
                )
            } {
                Ok(v) => v,
                Err(_) => {
                    unsafe {
                        self.device.destroy_descriptor_set_layout(set_layout, None);
                        self.device.destroy_shader_module(shader, None)
                    };
                    return Err(fault("native graph pipeline layout"));
                }
            };
            let (flags, mut required) =
                wgpu::hal::vulkan::kuno_subgroup_stage_options(p.required_subgroup, true);
            let mut stage = vk::PipelineShaderStageCreateInfo::default()
                .flags(flags)
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader)
                .name(&p.entry);
            // SAME exact width/flags/lifetime rule as ordinary Wgpu pipelines.
            if let Some(info) = required.as_mut() {
                stage = stage.push_next(&mut **info);
            }
            let pipeline_native = match unsafe {
                self.device.create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(stage)
                        .layout(layout)],
                    None,
                )
            } {
                Ok(v) => v[0],
                Err((v, _)) => {
                    unsafe {
                        for v in v {
                            self.device.destroy_pipeline(v, None);
                        }
                        self.device.destroy_pipeline_layout(layout, None);
                        self.device.destroy_descriptor_set_layout(set_layout, None);
                        self.device.destroy_shader_module(shader, None)
                    };
                    return Err(fault("native graph pipeline"));
                }
            };
            self.pipelines.insert(
                pipeline.clone(),
                Pipeline {
                    shader,
                    set_layout,
                    layout,
                    pipeline: pipeline_native,
                    program: p,
                    push_descriptors: push,
                },
            );
        }
        let p = &self.pipelines[pipeline];
        if resources.len() != p.program.bindings.len() {
            return Err(fault("native graph actual descriptor extent"));
        }
        if !read_dependencies.is_empty() && !p.program.physical_reads {
            return Err(fault("unregistered physical source consumer"));
        }
        if read_dependencies.len() > 4096 {
            return Err(fault("native physical dependency capacity"));
        }
        for r in resources.iter().chain(&read_dependencies) {
            if r.size == 0
                || r.offset
                    .checked_add(r.size)
                    .is_none_or(|end| end > r.buffer.size())
            {
                return Err(fault("native graph actual buffer extent"));
            }
        }
        let node = Node {
            shared: self.shared_request.is_some(),
            release_after: false,
            pipeline: Some(pipeline.clone()),
            resources,
            read_dependencies,
            dispatch,
            owners,
        };
        let accesses = self.node_ranges(&node)?;
        self.draft_scope.check_accesses(&accesses).map_err(fault)?;
        self.admit_accesses(
            &accesses,
            if node.shared {
                self.compute_queues
                    [strata_schedule::shared_peer(self.compute_lane).map_err(fault)?]
                .queue
            } else {
                self.queue
            },
        )?;
        self.nodes.push(node);
        Ok(())
    }
    /// Upload staging remains tracked by TransientIo. Transfers execute at
    /// their actual position between kernels, not ahead of earlier readers.
    pub(crate) fn push_transfer(
        &mut self,
        source: Option<WgpuResource>,
        target: WgpuResource,
    ) -> Result<(), ServerError> {
        self.push_transfer_owned(source, target, None)
    }
    pub(crate) fn push_transfer_owned(
        &mut self,
        source: Option<WgpuResource>,
        target: WgpuResource,
        owners: Option<Arc<Vec<cubecl_runtime::server::Binding>>>,
    ) -> Result<(), ServerError> {
        if self.nodes.len() >= 32768
            || target.size == 0
            || target.offset % 4 != 0
            || target.size % 4 != 0
            || target
                .offset
                .checked_add(target.size)
                .is_none_or(|end| end > target.buffer.size())
        {
            return Err(fault("native graph transfer extent"));
        }
        let (resources, dispatch) = if let Some(source) = source {
            if source.size != target.size
                || source.offset % 4 != 0
                || source
                    .offset
                    .checked_add(source.size)
                    .is_none_or(|end| end > source.buffer.size())
            {
                return Err(fault("native graph transfer source extent"));
            }
            (vec![source, target], Dispatch::Copy)
        } else {
            (vec![target], Dispatch::Fill)
        };
        let node = Node {
            shared: self.shared_request.is_some(),
            release_after: false,
            pipeline: None,
            resources,
            read_dependencies: Vec::new(),
            dispatch,
            owners,
        };
        let accesses = self.node_ranges(&node)?;
        self.draft_scope.check_accesses(&accesses).map_err(fault)?;
        self.admit_accesses(
            &accesses,
            if node.shared {
                self.compute_queues
                    [strata_schedule::shared_peer(self.compute_lane).map_err(fault)?]
                .queue
            } else {
                self.queue
            },
        )?;
        self.nodes.push(node);
        Ok(())
    }
    fn visit_node_bindings(
        &self,
        n: &Node,
        mut visit: impl FnMut(&WgpuResource, vk::AccessFlags, bool) -> Result<(), ServerError>,
    ) -> Result<(), ServerError> {
        match n.dispatch {
            Dispatch::Copy => {
                visit(&n.resources[0], vk::AccessFlags::TRANSFER_READ, false)?;
                visit(&n.resources[1], vk::AccessFlags::TRANSFER_WRITE, false)?;
            }
            Dispatch::Fill => visit(&n.resources[0], vk::AccessFlags::TRANSFER_WRITE, false)?,
            _ => {
                let p = &self.pipelines[n
                    .pipeline
                    .as_ref()
                    .ok_or_else(|| fault("native graph compute identity"))?];
                for (index, (r, access)) in n.resources.iter().zip(&p.program.accesses).enumerate()
                {
                    if !access.is_empty() {
                        visit(
                            r,
                            *access,
                            index < 32 && p.program.atomic_or & (1u32 << index) != 0,
                        )?;
                    }
                }
                for r in &n.read_dependencies {
                    visit(r, vk::AccessFlags::SHADER_READ, false)?;
                }
                if let Dispatch::Indirect(r) = &n.dispatch {
                    visit(r, vk::AccessFlags::INDIRECT_COMMAND_READ, false)?;
                }
            }
        }
        Ok(())
    }
    fn fill_node_accesses(
        &self,
        n: &Node,
        accesses: &mut HashMap<vk::Buffer, vk::AccessFlags>,
        native: &capture_buffers::Guarded<'_>,
    ) -> Result<(), ServerError> {
        accesses.clear();
        self.visit_node_bindings(n, |r, access, _| {
            *accesses.entry(native.get(r)?).or_default() |= access;
            Ok(())
        })
    }
    fn fill_node_ranges(
        &self,
        n: &Node,
        accesses: &mut Vec<strata_schedule::Access>,
        native: Option<&capture_buffers::Guarded<'_>>,
    ) -> Result<(), ServerError> {
        accesses.clear();
        self.visit_node_bindings(n, |r, access, _| {
            if r.size == 0
                || r.offset
                    .checked_add(r.size)
                    .is_none_or(|end| end > r.buffer.size())
            {
                return Err(fault("native queue actual resource interval"));
            }
            accesses.push(strata_schedule::Access {
                buffer: match native {
                    Some(native) => native.get(r)?,
                    None => raw_buffer(r)?,
                }
                .as_raw(),
                first: r.offset,
                bytes: r.size,
                write: access
                    .intersects(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE),
            });
            Ok(())
        })
    }
    fn fill_compute_ranges(
        &self,
        n: &Node,
        accesses: &mut Vec<compute_barriers::Access>,
        native: &capture_buffers::Guarded<'_>,
        mut planning: Option<&mut Vec<strata_schedule::Access>>,
    ) -> Result<(), ServerError> {
        accesses.clear();
        if let Some(ranges) = &mut planning {
            ranges.clear();
        }
        self.visit_node_bindings(n, |r, flags, atomic_or| {
            if r.size == 0
                || r.offset
                    .checked_add(r.size)
                    .is_none_or(|end| end > r.buffer.size())
            {
                return Err(fault("native compute barrier actual owner interval"));
            }
            let buffer = native.get(r)?.as_raw();
            if let Some(ranges) = &mut planning {
                ranges.push(strata_schedule::Access {
                    buffer,
                    first: r.offset,
                    bytes: r.size,
                    write: flags.intersects(
                        vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE,
                    ),
                });
            }
            accesses.push(compute_barriers::Access {
                buffer,
                first: r.offset,
                bytes: r.size,
                flags,
                atomic_or,
            });
            Ok(())
        })
    }
    fn node_ranges(&self, n: &Node) -> Result<Vec<strata_schedule::Access>, ServerError> {
        let mut accesses = Vec::new();
        self.fill_node_ranges(n, &mut accesses, None)?;
        Ok(accesses)
    }
    /// Tag genuinely enqueued target/draft nodes; no submission/completion grant.
    pub(crate) fn shared_boundary(
        &mut self,
        boundary: cubecl_runtime::server::NativeStrataSharedBoundary,
        request: u64,
    ) -> Result<(), ServerError> {
        use cubecl_runtime::server::NativeStrataSharedBoundary::*;
        strata_schedule::shared_queue(
            request,
            self.compute_lane,
            self.draft_scope.active_request(),
        )
        .map_err(fault)?;
        match boundary {
            Begin if self.shared_request.is_none() => {
                self.shared_request = Some(request);
            }
            End if self.shared_request == Some(request) => {
                self.shared_request = None;
            }
            ReleaseMain if self.shared_request.is_none() => {
                let last = self
                    .nodes
                    .last_mut()
                    .ok_or_else(|| fault("shared main release has no original node"))?;
                if last.shared {
                    return Err(fault("shared main release on peer"));
                }
                last.release_after = true;
            }
            _ => return Err(fault("shared queue boundary mismatch")),
        }
        Ok(())
    }
    /// Pin the actual verified R before accepted-prefix commit. The target
    /// fence already joins its shared queue before this lane is reused by MTP.
    pub(crate) fn prepare_draft_scope(
        &mut self,
        request: u64,
        source: cubecl_runtime::storage::ManagedResource<WgpuResource>,
    ) -> Result<cubecl_runtime::server::NativeStrataDraftScope, ServerError> {
        if self.compute_lane != 0 || self.pending() || self.draft_source.is_some() {
            return Err(fault(
                "native draft preparation requires original joined MAIN",
            ));
        }
        let r = source.resource();
        if !self.initialized.contains(&r.buffer)
            || !r
                .buffer
                .usage()
                .contains(wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC)
            || r.offset % 4 != 0
            || r.size % 4 != 0
            || r.offset
                .checked_add(r.size)
                .is_none_or(|end| end > r.buffer.size())
        {
            return Err(fault("native draft original initialized R owner"));
        }
        let id = self
            .draft_scope
            .prepare(
                request,
                strata_schedule::Access {
                    buffer: raw_buffer(r)?.as_raw(),
                    first: r.offset,
                    bytes: r.size,
                    write: false,
                },
            )
            .map_err(fault)?;
        self.draft_source = Some(source); // BEFORE any fallible native submission.
        if self.draft_readiness.is_none() {
            self.draft_readiness =
                Some(super::strata_handoff::Handoff::new(&self.device).map_err(fault)?);
        }
        // Both commands are queued now, before commit. Switching to the draft
        // lane later MUST NOT queue a fresh MAIN signal behind that commit.
        self.draft_readiness
            .as_mut()
            .unwrap()
            .transfer(self.compute_queues[0].queue, self.compute_queues[1].queue)
            .map_err(fault)?;
        self.draft_scope.dependency_submitted(id).map_err(fault)?;
        cubecl_runtime::server::NativeStrataDraftScope::from_queue(
            self.identity,
            id.generation,
            request,
        )
        .map_err(fault)
    }
    fn draft_identity(
        &self,
        token: cubecl_runtime::server::NativeStrataDraftScope,
        request: u64,
    ) -> Result<super::strata_draft_scope::Identity, ServerError> {
        let (graph, generation, original) = token.original_identity();
        if graph != self.identity || original != request {
            return Err(fault("native draft foreign graph/request"));
        }
        Ok(super::strata_draft_scope::Identity {
            generation,
            request,
        })
    }
    pub(crate) fn enter_draft_scope(
        &mut self,
        token: cubecl_runtime::server::NativeStrataDraftScope,
        request: u64,
    ) -> Result<(), ServerError> {
        let id = self.draft_identity(token, request)?;
        self.draft_scope.enter(id).map_err(fault)?;
        self.select_branch_queue(None)
    }
    pub(crate) fn end_draft_scope(
        &mut self,
        token: cubecl_runtime::server::NativeStrataDraftScope,
        request: u64,
    ) -> Result<(), ServerError> {
        let id = self.draft_identity(token, request)?;
        self.draft_scope.check_active(id).map_err(fault)?;
        self.complete_submission()?;
        if !self.nodes.is_empty() {
            return Err(fault("native draft end has unsubmitted consumers"));
        }
        self.switch_to_lane(0, true)?;
        self.handoff.complete().map_err(fault)?;
        self.draft_scope.finish_joined(id).map_err(fault)?;
        self.draft_source = None;
        Ok(())
    }
    /// Every unkeyed producer and named MTP branch stays inside the original
    /// active scope. Entry waits on the desired lane, never MAIN behind commit.
    pub(crate) fn select_branch_queue(
        &mut self,
        branch: Option<cubecl_runtime::server::NativeStrataBranch>,
    ) -> Result<(), ServerError> {
        if self.shared_request.is_some() {
            return Err(fault("native branch switch inside shared section"));
        }
        let named = branch.map(|b| usize::from(b.mtp_geometry().is_some()));
        let lane = self.draft_scope.lane(named).map_err(fault)?;
        if self.draft_scope.active_request().is_some() {
            self.preflight_flights()?;
            self.complete_lane_submission(lane)?;
            self.handoff.complete().map_err(fault)?;
        } else {
            self.complete_submission()?;
        }
        if !self.nodes.is_empty() {
            return Err(fault("native queue switch has unsubmitted nodes"));
        }
        let prepared_dependency = lane == 1 && self.draft_source.is_some();
        self.switch_to_lane(lane, !prepared_dependency)
    }
    fn switch_to_lane(&mut self, lane: usize, transfer: bool) -> Result<(), ServerError> {
        if lane == self.compute_lane {
            return Ok(());
        }
        // Only scratch pools are here. Original recording pools and resources
        // stay pinned by their own recording/flight inventories.
        unsafe {
            self.device.destroy_descriptor_pool(self.descriptors, None);
            self.device.destroy_command_pool(self.pool, None);
        }
        self.descriptors = vk::DescriptorPool::null();
        self.pool = vk::CommandPool::null();
        self.command = vk::CommandBuffer::null();
        let selected = self.compute_queues[lane];
        if transfer {
            self.handoff
                .transfer(self.queue, selected.queue)
                .map_err(fault)?;
        }
        self.family = selected.family;
        self.queue = selected.queue;
        self.timestamp_bits = selected.timestamp_bits;
        self.compute_lane = lane;
        Ok(())
    }
    fn info_queue(&self, lane: usize) -> vk::Queue {
        self.compute_queues[lane].queue
    }
    pub(super) fn upload_parameters(
        &mut self,
        copies: Vec<super::strata_info_transfer::Copy>,
    ) -> Result<(), ServerError> {
        if copies.is_empty() {
            return Ok(());
        }
        let accesses = super::strata_info_transfer::accesses(&copies).map_err(fault)?;
        self.draft_scope.check_accesses(&accesses).map_err(fault)?;
        let lane = self.compute_lane;
        let queue = self.info_queue(lane);
        self.admit_accesses(&accesses, queue)?;
        // Reuse only the INFO command pool's original completed copy. Its old
        // math consumers retain their own complete resources and header epochs.
        // Drop temporary ranges before allocating submit's validation metadata.
        drop(accesses);
        self.info_uploads[lane].complete().map_err(fault)?;
        self.info_uploads[lane].submit(queue, copies).map_err(fault)
    }
    /// Preflight BOTH original lanes before granting any completion credit.
    fn preflight_flights(&self) -> Result<(), ServerError> {
        for lane in 0..2 {
            if let Some(flight) = self.flights.peek(lane).map_err(fault)? {
                let recording = self
                    .recordings
                    .iter()
                    .find(|r| r.id == flight.recording)
                    .ok_or_else(|| fault("native pending access recording missing"))?;
                if !flight.submitted || recording.family != self.compute_queues[lane].family {
                    return Err(fault("native pending access original family/submission"));
                }
            }
        }
        Ok(())
    }
    fn access_joins(&self, incoming: &[strata_schedule::Access]) -> Result<[bool; 2], ServerError> {
        self.preflight_flights()?;
        let mut joins = [false; 2];
        for (lane, needed) in joins.iter_mut().enumerate() {
            if let Some(flight) = self.flights.peek(lane).map_err(fault)? {
                let recording = self
                    .recordings
                    .iter()
                    .find(|r| r.id == flight.recording)
                    .ok_or_else(|| fault("native pending access recording missing"))?;
                *needed = recording.work.plan.conflicts(incoming).map_err(fault)?;
            }
        }
        Ok(joins)
    }
    fn join_accesses(
        &mut self,
        mut joins: [bool; 2],
        destination: vk::Queue,
    ) -> Result<(), ServerError> {
        // The selected compute lane owns its command/fence scratch. Copies
        // may additionally use the transfer lane; conflicts still name their
        // original compute fence, which covers every segment of that work.
        joins[self.compute_lane] = true;
        for (lane, needed) in joins.into_iter().enumerate() {
            if needed && self.flights.peek(lane).map_err(fault)?.is_some() {
                self.complete_lane_submission(lane)?;
                let source = self.compute_queues[lane].queue;
                if source != destination {
                    self.handoff.transfer(source, destination).map_err(fault)?;
                    self.handoff.complete().map_err(fault)?;
                }
            }
        }
        Ok(())
    }
    fn admit_accesses(
        &mut self,
        incoming: &[strata_schedule::Access],
        destination: vk::Queue,
    ) -> Result<(), ServerError> {
        if self.draft_scope.active_request().is_none() {
            self.complete_submission()?;
            return Ok(());
        }
        let joins = self.access_joins(incoming)?;
        self.join_accesses(joins, destination)
    }
    fn admit_math_submission(&mut self) -> Result<(), ServerError> {
        if self.draft_scope.active_request().is_some() {
            self.complete_scoped_submission()
        } else {
            // Preserve the original math header's actual INFO timeline wait.
            self.complete_math_submission()
        }
    }
    /// Ordinary/final synchronization remains global. Only the original active
    /// draft scope can finish its own lane without joining unrelated MAIN.
    pub(crate) fn complete_scoped_submission(&mut self) -> Result<(), ServerError> {
        if self.draft_scope.active_request().is_none() {
            return self.complete_submission();
        }
        if self.compute_lane != 1 {
            return Err(fault("native active draft selected lane mismatch"));
        }
        self.preflight_flights()?;
        self.complete_lane_submission(1)?;
        self.handoff.complete().map_err(fault)?;
        if let Some(ready) = self.draft_readiness.as_mut() {
            ready.complete().map_err(fault)?;
        }
        // CPU reuse of an INFO source requires the real transfer fence, but
        // does not consume the timeline visibility owed to future math.
        self.info_uploads[1].complete().map_err(fault)
    }
    /// Real Wgpu work needs all old native users and MAIN visibility. Cold root
    /// initialization must not be silently skipped to manufacture overlap.
    pub(crate) fn complete_primary_access(&mut self) -> Result<(), ServerError> {
        self.complete_submission()?;
        if self.compute_lane != 0 {
            self.handoff
                .transfer(self.queue, self.compute_queues[0].queue)
                .map_err(fault)?;
            self.handoff.complete().map_err(fault)?;
        }
        Ok(())
    }
    pub(crate) fn queued(&self) -> bool {
        !self.nodes.is_empty()
    }
    /// Actual selective checkpoint for the same serialized stream's immutable
    /// full I/O owners. No new work/owner mutation may interleave before apply.
    /// Original named plans stay pinned for FUTURE direct replay, not just while
    /// a matching flight happens to be running.
    pub(super) fn certify_io_reuse(
        &mut self,
        owners: super::transient_io::NativeIoOwners<'_>,
    ) -> Result<super::strata_io_reuse::Cohort, ServerError> {
        if !self.nodes.is_empty() {
            return Err(fault("native I/O reuse has unsubmitted original work"));
        }
        // Preflight every original owner before joining/crediting any flight.
        for recording in &self.recordings {
            if !recording.ready || recording.key.is_some_and(|k| k.nonce != owners.request()) {
                return Err(fault("native I/O reuse partial/foreign retained recording"));
            }
        }
        for lane in 0..2 {
            if let Some(flight) = self.flights.peek(lane).map_err(fault)? {
                let original = self
                    .recordings
                    .iter()
                    .find(|r| r.id == flight.recording)
                    .ok_or_else(|| fault("native I/O original flight recording missing"))?;
                if !flight.submitted || original.family != self.compute_queues[lane].family {
                    return Err(fault("native I/O original flight family/submission"));
                }
            }
        }
        if let Some(request) = self.draft_scope.active_request() {
            if request != owners.request() || self.compute_lane != 1 {
                return Err(fault("native I/O reuse foreign original draft scope"));
            }
            self.complete_lane_submission(1)?;
            self.handoff.complete().map_err(fault)?;
            if let Some(ready) = self.draft_readiness.as_mut() {
                ready.complete().map_err(fault)?;
            }
            // Source CPU reuse needs the REAL copy fence, not target math's
            // completion. Do not credit an unrelated INFO header/visibility.
            self.info_uploads[1].complete().map_err(fault)?;
        } else {
            self.complete_submission()?;
        }
        let mut cohort =
            super::strata_io_reuse::Cohort::new(owners.request(), owners.epoch(), owners.len())
                .map_err(fault)?;
        for (buffer, pinned) in owners.owners() {
            if buffer.size() < 256 || !buffer.size().is_power_of_two() {
                return Err(fault("native I/O original dedicated full owner"));
            }
            let raw = unsafe { buffer.as_hal::<wgpu::hal::api::Vulkan>() }
                .ok_or_else(|| fault("native I/O original Vulkan allocation"))?;
            cohort
                .owner(unsafe { raw.raw_buffer_handle() }.as_raw(), pinned)
                .map_err(fault)?;
        }
        cohort.seal_inventory().map_err(fault)?;
        // A scoped MTP join does not join MAIN's original INFO copy. Its
        // source staging must not be reused/erased while that copy can read it.
        for uploads in &self.info_uploads {
            for access in uploads.pending_accesses().map_err(fault)? {
                cohort.exclude_buffer(access.buffer).map_err(fault)?;
            }
        }
        for recording in &self.recordings {
            if recording.key.is_some() || self.flights.live(recording.id) {
                for buffer in recording.work.plan.buffer_ids() {
                    cohort.exclude_buffer(buffer).map_err(fault)?;
                }
            }
        }
        Ok(cohort)
    }
    pub(crate) fn clear_completed_recordings(&mut self) -> Result<(), ServerError> {
        if self.flights.any() || self.pending() || self.info_uploads.iter().any(|u| u.pending()) {
            return Err(fault("native graph cleanup while pending"));
        }
        // Cleanup may follow a failed flush. It must not bypass an unknown
        // ordinary cross-lane handoff merely because math flights are empty.
        self.handoff.complete().map_err(fault)?;
        if let Some(ready) = self.draft_readiness.as_mut() {
            ready.complete().map_err(fault)?;
        }
        self.draft_scope.discard_after_drain();
        self.draft_source = None;
        self.shared_request = None; // Only after the original all-queue drain.
        self.flights.clear_joined_exports().map_err(fault)?;
        self.recordings.clear();
        self.initialized.clear();
        Ok(())
    }
    pub(crate) fn pending(&self) -> bool {
        !self.nodes.is_empty() || self.flights.any()
    }
    pub(crate) fn has_recording(&self, key: super::strata_parameters::RecordingKey) -> bool {
        !profile_active()
            && self.recordings.iter().any(|r| {
                r.ready
                    && r.key == Some(key)
                    && r.family == self.family
                    && r.profile.is_none()
                    && !self.flights.live(r.id)
            })
    }
    /// Submit the original retained commands/resources without building nodes,
    /// pipeline descriptors or a signature again. Only the owning stream may
    /// call this, after submitting its actual current INFO copies. The original
    /// header waits on their real transfer timeline before any math consumes it.
    pub(crate) fn replay_recording(
        &mut self,
        key: super::strata_parameters::RecordingKey,
        wait: bool,
        primary_writes: bool,
    ) -> Result<(), ServerError> {
        self.admit_math_submission()?;
        if self.queued() || !self.has_recording(key) {
            return Err(fault("native direct replay recording unavailable"));
        }
        let original = self
            .recordings
            .iter()
            .find(|r| {
                r.ready
                    && r.key == Some(key)
                    && r.family == self.family
                    && r.profile.is_none()
                    && !self.flights.live(r.id)
            })
            .ok_or_else(|| fault("native direct replay recording missing"))?;
        let id = original.id;
        // The actual retained plan supplies all ranges. No rebuilt-node or
        // caller-declared substitute is accepted for direct replay.
        let joins = self.access_joins(original.work.plan.accesses())?;
        self.join_accesses(joins, self.queue)?;
        // Joining sampled peer work can remove a Vec entry. Re-find the exact
        // stable original ID rather than using a pre-join position.
        let hit = self
            .recordings
            .iter()
            .position(|r| r.id == id)
            .ok_or_else(|| fault("native direct replay original disappeared"))?;
        let recording = self.recordings.remove(hit);
        let compute_count = recording.compute_count;
        self.recordings.push(recording);
        if self.compute_lane != 0 && primary_writes {
            self.handoff
                .transfer(self.compute_queues[0].queue, self.queue)
                .map_err(fault)?;
        }
        self.submit_last_recording(compute_count, true, false)?;
        if wait {
            self.complete_scoped_submission()?;
        }
        Ok(())
    }
    pub(crate) fn preflight_pending_commit(
        &self,
        key: cubecl_runtime::server::NativeStrataBranch,
    ) -> Result<(), ServerError> {
        if profile_active()
            || key.commit_geometry().is_none()
            || !self.flights.export_available(0).map_err(fault)?
        {
            return Err(fault(
                "native pending commit already exported or foreign family",
            ));
        }
        Ok(())
    }
    /// Export only the exact original submitted target-commit recording.
    pub(crate) fn export_pending_commit(
        &mut self,
        key: cubecl_runtime::server::NativeStrataBranch,
        scope: u64,
    ) -> Result<cubecl_runtime::server::NativeStrataSubmission, ServerError> {
        if self.compute_lane != 0 || key.commit_geometry().is_none() {
            return Err(fault("native pending commit family"));
        }
        let flight = self
            .flights
            .peek(0)
            .map_err(fault)?
            .ok_or_else(|| fault("native pending original flight missing"))?;
        let original = self
            .recordings
            .iter()
            .find(|r| r.id == flight.recording)
            .ok_or_else(|| fault("native pending original recording missing"))?;
        let recording = original
            .key
            .ok_or_else(|| fault("native pending commit recording unkeyed"))?;
        if recording.branch != key
            || recording.nonce != scope
            || !flight.submitted
            || original.family != self.compute_queues[0].family
        {
            return Err(fault("native pending original scope/recording mismatch"));
        }
        let submission = self.flights.expose(0, scope).map_err(fault)?;
        cubecl_runtime::server::NativeStrataSubmission::from_queue(self.identity, submission, scope)
            .map_err(fault)
    }
    /// A dependent drain may have already waited on this fence. Keep its exact
    /// exported completion until acknowledged; never match merely the LRU tail.
    pub(crate) fn join_exported(
        &mut self,
        token: cubecl_runtime::server::NativeStrataSubmission,
        scope: u64,
    ) -> Result<(), ServerError> {
        let (graph, submission, original_scope) = token.original_identity();
        if graph != self.identity || original_scope != scope {
            return Err(fault("native pending foreign graph/scope"));
        }
        let (lane, export) = self.flights.exported(submission, scope).map_err(fault)?;
        if !export.joined {
            let flight = self
                .flights
                .peek(lane)
                .map_err(fault)?
                .ok_or_else(|| fault("native pending original flight missing"))?;
            if flight.submission != submission || flight.recording != export.recording {
                return Err(fault("native pending original flight replaced"));
            }
            self.complete_lane_submission(lane)?;
        }
        self.flights.acknowledge(submission, scope).map_err(fault)
    }
    pub(crate) fn complete_submission(&mut self) -> Result<(), ServerError> {
        self.complete_math_submission()?;
        if let Some(ready) = self.draft_readiness.as_mut() {
            ready.complete().map_err(fault)?;
        }
        // An older math flight may have coexisted with a newer disjoint INFO
        // copy. Only its actual header epoch was consumed, so decide AFTER
        // joining the original flights whether any newer copy still lacks a
        // completed consumer. Merely having had math is not visibility proof.
        for lane in 0..2 {
            let info_only = self.info_uploads[lane].wait();
            self.info_uploads[lane].complete().map_err(fault)?;
            if let Some(info_only) = info_only {
                // Every debt keeps its original lane/epoch. A completed copy
                // alone never credits visibility to math or later Wgpu cleanup.
                let source = self.info_queue(lane);
                let native = self.compute_queues[lane].queue;
                // Even a same-queue INFO-only drain needs an ACTUAL GPU
                // dependency. A host copy fence alone does not make writes
                // visible to future native/Wgpu commands on this queue.
                self.handoff.visibility(source, native).map_err(fault)?;
                self.handoff.complete().map_err(fault)?;
                if native != self.compute_queues[0].queue {
                    self.handoff
                        .transfer(native, self.compute_queues[0].queue)
                        .map_err(fault)?;
                    self.handoff.complete().map_err(fault)?;
                }
                self.info_uploads[lane]
                    .math_fence_joined(info_only.value)
                    .map_err(fault)?;
            }
        }
        Ok(())
    }
    /// Attach the actual retained recording to its own lane fence before
    /// submitting anything. Partial failures remain owned and never complete.
    fn prepare_last_recording(
        &mut self,
        compute_count: usize,
        replay: bool,
        profiling: bool,
        require_ready: bool,
    ) -> Result<Option<super::strata_info_transfer::Wait>, ServerError> {
        let recording = self
            .recordings
            .last()
            .ok_or_else(|| fault("native submission recording missing"))?;
        if recording.ready != require_ready
            || recording.family != self.family
            || recording.compute_count != compute_count
        {
            return Err(fault("native submission original recording family"));
        }
        if let Some(source) = self.draft_scope.protected()
            && recording
                .work
                .plan
                .conflicts(std::slice::from_ref(source))
                .map_err(fault)?
        {
            return Err(fault(
                "native recorded writer aliases verified draft source",
            ));
        }
        let id = recording.id;
        let joins = self.access_joins(recording.work.plan.accesses())?;
        self.join_accesses(joins, self.queue)?;
        // Joining a sampled peer may remove an earlier Vec entry; ownership
        // must still resolve to this exact original tail, never an old index.
        if self.recordings.last().is_none_or(|r| r.id != id) {
            return Err(fault("native original submission recording changed"));
        }
        let fence = self.fences[self.compute_lane];
        self.flights
            .begin(self.compute_lane, id, compute_count, replay, profiling)
            .map_err(fault)?;
        let info_wait = self.info_uploads[self.compute_lane].wait();
        if let Some(wait) = info_wait {
            self.flights
                .header_wait(self.compute_lane, id, wait.value)
                .map_err(fault)?;
        }
        // Reserve first: never reset a lane fence while its previous owner is
        // still live. A reset failure leaves conservative terminal retention.
        unsafe { self.device.reset_fences(&[fence]) }
            .map_err(|_| fault("native lane fence reset"))?;
        Ok(info_wait)
    }
    fn mark_last_submitted(&mut self, id: u64) -> Result<(), ServerError> {
        if self
            .recordings
            .last()
            .is_none_or(|recording| recording.id != id || !recording.ready)
        {
            return Err(fault("native completed recording identity changed"));
        }
        self.flights
            .submitted(self.compute_lane, id)
            .map_err(fault)?;
        // The recording owns every original resource. Drop only duplicate
        // enqueue metadata so another lane can build its own subsequent work.
        self.nodes.clear();
        Ok(())
    }
    fn submit_last_recording(
        &mut self,
        compute_count: usize,
        replay: bool,
        profiling: bool,
    ) -> Result<(), ServerError> {
        let info_wait = self.prepare_last_recording(compute_count, replay, profiling, true)?;
        let id = self.recordings.last().unwrap().id;
        let fence = self.fences[self.compute_lane];
        self.recordings
            .last_mut()
            .unwrap()
            .work
            .submit(
                [
                    self.queue,
                    self.copy_queue.raw_identity(),
                    self.compute_queues
                        [strata_schedule::shared_peer(self.compute_lane).map_err(fault)?]
                    .queue,
                ],
                &self.copy_queue,
                fence,
                info_wait,
            )
            .map_err(fault)?;
        self.mark_last_submitted(id)
    }
    fn complete_math_submission(&mut self) -> Result<(), ServerError> {
        self.handoff.complete().map_err(fault)?;
        for lane in 0..2 {
            self.complete_lane_submission(lane)?;
        }
        // A newly staged INFO upload has no math flight yet: retain its real
        // timeline dependency until execute/replay consumes it.
        Ok(())
    }
    fn complete_lane_submission(&mut self, lane: usize) -> Result<(), ServerError> {
        let Some(flight) = self.flights.peek(lane).map_err(fault)? else {
            return Ok(());
        };
        if !flight.submitted {
            return Err(fault("native partial submission has no completed join"));
        }
        let at = self
            .recordings
            .iter()
            .position(|r| r.id == flight.recording)
            .ok_or_else(|| fault("native original in-flight recording missing"))?;
        if self.recordings[at].family != self.compute_queues[lane].family {
            return Err(fault("native flight original queue family mismatch"));
        }
        let wait_start = std::time::Instant::now();
        unsafe {
            self.device
                .wait_for_fences(&[self.fences[lane]], true, 120_000_000_000)
        }
        .map_err(|_| fault("native lane completion"))?;
        WAIT_NS.fetch_add(
            wait_start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        if flight.profiling {
            self.recordings[at]
                .profile
                .as_ref()
                .ok_or_else(|| fault("native sampled original flight owner"))?
                .collect()?;
        }
        self.info_uploads[lane]
            .math_fence_joined(flight.info_epoch)
            .map_err(fault)?;
        // Only the actual matching fence above authorizes bookkeeping.
        let completed = self
            .flights
            .finish_joined(lane, flight.recording)
            .map_err(fault)?;
        if completed.profiling {
            self.recordings.remove(at);
        }
        self.submissions += 1;
        self.replays += u64::from(completed.replay);
        self.dispatches += completed.compute_count as u64;
        credit(completed.compute_count, completed.replay);
        Ok(())
    }
    /// Packet I/O is an unnamed original recording, even if a caller still
    /// owns an open named math branch. Never replace its retained shape/key
    /// with a fill/copy-only signature or credit I/O as another math invocation.
    pub(crate) fn queued_transfer_only(&self) -> bool {
        !self.nodes.is_empty()
            && self
                .nodes
                .iter()
                .all(|n| matches!(n.dispatch, Dispatch::Copy | Dispatch::Fill))
    }
    pub(crate) fn execute_readback(&mut self) -> Result<(), ServerError> {
        if self
            .nodes
            .iter()
            .any(|n| !matches!(n.dispatch, Dispatch::Copy | Dispatch::Fill))
        {
            return Err(fault("native readback has unsubmitted math"));
        }
        self.execute(true, None, false)
    }
    /// Prepare only fresh, never-bound sets for one bounded ORIGINAL-node
    /// window. Recording still owns the pool/resources through its real fence;
    /// no set is cached, recycled, updated after use or granted completion here.
    fn fresh_descriptors(
        &self,
        nodes: &[Node],
        pool: vk::DescriptorPool,
        retained_metadata: usize,
        native: &capture_buffers::Guarded<'_>,
    ) -> Result<descriptors::Batch, ServerError> {
        let plan = descriptors::Plan::new(
            nodes.iter().map(|n| match n.dispatch {
                Dispatch::Copy | Dispatch::Fill => None,
                _ if n
                    .pipeline
                    .as_ref()
                    .is_some_and(|p| self.pipelines[p].push_descriptors) =>
                {
                    None
                }
                _ => Some(n.resources.len()),
            }),
            retained_metadata,
        )
        .map_err(fault)?;
        let mut layouts = Vec::with_capacity(plan.set_count());
        let mut infos = Vec::with_capacity(plan.binding_count());
        let mut types = Vec::with_capacity(plan.binding_count());
        // All actual layouts and full owner intervals are checked before the
        // bulk allocation. Finish infos before borrowing any of its pointers.
        for (index, n) in nodes.iter().enumerate() {
            if plan.set_index(index).map_err(fault)?.is_none() {
                continue;
            }
            let p = self
                .pipelines
                .get(
                    n.pipeline
                        .as_ref()
                        .ok_or_else(|| fault("native descriptor compute identity"))?,
                )
                .ok_or_else(|| fault("native descriptor original pipeline"))?;
            if n.resources.len() != p.program.bindings.len()
                || plan.binding_range(index).map_err(fault)?.start != infos.len()
            {
                return Err(fault("native descriptor original binding extent"));
            }
            layouts.push(p.set_layout);
            for (binding, r) in n.resources.iter().enumerate() {
                if r.size == 0
                    || r.offset
                        .checked_add(r.size)
                        .is_none_or(|end| end > r.buffer.size())
                {
                    return Err(fault("native descriptor actual owner extent"));
                }
                infos.push(
                    vk::DescriptorBufferInfo::default()
                        .buffer(native.get(r)?)
                        .offset(r.offset)
                        .range(r.size),
                );
                types.push(p.program.bindings[binding]);
            }
        }
        if layouts.len() != plan.set_count() || infos.len() != plan.binding_count() {
            return Err(fault("native descriptor complete window inventory"));
        }
        let sets = if layouts.is_empty() {
            Vec::new()
        } else {
            unsafe {
                self.device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(pool)
                        .set_layouts(&layouts),
                )
            }
            .map_err(|_| fault("native graph descriptor allocation"))?
        };
        // Pool ownership was transferred to Recording before this fallible
        // operation. On any failure the original rolling guard still waits or
        // quarantines real submitted users before the pool can be destroyed.
        if sets.len() != plan.set_count() || sets.capacity() > descriptors::NODE_WINDOW {
            return Err(fault("native descriptor actual allocated inventory"));
        }
        let writes = plan.writes(&sets, &infos, &types).map_err(fault)?;
        if writes.len() != plan.binding_count() {
            return Err(fault("native descriptor complete write inventory"));
        }
        if !writes.is_empty() {
            // Only new sets; no descriptor already named by any submitted or
            // recorded command is ever changed. Binding order/ranges are exact.
            unsafe { self.device.update_descriptor_sets(&writes, &[]) };
        }
        descriptors::Batch::new(plan, sets).map_err(fault)
    }
    pub(crate) fn execute(
        &mut self,
        wait: bool,
        key: Option<super::strata_parameters::RecordingKey>,
        primary_writes: bool,
    ) -> Result<(), ServerError> {
        self.admit_math_submission()?;
        if self.nodes.is_empty() {
            return Ok(());
        }
        let capture_start = std::time::Instant::now();
        // The caller joins real primary-queue initialization/readback work.
        // INFO copies carry their own actual native transfer dependency. Every
        // original owner stays retained through the real completion fence.
        if self.compute_lane != 0 && primary_writes {
            self.handoff
                .transfer(self.compute_queues[0].queue, self.queue)
                .map_err(fault)?;
        }
        let owners = binding_owners::Owners::collect(
            self.nodes.iter().filter_map(|node| node.owners.as_ref()),
            descriptors::METADATA_CAP,
        )
        .map_err(fault)?;
        let old_metadata = self
            .recordings
            .iter()
            .try_fold(owners.metadata_bytes(), |n, r| {
                n.checked_add(r.metadata_bytes()?)
            })
            .ok_or_else(|| fault("native capture previous metadata overflow"))?;
        let signature_words = self
            .nodes
            .iter()
            .try_fold(0usize, |sum, n| {
                n.resources
                    .len()
                    .checked_add(n.read_dependencies.len())?
                    .checked_mul(3)?
                    .checked_add(9)?
                    .checked_add(sum)
            })
            .ok_or_else(|| fault("native capture signature extent"))?;
        let mut signature = Vec::new();
        let available = descriptors::METADATA_CAP
            .checked_sub(old_metadata)
            .ok_or_else(|| fault("native capture previous metadata capacity"))?;
        if signature_words
            .checked_mul(size_of::<u64>())
            .is_none_or(|n| n > available)
        {
            return Err(fault("native capture signature metadata capacity"));
        }
        signature
            .try_reserve_exact(signature_words)
            .map_err(|_| fault("native capture signature allocation"))?;
        let resource_budget = available
            .checked_sub(
                signature
                    .capacity()
                    .checked_mul(size_of::<u64>())
                    .ok_or_else(|| fault("native capture signature capacity overflow"))?,
            )
            .ok_or_else(|| fault("native capture signature metadata capacity"))?;
        // These are the original lifetime roots, not replacement access views.
        let resources = binding_owners::unique_resources(
            self.nodes.iter().flat_map(|n| {
                n.resources
                    .iter()
                    .chain(&n.read_dependencies)
                    .chain(match &n.dispatch {
                        Dispatch::Indirect(r) => Some(r),
                        _ => None,
                    })
            }),
            |a, b| a.buffer.cmp(&b.buffer),
            resource_budget,
        )
        .map_err(fault)?;
        let resource_bytes = resources
            .capacity()
            .checked_mul(size_of::<WgpuResource>())
            .ok_or_else(|| fault("native capture resource metadata overflow"))?;
        let keys = capture_buffers::Keys::new(
            &resources,
            resource_budget
                .checked_sub(resource_bytes)
                .ok_or_else(|| fault("native capture resource metadata capacity"))?,
        )?;
        let native = keys.acquire()?;
        for n in &self.nodes {
            signature.push(
                n.pipeline
                    .as_ref()
                    .map_or(0, |p| self.pipelines[p].pipeline.as_raw()),
            );
            signature.extend([u64::from(n.shared), u64::from(n.release_after)]);
            signature.push(n.resources.len() as u64);
            for r in &n.resources {
                signature.extend([native.get(r)?.as_raw(), r.offset, r.size]);
            }
            signature.push(n.read_dependencies.len() as u64);
            for r in &n.read_dependencies {
                signature.extend([native.get(r)?.as_raw(), r.offset, r.size]);
            }
            match &n.dispatch {
                Dispatch::Direct(x, y, z) => signature.extend([0, *x as u64, *y as u64, *z as u64]),
                Dispatch::Indirect(r) => {
                    signature.extend([1, native.get(r)?.as_raw(), r.offset, r.size])
                }
                Dispatch::Copy => signature.extend([2, 0, 0, 0]),
                Dispatch::Fill => signature.extend([3, 0, 0, 0]),
            }
        }
        drop(native); // No guard crosses recording eviction, replay or cleanup.
        let compute_count = self.nodes.iter().filter(|n| n.pipeline.is_some()).count();
        let profiling = profile_active() && compute_count >= 2048;
        if let Some(key) = key {
            if let Some(previous) = self.recordings.iter().find(|r| r.key == Some(key)) {
                if !previous.ready
                    || previous.signature != signature
                    || previous.family != self.family
                {
                    return Err(fault("native named branch recording changed"));
                }
            }
        }
        if let Some(hit) = self
            .recordings
            .iter()
            .position(|r| {
                r.ready
                    && r.key == key
                    && r.family == self.family
                    && r.signature == signature
                    // A timestamped recording must not become the ordinary
                    // replay after the one sampled window has ended.
                    && r.profile.is_some() == profiling
                    && !self.flights.live(r.id)
            })
            .filter(|_| !profiling)
        {
            let recording = self.recordings.remove(hit);
            self.recordings.push(recording);
            drop(keys);
            drop(resources);
            self.submit_last_recording(compute_count, true, false)?;
            if wait {
                self.complete_scoped_submission()?;
            }
            return Ok(());
        }
        // Named branches pin their original owners until request erasure; only
        // unnamed opportunistic recordings participate in the sixteen-plan LRU.
        // Profiling may re-record an identical named shape after its real fence.
        if let Some(key) = key {
            if let Some(i) = self.recordings.iter().position(|r| r.key == Some(key)) {
                if self.flights.live(self.recordings[i].id) {
                    return Err(fault("native named recording still in flight"));
                }
                self.recordings.remove(i);
            }
        } else if self.recordings.iter().filter(|r| r.key.is_none()).count() >= 16 {
            if let Some(i) = self
                .recordings
                .iter()
                .position(|r| r.key.is_none() && !self.flights.live(r.id))
            {
                self.recordings.remove(i);
            }
        }
        if self.pool == vk::CommandPool::null() {
            self.pool = unsafe {
                self.device.create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                        .queue_family_index(self.family),
                    None,
                )
            }
            .map_err(|_| fault("native graph replay pool"))?;
            let fresh = self.nodes.iter().filter(|n| {
                n.pipeline
                    .as_ref()
                    .is_some_and(|p| !self.pipelines[p].push_descriptors)
            });
            let set_count = fresh.clone().count();
            let storage = fresh.map(|n| n.resources.len() as u32).sum::<u32>();
            let sizes = [
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: storage.max(1),
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: set_count.max(1) as u32,
                },
            ];
            self.descriptors = unsafe {
                self.device.create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(set_count.max(1) as u32)
                        .pool_sizes(&sizes),
                    None,
                )
            }
            .map_err(|_| fault("native graph replay descriptors"))?;
            self.command = unsafe {
                self.device.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(self.pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
            }
            .map_err(|_| fault("native graph replay command"))?[0];
        }
        let timestamp_bits = if self.compute_lane == 0
            && self
                .nodes
                .iter()
                .any(|n| matches!(n.dispatch, Dispatch::Copy | Dispatch::Fill))
        {
            self.timestamp_bits.min(self.copy_timestamp_bits)
        } else {
            self.timestamp_bits
        };
        let timestamp_bits = if self.nodes.iter().any(|n| n.shared) {
            timestamp_bits.min(
                self.compute_queues
                    [strata_schedule::shared_peer(self.compute_lane).map_err(fault)?]
                .timestamp_bits,
            )
        } else {
            timestamp_bits
        };
        let profile = if profiling {
            if self.nodes.len() > 16384
                || timestamp_bits == 0
                || timestamp_bits > 64
                || !self.timestamp_period.is_finite()
                || self.timestamp_period <= 0.0
            {
                return Err(fault("native public operator timing capacity"));
            }
            let names = self
                .nodes
                .iter()
                .map(|n| match n.dispatch {
                    Dispatch::Copy => "OrderedCopy",
                    Dispatch::Fill => "OrderedFill",
                    _ => self.pipelines[n.pipeline.as_ref().unwrap()].program.name,
                })
                .collect::<Vec<_>>();
            let pool = unsafe {
                self.device.create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::TIMESTAMP)
                        .query_count((names.len() * 2) as u32),
                    None,
                )
            }
            .map_err(|_| fault("native public timestamp pool"))?;
            Some(OperatorProfile {
                device: self.device.clone(),
                pool,
                names,
                period: self.timestamp_period,
                mask: if timestamp_bits == 64 {
                    u64::MAX
                } else {
                    (1u64 << timestamp_bits) - 1
                },
            })
        } else {
            None
        };
        let largest_access_count = self
            .nodes
            .iter()
            .map(|n| {
                n.resources.len()
                    + n.read_dependencies.len()
                    + usize::from(matches!(n.dispatch, Dispatch::Indirect(_)))
            })
            .max()
            .unwrap_or(0);
        let mut scratch = capture_scratch::Scratch::new(largest_access_count).map_err(fault)?;
        let mut compute_ranges = Vec::<compute_barriers::Access>::new();
        compute_ranges
            .try_reserve_exact(largest_access_count)
            .map_err(|_| fault("native compute barrier scratch allocation"))?;
        let compute_scratch_bytes = compute_ranges
            .capacity()
            .checked_mul(size_of::<compute_barriers::Access>())
            .and_then(|n| n.checked_add(size_of_val(&compute_ranges)))
            .ok_or_else(|| fault("native compute barrier scratch metadata overflow"))?;
        if compute_scratch_bytes > 1024 * 1024 {
            return Err(fault("native compute barrier scratch capacity"));
        }
        let scratch_bytes = scratch
            .metadata_bytes()
            .map_err(fault)?
            .checked_add(keys.metadata_bytes())
            .and_then(|n| n.checked_add(compute_scratch_bytes))
            .and_then(|n| {
                n.checked_add(if self.push_descriptors.is_some() {
                    push_descriptors::STACK_BYTES
                } else {
                    0
                })
            })
            .ok_or_else(|| fault("native capture guarded metadata overflow"))?;
        let native = keys.acquire()?;
        let preparation_used = self
            .recordings
            .iter()
            .try_fold(
                signature
                    .capacity()
                    .checked_mul(size_of::<u64>())
                    .and_then(|n| n.checked_add(scratch_bytes))
                    .and_then(|n| n.checked_add(resource_bytes))
                    .and_then(|n| n.checked_add(owners.metadata_bytes()))
                    .ok_or_else(|| fault("native transfer preparation metadata overflow"))?,
                |n, r| n.checked_add(r.metadata_bytes()?),
            )
            .ok_or_else(|| fault("native transfer preparation metadata overflow"))?;
        let preparation_budget = descriptors::METADATA_CAP
            .checked_sub(preparation_used)
            .ok_or_else(|| fault("native transfer preparation metadata capacity"))?;
        let copy_count = self
            .nodes
            .iter()
            .filter(|n| {
                self.compute_lane == 0
                    && !n.shared
                    && matches!(n.dispatch, Dispatch::Copy | Dispatch::Fill)
            })
            .count();
        let mut classes = compute_barriers::ClassBuilder::new(
            self.nodes.len(),
            preparation_budget.min(1024 * 1024),
        )
        .ok();
        let mut planner =
            strata_schedule::Planner::for_lane(self.nodes.len(), profiling, self.compute_lane)
                .map_err(fault)?;
        for node in &self.nodes {
            self.fill_compute_ranges(
                node,
                &mut compute_ranges,
                &native,
                Some(&mut scratch.ranges),
            )?;
            planner
                .push_accesses(
                    matches!(node.dispatch, Dispatch::Copy | Dispatch::Fill),
                    node.shared,
                    node.release_after
                        || node
                            .pipeline
                            .as_ref()
                            .is_some_and(|p| self.pipelines[p].program.release_after),
                    &scratch.ranges,
                )
                .map_err(fault)?;
            let segment = planner.last_segment().map_err(fault)?;
            if classes.as_mut().is_some_and(|c| {
                c.push(
                    segment.queue(),
                    if segment.copy {
                        None
                    } else {
                        Some(&compute_ranges)
                    },
                )
                .is_err()
            }) {
                // Optimization evidence unavailable: retain the original broad
                // destination policy. Original full-view planning still runs.
                classes = None;
            }
        }
        let plan = planner.finish().map_err(fault)?;
        let classes = classes.and_then(|c| c.finish().ok());
        let class_bytes = classes
            .as_ref()
            .map_or(Ok(0), |c| c.metadata_bytes())
            .map_err(fault)?;
        let plan_bytes = strata_schedule::plan_metadata_bytes(&plan)
            .ok_or_else(|| fault("native transfer original plan metadata overflow"))?;
        let preparation_budget = preparation_budget
            .checked_sub(plan_bytes)
            .and_then(|n| n.checked_sub(class_bytes))
            .ok_or_else(|| fault("native transfer original plan metadata capacity"))?;
        let visibility = transfer_barriers::Visibility::new(&plan.segments, preparation_budget)
            .map_err(fault)?;
        let transfer_budget = preparation_budget
            .checked_sub(visibility.metadata_bytes().map_err(fault)?)
            .ok_or_else(|| fault("native transfer visibility retained metadata capacity"))?;
        let mut transfers =
            transfer_barriers::Recorder::new(self.nodes.len(), copy_count, transfer_budget)
                .map_err(fault)?;
        for (index, node) in self.nodes.iter().enumerate() {
            let segment = plan.node_segments[index];
            let actual = &plan.segments[segment];
            let original_copy = self.compute_lane == 0
                && !node.shared
                && matches!(node.dispatch, Dispatch::Copy | Dispatch::Fill);
            if actual.copy != original_copy {
                return Err(fault("native transfer barrier original queue identity"));
            }
            let copy = if actual.copy {
                self.fill_node_ranges(node, &mut scratch.ranges, Some(&native))?;
                Some(scratch.ranges.as_slice())
            } else {
                None
            };
            transfers
                .push_scoped(
                    copy,
                    actual.signal,
                    visibility.before(segment).map_err(fault)?,
                )
                .map_err(fault)?;
        }
        let transfers = transfers.finish().map_err(fault)?;
        let compute_budget = transfer_budget
            .checked_sub(transfers.metadata_bytes().map_err(fault)?)
            .and_then(|n| n.checked_add(class_bytes)) // already charged above; Recorder owns it now
            .ok_or_else(|| fault("native compute barrier retained metadata capacity"))?;
        let mut compute = match classes {
            Some(classes) => {
                compute_barriers::Recorder::with_classes(self.nodes.len(), compute_budget, classes)
                    .or_else(|_| {
                        compute_barriers::Recorder::broad(self.nodes.len(), compute_budget)
                    })
            }
            None => compute_barriers::Recorder::broad(self.nodes.len(), compute_budget),
        }
        .map_err(fault)?;
        for (index, node) in self.nodes.iter().enumerate() {
            let segment = &plan.segments[plan.node_segments[index]];
            let ranges = if segment.copy {
                None
            } else {
                self.fill_compute_ranges(node, &mut compute_ranges, &native, None)?;
                Some(compute_ranges.as_mut_slice())
            };
            compute.push(segment.queue(), ranges).map_err(fault)?;
        }
        let compute = compute.finish().map_err(fault)?;
        let transfer_bytes = transfers
            .metadata_bytes()
            .map_err(fault)?
            .checked_add(visibility.metadata_bytes().map_err(fault)?)
            .ok_or_else(|| fault("native transfer visibility retained metadata overflow"))?;
        let scratch_bytes = scratch_bytes
            .checked_add(transfer_bytes)
            .and_then(|n| n.checked_add(compute.metadata_bytes().ok()?))
            .ok_or_else(|| fault("native transfer capture metadata overflow"))?;

        let other_metadata = signature
            .capacity()
            .checked_mul(size_of::<u64>())
            .and_then(|n| {
                n.checked_add(
                    resources
                        .capacity()
                        .checked_mul(size_of::<WgpuResource>())?,
                )
            })
            .and_then(|n| n.checked_add(owners.metadata_bytes()))
            .ok_or_else(|| fault("native queue metadata overflow"))?;
        let retained_metadata = self
            .recordings
            .iter()
            .try_fold(other_metadata, |n, r| n.checked_add(r.metadata_bytes()?))
            .ok_or_else(|| fault("native queue metadata overflow"))?;
        let capture_metadata = retained_metadata
            .checked_add(scratch_bytes)
            .ok_or_else(|| fault("native queue capture metadata overflow"))?;
        let command_budget = descriptors::METADATA_CAP
            .checked_sub(capture_metadata)
            .ok_or_else(|| fault("native queue retained metadata capacity"))?;
        let work = strata_schedule::Commands::new(
            &self.device,
            self.pool,
            self.command,
            self.copy_family,
            self.compute_queues[strata_schedule::shared_peer(self.compute_lane).map_err(fault)?]
                .family,
            plan,
            command_budget,
        )
        .map_err(fault)?;
        let metadata = capture_metadata
            .checked_add(work.metadata_bytes())
            .ok_or_else(|| fault("native queue metadata overflow"))?;
        if metadata > descriptors::METADATA_CAP {
            return Err(fault("native queue retained metadata capacity"));
        }
        self.next_recording = self
            .next_recording
            .checked_add(1)
            .ok_or_else(|| fault("native recording identity exhausted"))?;
        let recording = Recording {
            id: self.next_recording,
            family: self.family,
            key,
            ready: false,
            compute_count,
            work,
            profile,
            device: self.device.clone(),
            pool: self.pool,
            descriptors: self.descriptors,
            command: self.command,
            signature,
            _resources: resources,
            _owners: owners,
        };
        // Ownership transfers before any fallible recording/submission step.
        self.pool = vk::CommandPool::null();
        self.descriptors = vk::DescriptorPool::null();
        self.command = recording.command;
        self.recordings.push(recording);
        let boundaries = CommandBoundaries::from_queues(
            self.recordings
                .last()
                .unwrap()
                .work
                .plan
                .segments
                .iter()
                .map(|s| s.queue()),
        )
        .map_err(fault)?;
        drop(native); // prepare_last_recording may retire old Wgpu owners.
        let mut rolling = if rolling_prefill_selected(
            self.compute_lane,
            key.is_some(),
            profiling,
            self.nodes.len(),
            compute_count,
            self.nodes.iter().any(|node| {
                node.pipeline.as_ref().is_some_and(|pipeline| {
                    matches!(
                        self.pipelines[pipeline].program.name,
                        "StrataPromptExpertF16Image"
                            | "StrataPromptExpertResidentGuImage"
                            | "StrataPromptExpertResidentDownImage"
                    )
                })
            }),
        ) {
            // The complete original inventory, metadata caps, range hazards,
            // INFO epoch and recording-owned pools/resources were preflighted
            // above. Bind the ONE original flight before any prefix can run.
            let info_wait = self.prepare_last_recording(compute_count, false, false, false)?;
            Some(
                self.recordings
                    .last_mut()
                    .unwrap()
                    .work
                    .begin_rolling(
                        [
                            self.queue,
                            self.copy_queue.raw_identity(),
                            self.compute_queues
                                [strata_schedule::shared_peer(self.compute_lane).map_err(fault)?]
                            .queue,
                        ],
                        &self.copy_queue,
                        self.fences[self.compute_lane],
                        info_wait,
                    )
                    .map_err(fault)?,
            )
        } else {
            None
        };
        let native = keys.acquire()?;
        begin_native_command(&self.device, self.command, boundaries.before(0))?;
        if let Some(profile) = &self.recordings.last().unwrap().profile {
            unsafe {
                self.device.cmd_reset_query_pool(
                    self.command,
                    profile.pool,
                    0,
                    (self.nodes.len() * 2) as u32,
                );
            }
        }
        let mut segment = 0usize;
        let mut fresh_descriptors = None;
        let mut pushed_nodes = 0usize;
        let mut fresh_nodes = 0usize;
        for (index, n) in self.nodes.iter().enumerate() {
            let work = &self.recordings.last().unwrap().work;
            let next = work.plan.node_segments[index];
            if next != segment {
                end_native_command(&self.device, self.command, boundaries.after(segment))?;
                if let Some(rolling) = &mut rolling {
                    // Only already-ended command buffers enter the prefix.
                    // Recovery remains armed across every later fallible call.
                    rolling.closed(next, index, false).map_err(fault)?;
                }
                segment = next;
                self.command = work.commands[segment];
                begin_native_command(&self.device, self.command, boundaries.before(segment))?;
            }
            if index.is_multiple_of(descriptors::NODE_WINDOW) {
                // Issue an already-closed prior prefix before preparing this
                // fallible future window. Drop only host handle/index vectors.
                // Vulkan sets remain pool-owned through the original terminal fence.
                drop(fresh_descriptors.take());
                fresh_descriptors = Some(self.fresh_descriptors(
                    &self.nodes[index..(index + descriptors::NODE_WINDOW).min(self.nodes.len())],
                    self.recordings.last().unwrap().descriptors,
                    metadata,
                    &native,
                )?);
            }
            let copy = work.plan.segments[segment].copy;
            let supported_stages = if copy {
                vk::PipelineStageFlags::TRANSFER
            } else {
                vk::PipelineStageFlags::COMPUTE_SHADER
                    | vk::PipelineStageFlags::DRAW_INDIRECT
                    | vk::PipelineStageFlags::TRANSFER
            };
            scratch.barriers.clear();
            if copy {
                // Only disjoint ORIGINAL transfer views omit a dependency.
                // Any real overlap keeps the original WHOLE_SIZE barrier.
                for dependency in transfers.node(index).map_err(fault)? {
                    scratch.barriers.push(
                        vk::BufferMemoryBarrier::default()
                            .src_access_mask(dependency.before_flags())
                            .dst_access_mask(dependency.after_flags())
                            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .buffer(vk::Buffer::from_raw(dependency.buffer))
                            .offset(0)
                            .size(vk::WHOLE_SIZE),
                    );
                }
            } else {
                if !transfers.node(index).map_err(fault)?.is_empty() {
                    return Err(fault("native transfer barrier physical queue identity"));
                }
            }
            let global = if copy {
                None
            } else {
                compute_barriers::global_masks(compute.node(index).map_err(fault)?)
            };
            if !scratch.barriers.is_empty() || global.is_some() {
                let global_barrier = global.map(|(before, after)| {
                    vk::MemoryBarrier::default()
                        .src_access_mask(before)
                        .dst_access_mask(after)
                });
                unsafe {
                    self.device.cmd_pipeline_barrier(
                        self.command,
                        supported_stages,
                        supported_stages,
                        vk::DependencyFlags::empty(),
                        global_barrier.as_slice(),
                        &scratch.barriers,
                        &[],
                    );
                }
            }
            let profile = self.recordings.last().unwrap().profile.as_ref();
            if let Some(profile) = profile {
                profile.stamp(self.command, index, false);
            }
            match n.dispatch {
                Dispatch::Copy => {
                    let source = &n.resources[0];
                    let target = &n.resources[1];
                    unsafe {
                        self.device.cmd_copy_buffer(
                            self.command,
                            native.get(source)?,
                            native.get(target)?,
                            &[vk::BufferCopy {
                                src_offset: source.offset,
                                dst_offset: target.offset,
                                size: target.size,
                            }],
                        );
                    }
                    native_host_read_boundary(&self.device, self.command, target, &native)?;
                    if let Some(profile) = profile {
                        profile.stamp(self.command, index, true);
                    }
                    continue;
                }
                Dispatch::Fill => {
                    let target = &n.resources[0];
                    unsafe {
                        self.device.cmd_fill_buffer(
                            self.command,
                            native.get(target)?,
                            target.offset,
                            target.size,
                            0,
                        );
                    }
                    native_host_read_boundary(&self.device, self.command, target, &native)?;
                    if let Some(profile) = profile {
                        profile.stamp(self.command, index, true);
                    }
                    continue;
                }
                _ => {}
            }
            let p = &self.pipelines[n
                .pipeline
                .as_ref()
                .ok_or_else(|| fault("native graph compute identity"))?];
            let set = if p.push_descriptors {
                None
            } else {
                Some(
                    fresh_descriptors
                        .as_ref()
                        .ok_or_else(|| fault("native descriptor fresh window missing"))?
                        .set(index % descriptors::NODE_WINDOW)
                        .map_err(fault)?,
                )
            };
            if p.push_descriptors {
                // Re-write EVERY original descriptor, even after a new command
                // buffer or incompatible layout. Vulkan copies these views into
                // this command buffer; all original owners remain Recording-held.
                let mut infos =
                    [vk::DescriptorBufferInfo::default(); push_descriptors::MAX_BINDINGS];
                if !push_descriptors::selected(
                    true,
                    self.push_descriptors
                        .as_ref()
                        .ok_or_else(|| fault("native push authority missing"))?
                        .1,
                    n.resources.len(),
                ) || n.resources.len() != p.program.bindings.len()
                {
                    return Err(fault("native push original descriptor extent"));
                }
                for (i, r) in n.resources.iter().enumerate() {
                    if r.size == 0
                        || r.offset
                            .checked_add(r.size)
                            .is_none_or(|end| end > r.buffer.size())
                    {
                        return Err(fault("native push actual owner extent"));
                    }
                    infos[i] = vk::DescriptorBufferInfo::default()
                        .buffer(native.get(r)?)
                        .offset(r.offset)
                        .range(r.size);
                }
                let writes =
                    push_descriptors::writes(&infos[..n.resources.len()], &p.program.bindings)
                        .map_err(fault)?;
                unsafe {
                    self.push_descriptors
                        .as_ref()
                        .unwrap()
                        .0
                        .cmd_push_descriptor_set(
                            self.command,
                            vk::PipelineBindPoint::COMPUTE,
                            p.layout,
                            0,
                            &writes[..n.resources.len()],
                        );
                }
                pushed_nodes += 1;
            }
            unsafe {
                self.device.cmd_bind_pipeline(
                    self.command,
                    vk::PipelineBindPoint::COMPUTE,
                    p.pipeline,
                );
                if let Some(set) = set {
                    self.device.cmd_bind_descriptor_sets(
                        self.command,
                        vk::PipelineBindPoint::COMPUTE,
                        p.layout,
                        0,
                        &[set],
                        &[],
                    );
                    fresh_nodes += 1;
                }
                match &n.dispatch {
                    Dispatch::Direct(x, y, z) => self.device.cmd_dispatch(self.command, *x, *y, *z),
                    Dispatch::Indirect(r) => {
                        if r.offset
                            .checked_add(12)
                            .is_none_or(|end| end > r.buffer.size())
                        {
                            return Err(fault("native graph indirect extent"));
                        }
                        let raw = native.get(r)?;
                        self.device
                            .cmd_dispatch_indirect(self.command, raw, r.offset);
                    }
                    Dispatch::Copy | Dispatch::Fill => {
                        unreachable!("transfers recorded before compute")
                    }
                }
            }
            if let Some(profile) = profile {
                profile.stamp(self.command, index, true);
            }
        }
        end_native_command(&self.device, self.command, boundaries.after(segment))?;
        COPY_BARRIERS[0].fetch_add(copy_count as u64, std::sync::atomic::Ordering::Relaxed);
        COPY_BARRIERS[1].fetch_add(
            transfers.dependency_count() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        let call_counts = compute.call_counts();
        for (total, amount) in COMPUTE_BARRIERS.iter().zip([
            self.nodes.len() - copy_count,
            compute.candidates(),
            compute.count(),
            compute.proven_or_views(),
            call_counts[0],
            call_counts[1],
            call_counts[2],
            call_counts[3],
            pushed_nodes,
            fresh_nodes,
        ]) {
            total.fetch_add(amount as u64, std::sync::atomic::Ordering::Relaxed);
        }
        self.recordings.last_mut().unwrap().ready = true;
        if let Some(rolling) = &mut rolling {
            rolling
                .closed(
                    self.recordings.last().unwrap().work.plan.segments.len(),
                    self.nodes.len(),
                    true,
                )
                .map_err(fault)?;
            rolling.finish().map_err(fault)?;
        }
        drop(native); // Before node drops, Wgpu completion, cleanup or erasure.
        drop(keys);
        CAPTURE_NS.fetch_add(
            capture_start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        if let Some(rolling) = rolling {
            let id = self.recordings.last().unwrap().id;
            let [windows, segments] = rolling.submitted_counts();
            self.mark_last_submitted(id)?;
            rolling.accept().map_err(fault)?;
            for (total, amount) in ROLLING_TOTALS.iter().zip([1, windows, segments]) {
                total.fetch_add(amount, std::sync::atomic::Ordering::Relaxed);
            }
        } else {
            self.submit_last_recording(compute_count, false, profiling)?;
        }
        if wait {
            self.complete_scoped_submission()?;
        }
        Ok(())
    }
}
impl Drop for Graph {
    fn drop(&mut self) {
        unsafe {
            if self.complete_submission().is_err() {
                // Unknown partial work owns its original command/resource
                // inventories until protected-process/device teardown.
                std::mem::forget(std::mem::take(&mut self.nodes));
                std::mem::forget(std::mem::take(&mut self.recordings));
                std::mem::forget(std::mem::take(&mut self.initialized));
                std::mem::forget(self.draft_source.take());
                std::mem::forget(self.draft_readiness.take());
                return;
            }
            self.nodes.clear();
            self.recordings.clear();
            for (_, p) in self.pipelines.drain() {
                self.device.destroy_pipeline(p.pipeline, None);
                self.device.destroy_pipeline_layout(p.layout, None);
                self.device
                    .destroy_descriptor_set_layout(p.set_layout, None);
                self.device.destroy_shader_module(p.shader, None);
            }
            for fence in self.fences {
                self.device.destroy_fence(fence, None);
            }
            self.device.destroy_descriptor_pool(self.descriptors, None);
            self.device.destroy_command_pool(self.pool, None);
        }
    }
}

impl std::fmt::Debug for Graph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeStrataGraph")
            .field("pending_nodes", &self.nodes.len())
            .field("pipelines", &self.pipelines.len())
            .finish()
    }
}
