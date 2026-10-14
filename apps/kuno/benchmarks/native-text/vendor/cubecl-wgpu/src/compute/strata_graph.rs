//! Rust Vulkan execution for the coherent Intel Strata port. MIT upstream
//! fb58e0dbc8399662c0e47c76578c6e878b14f6cf records a whole verification graph.
//! This backend owns native compute pipelines/descriptor sets/command buffers;
//! Wgpu retains all buffer allocation, upload, readback and privacy ledgers.
//! Only serialized fixed-public laboratory streams may select this backend.
use super::{WgpuResource, transient_io::fault};
use ash::{vk, vk::Handle as _};
use cubecl_core::server::ServerError;
use std::{
    collections::{HashMap, HashSet},
    ffi::CString,
    sync::{Arc, Mutex, OnceLock},
};
use wgpu::naga;

static PROFILE_ACTIVE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub fn set_native_strata_profile_active(active: bool) {
    PROFILE_ACTIVE.store(active, std::sync::atomic::Ordering::Release);
}
fn profile_active() -> bool {
    enabled()
        && PROFILE_ACTIVE.load(std::sync::atomic::Ordering::Acquire)
        && std::env::var("KUNO_STRATA_NATIVE_PROFILE").as_deref() == Ok("round-8")
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
pub fn native_strata_graph_execution_stats() -> ([u64; 5], [u64; 5], u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (
        std::array::from_fn(|i| SHAPES[i].load(Relaxed)),
        std::array::from_fn(|i| REPLAY_SHAPES[i].load(Relaxed)),
        CAPTURE_NS.load(Relaxed),
        WAIT_NS.load(Relaxed),
    )
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
    *ON.get_or_init(|| std::env::var("KUNO_STRATA_INTEL_CORE").as_deref() == Ok("public"))
}
#[derive(Debug)]
pub(crate) struct Program {
    name: &'static str,
    words: Vec<u32>,
    bindings: Vec<vk::DescriptorType>,
    accesses: Vec<vk::AccessFlags>,
    entry: CString,
    expert_subgroup_32: bool,
    physical_reads: bool,
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
    if source.contains("STRATA_PHYSICAL_EXPERTS") {
        words = super::strata_physical::native_expert_shader_words(source)?;
    }
    let p = Arc::new(Program {
        name: "UnlabelledNativeOperator",
        words,
        expert_subgroup_32: source.contains("STRATA_PHYSICAL_EXPERTS")
            || source.contains("STRATA_DENSE_PACKED"),
        physical_reads: source.contains("STRATA_PHYSICAL_EXPERTS"),
        accesses: bindings.iter().map(|b| b.2).collect(),
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
pub(crate) fn label_module(
    module: &wgpu::ShaderModule,
    name: &'static str,
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
}
#[derive(Debug)]
struct Node {
    pipeline: Option<wgpu::ComputePipeline>,
    resources: Vec<WgpuResource>,
    read_dependencies: Vec<WgpuResource>,
    dispatch: Dispatch,
}
#[derive(Debug)]
pub(crate) enum Dispatch {
    Direct(u32, u32, u32),
    Indirect(WgpuResource),
    Copy,
    Fill,
}
pub(crate) struct Graph {
    timestamp_period: f64,
    timestamp_bits: u32,
    device: ash::Device,
    family: u32,
    queue: vk::Queue,
    pool: vk::CommandPool,
    descriptors: vk::DescriptorPool,
    command: vk::CommandBuffer,
    fence: vk::Fence,
    pipelines: HashMap<wgpu::ComputePipeline, Pipeline>,
    nodes: Vec<Node>,
    submitted: bool,
    pub submissions: u64,
    pub dispatches: u64,
    recordings: Vec<Recording>,
    pub replays: u64,
    initialized: HashSet<wgpu::Buffer>,
}
struct Recording {
    profile: Option<OperatorProfile>,
    device: ash::Device,
    pool: vk::CommandPool,
    descriptors: vk::DescriptorPool,
    command: vk::CommandBuffer,
    signature: Vec<u64>,
    _resources: Vec<WgpuResource>,
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
        let timestamp_period = props.properties.limits.timestamp_period as f64;
        let timestamp_bits = unsafe {
            h.shared_instance()
                .raw_instance()
                .get_physical_device_queue_family_properties(h.raw_physical_device())
        }
        .get(h.queue_family_index() as usize)
        .map_or(0, |q| q.timestamp_valid_bits);
        if subgroup.min_subgroup_size > 32
            || subgroup.max_subgroup_size < 32
            || !subgroup
                .required_subgroup_size_stages
                .contains(vk::ShaderStageFlags::COMPUTE)
        {
            return Err(fault("native Strata requires Intel expert subgroup32"));
        }
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
        let sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 262144,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 32768,
            },
        ];
        let descriptors = match unsafe {
            d.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(32768)
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
        Ok(Self {
            timestamp_period,
            timestamp_bits,
            device: d,
            family: h.queue_family_index(),
            queue,
            pool,
            descriptors,
            command,
            fence,
            pipelines: HashMap::new(),
            nodes: Vec::new(),
            submitted: false,
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
            let set_layout = match unsafe {
                self.device.create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
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
            let mut required = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
                .required_subgroup_size(32);
            let mut stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader)
                .name(&p.entry);
            // The pinned SYCL expert dispatch uses reqd_sub_group_size(32),
            // with four independent eight-lane rows inside that subgroup.
            if p.expert_subgroup_32 {
                stage = stage.push_next(&mut required);
            } else {
                stage =
                    stage.flags(vk::PipelineShaderStageCreateFlags::ALLOW_VARYING_SUBGROUP_SIZE);
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
        self.nodes.push(Node {
            pipeline: Some(pipeline.clone()),
            resources,
            read_dependencies,
            dispatch,
        });
        Ok(())
    }
    /// Upload staging remains tracked by TransientIo. Transfers execute at
    /// their actual position between kernels, not ahead of earlier readers.
    pub(crate) fn push_transfer(
        &mut self,
        source: Option<WgpuResource>,
        target: WgpuResource,
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
        self.nodes.push(Node {
            pipeline: None,
            resources,
            read_dependencies: Vec::new(),
            dispatch,
        });
        Ok(())
    }
    fn node_accesses(&self, n: &Node) -> Result<HashMap<vk::Buffer, vk::AccessFlags>, ServerError> {
        let mut accesses = HashMap::new();
        match n.dispatch {
            Dispatch::Copy => {
                *accesses.entry(raw_buffer(&n.resources[0])?).or_default() |=
                    vk::AccessFlags::TRANSFER_READ;
                *accesses.entry(raw_buffer(&n.resources[1])?).or_default() |=
                    vk::AccessFlags::TRANSFER_WRITE;
            }
            Dispatch::Fill => {
                accesses.insert(
                    raw_buffer(&n.resources[0])?,
                    vk::AccessFlags::TRANSFER_WRITE,
                );
            }
            _ => {
                let p = &self.pipelines[n
                    .pipeline
                    .as_ref()
                    .ok_or_else(|| fault("native graph compute identity"))?];
                for (r, access) in n.resources.iter().zip(&p.program.accesses) {
                    if !access.is_empty() {
                        *accesses.entry(raw_buffer(r)?).or_default() |= *access;
                    }
                }
                for r in &n.read_dependencies {
                    *accesses.entry(raw_buffer(r)?).or_default() |= vk::AccessFlags::SHADER_READ;
                }
                if let Dispatch::Indirect(r) = &n.dispatch {
                    *accesses.entry(raw_buffer(r)?).or_default() |=
                        vk::AccessFlags::INDIRECT_COMMAND_READ;
                }
            }
        }
        Ok(accesses)
    }
    pub(crate) fn clear_completed_recordings(&mut self) -> Result<(), ServerError> {
        if self.submitted || self.pending() {
            return Err(fault("native graph cleanup while pending"));
        }
        self.recordings.clear();
        self.initialized.clear();
        Ok(())
    }
    pub(crate) fn pending(&self) -> bool {
        !self.nodes.is_empty()
    }
    pub(crate) fn execute(&mut self) -> Result<(), ServerError> {
        if self.nodes.is_empty() {
            return Ok(());
        }
        let capture_start = std::time::Instant::now();
        // The caller drains Wgpu parameter/upload commands BEFORE entering.
        // Every owning buffer remains retained until our actual completion fence.
        unsafe { self.device.reset_fences(&[self.fence]) }
            .map_err(|_| fault("native graph reset fence"))?;
        let mut signature = Vec::new();
        for n in &self.nodes {
            signature.push(
                n.pipeline
                    .as_ref()
                    .map_or(0, |p| self.pipelines[p].pipeline.as_raw()),
            );
            signature.push(n.resources.len() as u64);
            for r in &n.resources {
                signature.extend([raw_buffer(r)?.as_raw(), r.offset, r.size]);
            }
            signature.push(n.read_dependencies.len() as u64);
            for r in &n.read_dependencies {
                signature.extend([raw_buffer(r)?.as_raw(), r.offset, r.size]);
            }
            match &n.dispatch {
                Dispatch::Direct(x, y, z) => signature.extend([0, *x as u64, *y as u64, *z as u64]),
                Dispatch::Indirect(r) => {
                    signature.extend([1, raw_buffer(r)?.as_raw(), r.offset, r.size])
                }
                Dispatch::Copy => signature.extend([2, 0, 0, 0]),
                Dispatch::Fill => signature.extend([3, 0, 0, 0]),
            }
        }
        let compute_count = self.nodes.iter().filter(|n| n.pipeline.is_some()).count();
        let profiling = profile_active() && compute_count >= 2048;
        if let Some(hit) = self
            .recordings
            .iter()
            .position(|r| r.signature == signature)
            .filter(|_| !profiling)
        {
            let recording = self.recordings.remove(hit);
            let command = recording.command;
            self.recordings.push(recording);
            let commands = [command];
            unsafe {
                self.device.queue_submit(
                    self.queue,
                    &[vk::SubmitInfo::default().command_buffers(&commands)],
                    self.fence,
                )
            }
            .map_err(|_| fault("native graph replay submit"))?;
            self.submitted = true;
            let wait_start = std::time::Instant::now();
            unsafe {
                self.device
                    .wait_for_fences(&[self.fence], true, 120_000_000_000)
            }
            .map_err(|_| fault("native graph replay completion"))?;
            WAIT_NS.fetch_add(
                wait_start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            self.submitted = false;
            self.submissions += 1;
            self.replays += 1;
            self.dispatches += compute_count as u64;
            credit(compute_count, true);
            self.nodes.clear();
            return Ok(());
        }
        // All earlier native work is complete. Cached command buffers contain
        // addresses, not tensor contents, and are bounded to sixteen exact plans.
        if self.recordings.len() == 16 {
            self.recordings.remove(0);
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
            let storage = self
                .nodes
                .iter()
                .map(|n| n.resources.len() as u32)
                .sum::<u32>();
            let sizes = [
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: storage.max(1),
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: self.nodes.len().max(1) as u32,
                },
            ];
            self.descriptors = unsafe {
                self.device.create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(self.nodes.len().max(1) as u32)
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
        let profile = if profiling {
            if self.nodes.len() > 16384
                || self.timestamp_bits == 0
                || self.timestamp_bits > 64
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
                mask: if self.timestamp_bits == 64 {
                    u64::MAX
                } else {
                    (1u64 << self.timestamp_bits) - 1
                },
            })
        } else {
            None
        };
        let recording = Recording {
            profile,
            device: self.device.clone(),
            pool: self.pool,
            descriptors: self.descriptors,
            command: self.command,
            signature,
            _resources: self
                .nodes
                .iter()
                .flat_map(|n| {
                    n.resources
                        .iter()
                        .chain(&n.read_dependencies)
                        .cloned()
                        .chain(match &n.dispatch {
                            Dispatch::Indirect(r) => Some(r.clone()),
                            _ => None,
                        })
                })
                .collect(),
        };
        // Ownership transfers before any fallible recording/submission step.
        self.pool = vk::CommandPool::null();
        self.descriptors = vk::DescriptorPool::null();
        self.command = recording.command;
        self.recordings.push(recording);
        unsafe {
            self.device
                .begin_command_buffer(self.command, &vk::CommandBufferBeginInfo::default())
        }
        .map_err(|_| fault("native graph record"))?;
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
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
        // One boundary covers Wgpu upload/initialization writes. Within the
        // graph, only real entrypoint accesses to the same buffer create edges.
        unsafe {
            self.device.cmd_pipeline_barrier(
                self.command,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::COMPUTE_SHADER
                    | vk::PipelineStageFlags::DRAW_INDIRECT
                    | vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
        let mut previous = HashMap::<vk::Buffer, vk::AccessFlags>::new();
        for (index, n) in self.nodes.iter().enumerate() {
            let accesses = self.node_accesses(n)?;
            let mut dependencies = Vec::new();
            for (buffer, access) in accesses {
                if let Some(before) = previous.get(&buffer).copied() {
                    if before
                        .intersects(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
                        || access.intersects(
                            vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE,
                        )
                    {
                        dependencies.push(
                            vk::BufferMemoryBarrier::default()
                                .src_access_mask(before)
                                .dst_access_mask(access)
                                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                                .buffer(buffer)
                                .offset(0)
                                .size(vk::WHOLE_SIZE),
                        );
                        previous.insert(buffer, access);
                    } else {
                        previous.insert(buffer, before | access);
                    }
                } else {
                    previous.insert(buffer, access);
                }
            }
            if !dependencies.is_empty() {
                unsafe {
                    self.device.cmd_pipeline_barrier(
                        self.command,
                        vk::PipelineStageFlags::COMPUTE_SHADER
                            | vk::PipelineStageFlags::DRAW_INDIRECT
                            | vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::COMPUTE_SHADER
                            | vk::PipelineStageFlags::DRAW_INDIRECT
                            | vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &dependencies,
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
                            raw_buffer(source)?,
                            raw_buffer(target)?,
                            &[vk::BufferCopy {
                                src_offset: source.offset,
                                dst_offset: target.offset,
                                size: target.size,
                            }],
                        );
                    }
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
                            raw_buffer(target)?,
                            target.offset,
                            target.size,
                            0,
                        );
                    }
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
            let layouts = [p.set_layout];
            let set = unsafe {
                self.device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(self.recordings.last().unwrap().descriptors)
                        .set_layouts(&layouts),
                )
            }
            .map_err(|_| fault("native graph descriptor allocation"))?[0];
            let mut infos = Vec::with_capacity(n.resources.len());
            for r in &n.resources {
                let h = unsafe { r.buffer.as_hal::<wgpu::hal::api::Vulkan>() }
                    .ok_or_else(|| fault("native graph non-Vulkan buffer"))?;
                infos.push(
                    vk::DescriptorBufferInfo::default()
                        .buffer(unsafe { h.raw_buffer_handle() })
                        .offset(r.offset)
                        .range(r.size),
                );
            }
            let writes = infos
                .iter()
                .enumerate()
                .map(|(i, b)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(i as u32)
                        .descriptor_type(p.program.bindings[i])
                        .buffer_info(std::slice::from_ref(b))
                })
                .collect::<Vec<_>>();
            unsafe {
                self.device.update_descriptor_sets(&writes, &[]);
                self.device.cmd_bind_pipeline(
                    self.command,
                    vk::PipelineBindPoint::COMPUTE,
                    p.pipeline,
                );
                self.device.cmd_bind_descriptor_sets(
                    self.command,
                    vk::PipelineBindPoint::COMPUTE,
                    p.layout,
                    0,
                    &[set],
                    &[],
                );
                match &n.dispatch {
                    Dispatch::Direct(x, y, z) => self.device.cmd_dispatch(self.command, *x, *y, *z),
                    Dispatch::Indirect(r) => {
                        if r.offset
                            .checked_add(12)
                            .is_none_or(|end| end > r.buffer.size())
                        {
                            return Err(fault("native graph indirect extent"));
                        }
                        let h = r
                            .buffer
                            .as_hal::<wgpu::hal::api::Vulkan>()
                            .ok_or_else(|| fault("native graph indirect buffer"))?;
                        self.device.cmd_dispatch_indirect(
                            self.command,
                            h.raw_buffer_handle(),
                            r.offset,
                        );
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
        unsafe {
            self.device.cmd_pipeline_barrier(
                self.command,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
            self.device.end_command_buffer(self.command)
        }
        .map_err(|_| fault("native graph end record"))?;
        CAPTURE_NS.fetch_add(
            capture_start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        let commands = [self.command];
        unsafe {
            self.device.queue_submit(
                self.queue,
                &[vk::SubmitInfo::default().command_buffers(&commands)],
                self.fence,
            )
        }
        .map_err(|_| fault("native graph submit"))?;
        self.submitted = true;
        // Protected public lab only. Bound to prevent a wedged device leaving
        // native descriptor/buffer lifetimes unowned. Never reset the GPU.
        let wait_start = std::time::Instant::now();
        unsafe {
            self.device
                .wait_for_fences(&[self.fence], true, 120_000_000_000)
        }
        .map_err(|_| fault("native graph completion"))?;
        WAIT_NS.fetch_add(
            wait_start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        self.submitted = false;
        if profiling {
            // Completed sampled command is never cached or replayed during an
            // unprofiled round. Failed in-flight pools stay with Recording and
            // are leaked with its other owners at terminal device teardown.
            let recording = self.recordings.pop().unwrap();
            recording.profile.as_ref().unwrap().collect()?;
        }
        self.submissions += 1;
        self.dispatches += compute_count as u64;
        credit(compute_count, false);
        self.nodes.clear();
        Ok(())
    }
}
impl Drop for Graph {
    fn drop(&mut self) {
        unsafe {
            if self.submitted
                && self
                    .device
                    .wait_for_fences(&[self.fence], true, 120_000_000_000)
                    .is_err()
            {
                // Do not free resources referenced by unknown in-flight execution.
                // Protected lab process teardown is the terminal device boundary.
                let nodes = std::mem::take(&mut self.nodes);
                std::mem::forget(nodes);
                let recordings = std::mem::take(&mut self.recordings);
                std::mem::forget(recordings);
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
            self.device.destroy_fence(self.fence, None);
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
