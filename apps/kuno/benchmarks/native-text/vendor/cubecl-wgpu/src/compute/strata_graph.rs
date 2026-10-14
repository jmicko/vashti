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
    words: Vec<u32>,
    bindings: Vec<vk::DescriptorType>,
    accesses: Vec<vk::AccessFlags>,
    entry: CString,
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
        words,
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
    pipeline: wgpu::ComputePipeline,
    resources: Vec<WgpuResource>,
    dispatch: Dispatch,
}
#[derive(Debug)]
pub(crate) enum Dispatch {
    Direct(u32, u32, u32),
    Indirect(WgpuResource),
}
pub(crate) struct Graph {
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
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader)
                .name(&p.entry)
                .flags(vk::PipelineShaderStageCreateFlags::ALLOW_VARYING_SUBGROUP_SIZE);
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
        for r in &resources {
            if r.size == 0
                || r.offset
                    .checked_add(r.size)
                    .is_none_or(|end| end > r.buffer.size())
            {
                return Err(fault("native graph actual buffer extent"));
            }
        }
        self.nodes.push(Node {
            pipeline: pipeline.clone(),
            resources,
            dispatch,
        });
        Ok(())
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
        // The caller drains Wgpu parameter/upload commands BEFORE entering.
        // Every owning buffer remains retained until our actual completion fence.
        unsafe { self.device.reset_fences(&[self.fence]) }
            .map_err(|_| fault("native graph reset fence"))?;
        let mut signature = Vec::new();
        for n in &self.nodes {
            signature.push(self.pipelines[&n.pipeline].pipeline.as_raw());
            signature.push(n.resources.len() as u64);
            for r in &n.resources {
                signature.extend([raw_buffer(r)?.as_raw(), r.offset, r.size]);
            }
            match &n.dispatch {
                Dispatch::Direct(x, y, z) => signature.extend([0, *x as u64, *y as u64, *z as u64]),
                Dispatch::Indirect(r) => {
                    signature.extend([1, raw_buffer(r)?.as_raw(), r.offset, r.size])
                }
            }
        }
        if let Some(hit) = self
            .recordings
            .iter()
            .position(|r| r.signature == signature)
        {
            let command = self.recordings[hit].command;
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
            unsafe {
                self.device
                    .wait_for_fences(&[self.fence], true, 120_000_000_000)
            }
            .map_err(|_| fault("native graph replay completion"))?;
            self.submitted = false;
            self.submissions += 1;
            self.replays += 1;
            self.dispatches += self.nodes.len() as u64;
            credit(self.nodes.len(), true);
            self.nodes.clear();
            return Ok(());
        }
        // All earlier native work is complete. Cached command buffers contain
        // addresses, not tensor contents, and are bounded to four exact plans.
        if self.recordings.len() == 4 {
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
        let recording = Recording {
            device: self.device.clone(),
            pool: self.pool,
            descriptors: self.descriptors,
            command: self.command,
            signature,
            _resources: self
                .nodes
                .iter()
                .flat_map(|n| {
                    n.resources.iter().cloned().chain(match &n.dispatch {
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
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
        // One boundary covers Wgpu upload/initialization writes. Within the
        // graph, only real entrypoint accesses to the same buffer create edges.
        unsafe {
            self.device.cmd_pipeline_barrier(
                self.command,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
        let mut previous = HashMap::<vk::Buffer, vk::AccessFlags>::new();
        for n in &self.nodes {
            let p = &self.pipelines[&n.pipeline];
            let mut accesses = HashMap::<vk::Buffer, vk::AccessFlags>::new();
            for (r, access) in n.resources.iter().zip(&p.program.accesses) {
                if !access.is_empty() {
                    *accesses.entry(raw_buffer(r)?).or_default() |= *access;
                }
            }
            if let Dispatch::Indirect(r) = &n.dispatch {
                *accesses.entry(raw_buffer(r)?).or_default() |=
                    vk::AccessFlags::INDIRECT_COMMAND_READ;
            }
            let mut dependencies = Vec::new();
            for (buffer, access) in accesses {
                if let Some(before) = previous.get(&buffer).copied() {
                    if before.contains(vk::AccessFlags::SHADER_WRITE)
                        || access.contains(vk::AccessFlags::SHADER_WRITE)
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
                            | vk::PipelineStageFlags::DRAW_INDIRECT,
                        vk::PipelineStageFlags::COMPUTE_SHADER
                            | vk::PipelineStageFlags::DRAW_INDIRECT,
                        vk::DependencyFlags::empty(),
                        &[],
                        &dependencies,
                        &[],
                    );
                }
            }
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
                }
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
        unsafe {
            self.device
                .wait_for_fences(&[self.fence], true, 120_000_000_000)
        }
        .map_err(|_| fault("native graph completion"))?;
        self.submitted = false;
        self.submissions += 1;
        self.dispatches += self.nodes.len() as u64;
        credit(self.nodes.len(), false);
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
