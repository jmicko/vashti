//! Vulkan compute for the Rust engine: one device, storage buffers, compute kernels and
//! batched dispatch. Kernels are GLSL in `shaders/`, checked in as SPIR-V next to their
//! source (see `shaders::tests`).
pub mod shaders;

use std::{ffi::CStr, sync::Arc};

use ash::vk;

/// Which adapter to open.
pub enum Select {
    /// The first discrete adapter of a PCI vendor (0x8086 Intel, 0x10de NVIDIA).
    Vendor(u32),
    /// The adapter at a PCI address such as `0000:04:00.0`.
    Pci(String),
}
struct Core {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    pool: vk::CommandPool,
    descriptors: vk::DescriptorPool,
    memory: vk::PhysicalDeviceMemoryProperties,
    name: String,
    /// Command and descriptor pools and the queue are used by one thread at a time.
    turn: std::sync::Mutex<()>,
}
impl Drop for Core {
    fn drop(&mut self) {
        // SAFETY: every object that borrows the device holds an `Arc<Core>`, so nothing
        // created from these handles is still alive.
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_descriptor_pool(self.descriptors, None);
            self.device.destroy_command_pool(self.pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
#[derive(Clone)]
pub struct Gpu {
    core: Arc<Core>,
}
pub struct Buffer {
    core: Arc<Core>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    bytes: u64,
}
impl Drop for Buffer {
    fn drop(&mut self) {
        // SAFETY: the buffer is unused once its last user dropped; passes wait before return.
        unsafe {
            self.core.device.destroy_buffer(self.buffer, None);
            self.core.device.free_memory(self.memory, None);
        }
    }
}
/// Host-visible memory kept mapped for reading results back.
pub struct Readback {
    stage: Buffer,
    at: *mut u8,
}
// SAFETY: the pointer is the buffer's own mapping, valid until the buffer is dropped, and
// only read through `Gpu::read_floats`, which holds the device turn while copying into it.
unsafe impl Send for Readback {}
unsafe impl Sync for Readback {}
impl Buffer {
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}
pub struct Kernel {
    core: Arc<Core>,
    module: vk::ShaderModule,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    bindings: usize,
    constants: usize,
}
impl Drop for Kernel {
    fn drop(&mut self) {
        // SAFETY: as for `Buffer`.
        unsafe {
            let d = &self.core.device;
            d.destroy_pipeline(self.pipeline, None);
            d.destroy_pipeline_layout(self.layout, None);
            d.destroy_descriptor_set_layout(self.set_layout, None);
            d.destroy_shader_module(self.module, None);
        }
    }
}
fn check<T>(result: Result<T, vk::Result>, code: &'static str) -> Result<T, &'static str> {
    result.map_err(|_| code)
}
impl Gpu {
    pub fn open(select: Select) -> Result<Self, &'static str> {
        // SAFETY: plain Vulkan setup; every handle is destroyed in `Core::drop`, and the
        // feature structures outlive the calls that read them.
        unsafe {
            let entry = ash::Entry::load().map_err(|_| "gpu_unavailable")?;
            let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
            let instance = check(
                entry.create_instance(
                    &vk::InstanceCreateInfo::default().application_info(&app),
                    None,
                ),
                "gpu_unavailable",
            )?;
            let physical = check(instance.enumerate_physical_devices(), "gpu_unavailable")?
                .into_iter()
                .find(|device| match &select {
                    Select::Vendor(vendor) => {
                        let p = instance.get_physical_device_properties(*device);
                        p.vendor_id == *vendor
                            && p.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
                    }
                    Select::Pci(address) => {
                        let mut bus = vk::PhysicalDevicePCIBusInfoPropertiesEXT::default();
                        let mut properties =
                            vk::PhysicalDeviceProperties2::default().push_next(&mut bus);
                        instance.get_physical_device_properties2(*device, &mut properties);
                        // Software adapters leave the structure zeroed; they are never wanted.
                        properties.properties.device_type != vk::PhysicalDeviceType::CPU
                            && format!(
                                "{:04x}:{:02x}:{:02x}.{:x}",
                                bus.pci_domain, bus.pci_bus, bus.pci_device, bus.pci_function
                            ) == address.to_ascii_lowercase()
                    }
                });
            let Some(physical) = physical else {
                instance.destroy_instance(None);
                return Err("gpu_unavailable");
            };
            let properties = instance.get_physical_device_properties(physical);
            let name = CStr::from_bytes_until_nul(std::slice::from_raw_parts(
                properties.device_name.as_ptr().cast::<u8>(),
                properties.device_name.len(),
            ))
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
            let family = instance
                .get_physical_device_queue_family_properties(physical)
                .iter()
                .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .ok_or("gpu_unavailable")? as u32;
            let queues = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(family)
                .queue_priorities(&[1.0])];
            let extensions = [c"VK_KHR_cooperative_matrix".as_ptr()];
            let mut v11 = vk::PhysicalDeviceVulkan11Features::default()
                .storage_buffer16_bit_access(true)
                .uniform_and_storage_buffer16_bit_access(true);
            let mut v12 = vk::PhysicalDeviceVulkan12Features::default()
                .shader_float16(true)
                .shader_int8(true)
                .vulkan_memory_model(true);
            let mut v13 = vk::PhysicalDeviceVulkan13Features::default()
                .subgroup_size_control(true)
                .compute_full_subgroups(true);
            let mut matrix =
                vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(true);
            let features = vk::PhysicalDeviceFeatures::default().shader_float64(true);
            let created = instance.create_device(
                physical,
                &vk::DeviceCreateInfo::default()
                    .queue_create_infos(&queues)
                    .enabled_extension_names(&extensions)
                    .enabled_features(&features)
                    .push_next(&mut v11)
                    .push_next(&mut v12)
                    .push_next(&mut v13)
                    .push_next(&mut matrix),
                None,
            );
            let device = match created {
                Ok(device) => device,
                Err(_) => {
                    instance.destroy_instance(None);
                    return Err("gpu_unsupported");
                }
            };
            let queue = device.get_device_queue(family, 0);
            let pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(family)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .map_err(|_| "gpu_unavailable")?;
            let sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1 << 16)];
            let descriptors = device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(1 << 13)
                        .pool_sizes(&sizes)
                        .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET),
                    None,
                )
                .map_err(|_| "gpu_unavailable")?;
            let memory = instance.get_physical_device_memory_properties(physical);
            Ok(Self {
                core: Arc::new(Core {
                    _entry: entry,
                    instance,
                    device,
                    queue,
                    pool,
                    descriptors,
                    memory,
                    name,
                    turn: std::sync::Mutex::new(()),
                }),
            })
        }
    }
    fn allocate(
        &self,
        bytes: u64,
        usage: vk::BufferUsageFlags,
        wanted: vk::MemoryPropertyFlags,
    ) -> Result<Buffer, &'static str> {
        let core = &self.core;
        // SAFETY: the buffer and its memory are owned by the returned `Buffer`.
        unsafe {
            let buffer = check(
                core.device.create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(bytes.max(4))
                        .usage(usage),
                    None,
                ),
                "gpu_memory",
            )?;
            let need = core.device.get_buffer_memory_requirements(buffer);
            let index = (0..core.memory.memory_type_count).find(|i| {
                need.memory_type_bits & (1 << i) != 0
                    && core.memory.memory_types[*i as usize]
                        .property_flags
                        .contains(wanted)
            });
            let memory = index.and_then(|index| {
                core.device
                    .allocate_memory(
                        &vk::MemoryAllocateInfo::default()
                            .allocation_size(need.size)
                            .memory_type_index(index),
                        None,
                    )
                    .ok()
            });
            let Some(memory) = memory else {
                core.device.destroy_buffer(buffer, None);
                return Err("gpu_memory");
            };
            let out = Buffer {
                core: core.clone(),
                buffer,
                memory,
                bytes,
            };
            check(
                core.device.bind_buffer_memory(buffer, memory, 0),
                "gpu_memory",
            )?;
            Ok(out)
        }
    }
    /// A zero-filled storage buffer in device memory.
    pub fn buffer(&self, bytes: u64) -> Result<Buffer, &'static str> {
        let buffer = self.allocate(
            bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        self.run(|device, commands| unsafe {
            // SAFETY: the fill covers the whole buffer just created.
            device.cmd_fill_buffer(commands, buffer.buffer, 0, vk::WHOLE_SIZE, 0);
        })?;
        Ok(buffer)
    }
    /// A storage buffer whose contents `fill` writes straight into mapped staging memory
    /// (zeroed first is not guaranteed: `fill` must write every byte).
    pub fn filled(
        &self,
        bytes: u64,
        fill: impl FnOnce(&mut [u8]) -> Result<(), &'static str>,
    ) -> Result<Buffer, &'static str> {
        let buffer = self.allocate(
            bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        if bytes == 0 {
            return Ok(buffer);
        }
        let stage = self.staging(bytes)?;
        // SAFETY: the mapping covers `bytes` bytes and is unmapped before the copy.
        unsafe {
            let device = &self.core.device;
            let at = check(
                device.map_memory(stage.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()),
                "gpu_memory",
            )?;
            let filled = fill(std::slice::from_raw_parts_mut(
                at.cast::<u8>(),
                bytes as usize,
            ));
            device.unmap_memory(stage.memory);
            filled?;
        }
        let region = [vk::BufferCopy::default().size(bytes)];
        self.run(|device, commands| unsafe {
            // SAFETY: both buffers hold `bytes` bytes.
            device.cmd_copy_buffer(commands, stage.buffer, buffer.buffer, &region);
        })?;
        Ok(buffer)
    }
    /// A reusable host-side landing area for reading results back.
    pub fn readback(&self, bytes: u64) -> Result<Readback, &'static str> {
        let stage = self.staging(bytes)?;
        // SAFETY: the mapping lives as long as the staging buffer, which `Readback` owns.
        let at = unsafe {
            check(
                self.core.device.map_memory(
                    stage.memory,
                    0,
                    vk::WHOLE_SIZE,
                    vk::MemoryMapFlags::empty(),
                ),
                "gpu_memory",
            )?
        };
        Ok(Readback {
            stage,
            at: at.cast::<u8>(),
        })
    }
    /// Copies `to.len()` f32 values starting at byte `offset` of `from` into `to`.
    pub fn read_floats(
        &self,
        from: &Buffer,
        offset: u64,
        through: &Readback,
        to: &mut [f32],
    ) -> Result<(), &'static str> {
        let bytes = std::mem::size_of_val(to) as u64;
        if offset + bytes > from.bytes || bytes > through.stage.bytes {
            return Err("gpu_range");
        }
        if bytes == 0 {
            return Ok(());
        }
        let region = [vk::BufferCopy::default().src_offset(offset).size(bytes)];
        self.run(|device, commands| unsafe {
            // SAFETY: the range was checked against both buffers above.
            device.cmd_copy_buffer(commands, from.buffer, through.stage.buffer, &region);
        })?;
        // SAFETY: the mapping holds at least `bytes` bytes; f32 has no invalid bit patterns.
        unsafe {
            std::ptr::copy_nonoverlapping(through.at, to.as_mut_ptr().cast::<u8>(), bytes as usize);
        }
        Ok(())
    }
    fn staging(&self, bytes: u64) -> Result<Buffer, &'static str> {
        self.allocate(
            bytes,
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
    }
    /// Records commands with `record`, submits them and waits for completion.
    fn run(
        &self,
        record: impl FnOnce(&ash::Device, vk::CommandBuffer),
    ) -> Result<(), &'static str> {
        let _turn = self.core.turn.lock().map_err(|_| "gpu_failed")?;
        self.run_locked(record)
    }
    /// `run` for callers that already hold the turn.
    fn run_locked(
        &self,
        record: impl FnOnce(&ash::Device, vk::CommandBuffer),
    ) -> Result<(), &'static str> {
        let core = &self.core;
        // SAFETY: the command buffer is recorded, submitted and freed here, and the wait
        // keeps every referenced object alive until the device is done with it.
        unsafe {
            let commands = check(
                core.device.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(core.pool)
                        .command_buffer_count(1),
                ),
                "gpu_failed",
            )?;
            let result = (|| {
                core.device.begin_command_buffer(
                    commands[0],
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )?;
                record(&core.device, commands[0]);
                core.device.end_command_buffer(commands[0])?;
                let fence = core
                    .device
                    .create_fence(&vk::FenceCreateInfo::default(), None)?;
                let submitted = core.device.queue_submit(
                    core.queue,
                    &[vk::SubmitInfo::default().command_buffers(&commands)],
                    fence,
                );
                let waited =
                    submitted.and_then(|()| core.device.wait_for_fences(&[fence], true, u64::MAX));
                if waited.is_err() {
                    let _ = core.device.device_wait_idle();
                }
                core.device.destroy_fence(fence, None);
                waited
            })();
            core.device.free_command_buffers(core.pool, &commands);
            check(result, "gpu_failed")
        }
    }
    /// Writes `data` at byte `offset` of `to`.
    pub fn upload(&self, to: &Buffer, offset: u64, data: &[u8]) -> Result<(), &'static str> {
        if offset + data.len() as u64 > to.bytes {
            return Err("gpu_range");
        }
        if data.is_empty() {
            return Ok(());
        }
        let stage = self.staging(data.len() as u64)?;
        // SAFETY: the mapping covers `data.len()` bytes of the staging buffer.
        unsafe {
            let device = &self.core.device;
            let at = check(
                device.map_memory(stage.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()),
                "gpu_memory",
            )?;
            std::ptr::copy_nonoverlapping(data.as_ptr(), at.cast::<u8>(), data.len());
            device.unmap_memory(stage.memory);
        }
        let region = [vk::BufferCopy::default()
            .dst_offset(offset)
            .size(data.len() as u64)];
        self.run(|device, commands| unsafe {
            // SAFETY: the range was checked against both buffers above.
            device.cmd_copy_buffer(commands, stage.buffer, to.buffer, &region);
        })
    }
    /// Reads `bytes` bytes at byte `offset` of `from`.
    pub fn download(
        &self,
        from: &Buffer,
        offset: u64,
        bytes: usize,
    ) -> Result<Vec<u8>, &'static str> {
        if offset + bytes as u64 > from.bytes {
            return Err("gpu_range");
        }
        let mut out = vec![0u8; bytes];
        if bytes == 0 {
            return Ok(out);
        }
        let stage = self.staging(bytes as u64)?;
        let region = [vk::BufferCopy::default()
            .src_offset(offset)
            .size(bytes as u64)];
        self.run(|device, commands| unsafe {
            // SAFETY: the range was checked against both buffers above.
            device.cmd_copy_buffer(commands, from.buffer, stage.buffer, &region);
        })?;
        // SAFETY: the mapping covers `bytes` bytes of the staging buffer.
        unsafe {
            let device = &self.core.device;
            let at = check(
                device.map_memory(stage.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()),
                "gpu_memory",
            )?;
            std::ptr::copy_nonoverlapping(at.cast::<u8>(), out.as_mut_ptr(), bytes);
            device.unmap_memory(stage.memory);
        }
        Ok(out)
    }
    /// Builds a compute kernel from SPIR-V. `bindings` storage buffers at set 0, bindings
    /// `0..bindings`, and `constants` push-constant words. `subgroup` pins the subgroup size.
    pub fn kernel(
        &self,
        spirv: &[u8],
        bindings: usize,
        constants: usize,
        subgroup: Option<u32>,
    ) -> Result<Kernel, &'static str> {
        let core = &self.core;
        let words: Vec<u32> = spirv
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        // SAFETY: the kernel owns every object created here; on failure the partly built
        // kernel is dropped, and destroying null handles is allowed.
        unsafe {
            let d = &core.device;
            let mut kernel = Kernel {
                core: core.clone(),
                module: vk::ShaderModule::null(),
                set_layout: vk::DescriptorSetLayout::null(),
                layout: vk::PipelineLayout::null(),
                pipeline: vk::Pipeline::null(),
                bindings,
                constants,
            };
            kernel.module = check(
                d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None),
                "gpu_kernel",
            )?;
            let slots: Vec<_> = (0..bindings as u32)
                .map(|binding| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(binding)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            kernel.set_layout = check(
                d.create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&slots),
                    None,
                ),
                "gpu_kernel",
            )?;
            let sets = [kernel.set_layout];
            let ranges = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .size(constants as u32 * 4)];
            let mut layout = vk::PipelineLayoutCreateInfo::default().set_layouts(&sets);
            if constants > 0 {
                layout = layout.push_constant_ranges(&ranges);
            }
            kernel.layout = check(d.create_pipeline_layout(&layout, None), "gpu_kernel")?;
            let mut size = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
                .required_subgroup_size(subgroup.unwrap_or(0));
            let mut stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(kernel.module)
                .name(c"main");
            if subgroup.is_some() {
                stage = stage
                    .flags(vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS)
                    .push_next(&mut size);
            }
            let pipelines = d
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(stage)
                        .layout(kernel.layout)],
                    None,
                )
                .map_err(|_| "gpu_kernel")?;
            kernel.pipeline = pipelines[0];
            Ok(kernel)
        }
    }
    /// Runs a batch of dispatches in order in one submission and waits for it.
    pub fn pass(&self, steps: &[Step]) -> Result<(), &'static str> {
        let core = &self.core;
        let _turn = core.turn.lock().map_err(|_| "gpu_failed")?;
        let mut sets = Vec::with_capacity(steps.len());
        let result = (|| {
            for step in steps {
                if step.buffers.len() != step.kernel.bindings
                    || step.constants.len() != step.kernel.constants
                {
                    return Err("gpu_kernel");
                }
                let layouts = [step.kernel.set_layout];
                // SAFETY: sets are freed below after the pass has completed.
                let set = unsafe {
                    core.device.allocate_descriptor_sets(
                        &vk::DescriptorSetAllocateInfo::default()
                            .descriptor_pool(core.descriptors)
                            .set_layouts(&layouts),
                    )
                }
                .map_err(|_| "gpu_failed")?[0];
                sets.push(set);
                let infos: Vec<_> = step
                    .buffers
                    .iter()
                    .map(|buffer| {
                        [vk::DescriptorBufferInfo::default()
                            .buffer(buffer.buffer)
                            .range(vk::WHOLE_SIZE)]
                    })
                    .collect();
                let writes: Vec<_> = infos
                    .iter()
                    .enumerate()
                    .map(|(binding, info)| {
                        vk::WriteDescriptorSet::default()
                            .dst_set(set)
                            .dst_binding(binding as u32)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(info)
                    })
                    .collect();
                // SAFETY: the set and every buffer are alive.
                unsafe { core.device.update_descriptor_sets(&writes, &[]) };
            }
            self.run_locked(|device, commands| {
                for (step, set) in steps.iter().zip(&sets) {
                    // SAFETY: bindings and constants were checked against the kernel; each
                    // dispatch is separated from the next by a full compute barrier.
                    unsafe {
                        device.cmd_bind_pipeline(
                            commands,
                            vk::PipelineBindPoint::COMPUTE,
                            step.kernel.pipeline,
                        );
                        device.cmd_bind_descriptor_sets(
                            commands,
                            vk::PipelineBindPoint::COMPUTE,
                            step.kernel.layout,
                            0,
                            &[*set],
                            &[],
                        );
                        if !step.constants.is_empty() {
                            let bytes: Vec<u8> = step
                                .constants
                                .iter()
                                .flat_map(|c| c.to_le_bytes())
                                .collect();
                            device.cmd_push_constants(
                                commands,
                                step.kernel.layout,
                                vk::ShaderStageFlags::COMPUTE,
                                0,
                                &bytes,
                            );
                        }
                        device.cmd_dispatch(
                            commands,
                            step.groups[0],
                            step.groups[1],
                            step.groups[2],
                        );
                        let barrier = [vk::MemoryBarrier::default()
                            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                            .dst_access_mask(
                                vk::AccessFlags::SHADER_READ
                                    | vk::AccessFlags::SHADER_WRITE
                                    | vk::AccessFlags::TRANSFER_READ,
                            )];
                        device.cmd_pipeline_barrier(
                            commands,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::PipelineStageFlags::COMPUTE_SHADER
                                | vk::PipelineStageFlags::TRANSFER,
                            vk::DependencyFlags::empty(),
                            &barrier,
                            &[],
                            &[],
                        );
                    }
                }
            })
        })();
        if !sets.is_empty() {
            // SAFETY: the pass has finished (or never started), so the sets are unused.
            unsafe {
                let _ = core.device.free_descriptor_sets(core.descriptors, &sets);
            }
        }
        result
    }
    /// The adapter name, for logs and test output.
    pub fn name(&self) -> &str {
        &self.core.name
    }
}
/// One dispatch: a kernel, its buffers in binding order, its push constants and the number
/// of workgroups on each axis.
pub struct Step<'a> {
    pub kernel: &'a Kernel,
    pub buffers: &'a [&'a Buffer],
    pub constants: &'a [u32],
    pub groups: [u32; 3],
}
/// Views plain numbers as bytes for upload.
pub fn bytes_of<T: Copy>(values: &[T]) -> &[u8] {
    // SAFETY: only used with f32, u16 and u32, which have no padding.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}
pub fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        super::safetensors::{narrow, widen},
        *,
    };

    /// Deterministic values in -1..1.
    fn noise(count: usize, seed: u64) -> Vec<f32> {
        let mut state = seed | 1;
        (0..count)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 40) as f32 / (1u64 << 23) as f32 - 1.0
            })
            .collect()
    }
    fn halves(values: &[f32]) -> Vec<u16> {
        values.iter().map(|v| narrow(*v)).collect()
    }
    /// Needs the Intel GPU: `cargo test --release -p kuno-image gpu -- --include-ignored`.
    #[test]
    #[ignore]
    fn gemm_matches_cpu_and_reports_speed() {
        let gpu = Gpu::open(Select::Vendor(0x8086)).unwrap();
        let kernel = gpu.kernel(shaders::GEMM, 3, 8, Some(16)).unwrap();
        let (batch, tokens, rows, columns) = (2usize, 256usize, 384usize, 96usize);
        let (input, weight) = (
            halves(&noise(batch * tokens * columns + 4, 1)),
            halves(&noise(batch * rows * columns + 8, 2)),
        );
        let (a, b) = (
            gpu.buffer(input.len() as u64 * 2).unwrap(),
            gpu.buffer(weight.len() as u64 * 2).unwrap(),
        );
        gpu.upload(&a, 0, bytes_of(&input)).unwrap();
        gpu.upload(&b, 0, bytes_of(&weight)).unwrap();
        let stride = rows + 5;
        let out = gpu
            .buffer((3 + batch * tokens * stride) as u64 * 4)
            .unwrap();
        gpu.pass(&[Step {
            kernel: &kernel,
            buffers: &[&a, &b, &out],
            constants: &[
                columns as u32,
                4,
                (tokens * columns) as u32,
                8,
                (rows * columns) as u32,
                3,
                (tokens * stride) as u32,
                stride as u32,
            ],
            groups: [(rows / 128) as u32, (tokens / 128) as u32, batch as u32],
        }])
        .unwrap();
        let got = floats(&gpu.download(&out, 0, out.bytes() as usize).unwrap());
        let mut worst = 0f64;
        for z in 0..batch {
            for t in 0..tokens {
                for r in 0..rows {
                    let want: f64 = (0..columns)
                        .map(|k| {
                            f64::from(widen(input[4 + (z * tokens + t) * columns + k]))
                                * f64::from(widen(weight[8 + (z * rows + r) * columns + k]))
                        })
                        .sum();
                    let have = f64::from(got[3 + (z * tokens + t) * stride + r]);
                    if (want - have).abs() > worst {
                        worst = (want - have).abs();
                        eprintln!("z{z} t{t} r{r} want {want} have {have}");
                    }
                }
            }
        }
        // Untouched gaps between rows stay zero.
        assert!(got[..3].iter().all(|v| *v == 0.0) && got[3 + rows] == 0.0);
        assert!(worst < 4e-6, "largest error {worst}");

        let (tokens, rows, columns) = (1920usize, 6144usize, 2048usize);
        let a = gpu.buffer((tokens * columns) as u64 * 2).unwrap();
        let b = gpu.buffer((rows * columns) as u64 * 2).unwrap();
        let out = gpu.buffer((tokens * rows) as u64 * 4).unwrap();
        gpu.upload(&a, 0, bytes_of(&halves(&noise(tokens * columns, 3))))
            .unwrap();
        gpu.upload(&b, 0, bytes_of(&halves(&noise(rows * columns, 4))))
            .unwrap();
        let constants = [columns as u32, 0, 0, 0, 0, 0, 0, rows as u32];
        let buffers = [&a, &b, &out];
        let step = || Step {
            kernel: &kernel,
            buffers: &buffers,
            constants: &constants,
            groups: [(rows / 128) as u32, (tokens / 128) as u32, 1],
        };
        gpu.pass(&[step()]).unwrap();
        let steps: Vec<Step> = (0..64).map(|_| step()).collect();
        gpu.pass(&steps).unwrap();
        let started = std::time::Instant::now();
        gpu.pass(&steps).unwrap();
        let seconds = started.elapsed().as_secs_f64() / steps.len() as f64;
        let flops = 2.0 * (tokens * rows * columns) as f64;
        println!(
            "{}: largest error {worst:.2e}; {tokens}x{rows}x{columns} in {:.2} ms, {:.1} TFLOPS",
            gpu.name(),
            seconds * 1e3,
            flops / seconds / 1e12
        );
    }
    /// The integer kernel is exact, so it is compared bit for bit, then timed at the shapes of
    /// the DiT's four linear layers.
    #[test]
    #[ignore = "needs the Intel GPU"]
    fn gemm8_is_exact_and_reports_speed() {
        let gpu = Gpu::open(Select::Vendor(0x8086)).unwrap();
        let kernel = gpu.kernel(shaders::GEMM8, 3, 5, Some(16)).unwrap();
        let bytes = |count: usize, seed: u64| -> Vec<i8> {
            noise(count, seed)
                .iter()
                .map(|v| (v * 90.0).clamp(-127.0, 127.0) as i8)
                .collect()
        };
        let (tokens, rows, columns) = (384usize, 512usize, 160usize);
        let (input, weight) = (
            bytes(tokens * columns + 16, 1),
            bytes(rows * columns + 32, 2),
        );
        let (a, b) = (
            gpu.buffer(input.len() as u64).unwrap(),
            gpu.buffer(weight.len() as u64).unwrap(),
        );
        gpu.upload(&a, 0, bytes_of(&input)).unwrap();
        gpu.upload(&b, 0, bytes_of(&weight)).unwrap();
        let stride = rows + 5;
        let out = gpu.buffer((3 + tokens * stride) as u64 * 4).unwrap();
        gpu.pass(&[Step {
            kernel: &kernel,
            buffers: &[&a, &b, &out],
            constants: &[columns as u32, 16, 32, 3, stride as u32],
            groups: [(rows / 256) as u32, (tokens / 128) as u32, 1],
        }])
        .unwrap();
        let got = gpu.download(&out, 0, out.bytes() as usize).unwrap();
        let got: Vec<i32> = got
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        for t in 0..tokens {
            for r in 0..rows {
                let want: i32 = (0..columns)
                    .map(|k| {
                        i32::from(input[16 + t * columns + k])
                            * i32::from(weight[32 + r * columns + k])
                    })
                    .sum();
                assert_eq!(got[3 + t * stride + r], want, "t{t} r{r}");
            }
        }
        // Untouched gaps between rows stay zero.
        assert!(got[..3].iter().all(|v| *v == 0) && got[3 + rows] == 0);

        // 4096 weight rows per dispatch keep the weight block cached while every token
        // passes over it; one dispatch for all rows is a quarter slower.
        const BLOCK: usize = 16;
        for (tokens, rows, columns) in [
            (8448usize, 21504usize, 5376usize),
            (8448, 28672, 5376),
            (8448, 5376, 14336),
            (8448, 5376, 7168),
        ] {
            let a = gpu.buffer((tokens * columns) as u64).unwrap();
            let b = gpu.buffer((rows * columns) as u64).unwrap();
            let out = gpu.buffer((tokens * rows) as u64 * 4).unwrap();
            gpu.upload(&a, 0, bytes_of(&bytes(tokens * columns, 3)))
                .unwrap();
            gpu.upload(&b, 0, bytes_of(&bytes(rows * columns, 4)))
                .unwrap();
            let buffers = [&a, &b, &out];
            let tiles = rows / 256;
            let constants: Vec<[u32; 5]> = (0..tiles)
                .step_by(BLOCK)
                .map(|first| {
                    [
                        columns as u32,
                        0,
                        (first * 256 * columns) as u32,
                        (first * 256) as u32,
                        rows as u32,
                    ]
                })
                .collect();
            let pass = || -> Vec<Step> {
                constants
                    .iter()
                    .enumerate()
                    .map(|(i, constants)| Step {
                        kernel: &kernel,
                        buffers: &buffers,
                        constants,
                        groups: [
                            BLOCK.min(tiles - i * BLOCK) as u32,
                            (tokens / 128) as u32,
                            1,
                        ],
                    })
                    .collect()
            };
            gpu.pass(&pass()).unwrap();
            let steps: Vec<Step> = (0..4).flat_map(|_| pass()).collect();
            let started = std::time::Instant::now();
            gpu.pass(&steps).unwrap();
            let seconds = started.elapsed().as_secs_f64() / 4.0;
            let ops = 2.0 * (tokens * rows * columns) as f64;
            println!(
                "{tokens}x{rows}x{columns} in {:.1} ms, {:.1} TOPS",
                seconds * 1e3,
                ops / seconds / 1e12
            );
        }
    }
}
