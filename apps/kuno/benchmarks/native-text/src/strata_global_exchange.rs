//! Owning original-weight exchange for variable global cells. Request-derived
//! offsets never enter a log or a public pointer table. Whole joined transaction.
use super::*;

pub(crate) struct Pair {
    pub gpu: WgpuResource,
    pub host: WgpuResource,
}
pub(super) struct Runtime {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}
impl Host {
    /// # Safety
    /// Same actual owning device/queue. All source consumers have completed,
    /// no CPU map or new consumer overlaps, and Sources authenticated every GPU
    /// cell. All three roles of each proposed pair must be published together.
    pub(crate) unsafe fn exchange(&mut self, queue: &wgpu::Queue, pairs: &[Pair]) -> KResult<()> {
        if self.closed
            || pairs.is_empty()
            || pairs.len() > crate::strata_global_adapt::MAX_SWAPS * 3
        {
            return Err(fail());
        }
        let align = u64::from(self.device.limits().min_storage_buffer_offset_alignment);
        for (i, pair) in pairs.iter().enumerate() {
            if pair.gpu.size != pair.host.size
                || pair.gpu.size == 0
                || pair.gpu.size > 1024 * 1024
                || !pair.gpu.size.is_multiple_of(4)
                || pair.gpu.buffer == pair.host.buffer
                || !pair.gpu.offset.is_multiple_of(align)
                || !pair.host.offset.is_multiple_of(align)
                || pair
                    .gpu
                    .offset
                    .checked_add(pair.gpu.size)
                    .is_none_or(|n| n > pair.gpu.buffer.size())
                || pair
                    .host
                    .offset
                    .checked_add(pair.host.size)
                    .is_none_or(|n| n > pair.host.buffer.size())
            {
                return Err(fail());
            }
            let owner = self
                .pages
                .iter()
                .find(|p| p.owner.buffer.as_ref() == Some(&pair.host.buffer))
                .ok_or_else(fail)?;
            if owner.owner.phase != Phase::Registered
                || owner.authority.is_none()
                || pair.host.offset + pair.host.size > owner.owner.logical
            {
                return Err(fail());
            }
            for earlier in &pairs[..i] {
                for a in [&earlier.gpu, &earlier.host] {
                    for b in [&pair.gpu, &pair.host] {
                        if a.buffer == b.buffer
                            && a.offset < b.offset + b.size
                            && b.offset < a.offset + a.size
                        {
                            return Err(fail());
                        }
                    }
                }
            }
        }
        let oom = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let validation = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let result = (|| -> KResult<()> {
            if self.exchange_runtime.is_none() {
                let layout =
                    self.device
                        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                            label: Some("OriginalGlobalCellExchange"),
                            entries: &std::array::from_fn::<_, 2, _>(|binding| {
                                wgpu::BindGroupLayoutEntry {
                                    binding: binding as u32,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                }
                            }),
                        });
                let module = self
                    .device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("OriginalGlobalCellExchange"),
                        source: wgpu::ShaderSource::Wgsl(
                            crate::public_host_expert_mirror::strata_gpu_exchange::SHADER.into(),
                        ),
                    });
                let pipeline_layout =
                    self.device
                        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                            label: Some("OriginalGlobalCellExchange"),
                            bind_group_layouts: &[Some(&layout)],
                            immediate_size: 0,
                        });
                let pipeline =
                    self.device
                        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                            label: Some("OriginalGlobalCellExchange"),
                            layout: Some(&pipeline_layout),
                            module: &module,
                            entry_point: Some("main"),
                            compilation_options: Default::default(),
                            cache: None,
                        });
                self.exchange_runtime = Some(Runtime { layout, pipeline });
            }
            let runtime = self.exchange_runtime.as_ref().ok_or_else(fail)?;
            let groups = pairs
                .iter()
                .map(|p| {
                    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("OriginalGlobalCellExchange"),
                        layout: &runtime.layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: p.gpu.as_wgpu_bind_resource(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: p.host.as_wgpu_bind_resource(),
                            },
                        ],
                    })
                })
                .collect::<Vec<_>>();
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("OriginalGlobalCellExchange"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("OriginalGlobalCellExchange"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&runtime.pipeline);
                for (group, pair) in groups.iter().zip(pairs) {
                    pass.set_bind_group(0, group, &[]);
                    pass.dispatch_workgroups((pair.gpu.size / 4).div_ceil(256) as u32, 1, 1);
                }
            }
            // SAFETY: compute pass ended, original public HOST roots retained;
            // checked fence completion precedes all CPU mappings/invalidations.
            unsafe {
                encoder.as_hal_mut::<wgpu::hal::api::Vulkan, _, _>(|encoder| {
                    encoder
                        .ok_or_else(fail)?
                        .kuno_public_host_read_barrier()
                        .map_err(Error::InvalidFormat)
                })?;
            }
            queue.submit([encoder.finish()]);
            join(&self.device).map_err(Error::InvalidFormat)
        })();
        let invalid = cubecl_common::future::block_on(validation.pop()).is_some();
        let out_of_memory = cubecl_common::future::block_on(oom.pop()).is_some();
        if invalid || out_of_memory {
            return Err(fail());
        }
        result
    }
}
