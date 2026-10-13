//! Joined, request-local exchange of original model weights. Activations and
//! request content have no binding in this pipeline. Ordinary external host
//! sources remain read-only; only the owning completed cache transaction calls
//! this raw device operation, before publishing its new logical bijection.
use super::*;
use zeroize::Zeroize;

pub(crate) const SHADER: &str = r#"
@group(0) @binding(0) var<storage,read_write> resident:array<u32>;
@group(0) @binding(1) var<storage,read_write> host:array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    if arrayLength(&resident)!=arrayLength(&host) {return;}
    if id.x<arrayLength(&resident) {
        let outgoing=resident[id.x];
        let incoming=host[id.x];
        resident[id.x]=incoming;
        host[id.x]=outgoing;
    }
}
"#;

pub(crate) struct GpuExpertExchange {
    pub(crate) resident: WgpuResource,
    pub(crate) layer: usize,
    pub(crate) role: usize,
    pub(crate) page: usize,
    pub(crate) host_first: usize,
    pub(crate) bytes: usize,
}
impl Drop for GpuExpertExchange {
    fn drop(&mut self) {
        self.resident.offset.zeroize();
        self.layer.zeroize();
        self.role.zeroize();
        self.page.zeroize();
        self.host_first.zeroize();
        self.bytes.zeroize();
    }
}
pub(super) struct ExchangeRuntime {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}
impl Mirror<'_, '_, '_> {
    /// # Safety
    /// Same device/queue pair as the original runtime initialization. Every
    /// CubeCL consumer has completed; no CPU map or new consumer may coexist.
    /// Resident resources must be the original role cells authenticated by the
    /// Sources owner, and all proposed swaps must have disjoint role cells.
    /// After success the caller publishes ALL swaps/table changes together.
    pub(crate) unsafe fn exchange_strata_gpu(
        &mut self,
        queue: &wgpu::Queue,
        swaps: &[GpuExpertExchange],
    ) -> Result<()> {
        if !crate::strata_port::enabled()
            || swaps.is_empty()
            || swaps.len() > 48 * crate::strata_cache::max_swaps_per_layer() * 3
        {
            return Err(rejected());
        }
        let alignment = self.device.limits().min_storage_buffer_offset_alignment as u64;
        for (index, swap) in swaps.iter().enumerate() {
            if swap.layer >= 48
                || swap.role >= 3
                || swap.page >= 3
                || swap.bytes == 0
                || swap.bytes > 1024 * 1024
                || !swap.bytes.is_multiple_of(4)
                || !swap.resident.offset.is_multiple_of(alignment)
                || !(swap.host_first as u64).is_multiple_of(alignment)
                || swap.resident.size != swap.bytes as u64
                || swap
                    .resident
                    .offset
                    .checked_add(swap.resident.size)
                    .is_none_or(|n| n > swap.resident.buffer.size())
            {
                return Err(rejected());
            }
            let owner = self
                .pages
                .get(swap.layer * 9 + swap.role * 3 + swap.page)
                .ok_or_else(rejected)?;
            require_readonly_phase(owner.phase)?;
            if swap
                .host_first
                .checked_add(swap.bytes)
                .is_none_or(|n| n > owner.logical as usize)
                || owner.buffer.as_ref() == Some(&swap.resident.buffer)
            {
                return Err(rejected());
            }
            for prior in &swaps[..index] {
                let resident_overlap = prior.resident.buffer == swap.resident.buffer
                    && prior.resident.offset < swap.resident.offset + swap.resident.size
                    && swap.resident.offset < prior.resident.offset + prior.resident.size;
                let host_overlap = [prior.layer, prior.role, prior.page]
                    == [swap.layer, swap.role, swap.page]
                    && prior.host_first < swap.host_first + swap.bytes
                    && swap.host_first < prior.host_first + prior.bytes;
                if resident_overlap || host_overlap {
                    return Err(rejected());
                }
            }
        }
        let oom_scope = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let validation_scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let result = (|| -> Result<()> {
            if self.strata_exchange.is_none() {
                let layout =
                    self.device
                        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                            label: Some("StrataPublicExpertExchange"),
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
                        label: Some("StrataPublicExpertExchange"),
                        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
                    });
                let pipeline_layout =
                    self.device
                        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                            label: Some("StrataPublicExpertExchange"),
                            bind_group_layouts: &[Some(&layout)],
                            immediate_size: 0,
                        });
                let pipeline =
                    self.device
                        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                            label: Some("StrataPublicExpertExchange"),
                            layout: Some(&pipeline_layout),
                            module: &module,
                            entry_point: Some("main"),
                            compilation_options: Default::default(),
                            cache: None,
                        });
                self.strata_exchange = Some(ExchangeRuntime { layout, pipeline });
            }
            let runtime = self.strata_exchange.as_ref().ok_or_else(rejected)?;
            let mut groups = Vec::with_capacity(swaps.len());
            for swap in swaps {
                let host = self.pages[swap.layer * 9 + swap.role * 3 + swap.page]
                    .buffer
                    .as_ref()
                    .ok_or_else(rejected)?;
                groups.push(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("StrataPublicExpertExchange"),
                    layout: &runtime.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &swap.resident.buffer,
                                offset: swap.resident.offset,
                                size: std::num::NonZeroU64::new(swap.bytes as u64),
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: host,
                                offset: swap.host_first as u64,
                                size: std::num::NonZeroU64::new(swap.bytes as u64),
                            }),
                        },
                    ],
                }));
            }
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("StrataPublicExpertExchange"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("StrataPublicExpertExchange"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&runtime.pipeline);
                for (group, swap) in groups.iter().zip(swaps) {
                    pass.set_bind_group(0, group, &[]);
                    pass.dispatch_workgroups((swap.bytes / 4).div_ceil(256) as u32, 1, 1);
                }
            }
            queue.submit([encoder.finish()]);
            join(&self.device)
        })();
        let validation = cubecl::future::block_on(validation_scope.pop());
        let oom = cubecl::future::block_on(oom_scope.pop());
        if validation.is_some() || oom.is_some() {
            return Err("Strata public GPU expert exchange rejected");
        }
        result
    }
}
