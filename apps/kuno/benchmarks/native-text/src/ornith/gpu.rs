//! Small direct command encoder for the fully resident, fixed-public speed lab.
//! Model weights are public; every mutable allocation is owned and erased here.
use std::collections::BTreeMap;
use std::future::Future;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};
use std::time::Duration;
use wgpu::util::DeviceExt;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub fn block_on<T>(future: impl Future<Output = T>) -> T {
    struct ThreadWake(std::thread::Thread);
    impl Wake for ThreadWake {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
        fn wake_by_ref(self: &Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = Waker::from(Arc::new(ThreadWake(std::thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::park(),
        }
    }
}

pub fn words(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub struct Step {
    pub name: String,
    pub pipeline: wgpu::ComputePipeline,
    pub bindings: wgpu::BindGroup,
    pub grid: [u32; 3],
}

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
    pub arena: wgpu::Buffer,
    pub routing: wgpu::Buffer,
    pub control: wgpu::Buffer,
    pub readback: wgpu::Buffer,
    pub dummy: wgpu::Buffer,
    pub states: Vec<wgpu::Buffer>,
    pub trusted_bounds: bool,
    layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    pipelines: BTreeMap<String, wgpu::ComputePipeline>,
}

impl Gpu {
    pub fn new() -> Result<Self> {
        let trusted_bounds = match std::env::var("KUNO_LAB_ORNITH_TRUSTED_BOUNDS") {
            Err(std::env::VarError::NotPresent) => false,
            Ok(value) if value == "1" => true,
            _ => return Err("Explicit fixed-geometry bounds selection rejected".into()),
        };
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapters = block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN));
        let adapter = adapters
            .into_iter()
            .find(|a| {
                let info = a.get_info();
                info.vendor == 0x8086 && info.device == 0xe223
            })
            .ok_or("The fully resident lab requires the inspected B70")?;
        let info = adapter.get_info();
        let mut features = wgpu::Features::SUBGROUP;
        if std::env::var("KUNO_LAB_ORNITH_PROFILE").as_deref() == Ok("1") {
            features |=
                wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES;
        }
        if !adapter.features().contains(features) {
            return Err("Subgroup arithmetic unavailable".into());
        }
        let mut limits = wgpu::Limits::default();
        let available = adapter.limits();
        limits.max_buffer_size = available.max_buffer_size.min(2 * 1024 * 1024 * 1024);
        limits.max_storage_buffer_binding_size = available.max_storage_buffer_binding_size;
        limits.max_storage_buffers_per_shader_stage = 5;
        limits.max_compute_workgroup_storage_size = 24 * 1024;
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("ornith-public-lab"),
            required_features: features,
            required_limits: limits,
            ..Default::default()
        }))?;
        let entries = (0..7)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 4 || binding == 6 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: binding < 2,
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect::<Vec<_>>();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let private = |size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("transient-public-lab-state"),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let private_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let arena = private(2 * 1024 * 1024, private_usage);
        let routing = private(4096, private_usage);
        let control = private(
            16,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let readback = private(
            16,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let dummy = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("public-zero-placeholder"),
            contents: &[0; 16],
            usage: wgpu::BufferUsages::STORAGE,
        });
        Ok(Self {
            device,
            queue,
            info,
            arena,
            routing,
            control,
            readback,
            dummy,
            states: Vec::new(),
            trusted_bounds,
            layout,
            pipeline_layout,
            pipelines: BTreeMap::new(),
        })
    }

    pub fn state(&mut self, bytes: u64) -> usize {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("transient-public-lab-recurrent-or-kv"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.states.push(buffer);
        self.states.len() - 1
    }

    pub fn pipeline(&mut self, name: &str, code: String) -> wgpu::ComputePipeline {
        if let Some(pipeline) = self.pipelines.get(name) {
            return pipeline.clone();
        }
        let descriptor = wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(code.into()),
        };
        let module = if self.trusted_bounds {
            // SAFETY: this is only the fixed-public laboratory. Graph::validate
            // checks every original tensor shape/type; token and context bounds
            // are checked before dispatch. Each subgroup row is guarded, routed
            // ranks select IDs0..255, and all fixed arena ranges fit524288f32.
            // GDN uses24576+32*128*128f32; full KV uses2*4096*512f32.
            // Public probes admit32 rows, columns2048, and original exact row
            // extents. No external geometry or production request reaches here.
            unsafe {
                self.device.create_shader_module_trusted(
                    descriptor,
                    wgpu::ShaderRuntimeChecks::unchecked(),
                )
            }
        } else {
            self.device.create_shader_module(descriptor)
        };
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&self.pipeline_layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        self.pipelines.insert(name.to_owned(), pipeline.clone());
        pipeline
    }

    pub fn step(
        &self,
        pipeline: wgpu::ComputePipeline,
        weight: Option<&wgpu::Buffer>,
        second: Option<&wgpu::Buffer>,
        state: Option<usize>,
        params: [u32; 8],
        grid: [u32; 3],
    ) -> Step {
        let parameters = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("public-fixed-geometry"),
                contents: &words(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let buffers = [
            weight.unwrap_or(&self.dummy),
            second.unwrap_or(&self.dummy),
            &self.arena,
            &self.states[state.unwrap_or(0)],
            &parameters,
            &self.routing,
            &self.control,
        ];
        let entries = buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>();
        let bindings = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &entries,
        });
        Step {
            name: String::new(),
            pipeline,
            bindings,
            grid,
        }
    }

    pub fn wait(&self) -> Result<()> {
        self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(60)),
        })?;
        Ok(())
    }

    /// Bounded laboratory readback. Callers admit only fixed public probes or
    /// public-fixture timestamp counters; real requests never use this path.
    pub fn public_read(
        &self,
        source: &wgpu::Buffer,
        offset: u64,
        bytes: u64,
    ) -> Result<zeroize::Zeroizing<Vec<u8>>> {
        if bytes == 0
            || bytes > 1024 * 1024
            || !bytes.is_multiple_of(4)
            || offset + bytes > source.size()
        {
            return Err("Public probe readback bound exceeded".into());
        }
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fixed-public-probe-readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(source, offset, &staging, 0, bytes);
        self.queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.wait()?;
        rx.recv()??;
        let view = staging.slice(..).get_mapped_range();
        let result = zeroize::Zeroizing::new(view.to_vec());
        drop(view);
        staging.unmap();
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.clear_buffer(&staging, 0, None);
        self.queue.submit([encoder.finish()]);
        self.wait()?;
        Ok(result)
    }

    pub fn result(&self) -> Result<[u32; 4]> {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.routing, 1024, &self.readback, 0, 16);
        self.queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        self.readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
        self.wait()?;
        rx.recv()??;
        let view = self.readback.slice(..).get_mapped_range();
        let mut result = [0; 4];
        for (value, bytes) in result.iter_mut().zip(view.chunks_exact(4)) {
            *value = u32::from_le_bytes(bytes.try_into().unwrap());
        }
        drop(view);
        self.readback.unmap();
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.clear_buffer(&self.readback, 0, None);
        self.queue.submit([encoder.finish()]);
        if !f32::from_bits(result[1]).is_finite() || !f32::from_bits(result[3]).is_finite() {
            return Err("Native final logits non-finite".into());
        }
        Ok(result)
    }

    pub fn clear(&self) -> Result<()> {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        for buffer in [&self.arena, &self.routing, &self.control, &self.readback]
            .into_iter()
            .chain(self.states.iter())
        {
            encoder.clear_buffer(buffer, 0, None);
        }
        self.queue.submit([encoder.finish()]);
        self.wait()
    }

    pub fn private_bytes(&self) -> u64 {
        self.arena.size()
            + self.routing.size()
            + self.control.size()
            + self.readback.size()
            + self.states.iter().map(wgpu::Buffer::size).sum::<u64>()
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        let _ = self.clear();
    }
}
