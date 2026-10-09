// Modified for Kuno's laboratory transient-I/O ownership/erasure patch (2026-10-05).
use super::storage::{WgpuResource, WgpuStorage};
use crate::schedule::{BindingsResource, ScheduleTask, ScheduledWgpuBackend};
use crate::{AutoCompiler, AutoRepresentation};
use alloc::sync::Arc;
use cubecl_common::{
    backtrace::BackTrace,
    bytes::Bytes,
    profile::{ProfileDuration, TimingMethod},
    stream_id::StreamId,
};
use cubecl_core::server::{Binding, StreamErrorMode};
use cubecl_core::zspace::Shape;
use cubecl_core::{
    MemoryConfiguration, WgpuCompilationOptions,
    future::DynFut,
    prelude::*,
    server::{
        CopyDescriptor, IoError, KernelArguments, LaunchError, ProfileError, ProfilingToken,
        ResourceLimitError, ServerCommunication, ServerError, ServerUtilities,
    },
    zspace::{Strides, strides},
};
#[cfg(feature = "spirv")]
use cubecl_core::{cache::CacheOption, compilation_cache::CompilationCache, hash::StableHash};
use cubecl_ir::MemoryDeviceProperties;
use cubecl_runtime::allocator::ContiguousMemoryLayoutPolicy;
use cubecl_runtime::memory_management::{ManagedMemoryHandle, MemoryUsage};
use cubecl_runtime::{
    compiler::CubeTask,
    config::{CubeClRuntimeConfig, RuntimeConfig},
    logging::ServerLogger,
    memory_management::MemoryAllocationMode,
    server::ComputeServer,
    storage::ManagedResource,
    stream::scheduler::{SchedulerMultiStream, SchedulerMultiStreamOptions, SchedulerStrategy},
    validation::{validate_cube_dim, validate_units},
};
use hashbrown::HashMap;
use wgpu::ComputePipeline;

/// Wgpu compute server.
#[derive(Debug)]
pub struct WgpuServer {
    pub(crate) device: wgpu::Device,
    // A buffer that can be used to store stream id without extra allocations.
    streams_pool: Vec<StreamId>,
    pipelines: HashMap<KernelId, Arc<ComputePipeline>>,
    scheduler: SchedulerMultiStream<ScheduledWgpuBackend>,
    #[cfg(feature = "spirv")]
    pub(crate) spirv_cache:
        Option<CompilationCache<(u64, StableHash), cubecl_spirv::SpirvCacheEntry>>,
    pub compilation_options: WgpuCompilationOptions,
    pub(crate) backend: wgpu::Backend,
    pub(crate) utilities: Arc<ServerUtilities<Self>>,
}

impl ServerCommunication for WgpuServer {
    const SERVER_COMM_ENABLED: bool = false;
}

impl WgpuServer {
    fn write_descriptors(
        &mut self,
        descriptors: Vec<(CopyDescriptor, Bytes)>,
        stream_id: StreamId,
        private: bool,
    ) {
        for (desc, data) in descriptors {
            let stream = self.scheduler.stream(&desc.handle.stream);

            if contiguous_strides(&desc.shape) != desc.strides {
                stream.error(ServerError::Io(IoError::UnsupportedStrides {
                    backtrace: BackTrace::capture(),
                }));
                return;
            }

            let resource = match stream.mem_manage.get_resource(desc.handle) {
                Ok(r) => r,
                Err(err) => {
                    stream.error(ServerError::Io(err));
                    return;
                }
            };
            let task = ScheduleTask::Write {
                data,
                buffer: resource,
                private,
            };

            self.scheduler.register(stream_id, task, &[]);
        }
    }

    /// Create a new server.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        memory_properties: MemoryDeviceProperties,
        memory_config: MemoryConfiguration,
        compilation_options: WgpuCompilationOptions,
        device: wgpu::Device,
        queue: wgpu::Queue,
        tasks_max: usize,
        backend: wgpu::Backend,
        timing_method: TimingMethod,
        utilities: ServerUtilities<Self>,
    ) -> Self {
        let backend_scheduler = ScheduledWgpuBackend::new(
            device.clone(),
            queue.clone(),
            memory_properties,
            memory_config,
            timing_method,
            tasks_max,
            utilities.logger.clone(),
        );

        let config = CubeClRuntimeConfig::get();
        let max_streams = config.streaming.max_streams;

        Self {
            compilation_options,
            streams_pool: Vec::new(),
            device,
            pipelines: HashMap::new(),
            scheduler: SchedulerMultiStream::new(
                utilities.logger.clone(),
                backend_scheduler,
                SchedulerMultiStreamOptions {
                    max_streams,
                    max_tasks: tasks_max,
                    strategy: SchedulerStrategy::Interleave,
                },
            ),
            #[cfg(feature = "spirv")]
            spirv_cache: {
                let config = cubecl_runtime::config::CubeClRuntimeConfig::get();
                if let Some(cache) = &config.compilation.cache {
                    let root = cache.root();
                    Some(CompilationCache::new(
                        "spirv",
                        CacheOption::default().name("vulkan").root(root),
                    ))
                } else {
                    None
                }
            },
            backend,
            utilities: Arc::new(utilities),
        }
    }

    fn prepare_bindings(&mut self, bindings: KernelArguments) -> Result<BindingsResource, IoError> {
        // Store all the resources we'll be using. This could be eliminated if
        // there was a way to tie the lifetime of the resource to the memory handle.
        let mut resources = Vec::with_capacity(bindings.buffers.len());

        for b in bindings.buffers.into_iter() {
            let stream = self.scheduler.stream(&b.stream);
            let resource = stream.mem_manage.get_resource(b)?;
            resources.push(resource);
        }

        Ok(BindingsResource {
            resources,
            info: bindings.info,
        })
    }

    fn pipeline(
        &mut self,
        kernel: <Self as ComputeServer>::Kernel,
        bindings: &KernelArguments,
        mode: ExecutionMode,
    ) -> Result<Arc<ComputePipeline>, LaunchError> {
        let mut kernel_id = kernel.id();
        kernel_id.mode(mode);

        if let Some(pipeline) = self.pipelines.get(&kernel_id) {
            return Ok(pipeline.clone());
        }

        let cached = self.load_cached_pipeline(&kernel_id, bindings, mode)?;

        if let Some(Ok(pipeline)) = cached {
            self.pipelines.insert(kernel_id, pipeline.clone());
            return Ok(pipeline);
        }

        validate_cube_dim(&self.utilities.properties, &kernel_id)?;
        validate_units(&self.utilities.properties, &kernel_id)?;

        let mut compiler = compiler(self.backend, &self.compilation_options);
        let mut compiled = compiler.compile(self, kernel, mode)?;

        if self.scheduler.logger.compilation_activated() {
            compiled.debug_info = Some(DebugInformation::new(
                compiler.lang_tag(),
                kernel_id.clone(),
            ));
        }
        self.scheduler.logger.log_compilation(&compiled);

        self.validate_shared(&compiled.repr)?;

        // /!\ Do not delete the following commented code.
        // This is useful while working on the metal compiler.
        // Also the errors are printed nicely which is not the case when this is the runtime
        // that does it.
        // println!("SOURCE:\n{}", compiled.source);
        // {
        //     // Write shader in metal file then compile it for error
        //     std::fs::write("shader.metal", &compiled.source).expect("should write to file");
        //     let _status = std::process::Command::new("xcrun")
        //         .args(vec![
        //             "-sdk",
        //             "macosx",
        //             "metal",
        //             "-o",
        //             "shader.ir",
        //             "-c",
        //             "shader.metal",
        //             "-w",
        //         ])
        //         .status()
        //         .expect("should launch the command");
        //     // std::process::exit(status.code().unwrap());
        // }
        let repr = compiled.repr.as_ref().map(|it| it.as_ref());
        let module = self.create_module(&compiled.entrypoint_name, repr, &compiled.source, mode)?;
        let pipeline = self.create_pipeline(&compiled.entrypoint_name, repr, module, bindings);
        self.pipelines.insert(kernel_id.clone(), pipeline.clone());

        #[cfg(feature = "spirv")]
        if let Some(Err(key)) = cached
            && let Some(crate::AutoRepresentation::SpirV(kernel)) = compiled.repr
        {
            let cache = self.spirv_cache.as_mut().unwrap();
            let result = cache.insert(
                key,
                cubecl_spirv::SpirvCacheEntry::new(compiled.entrypoint_name, kernel),
            );
            if let Err(err) = result {
                log::warn!("Unable to save the SPIR-V {err:?}");
            }
        }

        Ok(pipeline)
    }

    fn validate_shared(&self, repr: &Option<crate::AutoRepresentation>) -> Result<(), LaunchError> {
        let shared_bytes = repr.as_ref().map(|repr| match repr {
            AutoRepresentation::Wgsl(repr) => repr.shared_memory_bytes(),
            #[cfg(feature = "msl")]
            AutoRepresentation::Msl(repr) => repr.shared_memory_size(),
            #[cfg(feature = "spirv")]
            AutoRepresentation::SpirV(repr) => repr.shared_size,
        });
        let max_smem = self.utilities.properties.hardware.max_shared_memory_size;
        if let Some(shared_bytes) = shared_bytes
            && shared_bytes > max_smem
        {
            Err(ResourceLimitError::SharedMemory {
                requested: shared_bytes,
                max: max_smem,
                backtrace: BackTrace::capture(),
            }
            .into())
        } else {
            Ok(())
        }
    }
}

impl ComputeServer for WgpuServer {
    type Kernel = Box<dyn CubeTask<AutoCompiler>>;
    type Storage = WgpuStorage;
    type MemoryLayoutPolicy = ContiguousMemoryLayoutPolicy;
    type Info = wgpu::Backend;

    fn logger(&self) -> Arc<ServerLogger> {
        self.scheduler.logger.clone()
    }

    fn utilities(&self) -> Arc<ServerUtilities<Self>> {
        self.utilities.clone()
    }

    fn staging(
        &mut self,
        _sizes: &[usize],
        _stream_id: StreamId,
    ) -> Result<Vec<Bytes>, ServerError> {
        // TODO: Check if using a staging buffer is useful here.
        Err(IoError::UnsupportedIoOperation {
            backtrace: BackTrace::capture(),
        }
        .into())
    }

    fn initialize_memory(&mut self, memory: ManagedMemoryHandle, size: u64, stream_id: StreamId) {
        let stream = self.scheduler.stream(&stream_id);
        let reserved = stream.empty(size).unwrap();
        stream.mem_manage.bind(reserved, memory);
    }

    fn read(
        &mut self,
        descriptors: Vec<CopyDescriptor>,
        stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, ServerError>> {
        let mutation = self.scheduler.stream(&stream_id).private_mutation_guard();
        // Private ownership is stream-local; never read another stream's
        // bindings under this ledger (ordinary public read remains unchanged).
        if self.scheduler.stream(&stream_id).transient_io_enabled()
            && descriptors.iter().any(|d| d.handle.stream != stream_id)
        {
            return Box::pin(async {
                Err(super::transient_io::fault(
                    "private readback cross-stream binding",
                ))
            });
        }
        let mut streams = vec![stream_id];
        let mut resources = Vec::with_capacity(descriptors.len());
        for desc in descriptors {
            if contiguous_strides(&desc.shape) != desc.strides {
                return Box::pin(async {
                    Err(IoError::UnsupportedStrides {
                        backtrace: BackTrace::capture(),
                    }
                    .into())
                });
            }
            if !streams.contains(&desc.handle.stream) {
                streams.push(desc.handle.stream);
            }
            let stream = self.scheduler.stream(&desc.handle.stream);
            let resource = match stream.mem_manage.get_resource(desc.handle) {
                Ok(val) => val,
                Err(err) => return Box::pin(async move { Err(err.into()) }),
            };
            resources.push((resource, desc.shape, desc.elem_size));
        }

        self.scheduler.execute_streams(streams);

        let stream = self.scheduler.stream(&stream_id);
        let result = stream.read_resources(resources);
        if let Some(mutation) = mutation {
            mutation.accept();
        }
        result
    }

    fn write(&mut self, descriptors: Vec<(CopyDescriptor, Bytes)>, stream_id: StreamId) {
        self.write_descriptors(descriptors, stream_id, false);
    }

    fn configure_transient_io(
        &mut self,
        budget_bytes: u64,
        stream_id: StreamId,
    ) -> Result<(), ServerError> {
        let guard = self
            .scheduler
            .stream(&stream_id)
            .begin_transient_configure()?;
        guard.run(|| {
            self.scheduler.execute_streams(vec![stream_id]);
            self.scheduler
                .stream(&stream_id)
                .configure_transient_io(budget_bytes)
        })
    }

    fn configure_public_target_request_io(
        &mut self,
        stream_id: StreamId,
    ) -> Result<(), ServerError> {
        if !cubecl_common::device_handle::DeviceHandle::<Self>::is_blocking() {
            return Err(super::transient_io::fault(
                "public target I/O requires inline device",
            ));
        }
        let guard = self
            .scheduler
            .stream(&stream_id)
            .begin_transient_configure()?;
        guard.run(|| {
            self.scheduler.execute_streams(vec![stream_id]);
            self.scheduler
                .stream(&stream_id)
                .configure_public_target_request_io()
        })
    }

    fn public_target_request_io_configured(
        &mut self,
        stream_id: StreamId,
    ) -> Result<bool, ServerError> {
        self.scheduler
            .stream(&stream_id)
            .public_target_request_io_configured()
    }

    fn write_private(
        &mut self,
        descriptors: Vec<(CopyDescriptor, Bytes)>,
        stream_id: StreamId,
    ) -> Result<(), ServerError> {
        self.scheduler.stream(&stream_id).private_io_active()?;
        let mutation = self.scheduler.stream(&stream_id).private_mutation_guard();
        if descriptors
            .iter()
            .any(|(desc, _)| desc.handle.stream != stream_id)
        {
            return Err(super::transient_io::fault(
                "private upload cross-stream binding",
            ));
        }
        self.write_descriptors(descriptors, stream_id, true);
        if let Some(mutation) = mutation {
            mutation.accept();
        }
        Ok(())
    }

    fn read_private_cleanup(
        &mut self,
        descriptor: CopyDescriptor,
        stream_id: StreamId,
    ) -> Result<Bytes, ServerError> {
        // Enter BEFORE scheduler execution can invoke the ordinary regulator.
        // Live lease rejection precedes every scheduler/resource mutation.
        let guard = self
            .scheduler
            .stream(&stream_id)
            .begin_transient_cleanup()?;
        guard.run(|| {
            if descriptor.handle.stream != stream_id
                || descriptor.handle.offset_start.unwrap_or(0) != 0
                || descriptor.handle.offset_end.unwrap_or(0) != 0
                || descriptor.elem_size != 1
                || descriptor.strides.as_ref() != [1]
                || descriptor.shape.as_slice() != [descriptor.handle.size() as usize]
            {
                return Err(super::transient_io::fault(
                    "private cleanup read requires full span",
                ));
            }
            let resource = self
                .scheduler
                .stream(&stream_id)
                .mem_manage
                .get_resource(descriptor.handle)?;
            // Actual API-visible allocation must be the exact owning page.
            // Reject any hidden allocator tail or interior resource view.
            super::transient_io::cleanup_read_bounds(
                resource.offset,
                resource.size,
                resource.buffer.size(),
            )?;
            self.scheduler.execute_streams(vec![stream_id]);
            self.scheduler
                .stream(&stream_id)
                .read_private_cleanup_resource(resource)
        })
    }

    fn sanitize_transient_io(
        &mut self,
        stream_id: StreamId,
    ) -> DynFut<Result<cubecl_runtime::transient::TransientIoReceipt, ServerError>> {
        let result = self
            .scheduler
            .stream(&stream_id)
            .begin_transient_cleanup()
            .and_then(|guard| {
                guard.run(|| {
                    self.scheduler.execute_streams(vec![stream_id]);
                    self.scheduler
                        .stream(&stream_id)
                        .sanitize_transient_io(false)
                })
            });
        Box::pin(async move { result })
    }

    fn sanitize_transient_io_audited(
        &mut self,
        stream_id: StreamId,
    ) -> DynFut<Result<cubecl_runtime::transient::TransientIoReceipt, ServerError>> {
        let result = self
            .scheduler
            .stream(&stream_id)
            .begin_transient_cleanup()
            .and_then(|guard| {
                guard.run(|| {
                    self.scheduler.execute_streams(vec![stream_id]);
                    self.scheduler
                        .stream(&stream_id)
                        .sanitize_transient_io(true)
                })
            });
        Box::pin(async move { result })
    }

    fn begin_transient_io_reuse(&mut self, stream_id: StreamId) -> Result<u64, ServerError> {
        self.scheduler
            .stream(&stream_id)
            .validate_transient_io_reuse_mode(false)?;
        let guard = self
            .scheduler
            .stream(&stream_id)
            .begin_transient_cleanup()?;
        guard.run(|| {
            self.scheduler.execute_streams(vec![stream_id]);
            self.scheduler.stream(&stream_id).begin_transient_io_reuse()
        })
    }

    fn begin_public_target_request_io_reuse(
        &mut self,
        stream_id: StreamId,
    ) -> Result<u64, ServerError> {
        // Mode rejection precedes scheduler, flush, mapping and nonce mutation.
        self.scheduler
            .stream(&stream_id)
            .validate_transient_io_reuse_mode(true)?;
        let guard = self
            .scheduler
            .stream(&stream_id)
            .begin_transient_cleanup()?;
        guard.run(|| {
            self.scheduler.execute_streams(vec![stream_id]);
            self.scheduler
                .stream(&stream_id)
                .begin_public_target_request_io_reuse()
        })
    }

    fn recycle_transient_io(&mut self, nonce: u64, stream_id: StreamId) -> Result<(), ServerError> {
        let guard = self
            .scheduler
            .stream(&stream_id)
            .begin_transient_cleanup()?;
        guard.run(|| {
            self.scheduler.execute_streams(vec![stream_id]);
            self.scheduler
                .stream(&stream_id)
                .recycle_transient_io(nonce)
        })
    }

    fn accept_transient_io_reuse(
        &mut self,
        nonce: u64,
        stream_id: StreamId,
    ) -> Result<(), ServerError> {
        let guard = self
            .scheduler
            .stream(&stream_id)
            .begin_transient_cleanup()?;
        guard.run(|| {
            self.scheduler.execute_streams(vec![stream_id]);
            self.scheduler
                .stream(&stream_id)
                .accept_transient_io_reuse(nonce)
        })
    }

    fn end_transient_io_reuse(
        &mut self,
        nonce: u64,
        stream_id: StreamId,
    ) -> Result<cubecl_runtime::transient::TransientIoReceipt, ServerError> {
        let guard = self
            .scheduler
            .stream(&stream_id)
            .begin_transient_cleanup()?;
        guard.run(|| {
            self.scheduler.execute_streams(vec![stream_id]);
            self.scheduler
                .stream(&stream_id)
                .end_transient_io_reuse(nonce)
        })
    }

    fn get_resource(
        &mut self,
        binding: Binding,
        stream_id: StreamId,
    ) -> Result<ManagedResource<WgpuResource>, ServerError> {
        let mut streams = vec![stream_id];
        if binding.stream != stream_id {
            streams.push(binding.stream);
        }
        self.scheduler.execute_streams(streams);
        let stream = self.scheduler.stream(&binding.stream);
        let memory = binding.memory.clone();
        let resource = stream.mem_manage.get_resource(binding)?;

        Ok(ManagedResource::new(memory, resource))
    }

    fn retained_resource_metadata(
        &mut self,
        bindings: Vec<Binding>,
        stream_id: StreamId,
    ) -> Result<Vec<ManagedResource<WgpuResource>>, ServerError> {
        cubecl_runtime::transient::retained_metadata_bindings(&bindings, stream_id)?;
        let stream = self.scheduler.stream(&stream_id);
        stream.private_io_active()?;
        let mut resources = Vec::new();
        resources
            .try_reserve_exact(bindings.len())
            .map_err(|_| super::transient_io::fault("retained metadata bounded host allocation"))?;
        for binding in bindings {
            let memory = binding.memory.clone();
            let resource = stream.mem_manage.get_resource(binding)?;
            resources.push(ManagedResource::new(memory, resource));
        }
        // No execute_streams/flush/sync/cleanup/queue operation. The retained
        // bindings prevent relocation/reuse of these already initialized spans.
        Ok(resources)
    }

    fn clear_private_owners(
        &mut self,
        bindings: Vec<Binding>,
        stream_id: StreamId,
    ) -> Result<(), ServerError> {
        cubecl_runtime::transient::clear_owner_bindings(&bindings, stream_id)?;
        let mutation = self.scheduler.stream(&stream_id).private_mutation_guard();
        self.scheduler.stream(&stream_id).private_io_active()?;
        // Enqueue earlier scheduled consumers first; no submission or wait.
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        let mut resources = Vec::with_capacity(bindings.len());
        for binding in bindings {
            let size = binding.size;
            let memory = binding.memory.clone();
            let resource = stream.mem_manage.get_resource(binding)?;
            if resource.size != size {
                return Err(super::transient_io::fault("private clear owning extent"));
            }
            resources.push(ManagedResource::new(memory, resource));
        }
        stream.clear_private_owners(&resources)?;
        if let Some(mutation) = mutation {
            mutation.accept();
        }
        Ok(())
    }

    unsafe fn launch(
        &mut self,
        kernel: Self::Kernel,
        count: CubeCount,
        args: KernelArguments,
        mode: ExecutionMode,
        stream_id: StreamId,
    ) {
        let mutation = self.scheduler.stream(&stream_id).private_mutation_guard();
        let pipeline = match self.pipeline(kernel, &args, mode) {
            Ok(val) => val,
            Err(err) => {
                // We make the stream that would execute the kernel in error.
                let stream = self.scheduler.stream(&stream_id);
                stream.errors.push(ServerError::Launch(err));
                return;
            }
        };

        self.streams_pool.clear();
        args.buffers
            .iter()
            .for_each(|b| self.streams_pool.push(b.stream));

        let resources = match self.prepare_bindings(args) {
            Ok(val) => val,
            Err(err) => {
                // We make the stream that would execute the kernel in error.
                let stream = self.scheduler.stream(&stream_id);
                stream.errors.push(ServerError::Io(err));
                return;
            }
        };
        let task = ScheduleTask::Execute {
            pipeline,
            count,
            resources,
        };

        self.scheduler.register(stream_id, task, &self.streams_pool);
        if let Some(mutation) = mutation {
            mutation.accept();
        }
    }

    fn flush(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        let mutation = self.scheduler.stream(&stream_id).private_mutation_guard();
        self.scheduler.execute_streams(vec![stream_id]);

        let stream = self.scheduler.stream(&stream_id);

        let result = stream.flush(StreamErrorMode {
            ignore: false,
            flush: true,
        });
        if result.is_ok() {
            if let Some(mutation) = mutation {
                mutation.accept();
            }
        }
        result
    }

    /// Returns the total time of GPU work this sync completes.
    fn sync(&mut self, stream_id: StreamId) -> DynFut<Result<(), ServerError>> {
        let mutation = self.scheduler.stream(&stream_id).private_mutation_guard();
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);

        let result = stream.sync();
        if let Some(mutation) = mutation {
            mutation.accept();
        }
        result
    }

    fn start_profile(&mut self, stream_id: StreamId) -> Result<ProfilingToken, ServerError> {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        stream.start_profile()
    }

    fn end_profile(
        &mut self,
        stream_id: StreamId,
        token: ProfilingToken,
    ) -> Result<ProfileDuration, ProfileError> {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);

        stream.end_profile(token)
    }

    fn memory_usage(&mut self, stream_id: StreamId) -> Result<MemoryUsage, ServerError> {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        Ok(stream.mem_manage.memory_usage())
    }

    fn memory_cleanup(&mut self, stream_id: StreamId) {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        stream.mem_manage.memory_cleanup(true);
    }

    fn allocation_mode(&mut self, mode: MemoryAllocationMode, stream_id: StreamId) {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        stream.mem_manage.mode(mode);
    }
}

fn compiler(backend: wgpu::Backend, options: &WgpuCompilationOptions) -> AutoCompiler {
    let _ = options; // Unused without `spirv` feature
    match backend {
        #[cfg(feature = "spirv")]
        wgpu::Backend::Vulkan if options.supports_vulkan => AutoCompiler::SpirV(Default::default()),
        #[cfg(feature = "msl")]
        wgpu::Backend::Metal => AutoCompiler::Msl(Default::default()),
        _ => AutoCompiler::Wgsl(Default::default()),
    }
}

pub(crate) fn contiguous_strides(shape: &Shape) -> Strides {
    let rank = shape.len();
    let mut strides = strides![1; rank];
    for i in (0..rank - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}
