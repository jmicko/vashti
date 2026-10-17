// Modified for Kuno's laboratory transient-I/O ownership/erasure patch (2026-10-05).
use super::{mem_manager::WgpuMemManager, poll::WgpuPoll, timings::QueryProfiler};
use crate::{WgpuResource, controller::WgpuAllocController, schedule::ScheduleTask};
use cubecl_common::{
    backtrace::BackTrace,
    bytes::Bytes,
    profile::{ProfileDuration, TimingMethod},
};
use cubecl_core::{
    CubeCount, MemoryConfiguration,
    future::{self, DynFut},
    server::{IoError, LaunchError, ProfileError, ProfilingToken, ServerError, StreamErrorMode},
    zspace::Shape,
};
use cubecl_ir::MemoryDeviceProperties;
use cubecl_runtime::{
    logging::ServerLogger, memory_management::ManagedMemoryHandle,
    timestamp_profiler::TimestampProfiler,
};
use std::{future::Future, num::NonZero, pin::Pin, sync::Arc};
use wgpu::ComputePipeline;

enum NativeReplay {
    Missing,
    Joined,
    Pending(cubecl_runtime::server::NativeStrataSubmission),
}

static SUBMISSION_WAIT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SUBMISSION_WAITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Content-free process-global attribution, not an isolated GPU kernel timer.
pub fn transient_gpu_submission_wait_stats() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (
        SUBMISSION_WAIT_NS.load(Relaxed),
        SUBMISSION_WAITS.load(Relaxed),
    )
}

#[cfg(not(target_family = "wasm"))]
fn record_submission_wait(elapsed: std::time::Duration) {
    use std::sync::atomic::Ordering::Relaxed;
    let ns = elapsed.as_nanos().min(u64::MAX as u128) as u64;
    let _ = SUBMISSION_WAIT_NS.fetch_update(Relaxed, Relaxed, |old| Some(old.saturating_add(ns)));
    let _ = SUBMISSION_WAITS.fetch_update(Relaxed, Relaxed, |old| Some(old.saturating_add(1)));
}

#[derive(Debug)]
enum Timings {
    Device(QueryProfiler),
    System(TimestampProfiler),
}

#[derive(Debug)]
pub struct WgpuStream {
    pub mem_manage: WgpuMemManager,
    pub device: wgpu::Device,
    pub errors: Vec<ServerError>,
    compute_pass: Option<wgpu::ComputePass<'static>>,
    timings: Timings,
    tasks_count: usize,
    tasks_max: usize,
    queue: wgpu::Queue,
    encoder: wgpu::CommandEncoder,
    encoder_dirty: bool,
    poll: WgpuPoll,
    submission_load: SubmissionLoad,
    /// Number of consecutive `write_buffer` calls without a `queue.submit()`.
    /// Used to prevent wgpu staging buffer pool exhaustion during bulk writes
    /// (e.g. model loading with hundreds of tensors).
    pending_write_count: usize,
    transient_io: Option<super::transient_io::TransientIo>,
    strata_graph: Option<super::strata_graph::Graph>,
    native_init_pipeline: Option<wgpu::ComputePipeline>,
    cleanup_boundary: Arc<super::cleanup_boundary::CleanupBoundary>,
}

impl WgpuStream {
    /// Creates a new WGPU stream.
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        memory_properties: MemoryDeviceProperties,
        memory_config: MemoryConfiguration,
        timing_method: TimingMethod,
        tasks_max: usize,
        logger: Arc<ServerLogger>,
    ) -> Self {
        let timings = if timing_method == TimingMethod::Device {
            Timings::Device(QueryProfiler::new(&queue, &device))
        } else {
            if cfg!(target_family = "wasm") {
                // On WASM, there's not much we can do here anymore. This should be very rare however,
                // all modern GPU's support timestamp queries.
                panic!(
                    "Cannot profile on web assembly without timestamp_query feature as it requires blocking."
                );
            }
            Timings::System(TimestampProfiler::default())
        };

        let poll = WgpuPoll::new(device.clone());

        #[allow(unused_mut)]
        let mut mem_manage =
            WgpuMemManager::new(device.clone(), memory_properties, memory_config, logger);

        Self {
            mem_manage,
            compute_pass: None,
            timings,
            errors: Vec::new(),
            encoder: {
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("CubeCL Tasks Encoder"),
                })
            },
            strata_graph: if super::strata_graph::enabled() {
                Some(
                    super::strata_graph::Graph::new(&device)
                        .expect("native Strata graph initialization"),
                )
            } else {
                None
            },
            native_init_pipeline: None,
            encoder_dirty: false,
            device,
            queue,
            tasks_count: 0,
            tasks_max,
            poll,
            submission_load: SubmissionLoad::default(),
            pending_write_count: 0,
            transient_io: None,
            cleanup_boundary: Arc::default(),
        }
    }

    /// Enqueue a [`ScheduleTask`] on this stream.
    ///
    /// # Arguments
    ///
    /// * `task` - The task to execute.
    pub fn enqueue_task(&mut self, task: ScheduleTask) {
        if self.cleanup_boundary.retired()
            || self.transient_io.as_ref().is_some_and(|io| io.retired())
        {
            self.error(super::transient_io::fault("private I/O stream retired"));
            return;
        }
        let mutation = self.transient_io.as_ref().map(|io| io.mutation_guard());
        let scopes = self
            .transient_io
            .as_ref()
            .map(|_| super::transient_io::IoScopes::new(&self.device));
        self.enqueue_task_inner(task);
        if let Some(scopes) = scopes {
            scopes.finish(self.transient_io.as_mut().unwrap());
        }
        if let Some(mutation) = mutation {
            mutation.accept();
        }
    }
    /// One already-owned serialized scheduler batch; guards remain synchronous
    /// and thread-local. No ErrorScopeGuard is stored in a Send ComputeServer.
    pub fn enqueue_tasks(&mut self, tasks: impl Iterator<Item = ScheduleTask>) {
        if !self
            .transient_io
            .as_ref()
            .is_some_and(|io| io.batches_info())
        {
            for task in tasks {
                self.enqueue_task(task);
            }
            return;
        }
        if self.cleanup_boundary.retired()
            || self.transient_io.as_ref().is_some_and(|io| io.retired())
        {
            self.error(super::transient_io::fault("private I/O stream retired"));
            return;
        }
        let mut tasks = tasks.peekable();
        if tasks.peek().is_none() {
            return;
        }
        let mutation = self.transient_io.as_ref().unwrap().mutation_guard();
        let scopes = super::transient_io::IoScopes::new(&self.device);
        for task in tasks {
            if self.cleanup_boundary.retired() || self.transient_io.as_ref().unwrap().retired() {
                break;
            }
            self.enqueue_task_inner(task);
        }
        // Every typed error future exists BEFORE scheduler flush/readback or
        // token acceptance. Errors from all enclosed calls share these scopes.
        scopes.finish(self.transient_io.as_mut().unwrap());
        mutation.accept();
    }

    fn enqueue_task_inner(&mut self, task: ScheduleTask) {
        match task {
            ScheduleTask::Write {
                data,
                buffer,
                private,
            } => {
                if private {
                    if self.strata_graph.is_some()
                        && buffer.buffer.usage().contains(wgpu::BufferUsages::STORAGE)
                    {
                        self.compute_pass = None;
                        let result = self.private_io_active().and_then(|()| {
                            self.initialize_native_storage_root(&buffer)?;
                            let copies = self
                                .transient_io
                                .as_mut()
                                .unwrap()
                                .native_upload(&buffer, &data)?;
                            for (source, target) in copies {
                                self.strata_graph
                                    .as_mut()
                                    .unwrap()
                                    .push_transfer(source, target)?;
                            }
                            Ok(())
                        });
                        match result {
                            Ok(()) => {
                                self.tasks_count += 1;
                                self.flush_if_needed();
                            }
                            Err(error) => {
                                self.transient_io.as_ref().unwrap().retire();
                                self.error(error);
                            }
                        }
                        return;
                    }
                    if self.strata_graph.as_ref().is_some_and(|g| g.pending()) {
                        if let Err(error) = self.flush(StreamErrorMode {
                            ignore: false,
                            flush: true,
                        }) {
                            self.error(error);
                            return;
                        }
                    }
                    self.compute_pass = None;
                    self.encoder_dirty = true;
                    let result = self.private_io_active().and_then(|()| {
                        self.transient_io.as_mut().unwrap().upload(
                            &mut self.encoder,
                            &buffer,
                            &data,
                        )
                    });
                    match result {
                        Ok(()) => {
                            self.tasks_count += 1;
                            self.flush_if_needed();
                        }
                        Err(error) => {
                            if let Some(io) = &self.transient_io {
                                io.retire();
                            }
                            self.error(error);
                        }
                    }
                    return;
                }
                // It is important to flush before writing, as the write operation is inserted
                // into the QUEUE not the encoder. We want to make sure all outstanding work
                // happens _before_ the write operation.
                let _ = self
                    .flush(StreamErrorMode {
                        ignore: true,
                        flush: false,
                    })
                    .ok();
                if let Some(graph) = self.strata_graph.as_mut() {
                    if let Err(error) = graph.complete_primary_access() {
                        self.error(error);
                        return;
                    }
                }
                self.write_to_buffer(&buffer, &data);
            }
            ScheduleTask::Execute {
                pipeline,
                count,
                resources,
            } => match resources.into_resources(self) {
                Ok((resources, reads)) => {
                    self.register_pipeline(pipeline, resources.iter(), reads, &count)
                }
                Err(error) => {
                    if let Some(io) = &self.transient_io {
                        io.retire();
                    }
                    self.error(error);
                }
            },
        }
    }

    fn initialize_native_storage_root(
        &mut self,
        resource: &WgpuResource,
    ) -> Result<(), ServerError> {
        if !self
            .strata_graph
            .as_mut()
            .unwrap()
            .initialize_resources(std::slice::from_ref(resource))
        {
            return Ok(());
        }
        if self.native_init_pipeline.is_none() {
            let module=self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label:Some("Native Strata upload ownership"),
                source:wgpu::ShaderSource::Wgsl("@group(0) @binding(0) var<storage,read> data:array<u32>; @compute @workgroup_size(1) fn main() {if arrayLength(&data)==0u {return;} workgroupBarrier();}".into()),
            });
            self.native_init_pipeline = Some(self.device.create_compute_pipeline(
                &wgpu::ComputePipelineDescriptor {
                    label: Some("Native Strata upload initialization"),
                    layout: None,
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                },
            ));
        }
        let pipeline = self.native_init_pipeline.as_ref().unwrap();
        // Bind exactly one full owning root READONLY. Wgpu initializes only
        // untouched ranges; no original/public model write is authorized.
        // Carved read/write regions share a VkBuffer and cannot appear together
        // in a WebGPU usage scope even when their actual ranges are disjoint.
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Native Strata upload root"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &resource.buffer,
                    offset: 0,
                    size: None,
                }),
            }],
        });
        self.encoder_dirty = true;
        let mut pass = self
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Native Strata zero-work upload ownership"),
                timestamp_writes: None,
            });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(0, 1, 1);
        Ok(())
    }
    fn complete_native_before_readback(&mut self) -> Result<(), ServerError> {
        if self.strata_graph.as_ref().is_some_and(|g| g.pending()) {
            self.flush(StreamErrorMode {
                ignore: false,
                flush: true,
            })?;
        }
        if let Some(graph) = self.strata_graph.as_mut() {
            graph.complete_primary_access()?;
        }
        Ok(())
    }

    /// Read multiple buffers lazily to [Bytes], potentially using pinned memory.
    ///
    /// # Arguments
    ///
    /// * `self` - The current stream.
    /// * `descriptors` - A vector of copy descriptors specifying the source data.
    ///
    /// # Returns
    ///
    /// A [Result] containing a vector of [Bytes] with the copied data, or an [`IoError`] if any copy fails.
    pub fn read_resources(
        &mut self,
        descriptors: Vec<(WgpuResource, Shape, usize)>,
    ) -> DynFut<Result<Vec<Bytes>, ServerError>> {
        if self.transient_io.is_some() {
            return self.read_private_resources(descriptors);
        }
        if let Err(error) = self.complete_native_before_readback() {
            return Box::pin(async move { Err(error) });
        }
        self.compute_pass = None;
        let mut staging_info = Vec::with_capacity(descriptors.len());
        let mut callbacks = Vec::with_capacity(descriptors.len());

        for (resource, shape, elem_size) in descriptors {
            let size = shape.iter().product::<usize>() * elem_size;

            // Zero-sized resources don't need a GPU copy.
            if resource.size == 0 {
                staging_info.push(None);
                continue;
            }

            // Copying into a buffer has to be 4 byte aligned. We can safely do so, as
            // memory is 32 bytes aligned (see WgpuStorage).
            let align = wgpu::COPY_BUFFER_ALIGNMENT;
            let aligned_len = resource.size.div_ceil(align) * align;
            let (staging, binding) = self.mem_manage.reserve_staging(aligned_len).unwrap();

            self.tasks_count += 1;
            self.encoder_dirty = true;
            self.encoder.copy_buffer_to_buffer(
                &resource.buffer,
                resource.offset,
                &staging.buffer,
                0,
                aligned_len,
            );
            staging_info.push(Some((staging, binding, size)));
        }

        // Flush all commands to the queue, so GPU gets started on copying to the staging buffer.
        let _ = self
            .flush(StreamErrorMode {
                ignore: true,
                flush: false,
            })
            .ok();

        for entry in staging_info.iter() {
            if let Some((staging, _binding, _size)) = entry {
                let (sender, receiver) = async_channel::bounded(1);
                staging
                    .buffer
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |v| {
                        // This might fail if the channel is closed (eg. the future is dropped).
                        // This is fine, just means results aren't needed anymore.
                        let _ = sender.try_send(v);
                    });

                callbacks.push(Some(receiver));
            } else {
                callbacks.push(None);
            }
        }

        let poll = self.poll.start_polling();

        Box::pin(async move {
            for receiver in callbacks.iter().flatten() {
                receiver
                    .recv()
                    .await
                    .expect("Unable to receive buffer slice result.")
                    .expect("Failed to map buffer");
            }

            // Can stop polling now.
            core::mem::drop(poll);

            let result = {
                staging_info
                    .into_iter()
                    .map(|entry| {
                        if let Some((staging, binding, size)) = entry {
                            let controller =
                                Box::new(WgpuAllocController::init(binding, staging.buffer));
                            // SAFETY: The binding has initialized memory for at least `size` bytes.
                            unsafe { Bytes::from_controller(controller, size) }
                        } else {
                            Bytes::from_bytes_vec(vec![])
                        }
                    })
                    .collect()
            };

            Ok(result)
        })
    }

    // Bit silly but needed to make the borrow checker happy.
    fn system_profiler(&mut self) -> &mut TimestampProfiler {
        let Timings::System(timing) = &mut self.timings else {
            panic!("Unexpected timings type");
        };
        timing
    }

    pub fn start_profile(&mut self) -> Result<ProfilingToken, ServerError> {
        if matches!(self.timings, Timings::System(_)) {
            cubecl_common::future::block_on(self.sync())?;
        } else {
            self.flush(StreamErrorMode {
                ignore: false,
                flush: true,
            })?;
        }

        match &mut self.timings {
            Timings::System(_) => {
                let profiler = self.system_profiler();
                Ok(profiler.start())
            }
            Timings::Device(query) => {
                self.compute_pass = None;
                let token = query.start_profile();
                Ok(token)
            }
        }
    }

    pub fn profile_error(&mut self, error: ProfileError) {
        match &mut self.timings {
            Timings::Device(profiler) => {
                profiler.error(error);
            }
            Timings::System(profiler) => {
                profiler.error(error);
            }
        }
    }

    pub fn end_profile(&mut self, token: ProfilingToken) -> Result<ProfileDuration, ProfileError> {
        match &mut self.timings {
            Timings::System(..) => {
                // Nb: WASM _has_ to use device timing and will panic here if query timestamps are not supported.
                let result = future::block_on(self.sync());
                let profiler = self.system_profiler();

                if let Err(err) = result {
                    profiler.error(ProfileError::Server(Box::new(err)));
                }
                profiler.stop(token)
            }
            Timings::Device(..) => {
                let poll = self.poll.start_polling();
                self.compute_pass = None;

                // Submit commands needed for profiling.
                self.encoder_dirty = true;
                let buffer = {
                    let Timings::Device(timing) = &mut self.timings else {
                        return Err(ProfileError::Unknown {
                            reason: "Unexpected timings type".to_string(),
                            backtrace: BackTrace::capture(),
                        });
                    };
                    timing.stop_profile_setup(token, &self.device, &mut self.encoder)?
                };

                // This flushes the queue to execute the encoder write command to write the
                // timings.
                self.tasks_count += 1;
                let result = self.flush(StreamErrorMode {
                    ignore: false,
                    flush: true,
                });

                let Timings::Device(timing) = &mut self.timings else {
                    return Err(ProfileError::Unknown {
                        reason: "Unexpected timings type".to_string(),
                        backtrace: BackTrace::capture(),
                    });
                };

                match result {
                    Ok(_) => timing.stop_profile(buffer, poll),
                    Err(err) => {
                        // Just to clean the timing buffer.
                        let _ = timing.stop_profile(buffer, poll).ok();
                        Err(ProfileError::Server(Box::new(err)))
                    }
                }
            }
        }
    }

    pub fn sync(
        &mut self,
    ) -> Pin<Box<dyn Future<Output = Result<(), ServerError>> + Send + 'static>> {
        let error_scope = self.device.push_error_scope(wgpu::ErrorFilter::Internal);

        let flush_error = self
            .flush(StreamErrorMode {
                ignore: false,
                flush: true,
            })
            .and_then(|()| {
                if let Some(graph) = self.strata_graph.as_mut() {
                    graph.complete_primary_access()?;
                }
                Ok(())
            })
            .err();

        let queue = self.queue.clone();
        let error_future = error_scope.pop();
        let poll = self.poll.start_polling();

        Box::pin(async move {
            let (sender, receiver) = async_channel::bounded::<()>(1);
            queue.on_submitted_work_done(move || {
                // Signal that we're done.
                let _ = sender.try_send(());
                core::mem::drop(poll);
            });
            let _ = receiver.recv().await;

            if let Some(error) = error_future.await {
                return Err(ServerError::Generic {
                    reason: format!("{error}"),
                    backtrace: BackTrace::capture(),
                });
            }

            match flush_error {
                Some(err) => Err(err),
                None => Ok(()),
            }
        })
    }

    pub(crate) fn configure_transient_io(&mut self, budget: u64) -> Result<(), ServerError> {
        if self.transient_io.is_some() {
            return Err(super::transient_io::fault("private I/O already configured"));
        }
        // Configure only while the serialized stream is quiescent; historical
        // PUBLIC opaque stages are outside the erasure receipt's scope.
        self.flush(StreamErrorMode {
            ignore: false,
            flush: true,
        })?;
        let io =
            super::transient_io::TransientIo::new(self.device.clone(), self.queue.clone(), budget)?;
        io.drain_submitted()?;
        self.transient_io = Some(io);
        Ok(())
    }

    pub(crate) fn configure_public_target_request_io_with_backlog(
        &mut self,
        backlog: cubecl_runtime::server::PublicTargetSubmissionBacklog,
    ) -> Result<(), ServerError> {
        let inline_device =
            cubecl_common::device_handle::DeviceHandle::<super::server::WgpuServer>::is_blocking();
        validate_public_target_request_io_configuration(
            backlog,
            self.tasks_max,
            inline_device,
            self.cleanup_boundary.retired(),
            self.transient_io.is_some(),
        )?;
        self.flush(StreamErrorMode {
            ignore: false,
            flush: true,
        })?;
        let io = super::transient_io::TransientIo::new_public_target(
            self.device.clone(),
            self.queue.clone(),
            inline_device,
        )?;
        io.drain_submitted()?;
        self.transient_io = Some(io);
        // Prior setup work is drained above. No live request, owner reuse,
        // ledger or completion state is reset by this construction selection.
        self.submission_load.configure_public_target(backlog);
        Ok(())
    }

    pub(crate) fn public_target_submission_backlog(
        &self,
    ) -> Result<cubecl_runtime::server::PublicTargetSubmissionBacklog, ServerError> {
        if !self.public_target_request_io_configured()? {
            return Err(super::transient_io::fault(
                "public target I/O not configured",
            ));
        }
        Ok(self.submission_load.policy())
    }

    pub(crate) fn public_target_request_io_configured(&self) -> Result<bool, ServerError> {
        if self.cleanup_boundary.retired() {
            return Err(super::transient_io::fault("private I/O stream retired"));
        }
        self.transient_io
            .as_ref()
            .map_or(Ok(false), |io| io.public_target_configured())
    }

    pub(crate) fn begin_transient_configure(
        &self,
    ) -> Result<super::cleanup_boundary::CleanupGuard, ServerError> {
        if self.cleanup_boundary.retired() {
            return Err(super::transient_io::fault("private I/O stream retired"));
        }
        if self.transient_io.is_some() {
            return Err(super::transient_io::fault("private I/O already configured"));
        }
        self.cleanup_boundary.begin(0)
    }

    pub(crate) fn begin_transient_cleanup(
        &self,
    ) -> Result<super::cleanup_boundary::CleanupGuard, ServerError> {
        let io = self
            .transient_io
            .as_ref()
            .ok_or_else(|| super::transient_io::fault("private I/O not configured"))?;
        self.cleanup_boundary.begin(io.leases())
    }

    pub(crate) fn transient_io_enabled(&self) -> bool {
        self.transient_io.is_some()
    }

    pub(crate) fn private_mutation_guard(
        &self,
    ) -> Option<super::transient_io::PrivateMutationGuard> {
        self.transient_io.as_ref().map(|io| io.mutation_guard())
    }

    pub(crate) fn supports_physical_reads(&self) -> bool {
        self.strata_graph.is_some()
    }

    pub(crate) fn private_io_active(&self) -> Result<(), ServerError> {
        if self.cleanup_boundary.retired() {
            return Err(super::transient_io::fault("private I/O stream retired"));
        }
        self.transient_io
            .as_ref()
            .ok_or_else(|| super::transient_io::fault("private I/O not configured"))?
            .active()
    }

    pub(crate) fn clear_private_owners(
        &mut self,
        resources: &[cubecl_runtime::storage::ManagedResource<crate::WgpuResource>],
    ) -> Result<(), ServerError> {
        self.private_io_active()?;
        let mutation = self.private_mutation_guard();
        // All extents/aliases are checked before mutating the encoder. Clear
        // only the complete retained owning resource, never allocation gaps.
        for (i, managed) in resources.iter().enumerate() {
            let r = managed.resource();
            let end = r
                .offset
                .checked_add(r.size)
                .ok_or_else(|| super::transient_io::fault("private clear extent overflow"))?;
            if r.size == 0
                || !r.offset.is_multiple_of(4)
                || !r.size.is_multiple_of(4)
                || end > r.buffer.size()
                || !r.buffer.usage().contains(wgpu::BufferUsages::COPY_DST)
            {
                return Err(super::transient_io::fault("private clear physical extent"));
            }
            for other in &resources[..i] {
                let other = other.resource();
                if r.buffer == other.buffer
                    && r.offset < other.offset + other.size
                    && other.offset < end
                {
                    return Err(super::transient_io::fault(
                        "private clear overlapping owners",
                    ));
                }
            }
        }
        let scopes = super::transient_io::IoScopes::new(&self.device);
        self.compute_pass = None;
        for managed in resources {
            let r = managed.resource();
            if self.strata_graph.is_some() && r.buffer.usage().contains(wgpu::BufferUsages::STORAGE)
            {
                self.initialize_native_storage_root(r)?;
                self.strata_graph
                    .as_mut()
                    .unwrap()
                    .push_transfer(None, r.clone())?;
            } else {
                self.complete_native_before_readback()?;
                self.encoder_dirty = true;
                self.encoder.clear_buffer(&r.buffer, r.offset, Some(r.size));
            }
            self.tasks_count += 1;
        }
        scopes.finish(self.transient_io.as_mut().unwrap());
        self.flush_if_needed();
        if let Some(mutation) = mutation {
            mutation.accept();
        }
        Ok(())
    }

    #[allow(clippy::type_complexity)] // Retained source/target owners plus their model-cell extents.
    pub(crate) fn stage_public_host_copies(
        &mut self,
        copies: &[(
            cubecl_runtime::storage::ManagedResource<crate::WgpuResource>,
            u64,
            cubecl_runtime::storage::ManagedResource<crate::WgpuResource>,
            u64,
            u64,
        )],
    ) -> Result<(), ServerError> {
        self.private_io_active()?;
        // Validate every physical range before changing the encoder or graph.
        for (i, (source, sfirst, target, tfirst, bytes)) in copies.iter().enumerate() {
            let s = source.resource();
            let t = target.resource();
            if s.offset != 0
                || t.offset != 0
                || s.size != s.buffer.size()
                || t.size != t.buffer.size()
                || s.buffer == t.buffer
                || !s.buffer.usage().contains(wgpu::BufferUsages::COPY_SRC)
                || !t.buffer.usage().contains(wgpu::BufferUsages::COPY_DST)
                || sfirst.checked_add(*bytes).is_none_or(|end| end > s.size)
                || tfirst.checked_add(*bytes).is_none_or(|end| end > t.size)
            {
                return Err(super::transient_io::fault(
                    "public host copy physical owner extent",
                ));
            }
            for (_, _, earlier, efirst, eb) in &copies[..i] {
                if earlier.resource().buffer == t.buffer
                    && *tfirst < *efirst + *eb
                    && *efirst < *tfirst + *bytes
                {
                    return Err(super::transient_io::fault(
                        "public host copy overlapping destinations",
                    ));
                }
            }
        }
        let scopes = super::transient_io::IoScopes::new(&self.device);
        self.compute_pass = None;
        for (source, sfirst, target, tfirst, bytes) in copies {
            let s = source.resource();
            let t = target.resource();
            if let Some(graph) = self.strata_graph.as_mut() {
                // Actual vkCmdCopyBuffer nodes retain public/private roots and
                // get the graph's transfer<->shader dependency barriers.
                let source = crate::WgpuResource {
                    buffer: s.buffer.clone(),
                    offset: *sfirst,
                    size: *bytes,
                };
                let target = crate::WgpuResource {
                    buffer: t.buffer.clone(),
                    offset: *tfirst,
                    size: *bytes,
                };
                graph.push_transfer(Some(source), target)?;
            } else {
                self.encoder_dirty = true;
                self.encoder
                    .copy_buffer_to_buffer(&s.buffer, *sfirst, &t.buffer, *tfirst, *bytes);
            }
            self.tasks_count += 1;
        }
        scopes.finish(self.transient_io.as_mut().unwrap());
        self.flush_if_needed();
        Ok(())
    }

    pub(crate) fn begin_transient_io_reuse(&mut self) -> Result<u64, ServerError> {
        self.validate_transient_io_reuse_mode(false)?;
        self.private_io_active()?;
        self.flush(StreamErrorMode {
            ignore: false,
            flush: true,
        })?;
        self.transient_io.as_mut().unwrap().begin_reuse()
    }

    pub(crate) fn validate_transient_io_reuse_mode(
        &self,
        public_target: bool,
    ) -> Result<(), ServerError> {
        self.private_io_active()?;
        self.transient_io
            .as_ref()
            .unwrap()
            .validate_reuse_mode(public_target)
    }

    pub(crate) fn begin_public_target_request_io_reuse(&mut self) -> Result<u64, ServerError> {
        self.validate_transient_io_reuse_mode(true)?;
        self.flush(StreamErrorMode {
            ignore: false,
            flush: true,
        })?;
        self.transient_io
            .as_mut()
            .unwrap()
            .begin_public_target_reuse()
    }

    pub(crate) fn recycle_transient_io(&mut self, nonce: u64) -> Result<(), ServerError> {
        self.private_io_active()?;
        self.transient_io.as_ref().unwrap().check_reuse(nonce)?;
        self.flush(StreamErrorMode {
            ignore: false,
            flush: true,
        })?;
        if let Some(graph) = self.strata_graph.as_mut() {
            // SAFETY: original immutable owners are lent directly to this
            // stream's real Graph proof, then applied without enqueue/mutation
            // interleaving. The cohort excludes every retained named/live use;
            // only its actually admitted source may receive a native writer.
            unsafe {
                self.transient_io
                    .as_mut()
                    .unwrap()
                    .recycle_native_selected(nonce, |owners| graph.certify_io_reuse(owners))
            }
        } else {
            self.transient_io.as_mut().unwrap().recycle_reuse(nonce)
        }
    }

    pub(crate) fn accept_transient_io_reuse(&mut self, nonce: u64) -> Result<(), ServerError> {
        self.private_io_active()?;
        self.transient_io.as_ref().unwrap().check_reuse(nonce)?;
        self.flush(StreamErrorMode {
            ignore: false,
            flush: true,
        })?;
        self.transient_io.as_mut().unwrap().accept_reuse(nonce)
    }

    pub(crate) fn end_transient_io_reuse(
        &mut self,
        nonce: u64,
    ) -> Result<cubecl_runtime::transient::TransientIoReceipt, ServerError> {
        self.transient_io
            .as_ref()
            .ok_or_else(|| super::transient_io::fault("private I/O not configured"))?
            .check_reuse(nonce)?;
        let receipt = self.sanitize_transient_io(true)?;
        self.transient_io.as_mut().unwrap().finish_reuse(nonce)?;
        Ok(receipt)
    }

    pub(crate) fn sanitize_transient_io(
        &mut self,
        inspect: bool,
    ) -> Result<cubecl_runtime::transient::TransientIoReceipt, ServerError> {
        let io = self
            .transient_io
            .as_ref()
            .ok_or_else(|| super::transient_io::fault("private I/O not configured"))?;
        if io.leases() != 0 {
            return Err(super::transient_io::held_lease_fault());
        }
        // Deliberately synchronous at the server boundary, so dropping the
        // returned future cannot abandon cleanup midway or unlock the stream.
        // Flush the unsubmitted encoder, then the native eraser drains with
        // a checked 30-second poll. The ordinary sync callback is unbounded.
        let previous_error = self
            .flush(StreamErrorMode {
                ignore: false,
                flush: true,
            })
            .is_err()
            || self.cleanup_boundary.retired();
        if let Some(graph) = &mut self.strata_graph {
            graph.complete_primary_access()?;
            graph.clear_completed_recordings()?;
        }
        self.transient_io
            .as_mut()
            .unwrap()
            .sanitize_after_drain(previous_error, inspect)
    }

    /// Cleanup boundary only; no background ordinary/inference poll is used.
    pub(crate) fn read_private_cleanup_resource(
        &mut self,
        resource: WgpuResource,
    ) -> Result<Bytes, ServerError> {
        let timeout = self.cleanup_boundary.cleanup_timeout().ok_or_else(|| {
            super::transient_io::fault("private cleanup read outside cleanup boundary")
        })?;
        self.private_io_active()?;
        let mutation = self.transient_io.as_ref().unwrap().mutation_guard();
        self.compute_pass = None;
        let scopes = super::transient_io::IoScopes::new(&self.device);
        let setup: Result<_, ServerError> = (|| {
            self.complete_native_before_readback()?;
            let logical = usize::try_from(resource.size)
                .map_err(|_| super::transient_io::fault("private cleanup read span overflow"))?;
            self.encoder_dirty = true;
            let entry = self.transient_io.as_mut().unwrap().readback(
                &mut self.encoder,
                &resource,
                logical,
            )?;
            self.tasks_count += 2;
            self.flush(StreamErrorMode {
                ignore: false,
                flush: true,
            })?;
            Ok(entry)
        })();
        scopes.finish(self.transient_io.as_mut().unwrap());
        let entry = setup?;
        let scopes = super::transient_io::IoScopes::new(&self.device);
        let mapped = (|| {
            let deadline = std::time::Instant::now() + timeout;
            let (sender, receiver) = std::sync::mpsc::sync_channel(1);
            // entry's ReadbackLease already guards pending-map cancellation/unmap.
            entry.buffer.map_async(wgpu::MapMode::Read, .., move |r| {
                let _ = sender.try_send(r);
            });
            if let Err(error) = self
                .transient_io
                .as_ref()
                .unwrap()
                .await_cleanup_read_map(&receiver, deadline)
            {
                // Specific callback/drain code precedes generic ReadbackMap
                // retirement. Lease Drop still aborts/unmaps inside scopes.
                entry.fail();
                return Err(error);
            }
            entry.into_bytes()
        })();
        scopes.finish(self.transient_io.as_mut().unwrap());
        let bytes = mapped?;
        mutation.accept();
        Ok(bytes)
    }

    /// Selected native graph's real packet copy and synchronous HAL host map.
    /// Keep original I/O leases and wiping output controllers; no Wgpu primary
    /// copy/map callback/poll may serialize this observation behind MAIN work.
    /// Actual scoped admission and original allocator/I/O retention are still
    /// subject to full physical/numerical acceptance; no overlap is claimed.
    fn read_native_private_resources(
        &mut self,
        descriptors: Vec<(WgpuResource, Shape, usize)>,
    ) -> DynFut<Result<Vec<Bytes>, ServerError>> {
        let scoped = self.transient_io.as_ref().unwrap().reuse_active();
        let mut setup_timer = super::transient_io::ScopedReadTimer::new(scoped, 0);
        let mutation = self.transient_io.as_ref().unwrap().mutation_guard();
        self.compute_pass = None;
        let scopes = super::transient_io::IoScopes::new(&self.device);
        let mut pending = Vec::new();
        let setup = (|| {
            self.private_io_active()?;
            // Preserve Wgpu's real cold source initialization before raw
            // native copies. Actual inference roots are STORAGE/COPY_SRC;
            // external/private I/O staging owners are not exposed as sources.
            // Original verified R is already initialized before its scope pins
            // it, so this cannot clear a protected live source behind its back.
            for (source, _, _) in &descriptors {
                if source.size != 0 {
                    if !source
                        .buffer
                        .usage()
                        .contains(wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC)
                    {
                        return Err(super::transient_io::fault(
                            "native readback original inference source usage",
                        ));
                    }
                    self.initialize_native_storage_root(source)?;
                }
            }
            // Finish original queued/current-lane producers. Actual fresh
            // primary initialization still forces a full drain; warm native
            // observation does not join unrelated target commit work.
            self.flush(StreamErrorMode {
                ignore: false,
                flush: true,
            })?;
            pending
                .try_reserve_exact(descriptors.len())
                .map_err(|_| super::transient_io::fault("native readback metadata allocation"))?;
            for (resource, shape, elem_size) in descriptors {
                let logical = shape
                    .iter()
                    .try_fold(elem_size, |n, s| n.checked_mul(*s))
                    .ok_or_else(|| super::transient_io::fault("native readback shape overflow"))?;
                if resource.size == 0 {
                    if logical != 0 {
                        return Err(super::transient_io::fault("native empty readback mismatch"));
                    }
                    pending.push(None);
                    continue;
                }
                let (entry, transfers) = self
                    .transient_io
                    .as_mut()
                    .unwrap()
                    .native_readback(&resource, logical)?;
                // Retain the original lease BEFORE either enqueue can fail.
                pending.push(Some(entry));
                for (source, target) in transfers {
                    self.strata_graph
                        .as_mut()
                        .unwrap()
                        .push_transfer(source, target)?;
                    self.tasks_count += 1;
                }
            }
            setup_timer.finish();
            let _flush_timer = super::transient_io::ScopedReadTimer::new(scoped, 1);
            if self.encoder_dirty
                || self.pending_write_count != 0
                || self.transient_io.as_ref().unwrap().has_pending_info()
            {
                return Err(super::transient_io::fault(
                    "native readback has unstaged primary/INFO work",
                ));
            }
            self.strata_graph.as_mut().unwrap().execute_readback()?;
            self.tasks_count = 0;
            self.mem_manage.memory_cleanup(false);
            self.mem_manage.release_uniforms();
            self.flush_errors(StreamErrorMode {
                ignore: false,
                flush: true,
            })?;
            Ok(())
        })();
        scopes.finish(self.transient_io.as_mut().unwrap());
        if let Err(error) = setup {
            self.transient_io.as_ref().unwrap().retire();
            return Box::pin(async move { Err(error) });
        }
        let mut map_timer = super::transient_io::ScopedReadTimer::new(scoped, 2);
        let scopes = super::transient_io::IoScopes::new(&self.device);
        // One deadline for the whole original host-map batch, never one fresh
        // window per packet. The actual GPU fence completed in flush above.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        let mapped = (|| {
            let mut result = Vec::new();
            result
                .try_reserve_exact(pending.len())
                .map_err(|_| super::transient_io::fault("native readback result allocation"))?;
            for entry in pending {
                result.push(match entry {
                    // SAFETY: the exact original fill/copy commands, including
                    // all transfer segments, joined on Graph's real lane fence.
                    // Full initialization/HOST_READ barriers are recorded there;
                    // the original lease prevents reuse or erase during mapping.
                    Some(entry) => unsafe { entry.into_joined_bytes(&self.device, deadline)? },
                    None => Bytes::from_bytes_vec(Vec::new()),
                });
            }
            Ok(result)
        })();
        scopes.finish(self.transient_io.as_mut().unwrap());
        map_timer.finish();
        if mapped.is_err() {
            self.transient_io.as_ref().unwrap().retire();
        } else {
            mutation.accept();
        }
        Box::pin(async move { mapped })
    }
    fn read_private_resources(
        &mut self,
        descriptors: Vec<(WgpuResource, Shape, usize)>,
    ) -> DynFut<Result<Vec<Bytes>, ServerError>> {
        if self.strata_graph.is_some() {
            return self.read_native_private_resources(descriptors);
        }
        let scoped = self.transient_io.as_ref().unwrap().reuse_active();
        let mut setup_timer = super::transient_io::ScopedReadTimer::new(scoped, 0);
        let mutation = self.transient_io.as_ref().unwrap().mutation_guard();
        self.compute_pass = None;
        let scopes = super::transient_io::IoScopes::new(&self.device);
        let mut pending = Vec::new();
        let setup = (|| {
            self.private_io_active()?;
            self.complete_native_before_readback()?;
            pending
                .try_reserve_exact(descriptors.len())
                .map_err(|_| super::transient_io::fault("private readback metadata allocation"))?;
            for (resource, shape, elem_size) in descriptors {
                let logical = shape
                    .iter()
                    .try_fold(elem_size, |n, s| n.checked_mul(*s))
                    .ok_or_else(|| super::transient_io::fault("private readback shape overflow"))?;
                if resource.size == 0 {
                    if logical != 0 {
                        return Err(super::transient_io::fault(
                            "private readback empty resource mismatch",
                        ));
                    }
                    pending.push(None);
                    continue;
                }
                self.encoder_dirty = true;
                let entry = self.transient_io.as_mut().unwrap().readback(
                    &mut self.encoder,
                    &resource,
                    logical,
                )?;
                self.tasks_count += 2;
                pending.push(Some(entry));
            }
            setup_timer.finish();
            let _flush_timer = super::transient_io::ScopedReadTimer::new(scoped, 1);
            self.flush(StreamErrorMode {
                ignore: false,
                flush: true,
            })?;
            Ok(())
        })();
        scopes.finish(self.transient_io.as_mut().unwrap());
        if setup.is_err() {
            self.transient_io.as_ref().unwrap().retire();
            return Box::pin(async {
                Err(super::transient_io::fault("private readback setup failed"))
            });
        }
        let mut map_timer = super::transient_io::ScopedReadTimer::new(scoped, 2);
        let scopes = super::transient_io::IoScopes::new(&self.device);
        let inline_readback = self.transient_io.as_ref().unwrap().reads_inline();
        // One original bounded native map window starts before any request.
        let map_deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        let mut callbacks = Vec::with_capacity(pending.len());
        for entry in &pending {
            if let Some(entry) = entry {
                let (sender, receiver) = async_channel::bounded(1);
                entry
                    .buffer
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |r| {
                        let _ = sender.try_send(r);
                    });
                callbacks.push(Some(receiver));
            } else {
                callbacks.push(None);
            }
        }
        scopes.finish(self.transient_io.as_mut().unwrap());
        let poll = if inline_readback {
            if let Err(error) = self
                .transient_io
                .as_ref()
                .unwrap()
                .await_inline_read_maps(map_deadline)
            {
                for entry in pending.iter().flatten() {
                    entry.fail();
                }
                return Box::pin(async move { Err(error) });
            }
            // Completed native polls process map callbacks synchronously.
            // Refuse absent/rejected delivery rather than starting a new window.
            for (entry, callback) in pending.iter().zip(&callbacks) {
                if let (Some(entry), Some(callback)) = (entry, callback)
                    && !matches!(callback.try_recv(), Ok(Ok(())))
                {
                    entry.fail();
                    return Box::pin(async {
                        Err(super::transient_io::fault(
                            "inline private readback callback failed",
                        ))
                    });
                }
            }
            self.transient_io
                .as_ref()
                .unwrap()
                .record_inline_read_batch();
            None
        } else {
            Some(self.poll.start_polling())
        };
        map_timer.finish();
        mutation.accept();
        Box::pin(async move {
            if !inline_readback {
                for (entry, callback) in pending.iter().zip(&callbacks) {
                    if let (Some(entry), Some(callback)) = (entry, callback) {
                        if !matches!(callback.recv().await, Ok(Ok(()))) {
                            entry.fail();
                            return Err(super::transient_io::fault(
                                "private readback mapping failed",
                            ));
                        }
                    }
                }
            }
            drop(poll);
            let mut result = Vec::new();
            if result.try_reserve_exact(pending.len()).is_err() {
                for entry in pending.iter().flatten() {
                    entry.fail();
                }
                return Err(super::transient_io::fault(
                    "private readback result allocation",
                ));
            }
            for entry in pending {
                result.push(match entry {
                    Some(entry) => entry.into_bytes()?,
                    None => Bytes::from_bytes_vec(Vec::new()),
                });
            }
            Ok(result)
        })
    }

    /// Allocates a new empty buffer using the main memory pool.
    pub fn empty(&mut self, size: u64) -> Result<ManagedMemoryHandle, IoError> {
        self.mem_manage.reserve(size)
    }

    /// Registers a new error into the error sink.
    pub fn error(&mut self, error: ServerError) {
        if super::strata_graph::enabled()
            && std::env::var("KUNO_LAB_LANGUAGE_PUBLIC_ONLY").as_deref() == Ok("1")
        {
            // Fixed public completion only. Never print shader/backend error
            // descriptions, request data or arbitrary panic payloads.
            let class = match &error {
                ServerError::Launch(LaunchError::CompilationError(_)) => "launch_compile",
                ServerError::Launch(LaunchError::OutOfMemory { .. }) => "launch_oom",
                ServerError::Launch(LaunchError::TooManyResources(_)) => "launch_limits",
                ServerError::Launch(_) => "launch_other",
                ServerError::Io(_) => "io",
                ServerError::Profile(_) => "profile",
                ServerError::Generic { .. } => "generic",
                ServerError::ServerUnhealthy { .. } => "unhealthy",
            };
            eprintln!("native_public_stream_fault class={class}");
        }
        self.errors.push(error);
    }

    pub(crate) fn create_uniform(
        &mut self,
        data: &[u8],
        native: &[cubecl_runtime::server::NativeStrataParameter],
    ) -> Result<WgpuResource, ServerError> {
        if self.transient_io.is_some() {
            if self.transient_io.as_ref().unwrap().batches_info() {
                return self
                    .transient_io
                    .as_mut()
                    .unwrap()
                    .batched_uniform(data, native);
            }
            self.compute_pass = None;
            self.encoder_dirty = true;
            let resource = self
                .transient_io
                .as_mut()
                .unwrap()
                .uniform(&mut self.encoder, data)?;
            self.tasks_count += 2;
            Ok(resource)
        } else {
            let resource = self.mem_manage.reserve_uniform(data.len() as u64);
            self.write_to_buffer(&resource, data);
            Ok(resource)
        }
    }

    // Nb: this function submits a command to the _queue_ not to the encoder,
    // so you have to be really careful about the ordering of operations here.
    // Any buffer which has outstanding (not yet flushed) compute work should
    // NOT be copied to.
    fn write_to_buffer(&mut self, resource: &WgpuResource, data: &[u8]) {
        // Nothing to write for zero-sized resources.
        if resource.size == 0 {
            return;
        }

        // Copying into a buffer has to be 4 byte aligned. We can safely do so, as
        // memory is also aligned (see WgpuStorage). Per the WebGPU spec, this
        // just has to be a multiple of 4: https://www.w3.org/TR/webgpu/#dom-gpuqueue-writebuffer
        let copy_align = wgpu::COPY_BUFFER_ALIGNMENT;
        let size = resource.size.next_multiple_of(copy_align);

        if size == data.len() as u64 {
            // write_buffer is the recommended way to write this data, as:
            // - On WebGPU, from WASM, this can save a copy to the JS memory.
            // - On devices with unified memory, this could skip the staging buffer entirely.
            self.queue
                .write_buffer(&resource.buffer, resource.offset, data);
        } else {
            // For sizes not aligned we need to only write a part of the staging buffer, do this
            // with `write_buffer_with`.
            let mut buffer = self
                .queue
                .write_buffer_with(
                    &resource.buffer,
                    resource.offset,
                    NonZero::new(size).unwrap(),
                )
                .expect("Internal error: Failed to call `write_buffer_with`, this likely means no staging buffer could be allocated.");
            buffer.slice(0..data.len()).copy_from_slice(data);
        }

        self.pending_write_count += 1;

        // Prevent wgpu staging buffer pool exhaustion during bulk writes (e.g. model
        // loading with hundreds of tensors). queue.write_buffer() is async — wgpu
        // copies data into an internal staging buffer, then transfers to GPU on the
        // next queue.submit(). Without periodic submits, hundreds of writes accumulate
        // and staging buffers get recycled before the GPU copy completes, silently
        // corrupting early tensors.
        // See: https://github.com/tracel-ai/cubecl/issues/1120
        const MAX_PENDING_WRITES: usize = 64;

        if self.pending_write_count >= MAX_PENDING_WRITES {
            // Submit a fresh, empty command buffer to flush all pending write_buffer work.
            // wgpu flushes its internal staging-buffer copies on any queue.submit(),
            // so we don't need to touch the main compute encoder here.
            let write_flush_encoder =
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("CubeCL Write Flush Encoder"),
                    });
            let index = self.queue.submit([write_flush_encoder.finish()]);

            // Wait for the GPU to finish processing these writes before continuing.
            #[cfg(not(target_family = "wasm"))]
            if let Err(e) = self.device.poll(wgpu::PollType::Wait {
                submission_index: Some(index),
                timeout: self.cleanup_boundary.cleanup_timeout(),
            }) {
                if self.cleanup_boundary.cleanup_timeout().is_some() {
                    self.cleanup_boundary.retire();
                    self.error(super::transient_io::fault(
                        "private I/O bounded public-write drain failed",
                    ));
                } else {
                    log::warn!("wgpu: write flush poll failed ({e})");
                }
            }

            self.pending_write_count = 0;
        }
    }

    fn flush_if_needed(&mut self) {
        // Flush when there are too many tasks, or when too many handles are locked.
        // Locked handles should only accumulate in rare circumstances (where uniforms
        // are being created but no work is submitted).
        if self.tasks_count >= self.tasks_max {
            let _ = self
                .flush(StreamErrorMode {
                    ignore: true,
                    flush: false,
                })
                .ok();
        }
    }

    pub fn flush(&mut self, mode: StreamErrorMode) -> Result<(), ServerError> {
        self.flush_native_mode(mode, true)
    }
    pub(crate) fn begin_native_strata_branch(
        &mut self,
        key: cubecl_runtime::server::NativeStrataBranch,
        invocation: Option<cubecl_runtime::server::NativeStrataInvocation>,
    ) -> Result<(), ServerError> {
        self.private_io_active()?;
        if self.strata_graph.is_none() {
            return Err(super::transient_io::fault(
                "native branch requires selected graph",
            ));
        }
        // No parameter owner may be rewritten while its old recording is live.
        self.flush(StreamErrorMode {
            ignore: false,
            flush: true,
        })?;
        self.strata_graph
            .as_mut()
            .unwrap()
            .select_branch_queue(Some(key))?;
        self.transient_io
            .as_mut()
            .unwrap()
            .begin_native_branch(key, invocation)
    }
    pub(crate) fn end_native_strata_branch(
        &mut self,
        key: cubecl_runtime::server::NativeStrataBranch,
    ) -> Result<(), ServerError> {
        self.private_io_active()?;
        self.flush(StreamErrorMode {
            ignore: false,
            flush: true,
        })?;
        self.transient_io.as_mut().unwrap().end_native_branch(key)?;
        self.strata_graph
            .as_mut()
            .unwrap()
            .select_branch_queue(None)
    }
    pub(crate) fn set_native_strata_phase(
        &mut self,
        phase: cubecl_runtime::server::NativeStrataPhase,
    ) -> Result<(), ServerError> {
        self.private_io_active()?;
        self.transient_io.as_mut().unwrap().set_native_phase(phase)
    }
    pub(crate) fn replay_native_strata_branch(
        &mut self,
        key: cubecl_runtime::server::NativeStrataBranch,
        invocation: cubecl_runtime::server::NativeStrataInvocation,
    ) -> Result<bool, ServerError> {
        match self.replay_native_mode(key, invocation, true)? {
            NativeReplay::Missing => Ok(false),
            NativeReplay::Joined => Ok(true),
            NativeReplay::Pending(_) => {
                Err(super::transient_io::fault("joined replay returned pending"))
            }
        }
    }
    pub(crate) fn replay_native_strata_commit_pending(
        &mut self,
        key: cubecl_runtime::server::NativeStrataBranch,
        invocation: cubecl_runtime::server::NativeStrataInvocation,
    ) -> Result<Option<cubecl_runtime::server::NativeStrataSubmission>, ServerError> {
        if key.commit_geometry().is_none() {
            return Err(super::transient_io::fault(
                "pending replay requires target commit",
            ));
        }
        match self.replay_native_mode(key, invocation, false)? {
            NativeReplay::Missing => Ok(None),
            NativeReplay::Pending(token) => Ok(Some(token)),
            NativeReplay::Joined => {
                Err(super::transient_io::fault("pending replay returned joined"))
            }
        }
    }
    fn replay_native_mode(
        &mut self,
        key: cubecl_runtime::server::NativeStrataBranch,
        invocation: cubecl_runtime::server::NativeStrataInvocation,
        wait: bool,
    ) -> Result<NativeReplay, ServerError> {
        self.private_io_active()?;
        invocation
            .validate_for(key)
            .map_err(super::transient_io::fault)?;
        if self.strata_graph.is_none() {
            return Ok(NativeReplay::Missing);
        }
        if !wait {
            self.strata_graph
                .as_ref()
                .unwrap()
                .preflight_pending_commit(key)?;
        }
        // Input gathers, clears and prepared PLE uploads precede the retained
        // recording. Join them before touching its original INFO pages.
        self.flush(StreamErrorMode {
            ignore: false,
            flush: true,
        })?;
        let Some(recording) = self.transient_io.as_ref().unwrap().native_replay_key(key)? else {
            return Ok(NativeReplay::Missing);
        };
        self.strata_graph
            .as_mut()
            .unwrap()
            .select_branch_queue(Some(key))?;
        if !self.strata_graph.as_ref().unwrap().has_recording(recording) {
            self.strata_graph
                .as_mut()
                .unwrap()
                .select_branch_queue(None)?;
            return Ok(NativeReplay::Missing);
        }
        let mutation = self.private_mutation_guard();
        let scopes = super::transient_io::IoScopes::new(&self.device);
        let result = (|| {
            let io = self.transient_io.as_mut().unwrap();
            io.stage_native_replay(recording, invocation)?;
            if io.native_branch_recording()? != Some(recording) {
                return Err(super::transient_io::fault(
                    "native replay parameter identity",
                ));
            }
            let parameters = io.prepare_native_info()?;
            if parameters.is_empty() {
                return Err(super::transient_io::fault(
                    "native replay parameter upload missing",
                ));
            }
            self.strata_graph
                .as_mut()
                .unwrap()
                .upload_parameters(parameters)?;
            self.strata_graph
                .as_mut()
                .unwrap()
                .replay_recording(recording, wait, false)?;
            let io = self.transient_io.as_mut().unwrap();
            io.native_branch_submitted()?;
            io.end_native_branch(key)?;
            if wait {
                self.strata_graph
                    .as_mut()
                    .unwrap()
                    .select_branch_queue(None)?;
                self.mem_manage.memory_cleanup(false);
                self.mem_manage.release_uniforms();
                self.flush_errors(StreamErrorMode {
                    ignore: false,
                    flush: true,
                })?;
                Ok(NativeReplay::Joined)
            } else {
                let scope = self
                    .transient_io
                    .as_ref()
                    .unwrap()
                    .native_submission_scope()?;
                let token = self
                    .strata_graph
                    .as_mut()
                    .unwrap()
                    .export_pending_commit(key, scope)?;
                Ok(NativeReplay::Pending(token))
            }
        })();
        scopes.finish(self.transient_io.as_mut().unwrap());
        if result.is_err() {
            self.transient_io
                .as_mut()
                .unwrap()
                .observe_public_submission_failure();
        } else if let Some(mutation) = mutation {
            mutation.accept();
        }
        result
    }
    pub(crate) fn end_native_strata_commit_pending(
        &mut self,
        key: cubecl_runtime::server::NativeStrataBranch,
    ) -> Result<cubecl_runtime::server::NativeStrataSubmission, ServerError> {
        self.private_io_active()?;
        if key.commit_geometry().is_none() || self.strata_graph.is_none() {
            return Err(super::transient_io::fault(
                "pending capture requires native target commit",
            ));
        }
        self.strata_graph
            .as_ref()
            .unwrap()
            .preflight_pending_commit(key)?;
        self.flush_native_mode(
            StreamErrorMode {
                ignore: false,
                flush: true,
            },
            false,
        )?;
        let io = self.transient_io.as_mut().unwrap();
        io.end_native_branch(key)?;
        let scope = io.native_submission_scope()?;
        self.strata_graph
            .as_mut()
            .unwrap()
            .export_pending_commit(key, scope)
    }
    pub(crate) fn join_native_strata_submission(
        &mut self,
        token: cubecl_runtime::server::NativeStrataSubmission,
    ) -> Result<(), ServerError> {
        self.private_io_active()?;
        let scope = self
            .transient_io
            .as_ref()
            .unwrap()
            .native_submission_scope()?;
        self.strata_graph
            .as_mut()
            .ok_or_else(|| super::transient_io::fault("pending original graph missing"))?
            .join_exported(token, scope)?;
        // Do not release unrelated live INFO/recording resources. The original
        // request's fault/readback/cleanup consumers retain those obligations.
        self.flush_errors(StreamErrorMode {
            ignore: false,
            flush: true,
        })
    }
    pub(crate) fn prepare_native_strata_draft_scope(
        &mut self,
        source: cubecl_runtime::storage::ManagedResource<WgpuResource>,
    ) -> Result<cubecl_runtime::server::NativeStrataDraftScope, ServerError> {
        self.private_io_active()?;
        if self
            .transient_io
            .as_mut()
            .unwrap()
            .native_branch_recording()?
            .is_some()
        {
            return Err(super::transient_io::fault(
                "draft preparation inside named branch",
            ));
        }
        // Actual verified rows/fault have already been observed by the owner.
        // Drain original staged work before establishing the precommit signal.
        self.flush(StreamErrorMode {
            ignore: false,
            flush: true,
        })?;
        let request = self
            .transient_io
            .as_mut()
            .unwrap()
            .native_submission_scope()?;
        if self.strata_graph.is_none() {
            return Err(super::transient_io::fault(
                "draft requires original native graph",
            ));
        }
        self.mem_manage
            .begin_native_pins(request)
            .map_err(ServerError::Io)?;
        self.strata_graph
            .as_mut()
            .unwrap()
            .prepare_draft_scope(request, source)
    }
    pub(crate) fn native_strata_shared_boundary(
        &mut self,
        boundary: cubecl_runtime::server::NativeStrataSharedBoundary,
    ) -> Result<(), ServerError> {
        self.private_io_active()?;
        let io = self.transient_io.as_mut().unwrap();
        let request = io.native_submission_scope()?;
        // Prompt-final count1 is intentionally uncaptured; its authentic
        // InitialWindow scope supplies the nonce. Later captured target windows
        // must still match their actual branch/request. Graph checks the real
        // MAIN lane and absence of an active MTP draft scope independently.
        let recording = io
            .native_branch_recording()?
            .map(|key| (key.nonce, key.branch));
        super::strata_schedule::shared_recording(request, recording)
            .map_err(super::transient_io::fault)?;
        self.strata_graph
            .as_mut()
            .ok_or_else(|| super::transient_io::fault("shared original graph missing"))?
            .shared_boundary(boundary, request)
    }
    pub(crate) fn enter_native_strata_draft_scope(
        &mut self,
        scope: cubecl_runtime::server::NativeStrataDraftScope,
    ) -> Result<(), ServerError> {
        self.private_io_active()?;
        let io = self.transient_io.as_mut().unwrap();
        if io.native_branch_recording()?.is_some() || io.has_pending_info() {
            return Err(super::transient_io::fault(
                "draft entry has unsealed parameters",
            ));
        }
        let request = io.native_submission_scope()?;
        self.strata_graph
            .as_mut()
            .ok_or_else(|| super::transient_io::fault("draft original graph missing"))?
            .enter_draft_scope(scope, request)
    }
    pub(crate) fn end_native_strata_draft_scope(
        &mut self,
        scope: cubecl_runtime::server::NativeStrataDraftScope,
    ) -> Result<(), ServerError> {
        self.private_io_active()?;
        let io = self.transient_io.as_mut().unwrap();
        if io.native_branch_recording()?.is_some() {
            return Err(super::transient_io::fault(
                "draft end inside unsealed branch",
            ));
        }
        let request = io.native_submission_scope()?;
        self.flush(StreamErrorMode {
            ignore: false,
            flush: true,
        })?;
        self.strata_graph
            .as_mut()
            .ok_or_else(|| super::transient_io::fault("draft original graph missing"))?
            .end_draft_scope(scope, request)?;
        // Both original native lanes/readiness/visibility joined before pins,
        // uniform ownership or the allocator's deferred cleanup can be released.
        self.mem_manage
            .release_native_pins_joined(request)
            .map_err(ServerError::Io)?;
        self.mem_manage.memory_cleanup(false);
        self.mem_manage.release_uniforms();
        Ok(())
    }
    pub(crate) fn submit_native_strata_experts(&mut self) -> Result<(), ServerError> {
        self.private_io_active()?;
        if self.strata_graph.is_none() {
            return Err(super::transient_io::fault(
                "native expert submission requires selected graph",
            ));
        }
        self.flush_native_mode(
            StreamErrorMode {
                ignore: false,
                flush: true,
            },
            false,
        )
    }
    fn flush_native_mode(
        &mut self,
        mode: StreamErrorMode,
        wait_native: bool,
    ) -> Result<(), ServerError> {
        let mutation = self.transient_io.as_ref().map(|io| io.mutation_guard());
        let scopes = self
            .transient_io
            .as_ref()
            .map(|_| super::transient_io::IoScopes::new(&self.device));
        let result = self.flush_inner(mode, wait_native);
        if let Some(scopes) = scopes {
            scopes.finish(self.transient_io.as_mut().unwrap());
        }
        if result.is_err()
            && let Some(io) = self.transient_io.as_mut()
        {
            io.observe_public_submission_failure();
        }
        if result.is_ok() {
            if let Some(mutation) = mutation {
                mutation.accept();
            }
        }
        result
    }
    fn flush_inner(&mut self, mode: StreamErrorMode, wait_native: bool) -> Result<(), ServerError> {
        // Actual primary work still requires a global drain/visibility. Native
        // draft-only work finishes its own lane; original access admission and
        // managed allocation pins retain any unrelated pending target owners.
        if let Some(graph) = self.strata_graph.as_mut() {
            if self.encoder_dirty || self.pending_write_count != 0 {
                graph.complete_primary_access()?;
            } else {
                graph.complete_scoped_submission()?;
            }
        }
        if self.tasks_count == 0
            && !self.encoder_dirty
            && self.pending_write_count == 0
            && !self
                .transient_io
                .as_ref()
                .is_some_and(|io| io.has_pending_info())
        {
            return self.flush_errors(mode);
        }

        // End the current compute pass.
        self.compute_pass = None;

        let native = self.strata_graph.is_some();
        let native_info = if native {
            self.transient_io
                .as_mut()
                .map(|io| io.prepare_native_info())
                .transpose()?
        } else {
            None
        };
        let parameters = if native {
            None
        } else {
            self.transient_io
                .as_mut()
                .map(|io| io.prepare_batched_info())
                .transpose()?
                .flatten()
        };
        let primary =
            !native || self.encoder_dirty || self.pending_write_count != 0 || parameters.is_some();
        // Native-only math/INFO must not queue an empty primary submission:
        // its completion could sit behind an unrelated pending target commit.
        let index = if primary {
            let mut commands = Vec::with_capacity(2);
            commands.extend(parameters);
            if self.encoder_dirty || !native {
                let encoder = std::mem::replace(
                    &mut self.encoder,
                    self.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("CubeCL Tasks Encoder"),
                        }),
                );
                commands.push(encoder.finish());
                self.encoder_dirty = false;
            }
            Some(self.queue.submit(commands))
        } else {
            None
        };
        let pending_native = self.strata_graph.as_ref().is_some_and(|g| g.queued());
        if (pending_native || native_info.as_ref().is_some_and(|p| !p.is_empty()))
            && let Some(index) = &index
        {
            self.device
                .poll(wgpu::PollType::Wait {
                    submission_index: Some(index.clone()),
                    timeout: Some(std::time::Duration::from_secs(120)),
                })
                .map_err(|_| super::transient_io::fault("native graph primary work completion"))?;
        }
        if let Some(copies) = native_info {
            self.strata_graph
                .as_mut()
                .unwrap()
                .upload_parameters(copies)?;
        }
        if pending_native {
            let branch = self
                .transient_io
                .as_mut()
                .map(|io| io.native_branch_recording())
                .transpose()?
                .flatten();
            self.strata_graph
                .as_mut()
                .unwrap()
                .execute(wait_native, branch, index.is_some())?;
            if let Some(io) = self.transient_io.as_mut() {
                io.native_branch_submitted()?;
            }
        }
        if wait_native {
            // INFO source completion alone does not consume visibility owed
            // to a future native math header. Keep unrelated target work live.
            if let Some(graph) = self.strata_graph.as_mut() {
                graph.complete_scoped_submission()?;
            }
            if let Some(index) = index {
                self.cleanup_boundary.ordinary_regulation(|| {
                    self.submission_load
                        .regulate(&self.device, self.tasks_count, index);
                });
            }
            self.mem_manage.memory_cleanup(false);
            self.mem_manage.release_uniforms();
        }

        self.tasks_count = 0;
        self.pending_write_count = 0;

        self.flush_errors(mode)
    }

    fn flush_errors(&mut self, mode: StreamErrorMode) -> Result<(), ServerError> {
        if mode.flush {
            let errors = self.flush_errors_queue();

            if !mode.ignore && !errors.is_empty() {
                let error = ServerError::ServerUnhealthy {
                    errors,
                    backtrace: BackTrace::capture(),
                };
                return Err(error);
            }
        } else if !mode.ignore && !self.errors.is_empty() {
            let error = ServerError::ServerUnhealthy {
                errors: self.errors.clone(),
                backtrace: BackTrace::capture(),
            };
            return Err(error);
        }

        Ok(())
    }

    fn register_pipeline<'a>(
        &mut self,
        pipeline: Arc<ComputePipeline>,
        resources: impl Iterator<Item = &'a WgpuResource> + Clone,
        read_dependencies: Vec<WgpuResource>,
        dispatch: &CubeCount,
    ) {
        if !read_dependencies.is_empty() && self.strata_graph.is_none() {
            self.error(super::public_external::fault(
                "physical reads require native graph",
            ));
            return;
        }
        if dispatch.is_empty() {
            return;
        }

        if self.strata_graph.is_some() {
            let retained = resources.cloned().collect::<Vec<_>>();
            // Register each true root independently, not the inference
            // pipeline's disjoint aliases in one WebGPU usage scope. Native
            // commands retain their original exact range/hazard validation.
            for resource in &retained {
                if let Err(error) = self.initialize_native_storage_root(resource) {
                    self.error(error);
                    return;
                }
            }
            let actual = match dispatch.clone() {
                CubeCount::Static(x, y, z) => super::strata_graph::Dispatch::Direct(x, y, z),
                CubeCount::Dynamic(binding) => match self.mem_manage.get_resource(binding) {
                    Ok(r) => super::strata_graph::Dispatch::Indirect(r),
                    Err(e) => {
                        self.error(ServerError::Io(e));
                        return;
                    }
                },
            };
            let result = self.strata_graph.as_mut().unwrap().push(
                &pipeline,
                retained,
                read_dependencies,
                actual,
            );
            if let Err(error) = result {
                self.error(error);
                return;
            }
            self.tasks_count += 1;
            self.flush_if_needed();
            return;
        }
        let cached = self
            .transient_io
            .as_ref()
            .and_then(|io| io.cached_binding_group(&pipeline, resources.clone()));
        let bind_group = if let Some(group) = cached {
            group
        } else {
            let creation_start = std::time::Instant::now();
            let entries = resources
                .clone()
                .enumerate()
                .map(|(i, r)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: r.as_wgpu_bind_resource(),
                })
                .collect::<Vec<_>>();

            let group_layout = pipeline.get_bind_group_layout(0);
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &group_layout,
                entries: &entries,
            });

            if let Some(io) = &mut self.transient_io {
                io.retain_binding_group(
                    &pipeline,
                    resources.clone(),
                    &bind_group,
                    creation_start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                );
            }
            bind_group
        };
        // Start a new compute pass if needed. The forget_lifetime allows
        // to store this with a 'static lifetime, but the compute pass must
        // be dropped before the encoder. This isn't unsafe - it's still checked at runtime.
        self.encoder_dirty = true;
        let pass = self.compute_pass.get_or_insert_with(|| {
            let writes = if let Timings::Device(query_time) = &mut self.timings {
                query_time
                    .register_profile_device(&self.device)
                    .map(|query_set| wgpu::ComputePassTimestampWrites {
                        query_set,
                        beginning_of_pass_write_index: Some(0),
                        end_of_pass_write_index: Some(1),
                    })
            } else {
                None
            };
            self.encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: writes,
                })
                .forget_lifetime()
        });

        self.tasks_count += 1;

        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);

        match dispatch.clone() {
            CubeCount::Static(x, y, z) => {
                pass.dispatch_workgroups(x, y, z);
            }
            CubeCount::Dynamic(binding) => {
                let res = self.mem_manage.get_resource(binding).unwrap();
                pass.dispatch_workgroups_indirect(&res.buffer, res.offset);
            }
        }
        self.flush_if_needed();
    }

    pub(crate) fn flush_errors_queue(&mut self) -> Vec<ServerError> {
        let errors = core::mem::take(&mut self.errors);

        if !errors.is_empty() {
            self.profile_error(ProfileError::Unknown {
                reason: alloc::format!("{:?}", errors),
                backtrace: BackTrace::capture(),
            });
        }

        errors
    }
}

#[cfg(not(target_family = "wasm"))]
mod __submission_load {
    use cubecl_runtime::server::PublicTargetSubmissionBacklog;

    /// Both policies retain exactly one submission stamp and one task counter.
    /// Separate variants keep the policy immutable without another owner/list.
    #[derive(Debug)]
    pub enum SubmissionLoad<I = wgpu::SubmissionIndex> {
        Init {
            last_index: I,
            tasks_count_submitted: usize,
        },
        Empty,
        PublicTargetInit {
            last_index: I,
            tasks_count_submitted: usize,
        },
        PublicTargetEmpty,
    }

    impl<I> Default for SubmissionLoad<I> {
        fn default() -> Self {
            Self::Empty
        }
    }

    impl<I> SubmissionLoad<I> {
        pub fn policy(&self) -> PublicTargetSubmissionBacklog {
            match self {
                Self::PublicTargetInit { .. } | Self::PublicTargetEmpty => {
                    PublicTargetSubmissionBacklog::Queued2048
                }
                Self::Init { .. } | Self::Empty => PublicTargetSubmissionBacklog::Default512,
            }
        }

        /// Called only by the dedicated I/O constructor after its real drain.
        /// The default leaves the original regulator state/stamp unchanged.
        pub fn configure_public_target(&mut self, policy: PublicTargetSubmissionBacklog) {
            if policy == PublicTargetSubmissionBacklog::Queued2048 {
                *self = Self::PublicTargetEmpty;
            }
        }

        /// Return the ORIGINAL previous stamp to wait for, never the current
        /// submission. Counter reset remains after the real poll below.
        fn record_submission(&mut self, tasks_count: usize, mut index: I) -> Option<I> {
            let policy = self.policy();
            let window = policy.task_window();
            match self {
                Self::Init {
                    last_index,
                    tasks_count_submitted,
                }
                | Self::PublicTargetInit {
                    last_index,
                    tasks_count_submitted,
                } => {
                    if policy == PublicTargetSubmissionBacklog::Default512 {
                        *tasks_count_submitted += tasks_count;
                    } else {
                        // Every ACTUAL submission spends at least one slot,
                        // including a zero-task pending-INFO submission. Huge
                        // invalid counts cannot wrap away the selected wait.
                        *tasks_count_submitted = tasks_count_submitted
                            .saturating_add(tasks_count.max(policy.minimum_submission_cost()));
                    }
                    if *tasks_count_submitted >= window {
                        core::mem::swap(last_index, &mut index);
                        return Some(index);
                    }
                }
                Self::Empty => {
                    *self = Self::Init {
                        last_index: index,
                        tasks_count_submitted: 0,
                    }
                }
                Self::PublicTargetEmpty => {
                    *self = Self::PublicTargetInit {
                        last_index: index,
                        tasks_count_submitted: 0,
                    }
                }
            }
            None
        }

        fn finish_wait(&mut self) {
            match self {
                Self::Init {
                    tasks_count_submitted,
                    ..
                }
                | Self::PublicTargetInit {
                    tasks_count_submitted,
                    ..
                } => *tasks_count_submitted = 0,
                Self::Empty | Self::PublicTargetEmpty => unreachable!(),
            }
        }
    }

    impl SubmissionLoad {
        pub fn regulate(
            &mut self,
            device: &wgpu::Device,
            tasks_count: usize,
            index: wgpu::SubmissionIndex,
        ) {
            if let Some(previous_index) = self.record_submission(tasks_count, index) {
                let started = std::time::Instant::now();
                if let Err(e) = device.poll(wgpu::PollType::Wait {
                    submission_index: Some(previous_index),
                    timeout: None,
                }) {
                    log::warn!(
                        "wgpu: requested wait timed out before the submission was completed during sync. ({e})"
                    )
                }
                super::record_submission_wait(started.elapsed());
                self.finish_wait();
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn counter(load: &SubmissionLoad<usize>) -> usize {
            match load {
                SubmissionLoad::Init {
                    tasks_count_submitted,
                    ..
                }
                | SubmissionLoad::PublicTargetInit {
                    tasks_count_submitted,
                    ..
                } => *tasks_count_submitted,
                _ => 0,
            }
        }

        // A separate copy of the original 512-state transition is the oracle;
        // the candidate shares neither policy selection nor helper code here.
        #[derive(Default)]
        enum Original<I> {
            Init {
                last: I,
                count: usize,
            },
            #[default]
            Empty,
        }
        impl<I> Original<I> {
            fn submit(&mut self, tasks: usize, mut index: I) -> Option<I> {
                match self {
                    Self::Init { last, count } => {
                        *count += tasks;
                        if *count >= 512 {
                            core::mem::swap(last, &mut index);
                            *count = 0;
                            return Some(index);
                        }
                    }
                    Self::Empty => {
                        *self = Self::Init {
                            last: index,
                            count: 0,
                        }
                    }
                }
                None
            }
        }

        #[test]
        fn public_target_backlog_default_matches_original_stamp_sequence() {
            let mut load = SubmissionLoad::<usize>::default();
            let mut original = Original::<usize>::default();
            // Includes full scheduler batches, clear overshoot, explicit read
            // flushes and empty INFO-only submissions from the actual runtime.
            let batches = [256, 256, 271, 2, 0, 34, 255, 1, 256, 7];
            for index in 0..400 {
                let tasks = batches[index % batches.len()];
                let expected = original.submit(tasks, index);
                let actual = load.record_submission(tasks, index);
                assert_eq!(actual, expected);
                if actual.is_some() {
                    assert!(counter(&load) >= 512);
                    load.finish_wait();
                }
                assert_eq!(load.policy(), PublicTargetSubmissionBacklog::Default512);
            }
        }

        #[test]
        fn public_target_backlog_waits_previous_stamp_at_exact_2048_boundary() {
            let mut load = SubmissionLoad::<usize>::default();
            load.configure_public_target(PublicTargetSubmissionBacklog::Queued2048);
            assert_eq!(load.record_submission(256, 100), None);
            assert_eq!(counter(&load), 0); // Original first-submission rule.
            for index in 101..108 {
                assert_eq!(load.record_submission(256, index), None);
            }
            assert_eq!(counter(&load), 1792);
            assert_eq!(load.record_submission(256, 108), Some(100));
            assert_eq!(counter(&load), 2048); // Not reset before device.poll.
            load.finish_wait();
            assert_eq!(counter(&load), 0);
            for index in 109..116 {
                assert_eq!(load.record_submission(256, index), None);
            }
            assert_eq!(load.record_submission(256, 116), Some(108));
            load.finish_wait();
            assert_eq!(load.policy(), PublicTargetSubmissionBacklog::Queued2048);
        }

        #[test]
        fn public_target_backlog_finite_irregular_batch_envelope() {
            // The public target's largest enqueue overshoot is a batch of
            // sixteen owner clears: 255 existing tasks plus sixteen clears.
            const MAX_BATCH: usize = 271;
            for policy in [
                PublicTargetSubmissionBacklog::Default512,
                PublicTargetSubmissionBacklog::Queued2048,
            ] {
                let mut load = SubmissionLoad::<usize>::default();
                load.configure_public_target(policy);
                let mut submitted = 0usize;
                let mut completed = 0usize;
                let mut totals = vec![0usize];
                let mut peak = 0usize;
                let mut waits = 0;
                for index in 1..=2048 {
                    let tasks = [MAX_BATCH, 1, 0, 256, 34, 2][index % 6];
                    submitted += tasks.max(policy.minimum_submission_cost());
                    totals.push(submitted);
                    peak = peak.max(submitted - completed);
                    if let Some(previous) = load.record_submission(tasks, index) {
                        assert!(previous < index);
                        assert!(counter(&load) < policy.task_window() + MAX_BATCH);
                        completed = totals[previous];
                        load.finish_wait();
                        waits += 1;
                    }
                }
                assert!(waits > 0);
                assert!(peak <= 2 * (policy.task_window() + MAX_BATCH - 1));
            }
        }

        #[test]
        fn public_target_backlog_counts_only_actual_info_only_submissions_and_never_wraps() {
            let mut load = SubmissionLoad::<usize>::default();
            load.configure_public_target(PublicTargetSubmissionBacklog::Queued2048);
            // record_submission is reached ONLY after queue.submit. A flush
            // with neither tasks nor pending INFO returns before this point.
            assert_eq!(load.record_submission(0, 1), None);
            for index in 2..2049 {
                assert_eq!(load.record_submission(0, index), None);
            }
            assert_eq!(counter(&load), 2047);
            assert_eq!(load.record_submission(0, 2049), Some(1));
            assert_eq!(counter(&load), 2048);
            load.finish_wait();
            assert_eq!(load.record_submission(2047, 2050), None);
            assert_eq!(load.record_submission(usize::MAX, 2051), Some(2049));
            assert_eq!(counter(&load), usize::MAX);
            load.finish_wait();
            assert_eq!(counter(&load), 0);
        }

        #[test]
        fn public_target_backlog_retains_one_stamp_without_extra_state_allocation() {
            assert_eq!(
                core::mem::size_of::<SubmissionLoad<usize>>(),
                core::mem::size_of::<Original<usize>>()
            );
            assert_eq!(
                core::mem::size_of::<SubmissionLoad>(),
                core::mem::size_of::<Original<wgpu::SubmissionIndex>>()
            );
            let mut load = SubmissionLoad::<usize>::default();
            assert_eq!(load.record_submission(256, 10), None);
            assert_eq!(load.record_submission(256, 11), None);
            load.configure_public_target(PublicTargetSubmissionBacklog::Default512);
            assert_eq!(load.record_submission(256, 12), Some(10));
        }
    }
}

#[cfg(target_family = "wasm")]
mod __submission_load_wasm {
    #[derive(Default, Debug)]
    pub struct SubmissionLoad;

    impl SubmissionLoad {
        pub fn configure_public_target(
            &mut self,
            policy: cubecl_runtime::server::PublicTargetSubmissionBacklog,
        ) {
            debug_assert_eq!(
                policy,
                cubecl_runtime::server::PublicTargetSubmissionBacklog::Default512
            );
        }

        pub fn policy(&self) -> cubecl_runtime::server::PublicTargetSubmissionBacklog {
            cubecl_runtime::server::PublicTargetSubmissionBacklog::Default512
        }

        pub fn regulate(
            &mut self,
            _device: &wgpu::Device,
            _tasks_count: usize,
            _index: wgpu::SubmissionIndex,
        ) {
            // Nothing to do.
        }
    }
}

#[cfg(not(target_family = "wasm"))]
use __submission_load::*;
#[cfg(target_family = "wasm")]
use __submission_load_wasm::*;

fn validate_public_target_backlog(
    policy: cubecl_runtime::server::PublicTargetSubmissionBacklog,
    tasks_max: usize,
) -> Result<(), ServerError> {
    if policy == cubecl_runtime::server::PublicTargetSubmissionBacklog::Queued2048
        && (tasks_max != 256 || cfg!(target_family = "wasm"))
    {
        return Err(super::transient_io::fault(
            "public target backlog requires native 256-task submissions",
        ));
    }
    Ok(())
}

fn validate_public_target_request_io_configuration(
    policy: cubecl_runtime::server::PublicTargetSubmissionBacklog,
    tasks_max: usize,
    inline_device: bool,
    retired: bool,
    configured: bool,
) -> Result<(), ServerError> {
    validate_public_target_backlog(policy, tasks_max)?;
    if !inline_device {
        return Err(super::transient_io::fault(
            "public target I/O requires inline device",
        ));
    }
    if retired {
        return Err(super::transient_io::fault("private I/O stream retired"));
    }
    if configured {
        return Err(super::transient_io::fault("private I/O already configured"));
    }
    Ok(())
}

#[cfg(test)]
mod public_target_backlog_tests {
    use super::*;
    use cubecl_runtime::server::PublicTargetSubmissionBacklog;

    #[test]
    fn public_target_backlog_selection_preserves_default_and_exact_scheduler_bound() {
        for size in [0, 1, 255, 256, 257, 512, usize::MAX] {
            assert!(
                validate_public_target_backlog(PublicTargetSubmissionBacklog::Default512, size)
                    .is_ok()
            );
            assert_eq!(
                validate_public_target_backlog(PublicTargetSubmissionBacklog::Queued2048, size)
                    .is_ok(),
                size == 256 && !cfg!(target_family = "wasm")
            );
        }
        assert_eq!(PublicTargetSubmissionBacklog::default().task_window(), 512);
        assert_eq!(
            PublicTargetSubmissionBacklog::Queued2048.task_window(),
            2048
        );
        assert_eq!(
            PublicTargetSubmissionBacklog::default().minimum_submission_cost(),
            0
        );
        assert_eq!(
            PublicTargetSubmissionBacklog::Queued2048.minimum_submission_cost(),
            1
        );
    }

    #[test]
    fn public_target_backlog_constructor_rejects_late_reconfigure_and_retired_streams() {
        for policy in [
            PublicTargetSubmissionBacklog::Default512,
            PublicTargetSubmissionBacklog::Queued2048,
        ] {
            for inline in [false, true] {
                for retired in [false, true] {
                    for configured in [false, true] {
                        assert_eq!(
                            validate_public_target_request_io_configuration(
                                policy, 256, inline, retired, configured
                            )
                            .is_ok(),
                            inline
                                && !retired
                                && !configured
                                && (policy == PublicTargetSubmissionBacklog::Default512
                                    || !cfg!(target_family = "wasm"))
                        );
                    }
                }
            }
        }
    }
}
