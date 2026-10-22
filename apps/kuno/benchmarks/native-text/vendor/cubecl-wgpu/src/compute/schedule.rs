// Modified for Kuno's laboratory transient-I/O ownership/erasure patch (2026-10-05).
use crate::{WgpuResource, stream::WgpuStream};
use alloc::sync::Arc;
use cubecl_common::{bytes::Bytes, profile::TimingMethod};
use cubecl_core::{CubeCount, MemoryConfiguration, server::MetadataBindingInfo};
use cubecl_ir::MemoryDeviceProperties;
use cubecl_runtime::{
    logging::ServerLogger,
    stream::{StreamFactory, scheduler::SchedulerStreamBackend},
};

/// Defines tasks that can be scheduled on a WGPU stream.
pub enum ScheduleTask {
    /// Represents a task to write data to a buffer.
    Write {
        /// The data to be written.
        data: Bytes,
        /// The target buffer resource.
        buffer: WgpuResource,
        /// Explicit private upload; ordinary writes are PUBLIC only.
        private: bool,
    },
    /// Represents a task to execute a compute pipeline.
    Execute {
        /// The compute pipeline to execute.
        pipeline: Arc<super::strata_pipeline::BackendPipeline>,
        /// The number of workgroups to dispatch.
        count: CubeCount,
        /// The resources (bindings) required for execution.
        resources: BindingsResource,
    },
}

impl core::fmt::Debug for ScheduleTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Write { data, .. } => f.write_fmt(format_args!("Write(bytes={})", data.len())),
            Self::Execute {
                count, resources, ..
            } => f.write_fmt(format_args!(
                "Execute(resources={}, cube_count={count:?})",
                resources.resources.len()
            )),
        }
    }
}

/// Represents a collection of resources and bindings for a compute task.
#[derive(Debug)]
pub struct BindingsResource {
    /// List of WGPU resources used in the task.
    pub resources: Vec<WgpuResource>,
    /// Actual physical-pointer sources; never descriptor entries.
    pub read_dependencies: Vec<WgpuResource>,
    /// Metadata for uniform bindings.
    pub info: MetadataBindingInfo,
}

/// Represents a WGPU backend for scheduling tasks on streams.
#[derive(Debug)]
pub struct ScheduledWgpuBackend {
    /// Factory for creating WGPU streams.
    factory: WgpuStreamFactory,
}

/// Factory for creating WGPU streams with specific configurations.
#[derive(Debug)]
pub struct WgpuStreamFactory {
    device: wgpu::Device,
    queue: wgpu::Queue,
    memory_properties: MemoryDeviceProperties,
    memory_config: MemoryConfiguration,
    timing_method: TimingMethod,
    tasks_max: usize,
    logger: Arc<ServerLogger>,
    count: u64,
}

impl StreamFactory for WgpuStreamFactory {
    type Stream = WgpuStream;

    fn create(&mut self) -> Self::Stream {
        self.count += 1;

        WgpuStream::new(
            self.device.clone(),
            self.queue.clone(),
            self.memory_properties.clone(),
            self.memory_config.clone(),
            self.timing_method,
            self.tasks_max,
            self.logger.clone(),
        )
    }
}

impl ScheduledWgpuBackend {
    /// Creates a new `ScheduledWgpuBackend` with the given WGPU device, queue, and configurations.
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        memory_properties: MemoryDeviceProperties,
        memory_config: MemoryConfiguration,
        timing_method: TimingMethod,
        tasks_max: usize,
        logger: Arc<ServerLogger>,
    ) -> Self {
        Self {
            factory: WgpuStreamFactory {
                device,
                queue,
                memory_properties,
                memory_config,
                timing_method,
                tasks_max,
                logger,
                count: 0,
            },
        }
    }
}

impl BindingsResource {
    /// Converts metadata and scalar bindings into WGPU resources for a stream.
    pub fn into_resources(
        mut self,
        stream: &mut WgpuStream,
    ) -> Result<(Vec<WgpuResource>, Vec<WgpuResource>), cubecl_runtime::server::ServerError> {
        // Move request-derived metadata into a guard before any operation can
        // fail/panic. Wipe its FULL capacity, including old unused words.
        struct MetadataGuard(Vec<u64>);
        impl Drop for MetadataGuard {
            fn drop(&mut self) {
                let pointer = self.0.as_mut_ptr();
                for i in 0..self.0.capacity() {
                    // SAFETY: exclusively owned writable Vec allocation.
                    unsafe { pointer.add(i).write_volatile(0) };
                }
                std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
            }
        }
        let data = MetadataGuard(core::mem::take(&mut self.info.data));
        if data.0.is_empty() && !self.info.native_parameters.is_empty() {
            return Err(super::transient_io::fault(
                "native INFO schema without data",
            ));
        }
        if !data.0.is_empty() {
            let info = stream
                .create_uniform(bytemuck::cast_slice(&data.0), &self.info.native_parameters)?;
            self.resources.push(info);
        }
        Ok((self.resources, self.read_dependencies))
    }
}

impl SchedulerStreamBackend for ScheduledWgpuBackend {
    type Task = ScheduleTask;
    type Stream = WgpuStream;
    type Factory = WgpuStreamFactory;

    fn enqueue(task: Self::Task, stream: &mut Self::Stream) {
        stream.enqueue_task(task);
    }

    fn enqueue_many(tasks: impl Iterator<Item = Self::Task>, stream: &mut Self::Stream) {
        stream.enqueue_tasks(tasks);
    }

    fn flush(stream: &mut Self::Stream) {
        stream.flush_scheduled();
    }

    fn factory(&mut self) -> &mut Self::Factory {
        &mut self.factory
    }
}
