//! Owning native-global PUBLIC expert cache. Original GGUF bytes, exact persistent
//! GPU roots and bounded startup windows; no decoded model or request data.
//! Admission/precharge belongs to GpuExecutor; consumers must retain this owner.
use crate::strata_global_plan::{Plan, Source};
use cubecl::{prelude::ComputeClient, wgpu::WgpuRuntime};
use cubecl_runtime::server::Handle;
use kuno_text::{Error, Result, model::BoundGraph};
use std::sync::atomic::AtomicBool;
#[cfg(feature = "strata-parallel-loading")]
#[path = "strata_upload_completion.rs"]
mod upload_completion;
fn reject() -> Error {
    Error::InvalidFormat("Strata owning global GPU cache")
}
// Diagnostic is scoped to the fixed public artifact constructor. Emit only
// static stages/error classes, never backend descriptions or tensor contents.
fn public_load_fault(stage: &'static str, cause: cubecl_runtime::server::ServerError) -> Error {
    use cubecl_runtime::server::{IoError, LaunchError, ServerError};
    fn classify(error: &ServerError, depth: usize) {
        if depth > 8 {
            return;
        }
        let class = match error {
            ServerError::Generic { reason, .. }
                if reason == "public weight write requires exact aligned same-stream views" =>
            {
                "public_view_descriptor"
            }
            ServerError::Generic { .. } => "generic",
            ServerError::Launch(LaunchError::OutOfMemory { .. }) => "launch_oom",
            ServerError::Launch(LaunchError::CompilationError(_)) => "launch_compile",
            ServerError::Launch(LaunchError::TooManyResources(_)) => "launch_limits",
            ServerError::Launch(_) => "launch_other",
            ServerError::Profile(_) => "profile",
            ServerError::Io(IoError::BufferTooBig { .. }) => "io_buffer_size",
            ServerError::Io(IoError::UnsupportedStrides { .. }) => "io_strides",
            ServerError::Io(IoError::NotFound { .. }) => "io_missing_owner",
            ServerError::Io(IoError::FreeError { .. }) => "io_free",
            ServerError::Io(IoError::UnsupportedIoOperation { .. }) => "io_unsupported",
            ServerError::Io(IoError::Unknown { .. }) => "io_unknown",
            ServerError::Io(IoError::Execution(inner)) => {
                classify(inner, depth + 1);
                "io_execution"
            }
            ServerError::ServerUnhealthy { errors, .. } => {
                for error in errors.iter().take(8) {
                    classify(error, depth + 1);
                }
                "unhealthy"
            }
        };
        eprintln!("native_public_cache_fault class={class}");
    }
    if crate::strata_intel_core::enabled()
        && std::env::var("KUNO_LAB_LANGUAGE_PUBLIC_ONLY").as_deref() == Ok("1")
    {
        eprintln!("native_public_cache_fault stage={stage}");
        classify(&cause, 0);
    }
    Error::InvalidFormat("Strata original cache public load")
}
pub(crate) struct Cache {
    client: ComputeClient<WgpuRuntime>,
    roots: Vec<Handle>,
    bytes: u64,
    plan_instance: u64,
}
impl Cache {
    /// Caller has charged every exact root under exclusive persistent admission.
    pub(crate) fn load(
        client: &ComputeClient<WgpuRuntime>,
        graph: BoundGraph<'_, '_>,
        plan: &Plan,
    ) -> Result<Self> {
        if !crate::strata_intel_core::enabled() {
            return Err(reject());
        }
        plan.check_binding(graph)?;
        #[cfg(feature = "strata-parallel-loading")]
        let reader = graph
            .tensor("blk.0.ffn_gate_exps.weight")?
            .positional_reader()?;
        let mut roots = Vec::with_capacity(plan.chunks.len());
        let mut bytes = 0u64;
        let (mut read_s, mut submit_s) = (0f64, 0f64);
        // Allocate every true persistent GPU root first. Every byte, including
        // slot/root padding, is initialized by bounded original-model windows
        // before any table/consumer exists.
        let mut handles = Vec::with_capacity(plan.chunks.len());
        for chunk in &plan.chunks {
            if chunk.physical_bytes > client.properties().memory.max_page_size {
                return Err(Error::LimitExceeded(
                    "global cache root exceeds backend binding",
                ));
            }
            let size =
                usize::try_from(chunk.physical_bytes).map_err(|_| Error::ArithmeticOverflow)?;
            let handle = client.empty(size);
            if handle.offset_start.unwrap_or(0) != 0
                || handle.offset_end.unwrap_or(0) != 0
                || handle.size() != chunk.physical_bytes
                || handle.size_in_used() != chunk.physical_bytes
            {
                return Err(reject());
            }
            roots.push(handle.clone());
            handles.push(handle);
            bytes = bytes
                .checked_add(chunk.physical_bytes)
                .ok_or(Error::ArithmeticOverflow)?;
        }
        let window_bytes = crate::strata_global_plan::PUBLIC_UPLOAD_WINDOW;
        // (root index, first byte) for every bounded window of every root.
        let windows: Vec<(usize, u64)> = plan
            .chunks
            .iter()
            .enumerate()
            .flat_map(|(index, chunk)| {
                (0..chunk.physical_bytes)
                    .step_by(window_bytes as usize)
                    .map(move |first| (index, first))
            })
            .collect();
        let make_window = |(index, first): (usize, u64)| -> Result<Vec<u8>> {
            let physical = plan.chunks[index].physical_bytes;
            let count = usize::try_from((physical - first).min(window_bytes))
                .map_err(|_| Error::ArithmeticOverflow)?;
            let mut window = Vec::new();
            window
                .try_reserve_exact(count)
                .map_err(|_| Error::LimitExceeded("global cache public load window"))?;
            if count == 0 || window.capacity() != count {
                return Err(reject());
            }
            window.resize(count, 0);
            #[cfg(feature = "strata-parallel-loading")]
            plan.load_chunk_read_at(graph, &reader, index, first, &mut window)?;
            #[cfg(not(feature = "strata-parallel-loading"))]
            plan.load_chunk_range(graph, index, first, &mut window)?;
            Ok(window)
        };
        let view_of = |(index, first): (usize, u64), len: usize| -> Result<Handle> {
            let physical = plan.chunks[index].physical_bytes;
            let end = first
                .checked_add(len as u64)
                .filter(|&end| end <= physical)
                .ok_or(Error::ArithmeticOverflow)?;
            Ok(handles[index]
                .clone()
                .offset_start(first)
                .offset_end(physical - end))
        };
        #[cfg(not(feature = "strata-parallel-loading"))]
        for &at in &windows {
            let window = make_window(at)?;
            let view = view_of(at, window.len())?;
            client
                .write_public_owned_batch(vec![(view, window)])
                .map_err(|cause| public_load_fault("upload", cause))?;
            // Moved source/upload resources drain before the next window;
            // increasing GPU contiguity never enlarges the host envelope.
            cubecl_common::future::block_on(client.sync())
                .map_err(|cause| public_load_fault("completion", cause))?;
        }
        #[cfg(feature = "strata-parallel-loading")]
        {
            // Read UPLOAD_BATCH_WINDOWS windows in parallel threads and submit
            // them with ONE sync. The next batch is read on a background thread
            // while the current batch is copied to staging and uploaded, across
            // root boundaries. Bounded envelope: two batches of host windows.
            const UPLOAD_BATCH_WINDOWS: usize = 16;
            let read_batch = |batch: &[(usize, u64)]| -> Result<Vec<((usize, u64), Vec<u8>)>> {
                std::thread::scope(|scope| {
                    let workers: Vec<_> = batch
                        .iter()
                        .map(|&at| {
                            let make_window = &make_window;
                            scope.spawn(move || make_window(at).map(|w| (at, w)))
                        })
                        .collect();
                    workers
                        .into_iter()
                        .map(|worker| worker.join().map_err(|_| reject())?)
                        .collect()
                })
            };
            let mut batches = windows.chunks(UPLOAD_BATCH_WINDOWS);
            let t = std::time::Instant::now();
            let mut current = batches.next().map(read_batch).transpose()?;
            read_s += t.elapsed().as_secs_f64();
            while let Some(ready) = current {
                let mut writes = Vec::with_capacity(ready.len());
                for (at, window) in ready {
                    writes.push((view_of(at, window.len())?, window));
                }
                let next_batch = batches.next();
                let t = std::time::Instant::now();
                current = std::thread::scope(|scope| -> Result<_> {
                    let read_batch = &read_batch;
                    let reading = next_batch.map(|batch| scope.spawn(move || read_batch(batch)));
                    // Arm BEFORE synchronous scheduling. Setup/read errors and
                    // panics must drain actual prior work or retain its roots.
                    let mut pending = upload_completion::Pending::new(&mut roots, || {
                        cubecl_common::future::block_on(client.sync())
                    });
                    client
                        .write_public_owned_batch(writes)
                        .map_err(|cause| public_load_fault("upload", cause))?;
                    // sync() synchronously consumes Scheduler's moved sources,
                    // copies them to backend staging and submits the queue. Its
                    // RETURNED future owns real completion, not scheduling.
                    let completion = client.sync();
                    pending.submitted(move || cubecl_common::future::block_on(completion));
                    pending
                        .finish()
                        .map_err(|cause| public_load_fault("completion", cause))?;
                    submit_s += t.elapsed().as_secs_f64();
                    match reading {
                        Some(worker) => worker.join().map_err(|_| reject())?.map(Some),
                        None => Ok(None),
                    }
                })?;
                read_s = t.elapsed().as_secs_f64().max(read_s);
            }
        }
        if crate::strata_intel_core::enabled() && std::env::var("KUNO_LAB_LANGUAGE_PUBLIC_ONLY").as_deref() == Ok("1") {
            eprintln!("public_load_step gpu_cache_first_read {read_s:.3}");
            eprintln!("public_load_step gpu_cache_upload_total {submit_s:.3}");
        }
        if bytes != plan.resident_piece_bytes {
            return Err(reject());
        }
        let resources = crate::resident_util::retained_resources(client, roots.clone())?;
        if resources.len() != roots.len() {
            return Err(reject());
        }
        for (index, resource) in resources.iter().enumerate() {
            let r = resource.resource();
            if r.offset != 0
                || r.size != plan.chunks[index].physical_bytes
                || r.buffer.size() != r.size
                || resources[..index]
                    .iter()
                    .any(|other| other.resource().buffer == r.buffer)
            {
                return Err(reject());
            }
        }
        graph.verify_unchanged()?;
        Ok(Self {
            client: client.clone(),
            roots,
            bytes,
            plan_instance: plan.instance(),
        })
    }
    /// Exclusive source transaction only: private exposure has already been
    /// completely erased/inspected. Restore CURRENT logical placement, not the
    /// initial ranked profile. No allocation or private public-upload shortcut.
    pub(crate) fn restore_chunks(
        &self,
        graph: BoundGraph<'_, '_>,
        plan: &Plan,
        indices: [usize; 3],
        placement: &crate::strata_global_sources::Placement,
    ) -> Result<()> {
        plan.check_binding(graph)?;
        if self.plan_instance != plan.instance() {
            return Err(reject());
        }
        for (j, &index) in indices.iter().enumerate() {
            if indices[..j].contains(&index) {
                return Err(reject());
            }
            let root = self.roots.get(index).ok_or_else(reject)?;
            let chunk = plan.chunks.get(index).ok_or_else(reject)?;
            if root.size_in_used() != chunk.physical_bytes
                || root.size() != chunk.physical_bytes
                || root.offset_start.unwrap_or(0) != 0
                || root.offset_end.unwrap_or(0) != 0
            {
                return Err(reject());
            }
            let mut first = 0u64;
            while first < chunk.physical_bytes {
                let n = (chunk.physical_bytes - first)
                    .min(crate::strata_global_plan::PUBLIC_UPLOAD_WINDOW)
                    as usize;
                let mut bytes = Vec::new();
                bytes
                    .try_reserve_exact(n)
                    .map_err(|_| Error::LimitExceeded("cache restore public window"))?;
                if bytes.capacity() != n {
                    return Err(reject());
                }
                bytes.resize(n, 0);
                plan.load_chunk_placed(graph, index, first, &mut bytes, placement)?;
                let view = root
                    .clone()
                    .offset_start(first)
                    .offset_end(chunk.physical_bytes - first - n as u64);
                self.client
                    .write_public_owned_batch(vec![(view, bytes)])
                    .map_err(crate::resident_util::fault)?;
                cubecl_common::future::block_on(self.client.sync())
                    .map_err(crate::resident_util::fault)?;
                first += n as u64;
            }
        }
        graph.verify_unchanged()?;
        Ok(())
    }
    pub(crate) fn requests(plan: &Plan) -> Vec<u64> {
        plan.chunks.iter().map(|c| c.physical_bytes).collect()
    }
    pub(crate) fn roots(&self) -> &[Handle] {
        &self.roots
    }
    pub(crate) fn bytes(&self) -> u64 {
        self.bytes
    }
    /// Actual model-weight capability, bounded to the original expert role.
    /// No invented source root or residency credit for HOST complements.
    pub(crate) fn resident_role(
        &self,
        graph: BoundGraph<'_, '_>,
        plan: &Plan,
        layer: usize,
        expert: usize,
        role: usize,
    ) -> Result<Option<(Handle, u64, u64)>> {
        plan.check_binding(graph)?;
        if self.plan_instance != plan.instance() {
            return Err(reject());
        }
        let Source::Resident {
            chunk,
            offset,
            bytes,
        } = plan.source(layer, expert, role)?
        else {
            return Ok(None);
        };
        let root = self.roots.get(chunk).ok_or_else(reject)?;
        if offset
            .checked_add(bytes)
            .is_none_or(|n| n > plan.chunks[chunk].logical_bytes)
            || root.size_in_used() != plan.chunks[chunk].physical_bytes
        {
            return Err(reject());
        }
        Ok(Some((root.clone(), offset, bytes)))
    }
    pub(crate) fn persistent_load(
        client: &ComputeClient<WgpuRuntime>,
        graph: BoundGraph<'_, '_>,
        plan: &Plan,
    ) -> Result<Self> {
        use cubecl_runtime::{
            config::{CubeClRuntimeConfig, RuntimeConfig, memory::PersistentMemory},
            memory_management::MemoryAllocationMode,
        };
        if !matches!(
            CubeClRuntimeConfig::get().memory.persistent_memory,
            PersistentMemory::Enabled
        ) {
            return Err(reject());
        }
        let alignment = client.properties().memory.alignment;
        if alignment == 0
            || !alignment.is_power_of_two()
            || Self::requests(plan)
                .iter()
                .any(|b| !b.is_multiple_of(alignment))
        {
            return Err(reject());
        }
        let retired = AtomicBool::new(false);
        crate::expert_banks::exact_persistent::persistent_scope(
            |mode: MemoryAllocationMode| {
                // SAFETY: owning executor holds exclusive stream admission and the
                // complete persistent precharge while this scope is active.
                unsafe {
                    client.allocation_mode(mode);
                }
                Ok(())
            },
            &retired,
            || Self::load(client, graph, plan),
        )
    }
}
impl Drop for Cache {
    fn drop(&mut self) {
        // Joined model/request completion is required before parent close. The
        // runtime also retains submitted buffers until their completion fence.
        let _ = self.client.flush();
    }
}
