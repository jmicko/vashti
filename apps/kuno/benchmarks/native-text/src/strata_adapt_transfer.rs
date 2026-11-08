//! Original adaptive host job beside commit/MTP. The sole client thread
//! retains registry mutation, private-arena exclusion and MAIN submissions;
//! the worker owns only bounded ranking inputs and authenticated copy resources.
use super::*;
use cubecl_runtime::public_external::PublicExternalRebase;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
type WorkerOutput = (
    crate::strata_global_adapt::Cache,
    crate::strata_global_adapt::Proposal,
);
/// State of the ORIGINAL submitted fences; attempted submission never grants
/// completion. Only wait_for_fences success below calls observed().
#[derive(Default)]
struct State {
    pending: [bool; 3],
    copy_joined: bool,
    attempted: [bool; 3],
}
impl State {
    fn attempt(&mut self, index: usize) -> KResult<()> {
        if index >= 3
            || self.pending[index]
            || self.attempted[index]
            || match index {
                0 => self.pending.iter().any(|p| *p) || self.copy_joined || self.attempted[2],
                1 => !self.pending[0] || self.copy_joined || self.attempted[2],
                2 => !self.copy_joined || self.pending[0] || self.pending[1] || self.attempted[2],
                _ => true,
            }
        {
            return Err(fail());
        }
        self.pending[index] = true;
        self.attempted[index] = true;
        Ok(())
    }
    fn worker_joined(&mut self, attempted: bool) -> KResult<()> {
        if attempted && !self.attempted[1] {
            self.attempt(1)?;
        }
        Ok(()) // records debt only; original copy fence still must be observed
    }
    fn observed(&mut self, index: usize) -> KResult<()> {
        if index >= 3 || !self.pending[index] {
            return Err(fail());
        }
        self.pending[index] = false;
        if index == 1 {
            self.copy_joined = true;
        }
        Ok(())
    }
    fn needs_visibility(&self) -> bool {
        self.copy_joined && !self.attempted[2]
    }
}
pub(crate) struct Flight {
    device: Option<wgpu::Device>,
    raw: ash::Device,
    primary: vk::Queue,
    copy: wgpu::hal::vulkan::KunoTransferQueue,
    pool: vk::CommandPool,
    command: vk::CommandBuffer,
    ready: vk::Semaphore,
    done: vk::Semaphore,
    fences: [vk::Fence; 3],
    state: State,
    owners: Vec<WgpuResource>,
    inventory: Option<Arc<Inventory>>,
    worker: Option<std::thread::JoinHandle<KResult<WorkerOutput>>>,
    worker_output: Option<KResult<WorkerOutput>>,
    copy_attempted: Arc<AtomicBool>,
    token: Option<PublicExternalRebase>,
    client: ComputeClient<WgpuRuntime>,
    guard: crate::strata_external_flight::Flight, // !Send: actual original arena/TLS owner
}
fn raw_buffer(r: &WgpuResource) -> KResult<vk::Buffer> {
    let h = unsafe { r.buffer.as_hal::<wgpu::hal::api::Vulkan>() }.ok_or_else(fail)?;
    Ok(unsafe { h.raw_buffer_handle() })
}
/// The parent Flight retains/destructs ALL raw objects only AFTER joining this
/// worker. No client, allocator, registry token, arena or MAIN queue is sent.
struct CopyWork {
    raw: ash::Device,
    copy: wgpu::hal::vulkan::KunoTransferQueue,
    command: vk::CommandBuffer,
    ready: vk::Semaphore,
    done: vk::Semaphore,
    fence: vk::Fence,
    inventory: Arc<Inventory>,
    scratch: WgpuResource,
    attempted: Arc<AtomicBool>,
}
impl CopyWork {
    fn run(
        &self,
        cache: &crate::strata_global_adapt::Cache,
        usage: &[f32],
    ) -> KResult<crate::strata_global_adapt::Proposal> {
        let proposal = cache.propose(usage)?;
        if proposal.swaps.is_empty() {
            return Ok(proposal);
        }
        let pairs = self.inventory.pairs(&proposal.swaps, &self.scratch)?;
        let copies = copy_plan(pairs.iter().map(|p| p.gpu.size), self.scratch.size)?;
        // Paced batches: the native graph's INFO uploads share this in-order
        // COPY queue, so one monolithic swap submission stalled the next
        // window's parameters behind every swap copy. Each batch is whole
        // (host->scratch, gpu->host, scratch->gpu) triples, so the shared
        // scratch never crosses a batch. Only the first batch waits READY and
        // only the last signals DONE and the original fence.
        const TRIPLES_PER_BATCH: usize = 4;
        let batches: Vec<&[Copy]> = copies.chunks(TRIPLES_PER_BATCH * 3).collect();
        let pace = unsafe { self.raw.create_fence(&vk::FenceCreateInfo::default(), None) }
            .map_err(|_| fail())?;
        let result = (|| -> KResult<()> {
            for (index, batch) in batches.iter().enumerate() {
                let first = index == 0;
                let last = index + 1 == batches.len();
                unsafe {
                    self.raw
                        .begin_command_buffer(self.command, &vk::CommandBufferBeginInfo::default())
                        .map_err(|_| fail())?;
                    self.barrier(
                        vk::PipelineStageFlags::HOST,
                        vk::AccessFlags::HOST_WRITE,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE,
                    );
                    for op in batch.iter() {
                        let pair = &pairs[op.pair];
                        let resource = |which| match which {
                            Endpoint::Gpu => &pair.gpu,
                            Endpoint::Host => &pair.host,
                            Endpoint::Scratch => &self.scratch,
                        };
                        let a = resource(op.from);
                        let b = resource(op.to);
                        self.raw.cmd_copy_buffer(
                            self.command,
                            raw_buffer(a)?,
                            raw_buffer(b)?,
                            &[vk::BufferCopy::default()
                                .src_offset(a.offset)
                                .dst_offset(b.offset)
                                .size(op.bytes)],
                        );
                        self.barrier(
                            vk::PipelineStageFlags::TRANSFER,
                            vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE,
                            vk::PipelineStageFlags::TRANSFER,
                            vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE,
                        );
                    }
                    let (stage, access) =
                        wgpu::hal::vulkan::CommandEncoder::kuno_public_host_transfer_read_dependency();
                    self.barrier(stage, access, vk::PipelineStageFlags::HOST, vk::AccessFlags::HOST_READ);
                    self.raw
                        .end_command_buffer(self.command)
                        .map_err(|_| fail())?;
                    // Atomic records only attempted ownership debt, NEVER completion.
                    self.attempted.store(true, Ordering::Release);
                    let waits: &[vk::Semaphore] = if first { std::slice::from_ref(&self.ready) } else { &[] };
                    let stages: &[vk::PipelineStageFlags] =
                        if first { &[vk::PipelineStageFlags::TRANSFER] } else { &[] };
                    let signals: &[vk::Semaphore] = if last { std::slice::from_ref(&self.done) } else { &[] };
                    self.copy
                        .submit(
                            &[vk::SubmitInfo::default()
                                .wait_semaphores(waits)
                                .wait_dst_stage_mask(stages)
                                .command_buffers(std::slice::from_ref(&self.command))
                                .signal_semaphores(signals)],
                            if last { self.fence } else { pace },
                        )
                        .map_err(|_| fail())?;
                    if !last {
                        // The command buffer is reused only after its batch completed.
                        self.raw
                            .wait_for_fences(&[pace], true, 120_000_000_000)
                            .map_err(|_| fail())?;
                        self.raw.reset_fences(&[pace]).map_err(|_| fail())?;
                    }
                }
            }
            Ok(())
        })();
        unsafe { self.raw.destroy_fence(pace, None) };
        result?;
        Ok(proposal)
    }
    unsafe fn barrier(
        &self,
        src: vk::PipelineStageFlags,
        sa: vk::AccessFlags,
        dst: vk::PipelineStageFlags,
        da: vk::AccessFlags,
    ) {
        unsafe {
            self.raw.cmd_pipeline_barrier(
                self.command,
                src,
                dst,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default()
                    .src_access_mask(sa)
                    .dst_access_mask(da)],
                &[],
                &[],
            )
        };
    }
}
impl Flight {
    #[allow(clippy::too_many_arguments)] // Original admission/resource owners plus bounded worker inputs.
    pub(crate) fn submit(
        device: &wgpu::Device,
        client: &ComputeClient<WgpuRuntime>,
        inventory: Arc<Inventory>,
        scratch: WgpuResource,
        token: Option<PublicExternalRebase>,
        guard: crate::strata_external_flight::Flight,
        cache: crate::strata_global_adapt::Cache,
        usage: zeroize::Zeroizing<Vec<f32>>,
    ) -> KResult<Self> {
        let h = unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() }.ok_or_else(fail)?;
        let copy = h.kuno_transfer_queue_authority().ok_or_else(fail)?;
        let family = copy.family_index();
        if family == h.queue_family_index() {
            return Err(fail());
        }
        let mut s = Self {
            device: Some(device.clone()),
            raw: h.raw_device().clone(),
            primary: h.raw_queue(),
            copy,
            pool: vk::CommandPool::null(),
            command: vk::CommandBuffer::null(),
            ready: vk::Semaphore::null(),
            done: vk::Semaphore::null(),
            fences: [vk::Fence::null(); 3],
            state: State::default(),
            owners: vec![scratch.clone()],
            inventory: Some(inventory.clone()),
            worker: None,
            worker_output: None,
            copy_attempted: Arc::new(AtomicBool::new(false)),
            token,
            client: client.clone(),
            guard,
        };
        drop(h);
        unsafe {
            s.pool = s
                .raw
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(family)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .map_err(|_| fail())?;
            s.command = s
                .raw
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(s.pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .map_err(|_| fail())?[0];
            s.ready = s
                .raw
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                .map_err(|_| fail())?;
            s.done = s
                .raw
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                .map_err(|_| fail())?;
            for f in &mut s.fences {
                *f = s
                    .raw
                    .create_fence(&vk::FenceCreateInfo::default(), None)
                    .map_err(|_| fail())?;
            }
            // Original ready signal is enqueued by the sole MAIN authority
            // BEFORE spawning. The worker NEVER submits on a compute queue.
            s.state.attempt(0)?;
            s.raw
                .queue_submit(
                    s.primary,
                    &[vk::SubmitInfo::default().signal_semaphores(std::slice::from_ref(&s.ready))],
                    s.fences[0],
                )
                .map_err(|_| fail())?;
        }
        let work = CopyWork {
            raw: s.raw.clone(),
            copy: s.copy.clone(),
            command: s.command,
            ready: s.ready,
            done: s.done,
            fence: s.fences[1],
            inventory,
            scratch,
            attempted: s.copy_attempted.clone(),
        };
        s.worker = Some(crate::strata_global_adapt::spawn_host_job(move || {
            let proposal = work.run(&cache, &usage)?;
            Ok((cache, proposal))
        })?);
        Ok(s)
    }
    fn wait(&mut self, index: usize) -> KResult<()> {
        if self.state.pending[index] {
            unsafe {
                self.raw
                    .wait_for_fences(&[self.fences[index]], true, 120_000_000_000)
            }
            .map_err(|_| fail())?;
            self.state.observed(index)?;
        }
        Ok(())
    }
    fn join(&mut self) -> KResult<()> {
        if let Some(worker) = self.worker.take() {
            self.worker_output = Some(worker.join().unwrap_or_else(|_| Err(fail())));
        }
        self.state
            .worker_joined(self.copy_attempted.load(Ordering::Acquire))?;
        self.wait(0)?;
        self.wait(1)?;
        if self.state.needs_visibility() {
            self.state.attempt(2)?;
            // Only the original client thread submits MAIN visibility, AFTER
            // joining the actual worker and copy fence. Mutex guards are gone.
            unsafe {
                self.raw.queue_submit(
                    self.primary,
                    &[vk::SubmitInfo::default()
                        .wait_semaphores(std::slice::from_ref(&self.done))
                        .wait_dst_stage_mask(&[vk::PipelineStageFlags::ALL_COMMANDS])],
                    self.fences[2],
                )
            }
            .map_err(|_| fail())?;
        }
        self.wait(2)?;
        Ok(())
    }
    pub(crate) fn finish(mut self) -> KResult<WorkerOutput> {
        self.join()?;
        let output = self.worker_output.take().ok_or_else(fail)??;
        self.guard.joined().map_err(Error::InvalidFormat)?;
        if let Some(token) = self.token.take() {
            unsafe { self.client.finish_public_external_rebase(token) }.map_err(|_| fail())?;
        } else if !output.1.swaps.is_empty() {
            return Err(fail());
        }
        Ok(output)
    }
}
impl Drop for Flight {
    fn drop(&mut self) {
        if self.join().is_err() {
            std::mem::forget(std::mem::take(&mut self.owners));
            if let Some(i) = self.inventory.take() {
                std::mem::forget(i);
            }
            if let Some(d) = self.device.take() {
                std::mem::forget(d);
            }
            return; // original guard retires/quarantines; no cleanup credit
        }
        let _ = self.guard.joined();
        if let Some(token) = self.token.take() {
            let _ = unsafe { self.client.finish_public_external_rebase(token) };
        }
        unsafe {
            for fence in self.fences {
                self.raw.destroy_fence(fence, None);
            }
            self.raw.destroy_semaphore(self.ready, None);
            self.raw.destroy_semaphore(self.done, None);
            self.raw.destroy_command_pool(self.pool, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn completed_host_job_records_only_submission_debt_not_copy_or_visibility_credit() {
        let mut s = State::default();
        assert!(s.worker_joined(true).is_err());
        s.attempt(0).unwrap();
        s.worker_joined(false).unwrap();
        assert!(!s.pending[1]);
        assert!(!s.needs_visibility());
        s.worker_joined(true).unwrap();
        s.worker_joined(true).unwrap();
        assert!(s.pending[1]);
        assert!(!s.copy_joined);
        assert!(s.attempt(2).is_err());
        s.observed(0).unwrap();
        s.observed(1).unwrap();
        assert!(s.needs_visibility());
        s.attempt(2).unwrap();
        assert!(s.pending[2]);
    }
    #[test]
    fn actual_transfer_visibility_wait_requires_original_copy_fence_completion() {
        let mut s = State::default();
        assert!(s.attempt(2).is_err());
        assert!(s.observed(1).is_err());
        s.attempt(0).unwrap();
        s.attempt(1).unwrap();
        assert!(!s.needs_visibility());
        assert!(s.attempt(2).is_err());
        s.observed(0).unwrap();
        assert!(!s.needs_visibility());
        s.observed(1).unwrap();
        assert!(s.needs_visibility());
        s.attempt(2).unwrap();
        assert!(s.pending[2]);
        assert!(!s.needs_visibility());
        s.observed(2).unwrap();
        assert!(!s.pending.iter().any(|p| *p));
    }
    #[test]
    fn failed_or_unobserved_copy_submission_cannot_grant_visibility_or_duplicate_fence_use() {
        let mut s = State::default();
        s.attempt(0).unwrap();
        s.attempt(1).unwrap();
        s.observed(0).unwrap();
        assert!(s.pending[1]);
        assert!(s.attempt(1).is_err());
        assert!(s.attempt(2).is_err());
        assert!(!s.needs_visibility());
    }
    #[test]
    fn visibility_submission_is_attempted_only_once_even_if_observation_fails() {
        let mut s = State::default();
        s.attempt(0).unwrap();
        s.attempt(1).unwrap();
        s.observed(0).unwrap();
        s.observed(1).unwrap();
        s.attempt(2).unwrap();
        assert!(s.pending[2]);
        assert!(!s.needs_visibility());
        assert!(s.attempt(2).is_err());
        s.observed(2).unwrap();
        assert!(s.attempt(2).is_err());
        assert!(s.observed(2).is_err());
    }
    #[test]
    fn construction_without_submission_needs_no_invented_completion_or_semaphore_wait() {
        let mut s = State::default();
        assert!(!s.needs_visibility());
        assert!(!s.pending.iter().any(|p| *p));
        assert!(s.attempt(1).is_err());
        assert!(s.attempt(3).is_err());
        assert!(s.observed(3).is_err());
        s.attempt(0).unwrap();
        s.observed(0).unwrap();
        assert!(!s.needs_visibility());
        assert!(s.attempt(0).is_err());
    }
}
