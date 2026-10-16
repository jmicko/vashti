//! Owning original expert-copy flight on the already adopted transfer queue.
//! One serialized worker submits on all queues; Vulkan resources stay retained
//! through actual transfer and primary visibility fences, including unwinding.
use super::*;
use cubecl_runtime::public_external::PublicExternalRebase;
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
    copy: vk::Queue,
    pool: vk::CommandPool,
    command: vk::CommandBuffer,
    ready: vk::Semaphore,
    done: vk::Semaphore,
    fences: [vk::Fence; 3],
    state: State,
    owners: Vec<WgpuResource>,
    token: Option<PublicExternalRebase>,
    client: ComputeClient<WgpuRuntime>,
    guard: crate::strata_external_flight::Flight,
}
fn raw_buffer(r: &WgpuResource) -> KResult<vk::Buffer> {
    let h = unsafe { r.buffer.as_hal::<wgpu::hal::api::Vulkan>() }.ok_or_else(fail)?;
    Ok(unsafe { h.raw_buffer_handle() })
}
impl Flight {
    /// Actual same-device copies authenticated/revoked by Host::submit_exchange.
    pub(super) fn submit(
        device: &wgpu::Device,
        client: &ComputeClient<WgpuRuntime>,
        pairs: &[Pair],
        scratch: &WgpuResource,
        copies: &[Copy],
        token: PublicExternalRebase,
        guard: crate::strata_external_flight::Flight,
    ) -> KResult<Self> {
        let h = unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() }.ok_or_else(fail)?;
        let (family, copy) = h.kuno_transfer_queue().ok_or_else(fail)?;
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
            owners: Vec::new(),
            token: Some(token),
            client: client.clone(),
            guard,
        };
        drop(h);
        s.owners
            .try_reserve_exact(pairs.len() * 2 + 1)
            .map_err(|_| fail())?;
        s.owners.push(scratch.clone());
        for pair in pairs {
            s.owners.extend([pair.gpu.clone(), pair.host.clone()]);
        }
        unsafe {
            s.pool = s
                .raw
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default().queue_family_index(family),
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
            s.raw
                .begin_command_buffer(s.command, &vk::CommandBufferBeginInfo::default())
                .map_err(|_| fail())?;
            s.barrier(
                vk::PipelineStageFlags::HOST,
                vk::AccessFlags::HOST_WRITE,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE,
            );
            for op in copies {
                let pair = &pairs[op.pair];
                let region = |which| match which {
                    Endpoint::Gpu => &pair.gpu,
                    Endpoint::Host => &pair.host,
                    Endpoint::Scratch => scratch,
                };
                let a = region(op.from);
                let b = region(op.to);
                s.raw.cmd_copy_buffer(
                    s.command,
                    raw_buffer(a)?,
                    raw_buffer(b)?,
                    &[vk::BufferCopy::default()
                        .src_offset(a.offset)
                        .dst_offset(b.offset)
                        .size(op.bytes)],
                );
                // Original scratch is reused per role. This includes transfer
                // WAR/WAW, not only incoming scratch RAW dependencies.
                s.barrier(
                    vk::PipelineStageFlags::TRANSFER,
                    vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE,
                );
            }
            let (stage, access) =
                wgpu::hal::vulkan::CommandEncoder::kuno_public_host_transfer_read_dependency();
            s.barrier(
                stage,
                access,
                vk::PipelineStageFlags::HOST,
                vk::AccessFlags::HOST_READ,
            );
            s.raw.end_command_buffer(s.command).map_err(|_| fail())?;
            // Real original MAIN->copy visibility, even for a cold source load.
            // The existing rebase join already completed all source readers.
            s.state.attempt(0)?;
            s.raw
                .queue_submit(
                    s.primary,
                    &[vk::SubmitInfo::default().signal_semaphores(std::slice::from_ref(&s.ready))],
                    s.fences[0],
                )
                .map_err(|_| fail())?;
            s.state.attempt(1)?;
            s.raw
                .queue_submit(
                    s.copy,
                    &[vk::SubmitInfo::default()
                        .wait_semaphores(std::slice::from_ref(&s.ready))
                        .wait_dst_stage_mask(&[vk::PipelineStageFlags::TRANSFER])
                        .command_buffers(std::slice::from_ref(&s.command))
                        .signal_semaphores(std::slice::from_ref(&s.done))],
                    s.fences[1],
                )
                .map_err(|_| fail())?;
        }
        Ok(s)
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
            );
        }
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
        self.wait(0)?;
        self.wait(1)?;
        // No physical copy was submitted if command construction failed. In
        // that case there is no semaphore to wait and the original root is idle.
        if self.state.needs_visibility() {
            // Submit only after a SUCCESSFUL real copy-fence completion, never
            // after an uncertain/failed copy submission or a hypothetical event.
            self.state.attempt(2)?;
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
    pub(crate) fn finish(mut self) -> KResult<()> {
        self.join()?;
        // Physical root is idle before the arena exclusion is released. The
        // registered HOST authority still needs its real joined rebase finish.
        self.guard.joined().map_err(Error::InvalidFormat)?;
        let token = self.token.take().ok_or_else(fail)?;
        unsafe { self.client.finish_public_external_rebase(token) }.map_err(|_| fail())
    }
}
impl Drop for Flight {
    fn drop(&mut self) {
        if self.join().is_err() {
            std::mem::forget(std::mem::take(&mut self.owners));
            if let Some(d) = self.device.take() {
                std::mem::forget(d);
            }
            return; // guard retires/quarantines the original arena; no cleanup credit
        }
        // Unwinding still joins submitted work BEFORE original arena teardown.
        // A joined guard may already have been consumed by finish().
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
