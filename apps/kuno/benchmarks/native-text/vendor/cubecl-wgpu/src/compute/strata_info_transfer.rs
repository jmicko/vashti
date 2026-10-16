//! Actual current INFO owners copied on the adopted transfer queue. No model
//! content is cached here; retained handles remain under the original I/O ledger.
use crate::WgpuResource;
use ash::{vk, vk::Handle};
use std::collections::HashSet;

pub const HOST_BYTES: usize = 2 * 1024 * 1024;
pub const MAX_COPIES: usize = 4096;
pub(super) struct Copy {
    pub source: WgpuResource,
    pub target: WgpuResource,
}
/// Resolve the same complete original Vulkan owners used by cmd_copy_buffer.
/// Record offsets inside an INFO page do not narrow this operation's writes.
pub(super) fn accesses(
    copies: &[Copy],
) -> Result<Vec<super::strata_schedule::Access>, &'static str> {
    if copies.is_empty() || copies.len() > MAX_COPIES {
        return Err("native INFO copy capacity");
    }
    let mut spans = Vec::new();
    spans
        .try_reserve_exact(copies.len())
        .map_err(|_| "native INFO span allocation")?;
    for c in copies {
        spans.push(copy_span(c)?);
    }
    span_accesses(&spans)
}
fn span_accesses(spans: &[Span]) -> Result<Vec<super::strata_schedule::Access>, &'static str> {
    validate(spans)?;
    let mut accesses = Vec::new();
    accesses
        .try_reserve_exact(spans.len() * 2)
        .map_err(|_| "native INFO access allocation")?;
    if accesses.capacity() * size_of::<super::strata_schedule::Access>() > HOST_BYTES / 2 {
        return Err("native INFO access capacity");
    }
    for s in spans {
        accesses.extend([
            super::strata_schedule::Access {
                buffer: s.source,
                first: 0,
                bytes: s.bytes,
                write: false,
            },
            super::strata_schedule::Access {
                buffer: s.target,
                first: 0,
                bytes: s.bytes,
                write: true,
            },
        ]);
    }
    Ok(accesses)
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Wait {
    pub semaphore: vk::Semaphore,
    pub value: u64,
}
#[derive(Clone, Copy)]
struct Span {
    source: u64,
    target: u64,
    bytes: u64,
}
fn validate(spans: &[Span]) -> Result<(), &'static str> {
    if spans.is_empty() || spans.len() > MAX_COPIES {
        return Err("native INFO copy capacity");
    }
    let mut roots = HashSet::new();
    roots
        .try_reserve(spans.len() * 2)
        .map_err(|_| "native INFO root metadata allocation")?;
    if roots.capacity() * 32 > HOST_BYTES / 2 {
        return Err("native INFO root metadata capacity");
    }
    for s in spans {
        if s.source == 0
            || s.target == 0
            || s.bytes == 0
            || !s.bytes.is_multiple_of(4)
            || !roots.insert(s.source)
            || !roots.insert(s.target)
        {
            return Err("native INFO full-owner alias or extent");
        }
    }
    Ok(())
}
fn raw(r: &WgpuResource) -> Result<vk::Buffer, &'static str> {
    if r.offset != 0 || r.size == 0 || r.size != r.buffer.size() {
        return Err("native INFO full owner required");
    }
    let hal = unsafe { r.buffer.as_hal::<wgpu::hal::api::Vulkan>() }
        .ok_or("native INFO Vulkan owner required")?;
    Ok(unsafe { hal.raw_buffer_handle() })
}
fn copy_span(c: &Copy) -> Result<Span, &'static str> {
    if c.source.size != c.target.size
        || !c
            .source
            .buffer
            .usage()
            .contains(wgpu::BufferUsages::COPY_SRC)
        || !c
            .target
            .buffer
            .usage()
            .contains(wgpu::BufferUsages::COPY_DST)
    {
        return Err("native INFO copy usage or span");
    }
    Ok(Span {
        source: raw(&c.source)?.as_raw(),
        target: raw(&c.target)?.as_raw(),
        bytes: c.source.size,
    })
}
pub(super) struct Uploads {
    device: ash::Device,
    pool: vk::CommandPool,
    command: vk::CommandBuffer,
    timeline: vk::Semaphore,
    fence: vk::Fence,
    epoch: u64,
    joined_epoch: u64,
    pending: bool,
    copies: Vec<Copy>,
}
impl Uploads {
    pub fn new(device: &ash::Device, family: u32) -> Result<Self, &'static str> {
        let mut s = Self {
            device: device.clone(),
            pool: vk::CommandPool::null(),
            command: vk::CommandBuffer::null(),
            timeline: vk::Semaphore::null(),
            fence: vk::Fence::null(),
            epoch: 0,
            joined_epoch: 0,
            pending: false,
            copies: Vec::new(),
        };
        s.pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }
        .map_err(|_| "native INFO command pool")?;
        s.command = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(s.pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .map_err(|_| "native INFO command buffer")?[0];
        let mut kind = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        s.timeline = unsafe {
            device.create_semaphore(
                &vk::SemaphoreCreateInfo::default().push_next(&mut kind),
                None,
            )
        }
        .map_err(|_| "native INFO semaphore")?;
        s.fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .map_err(|_| "native INFO fence")?;
        Ok(s)
    }
    pub fn pending(&self) -> bool {
        self.pending
    }
    pub fn wait(&self) -> Option<Wait> {
        self.pending.then_some(Wait {
            semaphore: self.timeline,
            value: self.epoch,
        })
    }
    /// Only the actual original math-fence owner may credit this dependency.
    pub fn math_fence_joined(&mut self, epoch: u64) -> Result<(), &'static str> {
        record_joined(&mut self.joined_epoch, self.epoch, epoch)
    }
    pub fn needs_visibility(&self) -> bool {
        self.pending && self.joined_epoch < self.epoch
    }
    pub fn complete(&mut self) -> Result<(), &'static str> {
        if self.pending {
            unsafe {
                self.device
                    .wait_for_fences(&[self.fence], true, 120_000_000_000)
            }
            .map_err(|_| "native INFO completion")?;
            self.pending = false;
            self.copies.clear();
        }
        Ok(())
    }
    pub fn submit(&mut self, queue: vk::Queue, copies: Vec<Copy>) -> Result<(), &'static str> {
        if self.pending || copies.capacity() * size_of::<Copy>() > HOST_BYTES / 2 {
            return Err("native INFO pending owner or capacity");
        }
        let mut spans = Vec::new();
        spans
            .try_reserve_exact(copies.len())
            .map_err(|_| "native INFO span allocation")?;
        for c in &copies {
            spans.push(copy_span(c)?);
        }
        validate(&spans)?;
        self.epoch = self
            .epoch
            .checked_add(1)
            .ok_or("native INFO epoch exhausted")?;
        self.copies = copies;
        unsafe {
            self.device
                .reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())
                .map_err(|_| "native INFO pool reset")?;
            self.device
                .reset_fences(&[self.fence])
                .map_err(|_| "native INFO fence reset")?;
            self.device
                .begin_command_buffer(self.command, &vk::CommandBufferBeginInfo::default())
                .map_err(|_| "native INFO begin")?;
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::HOST_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            self.device.cmd_pipeline_barrier(
                self.command,
                vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
            for s in spans {
                self.device.cmd_copy_buffer(
                    self.command,
                    vk::Buffer::from_raw(s.source),
                    vk::Buffer::from_raw(s.target),
                    &[vk::BufferCopy::default().size(s.bytes)],
                );
            }
            self.device
                .end_command_buffer(self.command)
                .map_err(|_| "native INFO end")?;
        }
        let semaphores = [self.timeline];
        let values = [self.epoch];
        let mut timeline =
            vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&values);
        let submit = vk::SubmitInfo::default()
            .command_buffers(std::slice::from_ref(&self.command))
            .signal_semaphores(&semaphores)
            .push_next(&mut timeline);
        self.pending = true; // retain even after a partially failed submission
        unsafe { self.device.queue_submit(queue, &[submit], self.fence) }
            .map_err(|_| "native INFO submit")
    }
}
impl Drop for Uploads {
    fn drop(&mut self) {
        if self.complete().is_err() {
            std::mem::forget(std::mem::take(&mut self.copies));
            return; // unknown pending use: device/process teardown owns handles
        }
        unsafe {
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_semaphore(self.timeline, None);
            self.device.destroy_command_pool(self.pool, None);
        }
    }
}
const _: () = assert!(
    MAX_COPIES
        * (2 * size_of::<Copy>()
            + 2 * size_of::<Span>()
            + 2 * size_of::<super::strata_schedule::Access>()
            + 128)
        <= HOST_BYTES
);

fn record_joined(joined: &mut u64, latest: u64, epoch: u64) -> Result<(), &'static str> {
    if epoch > latest {
        return Err("native joined INFO epoch not submitted");
    }
    *joined = (*joined).max(epoch);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn whole_info_writes_conflict_even_when_recorded_uniform_views_are_disjoint() {
        use super::super::strata_schedule::{Access, NodeAccess, Planner};
        let incoming = span_accesses(&[Span {
            source: 1,
            target: 9,
            bytes: 4096,
        }])
        .unwrap();
        for (buffer, first, write, expected) in [
            (9, 0, false, true),
            (9, 3584, false, true),
            (9, 1024, true, true),
            (1, 1024, false, false),
            (1, 1024, true, true),
            (3, 1024, true, false),
        ] {
            let mut planner = Planner::new(1, false).unwrap();
            planner
                .push(&NodeAccess {
                    copy: false,
                    release_after: false,
                    accesses: vec![Access {
                        buffer,
                        first,
                        bytes: 64,
                        write,
                    }],
                })
                .unwrap();
            assert_eq!(
                planner.finish().unwrap().conflicts(&incoming).unwrap(),
                expected
            );
        }
        assert_eq!(incoming.len(), 2);
        assert_eq!(incoming[0].first, 0);
        assert_eq!(incoming[0].bytes, 4096);
        assert!(!incoming[0].write);
        assert_eq!(incoming[1].first, 0);
        assert_eq!(incoming[1].bytes, 4096);
        assert!(incoming[1].write);
    }
    #[test]
    fn joined_math_credits_only_its_original_info_wait_not_a_newer_copy() {
        let mut seen = 0;
        record_joined(&mut seen, 2, 1).unwrap();
        assert_eq!(seen, 1); // newer copy still requires visibility/completion
        record_joined(&mut seen, 2, 0).unwrap();
        assert_eq!(seen, 1);
        assert!(record_joined(&mut seen, 2, 3).is_err());
        assert_eq!(seen, 1);
        record_joined(&mut seen, 2, 2).unwrap();
        record_joined(&mut seen, 2, 1).unwrap();
        assert_eq!(seen, 2); // another lane cannot regress the accepted epoch
    }
    #[test]
    fn actual_current_info_copy_inventory_requires_distinct_complete_roots() {
        let s = [
            Span {
                source: 1,
                target: 2,
                bytes: 4096,
            },
            Span {
                source: 3,
                target: 4,
                bytes: 8192,
            },
        ];
        assert!(validate(&s).is_ok());
        for bad in [
            Span { source: 0, ..s[1] },
            Span { target: 0, ..s[1] },
            Span { source: 2, ..s[1] },
            Span { source: 1, ..s[1] },
            Span { target: 2, ..s[1] },
            Span { target: 3, ..s[1] },
            Span { bytes: 0, ..s[1] },
            Span {
                bytes: 4095,
                ..s[1]
            },
        ] {
            assert!(validate(&[s[0], bad]).is_err());
        }
    }
    #[test]
    fn full_native_info_page_bound_is_real_and_empty_or_excess_rejects() {
        let mut s: Vec<_> = (0..MAX_COPIES)
            .map(|i| Span {
                source: (2 * i + 1) as u64,
                target: (2 * i + 2) as u64,
                bytes: 4096,
            })
            .collect();
        assert!(validate(&s).is_ok());
        s.push(Span {
            source: 10001,
            target: 10002,
            bytes: 4096,
        });
        assert!(validate(&s).is_err());
        assert!(validate(&[]).is_err());
    }
}
