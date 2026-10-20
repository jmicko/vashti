//! Dependency-complete native primary/copy/shared submissions. No model content.
use ash::vk;
use std::collections::HashMap;
#[path = "strata_rolling.rs"]
mod rolling;
#[path = "strata_submit.rs"]
mod submit;
pub(crate) use rolling::{
    Submission as RollingSubmission, arm_public_failure, public_failure_stats,
};

/// Recording is optional for authentic uncaptured target/draft work. A named
/// recording must belong to this request and an inference (not commit) family.
/// This preflight grants neither a runtime scope nor device completion.
pub(crate) fn shared_recording(
    request: u64,
    recording: Option<(u64, cubecl_runtime::server::NativeStrataBranch)>,
) -> Result<(), &'static str> {
    if request == 0
        || recording.is_some_and(|(nonce, branch)| {
            nonce != request
                || (branch.target_geometry().is_none() && branch.mtp_geometry().is_none())
        })
    {
        return Err("shared foreign inference recording");
    }
    Ok(())
}
pub(crate) fn shared_queue(
    request: u64,
    lane: usize,
    draft_request: Option<u64>,
) -> Result<(), &'static str> {
    // The actual Graph scope, not a caller flag, authenticates draft work.
    let original = match (lane, draft_request) {
        (0, None) => true,
        (1, Some(original)) => original == request,
        _ => false,
    };
    if request == 0 || !original {
        return Err("shared requires authentic target or draft scope");
    }
    Ok(())
}
/// Two adopted compute queues. Target's side uses draft; draft's side uses
/// MAIN, ordered after already submitted commit work on that physical queue.
/// No additional hardware queue, ownership grant or completion is fabricated.
pub(crate) fn shared_peer(lane: usize) -> Result<usize, &'static str> {
    match lane {
        0 => Ok(1),
        1 => Ok(0),
        _ => Err("shared compute lane"),
    }
}

pub(crate) struct Access {
    pub buffer: u64,
    pub first: u64,
    pub bytes: u64,
    pub write: bool,
}
pub(crate) struct NodeAccess {
    pub copy: bool,
    pub shared: bool,
    pub release_after: bool,
    pub accesses: Vec<Access>,
}
#[derive(Debug)]
pub(crate) struct Segment {
    pub copy: bool,
    pub shared: bool,
    pub waits: [u64; 3],
    pub wait: u64,
    pub signal: u64,
}
impl Segment {
    pub fn queue(&self) -> usize {
        if self.shared {
            2
        } else {
            usize::from(self.copy)
        }
    }
}
pub(crate) struct Plan {
    pub segments: Vec<Segment>,
    pub node_segments: Vec<usize>,
    pub last_copy: u64,
    pub last_shared: u64,
    // Original descriptor/copy intervals, canonicalized without filling gaps.
    // Retained with the actual recording, not rebuilt from caller declarations.
    accesses: Vec<Access>,
}
impl Plan {
    pub fn accesses(&self) -> &[Access] {
        &self.accesses
    }
    /// Original canonical recorded descriptor/physical/indirect/copy owners.
    /// A whole dedicated I/O overwrite conflicts with ANY range on that buffer.
    /// No new allocation, caller-declared interval or data read is involved.
    pub fn buffer_ids(&self) -> impl Iterator<Item = u64> + '_ {
        self.accesses.iter().enumerate().filter_map(|(i, a)| {
            (i == 0 || self.accesses[i - 1].buffer != a.buffer).then_some(a.buffer)
        })
    }
    pub fn conflicts(&self, incoming: &[Access]) -> Result<bool, &'static str> {
        let mut conflict = false;
        for access in incoming {
            let end = access
                .first
                .checked_add(access.bytes)
                .filter(|_| access.buffer != 0 && access.bytes != 0)
                .ok_or("native pending access extent")?;
            if conflict {
                continue; // Still validate every later extent before joining.
            }
            // Original ranges are sorted and disjoint within each buffer.
            let first = self.accesses.partition_point(|old| {
                old.buffer < access.buffer
                    || old.buffer == access.buffer && old.first + old.bytes <= access.first
            });
            for old in &self.accesses[first..] {
                if old.buffer != access.buffer || old.first >= end {
                    break;
                }
                if old.write || access.write {
                    conflict = true;
                    break;
                }
            }
        }
        Ok(conflict)
    }
}
#[derive(Clone, Copy, Default, PartialEq, Eq)]
struct Hazard {
    readers: [u64; 3],
    writers: [u64; 3],
}
#[derive(Clone, Copy)]
struct Region {
    first: u64,
    end: u64,
    hazard: Hazard,
}
const MAX_REGIONS: usize = 262144;
/// Actual descriptor/copy intervals, not invented narrower declarations.
/// Segment0 releases Wgpu initialization before either queue's native work.
pub const NATIVE_QUEUE_HOST_BYTES: usize = 144 * 1024 * 1024
    + submit::HOST_BYTES
    + 2 * super::strata_info_transfer::HOST_BYTES
    + super::strata_io_reuse::HOST_BYTES
    + super::strata_allocator_scope::HOST_BYTES
    + 4096; // Bounded rolling lifecycle/device/queue/recovery-fence metadata.

fn insert_region(
    regions: &mut Vec<Region>,
    at: usize,
    region: Region,
    total: &mut usize,
) -> Result<(), &'static str> {
    if *total >= MAX_REGIONS {
        return Err("native queue interval metadata capacity");
    }
    regions.insert(at, region);
    *total += 1;
    Ok(())
}
fn split_region(regions: &mut Vec<Region>, at: u64, total: &mut usize) -> Result<(), &'static str> {
    let i = regions.partition_point(|r| r.end <= at);
    if i < regions.len() && regions[i].first < at {
        let original = regions[i];
        insert_region(
            regions,
            i + 1,
            Region {
                first: at,
                ..original
            },
            total,
        )?;
        regions[i].end = at;
    }
    Ok(())
}
fn update_regions(
    regions: &mut Vec<Region>,
    access: &Access,
    own: usize,
    signal: u64,
    total: &mut usize,
) -> Result<[u64; 3], &'static str> {
    let end = access
        .first
        .checked_add(access.bytes)
        .filter(|_| access.bytes != 0)
        .ok_or("native queue access interval")?;
    split_region(regions, access.first, total)?;
    split_region(regions, end, total)?;
    let mut i = regions.partition_point(|r| r.end <= access.first);
    let mut cursor = access.first;
    let mut wait = [0; 3];
    while cursor < end {
        if i == regions.len() || regions[i].first > cursor {
            let last = regions.get(i).map_or(end, |r| end.min(r.first));
            insert_region(
                regions,
                i,
                Region {
                    first: cursor,
                    end: last,
                    hazard: Hazard::default(),
                },
                total,
            )?;
        }
        let cell = &mut regions[i];
        let h = &mut cell.hazard;
        for peer in 0..3 {
            if peer != own {
                wait[peer] = wait[peer].max(h.writers[peer]);
                if access.write {
                    wait[peer] = wait[peer].max(h.readers[peer]);
                }
            }
        }
        if access.write {
            h.writers[own] = signal;
        } else {
            h.readers[own] = signal;
        }
        cursor = cell.end;
        i += 1;
    }
    // Rejoin equal neighboring states after edits so repeated views do not
    // accumulate permanent fragmentation. Outside intervals keep their history.
    let mut keep = 0usize;
    for read in 0..regions.len() {
        let value = regions[read];
        if keep > 0
            && regions[keep - 1].end == value.first
            && regions[keep - 1].hazard == value.hazard
        {
            regions[keep - 1].end = value.end;
        } else {
            regions[keep] = value;
            keep += 1;
        }
    }
    *total -= regions.len() - keep;
    regions.truncate(keep);
    Ok(wait)
}

pub(crate) struct Planner {
    result: Plan,
    hazards: HashMap<u64, Vec<Region>>,
    readonly: Vec<Access>,
    regions: usize,
    region_capacity: usize,
    capacity: usize,
    serialize: bool,
    inline_copies: bool,
    release_next: bool,
}
impl Planner {
    pub fn new(capacity: usize, serialize: bool) -> Result<Self, &'static str> {
        if capacity == 0 || capacity > 32768 {
            return Err("native queue plan node capacity");
        }
        let mut segments = Vec::with_capacity(capacity + 1);
        segments.push(Segment {
            copy: false,
            shared: false,
            waits: [0; 3],
            wait: 0,
            signal: 1,
        });
        Ok(Self {
            result: Plan {
                segments,
                node_segments: Vec::with_capacity(capacity),
                last_copy: 0,
                last_shared: 0,
                accesses: Vec::new(),
            },
            hazards: HashMap::new(),
            readonly: Vec::new(),
            regions: 0,
            region_capacity: 0,
            capacity,
            serialize,
            inline_copies: false,
            release_next: false,
        })
    }
    /// Actual original lane policy. MTP has its own compute queue, which also
    /// supports transfer commands; keep its copies ordered there, independently
    /// of expert adaptation. Main/prompt retains streamed dual-queue execution.
    pub fn for_lane(capacity: usize, serialize: bool, lane: usize) -> Result<Self, &'static str> {
        if lane > 1 {
            return Err("native queue plan lane");
        }
        let mut planner = Self::new(capacity, serialize)?;
        planner.inline_copies = lane == 1;
        Ok(planner)
    }
    /// Complete certified full-root read declarations. Internal tracking is
    /// factored only after an authentic whole-recording no-writer proof.
    pub fn retain_readonly(
        &mut self,
        views: impl Iterator<Item = Access>,
    ) -> Result<(), &'static str> {
        if !self.result.node_segments.is_empty() || !self.readonly.is_empty() {
            return Err("native readonly footprint order");
        }
        for v in views {
            if v.buffer == 0
                || v.first != 0
                || v.bytes == 0
                || v.write
                || self.readonly.last().is_some_and(|p| p.buffer >= v.buffer)
                || self.readonly.len() == 65536
            {
                return Err("native readonly full footprint inventory");
            };
            self.readonly.push(v)
        }
        Ok(())
    }
    pub fn push(&mut self, node: &NodeAccess) -> Result<(), &'static str> {
        self.push_accesses(node.copy, node.shared, node.release_after, &node.accesses)
    }
    /// Borrow authentic access intervals so capture can reuse its temporary
    /// vector. The complete original planning and hazard logic is unchanged.
    pub fn push_accesses(
        &mut self,
        node_copy: bool,
        node_shared: bool,
        release_after: bool,
        accesses: &[Access],
    ) -> Result<(), &'static str> {
        // This selects only the queue. The original node still carries its
        // COPY/FILL dispatch and TRANSFER accesses in Graph's actual encoder.
        let shared = node_shared;
        // Shared COPY/FILL commands execute on its compute-capable queue.
        let copy = node_copy && !self.inline_copies && !shared;
        let result = &mut self.result;
        let index = result.node_segments.len();
        if index >= self.capacity {
            return Err("native queue plan node capacity");
        }
        if index == 0
            || self.release_next
            || result.segments.last().unwrap().copy != copy
            || result.segments.last().unwrap().shared != shared
        {
            let signal = result.segments.len() as u64 + 1;
            result.segments.push(Segment {
                copy,
                shared,
                waits: [u64::from(copy || shared), 0, 0],
                wait: u64::from(copy || shared),
                signal,
            });
            if copy {
                result.last_copy = signal;
            }
            if shared {
                result.last_shared = signal;
            }
        }
        self.release_next = release_after;
        let s = result.segments.len() - 1;
        result.node_segments.push(s);
        let segment = &mut result.segments[s];
        let own = if shared { 2 } else { usize::from(copy) };
        for access in accesses {
            if self
                .readonly
                .binary_search_by_key(&access.buffer, |a| a.buffer)
                .is_ok()
            {
                return Err("native readonly proof unexpected unfactored access");
            }
            if access.buffer == 0 {
                return Err("native queue plan null buffer");
            }
            // Existing native streams have at most4096 imported roots plus
            // bounded private/INFO owners. Bound temporary hashing as well as
            // persistent command metadata, rather than copying all node lists.
            if !self.hazards.contains_key(&access.buffer) && self.hazards.len() >= 65536 {
                return Err("native queue root metadata capacity");
            }
            let regions = self.hazards.entry(access.buffer).or_default();
            let previous_capacity = regions.capacity();
            let waits = update_regions(regions, access, own, segment.signal, &mut self.regions)?;
            for (old, new) in segment.waits.iter_mut().zip(waits) {
                *old = (*old).max(new);
            }
            segment.wait = *segment.waits.iter().max().unwrap();
            // A bound on live intervals alone does not bound old Vec capacities
            // after contraction on many different buffers.
            if regions.capacity() > regions.len().saturating_mul(2) {
                regions.shrink_to_fit();
            }
            self.region_capacity = self
                .region_capacity
                .checked_sub(previous_capacity)
                .and_then(|n| n.checked_add(regions.capacity()))
                .ok_or("native queue interval capacity overflow")?;
            if self.region_capacity > MAX_REGIONS * 2 {
                return Err("native queue interval reserved capacity");
            }
        }
        Ok(())
    }
    /// Read only the actual queue selected for the most recent original node.
    pub fn last_segment(&self) -> Result<&Segment, &'static str> {
        self.result
            .node_segments
            .last()
            .and_then(|i| self.result.segments.get(*i))
            .ok_or("native queue original last node missing")
    }
    pub fn finish(mut self) -> Result<Plan, &'static str> {
        if self.result.node_segments.len() != self.capacity {
            return Err("native queue plan incomplete");
        }
        if self.serialize {
            for s in 1..self.result.segments.len() {
                let previous = &self.result.segments[s - 1];
                let peer = previous.queue();
                let signal = previous.signal;
                let current = &mut self.result.segments[s];
                if current.queue() != peer {
                    current.waits[peer] = current.waits[peer].max(signal);
                    current.wait = *current.waits.iter().max().unwrap();
                }
            }
        }
        let mut accesses: Vec<Access> = Vec::new();
        accesses
            .try_reserve_exact(
                self.regions
                    .checked_add(self.readonly.len())
                    .ok_or("native readonly footprint size overflow")?,
            )
            .map_err(|_| "native retained access allocation")?;
        if accesses.capacity() > MAX_REGIONS * 2 {
            return Err("native retained access reserved capacity");
        }
        for (buffer, regions) in &self.hazards {
            for region in regions {
                let write = region.hazard.writers != [0; 3];
                if !write && region.hazard.readers == [0; 3] {
                    return Err("native retained empty access history");
                }
                accesses.push(Access {
                    buffer: *buffer,
                    first: region.first,
                    bytes: region.end - region.first,
                    write,
                });
            }
        }
        accesses.extend(self.readonly.drain(..));
        accesses.sort_unstable_by_key(|a| (a.buffer, a.first));
        let mut keep = 0;
        for read in 0..accesses.len() {
            if keep > 0
                && accesses[keep - 1].buffer == accesses[read].buffer
                && accesses[keep - 1].write == accesses[read].write
                && accesses[keep - 1].first + accesses[keep - 1].bytes == accesses[read].first
            {
                let extension = accesses[read].bytes;
                accesses[keep - 1].bytes += extension;
            } else {
                accesses.swap(keep, read);
                keep += 1;
            }
        }
        accesses.truncate(keep);
        // Capacity remains bounded even when many old segment distinctions
        // compress to a tiny union. This is charged by Commands::metadata_bytes.
        if accesses.capacity() > accesses.len().saturating_mul(2) {
            accesses.shrink_to_fit();
        }
        self.result.accesses = accesses;
        Ok(self.result)
    }
}
#[cfg(test)]
fn plan(nodes: &[NodeAccess], serialize: bool) -> Result<Plan, &'static str> {
    let mut planner = Planner::new(nodes.len(), serialize)?;
    for node in nodes {
        planner.push(node)?;
    }
    planner.finish()
}

fn reserve_epoch(next: &mut u64, segments: usize) -> Result<u64, &'static str> {
    let base = *next;
    *next = base
        .checked_add(segments as u64 + 1)
        .ok_or("native queue timeline exhausted")?;
    Ok(base)
}

/// The actual submission uses this dependency, preserving the upload wait at
/// the original compute header and the plan's later compute/copy release edges.
fn incoming_dependencies(
    index: usize,
    segment: &Segment,
    timelines: [vk::Semaphore; 3],
    base: u64,
    uploads: Option<super::strata_info_transfer::Wait>,
) -> [Option<super::strata_info_transfer::Wait>; 3] {
    if index == 0 {
        return [uploads, None, None];
    }
    std::array::from_fn(|peer| {
        (segment.waits[peer] != 0).then_some(super::strata_info_transfer::Wait {
            semaphore: timelines[peer],
            value: base + segment.waits[peer],
        })
    })
}
#[cfg(test)]
fn incoming_dependency(
    index: usize,
    segment: &Segment,
    timelines: [vk::Semaphore; 2],
    base: u64,
    uploads: Option<super::strata_info_transfer::Wait>,
) -> Option<super::strata_info_transfer::Wait> {
    incoming_dependencies(
        index,
        segment,
        [timelines[0], timelines[1], vk::Semaphore::null()],
        base,
        uploads,
    )
    .into_iter()
    .flatten()
    .next()
}

// Only construction failures free MAIN buffers individually: the Graph still
// owns that pool. Once construction succeeds, the original Recording owns and
// destroys the whole pool; its field Drop ordering must not free them again.
struct ConstructingCommands<'a, F: FnMut(vk::CommandBuffer)> {
    plan: &'a Plan,
    commands: &'a mut Vec<vk::CommandBuffer>,
    scratch: &'a mut Vec<vk::CommandBuffer>,
    pending_queue: Option<usize>,
    release_main: F,
    armed: bool,
}
impl<F: FnMut(vk::CommandBuffer)> Drop for ConstructingCommands<'_, F> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        // A callback panic after writing actual handles but before adoption is
        // covered too. COPY/SHARED buffers remain owned by their original pools.
        if self.pending_queue == Some(0) {
            for &handle in self
                .scratch
                .iter()
                .filter(|h| **h != vk::CommandBuffer::null())
            {
                (self.release_main)(handle);
            }
        }
        for (segment, &handle) in self.plan.segments.iter().zip(self.commands.iter()).skip(1) {
            if segment.queue() == 0 && handle != vk::CommandBuffer::null() {
                (self.release_main)(handle);
            }
        }
    }
}
// Original commands are allocated once per ORIGINAL queue-family pool. Both
// actual Vec capacities are charged before Vulkan runs. Successful handles are
// immediately adopted into the real Commands vector before the next fallible
// operation; the construction guard handles failure/unwind before that point.
fn allocate_commands(
    plan: &Plan,
    header: vk::CommandBuffer,
    budget: usize,
    commands: &mut Vec<vk::CommandBuffer>,
    mut allocate: impl FnMut(usize, &mut [vk::CommandBuffer]) -> Result<(), &'static str>,
    release_main: impl FnMut(vk::CommandBuffer),
) -> Result<(), &'static str> {
    let count = plan.segments.len();
    if !(1..=32769).contains(&count)
        || header == vk::CommandBuffer::null()
        || !commands.is_empty()
        || plan.segments[0].queue() != 0
        || plan.segments.iter().any(|s| s.copy && s.shared)
    {
        return Err("native command allocation original inventory");
    }
    let mut counts = [0usize; 3];
    for segment in &plan.segments[1..] {
        counts[segment.queue()] += 1;
    }
    let scratch_count = *counts.iter().max().unwrap();
    let fixed = plan_metadata_bytes(plan)
        .and_then(|n| n.checked_add(size_of::<Commands>()))
        .ok_or("native command allocation metadata overflow")?;
    let peak = |commands: usize, scratch: usize| {
        commands
            .checked_add(scratch)
            .and_then(|n| n.checked_mul(size_of::<vk::CommandBuffer>()))
            .and_then(|n| n.checked_add(fixed))
            .filter(|&n| n <= budget)
            .ok_or("native command allocation metadata capacity")
    };
    peak(count, scratch_count)?;
    commands
        .try_reserve_exact(count)
        .map_err(|_| "native command handles allocation")?;
    peak(commands.capacity(), scratch_count)?;
    let mut scratch = Vec::new();
    scratch
        .try_reserve_exact(scratch_count)
        .map_err(|_| "native command temporary allocation")?;
    peak(commands.capacity(), scratch.capacity())?;
    commands.resize(count, vk::CommandBuffer::null());
    commands[0] = header;
    scratch.resize(scratch_count, vk::CommandBuffer::null());
    let mut guard = ConstructingCommands {
        plan,
        commands,
        scratch: &mut scratch,
        pending_queue: None,
        release_main,
        armed: true,
    };
    for (queue, &own_count) in counts.iter().enumerate() {
        if own_count == 0 {
            continue;
        }
        guard.pending_queue = Some(queue);
        let status = allocate(queue, &mut guard.scratch[..own_count]);
        // Adoption is unconditional, including a partial/failed allocation. No
        // allocation, validation or fallible driver call occurs in this loop.
        let mut own = 0;
        for (index, segment) in plan.segments.iter().enumerate().skip(1) {
            if segment.queue() == queue {
                guard.commands[index] =
                    std::mem::replace(&mut guard.scratch[own], vk::CommandBuffer::null());
                own += 1;
            }
        }
        guard.scratch[..own_count].fill(vk::CommandBuffer::null());
        guard.pending_queue = None;
        status?;
        if guard
            .commands
            .iter()
            .zip(&plan.segments)
            .skip(1)
            .any(|(h, s)| s.queue() == queue && *h == vk::CommandBuffer::null())
        {
            return Err("native command allocated null handle");
        }
    }
    guard.armed = false;
    Ok(())
}
pub(crate) fn plan_metadata_bytes(plan: &Plan) -> Option<usize> {
    plan.segments
        .capacity()
        .checked_mul(size_of::<Segment>())?
        .checked_add(
            plan.node_segments
                .capacity()
                .checked_mul(size_of::<usize>())?,
        )?
        .checked_add(plan.accesses.capacity().checked_mul(size_of::<Access>())?)
}

/// Recording owns all copy pools and timeline semaphores through the joined
/// fence. Main command buffers belong to its existing compute pool.
pub(crate) struct Commands {
    device: ash::Device,
    copy_pool: vk::CommandPool,
    shared_pool: vk::CommandPool,
    timelines: [vk::Semaphore; 3],
    next_base: u64,
    pub plan: Plan,
    pub commands: Vec<vk::CommandBuffer>,
}
impl Commands {
    pub fn new(
        device: &ash::Device,
        main_pool: vk::CommandPool,
        header: vk::CommandBuffer,
        copy_family: u32,
        shared_family: u32,
        plan: Plan,
        metadata_budget: usize,
    ) -> Result<Self, &'static str> {
        let mut result = Self {
            device: device.clone(),
            copy_pool: vk::CommandPool::null(),
            shared_pool: vk::CommandPool::null(),
            timelines: [vk::Semaphore::null(); 3],
            next_base: 0,
            commands: Vec::new(),
            plan,
        };
        result.copy_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                    .queue_family_index(copy_family),
                None,
            )
        }
        .map_err(|_| "native copy command pool")?;
        if result.plan.last_shared != 0 {
            result.shared_pool = unsafe {
                device.create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                        .queue_family_index(shared_family),
                    None,
                )
            }
            .map_err(|_| "native shared command pool")?;
        }
        for semaphore in &mut result.timelines {
            let mut timeline = vk::SemaphoreTypeCreateInfo::default()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(0);
            *semaphore = unsafe {
                device.create_semaphore(
                    &vk::SemaphoreCreateInfo::default().push_next(&mut timeline),
                    None,
                )
            }
            .map_err(|_| "native queue timeline")?;
        }
        allocate_commands(
            &result.plan,
            header,
            metadata_budget,
            &mut result.commands,
            |queue, handles| {
                let info = vk::CommandBufferAllocateInfo::default()
                    .command_pool(match queue {
                        0 => main_pool,
                        1 => result.copy_pool,
                        2 => result.shared_pool,
                        _ => return Err("native command original pool"),
                    })
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(handles.len() as u32);
                // The output slice was allocated, initialized and charged above;
                // unlike ash's allocating convenience wrapper, no hidden Vec exists.
                let status = unsafe {
                    (device.fp_v1_0().allocate_command_buffers)(
                        device.handle(),
                        &info,
                        handles.as_mut_ptr(),
                    )
                };
                if status == vk::Result::SUCCESS {
                    Ok(())
                } else {
                    Err("native queue command allocation")
                }
            },
            |handle| unsafe { device.free_command_buffers(main_pool, &[handle]) },
        )?;
        Ok(result)
    }
    pub fn metadata_bytes(&self) -> usize {
        self.plan.segments.capacity() * std::mem::size_of::<Segment>()
            + self.plan.node_segments.capacity() * std::mem::size_of::<usize>()
            + self.commands.capacity() * std::mem::size_of::<vk::CommandBuffer>()
            + self.plan.accesses.capacity() * std::mem::size_of::<Access>()
            + std::mem::size_of::<Self>()
    }
    fn prepare_submission(
        &mut self,
        queues: [vk::Queue; 3],
        copy_queue: &wgpu::hal::vulkan::KunoTransferQueue,
        uploads: Option<super::strata_info_transfer::Wait>,
    ) -> Result<(u64, submit::Batches), &'static str> {
        // Both native graph copies and the original adaptive worker submit
        // through the SAME actual adopted queue authority. Compute queues stay
        // on their original serialized engine thread; no cross-queue lock.
        if queues[0] == vk::Queue::null()
            || queues[1] == vk::Queue::null()
            || queues[0] == queues[1]
            || !copy_queue.is_original(self.device.handle(), queues[1])
            || (self.plan.last_shared != 0
                && (queues[2] == vk::Queue::null()
                    || queues[2] == queues[0]
                    || queues[2] == queues[1]))
        {
            return Err("native original copy submission authority");
        }
        // Reserve before any submission; even a partial failure never reuses a
        // timeline value. The caller retains this entire recording on failure.
        let base = reserve_epoch(&mut self.next_base, self.plan.segments.len())?;
        if self
            .plan
            .segments
            .first()
            .is_none_or(|s| s.copy || s.shared || s.wait != 0)
        {
            return Err("native upload dependency requires original compute header");
        }
        let batches = submit::Batches::new_with_budget(
            &self.plan,
            &self.commands,
            self.timelines,
            base,
            uploads,
            submit::HOST_BYTES,
        )?;
        Ok((base, batches))
    }
    pub fn begin_rolling(
        &mut self,
        queues: [vk::Queue; 3],
        copy_queue: &wgpu::hal::vulkan::KunoTransferQueue,
        fence: vk::Fence,
        uploads: Option<super::strata_info_transfer::Wait>,
    ) -> Result<RollingSubmission, &'static str> {
        let (base, batches) = self.prepare_submission(queues, copy_queue, uploads)?;
        RollingSubmission::new(
            &self.device,
            queues,
            copy_queue,
            batches,
            fence,
            self.timelines,
            base,
            [0, self.plan.last_copy, self.plan.last_shared],
            self.plan.segments.len(),
        )
    }
    pub fn submit(
        &mut self,
        queues: [vk::Queue; 3],
        copy_queue: &wgpu::hal::vulkan::KunoTransferQueue,
        fence: vk::Fence,
        uploads: Option<super::strata_info_transfer::Wait>,
    ) -> Result<(), &'static str> {
        let (base, batches) = self.prepare_submission(queues, copy_queue, uploads)?;
        // These semaphores are created as TIMELINE above, never binary. Vulkan
        // timeline waits may precede their signal submission on another queue.
        // Group only by the actual queue; original per-queue command order,
        // every signal/wait value and every segment command remain unchanged.
        // A single call per queue avoids tens of thousands of driver entries.
        // No host/device completion is granted here. On a partial submission
        // failure the original recording/owners remain retained by the caller.
        for (own, &queue) in queues.iter().enumerate() {
            batches
                .with_queue(own, |submits| {
                    if submits.is_empty() {
                        return Ok(());
                    }
                    unsafe {
                        if own == 1 {
                            copy_queue.submit(submits, vk::Fence::null())
                        } else {
                            self.device.queue_submit(queue, submits, vk::Fence::null())
                        }
                    }
                })?
                .map_err(|_| "native compute/copy batch submit")?;
        }
        // The original target fence owns all three queues, including peers
        // with no following compute consumer. Replays retain the same owners.
        join_queues(
            &self.device,
            queues[0],
            self.timelines,
            base,
            [0, self.plan.last_copy, self.plan.last_shared],
            fence,
        )
    }
}
fn join_queues(
    device: &ash::Device,
    primary: vk::Queue,
    timelines: [vk::Semaphore; 3],
    base: u64,
    last: [u64; 3],
    fence: vk::Fence,
) -> Result<(), &'static str> {
    let mut waits = [vk::Semaphore::null(); 2];
    let mut values = [0; 2];
    let mut count = 0;
    for (peer, last) in [(1, last[1]), (2, last[2])] {
        if last != 0 {
            waits[count] = timelines[peer];
            values[count] = base + last;
            count += 1;
        }
    }
    let stages = [vk::PipelineStageFlags::ALL_COMMANDS; 2];
    let mut timeline =
        vk::TimelineSemaphoreSubmitInfo::default().wait_semaphore_values(&values[..count]);
    let join = vk::SubmitInfo::default()
        .wait_semaphores(&waits[..count])
        .wait_dst_stage_mask(&stages[..count])
        .push_next(&mut timeline);
    unsafe { device.queue_submit(primary, &[join], fence) }.map_err(|_| "native all queue join")
}
impl Drop for Commands {
    fn drop(&mut self) {
        unsafe {
            for semaphore in self.timelines {
                if semaphore != vk::Semaphore::null() {
                    self.device.destroy_semaphore(semaphore, None);
                }
            }
            if self.shared_pool != vk::CommandPool::null() {
                self.device.destroy_command_pool(self.shared_pool, None);
            }
            if self.copy_pool != vk::CommandPool::null() {
                self.device.destroy_command_pool(self.copy_pool, None);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn retained_inventory_preserves_readers_writes_boundaries_and_real_gaps() {
        let nodes: Vec<_> = [(0, 4, false), (8, 4, false), (16, 8, true), (16, 8, false)]
            .into_iter()
            .map(|(first, bytes, write)| NodeAccess {
                shared: false,
                copy: false,
                release_after: true,
                accesses: vec![Access {
                    buffer: 7,
                    first,
                    bytes,
                    write,
                }],
            })
            .collect();
        let p = plan(&nodes, false).unwrap();
        for (buffer, first, bytes, write, expected) in [
            (7, 4, 4, true, false),  // Unused hole, not an invented union extent.
            (7, 0, 4, false, false), // Independent reader of a read-only span.
            (7, 0, 4, true, true),
            (7, 3, 2, true, true),
            (7, 12, 4, true, false),
            (7, 16, 8, false, true), // A later reader must not erase prior writes.
            (7, 24, 4, true, false),
            (8, 16, 8, true, false),
        ] {
            assert_eq!(
                p.conflicts(&[Access {
                    buffer,
                    first,
                    bytes,
                    write
                }])
                .unwrap(),
                expected
            );
        }
        for bad in [
            Access {
                buffer: 0,
                first: 0,
                bytes: 4,
                write: true,
            },
            Access {
                buffer: 7,
                first: 0,
                bytes: 0,
                write: true,
            },
            Access {
                buffer: 7,
                first: u64::MAX,
                bytes: 4,
                write: true,
            },
        ] {
            assert!(
                p.conflicts(&[
                    Access {
                        buffer: 7,
                        first: 16,
                        bytes: 4,
                        write: false
                    },
                    bad
                ])
                .is_err()
            ); // Malformed tail is not hidden by an earlier conflict.
        }
    }
    #[test]
    fn retained_union_matches_original_descriptor_accesses_across_both_queues() {
        let mut state = 31u64;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            state
        };
        let nodes: Vec<_> = (0..256)
            .map(|_| NodeAccess {
                shared: false,
                copy: next() & 1 != 0,
                release_after: next() & 1 != 0,
                accesses: (0..3)
                    .map(|_| Access {
                        buffer: 1 + next() % 11,
                        first: next() % 512,
                        bytes: 1 + next() % 32,
                        write: next() & 4 != 0,
                    })
                    .collect(),
            })
            .collect();
        let p = plan(&nodes, false).unwrap();
        for _ in 0..4096 {
            let incoming = Access {
                buffer: 1 + next() % 13,
                first: next() % 640,
                bytes: 1 + next() % 32,
                write: next() & 2 != 0,
            };
            let direct = nodes.iter().flat_map(|n| &n.accesses).any(|old| {
                old.buffer == incoming.buffer
                    && old.first < incoming.first + incoming.bytes
                    && incoming.first < old.first + old.bytes
                    && (old.write || incoming.write)
            });
            assert_eq!(p.conflicts(&[incoming]).unwrap(), direct);
        }
    }
    #[test]
    fn retained_adjacent_readers_compress_without_unbounded_reserved_capacity() {
        let nodes: Vec<_> = (0..2048)
            .map(|first| NodeAccess {
                shared: false,
                copy: false,
                release_after: true,
                accesses: vec![Access {
                    buffer: 7,
                    first,
                    bytes: 1,
                    write: false,
                }],
            })
            .collect();
        let p = plan(&nodes, false).unwrap();
        assert_eq!(p.accesses.len(), 1);
        assert!(p.accesses.capacity() <= 2);
        assert_eq!(p.accesses[0].first, 0);
        assert_eq!(p.accesses[0].bytes, 2048);
    }
    fn node(copy: bool, buffer: u64, write: bool) -> NodeAccess {
        NodeAccess {
            shared: false,
            copy,
            release_after: false,
            accesses: vec![Access {
                buffer,
                first: 0,
                bytes: 4096,
                write,
            }],
        }
    }
    #[test]
    fn copy_acquires_uploads_but_independent_compute_can_overlap() {
        let p = plan(
            &[
                node(false, 1, true),
                node(true, 2, true),
                node(false, 2, false),
            ],
            false,
        )
        .unwrap();
        assert_eq!(p.node_segments, [1, 2, 3]);
        assert_eq!(p.segments[2].wait, 1); // Not the unrelated compute signal2.
        assert_eq!(p.segments[3].wait, 3); // Actual copy -> consumer edge.
        assert_eq!(p.last_copy, 3);
    }
    #[test]
    fn raw_war_waw_and_copy_slot_reuse_wait_for_the_actual_other_queue() {
        for (first_write, second_write) in [(true, false), (false, true), (true, true)] {
            let p = plan(
                &[node(false, 1, first_write), node(true, 1, second_write)],
                false,
            )
            .unwrap();
            assert_eq!(p.segments[2].wait, 2);
        }
        let p = plan(
            &[
                node(true, 1, true),
                node(false, 1, false),
                node(true, 1, true),
            ],
            false,
        )
        .unwrap();
        assert_eq!(p.segments[2].wait, 2);
        assert_eq!(p.segments[3].wait, 3); // Cannot overwrite the live staging slot.
    }
    #[test]
    fn shared_readers_only_acquire_upload_boundary() {
        let p = plan(&[node(false, 1, false), node(true, 1, false)], false).unwrap();
        assert_eq!(p.segments[2].wait, 1);
    }
    #[test]
    fn every_hazard_is_a_backward_edge_and_queue_order_is_retained() {
        let mut state = 17u64;
        let mut nodes = Vec::new();
        for _ in 0..2048 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            nodes.push(node(
                state & 1 != 0,
                1 + ((state >> 4) % 11),
                state & 8 != 0,
            ));
        }
        let p = plan(&nodes, false).unwrap();
        for s in &p.segments {
            assert!(s.wait < s.signal);
        }
        // Independent pairwise oracle covers RAW/WAR/WAW without relying on
        // the planner's compressed last-access bookkeeping.
        for (j, n) in nodes.iter().enumerate() {
            let current = &p.segments[p.node_segments[j]];
            for (i, prior) in nodes[..j].iter().enumerate() {
                if prior.copy != n.copy
                    && prior.accesses[0].buffer == n.accesses[0].buffer
                    && (prior.accesses[0].write || n.accesses[0].write)
                {
                    assert!(current.wait >= p.segments[p.node_segments[i]].signal);
                }
            }
        }
    }
    fn range_node(copy: bool, buffer: u64, first: u64, bytes: u64, write: bool) -> NodeAccess {
        NodeAccess {
            shared: false,
            copy,
            release_after: false,
            accesses: vec![Access {
                buffer,
                first,
                bytes,
                write,
            }],
        }
    }
    #[test]
    fn disjoint_slots_in_one_buffer_overlap_but_reuse_waits_for_its_reader() {
        let p = plan(
            &[
                range_node(true, 1, 0, 4096, true),
                range_node(false, 1, 0, 4096, false),
                range_node(true, 1, 4096, 4096, true),
                range_node(false, 1, 4096, 4096, false),
                range_node(true, 1, 0, 4096, true),
            ],
            false,
        )
        .unwrap();
        assert_eq!(p.segments[2].wait, 2);
        assert_eq!(p.segments[3].wait, 1); // Slot1 need not wait for slot0 reader.
        assert_eq!(p.segments[4].wait, 4);
        assert_eq!(p.segments[5].wait, 3); // Slot0 reuse waits for its real reader.
    }
    #[test]
    fn dequant_release_allows_raw_reuse_before_following_gemms_finish() {
        let mut dequant = range_node(false, 1, 0, 4096, false);
        dequant.release_after = true;
        let p = plan(
            &[
                dequant,
                range_node(false, 2, 0, 4096, true), // GU/GEMM continuation on compute.
                range_node(true, 1, 0, 4096, true),  // Same raw slot, different queue.
            ],
            false,
        )
        .unwrap();
        assert_eq!(p.node_segments, [1, 2, 3]);
        assert_eq!(p.segments[3].wait, 2); // The dequant signal, not GEMM signal3.
        assert!(!p.segments[1].copy && !p.segments[2].copy);
    }
    #[test]
    fn partial_updates_keep_untouched_prefixes_gaps_and_suffixes() {
        let p = plan(
            &[
                range_node(true, 1, 0, 64, true),
                range_node(false, 1, 16, 16, false),
                range_node(true, 1, 48, 16, true),
                range_node(false, 1, 0, 16, false),
                range_node(true, 1, 24, 24, true),
                range_node(false, 1, 56, 8, false),
            ],
            false,
        )
        .unwrap();
        assert_eq!(p.segments[3].wait, 1);
        assert_eq!(p.segments[4].wait, 2);
        assert_eq!(p.segments[5].wait, 3);
        assert_eq!(p.segments[6].wait, 4);
        assert!(plan(&[range_node(true, 1, 0, 0, true)], false).is_err());
        assert!(plan(&[range_node(true, 1, u64::MAX, 2, true)], false).is_err());
    }
    #[test]
    fn mixed_intervals_and_multi_binding_aliases_match_pairwise_byte_oracle() {
        let mut nodes = Vec::new();
        let mut state = 71u64;
        for i in 0..2048 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let mut n = range_node(
                state & 1 != 0,
                1 + ((state >> 4) % 7),
                ((state >> 12) % 64) * 4,
                (1 + ((state >> 20) % 16)) * 4,
                state & 8 != 0,
            );
            if i % 7 == 0 {
                n.accesses.push(Access {
                    buffer: 1 + ((state >> 8) % 7),
                    first: ((state >> 16) % 64) * 4,
                    bytes: 32,
                    write: state & 32 != 0,
                });
            }
            nodes.push(n);
        }
        let p = plan(&nodes, false).unwrap();
        for (j, n) in nodes.iter().enumerate() {
            let current = &p.segments[p.node_segments[j]];
            assert!(current.wait < current.signal);
            for (i, previous) in nodes[..j].iter().enumerate() {
                if previous.copy == n.copy {
                    continue;
                }
                for a in &n.accesses {
                    for b in &previous.accesses {
                        if a.buffer == b.buffer
                            && a.first < b.first + b.bytes
                            && b.first < a.first + a.bytes
                            && (a.write || b.write)
                        {
                            assert!(current.wait >= p.segments[p.node_segments[i]].signal);
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn interval_contraction_releases_reserved_capacity_on_each_buffer() {
        let mut planner = Planner::new(1025, false).unwrap();
        for i in 0..512 {
            planner.push(&range_node(false, 1, i * 4, 4, true)).unwrap();
            planner.push(&range_node(true, 2, 0, 4, false)).unwrap();
        }
        assert_eq!(planner.hazards[&1].len(), 512);
        planner.push(&range_node(false, 1, 0, 2048, true)).unwrap();
        assert_eq!(planner.hazards[&1].len(), 1);
        assert!(planner.hazards[&1].capacity() <= 2);
        assert!(planner.region_capacity <= 4);
        planner.finish().unwrap();
    }
    #[test]
    fn replays_and_partial_epochs_never_reuse_timeline_values() {
        let p = plan(&[node(true, 1, true), node(false, 1, false)], false).unwrap();
        let mut next = 0;
        let mut previous_max = 0;
        for _ in 0..64 {
            let base = reserve_epoch(&mut next, p.segments.len()).unwrap();
            // This also applies if a prior epoch was only partially submitted.
            let first_signal = base + p.segments[0].signal;
            assert!(first_signal > previous_max);
            previous_max = base + p.segments.last().unwrap().signal;
        }
        let mut exhausted = u64::MAX - 1;
        assert!(reserve_epoch(&mut exhausted, p.segments.len()).is_err());
        assert_eq!(exhausted, u64::MAX - 1);
    }
    #[test]
    fn sampled_cross_queue_commands_wait_for_query_reset_and_each_other() {
        let p = plan(
            &[
                node(true, 1, false),
                node(false, 2, false),
                node(true, 3, false),
            ],
            true,
        )
        .unwrap();
        for pair in p.segments.windows(2) {
            if pair[0].copy != pair[1].copy {
                assert!(pair[1].wait >= pair[0].signal);
            }
        }
        assert!(plan(&[], false).is_err());
        assert!(plan(&[node(false, 0, false)], false).is_err());
    }
}

#[cfg(test)]
mod info_dependency_tests {
    use super::*;
    use ash::vk::Handle;
    #[test]
    fn actual_upload_wait_reaches_compute_copy_and_later_math_via_original_release_edges() {
        let c = vk::Semaphore::from_raw(11);
        let t = vk::Semaphore::from_raw(12);
        let upload = super::super::strata_info_transfer::Wait {
            semaphore: vk::Semaphore::from_raw(99),
            value: 77,
        };
        // Same real plan as native submissions: header releases copy; copy
        // completion releases its compute consumer. No fabricated completion.
        let nodes = [
            NodeAccess {
                shared: false,
                copy: true,
                release_after: false,
                accesses: vec![Access {
                    buffer: 1,
                    first: 0,
                    bytes: 4096,
                    write: true,
                }],
            },
            NodeAccess {
                shared: false,
                copy: false,
                release_after: false,
                accesses: vec![Access {
                    buffer: 1,
                    first: 0,
                    bytes: 4096,
                    write: false,
                }],
            },
        ];
        let p = plan(&nodes, false).unwrap();
        assert_eq!(
            incoming_dependency(0, &p.segments[0], [c, t], 100, Some(upload)),
            Some(upload)
        );
        assert_eq!(
            incoming_dependency(0, &p.segments[0], [c, t], 100, None),
            None
        );
        assert_eq!(
            incoming_dependency(1, &p.segments[1], [c, t], 100, Some(upload)),
            Some(super::super::strata_info_transfer::Wait {
                semaphore: c,
                value: 101
            })
        );
        assert_eq!(
            incoming_dependency(2, &p.segments[2], [c, t], 100, Some(upload)),
            Some(super::super::strata_info_transfer::Wait {
                semaphore: t,
                value: 102
            })
        );
    }
}

#[cfg(test)]
mod draft_lane_tests {
    use super::*;
    fn nodes() -> Vec<NodeAccess> {
        vec![
            NodeAccess {
                shared: false,
                copy: true,
                release_after: false,
                accesses: vec![
                    Access {
                        buffer: 11,
                        first: 0,
                        bytes: 4096,
                        write: false,
                    },
                    Access {
                        buffer: 12,
                        first: 0,
                        bytes: 4096,
                        write: true,
                    },
                ],
            },
            NodeAccess {
                shared: false,
                copy: false,
                release_after: true,
                accesses: vec![
                    Access {
                        buffer: 12,
                        first: 0,
                        bytes: 4096,
                        write: false,
                    },
                    Access {
                        buffer: 13,
                        first: 0,
                        bytes: 4096,
                        write: true,
                    },
                ],
            },
            NodeAccess {
                shared: false,
                copy: true,
                release_after: false,
                accesses: vec![
                    Access {
                        buffer: 13,
                        first: 0,
                        bytes: 4096,
                        write: false,
                    },
                    Access {
                        buffer: 14,
                        first: 0,
                        bytes: 4096,
                        write: true,
                    },
                ],
            },
            NodeAccess {
                shared: false,
                copy: true,
                release_after: false,
                accesses: vec![Access {
                    buffer: 12,
                    first: 8192,
                    bytes: 4096,
                    write: true,
                }],
            },
        ]
    }
    #[test]
    fn actual_mtp_lane_uses_compute_queue_for_copies_without_losing_original_access_inventory() {
        let nodes = nodes();
        let build = |lane| {
            let mut p = Planner::for_lane(nodes.len(), false, lane).unwrap();
            for node in &nodes {
                p.push(node).unwrap();
            }
            p.finish().unwrap()
        };
        let main = build(0);
        let draft = build(1);
        assert!(main.segments.iter().any(|s| s.copy));
        assert!(main.last_copy > 0);
        assert!(draft.segments.iter().all(|s| !s.copy && s.wait == 0));
        assert_eq!(draft.last_copy, 0);
        assert_eq!(draft.node_segments.len(), nodes.len());
        assert_eq!(draft.node_segments, [1, 1, 2, 2]); // Original release-after still divides recordings.
        assert_eq!(
            nodes.iter().map(|n| n.copy).collect::<Vec<_>>(),
            [true, false, true, true]
        );
        let inventory = |p: &Plan| {
            p.accesses()
                .iter()
                .map(|a| (a.buffer, a.first, a.bytes, a.write))
                .collect::<Vec<_>>()
        };
        assert_eq!(inventory(&draft), inventory(&main));
        assert_eq!(
            inventory(&draft),
            [
                (11, 0, 4096, false),
                (12, 0, 4096, true),
                (12, 8192, 4096, true),
                (13, 0, 4096, true),
                (14, 0, 4096, true)
            ]
        );
        assert!(
            draft
                .conflicts(&[Access {
                    buffer: 11,
                    first: 0,
                    bytes: 4096,
                    write: true
                }])
                .unwrap()
        );
        assert!(
            !draft
                .conflicts(&[Access {
                    buffer: 12,
                    first: 4096,
                    bytes: 4096,
                    write: true
                }])
                .unwrap()
        );
    }
    #[test]
    fn original_mtp_lane_selection_never_accepts_a_missing_queue_or_mints_copy_timeline_credit() {
        assert!(Planner::for_lane(0, false, 1).is_err());
        assert!(Planner::for_lane(1, false, 2).is_err());
        for profile in [false, true] {
            let nodes = nodes();
            let mut p = Planner::for_lane(nodes.len(), profile, 1).unwrap();
            for node in &nodes {
                p.push(node).unwrap();
            }
            let p = p.finish().unwrap();
            assert_eq!(p.last_copy, 0);
            assert!(p.segments.iter().all(|s| !s.copy && s.wait == 0));
            assert!(p.node_segments.windows(2).all(|s| s[0] <= s[1]));
        }
    }
}

#[cfg(test)]
mod shared_lane_tests {
    use super::*;
    use ash::vk::Handle;
    fn node(queue: usize, accesses: &[(u64, bool)]) -> NodeAccess {
        NodeAccess {
            copy: queue == 1,
            shared: queue == 2,
            release_after: false,
            accesses: accesses
                .iter()
                .map(|&(buffer, write)| Access {
                    buffer,
                    first: 0,
                    bytes: 32,
                    write,
                })
                .collect(),
        }
    }
    #[test]
    fn shared_scope_requires_real_nonce_and_original_target_or_active_draft() {
        use cubecl_runtime::server::NativeStrataBranch as Branch;
        for request in [1, 17, u64::MAX] {
            shared_recording(request, None).unwrap(); // Actual first prompt-final target is uncaptured.
            shared_queue(request, 0, None).unwrap();
            for rows in 1..=8 {
                for role in 0..4 {
                    let target = Branch::target_window_with_state(rows, 32768, role).unwrap();
                    shared_recording(request, Some((request, target))).unwrap();
                    assert!(
                        shared_recording(request, Some((request.wrapping_add(1), target))).is_err()
                    );
                }
            }
            for branch in [
                Branch::mtp_round_for_context(1, 3, 32768).unwrap(),
                Branch::mtp_step_for_context(1, 32768).unwrap(),
            ] {
                shared_recording(request, Some((request, branch))).unwrap();
                shared_queue(request, 1, Some(request)).unwrap();
                assert!(
                    shared_recording(request, Some((request.wrapping_add(1), branch))).is_err()
                );
                assert!(shared_queue(request, 1, Some(request.wrapping_add(1))).is_err());
            }
            assert!(
                shared_recording(
                    request,
                    Some((request, Branch::target_commit(2, 32768, 0).unwrap()))
                )
                .is_err()
            );
            assert!(shared_queue(request, 1, None).is_err());
            assert!(shared_queue(request, 0, Some(request)).is_err());
            assert!(shared_queue(request, 2, None).is_err());
        }
        assert!(shared_recording(0, None).is_err());
        assert!(shared_queue(0, 0, None).is_err());
        assert!(shared_recording(0, Some((0, Branch::target_window(1).unwrap()))).is_err());
    }
    #[test]
    fn draft_side_uses_original_opposite_compute_and_joins_both_fault_and_scaled_row() {
        assert_eq!(shared_peer(0).unwrap(), 1);
        assert_eq!(shared_peer(1).unwrap(), 0);
        assert!(shared_peer(2).is_err());
        assert!(shared_queue(0, 1, Some(0)).is_err());
        let mut routed = node(0, &[(11, false), (21, true), (31, true)]);
        routed.release_after = true;
        let mut side = node(2, &[(11, false), (22, true), (32, true)]);
        side.copy = true; // Actual side fill/copy stays on that compute family.
        let nodes = [
            node(0, &[(11, true), (21, true)]),
            side,
            routed,
            node(0, &[(22, false), (21, true)]),
            node(0, &[(31, false), (32, false), (21, true), (51, true)]),
        ];
        for lane in [0, 1] {
            let mut planner = Planner::for_lane(nodes.len(), false, lane).unwrap();
            for node in &nodes {
                planner.push(node).unwrap();
            }
            let plan = planner.finish().unwrap();
            assert_eq!(plan.last_copy, 0);
            let producer = &plan.segments[plan.node_segments[0]];
            let side = &plan.segments[plan.node_segments[1]];
            let routed = &plan.segments[plan.node_segments[2]];
            let joined = &plan.segments[plan.node_segments[3]];
            assert_eq!(side.queue(), 2);
            assert!(side.waits[0] >= producer.signal);
            assert_eq!(routed.waits[2], 0); // No shared fault alias serializes routed work.
            assert_ne!(plan.node_segments[2], plan.node_segments[3]);
            assert!(joined.waits[2] >= side.signal);
            assert_eq!(plan.last_shared, side.signal);
            assert!(nodes[4].accesses.iter().any(|a| a.buffer == 32 && !a.write));
            // The retained cross-flight inventory conservatively includes the
            // earlier writer; it must not be mislabeled as read-only.
            assert!(plan.accesses().iter().any(|a| a.buffer == 32 && a.write));
        }
    }
    #[test]
    fn late_shared_fork_overlaps_routed_roles_and_joins_real_fault_and_result_before_combine() {
        // 11=GR input, 21=global fault, 22=shared fault, 31=routed row,
        // 32=scaled shared row; genuinely independent original owner IDs.
        let mut roles = node(0, &[(11, false), (21, true), (31, true)]);
        roles.release_after = true;
        let p = plan(
            &[
                node(0, &[(11, true), (21, true)]),
                node(2, &[(11, false), (22, true), (32, true)]),
                roles,
                node(0, &[(22, false), (21, true)]),
                node(0, &[(31, false), (32, false), (21, true), (41, true)]),
            ],
            false,
        )
        .unwrap();
        assert_eq!(p.node_segments, [1, 2, 3, 4, 4]);
        let (pre, shared, routed, combine) = (
            &p.segments[1],
            &p.segments[2],
            &p.segments[3],
            &p.segments[4],
        );
        assert_eq!(shared.waits, [pre.signal, 0, 0]);
        assert_eq!(routed.waits, [0, 0, 0]); // no later combine edge drags these roles behind shared
        assert_eq!(combine.waits, [0, 0, shared.signal]);
        assert_eq!(p.last_shared, shared.signal);
        assert_eq!(p.last_copy, 0);
        let timelines = [
            vk::Semaphore::from_raw(1),
            vk::Semaphore::from_raw(2),
            vk::Semaphore::from_raw(3),
        ];
        assert_eq!(
            incoming_dependencies(4, combine, timelines, 100, None)[2],
            Some(super::super::strata_info_transfer::Wait {
                semaphore: timelines[2],
                value: 100 + shared.signal
            })
        );
    }
    #[test]
    fn three_queue_byte_hazards_match_independent_pairwise_oracle_and_keep_all_peer_waits() {
        let mut random = 19u64;
        let mut next = || {
            random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
            random
        };
        let nodes: Vec<_> = (0..768)
            .map(|_| {
                let q = (next() % 3) as usize;
                NodeAccess {
                    copy: q == 1,
                    shared: q == 2,
                    release_after: next() & 1 != 0,
                    accesses: (0..3)
                        .map(|_| Access {
                            buffer: 1 + next() % 9,
                            first: next() % 128,
                            bytes: 1 + next() % 24,
                            write: next() & 8 != 0,
                        })
                        .collect(),
                }
            })
            .collect();
        let p = plan(&nodes, false).unwrap();
        for (j, n) in nodes.iter().enumerate() {
            let current = &p.segments[p.node_segments[j]];
            assert!(current.waits.iter().all(|&w| w < current.signal));
            assert_eq!(current.waits[current.queue()], 0);
            for (i, old) in nodes[..j].iter().enumerate() {
                let prior = &p.segments[p.node_segments[i]];
                if prior.queue() == current.queue() {
                    continue;
                }
                if old.accesses.iter().any(|a| {
                    n.accesses.iter().any(|b| {
                        a.buffer == b.buffer
                            && a.first < b.first + b.bytes
                            && b.first < a.first + a.bytes
                            && (a.write || b.write)
                    })
                }) {
                    assert!(current.waits[prior.queue()] >= prior.signal);
                }
            }
        }
        // One consumer must wait on BOTH peer timelines, never max one away.
        let p = plan(
            &[
                node(1, &[(7, true)]),
                node(2, &[(8, true)]),
                node(0, &[(7, false), (8, false)]),
            ],
            false,
        )
        .unwrap();
        let join = &p.segments[3];
        assert!(join.waits[1] > 0 && join.waits[2] > 0);
        let sem = [
            vk::Semaphore::from_raw(11),
            vk::Semaphore::from_raw(12),
            vk::Semaphore::from_raw(13),
        ];
        let deps = incoming_dependencies(3, join, sem, 100, None);
        assert!(deps[0].is_none());
        assert_eq!(deps[1].unwrap().semaphore, sem[1]);
        assert_eq!(deps[2].unwrap().semaphore, sem[2]);
    }
    #[test]
    fn shared_fill_stays_on_original_opposite_compute_peer_for_target_and_mtp() {
        let mut fill = node(2, &[(7, true)]);
        fill.copy = true;
        let p = plan(&[fill, node(0, &[(7, false)])], false).unwrap();
        assert!(p.segments[1].shared && !p.segments[1].copy);
        assert_eq!(p.last_copy, 0);
        assert_eq!(p.last_shared, p.segments[1].signal);
        let mut mtp = Planner::for_lane(1, false, 1).unwrap();
        mtp.push(&node(2, &[(7, true)])).unwrap();
        let mtp = mtp.finish().unwrap();
        assert!(mtp.segments[1].shared && !mtp.segments[1].copy);
        assert_eq!(shared_peer(1).unwrap(), 0); // Never draft's own physical queue.
        assert_eq!(mtp.last_copy, 0);
        let p = plan(
            &[
                node(2, &[(7, false)]),
                node(0, &[(8, false)]),
                node(1, &[(9, false)]),
            ],
            true,
        )
        .unwrap();
        for pair in p.segments.windows(2) {
            if pair[0].queue() != pair[1].queue() {
                assert!(pair[1].waits[pair[0].queue()] >= pair[0].signal);
            }
        }
    }
    #[test]
    fn added_queue_metadata_fits_explicit_admission_allowance() {
        // Max reserved interval cells grew by16 bytes; every retained segment
        // gains24 wait bytes plus queue tags/alignment and a command-pool owner.
        let regions = 2 * MAX_REGIONS * 16;
        let segments = 32769 * 32;
        assert!(regions + segments < 16 * 1024 * 1024);
        assert_eq!(std::mem::size_of::<Region>(), 64);
        assert!(std::mem::size_of::<Segment>() <= 48);
    }
    fn range_node(copy: bool, buffer: u64, first: u64, bytes: u64, write: bool) -> NodeAccess {
        NodeAccess {
            copy,
            shared: false,
            release_after: false,
            accesses: vec![Access {
                buffer,
                first,
                bytes,
                write,
            }],
        }
    }
    fn test_allocate_commands(
        p: &Plan,
        h: vk::CommandBuffer,
        budget: usize,
        allocate: impl FnMut(usize, &mut [vk::CommandBuffer]) -> Result<(), &'static str>,
    ) -> Result<Vec<vk::CommandBuffer>, &'static str> {
        let mut commands = Vec::new();
        allocate_commands(p, h, budget, &mut commands, allocate, |_| ())?;
        Ok(commands)
    }
    #[test]
    fn pooled_command_allocation_preserves_original_global_handles_and_preflights_budget() {
        use ash::vk::Handle;
        let mut dq = range_node(false, 1, 0, 64, false);
        dq.release_after = true;
        let mut shared = range_node(false, 3, 0, 64, true);
        shared.shared = true;
        let p = plan(
            &[
                range_node(true, 1, 0, 64, true),
                dq,
                range_node(false, 2, 0, 64, true),
                shared,
                range_node(true, 1, 0, 64, true),
            ],
            false,
        )
        .unwrap();
        let header = vk::CommandBuffer::from_raw(9);
        let mut calls = Vec::new();
        let commands = test_allocate_commands(&p, header, 64 * 1024 * 1024, |queue, out| {
            calls.push((queue, out.len()));
            for (i, h) in out.iter_mut().enumerate() {
                *h = vk::CommandBuffer::from_raw(100 + queue as u64 * 100 + i as u64);
            }
            Ok(())
        })
        .unwrap();
        assert_eq!(commands.len(), p.segments.len());
        assert_eq!(commands[0], header);
        let mut seen = [0; 3];
        for (index, segment) in p.segments.iter().enumerate().skip(1) {
            let q = segment.queue();
            assert_eq!(commands[index].as_raw(), 100 + q as u64 * 100 + seen[q]);
            seen[q] += 1;
        }
        assert_eq!(
            calls.iter().map(|(q, _)| *q).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        let mut attempted = false;
        assert!(
            test_allocate_commands(&p, header, 1, |_, _| {
                attempted = true;
                Ok(())
            })
            .is_err()
        );
        assert!(!attempted);
        for failed in 0..3 {
            let mut attempts = Vec::new();
            assert!(
                test_allocate_commands(&p, header, 64 * 1024 * 1024, |queue, out| {
                    attempts.push(queue);
                    if queue == failed {
                        return Err("injected pool allocation");
                    }
                    out.fill(vk::CommandBuffer::from_raw(100 + queue as u64));
                    Ok(())
                })
                .is_err()
            );
            assert_eq!(attempts, (0..=failed).collect::<Vec<_>>());
        }
        assert!(test_allocate_commands(&p, header, 64 * 1024 * 1024, |_, _| Ok(())).is_err());
    }
    #[test]
    fn main_construction_owner_recovers_partial_pool_errors_and_panics_without_header_free() {
        use ash::vk::Handle;
        use std::{cell::RefCell, rc::Rc};
        let mut shared = range_node(false, 3, 0, 64, true);
        shared.shared = true;
        let p = plan(
            &[
                range_node(false, 1, 0, 64, true),
                range_node(true, 2, 0, 64, true),
                shared,
            ],
            false,
        )
        .unwrap();
        let header = vk::CommandBuffer::from_raw(9);
        for panic in [false, true] {
            for failed in 0..3 {
                let freed = Rc::new(RefCell::new(Vec::new()));
                let observed = freed.clone();
                let mut commands = Vec::new();
                let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    allocate_commands(
                        &p,
                        header,
                        64 * 1024 * 1024,
                        &mut commands,
                        |queue, out| {
                            for (i, h) in out.iter_mut().enumerate() {
                                *h = vk::CommandBuffer::from_raw(
                                    100 + queue as u64 * 100 + i as u64,
                                );
                            }
                            if queue == failed {
                                if panic {
                                    panic!("after actual allocation before adoption");
                                }
                                return Err("after partial allocation");
                            }
                            Ok(())
                        },
                        |h| observed.borrow_mut().push(h.as_raw()),
                    )
                }));
                assert!(attempt.is_err() || attempt.unwrap().is_err());
                assert_eq!(*freed.borrow(), vec![100]); // Each new MAIN once; header/COPY/SHARED never.
            }
        }
        let mut commands = Vec::new();
        let mut freed = Vec::new();
        allocate_commands(
            &p,
            header,
            64 * 1024 * 1024,
            &mut commands,
            |queue, out| {
                out.fill(vk::CommandBuffer::from_raw(100 + queue as u64));
                Ok(())
            },
            |h| freed.push(h),
        )
        .unwrap();
        assert!(freed.is_empty()); // Successful Recording retains the whole original MAIN pool.
    }
}
