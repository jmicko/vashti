//! Dependency-complete native compute/copy submissions. No model content.
use ash::vk;
use std::collections::HashMap;

pub(crate) struct Access {
    pub buffer: u64,
    pub first: u64,
    pub bytes: u64,
    pub write: bool,
}
pub(crate) struct NodeAccess {
    pub copy: bool,
    pub release_after: bool,
    pub accesses: Vec<Access>,
}
#[derive(Debug)]
pub(crate) struct Segment {
    pub copy: bool,
    pub wait: u64,
    pub signal: u64,
}
pub(crate) struct Plan {
    pub segments: Vec<Segment>,
    pub node_segments: Vec<usize>,
    pub last_copy: u64,
}
#[derive(Clone, Copy, Default, PartialEq, Eq)]
struct Hazard {
    readers: [u64; 2],
    writers: [u64; 2],
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
pub const NATIVE_QUEUE_HOST_BYTES: usize =
    128 * 1024 * 1024 + super::strata_info_transfer::HOST_BYTES;

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
) -> Result<u64, &'static str> {
    let end = access
        .first
        .checked_add(access.bytes)
        .filter(|_| access.bytes != 0)
        .ok_or("native queue access interval")?;
    split_region(regions, access.first, total)?;
    split_region(regions, end, total)?;
    let mut i = regions.partition_point(|r| r.end <= access.first);
    let mut cursor = access.first;
    let mut wait = 0;
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
        wait = wait.max(h.writers[1 - own]);
        if access.write {
            wait = wait.max(h.readers[1 - own]);
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
    regions: usize,
    region_capacity: usize,
    capacity: usize,
    serialize: bool,
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
            wait: 0,
            signal: 1,
        });
        Ok(Self {
            result: Plan {
                segments,
                node_segments: Vec::with_capacity(capacity),
                last_copy: 0,
            },
            hazards: HashMap::new(),
            regions: 0,
            region_capacity: 0,
            capacity,
            serialize,
            release_next: false,
        })
    }
    pub fn push(&mut self, node: &NodeAccess) -> Result<(), &'static str> {
        let result = &mut self.result;
        let index = result.node_segments.len();
        if index >= self.capacity {
            return Err("native queue plan node capacity");
        }
        if index == 0 || self.release_next || result.segments.last().unwrap().copy != node.copy {
            let signal = result.segments.len() as u64 + 1;
            result.segments.push(Segment {
                copy: node.copy,
                wait: u64::from(node.copy),
                signal,
            });
            if node.copy {
                result.last_copy = signal;
            }
        }
        self.release_next = node.release_after;
        let s = result.segments.len() - 1;
        result.node_segments.push(s);
        let segment = &mut result.segments[s];
        let own = usize::from(node.copy);
        for access in &node.accesses {
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
            segment.wait = segment.wait.max(update_regions(
                regions,
                access,
                own,
                segment.signal,
                &mut self.regions,
            )?);
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
    pub fn finish(mut self) -> Result<Plan, &'static str> {
        if self.result.node_segments.len() != self.capacity {
            return Err("native queue plan incomplete");
        }
        if self.serialize {
            for s in 1..self.result.segments.len() {
                if self.result.segments[s].copy != self.result.segments[s - 1].copy {
                    self.result.segments[s].wait = self.result.segments[s]
                        .wait
                        .max(self.result.segments[s - 1].signal);
                }
            }
        }
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
fn incoming_dependency(
    index: usize,
    segment: &Segment,
    timelines: [vk::Semaphore; 2],
    base: u64,
    uploads: Option<super::strata_info_transfer::Wait>,
) -> Option<super::strata_info_transfer::Wait> {
    if index == 0 {
        uploads
    } else {
        (segment.wait != 0).then_some(super::strata_info_transfer::Wait {
            semaphore: timelines[1 - usize::from(segment.copy)],
            value: base + segment.wait,
        })
    }
}

/// Recording owns all copy pools and timeline semaphores through the joined
/// fence. Main command buffers belong to its existing compute pool.
pub(crate) struct Commands {
    device: ash::Device,
    copy_pool: vk::CommandPool,
    timelines: [vk::Semaphore; 2],
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
        plan: Plan,
    ) -> Result<Self, &'static str> {
        let mut result = Self {
            device: device.clone(),
            copy_pool: vk::CommandPool::null(),
            timelines: [vk::Semaphore::null(); 2],
            next_base: 0,
            commands: Vec::with_capacity(plan.segments.len()),
            plan,
        };
        result.commands.push(header);
        result.copy_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                    .queue_family_index(copy_family),
                None,
            )
        }
        .map_err(|_| "native copy command pool")?;
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
        for segment in &result.plan.segments[1..] {
            let command = unsafe {
                device.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(if segment.copy {
                            result.copy_pool
                        } else {
                            main_pool
                        })
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
            }
            .map_err(|_| "native queue command allocation")?[0];
            result.commands.push(command);
        }
        Ok(result)
    }
    pub fn metadata_bytes(&self) -> usize {
        self.plan.segments.capacity() * std::mem::size_of::<Segment>()
            + self.plan.node_segments.capacity() * std::mem::size_of::<usize>()
            + self.commands.capacity() * std::mem::size_of::<vk::CommandBuffer>()
            + std::mem::size_of::<Self>()
    }
    pub fn submit(
        &mut self,
        queues: [vk::Queue; 2],
        fence: vk::Fence,
        uploads: Option<super::strata_info_transfer::Wait>,
    ) -> Result<(), &'static str> {
        // Reserve before any submission; even a partial failure never reuses a
        // timeline value. The caller retains this entire recording on failure.
        let base = reserve_epoch(&mut self.next_base, self.plan.segments.len())?;
        if self
            .plan
            .segments
            .first()
            .is_none_or(|s| s.copy || s.wait != 0)
        {
            return Err("native upload dependency requires original compute header");
        }
        for (index, (segment, command)) in self.plan.segments.iter().zip(&self.commands).enumerate()
        {
            let own = usize::from(segment.copy);
            let signals = [self.timelines[own]];
            let signal_values = [base + segment.signal];
            // Actual INFO transfer completion precedes the original header;
            // its timeline release carries that dependency to every segment.
            let dependency = incoming_dependency(index, segment, self.timelines, base, uploads);
            let waits = [dependency.map_or(vk::Semaphore::null(), |d| d.semaphore)];
            let wait_values = [dependency.map_or(0, |d| d.value)];
            let stages = [vk::PipelineStageFlags::ALL_COMMANDS];
            let mut timeline =
                vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&signal_values);
            let mut submit = vk::SubmitInfo::default()
                .command_buffers(std::slice::from_ref(command))
                .signal_semaphores(&signals);
            if dependency.is_some() {
                timeline = timeline.wait_semaphore_values(&wait_values);
                submit = submit.wait_semaphores(&waits).wait_dst_stage_mask(&stages);
            }
            let submit = submit.push_next(&mut timeline);
            unsafe {
                self.device
                    .queue_submit(queues[own], &[submit], vk::Fence::null())
            }
            .map_err(|_| "native compute/copy submit")?;
        }
        // Queue order joins all compute commands. Waiting for the last copy
        // timeline also joins every earlier copy before the single final fence.
        let waits = [self.timelines[1]];
        let values = [base + self.plan.last_copy];
        let stages = [vk::PipelineStageFlags::ALL_COMMANDS];
        let mut timeline = vk::TimelineSemaphoreSubmitInfo::default();
        let mut join = vk::SubmitInfo::default();
        if self.plan.last_copy != 0 {
            timeline = timeline.wait_semaphore_values(&values);
            join = join.wait_semaphores(&waits).wait_dst_stage_mask(&stages);
        }
        let join = join.push_next(&mut timeline);
        unsafe { self.device.queue_submit(queues[0], &[join], fence) }
            .map_err(|_| "native dual queue join")
    }
}
impl Drop for Commands {
    fn drop(&mut self) {
        unsafe {
            for semaphore in self.timelines {
                if semaphore != vk::Semaphore::null() {
                    self.device.destroy_semaphore(semaphore, None);
                }
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
    fn node(copy: bool, buffer: u64, write: bool) -> NodeAccess {
        NodeAccess {
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
