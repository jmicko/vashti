//! Dependency-complete native compute/copy submissions. No model content.
use ash::vk;
use std::collections::HashMap;

pub(crate) struct Access {
    pub buffer: u64,
    pub write: bool,
}
pub(crate) struct NodeAccess {
    pub copy: bool,
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
#[derive(Default)]
struct Hazard {
    readers: [u64; 2],
    writers: [u64; 2],
}
/// Full-buffer hazards are conservative for carved views. Independent buffers
/// may overlap; a read/read pair never invents a data dependency. Segment0 is
/// the compute-queue upload/initialization release, before any native work.
pub const NATIVE_QUEUE_HOST_BYTES: usize = 80 * 1024 * 1024;
pub(crate) struct Planner {
    result: Plan,
    hazards: HashMap<u64, Hazard>,
    capacity: usize,
    serialize: bool,
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
            capacity,
            serialize,
        })
    }
    pub fn push(&mut self, node: &NodeAccess) -> Result<(), &'static str> {
        let result = &mut self.result;
        let index = result.node_segments.len();
        if index >= self.capacity {
            return Err("native queue plan node capacity");
        }
        if index == 0 || result.segments.last().unwrap().copy != node.copy {
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
        let s = result.segments.len() - 1;
        result.node_segments.push(s);
        let segment = &mut result.segments[s];
        let own = usize::from(node.copy);
        let other = 1 - own;
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
            let h = self.hazards.entry(access.buffer).or_default();
            segment.wait = segment.wait.max(h.writers[other]);
            if access.write {
                segment.wait = segment.wait.max(h.readers[other]);
                h.writers[own] = segment.signal;
            } else {
                h.readers[own] = segment.signal;
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
    pub fn submit(&mut self, queues: [vk::Queue; 2], fence: vk::Fence) -> Result<(), &'static str> {
        // Reserve before any submission; even a partial failure never reuses a
        // timeline value. The caller retains this entire recording on failure.
        let base = reserve_epoch(&mut self.next_base, self.plan.segments.len())?;
        for (segment, command) in self.plan.segments.iter().zip(&self.commands) {
            let own = usize::from(segment.copy);
            let signals = [self.timelines[own]];
            let signal_values = [base + segment.signal];
            let waits = [self.timelines[1 - own]];
            let wait_values = [base + segment.wait];
            let stages = [vk::PipelineStageFlags::ALL_COMMANDS];
            let mut timeline =
                vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&signal_values);
            let mut submit = vk::SubmitInfo::default()
                .command_buffers(std::slice::from_ref(command))
                .signal_semaphores(&signals);
            if segment.wait != 0 {
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
            accesses: vec![Access { buffer, write }],
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
