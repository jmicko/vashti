//! Bounded timeline-only submission batches. No command or dependency is removed.
use super::super::strata_info_transfer::Wait;
use super::{Plan, incoming_dependencies};
use ash::vk;

/// Temporary original records, queue indices, timeline structs and submit views.
/// Added to the native host reservation, independently of retained graph metadata.
pub(super) const HOST_BYTES: usize = 16 * 1024 * 1024;
const MAX_SEGMENTS: usize = 32769;

pub(super) struct Record {
    command: vk::CommandBuffer,
    signal: [vk::Semaphore; 1],
    signal_value: [u64; 1],
    waits: [vk::Semaphore; 3],
    wait_values: [u64; 3],
    stages: [vk::PipelineStageFlags; 3],
    wait_count: usize,
}
pub(super) struct Batches {
    records: Vec<Record>,
    queues: [Vec<usize>; 3],
}

impl Batches {
    pub(super) fn new(
        plan: &Plan,
        commands: &[vk::CommandBuffer],
        timelines: [vk::Semaphore; 3],
        base: u64,
        uploads: Option<Wait>,
    ) -> Result<Self, &'static str> {
        let count = plan.segments.len();
        if !(1..=MAX_SEGMENTS).contains(&count)
            || commands.len() != count
            || commands.contains(&vk::CommandBuffer::null())
            || timelines
                .iter()
                .enumerate()
                .any(|(i, s)| *s == vk::Semaphore::null() || timelines[..i].contains(s))
            || base.checked_add(count as u64).is_none()
        {
            return Err("native batched submission inventory");
        }
        let mut counts = [0usize; 3];
        let mut last = [0u64; 3];
        for (index, segment) in plan.segments.iter().enumerate() {
            let own = segment.queue();
            if segment.copy && segment.shared
                || segment.signal != index as u64 + 1
                || segment.wait != *segment.waits.iter().max().unwrap()
                || segment.waits[own] != 0
                || index == 0 && (own != 0 || segment.wait != 0)
            {
                return Err("native batched submission segment");
            }
            for (peer, value) in segment.waits.into_iter().enumerate() {
                if value != 0
                    && (value >= segment.signal
                        || plan.segments[value as usize - 1].queue() != peer)
                {
                    return Err("native batched submission dependency");
                }
            }
            counts[own] += 1;
            last[own] = segment.signal;
        }
        if last[1] != plan.last_copy || last[2] != plan.last_shared {
            return Err("native batched submission final dependency");
        }
        let mut result = Self {
            records: Vec::new(),
            queues: Default::default(),
        };
        result
            .records
            .try_reserve_exact(count)
            .map_err(|_| "native submission records allocation")?;
        for (indices, count) in result.queues.iter_mut().zip(counts) {
            indices
                .try_reserve_exact(count)
                .map_err(|_| "native submission indices allocation")?;
        }
        // Charge even excess allocator capacity before populating the records.
        let view_bytes = count
            .checked_mul(
                size_of::<vk::TimelineSemaphoreSubmitInfo<'_>>() + size_of::<vk::SubmitInfo<'_>>(),
            )
            .ok_or("native submission view capacity overflow")?;
        let bytes = result.records.capacity() * size_of::<Record>()
            + result
                .queues
                .iter()
                .map(|v| v.capacity() * size_of::<usize>())
                .sum::<usize>()
            + view_bytes
            + size_of::<Self>();
        if bytes > HOST_BYTES {
            return Err("native submission host capacity");
        }
        for (index, (segment, command)) in plan.segments.iter().zip(commands).enumerate() {
            let own = segment.queue();
            let mut record = Record {
                command: *command,
                signal: [timelines[own]],
                signal_value: [base + segment.signal],
                waits: [vk::Semaphore::null(); 3],
                wait_values: [0; 3],
                stages: [vk::PipelineStageFlags::ALL_COMMANDS; 3],
                wait_count: 0,
            };
            for dependency in incoming_dependencies(index, segment, timelines, base, uploads)
                .into_iter()
                .flatten()
            {
                record.waits[record.wait_count] = dependency.semaphore;
                record.wait_values[record.wait_count] = dependency.value;
                record.wait_count += 1;
            }
            result.records.push(record);
            result.queues[own].push(index);
        }
        Ok(result)
    }

    /// The callback borrows all records, timeline extension structs and submit
    /// views. Their addresses stay stable until the synchronous Vulkan host call
    /// returns; Vulkan owns the submitted commands afterward, not these views.
    pub(super) fn with_queue<R>(
        &self,
        queue: usize,
        submit: impl FnOnce(&[vk::SubmitInfo<'_>]) -> R,
    ) -> Result<R, &'static str> {
        let indices = self
            .queues
            .get(queue)
            .ok_or("native submission queue index")?;
        let mut timelines = Vec::new();
        let mut views = Vec::new();
        timelines
            .try_reserve_exact(indices.len())
            .map_err(|_| "native timeline views allocation")?;
        views
            .try_reserve_exact(indices.len())
            .map_err(|_| "native submit views allocation")?;
        // All records plus these two arrays must remain in the precharged cap.
        let bytes = self.records.capacity() * size_of::<Record>()
            + self
                .queues
                .iter()
                .map(|v| v.capacity() * size_of::<usize>())
                .sum::<usize>()
            + timelines.capacity() * size_of::<vk::TimelineSemaphoreSubmitInfo<'_>>()
            + views.capacity() * size_of::<vk::SubmitInfo<'_>>()
            + size_of::<Self>();
        if bytes > HOST_BYTES {
            return Err("native submission live host capacity");
        }
        for &index in indices {
            let r = &self.records[index];
            timelines.push(
                vk::TimelineSemaphoreSubmitInfo::default()
                    .signal_semaphore_values(&r.signal_value)
                    .wait_semaphore_values(&r.wait_values[..r.wait_count]),
            );
        }
        for (&index, timeline) in indices.iter().zip(timelines.iter_mut()) {
            let r = &self.records[index];
            views.push(
                vk::SubmitInfo::default()
                    .command_buffers(std::slice::from_ref(&r.command))
                    .signal_semaphores(&r.signal)
                    .wait_semaphores(&r.waits[..r.wait_count])
                    .wait_dst_stage_mask(&r.stages[..r.wait_count])
                    .push_next(timeline),
            );
        }
        Ok(submit(&views))
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Access, NodeAccess, Planner};
    use super::*;
    use ash::vk::Handle;

    fn original(nodes: &[NodeAccess], lane: usize, serialize: bool) -> Plan {
        let mut planner = Planner::for_lane(nodes.len(), serialize, lane).unwrap();
        for node in nodes {
            planner.push(node).unwrap();
        }
        planner.finish().unwrap()
    }
    fn node(queue: usize, root: u64, first: u64, write: bool) -> NodeAccess {
        NodeAccess {
            copy: queue == 1,
            shared: queue == 2,
            release_after: true,
            accesses: vec![Access {
                buffer: root,
                first,
                bytes: 64,
                write,
            }],
        }
    }
    fn handles(plan: &Plan) -> Vec<vk::CommandBuffer> {
        (0..plan.segments.len())
            .map(|i| vk::CommandBuffer::from_raw(1000 + i as u64))
            .collect()
    }
    fn semaphores() -> [vk::Semaphore; 3] {
        [11, 12, 13].map(vk::Semaphore::from_raw)
    }

    #[test]
    fn original_three_queue_dag_and_borrowed_vulkan_views_are_preserved_exactly() {
        // The first host batch waits for work whose other-queue submission is
        // later. Timeline waits permit this; there is no same-queue forward edge.
        let nodes = [
            node(0, 1, 0, true),
            node(1, 2, 0, true),
            node(0, 2, 0, false),
            node(2, 3, 0, true),
            node(0, 3, 0, false),
            node(1, 2, 0, true),
        ];
        for lane in 0..2 {
            for serialize in [false, true] {
                let plan = original(&nodes, lane, serialize);
                let commands = handles(&plan);
                let semaphores = semaphores();
                let upload = Some(Wait {
                    semaphore: vk::Semaphore::from_raw(99),
                    value: 77,
                });
                let batches = Batches::new(&plan, &commands, semaphores, 123, upload).unwrap();
                let mut visited = vec![false; commands.len()];
                for queue in 0..3 {
                    let expected: Vec<_> = plan
                        .segments
                        .iter()
                        .enumerate()
                        .filter(|(_, s)| s.queue() == queue)
                        .map(|(i, _)| i)
                        .collect();
                    batches
                        .with_queue(queue, |views| {
                            assert_eq!(views.len(), expected.len());
                            for (&index, view) in expected.iter().zip(views) {
                                assert!(!visited[index]);
                                visited[index] = true;
                                assert_eq!(view.command_buffer_count, 1);
                                assert_eq!(view.signal_semaphore_count, 1);
                                // SAFETY: the actual helper's callback owns these views
                                // and their complete backing arrays for this whole scope.
                                unsafe {
                                    assert_eq!(*view.p_command_buffers, commands[index]);
                                    assert_eq!(*view.p_signal_semaphores, semaphores[queue]);
                                    let timeline =
                                        &*view.p_next.cast::<vk::TimelineSemaphoreSubmitInfo<'_>>();
                                    assert_eq!(
                                        timeline.s_type,
                                        vk::StructureType::TIMELINE_SEMAPHORE_SUBMIT_INFO
                                    );
                                    assert!(timeline.p_next.is_null());
                                    assert_eq!(timeline.signal_semaphore_value_count, 1);
                                    assert_eq!(
                                        *timeline.p_signal_semaphore_values,
                                        123 + plan.segments[index].signal
                                    );
                                    let dependencies: Vec<_> = incoming_dependencies(
                                        index,
                                        &plan.segments[index],
                                        semaphores,
                                        123,
                                        upload,
                                    )
                                    .into_iter()
                                    .flatten()
                                    .collect();
                                    assert_eq!(
                                        view.wait_semaphore_count as usize,
                                        dependencies.len()
                                    );
                                    assert_eq!(
                                        timeline.wait_semaphore_value_count as usize,
                                        dependencies.len()
                                    );
                                    for (i, dep) in dependencies.iter().enumerate() {
                                        assert_eq!(*view.p_wait_semaphores.add(i), dep.semaphore);
                                        assert_eq!(
                                            *timeline.p_wait_semaphore_values.add(i),
                                            dep.value
                                        );
                                        assert_eq!(
                                            *view.p_wait_dst_stage_mask.add(i),
                                            vk::PipelineStageFlags::ALL_COMMANDS
                                        );
                                    }
                                }
                            }
                        })
                        .unwrap();
                }
                assert!(visited.into_iter().all(|value| value));
            }
        }
    }

    #[test]
    fn every_original_queue_can_progress_without_a_host_dependency_cycle() {
        let mut random = 481u64;
        let nodes: Vec<_> = (0..4096)
            .map(|_| {
                random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
                node(
                    (random % 3) as usize,
                    1 + (random >> 5) % 19,
                    ((random >> 17) % 8) * 64,
                    random & 32 != 0,
                )
            })
            .collect();
        for lane in 0..2 {
            for serialize in [false, true] {
                let plan = original(&nodes, lane, serialize);
                let batches = Batches::new(&plan, &handles(&plan), semaphores(), 0, None).unwrap();
                let mut cursor = [0usize; 3];
                let mut completed = [0u64; 3];
                let mut done = 0;
                while done < plan.segments.len() {
                    let before = done;
                    for queue in 0..3 {
                        if let Some(&index) = batches.queues[queue].get(cursor[queue]) {
                            let segment = &plan.segments[index];
                            if segment
                                .waits
                                .iter()
                                .zip(completed)
                                .all(|(&wait, actual)| actual >= wait)
                            {
                                completed[queue] = segment.signal;
                                cursor[queue] += 1;
                                done += 1;
                            }
                        }
                    }
                    assert!(done > before, "all queue heads waiting is a deadlock");
                }
                assert!(completed[1] >= plan.last_copy && completed[2] >= plan.last_shared);
            }
        }
    }

    #[test]
    fn malformed_inventory_and_unknown_or_forward_edges_fail_before_submission() {
        let nodes = [node(1, 1, 0, true), node(0, 1, 0, false)];
        let mut plan = original(&nodes, 0, false);
        let mut commands = handles(&plan);
        let reject =
            |p: &Plan, c: &[vk::CommandBuffer], s, base| Batches::new(p, c, s, base, None).is_err();
        assert!(reject(
            &plan,
            &commands[..commands.len() - 1],
            semaphores(),
            0
        ));
        commands[0] = vk::CommandBuffer::null();
        assert!(reject(&plan, &commands, semaphores(), 0));
        commands = handles(&plan);
        assert!(reject(&plan, &commands, [semaphores()[0]; 3], 0));
        assert!(reject(&plan, &commands, [vk::Semaphore::null(); 3], 0));
        assert!(reject(&plan, &commands, semaphores(), u64::MAX));
        let last = plan.last_copy;
        plan.last_copy = 0;
        assert!(reject(&plan, &commands, semaphores(), 0));
        plan.last_copy = last;
        plan.segments[2].waits[1] = plan.segments[2].signal;
        plan.segments[2].wait = plan.segments[2].signal;
        assert!(reject(&plan, &commands, semaphores(), 0));
        plan = original(&nodes, 0, false);
        plan.segments[2].waits = [1, 0, 0];
        plan.segments[2].wait = 1; // own lane
        assert!(reject(&plan, &commands, semaphores(), 0));
        plan = original(&nodes, 0, false);
        plan.segments[2].waits = [0, 1, 0];
        plan.segments[2].wait = 1; // header is lane0
        assert!(reject(&plan, &commands, semaphores(), 0));
        plan = original(&nodes, 0, false);
        plan.segments[0].copy = true;
        assert!(reject(&plan, &commands, semaphores(), 0));
    }

    #[test]
    fn maximum_node_plan_stays_bounded_and_empty_queues_do_not_mint_a_submission() {
        let nodes: Vec<_> = (0..32768)
            .map(|i| node(0, 1, (i % 8) * 64, false))
            .collect();
        let plan = original(&nodes, 0, false);
        assert_eq!(plan.segments.len(), MAX_SEGMENTS);
        let batches = Batches::new(&plan, &handles(&plan), semaphores(), 0, None).unwrap();
        batches
            .with_queue(0, |views| assert_eq!(views.len(), MAX_SEGMENTS))
            .unwrap();
        for queue in [1, 2] {
            batches
                .with_queue(queue, |views| assert!(views.is_empty()))
                .unwrap();
        }
        assert!(batches.with_queue(3, |_| ()).is_err());
    }
}
