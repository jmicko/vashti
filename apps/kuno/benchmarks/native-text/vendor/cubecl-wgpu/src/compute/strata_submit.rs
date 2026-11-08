//! Bounded timeline-only submission batches. No command or dependency is removed.
use super::super::strata_info_transfer::Wait;
use super::{Plan, incoming_dependencies};
use ash::vk;

/// Temporary original records, queue indices, timeline structs and submit views.
/// Added to the native host reservation, independently of retained graph metadata.
pub(super) const HOST_BYTES: usize = 16 * 1024 * 1024;
const MAX_SEGMENTS: usize = 32769;

pub(super) struct Record {
    queue: usize,
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
    // Stable heap-backed C views, completed once before any rolling prefix.
    // Their raw pointers refer only to these immutable original records.
    timelines: Vec<vk::TimelineSemaphoreSubmitInfo<'static>>,
    views: Vec<vk::SubmitInfo<'static>>,
}

impl Batches {
    #[cfg(test)]
    pub(super) fn new(
        plan: &Plan,
        commands: &[vk::CommandBuffer],
        timelines: [vk::Semaphore; 3],
        base: u64,
        uploads: Option<Wait>,
    ) -> Result<Self, &'static str> {
        Self::new_with_budget(plan, commands, timelines, base, uploads, HOST_BYTES)
    }
    pub(super) fn new_with_budget(
        plan: &Plan,
        commands: &[vk::CommandBuffer],
        timelines: [vk::Semaphore; 3],
        base: u64,
        uploads: Option<Wait>,
        metadata_budget: usize,
    ) -> Result<Self, &'static str> {
        let metadata_budget = metadata_budget.min(HOST_BYTES);
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
        // Preflight requested peak before allocations, then charge actual
        // capacities. The temporary queue-indexed SubmitInfo array used only by
        // full replay is included; rolling runs allocate nothing after entry.
        let requested = count
            .checked_mul(
                size_of::<Record>()
                    + size_of::<usize>()
                    + size_of::<vk::TimelineSemaphoreSubmitInfo<'_>>()
                    + 2 * size_of::<vk::SubmitInfo<'_>>(),
            )
            .and_then(|n| n.checked_add(size_of::<Self>()))
            .ok_or("native submission host capacity overflow")?;
        if requested > metadata_budget {
            return Err("native submission host capacity");
        }
        let mut result = Self {
            records: Vec::new(),
            queues: Default::default(),
            timelines: Vec::new(),
            views: Vec::new(),
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
        result
            .timelines
            .try_reserve_exact(count)
            .map_err(|_| "native timeline records allocation")?;
        result
            .views
            .try_reserve_exact(count)
            .map_err(|_| "native submit records allocation")?;
        if result
            .metadata_bytes()
            .checked_add(count * size_of::<vk::SubmitInfo<'_>>())
            .is_none_or(|n| n > metadata_budget)
        {
            return Err("native submission actual host capacity");
        }
        for (index, (segment, command)) in plan.segments.iter().zip(commands).enumerate() {
            let own = segment.queue();
            let mut record = Record {
                queue: own,
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
        // All three backing arrays now have their final capacity. Moving the
        // Batches value moves Vec headers only; none of their pointees move.
        for r in &result.records {
            result.timelines.push(vk::TimelineSemaphoreSubmitInfo {
                signal_semaphore_value_count: 1,
                p_signal_semaphore_values: r.signal_value.as_ptr(),
                wait_semaphore_value_count: r.wait_count as u32,
                p_wait_semaphore_values: r.wait_values.as_ptr(),
                ..Default::default()
            });
        }
        for (r, timeline) in result.records.iter().zip(&result.timelines) {
            result.views.push(vk::SubmitInfo {
                p_next: (timeline as *const vk::TimelineSemaphoreSubmitInfo<'_>).cast(),
                command_buffer_count: 1,
                p_command_buffers: &r.command,
                signal_semaphore_count: 1,
                p_signal_semaphores: r.signal.as_ptr(),
                wait_semaphore_count: r.wait_count as u32,
                p_wait_semaphores: r.waits.as_ptr(),
                p_wait_dst_stage_mask: r.stages.as_ptr(),
                ..Default::default()
            });
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
        let mut views = Vec::new();
        views
            .try_reserve_exact(indices.len())
            .map_err(|_| "native submit views allocation")?;
        if self
            .metadata_bytes()
            .checked_add(views.capacity() * size_of::<vk::SubmitInfo<'_>>())
            .is_none_or(|n| n > HOST_BYTES)
        {
            return Err("native submission live host capacity");
        }
        for &index in indices {
            views.push(self.views[index]);
        }
        Ok(submit(&views))
    }

    /// A contiguous same-queue run of CLOSED original segments. Every original
    /// SubmitInfo, command, signal and wait remains separate and in global order;
    /// only the synchronous host Vulkan entry is shared. No allocation occurs.
    pub(super) fn with_run<R>(
        &self,
        start: usize,
        end: usize,
        submit: impl FnOnce(&[vk::SubmitInfo<'_>]) -> R,
    ) -> Result<R, &'static str> {
        let records = self
            .records
            .get(start..end)
            .filter(|r| !r.is_empty())
            .ok_or("native rolling submit run extent")?;
        if records.iter().any(|r| r.queue != records[0].queue) {
            return Err("native rolling submit run queue");
        }
        Ok(submit(&self.views[start..end]))
    }
    #[cfg(test)]
    pub(super) fn with_record<R>(
        &self,
        index: usize,
        submit: impl FnOnce(&[vk::SubmitInfo<'_>]) -> R,
    ) -> Result<R, &'static str> {
        self.with_run(
            index,
            index.checked_add(1).ok_or("native rolling record index")?,
            submit,
        )
    }
    pub(super) fn metadata_bytes(&self) -> usize {
        self.records.capacity() * size_of::<Record>()
            + self
                .queues
                .iter()
                .map(|v| v.capacity() * size_of::<usize>())
                .sum::<usize>()
            + self.timelines.capacity() * size_of::<vk::TimelineSemaphoreSubmitInfo<'_>>()
            + self.views.capacity() * size_of::<vk::SubmitInfo<'_>>()
            + size_of::<Self>()
    }
    pub(super) fn record_queue(&self, index: usize) -> Result<usize, &'static str> {
        self.records
            .get(index)
            .map(|record| record.queue)
            .ok_or("native rolling record queue")
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
    fn rolling_original_segments_keep_exact_dependencies_and_only_backward_signals() {
        let nodes: Vec<_> = (0..4096)
            .map(|i| {
                node(
                    i % 3,
                    (i % 11) as u64 + 1,
                    ((i / 11) % 5 * 64) as u64,
                    i % 4 != 0,
                )
            })
            .collect();
        for lane in 0..2 {
            for serialize in [false, true] {
                let plan = original(&nodes, lane, serialize);
                let commands = handles(&plan);
                let timelines = semaphores();
                let upload = Some(Wait {
                    semaphore: vk::Semaphore::from_raw(99),
                    value: 77,
                });
                let batches = Batches::new(&plan, &commands, timelines, 123, upload).unwrap();
                for (index, segment) in plan.segments.iter().enumerate() {
                    assert_eq!(batches.record_queue(index).unwrap(), segment.queue());
                    batches
                        .with_record(index, |views| {
                            assert_eq!(views.len(), 1);
                            let view = &views[0];
                            assert_eq!(view.command_buffer_count, 1);
                            assert_eq!(view.signal_semaphore_count, 1);
                            let expected: Vec<_> =
                                incoming_dependencies(index, segment, timelines, 123, upload)
                                    .into_iter()
                                    .flatten()
                                    .collect();
                            // SAFETY: actual with_record owns all these arrays/pNext
                            // through this synchronous callback, just as at submit.
                            unsafe {
                                assert_eq!(*view.p_command_buffers, commands[index]);
                                assert_eq!(*view.p_signal_semaphores, timelines[segment.queue()]);
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
                                    123 + segment.signal
                                );
                                assert_eq!(view.wait_semaphore_count as usize, expected.len());
                                assert_eq!(
                                    timeline.wait_semaphore_value_count as usize,
                                    expected.len()
                                );
                                for (offset, dependency) in expected.iter().enumerate() {
                                    assert_eq!(
                                        *view.p_wait_semaphores.add(offset),
                                        dependency.semaphore
                                    );
                                    assert_eq!(
                                        *timeline.p_wait_semaphore_values.add(offset),
                                        dependency.value
                                    );
                                    assert_eq!(
                                        *view.p_wait_dst_stage_mask.add(offset),
                                        vk::PipelineStageFlags::ALL_COMMANDS
                                    );
                                    if dependency.semaphore != vk::Semaphore::from_raw(99) {
                                        let original = dependency.value - 123;
                                        assert!(original < segment.signal);
                                        assert_eq!(
                                            timelines[plan.segments[original as usize - 1].queue()],
                                            dependency.semaphore
                                        );
                                    }
                                }
                            }
                        })
                        .unwrap();
                }
                assert!(batches.record_queue(commands.len()).is_err());
                assert!(batches.with_record(commands.len(), |_| ()).is_err());
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
    #[test]
    fn contiguous_same_queue_runs_preserve_every_original_signal_and_moved_owner_pointer() {
        let nodes = [
            node(0, 1, 0, true),
            node(0, 2, 0, true),
            node(1, 1, 0, false),
            node(0, 2, 0, false),
            node(0, 3, 0, true),
        ];
        let plan = original(&nodes, 0, false);
        let commands = handles(&plan);
        let batches = Batches::new(&plan, &commands, semaphores(), 123, None).unwrap();
        // Vec-backed C pointees survive an ordinary move; no pointer names a
        // temporary vk struct or a stack-owned original command handle.
        let moved = Box::new(batches);
        for start in 0..plan.segments.len() {
            let own = plan.segments[start].queue();
            let mut end = start + 1;
            while end < plan.segments.len() && plan.segments[end].queue() == own {
                end += 1;
            }
            moved
                .with_run(start, end, |views| {
                    assert_eq!(views.len(), end - start);
                    for (index, view) in (start..end).zip(views) {
                        unsafe {
                            assert_eq!(*view.p_command_buffers, commands[index]);
                            assert_eq!(*view.p_signal_semaphores, semaphores()[own]);
                            let timeline =
                                &*view.p_next.cast::<vk::TimelineSemaphoreSubmitInfo<'_>>();
                            assert_eq!(*timeline.p_signal_semaphore_values, 124 + index as u64);
                            let expected = incoming_dependencies(
                                index,
                                &plan.segments[index],
                                semaphores(),
                                123,
                                None,
                            );
                            let expected: Vec<_> = expected.into_iter().flatten().collect();
                            assert_eq!(
                                timeline.wait_semaphore_value_count as usize,
                                expected.len()
                            );
                            for (n, wait) in expected.iter().enumerate() {
                                assert_eq!(*view.p_wait_semaphores.add(n), wait.semaphore);
                                assert_eq!(*timeline.p_wait_semaphore_values.add(n), wait.value);
                            }
                        }
                    }
                })
                .unwrap();
        }
        for (start, end) in [(0, 0), (2, 1), (0, plan.segments.len() + 1)] {
            assert!(moved.with_run(start, end, |_| ()).is_err());
        }
        assert!(moved.with_run(1, 4, |_| ()).is_err()); // Cross-queue packing forbidden.
        assert!(Batches::new_with_budget(&plan, &commands, semaphores(), 123, None, 1).is_err());
        assert!(moved.metadata_bytes() < HOST_BYTES);
    }
}
