//! Bounded fresh-descriptor windows; no descriptor/authority cache or completion.
//! Vulkan pool ownership remains in the original fenced Recording.
use ash::vk;
use std::{mem::size_of, ops::Range};

pub(super) const NODE_WINDOW: usize = 128;
const BINDINGS_PER_NODE: usize = 32;
const TEMPORARY_CAP: usize = 512 * 1024;
pub(super) const METADATA_CAP: usize = 64 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
struct Node {
    set: Option<usize>,
    bindings: Range<usize>,
}
#[derive(Debug)]
pub(super) struct Plan {
    nodes: Vec<Node>,
    sets: usize,
    bindings: usize,
    peak_bytes: usize,
}
impl Plan {
    /// Exact production node order. None means original Copy/Fill or a
    /// separately validated command-owned push layout. Fresh compute nodes
    /// still require their complete original1..=32 bindings.
    pub(super) fn new(
        counts: impl IntoIterator<Item = Option<usize>>,
        retained_bytes: usize,
    ) -> Result<Self, &'static str> {
        let mut nodes = Vec::with_capacity(NODE_WINDOW);
        let mut sets = 0usize;
        let mut bindings = 0usize;
        for count in counts {
            if nodes.len() == NODE_WINDOW {
                return Err("native descriptor original-node window capacity");
            }
            let first = bindings;
            let set = if let Some(count) = count {
                if !(1..=BINDINGS_PER_NODE).contains(&count) {
                    return Err("native descriptor original binding capacity");
                }
                bindings = bindings
                    .checked_add(count)
                    .ok_or("native descriptor count overflow")?;
                let set = sets;
                sets += 1;
                Some(set)
            } else {
                None
            };
            nodes.push(Node {
                set,
                bindings: first..bindings,
            });
        }
        if nodes.is_empty() {
            return Err("native descriptor empty original-node window");
        }
        // Actual Vulkan structure sizes, not caller estimates. Includes the
        // retained node/set vectors and all temporary layouts/infos/writes;
        // all buffer owners already belong to the original Recording.
        let peak_bytes = 4096usize // Conservative reserve for headers/stack state.
            .checked_add(
                nodes
                    .capacity()
                    .checked_mul(size_of::<Node>())
                    .ok_or("native descriptor metadata overflow")?,
            )
            .and_then(|n| {
                n.checked_add(NODE_WINDOW.checked_mul(size_of::<vk::DescriptorSet>())?)
                    .and_then(|n| {
                        n.checked_add(sets.checked_mul(size_of::<vk::DescriptorSetLayout>())?)
                    })
            })
            .and_then(|n| {
                n.checked_add(bindings.checked_mul(
                    size_of::<vk::DescriptorBufferInfo>()
                        + size_of::<vk::WriteDescriptorSet<'static>>()
                        + size_of::<vk::DescriptorType>(),
                )?)
            })
            .ok_or("native descriptor metadata overflow")?;
        if peak_bytes > TEMPORARY_CAP
            || retained_bytes
                .checked_add(peak_bytes)
                .is_none_or(|n| n > METADATA_CAP)
        {
            return Err("native descriptor bounded temporary metadata capacity");
        }
        Ok(Self {
            nodes,
            sets,
            bindings,
            peak_bytes,
        })
    }
    pub(super) fn node_count(&self) -> usize {
        self.nodes.len()
    }
    pub(super) fn set_count(&self) -> usize {
        self.sets
    }
    pub(super) fn binding_count(&self) -> usize {
        self.bindings
    }
    pub(super) fn set_index(&self, node: usize) -> Result<Option<usize>, &'static str> {
        self.nodes
            .get(node)
            .map(|n| n.set)
            .ok_or("native descriptor original node index")
    }
    pub(super) fn binding_range(&self, node: usize) -> Result<Range<usize>, &'static str> {
        self.nodes
            .get(node)
            .map(|n| n.bindings.clone())
            .ok_or("native descriptor original binding range")
    }
    pub(super) fn peak_bytes(&self) -> usize {
        self.peak_bytes
    }
    /// Exact production writes. Infos are complete before any pointer is taken;
    /// the returned borrow prevents later vector growth until the update ends.
    pub(super) fn writes<'a>(
        &self,
        sets: &[vk::DescriptorSet],
        infos: &'a [vk::DescriptorBufferInfo],
        types: &[vk::DescriptorType],
    ) -> Result<Vec<vk::WriteDescriptorSet<'a>>, &'static str> {
        if sets.len() != self.sets || infos.len() != self.bindings || types.len() != self.bindings {
            return Err("native descriptor complete immutable write inventory");
        }
        let mut writes = Vec::with_capacity(self.bindings);
        for node in &self.nodes {
            let Some(set) = node.set else { continue };
            for (binding, index) in node.bindings.clone().enumerate() {
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(sets[set])
                        .dst_binding(binding as u32)
                        .descriptor_type(types[index])
                        .buffer_info(std::slice::from_ref(&infos[index])),
                );
            }
        }
        Ok(writes)
    }
}
#[derive(Debug)]
pub(super) struct Batch {
    plan: Plan,
    // Only host handles; this type NEVER frees/recycles Vulkan descriptor sets.
    // The original Recording's pool owns them through its real terminal fence.
    sets: Vec<vk::DescriptorSet>,
}
impl Batch {
    pub(super) fn new(plan: Plan, sets: Vec<vk::DescriptorSet>) -> Result<Self, &'static str> {
        if sets.len() != plan.set_count() || sets.capacity() > NODE_WINDOW {
            return Err("native descriptor actual allocated set inventory");
        }
        Ok(Self { plan, sets })
    }
    pub(super) fn set(&self, node: usize) -> Result<vk::DescriptorSet, &'static str> {
        let index = self
            .plan
            .set_index(node)?
            .ok_or("native descriptor compute set missing")?;
        self.sets
            .get(index)
            .copied()
            .ok_or("native descriptor allocated set index")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle as _;
    #[test]
    fn original_nodes_and_each_binding_keep_their_unique_fresh_set_in_order() {
        let plan = Plan::new([Some(4), None, Some(9), None, Some(6)], 0).unwrap();
        assert_eq!(plan.set_count(), 3);
        assert_eq!(plan.binding_count(), 19);
        for (node, set, range) in [
            (0, Some(0), 0..4),
            (1, None, 4..4),
            (2, Some(1), 4..13),
            (3, None, 13..13),
            (4, Some(2), 13..19),
        ] {
            assert_eq!(plan.set_index(node).unwrap(), set);
            assert_eq!(plan.binding_range(node).unwrap(), range);
        }
        let batch =
            Batch::new(plan, (101..104).map(vk::DescriptorSet::from_raw).collect()).unwrap();
        assert_eq!(batch.set(0).unwrap().as_raw(), 101);
        assert_eq!(batch.set(2).unwrap().as_raw(), 102);
        assert_eq!(batch.set(4).unwrap().as_raw(), 103);
        assert!(batch.set(1).is_err());
        assert!(batch.set(5).is_err());
    }
    #[test]
    fn bounded_windows_have_exact_bulk_call_inventory_without_reordering() {
        for total in [1usize, 127, 128, 129, 2048, 32768] {
            let original: Vec<_> = (0..total)
                .map(|n| if n % 7 == 0 { None } else { Some(n % 32 + 1) })
                .collect();
            let mut allocations = 0;
            let mut updates = 0;
            let mut emitted = Vec::new();
            for window in original.chunks(NODE_WINDOW) {
                let plan = Plan::new(window.iter().copied(), 0).unwrap();
                if plan.set_count() != 0 {
                    allocations += 1;
                    updates += 1;
                }
                let mut previous_set = None;
                for (node, count) in window.iter().enumerate() {
                    let set = plan.set_index(node).unwrap();
                    assert_eq!(set.is_some(), count.is_some());
                    if let Some(set) = set {
                        assert_eq!(set, previous_set.map_or(0, |p| p + 1));
                        previous_set = Some(set);
                    }
                    emitted.push(*count);
                }
            }
            assert_eq!(emitted, original);
            assert!(allocations <= total.div_ceil(NODE_WINDOW));
            assert_eq!(allocations, updates);
        }
        let plan = Plan::new([None; NODE_WINDOW], 0).unwrap();
        assert_eq!((plan.set_count(), plan.binding_count()), (0, 0));
        assert!(Batch::new(plan, Vec::new()).is_ok());
    }
    #[test]
    fn real_vulkan_metadata_and_original_caps_reject_incomplete_or_excess_inventory() {
        assert!(Plan::new([], 0).is_err());
        assert!(Plan::new([Some(0)], 0).is_err());
        assert!(Plan::new([Some(33)], 0).is_err());
        assert!(Plan::new([Some(1); NODE_WINDOW + 1], 0).is_err());
        let plan = Plan::new([Some(32); NODE_WINDOW], 0).unwrap();
        assert_eq!(
            (plan.node_count(), plan.set_count(), plan.binding_count()),
            (128, 128, 4096)
        );
        assert!(plan.peak_bytes() <= TEMPORARY_CAP);
        assert!(Plan::new([Some(32); NODE_WINDOW], METADATA_CAP - plan.peak_bytes()).is_ok());
        assert!(
            Plan::new(
                [Some(32); NODE_WINDOW],
                METADATA_CAP - plan.peak_bytes() + 1
            )
            .is_err()
        );
        assert!(Plan::new([None], usize::MAX).is_err());
        assert!(Batch::new(Plan::new([Some(1)], 0).unwrap(), Vec::new()).is_err());
    }
    #[test]
    fn production_writes_keep_original_buffers_ranges_types_and_pointer_lifetimes() {
        let plan = Plan::new([Some(2), None, Some(1)], 0).unwrap();
        let sets = [
            vk::DescriptorSet::from_raw(101),
            vk::DescriptorSet::from_raw(102),
        ];
        let infos: Vec<_> = (0..3)
            .map(|i| {
                vk::DescriptorBufferInfo::default()
                    .buffer(vk::Buffer::from_raw(201 + i))
                    .offset(256 + i * 128)
                    .range(64 + i * 32)
            })
            .collect();
        let types = [
            vk::DescriptorType::STORAGE_BUFFER,
            vk::DescriptorType::UNIFORM_BUFFER,
            vk::DescriptorType::STORAGE_BUFFER,
        ];
        let writes = plan.writes(&sets, &infos, &types).unwrap();
        assert_eq!(writes.len(), 3);
        for (i, w) in writes.iter().enumerate() {
            assert_eq!(w.dst_set, sets[usize::from(i == 2)]);
            assert_eq!(w.dst_binding, if i == 2 { 0 } else { i as u32 });
            assert_eq!(w.descriptor_type, types[i]);
            assert_eq!(w.descriptor_count, 1);
            assert_eq!(w.p_buffer_info, std::ptr::from_ref(&infos[i]));
            // SAFETY: production writes borrow these still-live actual infos.
            let original = unsafe { &*w.p_buffer_info };
            assert_eq!(
                (original.buffer, original.offset, original.range),
                (infos[i].buffer, infos[i].offset, infos[i].range)
            );
        }
        assert!(plan.writes(&sets[..1], &infos, &types).is_err());
        assert!(plan.writes(&sets, &infos[..2], &types).is_err());
        assert!(plan.writes(&sets, &infos, &types[..2]).is_err());
    }
    #[test]
    fn actual_capture_closes_and_issues_prior_prefix_before_future_window_preparation() {
        let source = include_str!("strata_graph.rs");
        let capture = source
            .split_once("let mut fresh_descriptors = None;")
            .unwrap()
            .1;
        let capture = capture
            .split_once("let copy = work.plan.segments[segment].copy;")
            .unwrap()
            .0;
        let mut previous = 0;
        for marker in [
            "if next != segment",
            "end_native_command",
            "rolling.closed(next, index, false)",
            "begin_native_command",
            "drop(fresh_descriptors.take())",
            "self.fresh_descriptors(",
        ] {
            let next = capture.find(marker).expect(marker);
            assert!(next >= previous, "capture ordering: {marker}");
            previous = next;
        }
        assert_eq!(capture.matches("self.fresh_descriptors(").count(), 1);
    }
}
