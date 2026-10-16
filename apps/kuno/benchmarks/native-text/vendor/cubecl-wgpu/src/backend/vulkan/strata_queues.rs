//! Select real independent Intel queues; never infer queue1 in family0.
use ash::vk;

pub(super) fn families(properties: &[vk::QueueFamilyProperties], main: u32) -> Option<(u32, u32)> {
    let primary = properties.get(main as usize)?;
    if primary.queue_count == 0 || !primary.queue_flags.contains(vk::QueueFlags::COMPUTE) {
        return None;
    }
    let copy = properties
        .iter()
        .enumerate()
        .find(|(i, q)| {
            *i != main as usize
                && q.queue_count > 0
                && q.queue_flags.contains(vk::QueueFlags::TRANSFER)
                && !q
                    .queue_flags
                    .intersects(vk::QueueFlags::COMPUTE | vk::QueueFlags::GRAPHICS)
        })?
        .0 as u32;
    let draft = properties
        .iter()
        .enumerate()
        .filter(|(i, q)| {
            *i != main as usize
                && *i != copy as usize
                && q.queue_count > 0
                && q.queue_flags.contains(vk::QueueFlags::COMPUTE)
        })
        .min_by_key(|(_, q)| q.queue_flags.contains(vk::QueueFlags::GRAPHICS))?
        .0 as u32;
    Some((draft, copy))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn q(flags: vk::QueueFlags, count: u32) -> vk::QueueFamilyProperties {
        vk::QueueFamilyProperties {
            queue_flags: flags,
            queue_count: count,
            ..Default::default()
        }
    }
    #[test]
    fn b70_uses_three_distinct_real_families_not_queue_one() {
        let c = vk::QueueFlags::COMPUTE;
        let t = vk::QueueFlags::TRANSFER;
        let g = vk::QueueFlags::GRAPHICS;
        let topology = [q(g | c | t, 1), q(c | t, 1), q(t, 1)];
        assert_eq!(families(&topology, 0), Some((1, 2)));
        assert_eq!(families(&topology, 1), Some((0, 2)));
    }
    #[test]
    fn missing_zero_count_or_noncompute_queues_are_not_silently_aliased() {
        let c = vk::QueueFlags::COMPUTE;
        let t = vk::QueueFlags::TRANSFER;
        for topology in [
            vec![],
            vec![q(c | t, 1)],
            vec![q(c | t, 1), q(t, 1)],
            vec![q(c | t, 1), q(c, 0), q(t, 1)],
            vec![q(t, 1), q(c, 1), q(t, 1)],
        ] {
            assert!(families(&topology, 0).is_none());
        }
        assert!(families(&[q(c, 1), q(c, 1), q(t, 1)], 9).is_none());
    }
    #[test]
    fn transfer_and_compute_selection_follows_properties_not_numeric_ids() {
        let c = vk::QueueFlags::COMPUTE;
        let t = vk::QueueFlags::TRANSFER;
        let g = vk::QueueFlags::GRAPHICS;
        let topology = [q(c | g, 1), q(c | g, 1), q(t, 0), q(c, 1), q(t, 1)];
        assert_eq!(families(&topology, 0), Some((3, 4)));
    }
}
