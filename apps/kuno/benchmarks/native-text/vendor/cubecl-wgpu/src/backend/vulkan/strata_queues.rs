//! Select real independent Intel queues; never infer queue1 in family0.
use ash::vk;

pub(super) fn supports_timestamps(properties: &[vk::QueueFamilyProperties], family: u32) -> bool {
    properties
        .get(family as usize)
        .is_some_and(|q| q.queue_count > 0 && q.timestamp_valid_bits >= 36)
}

/// Select a real primary family, never a logical lane or a second queue in a
/// one-queue family. The explicit compute candidate declines unsupported
/// topologies instead of silently falling back to the render engine.
pub(super) fn main_family(
    properties: &[vk::QueueFamilyProperties],
    compute_only: bool,
) -> Option<u32> {
    if !compute_only {
        families(properties, 0)?;
        return Some(0);
    }
    properties.iter().enumerate().find_map(|(i, q)| {
        if q.queue_count == 0
            || !q.queue_flags.contains(vk::QueueFlags::COMPUTE)
            || q.queue_flags.contains(vk::QueueFlags::GRAPHICS)
        {
            return None;
        }
        let main = u32::try_from(i).ok()?;
        families(properties, main)?;
        Some(main)
    })
}

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

    #[test]
    fn compute_primary_preserves_three_distinct_actual_queue_zero_roles() {
        let c = vk::QueueFlags::COMPUTE;
        let t = vk::QueueFlags::TRANSFER;
        let g = vk::QueueFlags::GRAPHICS;
        let topology = [q(g | c | t, 1), q(c | t, 1), q(t, 1)];
        assert_eq!(main_family(&topology, false), Some(0));
        let main = main_family(&topology, true).unwrap();
        let (draft, copy) = families(&topology, main).unwrap();
        assert_eq!([main, draft, copy], [1, 0, 2]);
        assert_eq!(
            [main, draft, copy].map(|i| topology[i as usize].queue_count),
            [1; 3]
        );
    }

    #[test]
    fn compute_primary_is_capability_based_even_when_families_are_reordered() {
        let c = vk::QueueFlags::COMPUTE;
        let t = vk::QueueFlags::TRANSFER;
        let g = vk::QueueFlags::GRAPHICS;
        let topology = [q(t, 1), q(c, 0), q(g | c | t, 1), q(c | t, 1)];
        assert_eq!(main_family(&topology, true), Some(3));
        assert_eq!(families(&topology, 3), Some((2, 0)));
    }

    #[test]
    fn compute_primary_refuses_missing_zero_count_or_aliased_roles() {
        let c = vk::QueueFlags::COMPUTE;
        let t = vk::QueueFlags::TRANSFER;
        let g = vk::QueueFlags::GRAPHICS;
        for topology in [
            vec![],
            vec![q(g | c, 1), q(t, 1)],
            vec![q(g | c, 1), q(c, 0), q(t, 1)],
            vec![q(c, 1), q(t, 1)],
            vec![q(g | c, 1), q(c, 1), q(t, 0)],
            vec![q(g | c, 1), q(c | t, 1)],
        ] {
            assert_eq!(main_family(&topology, true), None);
        }
    }

    #[test]
    fn timestamp_admission_checks_actual_selected_family_not_render_family_zero() {
        let mut topology = [
            q(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE, 1),
            q(vk::QueueFlags::COMPUTE, 1),
        ];
        topology[0].timestamp_valid_bits = 64;
        for bits in [0, 35, 36, 64] {
            topology[1].timestamp_valid_bits = bits;
            assert!(supports_timestamps(&topology, 0));
            assert_eq!(supports_timestamps(&topology, 1), bits >= 36);
        }
        topology[1].queue_count = 0;
        assert!(!supports_timestamps(&topology, 1));
        assert!(!supports_timestamps(&topology, 9));
    }
}
