//! Allocation-free permutation of an ORIGINAL completed command inventory.
//! This module grants no completion, pool-reset, source or descriptor authority.
use ash::vk;
const MAX_COMMANDS: usize = 32769;

pub(super) fn fits_budget(
    plan_bytes: Option<usize>,
    command_capacity: usize,
    object_bytes: usize,
    budget: usize,
) -> bool {
    plan_bytes
        .and_then(|n| n.checked_add(object_bytes))
        .and_then(|n| n.checked_add(command_capacity.checked_mul(size_of::<vk::CommandBuffer>())?))
        .is_some_and(|n| n <= budget)
}

pub(super) fn compatible<S>(
    old: &[S],
    queue: fn(&S) -> usize,
    new: impl Iterator<Item = usize> + Clone + ExactSizeIterator,
    commands: &[vk::CommandBuffer],
    scratch_len: usize,
) -> bool {
    let count = old.len();
    if !(1..=MAX_COMMANDS).contains(&count)
        || count != new.len()
        || count != commands.len()
        || scratch_len < count - 1
        || commands.contains(&vk::CommandBuffer::null())
        || queue(&old[0]) != 0
        || new.clone().next() != Some(0)
    {
        return false;
    }
    let mut before = [0usize; 3];
    let mut after = [0usize; 3];
    for own in old.iter().map(queue) {
        if own >= 3 {
            return false;
        }
        before[own] += 1;
    }
    for own in new {
        if own >= 3 {
            return false;
        }
        after[own] += 1;
    }
    before == after
}

pub(super) fn reorder<S>(
    old: &mut [S],
    queue: fn(&S) -> usize,
    new: impl Iterator<Item = usize> + Clone + ExactSizeIterator,
    commands: &mut [vk::CommandBuffer],
    scratch: &mut [usize],
) -> Result<(), &'static str> {
    if !compatible(old, queue, new.clone(), commands, scratch.len()) {
        return Err("native completed command inventory mismatch");
    }
    // Keep the original MAIN header at0. Partition every other original handle
    // by its actual pool's queue; move its old queue evidence with it.
    let mut cursor = [0usize; 3];
    let mut first = 1;
    for own in 0..3 {
        cursor[own] = first;
        for i in first..old.len() {
            if queue(&old[i]) == own {
                old.swap(first, i);
                commands.swap(first, i);
                first += 1;
            }
        }
    }
    // The old node->segment Vec already has at least segments-1 entries.
    // Its contents are no longer an executable plan and are replaced below.
    for (i, own) in new.enumerate().skip(1) {
        scratch[i - 1] = cursor[own];
        cursor[own] += 1;
    }
    // Apply source-index permutation cycles in O(n), using one actual handle.
    // No handle is created, lost, duplicated or moved to a different pool.
    for i in 1..commands.len() {
        if scratch[i - 1] == i {
            continue;
        }
        let saved = commands[i];
        let mut current = i;
        loop {
            let source = scratch[current - 1];
            scratch[current - 1] = current;
            if source == i {
                commands[current] = saved;
                break;
            }
            commands[current] = commands[source];
            current = source;
        }
    }
    Ok(())
}

/// Complete existing pool inventory, including inactive ORIGINAL command
/// handles. Preparation is CPU-only and cannot reset or submit any pool.
pub(super) struct Prepared {
    pub(super) commands: Vec<vk::CommandBuffer>,
    pub(super) spares: [Vec<vk::CommandBuffer>; 3],
    pub(super) header: vk::CommandBuffer,
}
impl Prepared {
    pub(super) fn metadata_bytes(&self, plan_bytes: usize, object_bytes: usize) -> Option<usize> {
        let capacity = self
            .spares
            .iter()
            .try_fold(self.commands.capacity(), |n, v| n.checked_add(v.capacity()))?;
        plan_bytes
            .checked_add(object_bytes)?
            .checked_add(size_of::<Self>())?
            .checked_add(capacity.checked_mul(size_of::<vk::CommandBuffer>())?)
    }
}

pub(super) fn prepare_capacity<S>(
    old: &[S],
    queue: fn(&S) -> usize,
    commands: &[vk::CommandBuffer],
    spares: &[Vec<vk::CommandBuffer>; 3],
    new: impl Iterator<Item = usize> + Clone + ExactSizeIterator,
    plan_bytes: usize,
    object_bytes: usize,
    budget: usize,
) -> Option<Prepared> {
    let count = new.len();
    if !(1..=MAX_COMMANDS).contains(&count)
        || old.is_empty()
        || commands.len() != old.len()
        || old.len() > MAX_COMMANDS
        || queue(&old[0]) != 0
        || new.clone().next() != Some(0)
        || commands.contains(&vk::CommandBuffer::null())
        || spares
            .iter()
            .any(|v| v.contains(&vk::CommandBuffer::null()))
    {
        return None;
    }
    let mut available = std::array::from_fn::<_, 3, _>(|i| spares[i].len());
    for own in old.iter().skip(1).map(queue) {
        if own >= 3 {
            return None;
        }
        available[own] = available[own].checked_add(1)?;
    }
    let total = available
        .iter()
        .try_fold(1usize, |n, x| n.checked_add(*x))?;
    if total > MAX_COMMANDS {
        return None;
    }
    let mut needed = [0usize; 3];
    for own in new.clone().skip(1) {
        if own >= 3 {
            return None;
        }
        needed[own] += 1;
    }
    if needed
        .iter()
        .zip(available)
        .any(|(need, have)| *need > have)
    {
        return None;
    }
    let requested = available.iter().try_fold(count, |n, x| n.checked_add(*x))?;
    let fixed = plan_bytes
        .checked_add(object_bytes)?
        .checked_add(size_of::<Prepared>())?;
    if !fits_budget(Some(0), requested, fixed, budget) {
        return None;
    }
    let mut result = Prepared {
        commands: Vec::new(),
        spares: std::array::from_fn(|_| Vec::new()),
        header: commands[0],
    };
    result.commands.try_reserve_exact(count).ok()?;
    if result.metadata_bytes(plan_bytes, object_bytes)? > budget {
        return None;
    }
    for own in 0..3 {
        result.spares[own].try_reserve_exact(available[own]).ok()?;
        if result.metadata_bytes(plan_bytes, object_bytes)? > budget {
            return None;
        }
        result.spares[own].extend_from_slice(&spares[own]);
    }
    for (segment, &handle) in old.iter().zip(commands).skip(1) {
        result.spares[queue(segment)].push(handle);
    }
    result.commands.push(commands[0]);
    for own in new.skip(1) {
        result.commands.push(result.spares[own].pop()?);
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;
    fn check(before: Vec<usize>, after: Vec<usize>) {
        let mut old = before.clone();
        let mut handles: Vec<_> = (1..=old.len())
            .map(|n| vk::CommandBuffer::from_raw(n as u64))
            .collect();
        let mut scratch = vec![usize::MAX; old.len() + 3];
        reorder(
            &mut old,
            |q| *q,
            after.iter().copied(),
            &mut handles,
            &mut scratch,
        )
        .unwrap();
        assert_eq!(handles[0].as_raw(), 1);
        for (i, h) in handles.iter().enumerate() {
            assert_eq!(before[h.as_raw() as usize - 1], after[i]);
        }
        let mut inventory: Vec<_> = handles.iter().map(|h| h.as_raw()).collect();
        inventory.sort_unstable();
        assert_eq!(inventory, (1..=before.len() as u64).collect::<Vec<_>>());
        assert!(scratch[before.len() - 1..].iter().all(|x| *x == usize::MAX));
    }
    #[test]
    fn capacity_reuse_keeps_unused_commands_in_their_original_pools_and_can_grow_again() {
        let original: Vec<_> = (0..4096).map(|i| if i == 0 { 0 } else { i % 3 }).collect();
        let mut queues = original.clone();
        let mut commands: Vec<_> = (1..=queues.len())
            .map(|n| vk::CommandBuffer::from_raw(n as u64))
            .collect();
        let mut spares = std::array::from_fn(|_| Vec::new());
        for size in [2048, 16, 3000, 512, 4096] {
            let mut next = original[..size].to_vec();
            next[1..].reverse();
            let p = prepare_capacity(
                &queues,
                |q| *q,
                &commands,
                &spares,
                next.iter().copied(),
                4096,
                256,
                4 * 1024 * 1024,
            )
            .unwrap();
            assert_eq!(p.commands[0].as_raw(), 1);
            assert!(p.metadata_bytes(4096, 256).unwrap() <= 4 * 1024 * 1024);
            let mut all = Vec::new();
            for (i, h) in p.commands.iter().enumerate() {
                assert_eq!(original[h.as_raw() as usize - 1], next[i]);
                all.push(h.as_raw());
            }
            for (own, list) in p.spares.iter().enumerate() {
                for h in list {
                    assert_eq!(original[h.as_raw() as usize - 1], own);
                    all.push(h.as_raw());
                }
            }
            all.sort_unstable();
            assert_eq!(all, (1..=4096).collect::<Vec<u64>>());
            commands = p.commands;
            spares = p.spares;
            queues = next;
        }
        assert!(spares.iter().all(Vec::is_empty));
    }
    #[test]
    fn capacity_preparation_rejects_insufficient_queues_budget_and_unknown_handles_without_mutation()
     {
        let old = [0, 1, 2, 0];
        let handles: Vec<_> = (1..=4).map(vk::CommandBuffer::from_raw).collect();
        let spares = std::array::from_fn(|_| Vec::new());
        for new in [vec![0, 1, 1], vec![0, 3], vec![1, 0], vec![], vec![0, 2, 2]] {
            assert!(
                prepare_capacity(
                    &old,
                    |q| *q,
                    &handles,
                    &spares,
                    new.iter().copied(),
                    0,
                    0,
                    4096
                )
                .is_none()
            );
        }
        assert!(
            prepare_capacity(
                &old,
                |q| *q,
                &handles,
                &spares,
                old.iter().copied(),
                0,
                0,
                1
            )
            .is_none()
        );
        assert!(
            prepare_capacity(
                &old,
                |q| *q,
                &handles,
                &spares,
                old.iter().copied(),
                usize::MAX,
                0,
                usize::MAX
            )
            .is_none()
        );
        let mut bad = spares.clone();
        bad[0].push(vk::CommandBuffer::null());
        assert!(
            prepare_capacity(
                &old,
                |q| *q,
                &handles,
                &bad,
                old.iter().copied(),
                0,
                0,
                4096
            )
            .is_none()
        );
        assert_eq!(
            handles.iter().map(|h| h.as_raw()).collect::<Vec<_>>(),
            [1, 2, 3, 4]
        );
        assert!(spares.iter().all(Vec::is_empty));
    }
    #[test]
    fn actual_old_command_capacity_and_new_plan_are_charged_without_overflow() {
        let handle = size_of::<vk::CommandBuffer>();
        assert!(fits_budget(
            Some(4096),
            32769,
            256,
            4096 + 32769 * handle + 256
        ));
        assert!(!fits_budget(
            Some(4096),
            32769,
            256,
            4096 + 32769 * handle + 255
        ));
        assert!(!fits_budget(None, 1, 1, usize::MAX));
        assert!(!fits_budget(Some(usize::MAX), 1, 1, usize::MAX));
        assert!(!fits_budget(Some(0), usize::MAX, 0, usize::MAX));
    }
    #[test]
    fn every_small_inventory_preserves_original_header_handles_and_physical_queues() {
        for n in 0..=7 {
            for mut bits in 0..3usize.pow(n) {
                let mut a = vec![0];
                for _ in 0..n {
                    a.push(bits % 3);
                    bits /= 3;
                }
                let mut b = a.clone();
                b[1..].reverse();
                check(a.clone(), b);
                let mut b = a.clone();
                b[1..].sort_unstable();
                check(a, b);
            }
        }
    }
    #[test]
    fn large_mixed_permutations_keep_every_real_command_without_extra_scratch() {
        for n in [16, 512, 4096, MAX_COMMANDS] {
            let a: Vec<_> = (0..n)
                .map(|i| if i == 0 { 0 } else { (i * 17 + i / 7) % 3 })
                .collect();
            let mut b = a.clone();
            b[1..].rotate_left(n / 3);
            b[1..].reverse();
            check(a, b);
        }
    }
    #[test]
    fn incompatible_or_incomplete_inventories_are_rejected_before_mutation() {
        let a = [0usize, 1, 2, 0];
        let h = vec![vk::CommandBuffer::from_raw(1); 4];
        for b in [
            vec![0, 1, 2],
            vec![1, 0, 2, 0],
            vec![0, 1, 3, 0],
            vec![0, 1, 1, 0],
        ] {
            let mut old = a;
            let mut handles = h.clone();
            let mut scratch = [usize::MAX; 3];
            assert!(
                reorder(
                    &mut old,
                    |q| *q,
                    b.iter().copied(),
                    &mut handles,
                    &mut scratch
                )
                .is_err()
            );
            assert_eq!(old, a);
            assert_eq!(handles, h);
            assert_eq!(scratch, [usize::MAX; 3]);
        }
        assert!(!compatible(&a, |q| *q, a.iter().copied(), &h, 2));
        let mut null = h;
        null[2] = vk::CommandBuffer::null();
        assert!(!compatible(&a, |q| *q, a.iter().copied(), &null, 3));
        assert!(!compatible(
            &[0, 3, 2, 0],
            |q| *q,
            a.iter().copied(),
            &null,
            3
        ));
    }
}
