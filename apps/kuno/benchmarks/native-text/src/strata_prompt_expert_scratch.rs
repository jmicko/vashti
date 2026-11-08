//! Two genuine disjoint expert scratch slots inside the original priced roots.
//! The complete six-phase program and original graph hazards remain authoritative.
use super::*;

pub(super) const SLOTS: usize = 2;
const INPUT_BYTES: u64 = (groups::CAPACITY * 2560 * 2) as u64;
const EXTRA_GU_FIRST: u64 = SLOTS as u64 * INPUT_BYTES;
const USED_BYTES: u64 = EXTRA_GU_FIRST + groups::GU_BYTES;

#[derive(Clone)]
pub(super) struct Slot {
    pub input: Handle,
    pub gu: Handle,
    pub gu_image: Handle,
    pub down_image: Handle,
}

fn view(owner: &Handle, first: u64, bytes: u64) -> Result<Handle> {
    if bytes == 0
        || !first.is_multiple_of(256)
        || !bytes.is_multiple_of(256)
        || first
            .checked_add(bytes)
            .is_none_or(|end| end > owner.size_in_used())
        || !owner.offset_start.unwrap_or(0).is_multiple_of(256)
    {
        return Err(Error::InvalidBufferLength);
    }
    Ok(owner
        .clone()
        .offset_start(first)
        .offset_end(owner.size_in_used() - first - bytes))
}

pub(super) fn slots(stage: &[Handle; 3], gu: &Handle) -> Result<[Slot; SLOTS]> {
    if USED_BYTES > storage::WORKING_BYTES
        || stage[0].size_in_used() != storage::WORKING_BYTES
        || stage[1].size_in_used() != storage::WORKING_BYTES
        || gu.size_in_used() != groups::GU_BYTES
        || !gu.offset_start.unwrap_or(0).is_multiple_of(256)
        || stage
            .iter()
            .chain([gu])
            .any(|h| h.stream != stage[0].stream)
        || DEQUANT_SLOTS != SLOTS
    {
        return Err(Error::InvalidBufferLength);
    }
    let make = |slot: usize| -> Result<Slot> {
        let first = (slot * DQ_WORDS * 4) as u64;
        Ok(Slot {
            input: view(&stage[0], slot as u64 * INPUT_BYTES, INPUT_BYTES)?,
            gu: if slot == 0 {
                gu.clone()
            } else {
                view(&stage[0], EXTRA_GU_FIRST, groups::GU_BYTES)?
            },
            gu_image: view(&stage[1], first, (GU_WORDS * 4) as u64)?,
            down_image: view(
                &stage[1],
                first + (GU_WORDS * 4) as u64,
                (DOWN_WORDS * 4) as u64,
            )?,
        })
    };
    let slots = [make(0)?, make(1)?];
    let views = slots
        .iter()
        .flat_map(|s| [&s.input, &s.gu, &s.gu_image, &s.down_image])
        .collect::<Vec<_>>();
    for (i, a) in views.iter().enumerate() {
        let a_binding = (*a).clone().binding();
        let first = a.offset_start.unwrap_or(0);
        let end = first
            .checked_add(a.size_in_used())
            .ok_or(Error::InvalidBufferLength)?;
        for b in &views[..i] {
            let b_binding = (*b).clone().binding();
            let b_first = b.offset_start.unwrap_or(0);
            let b_end = b_first
                .checked_add(b.size_in_used())
                .ok_or(Error::InvalidBufferLength)?;
            if a_binding.memory.allocation_id() == b_binding.memory.allocation_id()
                && first < b_end
                && b_first < end
            {
                return Err(Error::InvalidBufferLength);
            }
        }
    }
    Ok(slots)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl_common::stream_id::StreamId;

    fn owners() -> ([Handle; 3], Handle) {
        let stream = StreamId::current();
        let roots = [
            Handle::new(stream, 1 << 30),
            Handle::new(stream, 1 << 30),
            Handle::new(stream, allocation::RAW_RING_BYTES),
        ];
        let stage = [
            roots[0]
                .clone()
                .offset_end((1 << 30) - storage::WORKING_BYTES),
            roots[1]
                .clone()
                .offset_end((1 << 30) - storage::WORKING_BYTES),
            roots[2].clone(),
        ];
        let gu = view(&roots[1], storage::WORKING_BYTES, groups::GU_BYTES).unwrap();
        (stage, gu)
    }
    fn extent(h: &Handle) -> (cubecl_runtime::memory_management::ManagedMemoryId, u64, u64) {
        let b = h.clone().binding();
        (
            b.memory.allocation_id(),
            h.offset_start.unwrap_or(0),
            h.offset_start.unwrap_or(0) + h.size_in_used(),
        )
    }
    #[test]
    fn separate_role_shader_preserves_every_original_source_byte_and_release_class() {
        use cubecl_runtime::server::native_expert_batch::native_strata_expert_operator;
        for op in slot_operators() {
            let Op::SlotDequant {
                dtype,
                down,
                resident,
            } = op
            else {
                unreachable!()
            };
            let old = if resident {
                Op::ResidentDequant { dtype, down }
            } else {
                Op::Dequant { dtype, down }
            };
            let original = old.source();
            let actual = op.source();
            let guard = slot_image_guard(down);
            assert_eq!(actual.matches(&guard).count(), 1);
            assert_eq!(
                actual.replace(&guard, "arrayLength(&packed)!=PROMPT_WORKING_WORDS/4u"),
                original
            );
            assert_eq!(op.native_release_after(), old.native_release_after());
            assert_eq!(op.name(), old.name());
            assert_ne!(op.id(), old.id());
            assert!(native_strata_expert_operator(
                op.name(),
                op.native_release_after(),
                if down { 2 } else { 1 }
            ));
        }
    }
    #[test]
    fn actual_views_keep_original_owners_and_all_slots_and_roles_disjoint() {
        let (stage, gu) = owners();
        let slots = slots(&stage, &gu).unwrap();
        let views = slots
            .iter()
            .flat_map(|s| [&s.input, &s.gu, &s.gu_image, &s.down_image])
            .collect::<Vec<_>>();
        for (i, h) in views.iter().enumerate() {
            let (root, first, end) = extent(h);
            assert!(first.is_multiple_of(256) && end.is_multiple_of(256));
            for other in &views[..i] {
                let (r, a, b) = extent(other);
                assert!(r != root || end <= a || b <= first);
            }
        }
        assert_eq!(extent(&slots[0].gu), extent(&gu));
        assert_eq!(USED_BYTES, (crate::strata_prompt::groups::CAPACITY * 15360) as u64);
        assert!(USED_BYTES <= storage::WORKING_BYTES);
        for s in &slots {
            assert_eq!(s.input.size_in_used(), (groups::CAPACITY * 2560 * 2) as u64);
            assert_eq!(s.gu.size_in_used(), groups::GU_BYTES);
            assert_eq!(s.gu_image.size_in_used(), (GU_WORDS * 4) as u64);
            assert_eq!(s.down_image.size_in_used(), (DOWN_WORDS * 4) as u64);
            assert_eq!(extent(&s.input).0, extent(&stage[0]).0);
            assert_eq!(extent(&s.gu_image).0, extent(&stage[1]).0);
            assert_eq!(extent(&s.down_image).0, extent(&stage[1]).0);
        }
    }
    #[test]
    fn invalid_extents_alignment_and_capacity_are_rejected_before_views_escape() {
        let (stage, gu) = owners();
        for (first, bytes) in [
            (1, 256),
            (0, 0),
            (0, 257),
            (u64::MAX - 255, 256),
            (storage::WORKING_BYTES, 256),
        ] {
            assert!(view(&stage[0], first, bytes).is_err());
        }
        let mut bad = stage.clone();
        bad[0] = bad[0].clone().offset_end(256);
        assert!(slots(&bad, &gu).is_err());
        assert!(slots(&stage, &gu.clone().offset_end(256)).is_err());
        assert!(slots(&stage, &view(&stage[0], 0, groups::GU_BYTES).unwrap()).is_err());
        let mut bad = stage.clone();
        bad[1] = bad[0].clone();
        assert!(slots(&bad, &gu).is_err());
    }
}
