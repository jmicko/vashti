//! Compact actual-cache proof. No timing or publication authority is exposed.
//! The all-ten plan is temporary; only the canonical GPU subset survives in the
//! original native Pending, whose bank identity and full handles pin the owner.
use super::*;
use crate::cpu_gpu_expert_schedule::Split;
use kuno_text::cpu_expert::Partials;

pub(super) struct Proof {
    ids: Zeroizing<[usize; 10]>,
    probabilities: Zeroizing<[u32; 10]>,
    cap: Zeroizing<usize>,
    misses: Zeroizing<u16>,
    hit_slots: Zeroizing<[usize; 10]>,
    cpu_count: Zeroizing<usize>,
    joined: Zeroizing<bool>,
}
impl Proof {
    pub(super) fn prepare(
        source: &crate::bank_slots::BankSlots,
        ids: &[usize; 10],
        probabilities: &[f32; 10],
        cap: usize,
    ) -> Result<(Self, StagedSlots, Last, Split)> {
        if ![0, 10].contains(&cap) || probabilities.iter().any(|p| !p.is_finite() || *p < 0.) {
            return Err(err("native count1 fixed miss policy"));
        }
        let mut staged = StagedSlots::begin(source).map_err(|_| err("native count1 origin"))?;
        let all = staged
            .plan(source, ids)
            .map_err(|_| err("native count1 all-rank plan"))?;
        staged
            .validate_pending(source, &all, ids)
            .map_err(|_| err("native count1 all-rank classification"))?;
        let mut misses = Zeroizing::new([0usize; 10]);
        if all.uploads.len() > 10 {
            return Err(Error::InvalidBufferLength);
        }
        for (i, &(id, _)) in all.uploads.iter().enumerate() {
            misses[i] = id;
        }
        let split = Split::from_actual_misses(ids, &misses[..all.uploads.len()], cap)?;
        let mut proof = Self {
            ids: Zeroizing::new(*ids),
            probabilities: Zeroizing::new(probabilities.map(f32::to_bits)),
            cap: Zeroizing::new(cap),
            misses: Zeroizing::new(0),
            hit_slots: Zeroizing::new([0; 10]),
            cpu_count: Zeroizing::new(split.cpu().len()),
            joined: Zeroizing::new(false),
        };
        for (rank, &id) in ids.iter().enumerate() {
            if misses[..all.uploads.len()].contains(&id) {
                *proof.misses |= 1 << rank;
            } else {
                proof.hit_slots[rank] = all.slots[rank];
            }
        }
        // Drop its full Plan.next before retaining another candidate. The
        // compact exact miss/hit proof is rederived at final all-family preflight.
        drop(all);
        let gpu = staged
            .plan(source, split.gpu())
            .map_err(|_| err("native count1 canonical GPU subset"))?;
        let mut selected = Zeroizing::new([0usize; 80]);
        selected[..split.gpu().len()].copy_from_slice(split.gpu());
        let last = Last {
            pending: gpu,
            selected,
            len: Zeroizing::new(split.gpu().len()),
        };
        proof.validate_source(source, &staged, &last, false)?;
        Ok((proof, staged, last, split))
    }
    pub(super) fn native_map(split: &Split) -> Zeroizing<[u32; 20]> {
        let mut map = Zeroizing::new(*split.map());
        for rank in 0..10 {
            if map[rank * 2] == 0 {
                map[rank * 2 + 1] = rank as u32;
            }
        }
        map
    }
    pub(super) fn ids(&self) -> &[usize; 10] {
        &self.ids
    }
    pub(super) fn probabilities(&self) -> Zeroizing<[f32; 10]> {
        Zeroizing::new((*self.probabilities).map(f32::from_bits))
    }
    pub(super) fn misses(&self) -> usize {
        self.misses.count_ones() as usize
    }
    pub(super) fn cpu_count(&self) -> usize {
        *self.cpu_count
    }
    pub(super) fn cap(&self) -> usize {
        *self.cap
    }
    pub(super) fn complete(&mut self, partials: Option<&Partials>) -> Result<()> {
        if *self.joined {
            return Err(err("native count1 repeated CPU completion"));
        }
        match partials {
            None if *self.cpu_count == 0 => (),
            Some(p)
                if p.ranks() == *self.cpu_count
                    && p.hidden() == 2560
                    && p.full().len() == CPU_WORDS
                    && p.owned_bytes() <= HYBRID_HOST
                    && p.active().iter().all(|v| v.is_finite())
                    && p.full()[p.active().len()..]
                        .iter()
                        .all(|v| v.to_bits() == 0) => {}
            _ => return Err(err("native count1 actual joined CPU owner")),
        }
        *self.joined = true;
        Ok(())
    }
    pub(super) fn validate_source(
        &self,
        source: &crate::bank_slots::BankSlots,
        staged: &StagedSlots,
        last: &Last,
        require_join: bool,
    ) -> Result<()> {
        if ![0, 10].contains(&*self.cap)
            || require_join && !*self.joined
            || staged.accepted_waves() != 0
            || *last.len > 10
            || self.probabilities.iter().any(|&bits| {
                let p = f32::from_bits(bits);
                !p.is_finite() || p < 0.
            })
        {
            return Err(err("native count1 compact pending state"));
        }
        staged
            .validate_pending(source, &last.pending, &last.selected[..*last.len])
            .map_err(|_| err("native count1 unchanged GPU source"))?;
        let mut classifier = StagedSlots::begin(source)
            .map_err(|_| err("native count1 final original classifier"))?;
        let all = classifier
            .plan(source, &*self.ids)
            .map_err(|_| err("native count1 final all-rank classifier"))?;
        classifier
            .validate_pending(source, &all, &*self.ids)
            .map_err(|_| err("native count1 final all-rank proof"))?;
        let mut misses = Zeroizing::new([0usize; 10]);
        if all.uploads.len() > 10 {
            return Err(Error::InvalidBufferLength);
        }
        for (i, &(id, _)) in all.uploads.iter().enumerate() {
            misses[i] = id;
        }
        let split = Split::from_actual_misses(&self.ids, &misses[..all.uploads.len()], *self.cap)?;
        if split.cpu().len() != *self.cpu_count
            || split.gpu() != &last.selected[..*last.len]
            || last.selected[*last.len..].iter().any(|&x| x != 0)
            || last.pending.uploads.len() != all.uploads.len() - split.cpu().len()
        {
            return Err(err("native count1 actual miss ownership changed"));
        }
        let mut mask = Zeroizing::new(0u16);
        for (rank, &id) in self.ids.iter().enumerate() {
            if misses[..all.uploads.len()].contains(&id) {
                *mask |= 1 << rank;
            } else {
                let index = split
                    .gpu()
                    .iter()
                    .position(|&v| v == id)
                    .ok_or(err("native count1 resident hit moved to CPU"))?;
                if self.hit_slots[rank] != all.slots[rank]
                    || last.pending.slots[index] != all.slots[rank]
                {
                    return Err(err("native count1 protected hit slot changed"));
                }
            }
        }
        if *mask != *self.misses {
            return Err(err("native count1 miss classification changed"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bank_slots::BankSlots;
    #[test]
    fn compact_noncontiguous_misses_keep_original_gpu_ranks_and_hit_slots() {
        for cap in [0, 10] {
            let mut source = BankSlots::new(16, 512).unwrap();
            let prime = source
                .plan(&[0, 481, 2, 483, 4, 485, 6, 487, 8, 489])
                .unwrap();
            source.commit(prime).unwrap();
            let ids = std::array::from_fn(|r| r);
            let probabilities = [0.1; 10];
            let (mut p, staged, last, split) =
                Proof::prepare(&source, &ids, &probabilities, cap).unwrap();
            assert_eq!(p.misses(), 5);
            assert_eq!(
                split.cpu(),
                if cap == 10 { &[1, 3, 5, 7, 9][..] } else { &[] }
            );
            let map = Proof::native_map(&split);
            for rank in 0..10 {
                if map[rank * 2] == 0 {
                    assert_eq!(map[rank * 2 + 1], rank as u32);
                }
            }
            assert!(p.validate_source(&source, &staged, &last, true).is_err());
            if cap == 0 {
                p.complete(None).unwrap();
                p.validate_source(&source, &staged, &last, true).unwrap();
            } else {
                assert!(p.complete(None).is_err());
            }
            p.validate_source(&source, &staged, &last, false).unwrap();
            *p.misses ^= 1;
            assert!(p.validate_source(&source, &staged, &last, false).is_err());
        }
    }
    #[test]
    fn real_empty_gpu_plan_requires_all_cpu_and_rejects_stale_origin() {
        let mut source = BankSlots::new(16, 512).unwrap();
        let ids = std::array::from_fn(|r| r);
        let (p, staged, last, split) = Proof::prepare(&source, &ids, &[0.1; 10], 10).unwrap();
        assert!(split.gpu().is_empty());
        assert_eq!(*last.len, 0);
        p.validate_source(&source, &staged, &last, false).unwrap();
        let update = source.plan(&[481]).unwrap();
        source.commit(update).unwrap();
        assert!(p.validate_source(&source, &staged, &last, false).is_err());
        assert!(Proof::prepare(&source, &ids, &[0.1; 10], 1).is_err());
    }
}
