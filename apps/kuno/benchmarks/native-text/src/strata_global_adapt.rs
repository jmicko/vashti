//! MIT Strata generate.cpp:12040 original adaptive selection and publication.
//! Original physical cells stay fixed. Logical remap and bounded aggregate usage live with loaded weights only.
//! Request arenas are still erased in full; no telemetry is serialized/logged.
use crate::strata_global_plan::{EXPERTS, LAYERS, Plan};
use kuno_text::{Error, Result};
use zeroize::{Zeroize, Zeroizing};
pub(crate) const EVERY: u64 = 4;
pub(crate) const MAX_SWAPS: usize = 96;
pub(crate) const HOST_BYTES: usize = 2 * 1024 * 1024;
pub(crate) const ROUTE_MAP_BYTES: usize = 131072;
pub(crate) const USAGE_BYTES: usize = 131072;
/// Original full private owner, including its unused zero padding. The caller
/// transfers the wiping allocation directly into the actual tracked upload.
pub(crate) fn encode_route_map(mapping: &[u16]) -> Result<Zeroizing<Vec<u8>>> {
    validate_mapping(mapping)?;
    let mut bytes = Zeroizing::new(vec![0u8; ROUTE_MAP_BYTES]);
    for (word, &cell) in bytes.chunks_exact_mut(4).zip(mapping) {
        word.copy_from_slice(&u32::from(cell).to_le_bytes());
    }
    Ok(bytes)
}

fn validate_usage(usage: &[f32]) -> Result<()> {
    if usage.len() != LAYERS * EXPERTS || usage.iter().any(|u| !u.is_finite() || *u < 0.) {
        return Err(reject());
    }
    Ok(())
}
pub(crate) fn encode_usage(usage: &[f32]) -> Result<Zeroizing<Vec<u8>>> {
    validate_usage(usage)?;
    let mut bytes = Zeroizing::new(vec![0u8; USAGE_BYTES]);
    for (word, &score) in bytes.chunks_exact_mut(4).zip(usage) {
        word.copy_from_slice(&score.to_bits().to_le_bytes());
    }
    Ok(bytes)
}
const _: () = assert!(LAYERS * EXPERTS * 4 <= USAGE_BYTES && 8 * USAGE_BYTES < HOST_BYTES);
fn reject() -> Error {
    Error::InvalidFormat("Strata global adaptive cache")
}
pub(crate) struct Swap {
    pub layer: usize,
    pub incoming: usize,
    pub outgoing: usize,
    pub gpu_cell: usize,
    pub host_cell: usize,
    gain: f32,
}
impl Zeroize for Swap {
    fn zeroize(&mut self) {
        self.layer.zeroize();
        self.incoming.zeroize();
        self.outgoing.zeroize();
        self.gpu_cell.zeroize();
        self.host_cell.zeroize();
        self.gain.zeroize();
    }
}
pub(crate) struct Proposal {
    pub swaps: Zeroizing<Vec<Swap>>,
    pub mapping: Zeroizing<Vec<u16>>,
    pub usage: Zeroizing<Vec<f32>>,
    round: u64,
}
pub(crate) struct Cache {
    mapping: Zeroizing<Vec<u16>>,
    resident: Vec<bool>, // static PUBLIC initial physical-cell placement
    rounds: u64,
    usage: Zeroizing<Vec<f32>>, // bounded model aggregate; never token/seed/history payload
}
impl Cache {
    pub(crate) fn new(plan: &Plan) -> Result<Self> {
        if plan.layers.len() != LAYERS {
            return Err(reject());
        }
        let mut resident = vec![false; LAYERS * EXPERTS];
        for (layer, l) in plan.layers.iter().enumerate() {
            for &expert in l.resident_ids() {
                resident[layer * EXPERTS + expert as usize] = true;
            }
        }
        let mapping = Zeroizing::new(
            (0..LAYERS * EXPERTS)
                .map(|i| (i % EXPERTS) as u16)
                .collect(),
        );
        Ok(Self {
            mapping,
            resident,
            rounds: 0,
            usage: Zeroizing::new(vec![0.; LAYERS * EXPERTS]),
        })
    }
    pub(crate) fn check_retained_route_map(&self, mapping: &[u16]) -> Result<()> {
        validate_mapping(mapping)?;
        if self.mapping.as_slice() != mapping {
            return Err(reject());
        }
        Ok(())
    }
    pub(crate) fn retained_usage(&self) -> &[f32] {
        &self.usage
    }
    pub(crate) fn retain_usage(&mut self, usage: &[f32]) -> Result<()> {
        validate_usage(usage)?;
        if self.usage.len() != LAYERS * EXPERTS {
            return Err(reject());
        }
        self.usage.copy_from_slice(usage);
        Ok(())
    }
    pub(crate) fn tick(&mut self) -> Result<bool> {
        self.rounds = self
            .rounds
            .checked_add(1)
            .ok_or(Error::ArithmeticOverflow)?;
        Ok(self.rounds.is_multiple_of(EVERY))
    }
    pub(crate) fn propose(&self, usage: &[f32]) -> Result<Proposal> {
        if !self.rounds.is_multiple_of(EVERY)
            || self.rounds == 0
            || usage.len() != LAYERS * EXPERTS
            || usage.iter().any(|u| !u.is_finite() || *u < 0.)
        {
            return Err(reject());
        }
        let mut swaps = Zeroizing::new(Vec::new());
        for layer in 0..LAYERS {
            let start = layer * EXPERTS;
            let mut candidates = Zeroizing::new(Vec::<usize>::new());
            let mut victims = Zeroizing::new(Vec::<usize>::new());
            for expert in 0..EXPERTS {
                let cell = usize::from(self.mapping[start + expert]);
                if self.resident[start + cell] {
                    victims.push(expert);
                } else if usage[start + expert] >= 2. {
                    candidates.push(expert);
                }
            }
            // Upstream compares scores only; ID breaks equal scores
            // deterministically without changing the gain/eligibility rule.
            candidates.sort_by(|&a, &b| {
                usage[start + b]
                    .total_cmp(&usage[start + a])
                    .then(a.cmp(&b))
            });
            victims.sort_by(|&a, &b| {
                usage[start + a]
                    .total_cmp(&usage[start + b])
                    .then(a.cmp(&b))
            });
            for (&incoming, &outgoing) in candidates.iter().zip(victims.iter()) {
                if usage[start + incoming] < usage[start + outgoing] + 1.5 {
                    break;
                }
                swaps.push(Swap {
                    layer,
                    incoming,
                    outgoing,
                    gpu_cell: self.mapping[start + outgoing] as usize,
                    host_cell: self.mapping[start + incoming] as usize,
                    gain: usage[start + incoming] - usage[start + outgoing],
                });
            }
            retain_best(&mut swaps);
        }
        let mut mapping = Zeroizing::new(self.mapping.to_vec());
        for swap in swaps.iter() {
            let first = swap.layer * EXPERTS;
            mapping.swap(first + swap.incoming, first + swap.outgoing);
        }
        validate_mapping(&mapping)?;
        let usage = Zeroizing::new(usage.iter().map(|u| u * 0.7f32).collect());
        Ok(Proposal {
            swaps,
            mapping,
            usage,
            round: self.rounds,
        })
    }
    /// Only after all three original-role exchanges AND private GPU remap/usage
    /// publication have completed. No partially copied pair is committed.
    pub(crate) fn commit(&mut self, proposal: Proposal) -> Result<()> {
        if proposal.round != self.rounds {
            return Err(reject());
        }
        for swap in proposal.swaps.iter() {
            let first = swap.layer * EXPERTS;
            if self.mapping[first + swap.outgoing] as usize != swap.gpu_cell
                || self.mapping[first + swap.incoming] as usize != swap.host_cell
                || !self.resident[first + swap.gpu_cell]
                || self.resident[first + swap.host_cell]
            {
                return Err(reject());
            }
        }
        validate_mapping(&proposal.mapping)?;
        validate_usage(&proposal.usage)?;
        self.mapping.copy_from_slice(&proposal.mapping);
        self.usage.copy_from_slice(&proposal.usage);
        Ok(())
    }
}
fn validate_mapping(mapping: &[u16]) -> Result<()> {
    if mapping.len() != LAYERS * EXPERTS {
        return Err(reject());
    }
    for row in mapping.chunks_exact(EXPERTS) {
        let mut seen = Zeroizing::new([false; EXPERTS]);
        for &cell in row {
            let at = usize::from(cell);
            if at >= EXPERTS || seen[at] {
                return Err(reject());
            }
            seen[at] = true;
        }
    }
    Ok(())
}
impl Drop for Cache {
    fn drop(&mut self) {
        self.rounds.zeroize();
    }
}

impl Drop for Proposal {
    fn drop(&mut self) {
        self.round.zeroize();
    }
}

fn retain_best(swaps: &mut Zeroizing<Vec<Swap>>) {
    swaps.sort_by(|a, b| {
        b.gain
            .total_cmp(&a.gain)
            .then(a.layer.cmp(&b.layer))
            .then(a.incoming.cmp(&b.incoming))
    });
    for discarded in swaps.iter_mut().skip(MAX_SWAPS) {
        discarded.zeroize();
    }
    swaps.truncate(MAX_SWAPS);
}

#[cfg(test)]
mod tests {
    use super::*;
    fn cache(resident: Vec<bool>) -> Cache {
        Cache {
            mapping: Zeroizing::new(
                (0..LAYERS * EXPERTS)
                    .map(|i| (i % EXPERTS) as u16)
                    .collect(),
            ),
            resident,
            rounds: 4,
            usage: Zeroizing::new(vec![0.; LAYERS * EXPERTS]),
        }
    }
    #[test]
    fn retained_route_packet_tracks_real_exchanges_instead_of_resetting_to_identity() {
        let resident = (0..LAYERS * EXPERTS)
            .map(|i| (i % EXPERTS) % 3 == 0)
            .collect();
        let mut c = cache(resident);
        let mut physical = crate::strata_global_sources::Placement::identity();
        let cold = physical.route_map().unwrap();
        c.check_retained_route_map(&cold).unwrap();
        for round in 0..12 {
            let usage = (0..LAYERS * EXPERTS)
                .map(|i| ((i * 17 + round * 71) % 131) as f32)
                .collect::<Vec<_>>();
            let p = c.propose(&usage).unwrap();
            assert!(!p.swaps.is_empty());
            physical = physical.propose(&p.swaps).unwrap();
            c.commit(p).unwrap();
            let restored = physical.route_map().unwrap();
            c.check_retained_route_map(&restored).unwrap();
            assert!(c.check_retained_route_map(&cold).is_err());
            let bytes = encode_route_map(&restored).unwrap();
            assert_eq!(bytes.len(), ROUTE_MAP_BYTES);
            for (i, word) in bytes.chunks_exact(4).enumerate() {
                let cell = u32::from_le_bytes(word.try_into().unwrap());
                if i < LAYERS * EXPERTS {
                    assert_eq!(
                        physical.original_at(i / EXPERTS, cell as usize).unwrap(),
                        i % EXPERTS
                    );
                } else {
                    assert_eq!(cell, 0);
                }
            }
            for _ in 0..EVERY {
                c.tick().unwrap();
            }
        }
        let mut invalid = cold.to_vec();
        invalid[0] = invalid[1];
        assert!(encode_route_map(&invalid).is_err());
        invalid[0] = EXPERTS as u16;
        assert!(encode_route_map(&invalid).is_err());
        assert!(encode_route_map(&invalid[..invalid.len() - 1]).is_err());
    }
    #[test]
    fn reference_gain_rule_is_global_96_with_unequal_and_nonprefix_residency() {
        let mut resident = vec![false; LAYERS * EXPERTS];
        // GPU cells are actual initial expert IDs, not a uniform first240.
        for cell in 300..428 {
            resident[cell] = true;
        }
        for cell in 200..328 {
            resident[EXPERTS + cell] = true;
        }
        for cell in 0..EXPERTS {
            resident[3 * EXPERTS + cell] = true;
        } // fully GPU
        let c = cache(resident);
        let mut usage = vec![0f32; LAYERS * EXPERTS];
        for expert in 0..128 {
            usage[expert] = 3.;
            usage[EXPERTS + expert] = 10.;
        }
        usage[2 * EXPERTS] = 100.; // fully HOST: no GPU slot can be evicted
        usage[3 * EXPERTS + 100] = 100.; // fully GPU: no missing candidate
        let p = c.propose(&usage).unwrap();
        assert_eq!(p.swaps.len(), 96);
        assert!(
            p.swaps
                .iter()
                .all(|s| s.layer == 1 && s.gpu_cell >= 200 && s.host_cell < 128)
        );
        assert_eq!(p.usage[EXPERTS], 7.);
        assert_eq!(p.mapping[EXPERTS], 200);
        assert_eq!(p.mapping[EXPERTS + 200], 0);
    }
    #[test]
    fn repeated_two_way_physical_swaps_preserve_every_logical_expert() {
        let resident = (0..LAYERS * EXPERTS)
            .map(|i| (i % EXPERTS) % 3 == 0)
            .collect();
        let mut c = cache(resident);
        let mut physical = (0..LAYERS * EXPERTS)
            .map(|i| (i % EXPERTS) as u32)
            .collect::<Vec<_>>();
        for round in 0..12 {
            let usage = (0..LAYERS * EXPERTS)
                .map(|i| ((i * 17 + round * 71) % 131) as f32)
                .collect::<Vec<_>>();
            let p = c.propose(&usage).unwrap();
            assert!(!p.swaps.is_empty());
            for swap in p.swaps.iter() {
                let first = swap.layer * EXPERTS;
                assert!(c.resident[first + swap.gpu_cell]);
                assert!(!c.resident[first + swap.host_cell]);
                physical.swap(first + swap.gpu_cell, first + swap.host_cell);
            }
            c.commit(p).unwrap();
            for layer in 0..LAYERS {
                for expert in 0..EXPERTS {
                    assert_eq!(
                        physical[layer * EXPERTS + c.mapping[layer * EXPERTS + expert] as usize],
                        expert as u32
                    );
                }
            }
            for _ in 0..4 {
                c.tick().unwrap();
            }
        }
    }
    #[test]
    fn exact_eligibility_gain_boundary_and_stale_proposal_are_enforced() {
        let mut resident = vec![false; LAYERS * EXPERTS];
        resident[400] = true;
        let mut c = cache(resident);
        let mut usage = vec![0f32; LAYERS * EXPERTS];
        usage[400] = 0.5;
        usage[0] = 2.;
        let p = c.propose(&usage).unwrap();
        assert_eq!(p.swaps.len(), 1);
        usage[0] = 1.999;
        assert!(c.propose(&usage).unwrap().swaps.is_empty());
        c.tick().unwrap();
        assert!(c.commit(p).is_err());
        usage[0] = f32::NAN;
        c.rounds = 8;
        assert!(c.propose(&usage).is_err());
    }
}

#[cfg(test)]
mod retained_usage_tests {
    use super::*;
    fn model() -> Cache {
        Cache {
            mapping: Zeroizing::new(
                (0..LAYERS * EXPERTS)
                    .map(|i| (i % EXPERTS) as u16)
                    .collect(),
            ),
            resident: (0..LAYERS * EXPERTS)
                .map(|i| (i % EXPERTS).is_multiple_of(3))
                .collect(),
            rounds: 0,
            usage: Zeroizing::new(vec![0.; LAYERS * EXPERTS]),
        }
    }
    fn simulate(boundaries: &[usize]) -> (Vec<u16>, Vec<u32>, u64) {
        let mut cache = model();
        let mut device = vec![0f32; LAYERS * EXPERTS];
        for window in 1..=12 {
            for layer in 0..LAYERS {
                for pick in 0..8 {
                    device[layer * EXPERTS + (window * 31 + pick * 17) % EXPERTS] += 1.;
                }
            }
            if cache.tick().unwrap() {
                let p = cache.propose(&device).unwrap();
                device.copy_from_slice(&p.usage);
                cache.commit(p).unwrap();
            }
            if boundaries.contains(&window) {
                // Snapshot AFTER complete execution, then simulate a FULL
                // request erase; only model aggregate bytes are rebound.
                cache.retain_usage(&device).unwrap();
                device.fill(0.); // GPU owners retain their fixed extent after erase.
                assert!(device.iter().all(|v| v.to_bits() == 0));
                let bytes = encode_usage(cache.retained_usage()).unwrap();
                assert_eq!(bytes.len(), USAGE_BYTES);
                assert!(bytes[LAYERS * EXPERTS * 4..].iter().all(|&b| b == 0));
                for (value, word) in device.iter_mut().zip(bytes.chunks_exact(4)) {
                    *value = f32::from_bits(u32::from_le_bytes(word.try_into().unwrap()));
                }
            }
        }
        (
            cache.mapping.to_vec(),
            device.iter().map(|v| v.to_bits()).collect(),
            cache.rounds,
        )
    }
    #[test]
    fn full_request_erase_and_rebind_preserves_the_same_original_adaptive_sequence() {
        let uninterrupted = simulate(&[]);
        for ends in [
            &[1, 3, 7, 12][..],
            &[4, 8, 12][..],
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12][..],
        ] {
            assert_eq!(simulate(ends), uninterrupted);
        }
        assert_eq!(uninterrupted.2, 12);
    }
    #[test]
    fn retained_usage_rejects_bad_scores_before_mutation_and_encodes_complete_padding() {
        let mut cache = model();
        let mut valid = vec![1.25; LAYERS * EXPERTS];
        valid[0] = -0.;
        cache.retain_usage(&valid).unwrap();
        let original = cache
            .retained_usage()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>();
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.1] {
            let mut bad = valid.clone();
            bad[7] = invalid;
            assert!(cache.retain_usage(&bad).is_err());
            assert!(encode_usage(&bad).is_err());
            assert_eq!(
                cache
                    .retained_usage()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                original
            );
        }
        assert!(cache.retain_usage(&valid[1..]).is_err());
        assert!(encode_usage(&valid[1..]).is_err());
        let bytes = encode_usage(&valid).unwrap();
        assert_eq!(bytes.len(), USAGE_BYTES);
        for (expected, word) in original.iter().zip(bytes.chunks_exact(4)) {
            assert_eq!(*expected, u32::from_le_bytes(word.try_into().unwrap()));
        }
        assert!(bytes[original.len() * 4..].iter().all(|&b| b == 0));
    }
}
