//! Request-transient inverse placement for exact cache restoration after lending.
//! Nothing here is persisted, logged or exported in an operational profile.
use crate::strata_global_plan::{EXPERTS, LAYERS};
use kuno_text::{Error, Result};
use zeroize::Zeroizing;
pub(crate) const HOST_BYTES: u64 = 128 * 1024; // current + pending49152B maps and metadata
#[derive(Clone)]
pub(crate) struct Placement(Zeroizing<Vec<u16>>);
fn reject() -> Error {
    Error::InvalidFormat("cache physical placement")
}
impl Placement {
    pub(crate) fn identity() -> Self {
        Self(Zeroizing::new(
            (0..LAYERS * EXPERTS)
                .map(|i| (i % EXPERTS) as u16)
                .collect(),
        ))
    }
    pub(crate) fn original_at(&self, layer: usize, cell: usize) -> Result<usize> {
        if layer >= LAYERS || cell >= EXPERTS {
            return Err(reject());
        }
        self.0
            .get(layer * EXPERTS + cell)
            .copied()
            .map(usize::from)
            .ok_or_else(reject)
    }
    /// Invert the actual physical-cell identities. Initial profile placement
    /// cannot reconstruct this after adaptive exchanges have moved weights.
    pub(crate) fn route_map(&self) -> Result<Zeroizing<Vec<u16>>> {
        if self.0.len() != LAYERS * EXPERTS {
            return Err(reject());
        }
        let mut inverse = Zeroizing::new(vec![0u16; LAYERS * EXPERTS]);
        for layer in 0..LAYERS {
            let mut seen = Zeroizing::new([false; EXPERTS]);
            for cell in 0..EXPERTS {
                let expert = self.original_at(layer, cell)?;
                if expert >= EXPERTS || seen[expert] {
                    return Err(reject());
                }
                seen[expert] = true;
                inverse[layer * EXPERTS + expert] = cell as u16;
            }
        }
        Ok(inverse)
    }
    fn swap_checked(
        &mut self,
        layer: usize,
        incoming: usize,
        outgoing: usize,
        gpu: usize,
        host: usize,
    ) -> Result<()> {
        if incoming >= EXPERTS
            || outgoing >= EXPERTS
            || incoming == outgoing
            || gpu == host
            || self.original_at(layer, gpu)? != outgoing
            || self.original_at(layer, host)? != incoming
        {
            return Err(reject());
        }
        self.0.swap(layer * EXPERTS + gpu, layer * EXPERTS + host);
        Ok(())
    }
    pub(crate) fn propose(&self, swaps: &[crate::strata_global_adapt::Swap]) -> Result<Self> {
        let mut next = self.clone();
        for s in swaps {
            next.swap_checked(s.layer, s.incoming, s.outgoing, s.gpu_cell, s.host_cell)?;
        }
        Ok(next)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn retained_route_map_is_the_inverse_of_actual_cells_and_rejects_corrupt_placement() {
        let mut p = Placement::identity();
        for (incoming, outgoing, gpu, host) in [(99, 7, 7, 99), (81, 99, 7, 81), (6, 81, 7, 6)] {
            p.swap_checked(3, incoming, outgoing, gpu, host).unwrap();
            let route = p.route_map().unwrap();
            for layer in 0..LAYERS {
                for expert in 0..EXPERTS {
                    assert_eq!(
                        p.original_at(layer, route[layer * EXPERTS + expert] as usize)
                            .unwrap(),
                        expert
                    );
                }
            }
        }
        p.0[0] = p.0[1];
        assert!(p.route_map().is_err());
        p.0[0] = EXPERTS as u16;
        assert!(p.route_map().is_err());
        p.0.pop();
        assert!(p.route_map().is_err());
    }
    #[test]
    fn physical_to_original_survives_repeated_swaps_without_resetting_to_profile() {
        let old = Placement::identity();
        let mut next = old.clone();
        next.swap_checked(3, 99, 7, 7, 99).unwrap();
        assert_eq!(old.original_at(3, 7).unwrap(), 7);
        assert_eq!(next.original_at(3, 7).unwrap(), 99);
        next.swap_checked(3, 81, 99, 7, 81).unwrap();
        assert_eq!(next.original_at(3, 7).unwrap(), 81);
        assert_eq!(next.original_at(3, 81).unwrap(), 99);
        assert_eq!(next.original_at(3, 99).unwrap(), 7);
        for layer in 0..LAYERS {
            let mut ids = (0..EXPERTS)
                .map(|c| next.original_at(layer, c).unwrap())
                .collect::<Vec<_>>();
            ids.sort_unstable();
            assert_eq!(ids, (0..EXPERTS).collect::<Vec<_>>());
        }
        assert!(next.swap_checked(3, 80, 7, 7, 80).is_err()); // stale outgoing identity
        assert!(next.swap_checked(3, 81, 81, 7, 7).is_err());
        assert!(next.original_at(LAYERS, 0).is_err());
        assert!(next.original_at(0, EXPERTS).is_err());
    }
}
