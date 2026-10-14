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
