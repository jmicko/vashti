//! Native Intel prompt owner policy. Model/cache pricing includes every root
//! before any payload allocation; all large-row consumers use these same roots.
use crate::{
    private_arena::{BufferId, PrivateArena},
    resident_util as u,
};
use kuno_text::Result;
pub(crate) const DEFAULT_ROWS: usize = kuno_text::ops::ple_transition::MAX_PROMPT_BATCH;
pub(crate) const ROOT_BYTES: u64 = 1024 * 1024 * 1024;
/// The asynchronous raw transfer ring is not part of the phase overlay.
pub(crate) const RAW_RING_BYTES: u64 = 32 * 1024 * 1024;
pub(crate) const ROOT_SIZES: [u64; 3] = super::layout::root_sizes();
pub(crate) const MAIN_BYTES: u64 = ROOT_SIZES[0] + ROOT_SIZES[1] + ROOT_SIZES[2];
/// The selected owned prompt path alternates two genuine whole raw roots.
/// Keep the original three-root quote independent for cache-loan accounting.
pub(crate) const SECONDARY_RAW_BYTES: u64 = if cfg!(feature = "strata-expert-cohorts") {
    RAW_RING_BYTES
} else {
    0
};
/// Structural whole-root envelope, shared by private Scope, raw ring and source
/// attachment. Physical ownership still needs the arena/cache owner's backend
/// attestation; a Handle size alone is not a lending or erasure capability.
pub(crate) fn whole_root_bytes(root: &cubecl_runtime::server::Handle) -> Result<u64> {
    let bytes = root.size_in_used();
    if !(RAW_RING_BYTES..=ROOT_BYTES).contains(&bytes)
        || !bytes.is_multiple_of(256)
        || root.size() != bytes
        || root.offset_start.unwrap_or(0) != 0
        || root.offset_end.unwrap_or(0) != 0
    {
        return Err(crate::strata_prompt::invalid_buffer_length());
    }
    Ok(bytes)
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    Owned,
    CacheLoan,
}
#[derive(Clone, Copy)]
pub(crate) struct Plan {
    mode: Mode,
    rows: usize,
}
impl Plan {
    pub(crate) fn new(context: usize) -> Result<Self> {
        if context == 0 {
            return Err(crate::strata_prompt::invalid_buffer_length());
        }
        Ok(Self {
            mode: if context <= 32768 {
                Mode::Owned
            } else {
                Mode::CacheLoan
            },
            rows: DEFAULT_ROWS,
        })
    }
    pub(crate) fn requests(self) -> Vec<u64> {
        if self.mode == Mode::Owned {
            ROOT_SIZES
                .into_iter()
                .chain((self.secondary_raw_bytes() != 0).then_some(SECONDARY_RAW_BYTES))
                .collect()
        } else {
            Vec::new()
        }
    }
    pub(crate) fn secondary_raw_bytes(self) -> u64 {
        if self.mode == Mode::Owned {
            SECONDARY_RAW_BYTES
        } else {
            0
        }
    }
    pub(crate) fn allocate_secondary_raw_bank(
        self,
        arena: &mut PrivateArena,
    ) -> Result<Option<BufferId>> {
        if self.secondary_raw_bytes() == 0 {
            return Ok(None);
        }
        if self.rows != DEFAULT_ROWS {
            return Err(crate::strata_prompt::invalid_buffer_length());
        }
        Ok(Some(
            arena
                .allocate_zeroed(RAW_RING_BYTES as usize / 4)
                .map_err(u::fault)?,
        ))
    }
    pub(crate) fn allocate(self, arena: &mut PrivateArena) -> Result<Option<[BufferId; 3]>> {
        if self.mode == Mode::CacheLoan {
            return Ok(None);
        }
        if self.rows != DEFAULT_ROWS {
            return Err(crate::strata_prompt::invalid_buffer_length());
        }
        Ok(Some([
            arena
                .allocate_zeroed(ROOT_SIZES[0] as usize / 4)
                .map_err(u::fault)?,
            arena
                .allocate_zeroed(ROOT_SIZES[1] as usize / 4)
                .map_err(u::fault)?,
            arena
                .allocate_zeroed(ROOT_SIZES[2] as usize / 4)
                .map_err(u::fault)?,
        ]))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_policy_prices_default_owned_roots_and_distinguishes_real_loans() {
        for context in [1, 13, 4096, 32768] {
            let p = Plan::new(context).unwrap();
            assert_eq!(p.mode, Mode::Owned);
            assert_eq!(p.rows, DEFAULT_ROWS);
        }
        let loan = Plan::new(32769).unwrap();
        assert_eq!(loan.mode, Mode::CacheLoan);
        assert!(loan.requests().is_empty());
        assert_eq!(
            crate::strata_global_moe::Workspace::main_bytes_for_context(32769).unwrap()
                + MAIN_BYTES,
            crate::strata_global_moe::MAIN_BYTES - SECONDARY_RAW_BYTES
        );
        assert_eq!(
            crate::strata_global_moe::Workspace::main_bytes_for_context(32769).unwrap()
                + MAIN_BYTES
                + SECONDARY_RAW_BYTES,
            crate::strata_global_moe::MAIN_BYTES
        );
        assert_eq!(
            crate::strata_global_moe::Workspace::requests_for_context(32769)
                .unwrap()
                .len()
                + 3
                + usize::from(SECONDARY_RAW_BYTES != 0),
            crate::strata_global_moe::Workspace::requests().len()
        );
        assert!(Plan::new(0).is_err());
        assert_eq!(
            crate::strata_global_moe::Workspace::requests()
                .iter()
                .sum::<u64>(),
            crate::strata_global_moe::MAIN_BYTES
        );
        let layout = crate::private_arena::BufferLayout::new(ROOT_SIZES[0] as usize / 4);
        if cfg!(feature = "strata-intel-core") {
            assert_eq!(layout.unwrap().owned_bytes as u64, ROOT_SIZES[0]);
            let actual = ROOT_SIZES.map(|bytes| {
                crate::private_arena::BufferLayout::new(bytes as usize / 4)
                    .unwrap()
                    .owned_bytes as u64
            });
            assert_eq!(actual, ROOT_SIZES);
            assert_eq!(actual.iter().sum::<u64>(), MAIN_BYTES);
            assert_eq!(3 * ROOT_BYTES - MAIN_BYTES, 1642 * 1024 * 1024);
        } else {
            assert!(layout.is_err());
        }
    }
    #[test]
    fn secondary_raw_owner_is_quoted_independently_only_for_owned_contexts() {
        for context in [1, 4096, 32768] {
            let plan = Plan::new(context).unwrap();
            let requests = plan.requests();
            assert_eq!(&requests[..3], &ROOT_SIZES);
            assert_eq!(requests[..3].iter().sum::<u64>(), MAIN_BYTES);
            assert_eq!(plan.secondary_raw_bytes(), SECONDARY_RAW_BYTES);
            assert_eq!(requests[3..].iter().sum::<u64>(), SECONDARY_RAW_BYTES);
            assert_eq!(requests.len(), 3 + usize::from(SECONDARY_RAW_BYTES != 0));
        }
        for context in [32769, 65536, usize::MAX] {
            let plan = Plan::new(context).unwrap();
            assert_eq!(plan.secondary_raw_bytes(), 0);
            assert!(plan.requests().is_empty());
        }
        if cfg!(feature = "strata-intel-core") {
            let raw = crate::private_arena::BufferLayout::new(RAW_RING_BYTES as usize / 4).unwrap();
            assert_eq!(raw.owned_bytes as u64, RAW_RING_BYTES);
        }
        assert_eq!(
            SECONDARY_RAW_BYTES,
            if cfg!(feature = "strata-expert-cohorts") {
                RAW_RING_BYTES
            } else {
                0
            }
        );
    }
}
