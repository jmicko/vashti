//! Native Intel prompt owner policy. Model/cache pricing includes every root
//! before any payload allocation. The large-row consumer port is separate.
use crate::{
    private_arena::{BufferId, PrivateArena},
    resident_util as u,
};
use kuno_text::Result;
pub(crate) const DEFAULT_ROWS: usize = 2048;
pub(crate) const ROOT_BYTES: u64 = 512 * 1024 * 1024;
pub(crate) const MAIN_BYTES: u64 = 3 * ROOT_BYTES;
/// Structural whole-root envelope, shared by private Scope, raw ring and source
/// attachment. Physical ownership still needs the arena/cache owner's backend
/// attestation; a Handle size alone is not a lending or erasure capability.
pub(crate) fn whole_root_bytes(root: &cubecl_runtime::server::Handle) -> Result<u64> {
    let bytes = root.size_in_used();
    if !(64 * 1024 * 1024..=ROOT_BYTES).contains(&bytes)
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
            vec![ROOT_BYTES; 3]
        } else {
            Vec::new()
        }
    }
    pub(crate) fn allocate(self, arena: &mut PrivateArena) -> Result<Option<[BufferId; 3]>> {
        if self.mode == Mode::CacheLoan {
            return Ok(None);
        }
        if self.rows != DEFAULT_ROWS {
            return Err(crate::strata_prompt::invalid_buffer_length());
        }
        let words = ROOT_BYTES as usize / 4;
        Ok(Some([
            arena.allocate_zeroed(words).map_err(u::fault)?,
            arena.allocate_zeroed(words).map_err(u::fault)?,
            arena.allocate_zeroed(words).map_err(u::fault)?,
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
            assert_eq!(p.rows, 2048);
        }
        let loan = Plan::new(32769).unwrap();
        assert_eq!(loan.mode, Mode::CacheLoan);
        assert!(loan.requests().is_empty());
        assert_eq!(
            crate::strata_global_moe::Workspace::main_bytes_for_context(32769).unwrap()
                + MAIN_BYTES,
            crate::strata_global_moe::MAIN_BYTES
        );
        assert_eq!(
            crate::strata_global_moe::Workspace::requests_for_context(32769)
                .unwrap()
                .len()
                + 3,
            crate::strata_global_moe::Workspace::requests().len()
        );
        assert!(Plan::new(0).is_err());
        assert_eq!(
            crate::strata_global_moe::Workspace::requests()
                .iter()
                .sum::<u64>(),
            crate::strata_global_moe::MAIN_BYTES
        );
        let layout = crate::private_arena::BufferLayout::new(ROOT_BYTES as usize / 4);
        if cfg!(feature = "strata-intel-core") {
            assert_eq!(layout.unwrap().owned_bytes as u64 * 3, MAIN_BYTES);
        } else {
            assert!(layout.is_err());
        }
    }
}
