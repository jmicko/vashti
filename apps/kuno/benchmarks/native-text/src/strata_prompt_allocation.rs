//! Native Intel prompt owner policy. Model/cache pricing includes every root
//! before any payload allocation. The large-row consumer port is separate.
use crate::{
    private_arena::{BufferId, PrivateArena},
    resident_util as u,
};
use kuno_text::{Error, Result};
pub(crate) const DEFAULT_ROWS: usize = 2048;
pub(crate) const ROOT_BYTES: u64 = 512 * 1024 * 1024;
pub(crate) const MAIN_BYTES: u64 = 3 * ROOT_BYTES;
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
            return Err(Error::InvalidBufferLength);
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
    pub(crate) fn allocate(self, arena: &mut PrivateArena) -> Result<[BufferId; 3]> {
        if self.mode != Mode::Owned {
            // An unported loan is an explicit incomplete path, never an
            // unpriced owned fallback or an invented cache lease.
            return Err(Error::InvalidFormat(
                "native prompt cache lending not connected",
            ));
        }
        if self.rows != DEFAULT_ROWS {
            return Err(Error::InvalidBufferLength);
        }
        let words = ROOT_BYTES as usize / 4;
        Ok([
            arena.allocate_zeroed(words).map_err(u::fault)?,
            arena.allocate_zeroed(words).map_err(u::fault)?,
            arena.allocate_zeroed(words).map_err(u::fault)?,
        ])
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
        assert_eq!(Plan::new(32769).unwrap().mode, Mode::CacheLoan);
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
