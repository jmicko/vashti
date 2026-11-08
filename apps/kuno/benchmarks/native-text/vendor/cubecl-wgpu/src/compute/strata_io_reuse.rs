//! Bounded original-owner reuse inventory. This contains NO fence authority;
//! Graph must join original consumers and exclude original retained references.
pub const HOST_BYTES: usize = 4 * 1024 * 1024;
// Existing I/O metadata limit <=4MiB and actual Tracked<WgpuBuffer> >=32 bytes.
pub(super) const MAX_OWNERS: usize = 131072;

pub(super) struct Cohort {
    request: u64,
    epoch: u64,
    count: usize,
    original: Vec<(u64, usize)>,
    allowed: Vec<u64>,
    sealed: bool,
}
impl Cohort {
    pub fn new(request: u64, epoch: u64, count: usize) -> Result<Self, &'static str> {
        if request == 0 || epoch == 0 || count > MAX_OWNERS {
            return Err("native I/O original cohort extent");
        }
        let mut original = Vec::new();
        original
            .try_reserve_exact(count)
            .map_err(|_| "native I/O inventory allocation")?;
        let mut allowed = Vec::new();
        allowed
            .try_reserve_exact(count.div_ceil(64))
            .map_err(|_| "native I/O bitmap allocation")?;
        allowed.resize(count.div_ceil(64), 0);
        if original.capacity() * size_of::<(u64, usize)>() + allowed.capacity() * 8 > HOST_BYTES {
            return Err("native I/O cohort metadata capacity");
        }
        Ok(Self {
            request,
            epoch,
            count,
            original,
            allowed,
            sealed: false,
        })
    }
    /// Caller supplies each actual full dedicated I/O buffer in original slot
    /// order. No subview/main-arena alias may stand in for a tracked owner.
    pub fn owner(&mut self, buffer: u64, intrinsically_pinned: bool) -> Result<(), &'static str> {
        let index = self.original.len();
        if self.sealed || buffer == 0 || index >= self.count {
            return Err("native I/O original owner order/extent");
        }
        self.original.push((buffer, index));
        if !intrinsically_pinned {
            self.allowed[index / 64] |= 1u64 << (index % 64);
        }
        Ok(())
    }
    pub fn seal_inventory(&mut self) -> Result<(), &'static str> {
        if self.sealed || self.original.len() != self.count {
            return Err("native I/O original inventory incomplete/repeated");
        }
        self.original.sort_unstable_by_key(|p| p.0);
        if self.original.windows(2).any(|p| p[0].0 == p[1].0) {
            return Err("native I/O duplicate original allocation");
        }
        self.sealed = true;
        Ok(())
    }
    /// Host reuse overwrites the COMPLETE dedicated I/O buffer. Every genuine
    /// recorded range on it conflicts, even a readonly subview. Original Plan
    /// inventory already includes descriptor/physical/indirect/copy accesses.
    pub fn exclude_buffer(&mut self, buffer: u64) -> Result<(), &'static str> {
        if !self.sealed || buffer == 0 {
            return Err("native I/O exclusion inventory absent");
        }
        if let Ok(at) = self.original.binary_search_by_key(&buffer, |p| p.0) {
            let index = self.original[at].1;
            self.allowed[index / 64] &= !(1u64 << (index % 64));
        }
        Ok(())
    }
    pub fn validate(&self, request: u64, epoch: u64, count: usize) -> Result<(), &'static str> {
        if !self.sealed || (self.request, self.epoch, self.count) != (request, epoch, count) {
            return Err("native I/O original cohort changed");
        }
        Ok(())
    }
    pub fn admitted(&self, index: usize) -> bool {
        self.sealed && index < self.count && self.allowed[index / 64] & (1u64 << (index % 64)) != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_slots_survive_sorted_exclusions_and_cannot_change_request_epoch_or_extent() {
        let mut c = Cohort::new(9, 7, 4).unwrap();
        assert!(c.validate(9, 7, 4).is_err());
        assert!(c.exclude_buffer(3).is_err());
        for (buffer, pin) in [(71, false), (3, true), (999, false), (8, false)] {
            c.owner(buffer, pin).unwrap();
        }
        c.seal_inventory().unwrap();
        assert!(!c.admitted(1));
        c.exclude_buffer(999).unwrap();
        c.exclude_buffer(999).unwrap();
        c.exclude_buffer(1000).unwrap(); // Not an original I/O buffer.
        assert_eq!(
            (0..4).map(|i| c.admitted(i)).collect::<Vec<_>>(),
            [true, false, false, true]
        );
        c.validate(9, 7, 4).unwrap();
        for (r, e, n) in [(8, 7, 4), (9, 8, 4), (9, 7, 3), (0, 7, 4)] {
            assert!(c.validate(r, e, n).is_err());
        }
        assert!(!c.admitted(usize::MAX));
        assert!(c.owner(12, false).is_err());
        assert!(c.seal_inventory().is_err());
    }
    #[test]
    fn actual_recorded_read_subviews_exclude_the_whole_original_host_overwrite() {
        use super::super::strata_schedule::{Access, NodeAccess, Planner};
        let mut planner = Planner::new(1, false).unwrap();
        planner
            .push(&NodeAccess {
                shared: false,
                copy: false,
                release_after: false,
                accesses: vec![
                    Access {
                        buffer: 71,
                        first: 4,
                        bytes: 4,
                        write: false,
                    },
                    Access {
                        buffer: 71,
                        first: 16,
                        bytes: 4,
                        write: false,
                    },
                    Access {
                        buffer: 99,
                        first: 0,
                        bytes: 8,
                        write: true,
                    },
                ],
            })
            .unwrap();
        let original = planner.finish().unwrap();
        assert_eq!(original.buffer_ids().collect::<Vec<_>>(), [71, 99]);
        let mut cohort = Cohort::new(9, 1, 2).unwrap();
        cohort.owner(71, false).unwrap();
        cohort.owner(72, false).unwrap();
        cohort.seal_inventory().unwrap();
        // Graph uses the SAME original plan for a future named replay or live
        // flight. No physical fence is simulated by this host inventory check.
        for buffer in original.buffer_ids() {
            cohort.exclude_buffer(buffer).unwrap();
        }
        assert!(!cohort.admitted(0));
        assert!(cohort.admitted(1));
    }
    #[test]
    fn duplicate_partial_invalid_and_maximal_inventories_remain_bounded() {
        assert!(Cohort::new(0, 1, 0).is_err());
        assert!(Cohort::new(1, 0, 0).is_err());
        assert!(Cohort::new(1, 1, MAX_OWNERS + 1).is_err());
        let mut c = Cohort::new(1, 1, 2).unwrap();
        assert!(c.owner(0, false).is_err());
        c.owner(3, false).unwrap();
        assert!(c.seal_inventory().is_err());
        c.owner(3, false).unwrap();
        assert!(c.seal_inventory().is_err());
        let mut c = Cohort::new(1, 1, MAX_OWNERS).unwrap();
        for i in 0..MAX_OWNERS {
            c.owner((MAX_OWNERS - i) as u64, i % 2 == 0).unwrap();
        }
        c.seal_inventory().unwrap();
        for i in 0..MAX_OWNERS {
            assert_eq!(c.admitted(i), i % 2 == 1);
        }
        assert!(
            c.original.capacity() * size_of::<(u64, usize)>() + c.allowed.capacity() * 8
                <= HOST_BYTES
        );
        let mut empty = Cohort::new(1, 1, 0).unwrap();
        empty.seal_inventory().unwrap();
        empty.validate(1, 1, 0).unwrap();
        assert!(!empty.admitted(0));
    }
}
