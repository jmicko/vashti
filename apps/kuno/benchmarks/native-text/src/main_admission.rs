//! Main-pool admission for one exclusively owned, ordered resident context.
//!
//! The required allocator has exclusive power-of-two bins, 32 KiB..128 MiB.
//! `upper` includes its actual baseline plus every newly admitted page bound
//! since the last COMPLETED fence. Dropping/evicting handles grants no credit:
//! queued wgpu buffer references can outlive CubeCL's own allocation counters.
//! A telemetry query submits queued work; it is not a completed fence.
//!
//! Capacity planning and allocation admission are separate invariants. Check
//! dense capacity + bank page bounds + resident owned spans + I/O maximum +
//! complete public-vector pages + outstanding upload-source pages against the
//! cap before enabling the resident path. Then initialize this ledger from
//! actual main bytes ONLY. Do not add already allocated arena/public bytes a
//! second time. Charge actual allocations, not that separate capacity plan.
//! Uniform/readback/driver pools are not part of CubeCL main-pool telemetry.

pub const MIN_PAGE_BYTES: u64 = 32 * 1024;
pub const MAX_PAGE_BYTES: u64 = 128 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdmissionError {
    InvalidAllocation,
    ArithmeticOverflow,
    LimitExceeded,
    UnaccountedGrowth,
    Poisoned,
}
impl std::fmt::Display for AdmissionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::InvalidAllocation => "invalid main-pool allocation extent",
            Self::ArithmeticOverflow => "main-pool admission arithmetic overflow",
            Self::LimitExceeded => "main-pool admission limit exceeded",
            Self::UnaccountedGrowth => "unaccounted main-pool page growth",
            Self::Poisoned => "main-pool admission poisoned",
        })
    }
}
impl std::error::Error for AdmissionError {}
pub type Result<T> = std::result::Result<T, AdmissionError>;

/// Enclosing exclusive bin, not payload bytes or Handle::size_in_used().
pub fn page_bytes(requested: u64) -> Result<u64> {
    if requested == 0 || requested > MAX_PAGE_BYTES {
        return Err(AdmissionError::InvalidAllocation);
    }
    requested
        .max(MIN_PAGE_BYTES)
        .checked_next_power_of_two()
        .ok_or(AdmissionError::ArithmeticOverflow)
}

/// Pure startup capacity plan. Components must already be full page bounds;
/// each source/chunk must be rounded separately before summing.
pub fn check_capacity(limit: u64, components: &[u64]) -> Result<u64> {
    let total = components.iter().try_fold(0_u64, |sum, &bytes| {
        sum.checked_add(bytes)
            .ok_or(AdmissionError::ArithmeticOverflow)
    })?;
    if total > limit {
        return Err(AdmissionError::LimitExceeded);
    }
    Ok(total)
}

#[derive(Debug)]
pub struct MainAdmission {
    limit: u64,
    upper: u64,
    poisoned: bool,
}
impl MainAdmission {
    pub fn new(limit: u64, actual_main_baseline: u64) -> Result<Self> {
        if actual_main_baseline > limit {
            return Err(AdmissionError::LimitExceeded);
        }
        Ok(Self {
            limit,
            upper: actual_main_baseline,
            poisoned: false,
        })
    }
    pub fn upper_bytes(&self) -> u64 {
        self.upper
    }
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }
    pub fn poison(&mut self) {
        self.poisoned = true;
    }
    fn active(&self) -> Result<()> {
        if self.poisoned {
            Err(AdmissionError::Poisoned)
        } else {
            Ok(())
        }
    }
    /// Check a real sample taken BEFORE precharge/allocation. Increasing the
    /// sample cannot silently erase previously retained retirement debt.
    /// Setup/external owners must admit their allocations before this call.
    pub fn observe_before(&mut self, actual_main: u64) -> Result<()> {
        self.observe_after(actual_main)
    }
    /// Atomically charge every request's full bin before allocating any of the
    /// group. Admission rejection changes nothing; after charge there is no
    /// cancellation/refund API, even when allocation or later dispatch fails.
    pub fn precharge(&mut self, requested_allocations: &[u64]) -> Result<u64> {
        self.active()?;
        let charged = requested_allocations
            .iter()
            .try_fold(0_u64, |sum, &bytes| {
                sum.checked_add(page_bytes(bytes)?)
                    .ok_or(AdmissionError::ArithmeticOverflow)
            })?;
        let upper = self
            .upper
            .checked_add(charged)
            .ok_or(AdmissionError::ArithmeticOverflow)?;
        if upper > self.limit {
            return Err(AdmissionError::LimitExceeded);
        }
        self.upper = upper;
        Ok(charged)
    }
    pub(crate) fn precharge_persistent(
        &mut self,
        allocations: &[u64],
        alignment: u64,
    ) -> Result<u64> {
        self.active()?;
        if !crate::strata_port::enabled() || alignment == 0 || alignment > 4096 {
            return Err(AdmissionError::InvalidAllocation);
        }
        let charged = allocations.iter().try_fold(0u64, |sum, &n| {
            if n == 0 || n > MAX_PAGE_BYTES || !n.is_multiple_of(alignment) {
                return Err(AdmissionError::InvalidAllocation);
            }
            sum.checked_add(n).ok_or(AdmissionError::ArithmeticOverflow)
        })?;
        let upper = self
            .upper
            .checked_add(charged)
            .ok_or(AdmissionError::ArithmeticOverflow)?;
        if upper > self.limit {
            return Err(AdmissionError::LimitExceeded);
        }
        self.upper = upper;
        Ok(charged)
    }
    /// Post-allocation sample, before dependent computation/cache acceptance.
    /// Unexpected growth means an allocation owner/bin/config was missed;
    /// fail closed even when the physical sample is still below the cap.
    pub fn observe_after(&mut self, actual_main: u64) -> Result<()> {
        self.active()?;
        let error = if actual_main > self.limit {
            Some(AdmissionError::LimitExceeded)
        } else if actual_main > self.upper {
            Some(AdmissionError::UnaccountedGrowth)
        } else {
            None
        };
        if let Some(error) = error {
            self.poison();
            return Err(error);
        }
        Ok(())
    }
    /// Only the owner may call this AFTER a successful existing router/QSA/
    /// final readback or sync completes ALL earlier work on this context.
    /// Failed telemetry/fence must call poison(), never reset or retry.
    pub fn completed_fence(&mut self, actual_main: u64) -> Result<()> {
        self.observe_after(actual_main)?;
        self.upper = actual_main;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_bin_geometry_including_minimum_and_maximum() {
        assert_eq!(page_bytes(1), Ok(32768));
        assert_eq!(page_bytes(32768), Ok(32768));
        assert_eq!(page_bytes(32769), Ok(65536));
        assert_eq!(page_bytes(MAX_PAGE_BYTES), Ok(MAX_PAGE_BYTES));
        for n in [0, MAX_PAGE_BYTES + 1, u64::MAX] {
            assert_eq!(page_bytes(n), Err(AdmissionError::InvalidAllocation));
        }
    }
    #[test]
    fn a_hit_and_an_eviction_sample_grant_no_credit() {
        let mut ledger = MainAdmission::new(4 * MIN_PAGE_BYTES, 2 * MIN_PAGE_BYTES).unwrap();
        assert_eq!(ledger.precharge(&[]), Ok(0));
        ledger.observe_after(MIN_PAGE_BYTES).unwrap();
        assert_eq!(ledger.upper_bytes(), 2 * MIN_PAGE_BYTES);
        ledger.precharge(&[1]).unwrap();
        assert_eq!(ledger.upper_bytes(), 3 * MIN_PAGE_BYTES);
    }
    #[test]
    fn each_chunk_is_rounded_separately_and_exact_limit_succeeds() {
        let mut ledger = MainAdmission::new(3 * MIN_PAGE_BYTES, MIN_PAGE_BYTES).unwrap();
        assert_eq!(ledger.precharge(&[1, 1]), Ok(2 * MIN_PAGE_BYTES));
        assert_eq!(ledger.upper_bytes(), 3 * MIN_PAGE_BYTES);
        ledger.observe_after(3 * MIN_PAGE_BYTES).unwrap();
        assert_eq!(ledger.precharge(&[1]), Err(AdmissionError::LimitExceeded));
        assert_eq!(ledger.upper_bytes(), 3 * MIN_PAGE_BYTES);
        assert_eq!(
            MainAdmission::new(3 * MIN_PAGE_BYTES - 1, MIN_PAGE_BYTES)
                .unwrap()
                .precharge(&[1, 1]),
            Err(AdmissionError::LimitExceeded)
        );
    }
    #[test]
    fn invalid_group_and_arithmetic_overflow_never_partially_charge() {
        let mut ledger = MainAdmission::new(u64::MAX, u64::MAX - 1).unwrap();
        assert_eq!(
            ledger.precharge(&[1]),
            Err(AdmissionError::ArithmeticOverflow)
        );
        assert_eq!(ledger.upper_bytes(), u64::MAX - 1);
        let mut ledger = MainAdmission::new(4 * MIN_PAGE_BYTES, MIN_PAGE_BYTES).unwrap();
        assert_eq!(
            ledger.precharge(&[1, 0]),
            Err(AdmissionError::InvalidAllocation)
        );
        assert_eq!(ledger.upper_bytes(), MIN_PAGE_BYTES);
        assert_eq!(
            check_capacity(u64::MAX, &[u64::MAX, 1]),
            Err(AdmissionError::ArithmeticOverflow)
        );
    }
    #[test]
    fn only_a_completed_fence_reclaims_retirement_debt() {
        let mut ledger = MainAdmission::new(4 * MIN_PAGE_BYTES, MIN_PAGE_BYTES).unwrap();
        ledger.precharge(&[1, 1]).unwrap();
        ledger.observe_before(MIN_PAGE_BYTES).unwrap();
        assert_eq!(ledger.upper_bytes(), 3 * MIN_PAGE_BYTES);
        ledger.completed_fence(MIN_PAGE_BYTES).unwrap();
        assert_eq!(ledger.upper_bytes(), MIN_PAGE_BYTES);
    }
    #[test]
    fn failed_postcheck_or_fence_permanently_poison() {
        let mut ledger = MainAdmission::new(4 * MIN_PAGE_BYTES, MIN_PAGE_BYTES).unwrap();
        assert_eq!(
            ledger.observe_after(MIN_PAGE_BYTES + 1),
            Err(AdmissionError::UnaccountedGrowth)
        );
        assert!(ledger.is_poisoned());
        assert_eq!(ledger.completed_fence(0), Err(AdmissionError::Poisoned));
        assert_eq!(ledger.precharge(&[]), Err(AdmissionError::Poisoned));
        let mut ledger = MainAdmission::new(MIN_PAGE_BYTES, MIN_PAGE_BYTES).unwrap();
        assert_eq!(
            ledger.completed_fence(MIN_PAGE_BYTES + 1),
            Err(AdmissionError::LimitExceeded)
        );
        assert!(ledger.is_poisoned());
        let mut ledger = MainAdmission::new(MIN_PAGE_BYTES, 0).unwrap();
        ledger.poison();
        assert_eq!(ledger.observe_before(0), Err(AdmissionError::Poisoned));
    }
    #[test]
    fn capacity_reservation_does_not_double_charge_existing_baseline() {
        let arena_and_public = 4 * MIN_PAGE_BYTES;
        let future_io_and_source = 2 * MIN_PAGE_BYTES;
        let cache_capacity = 2 * MIN_PAGE_BYTES;
        let limit = 8 * MIN_PAGE_BYTES;
        assert_eq!(
            check_capacity(
                limit,
                &[arena_and_public, future_io_and_source, cache_capacity]
            ),
            Ok(limit)
        );
        let mut ledger = MainAdmission::new(limit, arena_and_public).unwrap();
        ledger.precharge(&[1, 1]).unwrap();
        assert_eq!(
            ledger.upper_bytes(),
            arena_and_public + future_io_and_source
        );
        assert!(MainAdmission::new(limit - 1, limit).is_err());
    }
}
