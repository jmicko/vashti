//! A real joined native submission, not numerical/state/erasure acceptance.
//! Construction is private and requires the original serialized runtime call.
use super::*;
use cubecl_runtime::server::{NativeStrataBranch, NativeStrataInvocation};

// The recording key packs two independent original pointer arrangements.
// Each family validates its own component, while the key retains both.
fn state_pair_roles(combined: u8) -> [u8; 2] {
    [combined & 1, combined >> 1]
}

pub(crate) struct Completed {
    fault: BufferId,
    controls: Zeroizing<[usize; 7]>, // position, count, state role, route epoch, usage, state mode, fold mode
}
impl Drop for Completed {
    fn drop(&mut self) {
        self.fault.zeroize();
    }
}
impl Completed {
    pub(crate) fn check(&self, fault: BufferId, position: u64, count: usize) -> Result<()> {
        if self.fault != fault || position != self.controls[0] as u64 || count != self.controls[1] {
            return Err(Error::InvalidFormat("native replay transaction mismatch"));
        }
        Ok(())
    }
    pub(crate) fn defers_gdn_state(&self) -> bool {
        self.controls[5] != 0
    }
    pub(crate) fn grouped_status_folds(&self) -> bool {
        self.controls[6] != 0
    }
    pub(crate) fn gdn_state_role(&self) -> u8 {
        state_pair_roles(self.controls[2] as u8)[0]
    }
    pub(crate) fn ple_history_role(&self) -> u8 {
        state_pair_roles(self.controls[2] as u8)[1]
    }
    pub(crate) fn route_epoch(&self) -> u32 {
        self.controls[3] as u32
    }
    pub(crate) fn route_usage(&self) -> bool {
        self.controls[4] != 0
    }
    pub(crate) fn check_fault_owner(&self, arena: &PrivateArena, fault: BufferId) -> Result<()> {
        if self.fault != fault || u::words(arena, fault)? != 1 {
            return Err(Error::InvalidFormat("native replay fault owner mismatch"));
        }
        Ok(())
    }
}
/// An actual submitted original prefix; no numerical/state authority. The
/// original arena borrow prevents continuing on a different client/context.
pub(crate) struct Prefix<'a> {
    arena: &'a PrivateArena,
    key: NativeStrataBranch,
    invocation: NativeStrataInvocation,
    fault: BufferId,
    controls: Zeroizing<[usize; 7]>,
    retired: Arc<AtomicBool>,
    armed: bool,
}
impl Drop for Prefix<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.retired.store(true, Ordering::Release);
        }
        self.invocation.wipe_private();
        self.fault.zeroize();
    }
}
impl Prefix<'_> {
    pub(crate) fn finish(mut self) -> Result<Completed> {
        self.arena
            .client()
            .map_err(u::fault)?
            .replay_native_strata_target_suffix(self.key, self.invocation)
            .map_err(u::fault)?;
        self.armed = false;
        Ok(Completed {
            fault: self.fault,
            controls: std::mem::replace(&mut self.controls, Zeroizing::new([0; 7])),
        })
    }
}
pub(crate) fn try_prefix<'a>(
    arena: &'a PrivateArena,
    key: NativeStrataBranch,
    invocation: NativeStrataInvocation,
    fault: BufferId,
    retired: Arc<AtomicBool>,
) -> Result<Option<Prefix<'a>>> {
    if !crate::strata_intel_core::enabled()
        || u::words(arena, fault)? != 1
        || key.target_ple_handoff() != Some(true)
    {
        return Err(Error::InvalidFormat("native target prefix owner"));
    }
    invocation.validate_for(key).map_err(u::fault)?;
    use cubecl_runtime::server::NativeStrataValue;
    let position = invocation
        .value(NativeStrataValue::Position { offset: 0 })
        .map_err(u::fault)?;
    let epoch = invocation
        .value(NativeStrataValue::RouteEpoch)
        .map_err(u::fault)?;
    let usage = invocation
        .value(NativeStrataValue::RouteUsage)
        .map_err(u::fault)?;
    let (count, _, state) = key
        .target_geometry()
        .ok_or(Error::InvalidFormat("native replay target shape required"))?;
    if !arena
        .client()
        .map_err(u::fault)?
        .replay_native_strata_target_prefix(key, invocation)
        .map_err(u::fault)?
    {
        return Ok(None);
    }
    Ok(Some(Prefix {
        arena,
        key,
        invocation,
        fault,
        controls: Zeroizing::new([
            position as usize,
            count as usize,
            state as usize,
            epoch as usize,
            usage as usize,
            key.target_defers_gdn_state()
                .ok_or(Error::InvalidFormat("native replay target state mode"))?
                as usize,
            key.target_grouped_status_folds()
                .ok_or(Error::InvalidFormat("native replay target fold mode"))?
                as usize,
        ]),
        retired,
        armed: true,
    }))
}
/// A miss leaves the capture path available. Any failed upload/submission/join
/// remains an error; callers cannot recover by manufacturing pending receipts.
pub(crate) fn try_submit(
    arena: &PrivateArena,
    key: NativeStrataBranch,
    invocation: NativeStrataInvocation,
    fault: BufferId,
) -> Result<Option<Completed>> {
    if !crate::strata_intel_core::enabled() || u::words(arena, fault)? != 1 {
        return Err(Error::InvalidFormat("native replay owning target required"));
    }
    invocation.validate_for(key).map_err(u::fault)?;
    use cubecl_runtime::server::NativeStrataValue;
    let position = invocation
        .value(NativeStrataValue::Position { offset: 0 })
        .map_err(u::fault)?;
    let epoch = invocation
        .value(NativeStrataValue::RouteEpoch)
        .map_err(u::fault)?;
    let usage = invocation
        .value(NativeStrataValue::RouteUsage)
        .map_err(u::fault)?;
    let (count, _, state_role) = key
        .target_geometry()
        .ok_or(Error::InvalidFormat("native replay target shape required"))?;
    if !arena
        .client()
        .map_err(u::fault)?
        .replay_native_strata_branch(key, invocation)
        .map_err(u::fault)?
    {
        return Ok(None);
    }
    Ok(Some(Completed {
        fault,
        controls: Zeroizing::new([
            position as usize,
            count as usize,
            state_role as usize,
            epoch as usize,
            usage as usize,
            key.target_defers_gdn_state()
                .ok_or(Error::InvalidFormat("native replay target state mode"))?
                as usize,
            key.target_grouped_status_folds()
                .ok_or(Error::InvalidFormat("native replay target fold mode"))?
                as usize,
        ]),
    }))
}

#[cfg(test)]
mod state_role_tests {
    #[test]
    fn original_recording_key_keeps_gdn_and_ple_pointer_orders_independent() {
        use cubecl_runtime::server::NativeStrataBranch;
        // Pure coordinate projection. No joined receipt or GPU authority is
        // fabricated: production Completed construction remains private.
        for gdn in 0..=1 {
            for ple in 0..=1 {
                let key = NativeStrataBranch::target_window_with_state(4, 32768, gdn | (ple << 1))
                    .unwrap();
                let (_, _, combined) = key.target_geometry().unwrap();
                assert_eq!(super::state_pair_roles(combined), [gdn, ple]);
            }
        }
        assert!(NativeStrataBranch::target_window_with_state(4, 32768, 4).is_err());
    }
}
