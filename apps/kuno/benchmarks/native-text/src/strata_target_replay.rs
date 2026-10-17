//! A real joined native submission, not numerical/state/erasure acceptance.
//! Construction is private and requires the original serialized runtime call.
use super::*;
use cubecl_runtime::server::{NativeStrataBranch, NativeStrataInvocation};

pub(crate) struct Completed {
    fault: BufferId,
    controls: Zeroizing<[usize; 6]>, // position, count, state role, route epoch, usage flag, state mode
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
    pub(crate) fn state_role(&self) -> u8 {
        self.controls[2] as u8
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
        ]),
    }))
}
