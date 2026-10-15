//! One native client uses one authoritative MAIN ledger. Independent legacy
//! laboratories keep their original account; no peer usage becomes a zero base.
use super::*;
use crate::executor::ResidentAdmission;

pub(super) enum Ledger {
    Exclusive(MainAdmission),
    Shared(ResidentAdmission),
}
impl Ledger {
    pub(super) fn precharge(&mut self, requests: &[u64]) -> Result<u64> {
        match self {
            Self::Exclusive(l) => l.precharge(requests).map_err(error),
            Self::Shared(a) => a.shared_main(|l| l.precharge(requests)),
        }
    }
    pub(super) fn observe_before(&mut self, n: u64) -> Result<()> {
        self.observe_after(n)
    }
    pub(super) fn observe_after(&mut self, n: u64) -> Result<()> {
        match self {
            Self::Exclusive(l) => l.observe_after(n).map_err(error),
            Self::Shared(a) => a.shared_main(|l| l.observe_after(n)),
        }
    }
    pub(super) fn completed_fence(&mut self, n: u64) -> Result<()> {
        match self {
            Self::Exclusive(l) => l.completed_fence(n).map_err(error),
            Self::Shared(a) => a.shared_main(|l| l.completed_fence(n)),
        }
    }
    pub(super) fn poison(&mut self) {
        match self {
            Self::Exclusive(l) => l.poison(),
            Self::Shared(a) => a.poison(),
        }
    }
    pub(super) fn shared(&self) -> Option<&ResidentAdmission> {
        match self {
            Self::Shared(a) => Some(a),
            _ => None,
        }
    }
}
impl<'g, 'f> ResidentMtp<'g, 'f> {
    pub(crate) fn shared_capacity() -> Result<u64> {
        if !crate::strata_intel_core::enabled() {
            return Err(Error::InvalidBufferLength);
        }
        public_plan_for(MtpProfile::StrataQ2)?
            .checked_sub(crate::strata_shared_weights::public_bytes())
            .ok_or(Error::ArithmeticOverflow)?
            .checked_add(PRIVATE_BYTES as u64)
            .and_then(|n| n.checked_add(PUBLIC_SOURCE_BYTES as u64))
            .ok_or(Error::ArithmeticOverflow)
    }
    pub(crate) fn new_shared_strata_q2(
        graph: &'g MtpGraph<'f>,
        client: ComputeClient<WgpuRuntime>,
        limits: &wgpu::Limits,
        permit: crate::executor::NativeMtpAdmission,
    ) -> Result<Self> {
        if !crate::strata_intel_core::enabled() {
            return Err(Error::InvalidBufferLength);
        }
        let admission = permit.consume(graph.target_binding_id(), Self::shared_capacity()?)?;
        admission.check_client(&client)?;
        let retired = admission.retired_context();
        Self::construct_mode(
            graph,
            client,
            limits,
            retired,
            true,
            MtpProfile::StrataQ2,
            false,
            Some(admission),
        )
    }
    pub(super) fn audit_io(&self) -> Result<()> {
        if let Some(a) = self.ledger.shared() {
            a.shared_observation_checkpoint()
        } else {
            io_audit(self.client()?, &self.retired)
        }
    }
    /// Called only after BOTH real model owners and the executor have closed.
    /// An own-arena close is not a claim that a shared allocator is empty.
    pub(crate) fn finish_shared_root(&self, receipt: &mut CloseReceipt) -> Result<()> {
        if !self.closed || self.ledger.shared().is_none() {
            return Err(Error::InvalidBufferLength);
        }
        crate::executor::audit_native_context_recovery(self.client()?, &self.retired)?;
        // The real final observation above proved zero; this does not credit
        // or resurrect the target's already poisoned admission ledger.
        receipt.main_reserved = 0;
        Ok(())
    }
}

#[cfg(test)]
mod shared_weight_tests {
    use super::*;
    #[test]
    fn native_shared_weights_mtp_consumes_the_same_original_tiles_without_a_second_charge() {
        if !crate::strata_intel_core::enabled() {
            return;
        }
        for role in [
            crate::strata_shared_weights::Tensor::Embedding,
            crate::strata_shared_weights::Tensor::Head,
        ] {
            let spec = if role == crate::strata_shared_weights::Tensor::Embedding {
                embedding_for(MtpProfile::StrataQ2)
            } else {
                *matrices_for(MtpProfile::StrataQ2)
                    .iter()
                    .find(|s| s.name == role.name())
                    .unwrap()
            };
            assert_eq!(spec.dtype, role.dtype());
            let actual = tile_plan(spec).unwrap();
            let expected = crate::strata_shared_weights::tiles(role)
                .into_iter()
                .map(|t| (t.first, t.rows, t.logical, t.owned))
                .collect::<Vec<_>>();
            assert_eq!(actual, expected);
        }
        assert_eq!(
            ResidentMtp::shared_capacity().unwrap() + crate::strata_shared_weights::public_bytes(),
            public_plan_for(MtpProfile::StrataQ2).unwrap()
                + PRIVATE_BYTES as u64
                + PUBLIC_SOURCE_BYTES as u64
        );
    }
}
