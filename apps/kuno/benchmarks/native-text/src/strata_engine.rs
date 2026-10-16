//! Request-scoped Rust owner of the selected Strata Intel inference path.
//! No transport, file output, prompt logging, receipt serialization or device
//! discovery lives here. The caller owns authenticated mapped artifacts and a
//! protected, exclusively admitted runtime; this function owns every inference
//! consumer until checked cleanup. Installed Kuno does not select this lab yet.
use crate::{
    executor::{self, GpuExecutor},
    expert_banks::OriginalPackedProfile,
    resident_model::{ResidentModel, batch::Config},
    resident_mtp::ResidentMtp,
};
use cubecl::{prelude::ComputeClient, wgpu::WgpuRuntime};
use kuno_text::{GgufFile, model::BoundGraph, mtp::MtpGraph};
use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
use zeroize::Zeroizing;

#[path = "strata_global_controller.rs"]
pub(crate) mod controller;
pub(crate) const VOCAB: usize = 248320;
pub(crate) const MAX_OUTPUTS: usize = crate::strata_intel_core::CONTEXT - 1;
pub(crate) const REQUEST_HOST_BYTES: usize = 4 * 1024 * 1024;

/// Owned, transient request. No Debug/Serialize; the token allocation and all
/// sampling values are wiped on every rejection, unwind and ordinary return.
pub(crate) struct Request {
    prompt: Zeroizing<Vec<u32>>,
    outputs: Zeroizing<usize>,
    eos: Zeroizing<u32>,
    minimum: Zeroizing<f32>,
    speculation: crate::strata_speculation::Config,
    sampling: crate::strata_sampler::Params,
}
impl Request {
    pub(crate) fn new(
        prompt: Zeroizing<Vec<u32>>,
        outputs: usize,
        eos: u32,
        sampling: crate::strata_sampler::Params,
        speculation: crate::strata_speculation::Config,
        minimum: f32,
    ) -> kuno_text::Result<Self> {
        sampling.validate()?;
        speculation.validate()?;
        if !crate::strata_intel_core::enabled()
            || prompt.is_empty()
            || !(1..=MAX_OUTPUTS).contains(&outputs)
            || eos as usize >= VOCAB
            || prompt.iter().any(|&v| v as usize >= VOCAB)
            || prompt
                .len()
                .checked_add(outputs)
                .is_none_or(|n| n > crate::strata_intel_core::CONTEXT)
            || !minimum.is_finite()
            || !(0.0..=1.0).contains(&minimum)
            || prompt
                .capacity()
                .checked_mul(4)
                .is_none_or(|n| n > crate::strata_intel_core::CONTEXT * 4)
            || crate::strata_speculation::HOST_BYTES + (prompt.capacity() + outputs) * 4 + 64 * 1024
                > REQUEST_HOST_BYTES
        {
            return Err(kuno_text::Error::InvalidFormat("native request bounds"));
        }
        Ok(Self {
            prompt,
            outputs: Zeroizing::new(outputs),
            eos: Zeroizing::new(eos),
            minimum: Zeroizing::new(minimum),
            speculation,
            sampling,
        })
    }
    pub(crate) fn geometry(&self) -> kuno_text::Result<(usize, usize, usize)> {
        Ok((
            self.prompt.len(),
            *self.outputs,
            self.prompt
                .len()
                .checked_add(*self.outputs)
                .ok_or(kuno_text::Error::ArithmeticOverflow)?,
        ))
    }
}

/// Fixed error categories; no backend/private error payload can be displayed.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Failure {
    Rejected,
    Inference,
    RecoveryRequired,
}
impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Rejected => "native request rejected",
            Self::Inference => "native inference failed",
            Self::RecoveryRequired => "native context recovery required",
        })
    }
}
impl std::error::Error for Failure {}

/// Content-free completion returned only after all original owners have closed.
/// Output tokens are delivered exclusively to the caller's callback.
pub(crate) struct Completion {
    pub generation: controller::Completion,
    pub target_main: crate::private_arena::OwnedZeroAudit,
    pub target_sources: crate::private_arena::OwnedZeroAudit,
    pub draft_public_bytes: Option<u64>,
    pub backlog: cubecl_runtime::server::PublicTargetSubmissionBacklog,
}
fn checked<T>(f: impl FnOnce() -> Result<T, Failure>) -> Result<T, Failure> {
    catch_unwind(AssertUnwindSafe(f)).map_err(|_| Failure::Inference)?
}

/// Actual selected source ownership: every constructor phase is held outside
/// the fallible inference closure so failures can close draft before target,
/// drop target before source detach, then audit the final shared physical root.
///
/// # Safety
/// Caller authenticates complete immutable target/MTP artifacts; `file` is the
/// original backing of `graph` (not another equivalent file). Caller retains their
/// backing, protects process memory from swap/core dumps and installs a redacted
/// panic hook. The exact client/device/queue/limits must share one exclusively
/// admitted Intel context with the full MAIN/HOST/driver reservation. No peer may
/// submit/reset/close it during this call. These are real resource obligations,
/// not assertions derived from a request or this function's metadata.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn execute<'m, 'f>(
    file: &'f GgufFile,
    graph: BoundGraph<'m, 'f>,
    draft_graph: Option<&MtpGraph<'_>>,
    client: &ComputeClient<WgpuRuntime>,
    limits: &wgpu::Limits,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    backlog: cubecl_runtime::server::PublicTargetSubmissionBacklog,
    request: Request,
    cancel: &AtomicBool,
    emit: &mut dyn FnMut(u32) -> kuno_text::Result<()>,
    phase: &mut dyn FnMut(&'static str),
) -> Result<Completion, Failure> {
    if !crate::strata_intel_core::enabled()
        || device.adapter_info().vendor != 0x8086
        || cancel.load(Ordering::Acquire)
        || crate::strata_port::profile_target()
        || crate::strata_port::profile_native_target()
        || graph.verify_unchanged().is_err()
        || draft_graph.is_some_and(|d| {
            d.target_binding_id() != Some(graph.binding_id()) || d.verify_unchanged().is_err()
        })
    {
        return Err(Failure::Rejected);
    }
    let (prompt, outputs, context) = request.geometry().map_err(|_| Failure::Rejected)?;
    let retired = Arc::new(AtomicBool::new(false));
    let mut runtime_owned = false;
    let mut tables = None;
    let mut executor = None;
    let mut target = None;
    let mut draft = None;
    let attempted = checked(|| {
        crate::target_lab_build::validate_runtime(true).map_err(|_| Failure::Rejected)?;
        crate::resident_util::select_retained_metadata(true);
        crate::resident_util::select_compute_scratch_clear(true);
        client
            .configure_public_target_request_io_with_backlog(backlog)
            .map_err(|_| Failure::Rejected)?;
        runtime_owned = true;
        tables = crate::strata_intel_core::Tables::new(client, Some(device))
            .map_err(|_| Failure::Inference)?;
        let mut e = GpuExecutor::new_with_profile(
            file,
            client.clone(),
            crate::strata_port::static_cache_bytes(),
            crate::daslab_profile::BANK_SLOTS,
            crate::target_lab_build::NATIVE_SHARED_MAIN_BYTES as usize,
            OriginalPackedProfile::DaslabIq3,
        )
        .map_err(|_| Failure::Inference)?;
        if draft_graph.is_some() {
            e.reserve_native_mtp(ResidentMtp::shared_capacity().map_err(|_| Failure::Inference)?)
                .map_err(|_| Failure::Inference)?;
        }
        e.target_rows_reserved = true;
        // Preserve original constructor reservations; global activation still
        // rejects legacy bank/source execution. These are not active old tiers.
        e.target_batch_router_reserved = true;
        e.target_frozen_host_router_reserved = true;
        e.target_frozen_gu_integer_reserved = true;
        e.target_frozen_down_integer_reserved = true;
        e.target_verify3_dense = true;
        e.target_verify3_fused = true;
        e.token_io_checkpoint = true;
        e.reserve_public_four_rows()
            .map_err(|_| Failure::Inference)?;
        e.enable_transient_resident(retired.clone())
            .map_err(|_| Failure::Inference)?;
        e.fast_dense = true;
        e.fast_dense_fault = true;
        e.shared_kernels = true;
        e.shared_gdn = true;
        executor = Some(e);
        let e = executor.as_mut().ok_or(Failure::Inference)?;
        let config = Config::new(client, limits, 8)
            .and_then(Config::reserve_target_rows)
            .and_then(Config::reserve_latest_target_checkpoint)
            .and_then(Config::reserve_public_gpu_qsa_selector)
            .and_then(Config::reserve_public_frozen_host_sources)
            .and_then(|c| c.reserve_public_language_request_io(prompt, outputs))
            .and_then(Config::reserve_public_frozen_host_batch_router)
            .and_then(Config::reserve_public_frozen_gu_integer)
            .and_then(Config::reserve_public_frozen_down_integer)
            .and_then(Config::reserve_public_grouped_ple_reads)
            .and_then(Config::reserve_public_grouped_status_folds)
            .and_then(Config::reserve_public_four_rows)
            .map_err(|_| Failure::Inference)?;
        phase("target");
        target = Some(
            ResidentModel::new_with_native_batch(graph, client.clone(), context, e, config, true)
                .map_err(|_| Failure::Inference)?,
        );
        let t = target.as_mut().ok_or(Failure::Inference)?;
        phase("global_sources");
        // SAFETY: same authenticated/protected/exclusive context required above;
        // no source loan is used in the selected32768-context serving mode.
        unsafe {
            t.prepare_global_sources_before_prefill(e, device, queue, 0)
                .map_err(|_| Failure::Inference)?;
        }
        phase("shared_target_weights");
        e.prepare_native_shared_target_weights(graph)
            .map_err(|_| Failure::Inference)?;
        if let Some(g) = draft_graph {
            phase("shared_draft");
            let permit = e.shared_mtp_admission().map_err(|_| Failure::Inference)?;
            draft = Some(
                ResidentMtp::new_shared_strata_q2(g, client.clone(), limits, permit)
                    .map_err(|_| Failure::Inference)?,
            );
        }
        let Request {
            prompt,
            outputs,
            eos,
            minimum,
            speculation,
            sampling,
        } = request;
        let mut delivered = Zeroizing::new(0usize);
        let done = controller::run(
            t,
            e,
            draft.as_mut(),
            controller::Request {
                prompt: &prompt,
                max_outputs: *outputs,
                eos: *eos,
                speculation,
                draft_min_probability: *minimum,
                sampling,
            },
            cancel,
            &mut |token| {
                if *delivered >= *outputs || token as usize >= VOCAB {
                    return Err("native output delivery bound".into());
                }
                emit(token)?;
                *delivered += 1;
                Ok(())
            },
            phase,
        )
        .map_err(|_| Failure::Inference)?;
        if done.outputs != *delivered {
            return Err(Failure::Inference);
        }
        Ok(done)
    });
    if !runtime_owned {
        // Failed configuration never grants ownership of a peer/preexisting runtime.
        return Err(attempted.err().unwrap_or(Failure::RecoveryRequired));
    }
    // Never short-circuit peer cleanup after an inference/constructor failure.
    let mut draft_close = checked(|| {
        draft
            .as_mut()
            .map(|d| d.close())
            .transpose()
            .map_err(|_| Failure::RecoveryRequired)
    });
    let target_close = checked(|| {
        target
            .as_mut()
            .map(|t| t.close())
            .transpose()
            .map_err(|_| Failure::RecoveryRequired)
    });
    let target_audit = target
        .as_ref()
        .and_then(ResidentModel::public_owned_zero_audit_receipt);
    let target_drop = checked(|| {
        drop(target.take());
        Ok(())
    });
    let executor_close = checked(|| {
        executor
            .as_mut()
            .map(|e| e.close_transient_resident())
            .transpose()
            .map_err(|_| Failure::RecoveryRequired)
    });
    let executor_drop = checked(|| {
        drop(executor.take());
        Ok(())
    });
    let tables_drop = checked(|| {
        drop(tables.take());
        Ok(())
    });
    let physical = checked(|| {
        if let Some(d) = draft.as_ref() {
            let receipt = draft_close
                .as_mut()
                .map_err(|_| Failure::RecoveryRequired)?
                .as_mut()
                .ok_or(Failure::RecoveryRequired)?;
            d.finish_shared_root(receipt)
                .map_err(|_| Failure::RecoveryRequired)?;
            crate::mtp_lab_profile::Profile::StrataQ2
                .validate_close_with_english(receipt, false)
                .map_err(|_| Failure::RecoveryRequired)?;
            Ok(Some(receipt.public_bytes))
        } else {
            executor::audit_native_context_recovery(client, &retired)
                .map_err(|_| Failure::RecoveryRequired)?;
            Ok(None)
        }
    });
    let draft_drop = checked(|| {
        drop(draft.take());
        Ok(())
    });
    if draft_close.is_err()
        || target_close.is_err()
        || executor_close.is_err()
        || physical.is_err()
        || target_drop.is_err()
        || executor_drop.is_err()
        || tables_drop.is_err()
        || draft_drop.is_err()
        || retired.load(Ordering::Acquire)
        || file.verify_file_unchanged().is_err()
        || graph.verify_unchanged().is_err()
        || draft_graph.is_some_and(|d| d.verify_unchanged().is_err())
        || client.public_target_request_io_configured().ok() != Some(true)
        || client.public_target_submission_backlog().ok() != Some(backlog)
    {
        retired.store(true, Ordering::Release);
        return Err(Failure::RecoveryRequired);
    }
    let generation = attempted?;
    let (target_main, target_sources) = target_audit.ok_or(Failure::RecoveryRequired)?;
    Ok(Completion {
        generation,
        target_main,
        target_sources,
        draft_public_bytes: physical?,
        backlog,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn request(tokens: Vec<u32>, outputs: usize) -> kuno_text::Result<Request> {
        Request::new(
            Zeroizing::new(tokens),
            outputs,
            248319,
            crate::strata_sampler::Params::default(),
            crate::strata_speculation::Config::native(4),
            0.5,
        )
    }
    #[test]
    fn owned_requests_accept_actual_varying_tokens_and_all_context_partitions() {
        if !crate::strata_intel_core::enabled() {
            return;
        }
        for prompt in [1, 17, 1024, 4096, 8193, 16384, 32767] {
            let outputs = crate::strata_intel_core::CONTEXT - prompt;
            let tokens = (0..prompt)
                .map(|i| (i * 13 % VOCAB) as u32)
                .collect::<Vec<_>>();
            let r = request(tokens, outputs).unwrap();
            assert_eq!(r.geometry().unwrap(), (prompt, outputs, 32768));
            assert_eq!(r.prompt[prompt - 1], ((prompt - 1) * 13 % VOCAB) as u32);
        }
        for (tokens, outputs) in [
            (vec![], 1),
            (vec![0], 0),
            (vec![0], 32768),
            (vec![VOCAB as u32], 1),
            (vec![0; 32768], 1),
        ] {
            assert!(request(tokens, outputs).is_err());
        }
        let mut bloated = Vec::with_capacity(32769);
        bloated.push(1);
        assert!(request(bloated, 1).is_err());
    }
    #[test]
    fn request_rejects_invalid_sampler_and_draft_parameters_before_construction() {
        let mut sampling = crate::strata_sampler::Params::default();
        sampling.top_p = f32::NAN;
        assert!(
            Request::new(
                Zeroizing::new(vec![1]),
                1,
                248319,
                sampling,
                crate::strata_speculation::Config::native(4),
                0.5
            )
            .is_err()
        );
        for minimum in [f32::NAN, -0.1, 1.1] {
            assert!(
                Request::new(
                    Zeroizing::new(vec![1]),
                    1,
                    248319,
                    crate::strata_sampler::Params::default(),
                    crate::strata_speculation::Config::native(4),
                    minimum
                )
                .is_err()
            );
        }
        assert!(
            Request::new(
                Zeroizing::new(vec![1]),
                1,
                248319,
                crate::strata_sampler::Params::default(),
                crate::strata_speculation::Config::native(9),
                0.5
            )
            .is_err()
        );
    }
    #[test]
    fn checked_inference_boundary_retains_fixed_errors_and_catches_unwinds() {
        assert_eq!(
            checked::<()>(|| Err(Failure::Rejected)),
            Err(Failure::Rejected)
        );
        assert_eq!(
            checked::<()>(|| panic!("synthetic public failure")),
            Err(Failure::Inference)
        );
        assert_eq!(checked(|| Ok(17)), Ok(17));
        assert_eq!(Failure::Inference.to_string(), "native inference failed");
    }
}
