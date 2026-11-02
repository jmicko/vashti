//! Loaded-session Rust owner of the selected Strata Intel inference path.
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
#[path = "strata_input.rs"]
pub(crate) mod input;
pub(crate) const VOCAB: usize = 248320;
// Pinned selected serve preflight (generate.cpp:9406) reserves eight cells
// beyond prompt+requested output. Target windows and the next MTP front remain
// full width even when delivery stops early at the cap; do not shrink them.
pub(crate) const CONTEXT_HEADROOM: usize = 8;
pub(crate) const MAX_OUTPUTS: usize = crate::strata_intel_core::CONTEXT - CONTEXT_HEADROOM - 1;
pub(crate) fn request_end(prompt: usize, outputs: usize) -> Option<usize> {
    prompt.checked_add(outputs).filter(|&end| {
        end.checked_add(CONTEXT_HEADROOM)
            .is_some_and(|reserved| reserved <= crate::strata_intel_core::CONTEXT)
    })
}
pub(crate) const REQUEST_HOST_BYTES: usize = 4 * 1024 * 1024;

// Fixed-public diagnostics only. Preserve the original failure and never
// format arbitrary backend/callback errors or request data.
#[track_caller]
pub(crate) fn report_private_reset_error(error: kuno_text::Error) -> kuno_text::Error {
    if crate::strata_intel_core::enabled()
        && std::env::var("KUNO_LAB_LANGUAGE_PUBLIC_ONLY").as_deref() == Ok("1")
    {
        let label = match &error {
            kuno_text::Error::InvalidFormat(label) | kuno_text::Error::LimitExceeded(label) => {
                *label
            }
            kuno_text::Error::InvalidBufferLength => "buffer_length",
            kuno_text::Error::OutOfBounds => "range",
            kuno_text::Error::FileChanged => "file_changed",
            _ => "model_error",
        };
        let site = std::panic::Location::caller();
        eprintln!(
            "native_public_reset_failure source_file={} source_line={} category={}",
            site.file(),
            site.line(),
            label
        );
    }
    error
}

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
            || request_end(prompt.len(), outputs).is_none()
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
            request_end(self.prompt.len(), *self.outputs)
                .ok_or(kuno_text::Error::InvalidFormat("native request headroom"))?,
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

// Counts are checked only AFTER real original-owner reset and final close.
// This comparison never grants a fence, release/refund or erasure capability.
fn retained_erasure_agrees_with_close(
    retained: (
        crate::private_arena::OwnedZeroAudit,
        crate::private_arena::OwnedZeroAudit,
    ),
    closed: (
        crate::private_arena::OwnedZeroAudit,
        crate::private_arena::OwnedZeroAudit,
    ),
    draft: (
        Option<crate::private_arena::OwnedZeroAudit>,
        Option<crate::private_arena::OwnedZeroAudit>,
    ),
) -> bool {
    retained.0 == closed.0
        && retained.1.bytes <= closed.1.bytes
        && retained.1.buffers <= closed.1.buffers
        && draft.0 == draft.1
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
    // SAFETY: identical caller protection/ownership contract; this wrapper
    // still performs exactly one real request followed by checked close.
    unsafe {
        with_loaded_session(
            file,
            graph,
            draft_graph,
            client,
            limits,
            device,
            queue,
            backlog,
            request,
            cancel,
            phase,
            |session, request, phase| {
                let done = session.run(request, cancel, emit, phase)?;
                session.erase_completed(cancel, phase)?;
                Ok(done)
            },
        )
    }
}

/// Run a bounded caller-owned sequence through the SAME loaded model. The
/// callback cannot export the session; every return/unwind still owes the
/// original checked all-peer close. No token/input storage is added here.
///
/// # Safety
/// The complete artifact, exclusive runtime and process protection contract
/// of `execute` must hold until this function returns, including all requests.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn with_loaded_session<'m, 'f, 'd, 'df>(
    file: &'f GgufFile,
    graph: BoundGraph<'m, 'f>,
    draft_graph: Option<&'d MtpGraph<'df>>,
    client: &ComputeClient<WgpuRuntime>,
    limits: &wgpu::Limits,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    backlog: cubecl_runtime::server::PublicTargetSubmissionBacklog,
    request: Request,
    cancel: &AtomicBool,
    phase: &mut dyn FnMut(&'static str),
    run: impl FnOnce(
        &mut Session<'m, 'f, 'd, 'df>,
        Request,
        &mut dyn FnMut(&'static str),
    ) -> Result<controller::Completion, Failure>,
) -> Result<Completion, Failure> {
    let mut session = unsafe {
        Session::load(
            file,
            graph,
            draft_graph,
            client,
            limits,
            device,
            queue,
            backlog,
            &request,
            cancel,
            phase,
        )?
    };
    let generation = match checked(|| run(&mut session, request, phase)) {
        Ok(done) => done,
        Err(failure) => {
            // A preflight rejection may deliberately leave a reusable session
            // untouched. This loaded wrapper still owes an explicit checked
            // close, so Drop cannot hide a recovery failure behind Rejected.
            if !session.close_attempted {
                session.close()?;
            }
            return Err(failure);
        }
    };
    let closed = session.close()?;
    let (target_main, target_sources) = closed.target.ok_or(Failure::RecoveryRequired)?;
    Ok(Completion {
        generation,
        target_main,
        target_sources,
        draft_public_bytes: closed.draft_public_bytes,
        backlog,
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Loading,
    Cold,
    Running,
    /// Generation finished; request state not yet erased.
    Generated,
    Ready,
    Closed,
    Broken,
}
impl State {
    fn start(&mut self, erased: bool) -> Result<(), Failure> {
        if !matches!((*self, erased), (Self::Cold, false) | (Self::Ready, true)) {
            return Err(Failure::Rejected);
        }
        *self = Self::Running;
        Ok(())
    }
    fn generated(&mut self) -> Result<(), Failure> {
        if *self != Self::Running {
            return Err(Failure::RecoveryRequired);
        }
        *self = Self::Generated;
        Ok(())
    }
    fn completed(&mut self) -> Result<(), Failure> {
        if *self != Self::Generated {
            return Err(Failure::RecoveryRequired);
        }
        *self = Self::Ready;
        Ok(())
    }
}
pub(crate) struct Closed {
    target: Option<(
        crate::private_arena::OwnedZeroAudit,
        crate::private_arena::OwnedZeroAudit,
    )>,
    draft_public_bytes: Option<u64>,
}

/// Original shared runtime, weights and request owners. No detached receipt can
/// reopen a closed/failed session. This internal owner is not a public transport.
/// Caller protection/exclusivity in `execute` must hold for its whole lifetime.
pub(crate) struct Session<'m, 'f, 'd, 'df> {
    // Native tables/recording scopes are bound to this calling thread.
    _thread: std::marker::PhantomData<std::rc::Rc<()>>,
    file: &'f GgufFile,
    graph: BoundGraph<'m, 'f>,
    draft_graph: Option<&'d MtpGraph<'df>>,
    client: ComputeClient<WgpuRuntime>,
    backlog: cubecl_runtime::server::PublicTargetSubmissionBacklog,
    retired: Arc<AtomicBool>,
    runtime_owned: bool,
    close_attempted: bool,
    state: State,
    tables: Option<crate::strata_intel_core::Tables>,
    executor: Option<GpuExecutor<'f>>,
    target: Option<ResidentModel<'m, 'f>>,
    draft: Option<ResidentMtp<'d, 'df>>,
    policy: Option<crate::strata_speculation::Policy>, // model aggregates only; no request content
    request_erasure: Option<(
        crate::resident_model::batch::public_warm_replay::NativeRequestResetReceipt,
        Option<crate::resident_mtp::NativeRequestResetReceipt>,
    )>,
}
impl<'m, 'f, 'd, 'df> Session<'m, 'f, 'd, 'df> {
    /// # Safety
    /// The authenticated backing, exact client/device/queue, exclusive MAIN/
    /// HOST/driver grants, no-swap/core protection and redacted panic hook from
    /// `execute` must remain valid on this thread until this session closes.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn load(
        file: &'f GgufFile,
        graph: BoundGraph<'m, 'f>,
        draft_graph: Option<&'d MtpGraph<'df>>,
        client: &ComputeClient<WgpuRuntime>,
        limits: &wgpu::Limits,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        backlog: cubecl_runtime::server::PublicTargetSubmissionBacklog,
        first: &Request,
        cancel: &AtomicBool,
        phase: &mut dyn FnMut(&'static str),
    ) -> Result<Self, Failure> {
        if !crate::strata_intel_core::enabled()
            || device.adapter_info().vendor != 0x8086
            || cancel.load(Ordering::Acquire)
            || crate::strata_port::profile_target()
            || !crate::strata_port::native_profile_authorized()
            || graph.verify_unchanged().is_err()
            || draft_graph.is_some_and(|d| {
                d.target_binding_id() != Some(graph.binding_id()) || d.verify_unchanged().is_err()
            })
        {
            return Err(Failure::Rejected);
        }
        let (prompt, outputs, _) = first.geometry().map_err(|_| Failure::Rejected)?;
        let mut session = Self {
            _thread: std::marker::PhantomData,
            file,
            graph,
            draft_graph,
            client: client.clone(),
            backlog,
            retired: Arc::new(AtomicBool::new(false)),
            runtime_owned: false,
            close_attempted: false,
            state: State::Loading,
            tables: None,
            executor: None,
            target: None,
            draft: None,
            policy: None,
            request_erasure: None,
        };
        let attempted = checked(|| unsafe {
            session.load_inner(limits, device, queue, prompt, outputs, phase)
        });
        if let Err(failure) = attempted {
            if session.runtime_owned {
                session.close()?;
            }
            return Err(failure);
        }
        session.state = State::Cold;
        Ok(session)
    }
    #[allow(clippy::too_many_arguments)]
    unsafe fn load_inner(
        &mut self,
        limits: &wgpu::Limits,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        prompt: usize,
        outputs: usize,
        phase: &mut dyn FnMut(&'static str),
    ) -> Result<(), Failure> {
        let file = self.file;
        let graph = self.graph;
        let draft_graph = self.draft_graph;
        let client = &self.client;
        let backlog = self.backlog;
        crate::target_lab_build::validate_runtime(true).map_err(|_| Failure::Rejected)?;
        crate::resident_util::select_retained_metadata(true);
        crate::resident_util::select_compute_scratch_clear(true);
        client
            .configure_public_target_request_io_with_backlog(backlog)
            .map_err(|_| Failure::Rejected)?;
        self.runtime_owned = true;
        self.tables = crate::strata_intel_core::Tables::new(client, Some(device))
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
        e.enable_transient_resident(self.retired.clone())
            .map_err(|_| Failure::Inference)?;
        e.fast_dense = true;
        e.fast_dense_fault = true;
        e.shared_kernels = true;
        e.shared_gdn = true;
        self.executor = Some(e);
        let e = self.executor.as_mut().ok_or(Failure::Inference)?;
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
        self.target = Some(
            ResidentModel::new_with_native_batch(
                graph,
                client.clone(),
                crate::strata_intel_core::CONTEXT,
                e,
                config,
                true,
            )
            .map_err(|error| {
                // Constructor diagnostics are compile-time labels only. Never
                // format a backend error, artifact value or request payload.
                phase("target_construction_error");
                phase(match error {
                    kuno_text::Error::InvalidFormat(label)
                    | kuno_text::Error::LimitExceeded(label) => label,
                    kuno_text::Error::Io(_) => "target_model_io",
                    kuno_text::Error::UnsupportedVersion(_) => "target_model_version",
                    kuno_text::Error::UnsupportedTensorType(_) => "target_tensor_type",
                    kuno_text::Error::ArithmeticOverflow => "target_size_overflow",
                    kuno_text::Error::OutOfBounds => "target_range",
                    kuno_text::Error::InvalidBufferLength => "target_buffer_length",
                    kuno_text::Error::FileChanged => "target_file_changed",
                });
                Failure::Inference
            })?,
        );
        let t = self.target.as_mut().ok_or(Failure::Inference)?;
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
            self.draft = Some(
                ResidentMtp::new_shared_strata_q2(g, client.clone(), limits, permit)
                    .map_err(|_| Failure::Inference)?,
            );
        }
        Ok(())
    }
    /// Compiled-public startup diagnostic only. The original loaded owner is
    /// still cold; no prompt, router, usage or speculative state is inspected.
    pub(crate) fn compiled_public_capacity(&self) -> Result<serde_json::Value, Failure> {
        if !crate::strata_prefill_profile::compiled_public_enabled()
            || self.state != State::Cold
            || !self.runtime_owned
            || self.close_attempted
        {
            return Err(Failure::Rejected);
        }
        self.executor
            .as_ref()
            .and_then(|executor| executor.strata_global_sources.as_ref())
            .ok_or(Failure::Inference)?
            .compiled_public_capacity(self.graph)
            .map_err(|_| Failure::Inference)
    }
    #[cfg(feature="strata-host-expert-blobs")]
    pub(crate) fn compiled_public_host_blob_probe(&self)->Result<serde_json::Value,Failure>{
        if !crate::strata_prefill_profile::compiled_public_enabled() || self.state!=State::Cold || !self.runtime_owned || self.close_attempted {return Err(Failure::Rejected);}
        self.executor.as_ref().and_then(|e|e.strata_global_sources.as_ref()).ok_or(Failure::Inference)?.compiled_public_host_blob_probe(self.graph).map_err(|_|Failure::Inference)
    }
    pub(crate) fn run(
        &mut self,
        request: Request,
        cancel: &AtomicBool,
        emit: &mut dyn FnMut(u32) -> kuno_text::Result<()>,
        phase: &mut dyn FnMut(&'static str),
    ) -> Result<controller::Completion, Failure> {
        let (prompt_count, output_count, _) = request.geometry().map_err(|_| Failure::Rejected)?;
        if cancel.load(Ordering::Acquire) {
            return Err(Failure::Rejected);
        }
        self.state.start(self.request_erasure.is_some())?;
        let attempted = checked(|| {
            phase("request_rebind_begin");
            let t = self.target.as_mut().ok_or(Failure::Inference)?;
            let e = self.executor.as_mut().ok_or(Failure::Inference)?;
            if let Some((erased, draft_erased)) = self.request_erasure.take() {
                match (draft_erased, self.draft.as_ref()) {
                    (Some(proof), Some(draft)) => proof
                        .check_original(draft)
                        .map_err(|_| Failure::RecoveryRequired)?,
                    (None, None) => (),
                    _ => return Err(Failure::RecoveryRequired),
                }
                t.bind_completed_native_request(e, erased, prompt_count, output_count, cancel)
                    .map_err(|_| Failure::RecoveryRequired)?;
            }
            phase("request_rebind_complete");
            let Request {
                prompt,
                outputs,
                eos,
                minimum,
                speculation,
                sampling,
            } = request;
            let mut delivered = Zeroizing::new(0usize);
            let done = crate::strata_ple_reader::with_request(self.graph, cancel, || {
                controller::run(
                    t,
                    e,
                    self.draft.as_mut(),
                    &mut self.policy,
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
                .map_err(|error| {
                    // Keep original fixed verifier labels before checked
                    // cleanup can replace the outer result with retirement.
                    // Arbitrary callback/backend error payloads stay redacted.
                    phase("controller_execution_error");
                    phase(match error.downcast_ref::<kuno_text::Error>() {
                        Some(
                            kuno_text::Error::InvalidFormat(label)
                            | kuno_text::Error::LimitExceeded(label),
                        ) => label,
                        Some(kuno_text::Error::InvalidBufferLength) => "controller_buffer_length",
                        Some(kuno_text::Error::OutOfBounds) => "controller_range",
                        Some(kuno_text::Error::FileChanged) => "controller_file_changed",
                        Some(_) => "controller_model_error",
                        None => "controller_state_or_callback_error",
                    });
                    kuno_text::Error::InvalidFormat("native controller execution failed")
                })
            })
            .map_err(|_| Failure::Inference)?;
            if done.outputs != *delivered {
                return Err(Failure::Inference);
            }
            if std::env::var_os("KUNO_CLAUDE_PIPE_TRACE").is_some() {
                eprintln!("public_decode_windows outputs={} windows={} repairs={} window={} routes_vram={} routes_host={}", done.outputs, done.windows, done.state_repairs, speculation.window, crate::strata_global_adapt::ROUTE_TRACE[0].load(std::sync::atomic::Ordering::Relaxed), crate::strata_global_adapt::ROUTE_TRACE[1].load(std::sync::atomic::Ordering::Relaxed));
                let ph: Vec<String> = controller::PHASE_NS.iter().map(|v| v.load(std::sync::atomic::Ordering::Relaxed).to_string()).collect();
                eprintln!("public_decode_phases {}", ph.join(" "));
            }
            // Generation is complete. Request state is erased separately by
            // `erase_completed`, so the caller can deliver DONE first; no new
            // request can start until that erasure has succeeded.
            phase("request_private_erasure");
            Ok(done)
        });
        match attempted {
            Ok(done) => {
                self.state.generated()?;
                Ok(done)
            }
            Err(failure) => {
                self.state = State::Broken;
                self.close()?;
                Err(failure)
            }
        }
    }
    /// Erase the completed request's private state, retaining the model owners.
    /// Must follow every successful `run` before the next `run`. A subsequent
    /// run consumes these original receipts to bind its new geometry before
    /// prefill; public weights stay owned.
    pub(crate) fn erase_completed(
        &mut self,
        cancel: &AtomicBool,
        phase: &mut dyn FnMut(&'static str),
    ) -> Result<(), Failure> {
        if self.state != State::Generated || self.request_erasure.is_some() {
            return Err(Failure::Rejected);
        }
        let attempted = checked(|| {
            let t = self.target.as_mut().ok_or(Failure::Inference)?;
            let e = self.executor.as_mut().ok_or(Failure::Inference)?;
            phase("draft_private_erasure");
            let draft_erasure = self
                .draft
                .as_mut()
                .map(|d| -> kuno_text::Result<_> {
                    let receipt = d.reset_completed_native_request(cancel)?;
                    receipt.check_original(d)?;
                    Ok(receipt)
                })
                .transpose()
                .map_err(|error| {
                    report_private_reset_error(error);
                    Failure::RecoveryRequired
                })?;
            phase("target_private_erasure");
            let target_erasure = t
                .reset_completed_native_request(e, cancel)
                .map_err(|error| {
                    report_private_reset_error(error);
                    Failure::RecoveryRequired
                })?;
            // Original retained ownership is distinct from a never-used PLE
            // generation0 constructor. Do not manufacture fresh-model authority.
            target_erasure
                .check_original(t)
                .map_err(|_| Failure::RecoveryRequired)?;
            self.request_erasure = Some((target_erasure, draft_erasure));
            phase("request_erasure_complete");
            Ok(())
        });
        match attempted {
            Ok(()) => self.state.completed(),
            Err(failure) => {
                self.state = State::Broken;
                self.close()?;
                Err(failure)
            }
        }
    }
    pub(crate) fn close(&mut self) -> Result<Closed, Failure> {
        if !self.runtime_owned || self.close_attempted {
            return Err(Failure::RecoveryRequired);
        }
        self.close_attempted = true;
        let result = checked(|| self.close_inner());
        self.state = if result.is_ok() {
            State::Closed
        } else {
            State::Broken
        };
        if result.is_err() {
            self.retired.store(true, Ordering::Release);
        }
        result
    }
    fn close_inner(&mut self) -> Result<Closed, Failure> {
        drop(self.policy.take()); // wipe model telemetry even if peer cleanup later fails
        // Never short-circuit peer cleanup after an inference/constructor failure.
        let mut draft_close = checked(|| {
            self.draft
                .as_mut()
                .map(|d| d.close())
                .transpose()
                .map_err(|_| Failure::RecoveryRequired)
        });
        let target_close = checked(|| {
            self.target
                .as_mut()
                .map(|t| t.close())
                .transpose()
                .map_err(|_| Failure::RecoveryRequired)
        });
        let target_audit = self
            .target
            .as_ref()
            .and_then(ResidentModel::public_owned_zero_audit_receipt);
        let target_drop = checked(|| {
            drop(self.target.take());
            Ok(())
        });
        let executor_close = checked(|| {
            self.executor
                .as_mut()
                .map(|e| e.close_transient_resident())
                .transpose()
                .map_err(|_| Failure::RecoveryRequired)
        });
        let executor_drop = checked(|| {
            drop(self.executor.take());
            Ok(())
        });
        let tables_drop = checked(|| {
            drop(self.tables.take());
            Ok(())
        });
        let physical = checked(|| {
            if let Some(d) = self.draft.as_ref() {
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
                executor::audit_native_context_recovery(&self.client, &self.retired)
                    .map_err(|_| Failure::RecoveryRequired)?;
                Ok(None)
            }
        });
        let draft_drop = checked(|| {
            drop(self.draft.take());
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
            || self.retired.load(Ordering::Acquire)
            || self.file.verify_file_unchanged().is_err()
            || self.graph.verify_unchanged().is_err()
            || self
                .draft_graph
                .is_some_and(|d| d.verify_unchanged().is_err())
            || self.client.public_target_request_io_configured().ok() != Some(true)
            || self.client.public_target_submission_backlog().ok() != Some(self.backlog)
        {
            self.retired.store(true, Ordering::Release);
            return Err(Failure::RecoveryRequired);
        }
        if let Some((retained, draft_erasure)) = self.request_erasure.take() {
            let (target_main, target_sources) = target_audit.ok_or(Failure::RecoveryRequired)?;
            // Retained erasure is not a close/refund receipt. Final close independently
            // audits the SAME target owners; source close counts include earlier resets.
            let final_draft = draft_close
                .as_ref()
                .map_err(|_| Failure::RecoveryRequired)?
                .as_ref()
                .map(|r| r.private);
            if !retained_erasure_agrees_with_close(
                (retained.main, retained.source),
                (target_main, target_sources),
                (draft_erasure.as_ref().map(|r| r.private), final_draft),
            ) {
                self.retired.store(true, Ordering::Release);
                return Err(Failure::RecoveryRequired);
            }
        }
        Ok(Closed {
            target: target_audit,
            draft_public_bytes: physical?,
        })
    }
}
impl Drop for Session<'_, '_, '_, '_> {
    fn drop(&mut self) {
        if self.runtime_owned && !self.close_attempted {
            let _ = checked(|| self.close());
        }
    }
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
    fn loaded_session_requires_matching_erasure_state_and_never_reopens_terminal_states() {
        let mut state = State::Cold;
        assert!(state.start(true).is_err());
        state.start(false).unwrap();
        assert!(state.start(false).is_err());
        assert!(state.start(true).is_err());
        assert!(state.completed().is_err(), "erasure requires finished generation");
        state.generated().unwrap();
        assert!(state.start(true).is_err(), "no new request before erasure");
        assert!(state.generated().is_err());
        state.completed().unwrap();
        for _ in 0..32 {
            assert!(state.start(false).is_err());
            state.start(true).unwrap();
            assert!(state.start(true).is_err());
            state.generated().unwrap();
            assert!(state.start(true).is_err());
            state.completed().unwrap();
        }
        for mut terminal in [State::Loading, State::Closed, State::Broken] {
            assert!(terminal.start(false).is_err());
            assert!(terminal.start(true).is_err());
            assert!(terminal.generated().is_err());
            assert!(terminal.completed().is_err());
        }
    }
    #[test]
    fn retained_audit_counts_do_not_replace_independent_close_or_cumulative_source_accounting() {
        use crate::private_arena::OwnedZeroAudit;
        let main = OwnedZeroAudit {
            bytes: 65536,
            buffers: 2,
        };
        let source = OwnedZeroAudit {
            bytes: 32768,
            buffers: 1,
        };
        let cumulative = OwnedZeroAudit {
            bytes: 98304,
            buffers: 3,
        };
        assert!(retained_erasure_agrees_with_close(
            (main, source),
            (main, cumulative),
            (Some(source), Some(source))
        ));
        assert!(retained_erasure_agrees_with_close(
            (main, source),
            (main, source),
            (None, None)
        ));
        assert!(!retained_erasure_agrees_with_close(
            (main, source),
            (source, cumulative),
            (None, None)
        ));
        assert!(!retained_erasure_agrees_with_close(
            (main, cumulative),
            (main, source),
            (None, None)
        ));
        assert!(!retained_erasure_agrees_with_close(
            (main, source),
            (main, source),
            (Some(source), None)
        ));
        assert!(!retained_erasure_agrees_with_close(
            (main, source),
            (main, source),
            (Some(source), Some(main))
        ));
    }
    #[test]
    fn owned_requests_accept_actual_varying_tokens_and_all_context_partitions() {
        if !crate::strata_intel_core::enabled() {
            return;
        }
        for prompt in [1, 17, 1024, 4096, 8193, 16384, MAX_OUTPUTS] {
            let outputs = crate::strata_intel_core::CONTEXT - CONTEXT_HEADROOM - prompt;
            let tokens = (0..prompt)
                .map(|i| (i * 13 % VOCAB) as u32)
                .collect::<Vec<_>>();
            let r = request(tokens, outputs).unwrap();
            assert_eq!(
                r.geometry().unwrap(),
                (prompt, outputs, 32768 - CONTEXT_HEADROOM)
            );
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
    fn selected_serve_headroom_rejects_short_reservations_and_keeps_full_windows_in_bounds() {
        if !crate::strata_intel_core::enabled() {
            return;
        }
        let context = crate::strata_intel_core::CONTEXT;
        // Literal pinned serving preflight oracle, independent from request_end.
        for prompt in [1, 17, 4096, 16384, context - 10, context - 9] {
            for slack in 0..=12 {
                let Some(outputs) = context.checked_sub(prompt + slack).filter(|&n| n > 0) else {
                    continue;
                };
                let expected = prompt + outputs + 8 <= context;
                assert_eq!(request_end(prompt, outputs).is_some(), expected);
                assert_eq!(request(vec![0; prompt], outputs).is_ok(), expected);
                if expected {
                    // The latest continuing round can feed its entire target
                    // window and next MTP producer, even with only one output
                    // left to deliver. Do not clip either at the delivery cap.
                    let latest_input = prompt + outputs - 2;
                    for width in 1..=crate::strata_port::verify_capacity() {
                        assert!(latest_input + width <= context);
                        let accepted_next = latest_input + 1;
                        assert!(accepted_next + width - 1 <= context);
                    }
                }
            }
        }
        assert!(request_end(usize::MAX, 1).is_none());
        assert!(request_end(1, usize::MAX).is_none());
        assert!(request_end(usize::MAX - 9, 2).is_none());
        assert_eq!(MAX_OUTPUTS + 1 + CONTEXT_HEADROOM, context);
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
