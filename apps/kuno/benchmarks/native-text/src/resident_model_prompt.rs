//! Connected layer-major large prompt controller. MAX8 speculative receipts
//! remain separate; these tickets publish only after the full model fence.
use super::*;
use crate::strata_prompt as p;
use cubecl::server::Handle;

// Embedding upload and runtime copies, final logits/carry, PLE tail/history,
// all-family tickets and bounded argument tables coexist during acceptance.
// Native recordings retain dual-queue plans across prompt and decode. Reserve
// their bounded cache/temporary metadata in this same core lifetime quote.
pub(super) const HOST_BYTES: usize = 128 * 1024 * 1024
    + cubecl::wgpu::NATIVE_QUEUE_HOST_BYTES
    + if cfg!(feature = "strata-native-expert-bulk") {
        // One serialized complete expert batch; charged in PROMPT_SOURCE_HOST
        // and the parent host-reservation quote before model/payload creation.
        cubecl_runtime::server::native_expert_batch::NATIVE_EXPERT_BATCH_HOST_BYTES
    } else {
        0
    };
pub(super) const RUNTIME_BYTES: u64 = 128 * 1024 * 1024;
/// Constructed here only after real completed model/fault/I/O proofs. The
/// MAX8 workspace can record that boundary without fabricating issued rows.
pub(crate) struct Completed {
    binding: kuno_text::model::GraphBindingId,
    poison: Arc<AtomicBool>,
}
impl Completed {
    pub(crate) fn check(
        &self,
        binding: kuno_text::model::GraphBindingId,
        poison: &Arc<AtomicBool>,
    ) -> Result<()> {
        if self.binding != binding
            || !Arc::ptr_eq(&self.poison, poison)
            || poison.load(Ordering::Acquire)
        {
            return Err(error());
        }
        Ok(())
    }
}
enum Stage {
    Gdn(crate::resident_gdn::prompt::Pending),
    Qsa(crate::resident_qsa::prompt::Pending),
}

impl<'m, 'f> ResidentModel<'m, 'f> {
    pub(crate) fn begin_prompt_cache_loan(
        &self,
        executor: &mut GpuExecutor<'f>,
    ) -> Result<Option<crate::strata_global_sources::BorrowedPrompt>> {
        let native = self.native_batch.as_ref().ok_or_else(error)?;
        if native.core_prompt_stages.is_some() {
            return Ok(None);
        }
        if native
            .global_moe
            .as_ref()
            .ok_or_else(error)?
            .prompt_secondary_raw_id()
            .is_some()
        {
            return Err(error());
        }
        if self.poisoned || self.retired.load(Ordering::Acquire) || !self.admission.transient_io() {
            return Err(error());
        }
        let sources = executor
            .strata_global_sources
            .as_mut()
            .and_then(Arc::get_mut)
            .ok_or_else(error)?;
        // Protected model construction owns the complete cache/RAM grants;
        // request execution is serialized on this worker through finish.
        Ok(Some(unsafe { sources.begin_prompt_loan(self.graph)? }))
    }
    pub(crate) fn prompt_scope(
        &self,
        loan: Option<&crate::strata_global_sources::BorrowedPrompt>,
    ) -> Result<p::Scope> {
        let native = self.native_batch.as_ref().ok_or_else(error)?;
        let secondary = native
            .global_moe
            .as_ref()
            .ok_or_else(error)?
            .prompt_secondary_raw_id();
        match (native.core_prompt_stages, loan, secondary) {
            (Some(ids), None, secondary) => {
                p::Scope::new_with_secondary_raw_bank(&self.arena, ids, secondary)
            }
            (None, Some(loan), None) => loan.scope(),
            _ => Err(error()),
        }
    }
    pub(crate) fn finish_prompt_cache_loan(
        &self,
        executor: &mut GpuExecutor<'f>,
        loan: Option<crate::strata_global_sources::BorrowedPrompt>,
    ) -> Result<()> {
        if let Some(loan) = loan {
            executor
                .strata_global_sources
                .as_mut()
                .and_then(Arc::get_mut)
                .ok_or_else(error)?
                .finish_prompt_loan(self.graph, loan)?;
        }
        Ok(())
    }

    /// Batched prompt prefix plus device-only MTP teacher forcing. This path
    /// never runs a vocabulary head; the final prompt input uses count1 decode.
    pub(crate) fn prefill_prompt_chunk(
        &mut self,
        executor: &mut GpuExecutor<'f>,
        tokens: &[u32],
        teacher: &[u32],
        mut draft: Option<&mut crate::resident_mtp::ResidentMtp<'_, '_>>,
        cancel: &AtomicBool,
    ) -> Result<()> {
        if teacher.len() != tokens.len()
            || draft
                .as_ref()
                .is_some_and(|d| d.position() != self.position)
        {
            return Err(error());
        }
        if let Some(rows) = self
            .native_batch
            .as_ref()
            .and_then(|n| n.target_rows.as_ref())
        {
            rows.check_no_live_output()?;
            rows.invalidate_carry()?;
        }
        let mut native = self.native_batch.take().ok_or_else(error)?;
        let mut flight = Flight {
            poison: native.poison.clone(),
            accepted: false,
        };
        let attempted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if !p::active() {
                return Err(error());
            }
            native.run_prompt_full(self, executor, tokens, cancel)?;
            let mut timing =
                crate::strata_prefill_profile::Timer::new(self.arena.client().map_err(u::fault)?)?;
            if let Some(draft) = draft.as_deref_mut() {
                let carry =
                    p::outer_workspace()?[native.prompt_final_residual(tokens.len())?].clone();
                if !teacher.is_empty() {
                    draft.append_prompt_device(
                        teacher,
                        self.arena.client().map_err(u::fault)?,
                        carry,
                        cancel,
                    )?;
                }
            }
            timing.mark("mtp_teacher")?;
            u::cancel(cancel)?;
            Ok(())
        }));
        self.native_batch = Some(native);
        match attempted {
            Ok(Ok(output)) => {
                flight.accepted = true;
                Ok(output)
            }
            failed => {
                flight.poison.store(true, Ordering::Release);
                self.poisoned = true;
                executor.poison_resident();
                let _ = self.close();
                let _ = executor.close_transient_resident();
                Err(match failed {
                    Ok(Err(e)) => e,
                    _ => error(),
                })
            }
        }
    }
}
impl Workspace {
    // Native prompt GR and PLE update the sole original target R in place.
    // MTP teaching preserves this carry and uses the idle PLE key for scratch.
    fn prompt_final_residual(&self, count: usize) -> Result<usize> {
        if !(1..=p::groups::CAPACITY).contains(&count) {
            return Err(error());
        }
        Ok(0)
    }
    #[allow(clippy::too_many_arguments)]
    fn prompt_gr(
        m: &mut ResidentModel<'_, '_>,
        e: &mut GpuExecutor<'_>,
        base: &str,
        residual: Handle,
        previous: Option<(Handle, Handle)>,
        injection: Handle,
        mixed: Handle,
        count: usize,
        first: usize,
        inject: bool,
    ) -> Result<()> {
        let gamma = m.public.vector(m.graph, &format!("{base}_norm.weight"))?;
        let weights = e.strata_gr_weights(
            [
                m.graph.tensor(&format!("{base}_down.weight"))?,
                m.graph.tensor(&format!("{base}_up.weight"))?,
            ],
            if inject {
                Some(m.graph.tensor(&format!("{base}_inject.weight"))?)
            } else {
                None
            },
        )?;
        p::gr_handles(
            p::GrHandles {
                client: m.arena.client().map_err(u::fault)?,
                residual,
                previous,
                gamma,
                injection,
                mixed,
                fault: u::handle(&m.arena, m.b.fault)?,
                count,
                first,
                epsilon: m.graph.geometry().epsilon,
                inject,
            },
            weights,
        )
    }
    fn run_prompt_full(
        &mut self,
        m: &mut ResidentModel<'_, '_>,
        e: &mut GpuExecutor<'_>,
        tokens: &[u32],
        cancel: &AtomicBool,
    ) -> Result<()> {
        let mut timing =
            crate::strata_prefill_profile::Timer::new(m.arena.client().map_err(u::fault)?)?;
        self.prove_outer(m)?;
        self.validate_global_scope(e, &m.admission)?;
        let start = m.position;
        let count = tokens.len();
        let end = start.checked_add(count).ok_or(Error::ArithmeticOverflow)?;
        if !p::active()
            || !(1..=p::groups::CAPACITY).contains(&count)
            || end > m.context
            || m.poisoned
            || m.retired.load(Ordering::Acquire)
            || self.poison.load(Ordering::Acquire)
            || m.layers.len() != LAYERS
            || !m.pending_qsa.is_empty()
            || !m.pending_moe.is_empty()
            || m.ple_batch_pending.is_some()
            || tokens.iter().any(|&t| t >= 248320)
        {
            return Err(error());
        }
        self.public_language_request_io
            .as_ref()
            .ok_or_else(error)?
            .validate_prompt_chunk(
                m.graph.binding_id(),
                m.context,
                start,
                count,
                m.admission.request_io_reuse_active(),
            )?;
        if let Some(rows) = &self.target_rows {
            rows.check_no_live_output()?;
        }
        self.gr.validate_all_consumers_queued()?;
        self.shared.validate_all_consumers_queued()?;
        self.global_moe
            .as_ref()
            .ok_or_else(error)?
            .validate_idle()?;
        e.check_client(m.arena.client().map_err(u::fault)?)?;
        e.check_resident_admission(&m.admission)?;
        m.graph.verify_unchanged()?;
        u::cancel(cancel)?;
        u::zero_many(&m.arena, &[m.b.fault, m.b.logits, m.b.logit_packet])?;
        let v = p::outer_workspace()?;
        let fault = u::handle(&m.arena, m.b.fault)?;
        p::embedding(
            m.graph,
            &m.arena,
            &m.admission,
            m.b.logit_packet,
            m.b.fault,
            tokens,
            v[10].clone(),
            v[0].clone(),
            cancel,
        )?;
        self.global_moe
            .as_mut()
            .ok_or_else(error)?
            .begin(m.graph, &m.arena, count)?;
        let mut stages = Vec::with_capacity(LAYERS);
        let mut ple = None;
        let mut previous: Option<(Handle, Handle)> = None;
        timing.mark("setup_embedding")?;
        for layer in 0..LAYERS {
            u::cancel(cancel)?;
            let layer_trace = crate::strata_prefill_profile::FullHostLayer::new(start,count,layer);
            if layer == 1 {
                if let Some((source, block)) = previous.take() {
                    p::write_handles(
                        m.arena.client().map_err(u::fault)?,
                        [source, block, v[4].clone(), v[0].clone(), fault.clone()],
                        count,
                    )?;
                }
                ple = Some(self.ple.enqueue_prompt_full(
                    m,
                    e,
                    v[0].clone(),
                    v[0].clone(),
                    start,
                    tokens,
                    cancel,
                )?);
            }
            let gr_sample = crate::strata_prefill_profile::GrSample::new(start, count, layer);
            Self::prompt_gr(
                m,
                e,
                &format!("blk.{layer}.hc_attn"),
                v[0].clone(),
                previous.take(),
                v[4].clone(),
                v[1].clone(),
                count,
                0,
                true,
            )?;
            drop(gr_sample);
            timing.mark("gr_attention_input")?;
            stages.push(match &mut m.layers[layer] {
                Layer::Gdn(stage, public) => Stage::Gdn(stage.enqueue_prompt_full(
                    m.graph,
                    &m.arena,
                    e,
                    public,
                    v[1].clone(),
                    v[2].clone(),
                    start,
                    count,
                    m.b.fault,
                    self.poison.clone(),
                    cancel,
                )?),
                Layer::Qsa(stage, public) => Stage::Qsa(stage.enqueue_prompt_full(
                    m.graph,
                    &m.arena,
                    e,
                    public,
                    v[1].clone(),
                    v[2].clone(),
                    start,
                    count,
                    m.b.fault,
                    self.poison.clone(),
                    cancel,
                )?),
            });
            timing.mark("attention")?;
            previous = Some((v[0].clone(), v[2].clone()));
            Self::prompt_gr(
                m,
                e,
                &format!("blk.{layer}.hc_ffn"),
                v[0].clone(),
                previous.take(),
                v[4].clone(),
                v[1].clone(),
                count,
                0,
                true,
            )?;
            let tensor = m
                .graph
                .tensor(&format!("blk.{layer}.ffn_gate_inp.weight"))?;
            e.enqueue_strata_prompt_projection(
                tensor,
                &v[1],
                &v[5],
                &fault,
                &BatchSpans::new(count, 0, 0, D, 0, 512),
                cancel,
            )?;
            timing.mark("gr_router")?;
            let sources = e.strata_global_sources.as_ref().ok_or_else(error)?.clone();
            let native_sample =
                crate::strata_prefill_profile::NativeExpertLayer::new(start, count, layer);
            self.global_moe.as_mut().ok_or_else(error)?.enqueue_prompt(
                m.graph,
                &m.arena,
                &sources,
                &m.admission,
                layer,
                [v[1].clone(), v[2].clone(), v[5].clone()],
                m.b.fault,
                cancel,
            )?;
            timing.mark("experts")?;
            native_sample.finish();
            for (name, index, width) in [
                ("ffn_gate_shexp.weight", 6, 640),
                ("ffn_up_shexp.weight", 7, 640),
                ("ffn_gate_inp_shexp.weight", 9, 1),
            ] {
                let tensor = m.graph.tensor(&format!("blk.{layer}.{name}"))?;
                e.enqueue_strata_prompt_projection(
                    tensor,
                    &v[1],
                    &v[index],
                    &fault,
                    &BatchSpans::new(count, 0, 0, D, 0, width),
                    cancel,
                )?;
            }
            let mut info = [0; 16];
            info[0] = count as u32;
            p::launch(
                m.arena.client().map_err(u::fault)?,
                p::Kernel::SharedPair,
                [(count * 320).div_ceil(256), 1, 1],
                vec![
                    v[6].clone(),
                    v[7].clone(),
                    p::activation_stage()?,
                    fault.clone(),
                ],
                info,
            )?;
            let tensor = m
                .graph
                .tensor(&format!("blk.{layer}.ffn_down_shexp.weight"))?;
            e.enqueue_strata_prompt_prepacked_projection(
                tensor,
                &p::activation_stage()?,
                &v[8],
                &fault,
                &BatchSpans::new(count, 0, 0, 640, 0, D),
                cancel,
            )?;
            #[cfg(feature = "strata-expert-ordered-output")]
            self.global_moe.as_mut().ok_or_else(error)?.finish_prompt_ordered_shared(
                &m.arena, layer, v[2].clone(), v[8].clone(), v[9].clone(),
                v[3].clone(), m.b.fault,
            )?;
            #[cfg(not(feature = "strata-expert-ordered-output"))]
            {
            self.global_moe.as_mut().ok_or_else(error)?.finish_prompt(
                &m.arena,
                layer,
                v[2].clone(),
                m.b.fault,
            )?;
            p::launch(
                m.arena.client().map_err(u::fault)?,
                p::Kernel::SharedMerge,
                [(count * D).div_ceil(256), 1, 1],
                vec![
                    v[2].clone(),
                    v[8].clone(),
                    v[9].clone(),
                    v[3].clone(),
                    fault.clone(),
                ],
                info,
            )?;
            }
            timing.mark("shared_expert_merge")?;
            previous = Some((v[0].clone(), v[3].clone()));
            layer_trace.finish();
        }
        if let Some((source, block)) = previous.take() {
            p::write_handles(
                m.arena.client().map_err(u::fault)?,
                [source, block, v[4].clone(), v[0].clone(), fault.clone()],
                count,
            )?;
        }
        if self.prompt_final_residual(count)? != 0 {
            return Err(error());
        }
        // First completed whole-model proof precedes every state readback.
        batch_io::read_fault(&m.arena, &m.admission, m.b.fault, cancel)?;
        let ple = ple.ok_or_else(error)?;
        self.ple.validate_prompt_full(m, &ple, start, count)?;
        let history = if !crate::strata_intel_core::enabled() || self.ple.public_grouped_reads() {
            Some(self.ple.prompt_readback(m, &ple, cancel)?)
        } else {
            None
        };
        batch_io::read_fault(&m.arena, &m.admission, m.b.fault, cancel)?;
        if stages.len() != m.layers.len() {
            return Err(error());
        }
        for (owner, ticket) in m.layers.iter().zip(&stages) {
            match (owner, ticket) {
                (Layer::Gdn(stage, _), Stage::Gdn(p)) => {
                    stage.validate_prompt_full(p, start, count, m.b.fault, &self.poison)?
                }
                (Layer::Qsa(stage, _), Stage::Qsa(p)) => {
                    stage.validate_prompt_full(p, start, count, m.b.fault, &self.poison)?
                }
                _ => return Err(error()),
            }
        }
        self.global_moe
            .as_mut()
            .ok_or_else(error)?
            .accept(&m.arena, &m.admission, cancel)?;
        m.graph.verify_unchanged()?;
        u::cancel(cancel)?;
        if m.retired.load(Ordering::Acquire)
            || self.poison.load(Ordering::Acquire)
            || m.position != start
        {
            return Err(error());
        }
        let completed = Completed {
            binding: m.graph.binding_id(),
            poison: self.poison.clone(),
        };
        let gdn_boundary = self
            .gdn
            .prepare_prompt_boundary(&m.arena, &completed, cancel)?;
        let qsa_boundary = self
            .qsa
            .prepare_prompt_boundary(&m.arena, &completed, cancel)?;
        let host = self
            .ple
            .prepare_prompt_host(m, &ple, history.as_ref(), cancel)?;
        host.preflight(cancel)?;
        // Global finish is still fallible, so execute it before publishing any
        // family. All subsequent state operations are infallible and exclusive.
        self.global_moe.as_mut().ok_or_else(error)?.finish()?;
        host.commit();
        for (owner, ticket) in m.layers.iter_mut().zip(stages) {
            match (owner, ticket) {
                (Layer::Gdn(stage, _), Stage::Gdn(p)) => stage.commit_prompt_full(p),
                (Layer::Qsa(stage, _), Stage::Qsa(p)) => stage.commit_prompt_full(p),
                _ => unreachable!("preflighted prompt stage family"),
            }
        }
        self.ple.commit_prompt_full(ple);
        gdn_boundary.commit();
        qsa_boundary.commit();
        m.position = end;
        timing.mark("boundary_commit")?;
        Ok(())
    }
}
