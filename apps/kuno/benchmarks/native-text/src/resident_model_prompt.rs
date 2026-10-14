//! Connected layer-major large prompt controller. MAX8 speculative receipts
//! remain separate; these tickets publish only after the full model fence.
use super::*;
use crate::strata_prompt as p;
use cubecl::server::Handle;

// Embedding upload and runtime copies, final logits/carry, PLE tail/history,
// all-family tickets and bounded argument tables coexist during acceptance.
// Native recordings retain dual-queue plans across prompt and decode. Reserve
// their bounded cache/temporary metadata in this same core lifetime quote.
pub(super) const HOST_BYTES: usize = 8 * 1024 * 1024 + cubecl::wgpu::NATIVE_QUEUE_HOST_BYTES;
pub(super) const RUNTIME_BYTES: u64 = 8 * 1024 * 1024;
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
    /// Full target chunk plus device-only MTP teacher forcing. The last chunk
    /// alone produces a head and one host carry for the subsequent decode loop.
    pub(crate) fn prefill_prompt_full(
        &mut self,
        executor: &mut GpuExecutor<'f>,
        tokens: &[u32],
        teacher: &[u32],
        mut draft: Option<&mut crate::resident_mtp::ResidentMtp<'_, '_>>,
        final_chunk: bool,
        cancel: &AtomicBool,
    ) -> Result<Option<(u32, Zeroizing<Vec<f32>>)>> {
        let mut native = self.native_batch.take().ok_or_else(error)?;
        let mut flight = Flight {
            poison: native.poison.clone(),
            accepted: false,
        };
        let attempted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _scope = p::Scope::new(&self.arena, native.core_prompt_stages.ok_or_else(error)?)?;
            let start = self.position;
            let result = native.run_prompt_full(self, executor, tokens, final_chunk, cancel)?;
            if let Some(draft) = draft.as_deref_mut() {
                if teacher.len() > tokens.len() || draft.position() != start {
                    return Err(error());
                }
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
            u::cancel(cancel)?;
            Ok(result)
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
    // 96 GR writes preserve the incoming R0 parity, with one additional PLE
    // write after layer0. Every chunk therefore ends in R1.
    fn prompt_final_residual(&self, count: usize) -> Result<usize> {
        if !(1..=p::groups::CAPACITY).contains(&count) {
            return Err(error());
        }
        Ok(1)
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
        final_chunk: bool,
        cancel: &AtomicBool,
    ) -> Result<Option<(u32, Zeroizing<Vec<f32>>)>> {
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
            m.arena.client().map_err(u::fault)?,
            tokens,
            v[11].clone(),
            v[0].clone(),
            fault.clone(),
            cancel,
        )?;
        self.global_moe
            .as_mut()
            .ok_or_else(error)?
            .begin(m.graph, &m.arena, count)?;
        let mut stages = Vec::with_capacity(LAYERS);
        let mut ple = None;
        let mut current = 0;
        let mut previous: Option<(Handle, Handle)> = None;
        for layer in 0..LAYERS {
            u::cancel(cancel)?;
            if layer == 1 {
                if let Some((source, block)) = previous.take() {
                    p::write_handles(
                        m.arena.client().map_err(u::fault)?,
                        [
                            source,
                            block,
                            v[5].clone(),
                            v[current].clone(),
                            fault.clone(),
                        ],
                        count,
                    )?;
                }
                ple = Some(self.ple.enqueue_prompt_full(
                    m,
                    e,
                    v[current].clone(),
                    v[1 - current].clone(),
                    start,
                    tokens,
                    cancel,
                )?);
                current = 1 - current;
            }
            Self::prompt_gr(
                m,
                e,
                &format!("blk.{layer}.hc_attn"),
                v[current].clone(),
                previous.take(),
                v[5].clone(),
                v[2].clone(),
                count,
                0,
                true,
            )?;
            stages.push(match &mut m.layers[layer] {
                Layer::Gdn(stage, public) => Stage::Gdn(stage.enqueue_prompt_full(
                    m.graph,
                    &m.arena,
                    e,
                    public,
                    v[2].clone(),
                    v[3].clone(),
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
                    v[2].clone(),
                    v[3].clone(),
                    start,
                    count,
                    m.b.fault,
                    self.poison.clone(),
                    cancel,
                )?),
            });
            previous = Some((v[current].clone(), v[3].clone()));
            current = 1 - current;
            Self::prompt_gr(
                m,
                e,
                &format!("blk.{layer}.hc_ffn"),
                v[current].clone(),
                previous.take(),
                v[5].clone(),
                v[2].clone(),
                count,
                0,
                true,
            )?;
            let tensor = m
                .graph
                .tensor(&format!("blk.{layer}.ffn_gate_inp.weight"))?;
            e.enqueue_strata_prompt_projection(
                tensor,
                &v[2],
                &v[6],
                &fault,
                &BatchSpans::new(count, 0, 0, D, 0, 512),
                cancel,
            )?;
            let sources = e.strata_global_sources.as_ref().ok_or_else(error)?.clone();
            self.global_moe.as_mut().ok_or_else(error)?.enqueue_prompt(
                m.graph,
                &m.arena,
                &sources,
                &m.admission,
                layer,
                [v[2].clone(), v[3].clone(), v[6].clone()],
                m.b.fault,
                cancel,
            )?;
            for (name, index, width) in [
                ("ffn_gate_shexp.weight", 7, 640),
                ("ffn_up_shexp.weight", 8, 640),
                ("ffn_gate_inp_shexp.weight", 10, 1),
            ] {
                let tensor = m.graph.tensor(&format!("blk.{layer}.{name}"))?;
                e.enqueue_strata_prompt_projection(
                    tensor,
                    &v[2],
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
                    v[7].clone(),
                    v[8].clone(),
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
                &v[9],
                &fault,
                &BatchSpans::new(count, 0, 0, 640, 0, D),
                cancel,
            )?;
            self.global_moe.as_mut().ok_or_else(error)?.finish_prompt(
                &m.arena,
                layer,
                v[3].clone(),
                m.b.fault,
            )?;
            p::launch(
                m.arena.client().map_err(u::fault)?,
                p::Kernel::SharedMerge,
                [(count * D).div_ceil(256), 1, 1],
                vec![
                    v[3].clone(),
                    v[9].clone(),
                    v[10].clone(),
                    v[4].clone(),
                    fault.clone(),
                ],
                info,
            )?;
            previous = Some((v[current].clone(), v[4].clone()));
            current = 1 - current;
        }
        if let Some((source, block)) = previous.take() {
            p::write_handles(
                m.arena.client().map_err(u::fault)?,
                [
                    source,
                    block,
                    v[5].clone(),
                    v[current].clone(),
                    fault.clone(),
                ],
                count,
            )?;
        }
        if current != self.prompt_final_residual(count)? {
            return Err(error());
        }
        if final_chunk {
            Self::prompt_gr(
                m,
                e,
                "output_hc",
                v[current].clone(),
                None,
                v[5].clone(),
                v[2].clone(),
                1,
                (count - 1) * HC * D,
                false,
            )?;
            e.enqueue_strata_prompt_projection(
                m.graph.tensor("output.weight")?,
                &v[2],
                &u::handle(&m.arena, m.b.logits)?,
                &fault,
                &BatchSpans::new(1, 0, 0, D, 0, 248320),
                cancel,
            )?;
        }
        // First completed whole-model proof precedes every state readback.
        batch_io::read_fault(&m.arena, &m.admission, m.b.fault, cancel)?;
        let ple = ple.ok_or_else(error)?;
        self.ple.validate_prompt_full(m, &ple, start, count)?;
        let history = self.ple.prompt_readback(m, &ple, cancel)?;
        let output = if final_chunk {
            let logits = batch_io::read_last_logits(
                m.graph,
                &m.arena,
                &m.admission,
                m.b.logits,
                m.b.logit_packet,
                m.b.fault,
                cancel,
            )?;
            let mut best = 0;
            for i in 1..logits.len() {
                if logits[i] > logits[best] {
                    best = i;
                }
            }
            let carry = m.admission.readback_audited(cancel, || {
                let row = v[current]
                    .clone()
                    .offset_start(((count - 1) * HC * D * 4) as u64)
                    .offset_end(((p::groups::CAPACITY - count) * HC * D * 4) as u64);
                struct Lease(cubecl::bytes::Bytes);
                impl Drop for Lease {
                    fn drop(&mut self) {
                        self.0.zeroize();
                    }
                }
                let bytes = Lease(
                    m.arena
                        .client()
                        .map_err(u::fault)?
                        .read_one(row)
                        .map_err(u::fault)?,
                );
                if bytes.0.len() != HC * D * 4 {
                    return Err(error());
                }
                let mut carry = Zeroizing::new(Vec::with_capacity(HC * D));
                for word in bytes.0.chunks_exact(4) {
                    let x = f32::from_le_bytes(word.try_into().map_err(|_| error())?);
                    if !x.is_finite() {
                        return Err(error());
                    }
                    carry.push(x);
                }
                Ok(carry)
            })?;
            Some((best as u32, carry))
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
            .prepare_prompt_host(m, &ple, &history.tail, &history.next, cancel)?;
        host.preflight(cancel).map_err(u::fault)?;
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
        Ok(output)
    }
}
