//! Native large prompt PLE. Intermediates overlay attention scratch; only the
//! final nine normalized rows remain in the original GPU history. Explicit
//! diagnostics may read them; native publication advances only CPU token IDs.
use super::*;
use crate::executor::GpuExecutor;
pub(crate) struct HistoryReadback {
    pub(crate) tail: Zeroizing<Vec<f32>>,
    pub(crate) next: Zeroizing<Vec<f32>>,
}
pub(crate) struct Pending {
    binding: Option<GraphBindingId>,
    stamp: Zeroizing<[usize; 3]>,
    tokens: Zeroizing<Vec<u32>>,
    old_tokens: Zeroizing<[i32; 2]>,
    roots: [BufferId; 3],
    poison: Arc<AtomicBool>,
    armed: bool,
}
impl Drop for Pending {
    fn drop(&mut self) {
        if self.armed {
            self.poison.store(true, Ordering::Release);
        }
        self.roots.zeroize();
        self.binding = None;
    }
}
impl PleBatchWorkspace {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn enqueue_prompt_full(
        &mut self,
        model: &mut ResidentModel<'_, '_>,
        executor: &mut GpuExecutor<'_>,
        input: Handle,
        output: Handle,
        start: usize,
        tokens: &[u32],
        cancel: &AtomicBool,
    ) -> Result<Pending> {
        graph_preflight(model.graph)?;
        u::cancel(cancel)?;
        let count = tokens.len();
        if !crate::strata_prompt::active()
            || !(1..=crate::strata_prompt::groups::CAPACITY).contains(&count)
            || start
                .checked_add(count)
                .is_none_or(|end| end > model.context)
            || start != self.next
            || start != model.position
            || self.pending
            || model.ple_batch_pending.is_some()
            || model.poisoned
            || self.poison.load(Ordering::Acquire)
            || tokens.iter().any(|&t| t >= 248320)
            || self.binding != model.graph.binding_id()
        {
            return Err(error());
        }
        executor.check_client(model.arena.client().map_err(u::fault)?)?;
        crate::strata_prompt::validate_output(&input)?;
        crate::strata_prompt::validate_output(&output)?;
        if input.size_in_used() < (count * CHANNELS * 4) as u64
            || output.size_in_used() < (count * CHANNELS * 4) as u64
        {
            return Err(error());
        }
        let roots = [self.ids[OLD], self.ids[NEXT], model.b.fault];
        let physical = roots
            .into_iter()
            .zip([CHANNELS * HISTORY, CHANNELS * HISTORY, 1])
            .map(|(id, n)| full_private(&model.arena, id, n))
            .collect::<Result<Vec<_>>>()?;
        crate::resident_gr::batch::physical(&model.arena, &physical)?;
        let generation = self
            .generation
            .checked_add(1)
            .ok_or(Error::ArithmeticOverflow)?;
        self.pending = true;
        self.generation = generation;
        let old_tokens =
            Zeroizing::new([model.ple_history.tokens()[0], model.ple_history.tokens()[1]]);
        let pending = Pending {
            binding: Some(self.binding),
            stamp: Zeroizing::new([generation as usize, start, count]),
            tokens: Zeroizing::new(tokens.to_vec()),
            old_tokens,
            roots,
            poison: self.poison.clone(),
            armed: true,
        };
        let mut rows = Zeroizing::new(vec![[0u32; 16]; count]);
        let mut previous = Zeroizing::new(*pending.old_tokens);
        for (t, &token) in tokens.iter().enumerate() {
            crate::ple_chunk::rows_into(
                model.graph.ple_hash().metadata(),
                token,
                &previous,
                &mut rows[t],
            )
            .map_err(u::fault)?;
            previous[0] = previous[1];
            previous[1] = token as i32;
        }
        let mut data = if let Some(data) = prompt_source::take(model.graph, &rows, start)? {
            data
        } else {
            native_embedding(
                model.graph,
                model.graph.tensor("per_layer_token_embd.weight")?,
                self.profile,
                &rows,
                cancel,
                crate::strata_ple_reader::current().as_ref(),
            )?
        };
        let v = crate::strata_prompt::ple_workspace()?;
        data.truncate(count * WIDTH * 4);
        if data.len() != count * WIDTH * 4 {
            return Err(error());
        }
        let client = model.arena.client().map_err(u::fault)?;
        let fault = u::handle(&model.arena, model.b.fault)?;
        crate::strata_prompt::stage_private_view(
            &model.arena,
            model.b.logit_packet,
            v[0].clone(),
            &data,
        )?;
        // GPU history is authoritative across full prompt chunks. At the first
        // chunk both original owners must still describe the empty sequence.
        if start == 0 && model.ple_conv.history().iter().any(|x| x.to_bits() != 0) {
            return Err(error());
        }
        for (name, target, width) in [
            ("blk.1.ple_key.weight", 1, CHANNELS),
            ("blk.1.ple_value.weight", 2, WIDTH),
        ] {
            let tensor = model.graph.tensor(name)?;
            let m = Matrix::new(tensor)?;
            if m.inputs() != WIDTH || m.outputs() != width || m.experts() != 1 {
                return Err(error());
            }
            executor.enqueue_strata_prompt_projection(
                tensor,
                &v[0],
                &v[target],
                &fault,
                &BatchSpans::new(count, 0, 0, WIDTH, 0, width),
                cancel,
            )?;
        }
        use crate::strata_ple as n;
        let info = |is: usize, os: usize, ss: usize, eps: f32| {
            [
                count as u32,
                0,
                is as u32,
                0,
                os as u32,
                ss as u32,
                0,
                eps.to_bits(),
                0,
                0,
                0,
                0,
            ]
        };
        // Upstream normalizes key in place. The single writable descriptor
        // retains that exact original projection owner through its gate reader.
        n::launch(
            &model.arena,
            n::Kernel::PromptNormInPlace,
            [4, count, 1],
            vec![v[1].clone(), self.public.gamma[0].clone(), v[6].clone()],
            info(CHANNELS, CHANNELS, 4, 1e-6),
        )?;
        crate::strata_prompt::head_flags(client, v[6].clone(), fault.clone(), count, 4)?;
        n::launch(
            &model.arena,
            n::Kernel::PromptNorm,
            [4, count, 1],
            vec![
                input.clone(),
                self.public.gamma[1].clone(),
                v[3].clone(),
                v[7].clone(),
            ],
            info(CHANNELS, CHANNELS, 4, 1e-6),
        )?;
        crate::strata_prompt::head_flags(client, v[7].clone(), fault.clone(), count, 4)?;
        n::launch(
            &model.arena,
            n::Kernel::PromptGate,
            [4, count, 1],
            vec![v[1].clone(), v[3].clone(), v[4].clone(), v[8].clone()],
            info(CHANNELS, 4, 4, (WIDTH as f32).sqrt().recip()),
        )?;
        crate::strata_prompt::head_flags(client, v[8].clone(), fault.clone(), count, 4)?;
        n::launch(
            &model.arena,
            n::Kernel::PromptBroadcast,
            [40, count, 1],
            vec![v[2].clone(), v[4].clone(), v[5].clone(), fault.clone()],
            info(0, 0, 0, 0.),
        )?;
        // The gate has queued its final query reader. Reuse that exact region
        // for normalized convolution rows; keep unnormalized gated and R live.
        n::launch(
            &model.arena,
            n::Kernel::PromptNorm,
            [4, count, 1],
            vec![
                v[5].clone(),
                self.public.gamma[2].clone(),
                v[3].clone(),
                v[9].clone(),
            ],
            info(CHANNELS, CHANNELS, 4, 1e-6),
        )?;
        crate::strata_prompt::head_flags(client, v[9].clone(), fault.clone(), count, 4)?;
        let in_place = crate::strata_prompt::storage::in_place_or_disjoint(&input, &output)?;
        let mut conv = vec![
            u::handle(&model.arena, roots[0])?,
            v[3].clone(),
            self.public.conv.clone(),
        ];
        if !in_place {
            conv.push(input);
        }
        conv.extend([v[5].clone(), output, fault.clone()]);
        n::launch(
            &model.arena,
            if in_place {
                n::Kernel::PromptConvInPlace
            } else {
                n::Kernel::PromptConv
            },
            [40, count, 1],
            conv,
            info(CHANNELS, CHANNELS, 0, 0.),
        )?;
        n::launch(
            &model.arena,
            n::Kernel::PromptHistory,
            [40, 1, 1],
            vec![
                u::handle(&model.arena, roots[0])?,
                v[3].clone(),
                u::handle(&model.arena, roots[1])?,
                fault,
            ],
            info(0, 0, 0, 0.),
        )?;
        if !crate::strata_intel_core::enabled() || self.public_grouped_reads() {
            let keep = count.min(HISTORY);
            crate::strata_prompt::copy_handles(
                client,
                v[3].clone(),
                crate::strata_prompt::outer_workspace()?[11].clone(),
                (count - keep) * CHANNELS,
                keep * CHANNELS,
            )?;
        }
        u::cancel(cancel)?;
        Ok(pending)
    }
    pub(crate) fn validate_prompt_full(
        &self,
        model: &ResidentModel<'_, '_>,
        p: &Pending,
        start: usize,
        count: usize,
    ) -> Result<()> {
        if !p.armed
            || p.binding != Some(self.binding)
            || *p.stamp != [self.generation as usize, start, count]
            || !self.pending
            || self.next != start
            || p.roots != [self.ids[OLD], self.ids[NEXT], model.b.fault]
            || !Arc::ptr_eq(&p.poison, &self.poison)
            || self.poison.load(Ordering::Acquire)
            || model.ple_history.tokens() != *p.old_tokens
        {
            return Err(error());
        }
        Ok(())
    }
    pub(crate) fn prompt_readback(
        &self,
        model: &ResidentModel<'_, '_>,
        p: &Pending,
        cancel: &AtomicBool,
    ) -> Result<HistoryReadback> {
        self.validate_prompt_full(model, p, p.stamp[1], p.stamp[2])?;
        let rows = p.stamp[2].min(HISTORY);
        let tail = crate::strata_prompt::outer_workspace()?[11].clone();
        let tail = tail
            .clone()
            .offset_end(tail.size_in_used() - (rows * CHANNELS * 4) as u64);
        model.admission.readback_audited(cancel, || {
            let read = |h: Handle| -> Result<Zeroizing<Vec<f32>>> {
                struct Lease(cubecl::bytes::Bytes);
                impl Drop for Lease {
                    fn drop(&mut self) {
                        self.0.zeroize();
                    }
                }
                let bytes = Lease(
                    model
                        .arena
                        .client()
                        .map_err(u::fault)?
                        .read_one(h)
                        .map_err(u::fault)?,
                );
                let mut values = Zeroizing::new(Vec::with_capacity(bytes.0.len() / 4));
                for b in bytes.0.chunks_exact(4) {
                    let v = f32::from_le_bytes(b.try_into().map_err(|_| error())?);
                    if !v.is_finite() {
                        return Err(error());
                    }
                    values.push(v);
                }
                Ok(values)
            };
            Ok(HistoryReadback {
                tail: read(tail)?,
                next: read(u::handle(&model.arena, p.roots[1])?)?,
            })
        })
    }
    pub(crate) fn prepare_prompt_host<'a>(
        &self,
        model: &'a mut ResidentModel<'_, '_>,
        p: &Pending,
        diagnostic: Option<&HistoryReadback>,
        cancel: &AtomicBool,
    ) -> Result<device_history::Commit<'a>> {
        self.validate_prompt_full(model, p, p.stamp[1], p.stamp[2])?;
        if crate::strata_intel_core::enabled() {
            return core::prepare_device_history_tokens(
                &mut model.ple_history,
                model.graph.ple_hash(),
                *p.old_tokens,
                &p.tokens,
                cancel,
            )
            .map(device_history::Commit::Device)
            .map_err(u::fault);
        }
        let HistoryReadback { tail, next } = diagnostic.ok_or(error())?;
        let old = Zeroizing::new(model.ple_conv.history().to_vec());
        core::prepare_prompt(
            &mut model.ple_conv,
            &mut model.ple_history,
            model.graph.ple_hash(),
            &p.old_tokens,
            &p.tokens,
            &old,
            tail,
            next,
            2 * 1024 * 1024,
            cancel,
        )
        .map(device_history::Commit::Legacy)
        .map_err(u::fault)
    }
    pub(crate) fn commit_prompt_full(&mut self, mut p: Pending) {
        self.ids.swap(OLD, NEXT);
        self.next = p.stamp[1] + p.stamp[2];
        self.pending = false;
        p.armed = false;
    }
}
