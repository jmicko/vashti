//! Explicit count1 model seam; ordinary native MAX8 remains unchanged.
//! Explicit serial/overlap scheduling; all consumers join before acceptance.
use super::*;
use crate::resident_model::batch_io::Count1RouterInput;
use kuno_text::cpu_expert::PackedExperts;
#[path = "q3_expert_oracle.rs"]
mod public_expert_oracle;

impl GpuExecutor<'_> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn enqueue_routed_moe_count1_hybrid(
        &mut self,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        w: &mut Workspace,
        s: &Session,
        layer: usize,
        input: BufferId,
        output: BufferId,
        views: &BatchSpans,
        routes: &token_waves::Plan,
        packet: &Count1RouterInput,
        cancel: &AtomicBool,
    ) -> Result<Pending> {
        let result = guarded(&w.poison.clone(), || {
            let result = self.count1_checked(
                graph, arena, w, s, layer, input, output, views, routes, packet, cancel,
            );
            if self.secondary.is_some()
                && let Err(error) = &result
            {
                let (kind, reason) = match error {
                    Error::InvalidFormat(reason) => (1u32, *reason),
                    Error::LimitExceeded(reason) => (2, *reason),
                    Error::InvalidBufferLength => (3, ""),
                    Error::ArithmeticOverflow => (4, ""),
                    Error::OutOfBounds => (5, ""),
                    _ => (6, ""),
                };
                // Only a source-controlled static reason identifier; never raw
                // backend errors, routing IDs, prompts or model output.
                let id = reason.as_bytes().iter().fold(2_166_136_261u32, |h, b| {
                    (h ^ u32::from(*b)).wrapping_mul(16_777_619)
                });
                eprintln!("public_secondary_count1_failure kind={kind} reason_id={id}");
            }
            result
        });
        if result.is_err() {
            self.poison_resident();
        }
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn count1_checked(
        &mut self,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        w: &mut Workspace,
        s: &Session,
        layer: usize,
        input: BufferId,
        output: BufferId,
        views: &BatchSpans,
        routes: &token_waves::Plan,
        packet: &Count1RouterInput,
        cancel: &AtomicBool,
    ) -> Result<Pending> {
        // All public policy/count/binding checks precede scratch mutation or
        // private CPU fills. There is no scalar-cache or GPU-only fallback.
        u::cancel(cancel)?;
        w.check(self, s, arena)?;
        graph.verify_unchanged()?;
        self.model.verify_file_unchanged()?;
        let h = w
            .hybrid
            .as_ref()
            .ok_or(err("native count1 reservation absent"))?;
        let cap = h.cap.ok_or(err("native count1 policy absent"))?;
        let overlap = h.overlap;
        if self.q3_skip_empty_segments && !self.q3_empty_slot_fill {
            return Err(err("native Q3 empty-segment runtime policy"));
        }
        if self.q3_subgroup_moe && (!self.q3_empty_slot_fill || !self.q3_skip_empty_segments) {
            return Err(err("native Q3 subgroup runtime policy"));
        }
        if (self.q3_integer_moe || self.q3_shared_original_moe)
            && (!self.packed_profile.is_low_bit()
                || !self.q3_empty_slot_fill
                || !self.q3_skip_empty_segments
                || !self.fast_dense
                || !self.gpu_router
                || !self.shared_kernels
                || self.q3_subgroup_moe
                || (self.q3_integer_moe && self.q3_shared_original_moe)
                || self.fused_experts
                || self.integer_experts
                || w.cpu_q8_activations() != Some(false)
                || w.cpu_q8_residual() != Some(false))
        {
            return Err(err("native Q3 integer runtime policy"));
        }
        if self.q3_empty_slot_fill
            && (!self.packed_profile.is_low_bit()
                || cap != 10
                || self.gpu_expert_limit != 10
                || !overlap
                || !self.model.weights_mapped()
                || self.adaptive.is_some()
                || self.secondary.is_some())
        {
            return Err(err("native Q3 empty-slot runtime policy"));
        }
        if cap > 10
            || self.gpu_expert_limit > 10
            || self.gpu_expert_limit < 10 && (cap != 10 || !overlap || !self.model.weights_mapped())
            || self.integer_experts && (!self.fused_experts || self.gpu_expert_limit != 10)
            || self.fused_experts && !self.shared_kernels
            || routes.count() != 1
            || s.stamp.count != 1
            || views.count() != 1
            || routes.wave_count() != 1
            || views.packed_first() != 0
            || views.input_stride() < 2560
            || views.output_stride() < 2560
            || graph.binding_id() != w.binding
            || self.batch_experts
            || self.ordered_public_upload
            || self.word_down
        {
            return Err(err("native count1 exact configured transaction"));
        }
        packet.validate_for(
            graph,
            input,
            s.stamp.fault,
            layer,
            s.stamp.start,
            views.input_first(),
            views.input_stride(),
        )?;
        let wave = routes.wave(0).ok_or(Error::InvalidBufferLength)?;
        if wave.tokens() != (0..1) || wave.protected_experts().len() != 10 {
            return Err(err("native count1 original ten-rank wave"));
        }
        let g = *w.layout.geometries.get(layer).ok_or(Error::OutOfBounds)?;
        g.validate_count(1).map_err(err)?;
        if self.q3_skip_empty_segments && (g.tile() != 1 || g.capacity() != 144) {
            return Err(err("native Q3 empty-segment count1 geometry"));
        }
        let weights = weights(graph, layer)?;
        if [weights.gate, weights.up, weights.down]
            .iter()
            .any(|t| !t.belongs_to(self.model))
        {
            return Err(err("native count1 original artifact"));
        }
        let key = weights.gate.info().name().to_owned();
        // Completed prefill must already have created every original bank.
        let bank = self
            .banks
            .get(&key)
            .ok_or(err("native count1 original bank absent"))?;
        bank.check_names(weights)?;
        bank.check_client(arena.client().map_err(u::fault)?)?;
        if self.q3_skip_empty_segments
            && (!bank.profile().is_low_bit()
                || [bank.gate(), bank.up(), bank.down()]
                    .iter()
                    .any(|b| b.segments().len() != 2))
        {
            return Err(err("native Q3 empty-segment original banks"));
        }
        let ids = w.ids(input, output);
        let spans = k::Spans::new([
            0,
            views.input_first(),
            views.input_stride(),
            0,
            g.ff(),
            0,
            g.hidden(),
            0,
            views.output_first(),
            views.output_stride(),
            0,
        ]);
        let mut words = Zeroizing::new([0usize; 7]);
        for (i, &id) in ids.iter().enumerate() {
            words[i] = u::words(arena, id)?;
        }
        spans.validate(g, *words).map_err(err)?;
        physical(arena, bank, g, &ids, s.stamp.fault, w)?;
        let owner_ids = *h.ids;
        let operator = h.operator.clone();
        let row = routes.token(0).ok_or(Error::InvalidBufferLength)?;
        if row.len() != 10 {
            return Err(Error::InvalidBufferLength);
        }
        let mut experts = Zeroizing::new([0usize; 10]);
        let mut probabilities = Zeroizing::new([0f32; 10]);
        for (rank, v) in row.iter().enumerate() {
            experts[rank] = v.expert() as usize;
            probabilities[rank] = v.probability();
        }
        let prepare = if self.q3_empty_slot_fill {
            count1::Proof::prepare_empty_slots
        } else {
            count1::Proof::prepare
        };
        let (mut proof, staged, last, split) = prepare(
            &bank.slots,
            &experts,
            &probabilities,
            cap,
            self.gpu_expert_limit,
        )?;
        // prepare() has validated this exact immutable pending plan. Derive
        // dispatch only from its GPU slots, never the live bank's reverse map.
        // Both banks still pass physical() before and after every fill, and the
        // original source proof is revalidated before the headers/launches.
        let active_segments = if self.q3_skip_empty_segments {
            let parts = [bank.gate(), bank.up(), bank.down()].map(|b| {
                std::array::from_fn(|i| {
                    let p = &b.segments()[i];
                    (p.first_slot(), p.capacity())
                })
            });
            Some(Zeroizing::new(q3_segment_activity(
                bank.slots.capacity(),
                parts,
                &last.pending.slots,
            )?))
        } else {
            None
        };
        let mut pairs = Zeroizing::new([(0u32, 0u32); 10]);
        for (i, &id) in split.gpu().iter().enumerate() {
            pairs[i] = (id as u32, last.pending.slots[i] as u32);
        }
        let rank_map = count1::Proof::native_map(&split);
        let original_probabilities = proof.probabilities();
        let secondary_selection = self
            .secondary
            .as_ref()
            .map(|s| s.select(layer, split.cpu(), bank))
            .transpose()?;
        let mut map = hybrid_k::Map::new(
            proof.ids(),
            &original_probabilities,
            &rank_map,
            &pairs[..split.gpu().len()],
            bank.slots.capacity(),
        )
        .map_err(err)?;
        let router_reference_host = self
            .q3_router_reference
            .as_ref()
            .map(|reference| {
                reference
                    .owned_bytes()?
                    .checked_add(16 * 1024)
                    .ok_or(Error::ArithmeticOverflow)
            })
            .transpose()?
            .unwrap_or(0);
        let live = packet
            .owned_bytes()
            .checked_add(router_reference_host)
            .ok_or(Error::ArithmeticOverflow)?
            .checked_add(
                staged
                    .retained_native_metadata_bytes(&last.pending)
                    .map_err(|_| err("native count1 actual candidate capacity"))?,
            )
            .and_then(|n| {
                n.checked_add(
                    std::mem::size_of::<count1::Proof>()
                        + std::mem::size_of::<crate::cpu_gpu_expert_schedule::Split>()
                        + map.owned_bytes()
                        + std::mem::size_of::<Last>()
                        + std::mem::size_of::<Pending>()
                        + 4096,
                )
            })
            .ok_or(Error::ArithmeticOverflow)?;
        let secondary_enqueue = if secondary_selection
            .as_ref()
            .is_some_and(|s| s.gpu_count() > 0)
        {
            crate::secondary_experts::ENQUEUE_HOST
        } else {
            0
        };
        let private_limit = HYBRID_HOST
            .checked_sub(live)
            .and_then(|n| n.checked_sub(secondary_enqueue))
            .and_then(|n| {
                n.checked_sub(if self.adaptive.is_some() {
                    crate::adaptive_experts::HOST
                } else {
                    0
                })
            })
            .and_then(|n| {
                n.checked_sub(if self.oracle_trace.is_some() {
                    crate::oracle_trace::Trace::HOST
                } else {
                    0
                })
            })
            .filter(|&n| n > 0)
            .ok_or(Error::LimitExceeded("native count1 shared CPU host"))?;
        w.policy.issue(layer)?;
        if self.shared_kernels {
            let ids = Zeroizing::new([
                w.ids[0],
                w.ids[1],
                w.ids[2],
                w.ids[3],
                w.ids[4],
                owner_ids[0],
                owner_ids[1],
            ]);
            u::zero_batch(arena, &ids[..])?;
        } else {
            for &id in w.ids.iter().chain(owner_ids.iter()) {
                u::zero(arena, id)?;
            }
        }
        // GPU application encoding can coexist with the CPU owner. Precharge
        // both actual owning header pages plus conservative parent frame bytes;
        // the core additionally subtracts its 32-KiB orchestrator allowance.
        let concurrent_gpu_host =
            [ids[MAP], ids[PROB]]
                .into_iter()
                .try_fold(4096usize, |n, id| {
                    n.checked_add(arena.layout(id).map_err(u::fault)?.owned_bytes)
                        .ok_or(Error::ArithmeticOverflow)
                })?;
        let cpu_limit = if overlap {
            private_limit
                .checked_sub(concurrent_gpu_host)
                .filter(|&n| n > 0)
                .ok_or(Error::LimitExceeded("native count1 concurrent host"))?
        } else {
            private_limit
        };
        let mut parent = || -> Result<expert_banks::FillStats> {
            let mut stats = expert_banks::FillStats::default();
            for window in last.pending.uploads.chunks(10) {
                proof.validate_source(&bank.slots, &staged, &last, false)?;
                u::cancel(cancel)?;
                let requests = fill_requests(
                    [bank.gate(), bank.up(), bank.down()].map(|b| b.expert_bytes() as u64),
                    window.len(),
                    false,
                    self.expert_batch_bytes,
                )?;
                let charged = requests.iter().try_fold(0u64, |n, &b| {
                    n.checked_add(main_admission::page_bytes(b).map_err(admission_error)?)
                        .ok_or(Error::ArithmeticOverflow)
                })?;
                if charged > 40 * 1024 * 1024 {
                    return Err(Error::LimitExceeded("native count1 real GPU source"));
                }
                let add = w.admission.allocate(&requests, || {
                    let mut add = expert_banks::FillStats::default();
                    for &(expert, slot) in window {
                        for (packed, tensor) in [
                            (bank.gate(), weights.gate),
                            (bank.up(), weights.up),
                            (bank.down(), weights.down),
                        ] {
                            proof.validate_source(&bank.slots, &staged, &last, false)?;
                            u::cancel(cancel)?;
                            if w.poison.load(Ordering::Acquire) {
                                return Err(err("native count1 aborted GPU fill"));
                            }
                            let (bytes, read, upload) = expert_banks::fill(
                                &self.client,
                                packed,
                                tensor,
                                expert,
                                slot,
                                cancel,
                            )?;
                            add.bytes = add
                                .bytes
                                .checked_add(bytes)
                                .ok_or(Error::ArithmeticOverflow)?;
                            add.read_ns = add
                                .read_ns
                                .checked_add(read)
                                .ok_or(Error::ArithmeticOverflow)?;
                            add.upload_ns = add
                                .upload_ns
                                .checked_add(upload)
                                .ok_or(Error::ArithmeticOverflow)?;
                        }
                        add.experts += 1;
                    }
                    Ok(add)
                })?;
                stats.bytes = stats
                    .bytes
                    .checked_add(add.bytes)
                    .ok_or(Error::ArithmeticOverflow)?;
                stats.read_ns = stats
                    .read_ns
                    .checked_add(add.read_ns)
                    .ok_or(Error::ArithmeticOverflow)?;
                stats.upload_ns = stats
                    .upload_ns
                    .checked_add(add.upload_ns)
                    .ok_or(Error::ArithmeticOverflow)?;
                stats.experts = stats
                    .experts
                    .checked_add(add.experts)
                    .ok_or(Error::ArithmeticOverflow)?;
            }
            proof.validate_source(&bank.slots, &staged, &last, false)?;
            write_gpu_header(arena, ids[MAP], map.words(), &w.poison)?;
            let mut probs = Zeroizing::new([0u32; 80]);
            for (i, &p) in map.probabilities().iter().enumerate() {
                probs[i] = p.to_bits();
            }
            write_gpu_header(arena, ids[PROB], &*probs, &w.poison)?;
            physical(arena, bank, g, &ids, s.stamp.fault, w)?;
            if !split.gpu().is_empty() {
                if self.q3_subgroup_moe || self.q3_integer_moe || self.q3_shared_original_moe {
                    let active = active_segments
                        .as_ref()
                        .ok_or(err("native Q3 selected segment proof absent"))?;
                    launch_q3_selected_segments(
                        arena,
                        bank,
                        g,
                        &spans,
                        &ids,
                        active,
                        if self.q3_integer_moe {
                            SelectedIqKernel::Integer
                        } else if self.q3_shared_original_moe {
                            SelectedIqKernel::SharedOriginal
                        } else {
                            SelectedIqKernel::SubgroupOriginal
                        },
                    )?;
                } else if self.shared_kernels
                    && bank.profile() == crate::expert_banks::OriginalPackedProfile::Q4
                {
                    let owner = |i| arena.handle(ids[i]).cloned().map_err(u::fault);
                    let owners = || -> Result<[Handle; 7]> {
                        Ok([
                            owner(0)?,
                            owner(1)?,
                            owner(2)?,
                            owner(3)?,
                            owner(4)?,
                            owner(5)?,
                            owner(6)?,
                        ])
                    };
                    if bank.gate().segments().len() == 2 {
                        // physical() already proved all original contiguous segments,
                        // exact extents/origin/aliases, including both paired banks.
                        let banks = std::array::from_fn(|shard| {
                            [
                                bank.gate().segments()[shard].handle().clone(),
                                bank.up().segments()[shard].handle().clone(),
                                bank.down().segments()[shard].handle().clone(),
                            ]
                        });
                        let capacities =
                            std::array::from_fn(|shard| bank.gate().segments()[shard].capacity());
                        crate::next_shared_moe::launch_pair(
                            &self.client,
                            g,
                            &spans,
                            banks,
                            owners()?,
                            capacities,
                            self.fused_experts,
                            self.integer_experts,
                        )?;
                    } else {
                        for shard in 0..bank.gate().segments().len() {
                            let ga = &bank.gate().segments()[shard];
                            let up = &bank.up().segments()[shard];
                            let down = &bank.down().segments()[shard];
                            crate::next_shared_moe::launch(
                                &self.client,
                                g,
                                &spans,
                                [
                                    ga.handle().clone(),
                                    up.handle().clone(),
                                    down.handle().clone(),
                                ],
                                [
                                    owner(0)?,
                                    owner(1)?,
                                    owner(2)?,
                                    owner(3)?,
                                    owner(4)?,
                                    owner(5)?,
                                    owner(6)?,
                                ],
                                ga.first_slot(),
                                ga.capacity(),
                                self.fused_experts,
                                self.integer_experts,
                            )?;
                        }
                    }
                } else if let Some(active) = &active_segments {
                    launch_q3_active_segments(arena, bank, g, &spans, &ids, active)?;
                } else {
                    launch_wave(arena, bank, g, &spans, &ids, 10)?;
                }
            }
            launch_finite(arena, g, &spans, &ids)?;
            if overlap {
                // Submit on the original parent stream without waiting for GPU
                // completion. The original router/whole-model proofs still drain
                // and audit every consumer before publication or scratch reuse.
                self.client.flush().map_err(u::fault)?;
            }
            if let Some(selection) = &secondary_selection
                && selection.gpu_count() > 0
            {
                self.secondary
                    .as_mut()
                    .ok_or(err("secondary owner absent"))?
                    .enqueue(layer, selection, packet.mixed(), cancel)?;
            }
            Ok(stats)
        };
        // Parent work never joins a CPU consumer inside its admission allocation
        // closure. CPU/parent error or panic finishes all borrowed CPU jobs before
        // the enclosing sticky poison/drain/erase path can dispose model owners.
        let cpu_ids = secondary_selection.as_ref().map_or(split.cpu(), |s| &s.cpu);
        let (partials, stats) = if cpu_ids.is_empty() {
            (None, parent()?)
        } else {
            let reader = self.model.positional_reader()?;
            let cpu_weights = PackedExperts::new(&reader, weights.gate, weights.up, weights.down)?;
            reader.verify_file_unchanged()?;
            let (result, stats) = if overlap {
                operator.execute_with_private_limit_alongside(
                    &cpu_weights,
                    cpu_ids,
                    packet.mixed(),
                    cancel,
                    cpu_limit,
                    parent,
                )?
            } else {
                let result = operator.execute_with_private_limit(
                    &cpu_weights,
                    cpu_ids,
                    packet.mixed(),
                    cancel,
                    cpu_limit,
                )?;
                (result, parent()?)
            };
            reader.verify_file_unchanged()?;
            (Some(result), stats)
        };
        let partials = if let Some(selection) = &secondary_selection {
            if split.cpu().is_empty() {
                None
            } else {
                Some(
                    self.secondary
                        .as_mut()
                        .ok_or(err("secondary joined owner absent"))?
                        .finish(
                            selection,
                            split.cpu(),
                            partials,
                            HYBRID_HOST
                                .checked_sub(live)
                                .ok_or(Error::ArithmeticOverflow)?,
                            cancel,
                        )?,
                )
            }
        } else {
            partials.map(crate::secondary_experts::HostPartials::Cpu)
        };
        proof.complete(partials.as_ref())?;
        if let Some(selection) = &secondary_selection
            && selection.gpu_count() > 0
        {
            let secondary = self
                .secondary
                .as_ref()
                .ok_or(err("secondary completion owner absent"))?;
            let ids = zeroize::Zeroizing::new(
                split
                    .cpu()
                    .iter()
                    .copied()
                    .filter(|id| !selection.cpu.contains(id))
                    .collect::<Vec<_>>(),
            );
            proof.mark_secondary(&ids, selection.instance)?;
            proof.validate_secondary(Some(secondary), layer, bank)?;
        }
        proof.validate_source(&bank.slots, &staged, &last, true)?;
        map.complete_cpu(partials.as_ref(), g.hidden())
            .map_err(err)?;
        let live = live
            .checked_add(self.adaptive.as_ref().map_or(0, |c| c.owned_bytes()))
            .and_then(|n| {
                n.checked_add(
                    self.oracle_trace
                        .as_ref()
                        .map_or(0, |trace| trace.owned_bytes()),
                )
            })
            .ok_or(Error::ArithmeticOverflow)?;
        let partial_bytes = partials.as_ref().map_or(0, |p| p.owned_bytes());
        let encoding = arena.layout(owner_ids[0]).map_err(u::fault)?.owned_bytes
            + arena.layout(owner_ids[1]).map_err(u::fault)?.owned_bytes;
        if live
            .checked_add(partial_bytes)
            .and_then(|n| n.checked_add(encoding))
            .is_none_or(|n| n > HYBRID_HOST)
        {
            return Err(Error::LimitExceeded("native count1 joined bridge host"));
        }
        // CPU readiness/results enter distinct owners only AFTER all CPU jobs
        // finish. The original rank0..9 merge is ordered after both producers.
        write_cpu(arena, owner_ids[0], partials.as_ref(), live, &w.poison)?;
        write_words(arena, owner_ids[1], map.owner_words(), &w.poison)?;
        drop(partials);
        if self.q3_expert_oracle && s.stamp.start == public_expert_oracle::POSITION {
            if !self.q3_empty_slot_fill
                || self.packed_profile != expert_banks::OriginalPackedProfile::UnslothQ3
                || cap != 10
                || self.gpu_expert_limit != 10
                || self.adaptive.is_some()
                || self.secondary.is_some()
                || operator.q8_activations()
                || g.hidden() != public_expert_oracle::HIDDEN
                || spans.partial_first() != 0
                || spans.partial_stride() != public_expert_oracle::HIDDEN
            {
                return Err(err("public Q3 actual expert oracle policy"));
            }
            if let Some(reference) = &self.q3_router_reference {
                let report = reference.layer(layer)?.compare(
                    packet.logits().row(0)?,
                    packet
                        .gpu_selection()
                        .ok_or(err("public Q3 GPU routes required"))?,
                )?;
                if !report.raw_gpu_valid {
                    return Err(err("public Q3 raw router invalid"));
                }
                println!(
                    "public_q3_router_compare position=23 layer={layer} cpu_membership_matches={} cpu_rank_matches={} same_input_ops_membership_matches={} same_input_ops_rank_matches={} max_score_difference={} max_probability_difference_by_rank={} max_same_input_ops_probability_difference_by_rank={} cpu_logit_margin={} gpu_logit_margin={} cpu_probability_margin={} gpu_probability_margin={} gpu_ops_logit_margin={}",
                    report.cpu_membership_matches,
                    report.cpu_rank_matches,
                    report.ops_membership_matches,
                    report.ops_rank_matches,
                    report.max_score_difference,
                    report.max_probability_difference,
                    report.max_ops_probability_difference,
                    report.cpu_logit_margin,
                    report.gpu_logit_margin,
                    report.cpu_probability_margin,
                    report.gpu_probability_margin,
                    report.gpu_ops_logit_margin,
                );
            }
            if !split.gpu().is_empty() {
                w.check(self, s, arena)?;
                proof.validate_source(&bank.slots, &staged, &last, true)?;
                let shadow_limit = cpu_limit
                    .checked_sub(public_expert_oracle::READ_HOST)
                    .filter(|&limit| limit > 0)
                    .ok_or(Error::LimitExceeded("public Q3 expert oracle host reserve"))?;
                let reader = self.model.positional_reader()?;
                let cpu_weights =
                    PackedExperts::new(&reader, weights.gate, weights.up, weights.down)?;
                reader.verify_file_unchanged()?;
                // Same actual immutable weights and packet input, sequentially
                // after the original CPU jobs/partials released their owners.
                let shadow = operator.execute_with_private_limit(
                    &cpu_weights,
                    split.gpu(),
                    packet.mixed(),
                    cancel,
                    shadow_limit,
                )?;
                reader.verify_file_unchanged()?;
                if live
                    .checked_add(shadow.owned_bytes())
                    .and_then(|bytes| bytes.checked_add(public_expert_oracle::READ_HOST))
                    .is_none_or(|bytes| bytes > HYBRID_HOST)
                {
                    return Err(Error::LimitExceeded("public Q3 expert oracle live host"));
                }
                let gpu = u::read_prefix_f32(arena, ids[PART], public_expert_oracle::GPU_WORDS)?;
                if gpu
                    .capacity()
                    .checked_mul(std::mem::size_of::<f32>())
                    .is_none_or(|bytes| bytes > 131072)
                {
                    return Err(Error::LimitExceeded("public Q3 expert oracle copied host"));
                }
                let report =
                    public_expert_oracle::compare(proof.ids(), split.gpu(), &gpu, shadow.active())?;
                let arithmetic = match operator.backend() {
                    kuno_text::cpu_expert::Backend::Portable => "portable64",
                    kuno_text::cpu_expert::Backend::Avx2 => "avx2-original64",
                    kuno_text::cpu_expert::Backend::Avx2Fma => "avx2-fma32-other-original64",
                };
                println!(
                    "public_q3_actual_expert_oracle position=23 layer={layer} arithmetic={arithmetic} gpu_ranks={} values={} different_bits={} max_abs_error={} max_scaled_error={} first_different_rank={:?} first_different_column={:?}",
                    report.ranks,
                    report.values,
                    report.different_bits,
                    report.max_abs_error,
                    report.max_scaled_error,
                    report.first_different_rank,
                    report.first_different_column
                );
                drop(gpu);
                drop(shadow);
                proof.validate_source(&bank.slots, &staged, &last, true)?;
                w.check(self, s, arena)?;
            } else {
                println!(
                    "public_q3_actual_expert_oracle position=23 layer={layer} gpu_ranks=0 skipped=true"
                );
            }
        }
        hybrid_k::merge(
            arena,
            g,
            &ids,
            owner_ids[0],
            owner_ids[1],
            s.stamp.fault,
            &spans,
            1,
            0,
            g.hidden(),
            0,
        )?;
        launch_finite(arena, g, &spans, &ids)?;
        hybrid_k::fold(
            arena,
            g,
            &ids,
            owner_ids[0],
            owner_ids[1],
            s.stamp.fault,
            &spans,
            1,
            0,
            g.hidden(),
            0,
        )?;
        u::cancel(cancel)?;
        w.check(self, s, arena)?;
        proof.validate_source(&bank.slots, &staged, &last, true)?;
        let pins: Vec<Handle> = [bank.gate(), bank.up(), bank.down()]
            .into_iter()
            .flat_map(|b| b.segments().iter().map(|p| p.handle().clone()))
            .collect();
        let metadata = staged
            .retained_native_metadata_bytes(&last.pending)
            .map_err(|_| Error::LimitExceeded("native count1 retained plan"))?;
        if metadata
            .checked_add(
                std::mem::size_of::<Pending>()
                    + std::mem::size_of::<Last>()
                    + pins.capacity() * std::mem::size_of::<Handle>(),
            )
            .is_none_or(|n| n > 32 * 1024)
        {
            return Err(Error::LimitExceeded(
                "native count1 compact pending envelope",
            ));
        }
        let misses = proof.misses();
        self.cpu_resident_expert_calls = self
            .cpu_resident_expert_calls
            .checked_add((proof.cpu_count() - proof.cpu_miss_count()) as u64)
            .ok_or(Error::ArithmeticOverflow)?;
        let out = Pending {
            views: Views {
                ids: Zeroizing::new([input, output]),
                offsets: Zeroizing::new([
                    views.input_first(),
                    views.input_stride(),
                    views.output_first(),
                    views.output_stride(),
                ]),
            },
            stamp: Zeroizing::new((*s.stamp).clone()),
            layer: Zeroizing::new(layer),
            key,
            instance: bank.instance(),
            pins,
            staged: Some(staged),
            last: Some(last),
            count1: Some(proof),
            waves: Zeroizing::new(1),
            poison: w.poison.clone(),
            armed: true,
        };
        // CPU misses are not resident GPU hits and not GPU uploads.
        self.expert_hits = self
            .expert_hits
            .checked_add((10 - misses) as u64)
            .ok_or(Error::ArithmeticOverflow)?;
        self.expert_uploads = self
            .expert_uploads
            .checked_add(stats.experts)
            .ok_or(Error::ArithmeticOverflow)?;
        self.uploaded_bytes = self
            .uploaded_bytes
            .checked_add(stats.bytes)
            .ok_or(Error::ArithmeticOverflow)?;
        self.weight_read_ns = self
            .weight_read_ns
            .checked_add(stats.read_ns)
            .ok_or(Error::ArithmeticOverflow)?;
        self.weight_upload_ns = self
            .weight_upload_ns
            .checked_add(stats.upload_ns)
            .ok_or(Error::ArithmeticOverflow)?;
        if self.fused_experts && !split.gpu().is_empty() {
            self.fused_expert_calls = self
                .fused_expert_calls
                .checked_add(1)
                .ok_or(Error::ArithmeticOverflow)?;
        }
        self.moe_calls = self
            .moe_calls
            .checked_add(1)
            .ok_or(Error::ArithmeticOverflow)?;
        Ok(out)
    }
}

/// Only the fixed Q3 72+72 owner is admitted. A segment is inactive exactly
/// when the validated frozen GPU subset contains no original slot in it.
fn q3_segment_activity(
    capacity: usize,
    parts: [[(usize, usize); 2]; 3],
    slots: &[usize],
) -> Result<[bool; 2]> {
    if capacity != 144 || parts != [[(0, 72), (72, 72)]; 3] || slots.len() > 10 {
        return Err(err("native Q3 empty-segment original geometry"));
    }
    let mut active = Zeroizing::new([false; 2]);
    for (rank, &slot) in slots.iter().enumerate() {
        if slot >= capacity || slots[..rank].contains(&slot) {
            return Err(err("native Q3 empty-segment frozen slots"));
        }
        active[slot / 72] = true;
    }
    Ok(*active)
}

/// Same original generic kernels, map, rank count and owning spans as
/// launch_wave(). Only a wholly inactive segment's gate/down pair is omitted;
/// its buffers remain retained and subject to every original owner audit.
fn launch_q3_active_segments(
    arena: &PrivateArena,
    bank: &ExpertBanks,
    g: k::Geometry,
    s: &k::Spans,
    ids: &[BufferId; 7],
    active: &[bool; 2],
) -> Result<()> {
    let [gate, up, down] = g.projections();
    let client = arena.client().map_err(u::fault)?;
    for (shard, &enabled) in active.iter().enumerate() {
        if !enabled {
            continue;
        }
        let ga = &bank.gate().segments()[shard];
        let u = &bank.up().segments()[shard];
        let d = &bank.down().segments()[shard];
        unsafe {
            k::grouped_gate_up::launch_unchecked::<WgpuRuntime>(
                client,
                g.grid(10, false).map_err(err)?,
                CubeDim::new_1d(64),
                ArrayArg::from_raw_parts(ga.handle().clone(), ga.words()),
                ArrayArg::from_raw_parts(u.handle().clone(), u.words()),
                arg(arena, ids[INPUT])?,
                arg(arena, ids[MAP])?,
                arg(arena, ids[ACT])?,
                arg(arena, ids[STATUS])?,
                s.map_first(),
                s.input_first(),
                s.input_stride(),
                s.act_first(),
                s.act_stride(),
                s.status_first(),
                ga.first_slot(),
                ga.capacity(),
                g.hidden(),
                g.ff(),
                gate.dtype(),
                up.dtype(),
                gate.row_bytes(),
                up.row_bytes(),
                g.tile(),
                g.gate_policy() == k::GatePolicy::Q4Block,
            );
            k::grouped_down::launch_unchecked::<WgpuRuntime>(
                client,
                g.grid(10, true).map_err(err)?,
                CubeDim::new_1d(64),
                ArrayArg::from_raw_parts(d.handle().clone(), d.words()),
                arg(arena, ids[ACT])?,
                arg(arena, ids[MAP])?,
                arg(arena, ids[PART])?,
                arg(arena, ids[STATUS])?,
                s.map_first(),
                s.act_first(),
                s.act_stride(),
                s.partial_first(),
                s.partial_stride(),
                s.status_first(),
                d.first_slot(),
                d.capacity(),
                g.hidden(),
                g.ff(),
                down.dtype(),
                down.row_bytes(),
                g.tile(),
            );
        }
    }
    Ok(())
}

/// Same frozen original plan/physical owners as the generic dispatch above.
/// Only explicit fixed-public Q3 options select subgroup original64 or the
/// alternative integer arithmetic. Finite/fault/merge and all-family
/// publication remain unchanged; no failure selects an alternative leaf.
#[derive(Clone, Copy)]
enum SelectedIqKernel {
    Integer,
    SharedOriginal,
    SubgroupOriginal,
}
fn launch_q3_selected_segments(
    arena: &PrivateArena,
    bank: &ExpertBanks,
    g: k::Geometry,
    s: &k::Spans,
    ids: &[BufferId; 7],
    active: &[bool; 2],
    choice: SelectedIqKernel,
) -> Result<()> {
    let client = arena.client().map_err(u::fault)?;
    let owner = |index: usize| arena.handle(ids[index]).cloned().map_err(u::fault);
    let owners = [
        owner(0)?,
        owner(1)?,
        owner(2)?,
        owner(3)?,
        owner(4)?,
        owner(5)?,
        owner(6)?,
    ];
    for (shard, &enabled) in active.iter().enumerate() {
        if !enabled {
            continue;
        }
        let ga = &bank.gate().segments()[shard];
        let up = &bank.up().segments()[shard];
        let down = &bank.down().segments()[shard];
        let launch = match choice {
            SelectedIqKernel::Integer => crate::quant_matvec::iq_moe_integer::launch,
            SelectedIqKernel::SharedOriginal => crate::quant_matvec::iq_moe_shared_original::launch,
            SelectedIqKernel::SubgroupOriginal => crate::quant_matvec::q3_subgroup_moe::launch,
        };
        launch(
            client,
            g,
            s,
            [
                ga.handle().clone(),
                up.handle().clone(),
                down.handle().clone(),
            ],
            owners.clone(),
            ga.first_slot(),
            ga.capacity(),
            10,
        )?;
    }
    Ok(())
}

// Unlike the ordinary serial header helper, this seam checks actual capacity
// against the public owning-page charge before any private header is encoded.
fn write_gpu_header(
    arena: &PrivateArena,
    id: BufferId,
    words: &[u32],
    poison: &AtomicBool,
) -> Result<()> {
    let l = arena.layout(id).map_err(u::fault)?;
    if l.logical_words != words.len() {
        return Err(Error::InvalidBufferLength);
    }
    let mut bytes = Zeroizing::new(Vec::new());
    bytes
        .try_reserve_exact(l.owned_bytes)
        .map_err(|_| Error::LimitExceeded("native count1 GPU header allocation"))?;
    if bytes.capacity() != l.owned_bytes {
        return Err(Error::LimitExceeded("native count1 GPU header capacity"));
    }
    bytes.resize(l.owned_bytes, 0);
    for (index, value) in words.iter().enumerate() {
        bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
    arena
        .client()
        .map_err(u::fault)?
        .write_private(
            arena.handle(id).map_err(u::fault)?.clone(),
            std::mem::take(&mut *bytes),
        )
        .map_err(|_| {
            poison.store(true, Ordering::Release);
            err("native count1 GPU header upload")
        })
}

fn write_cpu(
    arena: &PrivateArena,
    id: BufferId,
    partials: Option<&crate::secondary_experts::HostPartials>,
    live: usize,
    poison: &AtomicBool,
) -> Result<()> {
    let layout = arena.layout(id).map_err(u::fault)?;
    if layout.logical_words != CPU_WORDS {
        return Err(Error::InvalidBufferLength);
    }
    let mut bytes = Zeroizing::new(Vec::new());
    bytes
        .try_reserve_exact(layout.owned_bytes)
        .map_err(|_| Error::LimitExceeded("native count1 guarded CPU upload"))?;
    let retained = partials.map_or(0, |p| p.owned_bytes());
    if bytes.capacity() != layout.owned_bytes
        || live
            .checked_add(retained)
            .and_then(|n| n.checked_add(bytes.capacity() + 32768))
            .is_none_or(|n| n > HYBRID_HOST)
    {
        return Err(Error::LimitExceeded("native count1 actual upload host"));
    }
    bytes.resize(layout.owned_bytes, 0);
    if let Some(p) = partials {
        if p.full().len() != CPU_WORDS {
            return Err(Error::InvalidBufferLength);
        }
        for (i, &value) in p.full().iter().enumerate() {
            bytes[i * 4..i * 4 + 4].copy_from_slice(&value.to_bits().to_le_bytes());
        }
    }
    arena
        .client()
        .map_err(u::fault)?
        .write_private(
            arena.handle(id).map_err(u::fault)?.clone(),
            std::mem::take(&mut *bytes),
        )
        .map_err(|_| {
            poison.store(true, Ordering::Release);
            err("native count1 actual owning upload")
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    const PARTS: [[(usize, usize); 2]; 3] = [[(0, 72), (72, 72)]; 3];

    #[test]
    fn q3_segment_dispatch_keeps_first_last_slots_and_original_order() {
        let first = [71, 0, 35, 1, 70, 2, 69, 3, 68, 4];
        let second = [143, 72, 108, 73, 142, 74, 141, 75, 140, 76];
        assert_eq!(
            q3_segment_activity(144, PARTS, &first).unwrap(),
            [true, false]
        );
        assert_eq!(
            q3_segment_activity(144, PARTS, &second).unwrap(),
            [false, true]
        );
        assert_eq!(
            q3_segment_activity(144, PARTS, &[143, 0, 72, 71]).unwrap(),
            [true, true]
        );
        assert_eq!(first, [71, 0, 35, 1, 70, 2, 69, 3, 68, 4]);
        assert_eq!(second, [143, 72, 108, 73, 142, 74, 141, 75, 140, 76]);
    }

    #[test]
    fn q3_segment_dispatch_empty_gpu_subset_has_no_active_segment() {
        assert_eq!(
            q3_segment_activity(144, PARTS, &[]).unwrap(),
            [false, false]
        );
    }

    #[test]
    fn q3_segment_dispatch_rejects_invalid_frozen_slots() {
        for slots in [
            &[144][..],
            &[usize::MAX][..],
            &[0, 0][..],
            &[143, 143][..],
            &[0; 11][..],
        ] {
            assert!(q3_segment_activity(144, PARTS, slots).is_err());
        }
        assert!(q3_segment_activity(144, PARTS, &(0..11).collect::<Vec<_>>()).is_err());
    }

    #[test]
    fn q3_segment_dispatch_rejects_nonoriginal_or_mismatched_segments() {
        for capacity in [0, 72, 143, 145, usize::MAX] {
            assert!(q3_segment_activity(capacity, PARTS, &[0]).is_err());
        }
        for projection in 0..3 {
            for replacement in [
                [(72, 72), (0, 72)],
                [(0, 71), (72, 72)],
                [(0, 72), (73, 71)],
                [(0, 72), (usize::MAX, 72)],
            ] {
                let mut parts = PARTS;
                parts[projection] = replacement;
                assert!(q3_segment_activity(144, parts, &[0]).is_err());
            }
        }
    }

    #[test]
    fn q3_segment_dispatch_uses_the_validated_pending_gpu_plan() {
        let mut source = crate::bank_slots::BankSlots::new(144, 512).unwrap();
        let prime: Vec<_> = (100..244).collect();
        let initial = source.plan(&prime).unwrap();
        source.commit(initial).unwrap();
        // Expert IDs deliberately do not equal slots. Rank order is preserved.
        let ids = [243, 100, 172, 171, 101, 170, 173, 242, 135, 208];
        let (proof, staged, last, split) =
            count1::Proof::prepare_empty_slots(&source, &ids, &[0.1; 10], 10, 10).unwrap();
        proof
            .validate_source(&source, &staged, &last, false)
            .unwrap();
        assert_eq!(split.gpu(), &ids);
        assert_eq!(
            last.pending.slots,
            [143, 0, 72, 71, 1, 70, 73, 142, 35, 108]
        );
        assert_eq!(
            q3_segment_activity(144, PARTS, &last.pending.slots).unwrap(),
            [true, true]
        );
        assert_eq!(
            &count1::Proof::native_map(&split)[..20],
            &[0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0, 9]
        );
        proof
            .validate_source(&source, &staged, &last, false)
            .unwrap();
    }
}
