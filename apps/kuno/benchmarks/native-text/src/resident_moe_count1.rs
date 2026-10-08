//! Explicit count1 model seam; ordinary native MAX8 remains unchanged.
//! Explicit serial/overlap scheduling; all consumers join before acceptance.
use super::*;
use crate::resident_model::batch_io::Count1RouterInput;
use kuno_text::cpu_expert::PackedExperts;

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
            self.count1_checked(
                graph, arena, w, s, layer, input, output, views, routes, packet, cancel,
            )
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
        let (mut proof, staged, last, split) = count1::Proof::prepare(
            &bank.slots,
            &experts,
            &probabilities,
            cap,
            self.gpu_expert_limit,
        )?;
        let mut pairs = Zeroizing::new([(0u32, 0u32); 10]);
        for (i, &id) in split.gpu().iter().enumerate() {
            pairs[i] = (id as u32, last.pending.slots[i] as u32);
        }
        let rank_map = count1::Proof::native_map(&split);
        let original_probabilities = proof.probabilities();
        let mut map = hybrid_k::Map::new(
            proof.ids(),
            &original_probabilities,
            &rank_map,
            &pairs[..split.gpu().len()],
            bank.slots.capacity(),
        )
        .map_err(err)?;
        let live = packet
            .owned_bytes()
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
        let private_limit = HYBRID_HOST
            .checked_sub(live)
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
        let parent = || -> Result<expert_banks::FillStats> {
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
                if self.shared_kernels {
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
            Ok(stats)
        };
        // Parent work never joins a CPU consumer inside its admission allocation
        // closure. CPU/parent error or panic finishes all borrowed CPU jobs before
        // the enclosing sticky poison/drain/erase path can dispose model owners.
        let (partials, stats) = if split.cpu().is_empty() {
            (None, parent()?)
        } else {
            let reader = self.model.positional_reader()?;
            let cpu_weights = PackedExperts::new(&reader, weights.gate, weights.up, weights.down)?;
            reader.verify_file_unchanged()?;
            let (result, stats) = if overlap {
                operator.execute_with_private_limit_alongside(
                    &cpu_weights,
                    split.cpu(),
                    packet.mixed(),
                    cancel,
                    cpu_limit,
                    parent,
                )?
            } else {
                let result = operator.execute_with_private_limit(
                    &cpu_weights,
                    split.cpu(),
                    packet.mixed(),
                    cancel,
                    cpu_limit,
                )?;
                (result, parent()?)
            };
            reader.verify_file_unchanged()?;
            (Some(result), stats)
        };
        proof.complete(partials.as_ref())?;
        proof.validate_source(&bank.slots, &staged, &last, true)?;
        map.complete_cpu(partials.as_ref(), g.hidden())
            .map_err(err)?;
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
    partials: Option<&kuno_text::cpu_expert::Partials>,
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
