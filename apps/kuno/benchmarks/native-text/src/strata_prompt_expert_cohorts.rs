//! Closed small-expert cohorts. Original map/source/math guards, real parent
//! owners and native range/timeline hazards remain authoritative.
use super::*;
use zeroize::Zeroizing;
#[path = "strata_prompt_expert_cohort_probe.rs"]
mod probe;
pub(crate) use probe::cases as public_producers;
#[path = "strata_prompt_cohort_multi_dequant_probe.rs"]
mod multi_dequant;
pub(crate) use multi_dequant::cases as public_multi_dequant;
#[cfg(feature = "strata-small-quantized-matrix")]
#[path = "strata_prompt_cohort_compact.rs"]
pub(crate) mod compact;
pub(crate) const MAX: usize = 8;
pub(crate) const WORDS: usize = 48;
pub(crate) const X_BYTES: u64 = 20 * 1024 * 1024;
pub(crate) const H_BYTES: u64 = 5 * 1024 * 1024;
pub(crate) const IMAGE_BYTES: u64 = (DQ_WORDS * 4) as u64;
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(crate) enum CohortOp {
    Gather,
    Image { dtype: u32, down: bool },
    Activate,
}
impl CohortOp {
    fn lanes(self)->u32 {if cfg!(feature="strata-dequant-direct8") && matches!(self,Self::Image{..}) {32}else{256}}
    /// Original explicit fixture identity pairs with source() and /32 grids.
    pub(crate) fn original_name(self) -> &'static str {
        match self {
            Self::Gather => "StrataExpertCohortGather",
            Self::Image { down: false, .. } => "StrataExpertCohortGuImage",
            Self::Image { down: true, .. } => "StrataExpertCohortDownImage",
            Self::Activate => "StrataExpertCohortActivate",
        }
    }
    pub(super) fn selected_source(self) -> String {
        let source=self.source();
        #[cfg(feature="strata-dequant-direct8")]
        if let Self::Image{dtype,down}=self {return super::direct8::source(source,dtype,down);}
        #[cfg(feature="strata-dequant-quad32")]
        if let Self::Image{down,..}=self {return super::quad32::source(source,down);}
        source
    }
    pub(crate) fn source(self) -> String {
        let old = match self {
            Self::Gather => Op::Gather,
            Self::Activate => Op::ActivateCheckGu,
            Self::Image { dtype, down } => Op::Dequant { dtype, down },
        };
        let mut source = old.source();
        source = source.replace("var<uniform> p:Params;", "var<uniform> cohort:CohortInfo;\nvar<private> p:Params;\nvar<private> cohort_prefix:u32;\nvar<private> cohort_slot:u32;");
        source.push_str(include_str!("strata_prompt_expert_cohort_header.wgsl"));
        source = source.replace("fn main(@builtin(global_invocation_id) id:vec3<u32>) {", "fn main(@builtin(global_invocation_id) id:vec3<u32>) {\n    if !cohort_begin(id.z) {return;}");
        match self {
            Self::Gather => {
                source = source.replace("packed[at]=0u", "packed[cohort_prefix*1280u+at]=0u");
                source = source.replace(
                    "if end>arrayLength(&packed)",
                    "if cohort_prefix*1280u+end>arrayLength(&packed)",
                );
                source = source.replace(
                    "if i>=arrayLength(&packed)",
                    "if cohort_prefix*1280u+i>=arrayLength(&packed)",
                );
                source = source.replace("packed[i]=", "packed[cohort_prefix*1280u+i]=");
                source = source.replace(
                    "let stride=((first+255u)/256u)*256u;",
                    "let stride=cohort.base.r4*1280u/256u*256u;",
                );
            }
            Self::Activate => {
                source = source.replace("hidden[at]=0u", "hidden[cohort_prefix*320u+at]=0u");
                source = source.replace(
                    "if end>arrayLength(&hidden)",
                    "if cohort_prefix*320u+end>arrayLength(&hidden)",
                );
                source = source.replace(
                    "let stride=((first+255u)/256u)*256u;",
                    "let stride=((cohort.base.r4*320u+255u)/256u)*256u;",
                );
                source = source.replace(
                    "let base=member*1280u+col*2u;",
                    "let base=(cohort_prefix+member)*1280u+col*2u;",
                );
                source = source.replace("hidden[i]=", "hidden[cohort_prefix*320u+i]=");
            }
            Self::Image { .. } => {
                // This is a distinct authenticated whole-ring/image layout.
                // Retain the original exact per-role extent and source checks.
                source = source.replace("arrayLength(&raw)!=p.r5", "arrayLength(&raw)!=8388608u");
                source = source.replace("p.r0!=0u", "p.r0+p.r5>arrayLength(&raw)");
                source = source.replace("p.r1>p.r5", "p.r1>p.r5");
                source = source.replace("let src=addresses[p.role*512u+map[4u+p.group]];", "let src=addresses[p.role*512u+map[4u+p.group]];\n    if cohort_slot==8u && src.w!=0u {atomicOr(&fault[0],128u);return;}");
                source = source.replace("strata_up_address=up.xy;host_up=up.w==1u;", "if cohort_slot==8u && up.w!=0u {atomicOr(&fault[0],128u);return;}strata_up_address=up.xy;host_up=up.w==1u;");
            }
        }
        source
    }
}
impl KernelMetadata for CohortOp {
    fn name(&self) -> &'static str {
        match self {
            Self::Gather => "StrataExpertCohortGather",
            Self::Image { down: false, .. } => if cfg!(feature="strata-dequant-direct8") {
                "StrataExpertCohortGuDirect8Image"
            } else if cfg!(feature="strata-dequant-quad32") {
                "StrataExpertCohortGuQuad32Image"
            } else { self.original_name() },
            Self::Image { down: true, .. } => if cfg!(feature="strata-dequant-direct8") {
                "StrataExpertCohortDownDirect8Image"
            } else if cfg!(feature="strata-dequant-quad32") {
                "StrataExpertCohortDownQuad32Image"
            } else { self.original_name() },
            Self::Activate => "StrataExpertCohortActivate",
        }
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
            .info((*self,cfg!(feature="strata-dequant-quad32"),cfg!(feature="strata-dequant-direct8")))
            .cube_dim(CubeDim::new_1d(self.lanes()))
    }
    fn address_type(&self) -> StorageType {
        StorageType::Scalar(ElemType::UInt(UIntKind::U32))
    }
    fn native_release_after(&self) -> bool {
        matches!(self, Self::Image { down: true, .. })
    }
}
impl CubeTask<AutoCompiler> for CohortOp {
    fn compile(
        &self,
        _: &mut AutoCompiler,
        _: &cubecl::WgpuCompilationOptions,
        _: ExecutionMode,
        _: StorageType,
    ) -> std::result::Result<CompiledKernel<AutoCompiler>, CompilationError> {
        Ok(CompiledKernel {
            entrypoint_name: "main".into(),
            debug_name: None,
            source: self.selected_source(),
            repr: None,
            cube_dim: CubeDim::new_1d(self.lanes()),
            debug_info: None,
        })
    }
}
fn view(root: &Handle, first: u64, bytes: u64) -> Result<Handle> {
    if first % 256 != 0
        || bytes == 0
        || bytes % 256 != 0
        || first
            .checked_add(bytes)
            .is_none_or(|end| end > root.size_in_used())
    {
        return Err(Error::InvalidBufferLength);
    }
    Ok(root
        .clone()
        .offset_start(first)
        .offset_end(root.size_in_used() - first - bytes))
}
fn launch(
    client: &ComputeClient<WgpuRuntime>,
    op: CohortOp,
    grid: [usize; 3],
    mut args: KernelArguments,
    words: &[u32; WORDS],
) -> Result<()> {
    #[cfg(all(feature="strata-dequant-quad32",not(feature="strata-dequant-direct8")))]
    if matches!(op,CohortOp::Image{..}) {super::quad32::check_hardware(client)?;}
    args = args.with_info(MetadataBindingInfo::custom(
        words
            .chunks_exact(2)
            .map(|v| u64::from(v[0]) | (u64::from(v[1]) << 32))
            .collect(),
    ));
    unsafe {
        client.launch_unchecked(
            Box::new(op),
            CubeCount::Static(grid[0] as u32, grid[1] as u32, grid[2] as u32),
            args,
        );
    }
    Ok(())
}
pub(crate) fn sources() -> Vec<CompiledKernel<AutoCompiler>> {
    let mut ops = vec![CohortOp::Gather, CohortOp::Activate];
    ops.extend(
        [q::IQ2_S, q::IQ3_S, q::IQ3_XXS, q::IQ4_XS]
            .map(|dtype| CohortOp::Image { dtype, down: false }),
    );
    ops.extend([q::IQ4_NL, q::Q2_0].map(|dtype| CohortOp::Image { dtype, down: true }));
    ops.into_iter()
        .map(|op| {
            op.compile(
                &mut AutoCompiler::Wgsl(Default::default()),
                &Default::default(),
                ExecutionMode::Unchecked,
                StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
            )
            .map(|mut compiled| {compiled.source=op.source();compiled.cube_dim=CubeDim::new_1d(256);compiled})
            .unwrap()
        })
        .collect()
}
/// Structural metadata authority shared by the real matrix launch and CPU controls.
/// Original map contents and per-role physical source authority remain GPU/read-lease checks.
pub(crate) fn validate_words(words: &[u32; WORDS]) -> Result<usize> {
    if words[0] == 0
        || words[0] > 4096
        || words[9] == 0
        || words[9] > 8
        || words[4] > 512 - words[9]
        || words[6] == 0
        || words[7] != words[6]
        || words[6] > 1048576
        || words[8] <= 2 * words[6]
        || u64::from(words[8]) * 4 * 8 > allocation::RAW_RING_BYTES
        || words[12] > 1
        || words[13..16] != [0; 3]
    {
        return Err(Error::InvalidBufferLength);
    }
    let mut rows = 0;
    let mut maximum = 0;
    let mut raw = [false; 8];
    for i in 0..MAX {
        let [group, members, prefix, slot] =
            <[u32; 4]>::try_from(&words[16 + i * 4..20 + i * 4]).unwrap();
        if i >= words[9] as usize {
            if [group, members, prefix, slot] != [0; 4] {
                return Err(Error::InvalidBufferLength);
            }
            continue;
        }
        if group != words[4] + i as u32
            || members == 0
            || members >= 32
            || members > words[0]
            || prefix != rows
            || slot > 8
            || (slot < 8 && raw[slot as usize])
        {
            return Err(Error::InvalidBufferLength);
        }
        if slot < 8 {
            raw[slot as usize] = true;
        }
        let tile = if members <= 8 { 8 } else { 16 };
        rows += members.div_ceil(tile) * tile;
        maximum = maximum.max(members);
    }
    if rows > 4096 || maximum != words[10] {
        return Err(Error::InvalidBufferLength);
    }
    Ok(rows as usize)
}
#[allow(clippy::too_many_arguments)]
pub(super) fn enqueue_small(
    client: &ComputeClient<WgpuRuntime>,
    _graph: BoundGraph<'_, '_>,
    _sources: &Sources,
    _layer: usize,
    geometry: q::grouped_moe::Geometry,
    owners: &[Handle; 7],
    fault: &Handle,
    count: usize,
    map: &[u32],
    stage: &[Handle; 3],
    layer_sources: &crate::strata_global_sources::PromptLayer<'_, '_, '_>,
    ring: &mut miss_ring::Ring,
    first: usize,
    total: usize,
) -> Result<Option<usize>> {
    // Larger genuine cache-loan roots retain their original whole-owner path.
    // Return before any ring issue/consume; do not slice that authority.
    if stage[2].size_in_used() != allocation::RAW_RING_BYTES || map[516 + first] >= 32 {
        return Ok(None);
    }
    let roles = geometry.projections().map(|p| p.expert_bytes());
    let stride = roles.iter().sum::<usize>();
    // Fill original miss-only ring first. Cohorts never include an unissued
    // slot, and peek does not consume/authorize its overwrite.
    while let Some((next, slot)) = ring.issue_next()? {
        layer_sources.stage_group_on(
            client,
            map[4 + next] as usize,
            roles,
            &stage[2],
            slot.unwrap_or(RAW_SLOTS) * stride,
        )?;
    }
    let mut slots = Zeroizing::new([8u8; MAX]);
    let mut end = first;
    while end < total && end < first + MAX && map[516 + end] < 32 && ring.peek_slot(end)?.is_some() {
        slots[end-first] = ring.peek_slot(end)?.unwrap().unwrap_or(RAW_SLOTS) as u8;
        end += 1;
    }
    enqueue_unit(client, geometry, owners, fault, count, map, stage, layer_sources, first, &slots[..end-first])?;
    // Preserve the original fallback ring and the complete post-matrix COPY cut.
    for group in first..end { ring.consumed(group)?; }
    while let Some((next, slot)) = ring.issue_next()? {
        layer_sources.stage_group_on(client, map[4+next] as usize, roles, &stage[2], slot.unwrap_or(RAW_SLOTS)*stride)?;
    }
    Ok(Some(end))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn enqueue_unit(
    client: &ComputeClient<WgpuRuntime>,
    geometry: q::grouped_moe::Geometry,
    owners: &[Handle; 7], fault: &Handle, count: usize, map: &[u32],
    stage: &[Handle; 3],
    layer_sources: &crate::strata_global_sources::PromptLayer<'_, '_, '_>,
    first: usize, slots: &[u8],
) -> Result<()> {
    if slots.is_empty() || slots.len()>MAX || first.checked_add(slots.len()).is_none_or(|n|n>512)
        || stage[2].size_in_used()!=allocation::RAW_RING_BYTES { return Err(Error::InvalidBufferLength); }
    let roles=geometry.projections().map(|p|p.expert_bytes());
    let stride=roles.iter().sum::<usize>();
    let mut words = Zeroizing::new([0u32; WORDS]);
    words[0] = count as u32;
    words[1] = 2560;
    words[2] = 1280;
    words[4] = first as u32;
    words[6] = (roles[0] / 4) as u32;
    words[7] = (roles[1] / 4) as u32;
    words[8] = (stride / 4) as u32;
    let mixed = super::super::mixed_input_image(&owners[1], count, 0, 2560, false);
    words[12] = u32::from(mixed.is_some());
    let n=slots.len();
    let mut rows=0;
    for (i,&slot) in slots.iter().enumerate() {
        let group=first+i;
        let members=map[516+group] as usize;
        if members==0 || members>=32 { return Err(Error::InvalidBufferLength); }
        let tile=if members<=8 {8}else{16};
        let padded=members.div_ceil(tile)*tile;
        if rows+padded>groups::CAPACITY { return Err(Error::InvalidBufferLength); }
        words[16+i*4..20+i*4].copy_from_slice(&[group as u32,members as u32,rows as u32,u32::from(slot)]);
        rows+=padded;
        words[10]=words[10].max(members as u32);
    }
    words[9]=n as u32;
    validate_words(&words)?;
    crate::strata_prefill_profile::count("expert_narrow_global_groups", n as u64);
    #[cfg(any(not(feature = "strata-small-quantized-matrix"),feature="strata-global-image-experts"))]
    {
    crate::strata_prefill_profile::count(
        "expert_dequant_record32_records",
        (n * (2560 * 1280 + 640 * 2560) / 32) as u64,
    );
    #[cfg(all(feature="strata-dequant-quad32",not(feature="strata-dequant-direct8")))]
    crate::strata_prefill_profile::count("expert_dequant_quad32_records",
        (n*(2560*1280+640*2560)/32) as u64);
    #[cfg(feature="strata-dequant-direct8")]
    crate::strata_prefill_profile::count("expert_dequant_direct8_records",
        (n*(2560*1280+640*2560)/32) as u64);

    }

    let x = view(&stage[0], 0, X_BYTES)?;
    let h = view(&stage[0], X_BYTES, H_BYTES)?;
    if stage[1].size_in_used() != storage::WORKING_BYTES
        || IMAGE_BYTES * MAX as u64 > stage[1].size_in_used()
    {
        return Err(Error::InvalidBufferLength);
    }
    let args = KernelArguments::new().with_buffers(vec![
        owners[0].clone().binding(),
        mixed.unwrap_or_else(|| owners[1].clone()).binding(),
        x.clone().binding(),
        fault.clone().binding(),
    ]);
    launch(
        client,
        CohortOp::Gather,
        [words[10] as usize * 5, 1, n],
        args,
        &words,
    )?;
    let ps = geometry.projections();
    for down in [false, true] {
        let p = ps[if down { 2 } else { 0 }];
        words[1] = if down { 640 } else { 2560 };
        words[2] = if down { 2560 } else { 1280 };
        words[3] = p.row_bytes() as u32;
        words[5] = if down { 2 } else { 0 };
        let mut args = KernelArguments::new()
            .with_buffer(layer_sources.prototype(if down { 2 } else { 0 })?.binding());
        if !down {
            args = args.with_buffer(layer_sources.prototype(1)?.binding());
        }
        args = args.with_buffers(vec![
            owners[0].clone().binding(),
            stage[2].clone().binding(),
            stage[1].clone().binding(),
            fault.clone().binding(),
        ]);
        args = crate::strata_intel_core::bind(args, owners[0].stream)?;
        args = layer_sources.bind_image(args)?;
        #[cfg(all(feature = "strata-small-quantized-matrix",not(feature="strata-global-image-experts")))]
        compact::launch(client, compact::CompactOp { dtype: p.dtype(), down }, args, &words)?;
        #[cfg(any(not(feature = "strata-small-quantized-matrix"),feature="strata-global-image-experts"))]
        launch(
            client,
            CohortOp::Image {
                dtype: p.dtype(),
                down,
            },
            [
                (words[1] as usize * words[2] as usize / Op::selected_dequant_values()).div_ceil(Op::selected_dequant_lanes()),
                1,
                n,
            ],
            args,
            &words,
        )?;
    }
    for down in [false, true] {
        words[1] = if down { 640 } else { 2560 };
        words[2] = if down { 2560 } else { 1280 };
        #[cfg(any(not(feature = "strata-small-quantized-matrix"),feature="strata-global-image-experts"))]
        { words[3] = 0; }
        #[cfg(all(feature = "strata-small-quantized-matrix",not(feature="strata-global-image-experts")))]
        { words[3] = ps[if down { 2 } else { 0 }].row_bytes() as u32; }
        words[5] = if down { 2 } else { 0 };
        #[cfg(all(feature = "strata-small-quantized-matrix",not(feature="strata-global-image-experts")))]
        for tile in [8usize, 16, 32] {
            if (0..n).any(|i| compact::tile_for(words[17 + i * 4] as usize) == tile) {
                words[11] = tile as u32;
                crate::strata_matrix::quant_cohort::launch(
                    client,
                    [if down { h.clone() } else { x.clone() }, stage[1].clone(),
                        if down { owners[3].clone() } else { owners[2].clone() },
                        owners[0].clone(), owners[6].clone(), fault.clone(),
                        crate::strata_intel_core::table_handle(owners[0].stream)?],
                    &words, down, tile, ps[if down { 2 } else { 0 }].dtype(),
                )?;
                crate::strata_prefill_profile::count(
                    if down { "expert_small_quant_down_launches" } else { "expert_small_quant_gu_launches" }, 1);
            }
        }
        #[cfg(any(not(feature = "strata-small-quantized-matrix"),feature="strata-global-image-experts"))]
        for tile in [8usize, 16] {
            if (0..n).any(|i| (words[17 + i * 4] <= 8) == (tile == 8)) {
                words[11] = tile as u32;
                #[cfg(feature="strata-small-direct-publish")]
                let product=if cfg!(feature="strata-global-small-products") && tile==16 {
                    #[cfg(feature="strata-global-small-products")]
                    {if cfg!(feature="strata-small-full-walk") {
                        #[cfg(feature="strata-small-full-walk")]
                        {crate::strata_matrix::cohorts::small_full_walk::launch}
                        #[cfg(not(feature="strata-small-full-walk"))]
                        {crate::strata_matrix::cohorts::global_small::launch}
                    }else{crate::strata_matrix::cohorts::global_small::launch}}
                    #[cfg(not(feature="strata-global-small-products"))]
                    {crate::strata_matrix::cohorts::small_direct::launch}
                } else {crate::strata_matrix::cohorts::small_direct::launch};
                #[cfg(all(feature = "strata-expert-ordered-output",not(feature="strata-small-direct-publish")))]
                let product = if down { crate::strata_matrix::cohorts::launch_ordered }
                    else { crate::strata_matrix::cohorts::launch };
                #[cfg(all(not(feature = "strata-expert-ordered-output"),not(feature="strata-small-direct-publish")))]
                let product = crate::strata_matrix::cohorts::launch;
                product(
                    client,
                    [
                        if down { h.clone() } else { x.clone() },
                        stage[1].clone(),
                        if down {
                            owners[3].clone()
                        } else {
                            owners[2].clone()
                        },
                        owners[0].clone(),
                        owners[6].clone(),
                        fault.clone(),
                    ],
                    &words,
                    down,
                    tile,
                )?;
            }
        }
        if !down {
            words[1] = 2560;
            words[2] = 1280;
            words[3] = 0;
            words[5] = 0;
            words[11] = 0;
            let args = KernelArguments::new().with_buffers(vec![
                owners[0].clone().binding(),
                owners[2].clone().binding(),
                h.clone().binding(),
                owners[6].clone().binding(),
                fault.clone().binding(),
            ]);
            launch(
                client,
                CohortOp::Activate,
                [(words[10] as usize * 320).div_ceil(256), 1, n],
                args,
                &words,
            )?;
        }
    }
    for (name, value) in [
        ("expert_cohort_groups", n as u64),
        ("expert_cohort_batches", 1),
        (if (cfg!(feature = "strata-small-quantized-matrix") && !cfg!(feature="strata-global-image-experts")) { "expert_small_quant_compact_slots_used" } else { "expert_cohort_image_slots_used" }, n as u64),
        (if (cfg!(feature = "strata-small-quantized-matrix") && !cfg!(feature="strata-global-image-experts")) { "expert_small_quant_compact_slots_capacity" } else { "expert_cohort_image_slots_capacity" }, MAX as u64),
        (if (cfg!(feature = "strata-small-quantized-matrix") && !cfg!(feature="strata-global-image-experts")) { "expert_small_quant_compact_bytes_capacity" } else { "expert_cohort_image_bytes_capacity" },
            MAX as u64 * if (cfg!(feature = "strata-small-quantized-matrix") && !cfg!(feature="strata-global-image-experts")) { stride as u64 } else { IMAGE_BYTES }),
        ("expert_cohort_x_owner_bytes", X_BYTES),
        ("expert_cohort_h_owner_bytes", H_BYTES),
        ("expert_cohort_logical_phases", (n * 6) as u64),
        ("expert_cohort_fp16_image_bytes", if (cfg!(feature = "strata-small-quantized-matrix") && !cfg!(feature="strata-global-image-experts")) { 0 } else { n as u64 * IMAGE_BYTES }),
    ] {
        crate::strata_prefill_profile::count(name, value);
    }
    #[cfg(all(feature = "strata-small-quantized-matrix",not(feature="strata-global-image-experts")))]
    crate::strata_prefill_profile::count("expert_small_quant_working_owner_bytes", storage::WORKING_BYTES);
    Ok(())
}

#[cfg(test)]
#[path = "strata_cohort_grammar_cpu_tests.rs"]
mod grammar_cpu_tests;
#[cfg(test)]
mod tests {
    use super::*;
    pub(super) fn fixture() -> [u32; WORDS] {
        let mut w = [0; WORDS];
        w[0] = 32;
        w[1] = 2560;
        w[2] = 1280;
        w[4] = 7;
        w[6] = 100;
        w[7] = 100;
        w[8] = 300;
        w[9] = 8;
        w[10] = 31;
        let mut prefix = 0;
        for (i, members) in [1u32, 3, 7, 8, 9, 16, 17, 31].into_iter().enumerate() {
            w[16 + i * 4..20 + i * 4].copy_from_slice(&[7 + i as u32, members, prefix, i as u32]);
            let tile = if members <= 8 { 8 } else { 16 };
            prefix += members.div_ceil(tile) * tile;
        }
        w
    }
    #[test]
    fn cohort_exact_original_roots_fit_eight_images_and_keep_x_hidden_gu_disjoint() {
        assert_eq!(IMAGE_BYTES, 9_830_400);
        assert_eq!(IMAGE_BYTES * 8, 75 * 1024 * 1024);
        assert!(IMAGE_BYTES * 8 <= storage::WORKING_BYTES);
        assert_eq!(X_BYTES + H_BYTES, 25 * 1024 * 1024);
        assert!(X_BYTES + H_BYTES <= storage::WORKING_BYTES);
        let w = fixture();
        let rows = validate_words(&w).unwrap();
        let mut x = vec![false; rows];
        for i in 0..w[9] as usize {
            let n = w[17 + i * 4];
            let p = w[18 + i * 4] as usize;
            let tile = if n <= 8 { 8 } else { 16 };
            let padded = n.div_ceil(tile) * tile;
            for at in p..p + padded as usize {
                assert!(!x[at]);
                x[at] = true;
            }
            assert!((p + padded as usize) * 2560 * 2 <= X_BYTES as usize);
            assert!((p + padded as usize) * 640 * 2 <= H_BYTES as usize);
            assert!((p + padded as usize) * 1280 * 4 <= groups::GU_BYTES as usize);
        }
        assert!(x.into_iter().all(|v| v));
    }
    #[test]
    fn cohort_metadata_rejects_aliases_spare_rows_oversize_slots_and_changed_original_groups() {
        let baseline = fixture();
        assert!(validate_words(&baseline).is_ok());
        for (at, value) in [
            (0, 0),
            (0, 4097),
            (9, 0),
            (9, 9),
            (10, 30),
            (17, 0),
            (17, 32),
            (18, 8),
            (19, 9),
            (20, 88),
            (23, 0),
            (47, 9),
            (6, 0),
            (7, 99),
            (8, 199),
            (13, 1),
            (4, u32::MAX),
            (6, u32::MAX),
            (18, u32::MAX - 7),
        ] {
            let mut w = baseline;
            w[at] = value;
            assert!(validate_words(&w).is_err(), "field {at} value {value}");
        }
        let mut w = baseline;
        w[9] = 7;
        assert!(validate_words(&w).is_err());
        w[44..48].fill(0);
        w[10] = 17;
        assert!(validate_words(&w).is_ok());
    }
    #[test]
    fn selected_quad_image_identity_keeps_original_explicit_fixture_names() {
        for down in [false,true] {
            let op = CohortOp::Image { dtype:q::IQ3_S,down };
            let original = if down { "StrataExpertCohortDownImage" } else { "StrataExpertCohortGuImage" };
            assert_eq!(op.original_name(),original);
            let quad = if down { "StrataExpertCohortDownQuad32Image" } else { "StrataExpertCohortGuQuad32Image" };
            let direct=if down {"StrataExpertCohortDownDirect8Image"} else {"StrataExpertCohortGuDirect8Image"};
            assert_eq!(op.name(),if cfg!(feature="strata-dequant-direct8") {direct} else if cfg!(feature="strata-dequant-quad32") {quad} else {original});
            assert_eq!(CohortOp::Gather.name(),CohortOp::Gather.original_name());
            assert_eq!(CohortOp::Activate.name(),CohortOp::Activate.original_name());
        }
    }
    #[test]
    fn actual_cohort_sources_validate_and_keep_original_source_rounding_faults_and_complete_owners()
    {
        for op in [
            CohortOp::Gather,
            CohortOp::Activate,
            CohortOp::Image {
                dtype: q::IQ3_S,
                down: false,
            },
            CohortOp::Image {
                dtype: q::Q2_0,
                down: true,
            },
        ] {
            let s = op.source();
            let module = wgpu::naga::front::wgsl::parse_str(&s).unwrap();
            wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
            assert!(s.contains("cohort_begin(id.z)"));
            assert!(s.contains("map[516u+p.group]!=v.y"));
            assert!(s.contains("padded>4096u-min(v.z,4096u)"));
            assert!(s.contains("v.w>8u"));
            if let CohortOp::Image { .. } = op {
                assert!(s.contains("arrayLength(&raw)!=8388608u"));
                assert!(s.contains("arrayLength(&packed)!=PROMPT_WORKING_WORDS/4u"));
                assert!(s.contains("src.z!=bytes/4u"));
                assert!(s.contains("pack2x16float(values.a.xy)"));
                assert!(s.contains("cohort_slot==8u"));
                assert!(s.contains("p.r0+p.r5>arrayLength(&raw)"));
            }
            if op == CohortOp::Activate {
                assert!(s.contains("hidden[cohort_prefix*320u+i]"));
                assert!(s.contains("gu[base]=0.0"));
            }
        }
    }
    #[test]
    fn grouped_real_grids_initialize_each_padding_word_once_for_both_original_families() {
        for members in [1usize, 3, 7, 8, 9, 15, 16, 17, 31] {
            for largest in members..32 {
                for row_words in [1280usize, 320] {
                    let tile = if members <= 8 { 8 } else { 16 };
                    let padded = members.div_ceil(tile) * tile;
                    let stride = (largest * row_words).div_ceil(256) * 256;
                    let mut visits = vec![0u8; (padded - members) * row_words];
                    for id in 0..stride {
                        for at in (members * row_words + id..padded * row_words).step_by(stride) {
                            visits[at - members * row_words] += 1;
                        }
                    }
                    assert!(visits.into_iter().all(|n| n == 1));
                }
            }
        }
    }
}
