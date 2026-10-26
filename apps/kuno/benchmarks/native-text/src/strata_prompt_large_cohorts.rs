//! Distinct large256..4096-member expert cohorts. Original map/source/math guards, real parent
//! owners and native range/timeline hazards remain authoritative.
use super::*;
use zeroize::Zeroizing;
pub(crate) const MAX: usize = 8;
pub(crate) const WORDS: usize = 48;
pub(crate) const X_BYTES: u64 = 20 * 1024 * 1024;
pub(crate) const H_BYTES: u64 = 5 * 1024 * 1024;
pub(crate) const IMAGE_BYTES: u64 = (DQ_WORDS * 4) as u64;
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(crate) enum LargeOp {
    Gather,
    Image { dtype: u32, down: bool },
    Activate,
}
impl LargeOp {
    fn lanes(self) -> u32 {
        if cfg!(feature = "strata-dequant-direct8") && matches!(self, Self::Image { .. }) {
            32
        } else {
            256
        }
    }
    /// Original explicit fixture identity pairs with source() and /32 grids.
    pub(crate) fn original_name(self) -> &'static str {
        match self {
            Self::Gather => "StrataExpertLargeCohortGather",
            Self::Image { down: false, .. } => "StrataExpertLargeCohortGuImage",
            Self::Image { down: true, .. } => "StrataExpertLargeCohortDownImage",
            Self::Activate => "StrataExpertLargeCohortActivate",
        }
    }
    pub(super) fn selected_source(self) -> String {
        let source = self.source();
        #[cfg(feature = "strata-dequant-direct8")]
        if let Self::Image { dtype, down } = self {
            return super::direct8::source(source, dtype, down);
        }
        #[cfg(feature = "strata-dequant-quad32")]
        if let Self::Image { down, .. } = self {
            return super::quad32::source(source, down);
        }
        source
    }
    pub(crate) fn source(self) -> String {
        let old = match self {
            Self::Gather => Op::Gather,
            Self::Activate => Op::ActivateCheckGu,
            Self::Image { dtype, down } => Op::Dequant { dtype, down },
        };
        let mut source = old.source();
        if matches!(self, Self::Gather | Self::Activate) {
            // Both producers initialize only this group's charged Pad32 suffix.
            // Charge the same Pad32 extent in the planner, WGSL guards,
            // matrix views and native grammar; surplus rows belong to no neighbor.
            let previous = "let tile=select(select(16u,8u,real<=8u),32u,real>=32u);";
            assert_eq!(source.matches(previous).count(), 1);
            source = source.replace(previous, "let tile=32u;");
        }

        source = source.replace("var<uniform> p:Params;", "var<uniform> cohort:CohortInfo;\nvar<private> p:Params;\nvar<private> cohort_prefix:u32;\nvar<private> cohort_slot:u32;");
        source.push_str(include_str!("strata_prompt_large_cohort_header.wgsl"));
        source = source.replace("fn main(@builtin(global_invocation_id) id:vec3<u32>) {", "fn main(@builtin(global_invocation_id) id:vec3<u32>) {\n    if !cohort_begin(id.z) {return;}");
        #[cfg(feature="strata-layer-gather")]
        if matches!(self,Self::Activate) {
            source=source.replace("fn main(@builtin(global_invocation_id) id:vec3<u32>) {", "fn main(@builtin(global_invocation_id) actual_id:vec3<u32>) {\n    var id=actual_id;\n    if cohort.base.r7==1u {\n        if cohort.base.r3<1u || cohort.base.r3>8u {atomicOr(&fault[0],128u);return;}\n        var found=false;\n        for(var s=0u;s<8u;s++){if s<cohort.base.r3 {let v=cohort.records[s];if v.y<256u||v.y>4096u||v.z>4096u||((v.y+31u)/32u)*32u>4096u-v.z {atomicOr(&fault[0],128u);return;}let first=v.z*320u;let end=first+((v.y+31u)/32u)*32u*320u;if actual_id.x>=first&&actual_id.x<end {id=vec3<u32>(actual_id.x-first,0u,s);found=true;}}}\n        if !found {return;}\n    }");
        }
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
impl KernelMetadata for LargeOp {
    fn name(&self) -> &'static str {
        match self {
            Self::Gather => "StrataExpertLargeCohortGather",
            Self::Image { down: false, .. } => {
                if cfg!(feature = "strata-dequant-direct8") {
                    "StrataExpertLargeCohortGuDirect8Image"
                } else if cfg!(feature = "strata-dequant-quad32") {
                    "StrataExpertLargeCohortGuQuad32Image"
                } else {
                    self.original_name()
                }
            }
            Self::Image { down: true, .. } => {
                if cfg!(feature = "strata-dequant-direct8") {
                    "StrataExpertLargeCohortDownDirect8Image"
                } else if cfg!(feature = "strata-dequant-quad32") {
                    "StrataExpertLargeCohortDownQuad32Image"
                } else {
                    self.original_name()
                }
            }
            Self::Activate => "StrataExpertLargeCohortActivate",
        }
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
            .info((
                *self,
                cfg!(feature = "strata-dequant-quad32"),
                cfg!(feature = "strata-dequant-direct8"),
            ))
            .cube_dim(CubeDim::new_1d(self.lanes()))
    }
    fn address_type(&self) -> StorageType {
        StorageType::Scalar(ElemType::UInt(UIntKind::U32))
    }
    fn native_release_after(&self) -> bool {
        matches!(self, Self::Image { down: true, .. })
    }
}
impl CubeTask<AutoCompiler> for LargeOp {
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
pub(crate) fn launch(
    client: &ComputeClient<WgpuRuntime>,
    op: LargeOp,
    grid: [usize; 3],
    mut args: KernelArguments,
    words: &[u32; WORDS],
) -> Result<()> {
    #[cfg(all(
        feature = "strata-dequant-quad32",
        not(feature = "strata-dequant-direct8")
    ))]
    if matches!(op, LargeOp::Image { .. }) {
        super::quad32::check_hardware(client)?;
    }
    if matches!(op,LargeOp::Activate) {crate::strata_prefill_profile::count("expert_large_cohort_activation_workgroups",(grid[0]*grid[1]*grid[2]) as u64);}
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
    let mut ops = vec![LargeOp::Gather, LargeOp::Activate];
    ops.extend(
        [q::IQ2_S, q::IQ3_S, q::IQ3_XXS, q::IQ4_XS]
            .map(|dtype| LargeOp::Image { dtype, down: false }),
    );
    ops.extend([q::IQ4_NL, q::Q2_0].map(|dtype| LargeOp::Image { dtype, down: true }));
    ops.into_iter()
        .map(|op| {
            op.compile(
                &mut AutoCompiler::Wgsl(Default::default()),
                &Default::default(),
                ExecutionMode::Unchecked,
                StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
            )
            .map(|mut compiled| {
                compiled.source = op.selected_source();
                compiled.cube_dim = CubeDim::new_1d(op.lanes());
                compiled
            })
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
        || words[13] > u32::from(cfg!(feature="strata-layer-gather")) || words[14..16] != [0; 2]
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
            || !(256..=4096).contains(&members)
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
        let tile = 32;
        rows += members.div_ceil(tile) * tile;
        maximum = maximum.max(members);
    }
    if rows > 4096 || maximum != words[10] {
        return Err(Error::InvalidBufferLength);
    }
    Ok(rows as usize)
}
#[allow(clippy::too_many_arguments)]
pub(super) fn enqueue_unit(
    client: &ComputeClient<WgpuRuntime>,
    geometry: q::grouped_moe::Geometry,
    owners: &[Handle; 7],
    fault: &Handle,
    count: usize,
    map: &[u32],
    stage: &[Handle; 3],
    layer_sources: &crate::strata_global_sources::PromptLayer<'_, '_, '_>,
    first: usize,
    slots: &[u8],
) -> Result<()> {
    if slots.is_empty()
        || slots.len() > MAX
        || first.checked_add(slots.len()).is_none_or(|n| n > 512)
        || stage[2].size_in_used() != allocation::RAW_RING_BYTES
    {
        return Err(Error::InvalidBufferLength);
    }
    let roles = geometry.projections().map(|p| p.expert_bytes());
    let stride = roles.iter().sum::<usize>();
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
    let n = slots.len();
    let mut rows = 0;
    for (i, &slot) in slots.iter().enumerate() {
        let group = first + i;
        let members = map[516 + group] as usize;
        if members == 0 || !(256..=4096).contains(&members) {
            return Err(Error::InvalidBufferLength);
        }
        let tile = 32;
        let padded = members.div_ceil(tile) * tile;
        if rows + padded > groups::CAPACITY {
            return Err(Error::InvalidBufferLength);
        }
        words[16 + i * 4..20 + i * 4].copy_from_slice(&[
            group as u32,
            members as u32,
            rows as u32,
            u32::from(slot),
        ]);
        rows += padded;
        words[10] = words[10].max(members as u32);
    }
    words[9] = n as u32;
    words[13]=u32::from(cfg!(feature="strata-layer-gather"));
    validate_words(&words)?;
    #[cfg(any(
        not(feature = "strata-small-quantized-matrix"),
        feature = "strata-global-image-experts"
    ))]
    {
        crate::strata_prefill_profile::count(
            "expert_dequant_record32_records",
            (n * (2560 * 1280 + 640 * 2560) / 32) as u64,
        );
        #[cfg(all(
            feature = "strata-dequant-quad32",
            not(feature = "strata-dequant-direct8")
        ))]
        crate::strata_prefill_profile::count(
            "expert_dequant_quad32_records",
            (n * (2560 * 1280 + 640 * 2560) / 32) as u64,
        );
        #[cfg(feature = "strata-dequant-direct8")]
        crate::strata_prefill_profile::count(
            "expert_dequant_direct8_records",
            (n * (2560 * 1280 + 640 * 2560) / 32) as u64,
        );
    }

    let x = {
        #[cfg(feature="strata-layer-gather")] {super::super::layer_gather::x()?}
        #[cfg(not(feature="strata-layer-gather"))] {view(&stage[0],0,X_BYTES)?}
    };
    let h = view(&stage[0], X_BYTES, H_BYTES)?;
    if stage[1].size_in_used() != storage::WORKING_BYTES
        || IMAGE_BYTES * MAX as u64 > stage[1].size_in_used()
    {
        return Err(Error::InvalidBufferLength);
    }
    #[cfg(not(feature="strata-layer-gather"))]
    {
    let args = KernelArguments::new().with_buffers(vec![
        owners[0].clone().binding(),
        mixed.unwrap_or_else(|| owners[1].clone()).binding(),
        x.clone().binding(),
        fault.clone().binding(),
    ]);
    launch(
        client,
        LargeOp::Gather,
        [words[10] as usize * 5, 1, n],
        args,
        &words,
    )?;
    }
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
        #[cfg(all(
            feature = "strata-small-quantized-matrix",
            not(feature = "strata-global-image-experts")
        ))]
        compact::launch(
            client,
            compact::CompactOp {
                dtype: p.dtype(),
                down,
            },
            args,
            &words,
        )?;
        #[cfg(any(
            not(feature = "strata-small-quantized-matrix"),
            feature = "strata-global-image-experts"
        ))]
        launch(
            client,
            LargeOp::Image {
                dtype: p.dtype(),
                down,
            },
            [
                (words[1] as usize * words[2] as usize / Op::selected_dequant_values())
                    .div_ceil(Op::selected_dequant_lanes()),
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
        #[cfg(any(
            not(feature = "strata-small-quantized-matrix"),
            feature = "strata-global-image-experts"
        ))]
        {
            words[3] = 0;
        }
        #[cfg(all(
            feature = "strata-small-quantized-matrix",
            not(feature = "strata-global-image-experts")
        ))]
        {
            words[3] = ps[if down { 2 } else { 0 }].row_bytes() as u32;
        }
        words[5] = if down { 2 } else { 0 };
        for tile in [128,8,16,32] {
            if crate::strata_matrix::cohorts::large::part_rows(&words,tile)==0 {continue;}
            words[11]=tile as u32;
            crate::strata_matrix::cohorts::large::launch(client,[
                if down {h.clone()}else{x.clone()},stage[1].clone(),
                if down {owners[3].clone()}else{owners[2].clone()},
                owners[0].clone(),owners[6].clone(),fault.clone()],&words,down,tile)?;
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
                LargeOp::Activate,
                [if words[13]==1 {(rows*320).div_ceil(256)}else{(words[10] as usize*320).div_ceil(256)}, 1, if words[13]==1 {1}else{n}],
                args,
                &words,
            )?;
        }
    }
    for (name, value) in [
        ("expert_large_cohort_groups", n as u64),
        ("expert_cohort_groups", n as u64),
        ("expert_cohort_batches", 1),
        ("expert_large_cohort_batches", 1),
        ("expert_large_cohort_padded_rows", rows as u64),
        (
            if (cfg!(feature = "strata-small-quantized-matrix")
                && !cfg!(feature = "strata-global-image-experts"))
            {
                "expert_small_quant_compact_slots_used"
            } else {
                "expert_cohort_image_slots_used"
            },
            n as u64,
        ),
        (
            if (cfg!(feature = "strata-small-quantized-matrix")
                && !cfg!(feature = "strata-global-image-experts"))
            {
                "expert_small_quant_compact_slots_capacity"
            } else {
                "expert_cohort_image_slots_capacity"
            },
            MAX as u64,
        ),
        (
            if (cfg!(feature = "strata-small-quantized-matrix")
                && !cfg!(feature = "strata-global-image-experts"))
            {
                "expert_small_quant_compact_bytes_capacity"
            } else {
                "expert_cohort_image_bytes_capacity"
            },
            MAX as u64
                * if (cfg!(feature = "strata-small-quantized-matrix")
                    && !cfg!(feature = "strata-global-image-experts"))
                {
                    stride as u64
                } else {
                    IMAGE_BYTES
                },
        ),
        ("expert_cohort_x_owner_bytes", if cfg!(feature="strata-layer-gather") {0}else{X_BYTES}),
        ("expert_cohort_h_owner_bytes", H_BYTES),
        ("expert_cohort_logical_phases", (n * 6) as u64),
        (
            "expert_cohort_fp16_image_bytes",
            if (cfg!(feature = "strata-small-quantized-matrix")
                && !cfg!(feature = "strata-global-image-experts"))
            {
                0
            } else {
                n as u64 * IMAGE_BYTES
            },
        ),
    ] {
        crate::strata_prefill_profile::count(name, value);
    }
    #[cfg(all(
        feature = "strata-small-quantized-matrix",
        not(feature = "strata-global-image-experts")
    ))]
    crate::strata_prefill_profile::count(
        "expert_small_quant_working_owner_bytes",
        storage::WORKING_BYTES,
    );
    Ok(())
}

#[cfg(test)]
#[path="strata_large_cohort_cpu_tests.rs"]
mod tests;
