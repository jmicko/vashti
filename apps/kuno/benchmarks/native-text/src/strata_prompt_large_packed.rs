//! Size-preserving LARGE operand permutation. Original producers stay unchanged.
use super::*;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(crate) struct PackedLargeOp(pub(crate) LargeOp);

fn replace_once(source: &mut String, old: &str, new: &str) {
    assert_eq!(
        source.matches(old).count(),
        1,
        "packed producer source anchor: {old}"
    );
    *source = source.replacen(old, new, 1);
}

impl PackedLargeOp {
    pub(crate) fn source(self) -> String {
        assert!(cfg!(feature = "strata-dequant-direct8"));
        let mut source = self.0.selected_source();
        replace_once(
            &mut source,
            "cohort.base.r7>1u || cohort.base.r8!=0u || cohort.base.r9!=0u",
            "cohort.base.r7!=1u || cohort.base.r8!=1u || cohort.base.r9!=0u",
        );
        if matches!(self.0, LargeOp::Gather | LargeOp::Activate) {
            // Every large Pad32 gutter belongs to this group and must be zeroed.
            // The original small producer's <256 condition never initialized it.
            replace_once(&mut source, "if real<256u", "if real<=PROMPT_MAX_ROWS");
        }
        match self.0 {
            LargeOp::Image { .. } => {
                // store_original8 writes one vec4<u32>, i.e. EIGHT halfs.
                // Source decoding, round-to-half and finite predicates stay intact.
                replace_once(
                    &mut source,
                    "let first=select(p.r3,p.r4,p.role==2u)/4u;",
                    "let first=select(p.r3,p.r4,p.role==2u)/4u;\n    let packed_row=i/(p.columns/8u);let packed_col=i%(p.columns/8u);\n    let dest=(packed_row/8u)*p.columns+(packed_col/2u)*16u+(packed_row%8u)*2u+packed_col%2u;",
                );
                replace_once(
                    &mut source,
                    "if first+i>=arrayLength(&packed)",
                    "if first+dest>=arrayLength(&packed)",
                );
                replace_once(&mut source, "packed[first+i]=", "packed[first+dest]=");
            }
            LargeOp::Activate => {
                source.push_str("\nfn packed_hidden_pair(i:u32)->u32 {let row=i/320u;let pair=i%320u;return (row/16u)*5120u+(pair/8u)*128u+(row%16u)*8u+pair%8u;}\n");
                replace_once(
                    &mut source,
                    "hidden[cohort_prefix*320u+at]=0u",
                    "hidden[cohort_prefix*320u+packed_hidden_pair(at)]=0u",
                );
                replace_once(
                    &mut source,
                    "hidden[cohort_prefix*320u+i]=",
                    "hidden[cohort_prefix*320u+packed_hidden_pair(i)]=",
                );
            }
            LargeOp::Gather => {
                source.push_str("\nfn packed_input_pair(i:u32)->u32 {let row=i/1280u;let pair=i%1280u;return (row/16u)*20480u+(pair/8u)*128u+(row%16u)*8u+pair%8u;}\n");
                replace_once(
                    &mut source,
                    "packed[cohort_prefix*1280u+at]=0u",
                    "packed[cohort_prefix*1280u+packed_input_pair(at)]=0u",
                );
                replace_once(
                    &mut source,
                    "packed[cohort_prefix*1280u+i]=",
                    "packed[cohort_prefix*1280u+packed_input_pair(i)]=",
                );
            }
        }
        source
    }
}

impl KernelMetadata for PackedLargeOp {
    fn name(&self) -> &'static str {
        match self.0 {
            LargeOp::Gather => "StrataExpertLargePackedCohortGather",
            LargeOp::Activate => "StrataExpertLargePackedCohortActivate",
            LargeOp::Image { down: false, .. } => "StrataExpertLargePackedCohortGuDirect8Image",
            LargeOp::Image { down: true, .. } => "StrataExpertLargePackedCohortDownDirect8Image",
        }
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
            .info(*self)
            .cube_dim(CubeDim::new_1d(self.0.lanes()))
    }
    fn address_type(&self) -> StorageType {
        self.0.address_type()
    }
    fn native_release_after(&self) -> bool {
        self.0.native_release_after()
    }
}
impl CubeTask<AutoCompiler> for PackedLargeOp {
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
            source: self.source(),
            repr: None,
            cube_dim: CubeDim::new_1d(self.0.lanes()),
            debug_info: None,
        })
    }
}

pub(crate) fn validate_words(words: &[u32; WORDS]) -> Result<usize> {
    if words[13] != 1 || words[14] != 1 {
        return Err(Error::InvalidBufferLength);
    }
    let mut original = *words;
    original[14] = 0;
    super::validate_words(&original)
}

pub(crate) fn launch(
    client: &ComputeClient<WgpuRuntime>,
    op: LargeOp,
    grid: [usize; 3],
    args: KernelArguments,
    words: &[u32; WORDS],
) -> Result<()> {
    validate_words(words)?;
    if matches!(op, LargeOp::Activate) {
        crate::strata_prefill_profile::count(
            "expert_large_cohort_activation_workgroups",
            (grid[0] * grid[1] * grid[2]) as u64,
        );
    }
    let args = args.with_info(MetadataBindingInfo::custom(
        words
            .chunks_exact(2)
            .map(|v| u64::from(v[0]) | (u64::from(v[1]) << 32))
            .collect(),
    ));
    unsafe {
        client.launch_unchecked(
            Box::new(PackedLargeOp(op)),
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
            PackedLargeOp(op)
                .compile(
                    &mut AutoCompiler::Wgsl(Default::default()),
                    &Default::default(),
                    ExecutionMode::Unchecked,
                    StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
                )
                .unwrap()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packed_producers_preserve_original_decoders_and_validate_actual_wgsl() {
        for compiled in sources() {
            let module = wgpu::naga::front::wgsl::parse_str(&compiled.source).unwrap();
            wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
            assert!(
                compiled
                    .source
                    .contains("cohort.base.r7!=1u || cohort.base.r8!=1u")
            );
        }
        for (dtype, down) in [
            (q::IQ2_S, false),
            (q::IQ3_S, false),
            (q::IQ3_XXS, false),
            (q::IQ4_XS, false),
            (q::IQ4_NL, true),
            (q::Q2_0, true),
        ] {
            let op = LargeOp::Image { dtype, down };
            let old = op.selected_source();
            let new = PackedLargeOp(op).source();
            // No changes to the complete decode helpers after the store body.
            let anchor = "fn write_record32(";
            assert_eq!(
                old.split_once(anchor).unwrap().1,
                new.split_once(anchor).unwrap().1.replace(
                    "cohort.base.r7!=1u || cohort.base.r8!=1u || cohort.base.r9!=0u",
                    "cohort.base.r7>1u || cohort.base.r8!=0u || cohort.base.r9!=0u"
                )
            );
            assert!(new.contains("packed[first+dest]=select(vec4<u32>(0u),bits,good)"));
            assert!(old.contains("cohort.base.r8!=0u"));
        }
    }
}
