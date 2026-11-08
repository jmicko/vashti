//! Genuine MAIN-queue preparation of ORIGINAL encoded model roles. The source
//! resolver, current tier/loan table, RAW sentinel and whole-owner checks are
//! inherited from the original Image producer. No dequantization or CPU copy.
use super::*;
#[path = "strata_prompt_cohort_compact_probe.rs"]
mod probe;
pub(crate) use probe::cases as public_cases;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(crate) struct CompactOp {
    pub dtype: u32,
    pub down: bool,
}
impl CompactOp {
    pub(crate) fn source(self) -> String {
        assert!(matches!(
            (self.down, self.dtype),
            (false, q::IQ2_S | q::IQ3_S | q::IQ3_XXS | q::IQ4_XS) | (true, q::IQ4_NL | q::Q2_0)
        ));
        // The old source() deliberately precedes any optional selected decoder.
        // Original factories and their exact compiler population remain intact.
        let mut source = CohortOp::Image {
            dtype: self.dtype,
            down: self.down,
        }
        .source();
        let main = source.find("fn main(").unwrap();
        let first = source[main..]
            .find("    let record=id.x;")
            .or_else(|| source[main..].find("    let i=id.x;if i>=p.rows*p.columns/8u"))
            .map(|at| main + at)
            .unwrap();
        let end = crate::strata_port::function_end(&source, main);
        let body = if self.down {
            include_str!("strata_prompt_cohort_compact_down.wgsl")
        } else {
            include_str!("strata_prompt_cohort_compact_gu.wgsl")
        };
        source.replace_range(first..end - 1, body);
        source.push_str("\n// STRATA_SMALL_QUANT_COMPACT_ORIGINAL_ENCODED_BITS\n");
        source
    }
}
impl KernelMetadata for CompactOp {
    fn name(&self) -> &'static str {
        if self.down {
            "StrataExpertCohortDownCompact"
        } else {
            "StrataExpertCohortGuCompact"
        }
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
            .info(*self)
            .cube_dim(CubeDim::new_1d(256))
    }
    fn address_type(&self) -> StorageType {
        StorageType::Scalar(ElemType::UInt(UIntKind::U32))
    }
    fn native_release_after(&self) -> bool {
        self.down
    }
}
impl CubeTask<AutoCompiler> for CompactOp {
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
            debug_info: None,
            source: self.source(),
            repr: None,
            cube_dim: CubeDim::new_1d(256),
        })
    }
}
pub(crate) fn tile_for(members: usize) -> usize {
    if members <= 8 {
        8
    } else if members <= 16 {
        16
    } else {
        32
    }
}
fn grid(op: CompactOp, words: &[u32; WORDS]) -> Result<[usize; 3]> {
    validate_words(words)?;
    let (columns, rows, role) = if op.down {
        (640, 2560, 2)
    } else {
        (2560, 1280, 0)
    };
    if words[1] != columns
        || words[2] != rows
        || words[5] != role
        || words[6] % 4 != 0
        || words[7] % 4 != 0
        || words[8] % 4 != 0
    {
        return Err(Error::InvalidBufferLength);
    }
    let (values, bytes) = q::block_layout(op.dtype).ok_or(Error::InvalidBufferLength)?;
    if columns as usize % values != 0 || words[3] as usize != columns as usize / values * bytes {
        return Err(Error::InvalidBufferLength);
    }
    let role_words = if op.down {
        words[8] - words[6] - words[7]
    } else {
        words[6]
    };
    let original_rows = if op.down { rows } else { rows / 2 };
    if u64::from(role_words) * 4 != u64::from(original_rows) * u64::from(words[3]) {
        return Err(Error::InvalidBufferLength);
    }
    Ok([
        (role_words as usize / 4).div_ceil(256),
        1,
        words[9] as usize,
    ])
}
pub(super) fn launch(
    client: &ComputeClient<WgpuRuntime>,
    op: CompactOp,
    mut args: KernelArguments,
    words: &[u32; WORDS],
) -> Result<()> {
    let dispatch = grid(op, words)?;
    args = args.with_info(MetadataBindingInfo::custom(
        words
            .chunks_exact(2)
            .map(|v| u64::from(v[0]) | (u64::from(v[1]) << 32))
            .collect(),
    ));
    unsafe {
        client.launch_unchecked(
            Box::new(op),
            CubeCount::Static(dispatch[0] as u32, 1, dispatch[2] as u32),
            args,
        );
    }
    crate::strata_prefill_profile::count(
        if op.down {
            "expert_small_quant_pack_down_launches"
        } else {
            "expert_small_quant_pack_gu_launches"
        },
        1,
    );
    let copied_words = if op.down {
        words[8] - words[6] - words[7]
    } else {
        words[6] + words[7]
    };
    crate::strata_prefill_profile::count(
        "expert_small_quant_compact_bytes",
        u64::from(copied_words) * 4 * u64::from(words[9]),
    );
    Ok(())
}
pub(crate) fn validate_sources() -> std::result::Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;
    for (down, types) in [
        (false, &[q::IQ2_S, q::IQ3_S, q::IQ3_XXS, q::IQ4_XS][..]),
        (true, &[q::IQ4_NL, q::Q2_0][..]),
    ] {
        for &dtype in types {
            let op = CompactOp { dtype, down };
            let words = cubecl::wgpu::native_expert_shader_words(&op.source())?;
            let mut at = 5;
            let mut literal_local_size = 0;
            while at < words.len() {
                let count = (words[at] >> 16) as usize;
                if count == 0 || at + count > words.len() {
                    return Err("compact SPIRV instruction extent".into());
                }
                if words[at] & 65535 == 16 && count >= 3 && words[at + 2] == 17 {
                    if count != 6 || words[at + 3..at + 6] != [256, 1, 1] {
                        return Err("compact actual SPIRV local size".into());
                    }
                    literal_local_size += 1;
                }
                at += count;
            }
            if literal_local_size != 1 {
                return Err("compact actual SPIRV entry count".into());
            }
            let mut validator = std::process::Command::new("spirv-val")
                .args(["--target-env", "vulkan1.3", "-"])
                .stdin(std::process::Stdio::piped())
                .spawn()?;
            validator
                .stdin
                .take()
                .ok_or("compact validator stdin")?
                .write_all(
                    &words
                        .iter()
                        .flat_map(|w| w.to_le_bytes())
                        .collect::<Vec<_>>(),
                )?;
            if !validator.wait()?.success() {
                return Err("compact actual SPIRV validation".into());
            }
            println!(
                "strata_small_quant_compact_shader_accepted dtype={dtype} down={down} workgroup=256 original_encoded_copy=true"
            );
        }
    }
    Ok(())
}
#[cfg(test)]
#[path = "strata_prompt_cohort_compact_tests.rs"]
mod tests;
