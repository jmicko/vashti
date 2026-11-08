//! Separate32-module validation, never appended to the original matrix population.
use super::*;
use cubecl::wgpu::{AutoCompiler, AutoRepresentation};
use cubecl::{
    Compiler, WgpuCompilationOptions,
    ir::{ElemType, StorageType, UIntKind},
};
use cubecl_runtime::kernel::KernelDefinition;
use std::{
    collections::{BTreeMap, BTreeSet},
    io::Write,
};
fn definitions() -> Vec<(&'static str, KernelDefinition)> {
    let mut out = native_expert_wide::shared_reuse::definitions();
    out.push((
        "shared_reuse_ordered_down",
        native_expert_wide::ordered::shared_reuse::definition(),
    ));
    out
}
/// Inspect actual emitted Workgroup block types and byte offsets. Half operands
/// must remain disjoint while the later float publication aliases an operand.
fn verify_layout(
    words: &[u32],
    explicit: bool,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut types = BTreeMap::<u32, Vec<u32>>::new();
    let mut constants = BTreeMap::new();
    let mut offsets = BTreeMap::new();
    let mut aliased = BTreeSet::new();
    let mut variables = Vec::new();
    let mut at = 5;
    while at < words.len() {
        let n = (words[at] >> 16) as usize;
        if n == 0 || at + n > words.len() {
            return Err("reuse instruction extent".into());
        }
        let i = &words[at..at + n];
        match i[0] & 65535 {
            21 | 22 | 23 | 28 | 30 | 32 => {
                types.insert(i[1], i.to_vec());
            }
            43 if n == 4 => {
                constants.insert(i[2], i[3] as usize);
            }
            71 if n == 3 && i[2] == 20 => {
                aliased.insert(i[1]);
            }
            72 if n == 5 && i[2] == 0 && i[3] == 35 => {
                offsets.insert(i[1], i[4] as usize);
            }
            59 if n >= 4 && i[3] == 4 => {
                variables.push((i[2], i[1]));
            }
            _ => {}
        }
        at += n;
    }
    fn bytes(ty: u32, types: &BTreeMap<u32, Vec<u32>>, c: &BTreeMap<u32, usize>) -> Option<usize> {
        let i = types.get(&ty)?;
        match i[0] & 65535 {
            21 | 22 => Some(i[2] as usize / 8),
            23 => bytes(i[2], types, c)?.checked_mul(i[3] as usize),
            28 => bytes(i[2], types, c)?.checked_mul(*c.get(&i[3])?),
            _ => None,
        }
    }
    let mut half = Vec::new();
    let mut floats = Vec::new();
    for (id, ptr) in variables {
        let p = types.get(&ptr).ok_or("reuse pointer type")?;
        if p[0] & 65535 != 32 {
            return Err("reuse real Workgroup pointer".into());
        }
        let mut inner = p[3];
        let mut offset = 0;
        if explicit {
            let b = types.get(&inner).ok_or("reuse Block type")?;
            if b[0] & 65535 != 30 || b.len() != 3 || !aliased.contains(&id) {
                return Err("reuse genuine Aliased Block".into());
            }
            offset = *offsets
                .get(&inner)
                .ok_or("reuse actual Block byte offset")?;
            inner = b[2];
        }
        let a = types.get(&inner).ok_or("reuse array type")?;
        if a[0] & 65535 != 28 {
            return Err("reuse genuine shared array".into());
        }
        let size = bytes(inner, &types, &constants).ok_or("reuse array extent")?;
        if size != 16384 {
            return Err("reuse exact original16KiB array".into());
        }
        let element = types.get(&a[2]).ok_or("reuse element type")?;
        if element[0] & 65535 == 23 {
            let scalar = types.get(&element[2]).ok_or("reuse half scalar")?;
            if element[3] != 4 || scalar[0] & 65535 != 22 || scalar[2] != 16 {
                return Err("reuse typed original half4 operand".into());
            }
            half.push((offset, offset + size));
        } else if element[0] & 65535 == 22 && element[2] == 32 {
            floats.push((offset, offset + size));
        } else {
            return Err("reuse typed originalF32 publication".into());
        }
    }
    if half.len() != 2 || floats.len() != 1 {
        return Err("reuse all three original shared owners".into());
    }
    if explicit {
        let overlap = |a: (usize, usize), b: (usize, usize)| a.0 < b.1 && b.0 < a.1;
        if overlap(half[0], half[1])
            || !half.iter().any(|&h| overlap(h, floats[0]))
            || half.iter().chain(&floats).map(|v| v.1).max() != Some(32768)
        {
            return Err("reuse actual disjoint operands and publication alias".into());
        }
    }
    Ok(())
}
pub(crate) fn validate() -> std::result::Result<(), Box<dyn std::error::Error>> {
    for mask in 0u8..8 {
        for (name, definition) in definitions() {
            let mut options = WgpuCompilationOptions {
                supports_vulkan: true,
                ..Default::default()
            };
            options.vulkan.max_spirv_version = (1, 6);
            options.vulkan.supports_uniform_standard_layout = mask & 1 != 0;
            options.vulkan.supports_uniform_unsized_array = mask & 2 != 0;
            options.vulkan.supports_explicit_smem = mask & 4 != 0;
            let AutoRepresentation::SpirV(kernel) = Compiler::compile(
                &mut AutoCompiler::SpirV(Default::default()),
                definition,
                &options,
                ExecutionMode::Unchecked,
                StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
            )?
            else {
                return Err("reuse actual SPIRV representation".into());
            };
            let expected = if mask & 4 != 0 { 32768 } else { 49152 };
            if kernel.shared_size != expected {
                return Err(format!(
                    "reuse actual physical shared{name} mask{mask}:{} != {expected}",
                    kernel.shared_size
                )
                .into());
            }
            verify_layout(&kernel.assembled_module, mask & 4 != 0)?;
            let words = wgpu::hal::vulkan::kuno_native_workgroup_layout(&kernel.assembled_module)?;
            let words = wgpu::hal::vulkan::kuno_cooperative_subgroup_shared_barriers(&words)?
                .unwrap_or(words);
            let words = wgpu::hal::vulkan::kuno_cooperative_fp16_preserve(&words)?.unwrap_or(words);
            if wgpu::hal::vulkan::kuno_cooperative_subgroup(&words) != Some(16) {
                return Err("reuse actual subgroup16".into());
            }
            let mut at = 5;
            let (mut products, mut loads, mut stores, mut fault) = (0, 0, 0, false);
            while at < words.len() {
                let n = (words[at] >> 16) as usize;
                match words[at] & 65535 {
                    4459 => products += 1,
                    4457 => loads += 1,
                    4458 => stores += 1,
                    241 => fault = true,
                    _ => {}
                }
                at += n;
            }
            if products != 8 || loads != 8 || stores < 4 || !fault {
                return Err("reuse original K16 products/loads/publication/fault".into());
            }
            #[cfg(feature = "strata-wide-register-stage")]
            wide_pipeline::validate_program(&words)?;
            let mut c = std::process::Command::new("spirv-val");
            c.args(["--target-env", "vulkan1.3"]);
            if mask & 1 != 0 {
                c.arg("--uniform-buffer-standard-layout");
            }
            let mut c = c.arg("-").stdin(std::process::Stdio::piped()).spawn()?;
            c.stdin.take().ok_or("reuse validator stdin")?.write_all(
                &words
                    .iter()
                    .flat_map(|w| w.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            if !c.wait()?.success() {
                return Err("reuse actual SPIRV validation".into());
            }
            println!(
                "strata_wide_shared_reuse_shader_accepted name={name} layout_mask={mask} physical_shared={expected} subgroup=16"
            );
        }
    }
    Ok(())
}
#[cfg(test)]
#[path = "strata_matrix_shared_reuse_tests.rs"]
mod tests;
