//! Inspect the actual compiler module, not a hand-written stand-in shader.
use super::*;
use cubecl::{Compiler, WgpuCompilationOptions};

fn original() -> Vec<u32> {
    static MODULE: std::sync::OnceLock<Vec<u32>> = std::sync::OnceLock::new();
    MODULE
        .get_or_init(|| {
            let mut options = WgpuCompilationOptions {
                supports_vulkan: true,
                ..Default::default()
            };
            options.vulkan.max_spirv_version = (1, 6);
            let cubecl::wgpu::AutoRepresentation::SpirV(kernel) = Compiler::compile(
                &mut cubecl::wgpu::AutoCompiler::SpirV(Default::default()),
                definition(false, 8, 23),
                &options,
                ExecutionMode::Unchecked,
                StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
            )
            .unwrap() else {
                panic!("actual native half SPIRV compiler")
            };
            wgpu::hal::vulkan::kuno_native_workgroup_layout(&kernel.assembled_module).unwrap()
        })
        .clone()
}
fn instructions(words: &[u32]) -> Vec<(usize, usize, u32)> {
    let mut out = Vec::new();
    let mut at = 5;
    while at < words.len() {
        let n = (words[at] >> 16) as usize;
        assert!(n != 0 && at + n <= words.len());
        out.push((at, n, words[at] & 65535));
        at += n;
    }
    out
}
fn actual_module(dtype: u32, tile: usize) -> Vec<u32> {
    let mut options = WgpuCompilationOptions {
        supports_vulkan: true,
        ..Default::default()
    };
    options.vulkan.max_spirv_version = (1, 6);
    let cubecl::wgpu::AutoRepresentation::SpirV(kernel) = Compiler::compile(
        &mut cubecl::wgpu::AutoCompiler::SpirV(Default::default()),
        definition(matches!(dtype, 20 | 42), tile, dtype),
        &options,
        ExecutionMode::Unchecked,
        StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
    )
    .unwrap() else {
        panic!("actual quant loop compiler")
    };
    wgpu::hal::vulkan::kuno_native_workgroup_layout(&kernel.assembled_module).unwrap()
}

#[test]
fn quant_native_preserves_genuine_inner_product_loop_and_numeric_words() {
    for dtype in [18, 21, 22, 23, 20, 42] {
        for tile in [8, 16, 32] {
            let original = actual_module(dtype, tile);
            let native = wgpu::hal::vulkan::kuno_quant_half_pack(&original)
                .unwrap()
                .unwrap();
            let controls: Vec<_> = instructions(&native)
                .into_iter()
                .filter_map(|(at, n, op)| {
                    (op == 246 && n == 4 && native[at + 3] == 2).then_some(at + 3)
                })
                .collect();
            assert_eq!(controls.len(), usize::from(tile == 32));
            let mut unhinted = native.clone();
            for &control in &controls {
                unhinted[control] = 0;
            }
            let restored = wgpu::hal::vulkan::kuno_quant_half_pack(&unhinted)
                .unwrap()
                .unwrap();
            assert_eq!(restored, native);
            assert_eq!(
                wgpu::hal::vulkan::kuno_quant_half_pack(&native)
                    .unwrap()
                    .unwrap(),
                native
            );
            assert_eq!(
                unhinted.iter().zip(&native).filter(|(a, b)| a != b).count(),
                controls.len()
            );
        }
    }
}

#[test]
fn quant_native_refuses_conflicting_product_loop_hint_and_unknown_cfg_edge() {
    let native = wgpu::hal::vulkan::kuno_quant_half_pack(&actual_module(23, 32))
        .unwrap()
        .unwrap();
    let at = instructions(&native)
        .into_iter()
        .find_map(|(at, n, op)| (op == 246 && n == 4 && native[at + 3] == 2).then_some(at))
        .unwrap();
    let mut conflict = native.clone();
    conflict[at + 3] = 1;
    assert!(wgpu::hal::vulkan::kuno_quant_half_pack(&conflict).is_err());
    let continuation = native[at + 2];
    let start = instructions(&native)
        .into_iter()
        .find_map(|(at, n, op)| {
            (op == 248 && n == 2 && native[at + 1] == continuation).then_some(at)
        })
        .unwrap();
    let edge = instructions(&native)
        .into_iter()
        .find_map(|(at, n, op)| (at > start && op == 249 && n == 2).then_some(at))
        .unwrap();
    let mut unknown = native.clone();
    unknown[edge + 1] = native[3];
    assert!(wgpu::hal::vulkan::kuno_quant_half_pack(&unknown).is_err());
}
#[test]
fn quant_half_native_real_module_uses_original_glsl_packing_and_is_idempotent() {
    let original = original();
    let lowered = wgpu::hal::vulkan::kuno_quant_half_pack(&original)
        .unwrap()
        .unwrap();
    assert_ne!(original, lowered);
    let native = instructions(&lowered);
    assert!(native
        .iter()
        .any(|&(at, n, op)| op == 12 && n == 6 && lowered[at + 4] == 58));
    assert!(native
        .iter()
        .any(|&(at, n, op)| op == 12 && n == 6 && lowered[at + 4] == 62));
    // The authentic source has no unrelated floating-point conversions: every
    // original F16 cast now executes the original GLSL operation explicitly.
    assert!(!native.iter().any(|&(_, _, op)| op == 115));
    assert_eq!(
        wgpu::hal::vulkan::kuno_quant_half_pack(&lowered)
            .unwrap()
            .unwrap(),
        lowered
    );
    let declarations = |words: &[u32]| {
        instructions(words)
            .into_iter()
            .filter(|(_, _, op)| matches!(op, 59 | 71 | 72))
            .flat_map(|(at, n, _)| words[at..at + n].to_vec())
            .collect::<Vec<_>>()
    };
    assert_eq!(declarations(&original), declarations(&lowered));
}
#[test]
fn quant_half_native_refuses_real_wrong_workgroup_descriptor_and_missing_matrix() {
    let original = original();
    let mut wrong_workgroup = original.clone();
    let at = instructions(&original)
        .into_iter()
        .find_map(|(at, n, op)| (op == 16 && n == 6 && original[at + 2] == 17).then_some(at))
        .unwrap();
    wrong_workgroup[at + 3] = 128;
    assert!(wgpu::hal::vulkan::kuno_quant_half_pack(&wrong_workgroup).is_err());
    let mut wrong_set = original.clone();
    let at = instructions(&original)
        .into_iter()
        .find_map(|(at, n, op)| (op == 71 && n == 4 && original[at + 2] == 34).then_some(at))
        .unwrap();
    wrong_set[at + 3] = 1;
    assert!(wgpu::hal::vulkan::kuno_quant_half_pack(&wrong_set).is_err());
    let mut no_matrix = original.clone();
    let at = instructions(&original)
        .into_iter()
        .find_map(|(at, n, op)| (op == 17 && n == 2 && original[at + 1] == 6022).then_some(at))
        .unwrap();
    no_matrix[at + 1] = 1;
    assert!(wgpu::hal::vulkan::kuno_quant_half_pack(&no_matrix).is_err());
}
#[test]
fn quant_half_native_refuses_other_f16_width_and_does_not_touch_other_entry() {
    let original = original();
    let mut wrong_width = original.clone();
    let half = instructions(&original)
        .into_iter()
        .find_map(|(at, n, op)| {
            (op == 22 && n == 3 && original[at + 2] == 16).then_some(original[at + 1])
        })
        .unwrap();
    let at = instructions(&original)
        .into_iter()
        .find_map(|(at, n, op)| {
            (op == 23 && n == 4 && original[at + 2] == half && original[at + 3] == 2).then_some(at)
        })
        .unwrap();
    wrong_width[at + 3] = 4;
    assert!(wgpu::hal::vulkan::kuno_quant_half_pack(&wrong_width).is_err());
    let mut missing_operand = original.clone();
    let at = instructions(&original)
        .into_iter()
        .find_map(|(at, _, op)| (op == 115).then_some(at))
        .unwrap();
    missing_operand[at + 3] = original[3];
    assert!(wgpu::hal::vulkan::kuno_quant_half_pack(&missing_operand).is_err());
    let mut unrelated = original.clone();
    let (at, _, _) = instructions(&original)
        .into_iter()
        .find(|(_, _, op)| *op == 15)
        .unwrap();
    // Change the literal entry identity without modifying its extent/interface.
    unrelated[at + 3] ^= 1;
    assert_eq!(
        wgpu::hal::vulkan::kuno_quant_half_pack(&unrelated),
        Ok(None)
    );
}
