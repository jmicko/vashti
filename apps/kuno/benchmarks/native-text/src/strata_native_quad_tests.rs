//! Independent word-grammar proofs for the exact native subgroup postpass.
use super::{Kernel, prompt_specialize};
use cubecl::wgpu::{lower_native_subgroup_quads, native_compute_quads_supported};
use std::collections::BTreeMap;
fn instructions(w: &[u32]) -> Vec<Vec<u32>> {
    let mut out = Vec::new();
    let mut at = 5;
    while at < w.len() {
        let n = (w[at] >> 16) as usize;
        assert!(n > 0 && at + n <= w.len());
        out.push(w[at..at + n].to_vec());
        at += n;
    }
    out
}
fn assemble(header: &[u32], ins: &[Vec<u32>]) -> Vec<u32> {
    header[..5]
        .iter()
        .copied()
        .chain(ins.iter().flatten().copied())
        .collect()
}
fn constants(ins: &[Vec<u32>]) -> BTreeMap<(u32, u32), u32> {
    ins.iter()
        .filter(|i| i[0] & 65535 == 43 && i.len() == 4)
        .map(|i| ((i[1], i[3]), i[2]))
        .collect()
}
fn original(selected: &[u32]) -> Vec<u32> {
    let mut ins = instructions(selected);
    let c = constants(&ins);
    let values: BTreeMap<_, _> = c.iter().map(|(&(t, v), &id)| (id, (t, v))).collect();
    for i in &mut ins {
        if i[0] & 65535 == 366 {
            let (ty, dir) = values[&i[5]];
            assert!(dir < 2);
            i[0] = (6 << 16) | 346;
            i[5] = c[&(ty, dir + 1)];
        }
    }
    ins.retain(|i| !(i[0] & 65535 == 17 && i[1] == 68)); // Only the added Quad capability.
    assemble(selected, &ins)
}
#[test]
fn actual_selected_modules_only_replace_52_exact_f32_participants_and_preserve_every_other_word() {
    assert!(cfg!(feature = "strata-subgroup-quad"));
    for k in [Kernel::PromptAttention, Kernel::PromptAttentionFault] {
        let (_, selected) = prompt_specialize::tests::compiled(&k.source());
        let ins = instructions(&selected);
        assert_eq!(ins.iter().filter(|i| i[0] & 65535 == 366).count(), 52);
        assert_eq!(ins.iter().filter(|i| i[0] & 65535 == 346).count(), 78);
        assert_eq!(ins.iter().filter(|i| i[0] & 65535 == 345).count(), 48);
        let old = original(&selected);
        assert_eq!(lower_native_subgroup_quads(&old).unwrap(), selected);
        assert_eq!(lower_native_subgroup_quads(&selected).unwrap(), selected);
        assert_eq!(old[..5], selected[..5]); // complete ID/header/ABI unchanged.
        // Reverse the replacement independently, requiring all other bytes.
        assert_eq!(original(&lower_native_subgroup_quads(&old).unwrap()), old);
    }
}
#[test]
fn quad_participants_preserve_all_lane_bits_and_reduction_order() {
    for size in [16u32, 32, 64, 128] {
        for lane in 0..size {
            for (mask, dir) in [(1, 0), (2, 1)] {
                let quad = (lane & !3) | ((lane & 3) ^ if dir == 0 { 1 } else { 2 });
                assert_eq!(quad, lane ^ mask);
                for salt in [0, 0x8000_0000, 0x7fc0_1234, 0x7f80_0000, u32::MAX] {
                    let bits = |l: u32| l.wrapping_mul(0x9e37_79b9) ^ salt;
                    assert_eq!(
                        f32::from_bits(bits(quad)).to_bits(),
                        f32::from_bits(bits(lane ^ mask)).to_bits()
                    );
                }
            }
        }
    }
}
#[test]
fn non_f32_wrong_scope_dynamic_and_larger_masks_are_left_exactly_unchanged() {
    let (_, selected) = prompt_specialize::tests::compiled(&Kernel::PromptAttentionFault.source());
    let old = original(&selected);
    let ins = instructions(&old);
    let c = constants(&ins);
    let pos = ins
        .iter()
        .position(|i| i[0] & 65535 == 346 && c.values().any(|id| *id == i[5]))
        .unwrap();
    // Find a genuine eligible XOR1/2, using the selected Quad result ID.
    let target = instructions(&selected)
        .iter()
        .find(|i| i[0] & 65535 == 366)
        .unwrap()[2];
    let pos = ins
        .iter()
        .position(|i| i.get(2) == Some(&target) && i[0] & 65535 == 346)
        .unwrap_or(pos);
    let uint = ins
        .iter()
        .find(|i| i[0] & 65535 == 21 && i[2..] == [32, 0])
        .unwrap()[1];
    for mode in 0..4 {
        let mut input = ins.clone();
        match mode {
            0 => input[pos][1] = uint,
            1 => input[pos][3] = c[&(uint, 0)],
            2 => input[pos][5] = input[pos][4], // nonconstant F32 SSA value, no literal authority.
            _ => input[pos][5] = c[&(uint, 4)],
        }
        let before = assemble(&old, &input);
        let after = lower_native_subgroup_quads(&before).unwrap();
        let out = instructions(&after);
        assert!(out.iter().any(|i| i == &input[pos]));
        assert_eq!(out.iter().filter(|i| i[0] & 65535 == 366).count(), 51);
        // Restore eligible sites only: the rejected instruction must remain byte exact.
        assert_eq!(original(&after), before);
    }
    assert!(lower_native_subgroup_quads(&[]).is_err());
    assert!(lower_native_subgroup_quads(&[0; 5]).is_err());
}
#[test]
fn actual_device_selection_requires_compute_quad_and_all_stage_support() {
    for compute in [false, true] {
        for quad in [false, true] {
            for all in [false, true] {
                assert_eq!(
                    native_compute_quads_supported(
                        0x10 | if compute { 0x20 } else { 0 },
                        0x17 | if quad { 0x80 } else { 0 },
                        all
                    ),
                    compute && quad && all
                );
            }
        }
    }
}
