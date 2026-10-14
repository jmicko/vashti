//! CPU/source proofs for the actual bounded joined method. No device or model
//! is constructed; real CPU13/cache/failure/closure gates remain separate.
use super::*;

#[test]
fn core_eight_repair_keeps_confirmed_cell_and_replaces_only_provisional_tail() {
    if !crate::strata_intel_core::enabled() {
        return;
    }
    let start = 101;
    let preceding = carries(1);
    for window in 1..=8 {
        let old = start + window - 2;
        for accepted in 1..=window {
            let tokens: Vec<_> = (0..accepted as u32).map(|i| 31 + i).collect();
            let actual = carries(accepted);
            let plan =
                AcceptedCacheRepair::new(old, start, window > 1, &tokens, &preceding, &actual)
                    .unwrap();
            assert_eq!(plan.end, start + accepted - 1);
            assert_eq!(plan.rewind, window > 1);
            assert_eq!(plan.first_row, usize::from(window > 1));
            if window > 1 {
                assert!(
                    AcceptedCacheRepair::new(old, start, false, &tokens, &preceding, &actual)
                        .is_err()
                );
            }
        }
    }
    let tokens = vec![31; 9];
    assert!(
        AcceptedCacheRepair::new(start + 7, start, true, &tokens, &preceding, &carries(9)).is_err()
    );
    assert!(
        AcceptedCacheRepair::new(start + 7, start, true, &[31], &preceding, &carries(1)).is_err()
    );
}

fn carries(count: usize) -> Vec<f32> {
    (0..count * k::WIDE)
        .map(|i| {
            if i % 23 == 0 {
                -0.0
            } else {
                (i % 79) as f32 / 83.0
            }
        })
        .collect()
}

#[test]
fn actual_cursor_distinguishes_rejected_one_row_prefix_from_no_draft_tail() {
    let start = 17;
    let preceding = carries(1);
    for drafts in 0..=2 {
        for accepted in 1..=3 {
            let old = start - 1 + drafts;
            let tokens = [31, 37, 41];
            let c = carries(accepted);
            let plan = AcceptedCacheRepair::new(
                old,
                start,
                drafts != 0,
                &tokens[..accepted],
                &preceding,
                &c,
            );
            if accepted > drafts + 1 || drafts == 0 && accepted != 1 {
                assert!(plan.is_err());
                continue;
            }
            let plan = plan.unwrap();
            assert_eq!(plan.rewind, drafts != 0);
            assert_eq!(plan.first_row, usize::from(drafts != 0));
            assert_eq!(plan.end, start + accepted - 1);
            // An accepted length of one after either actual draft retains the
            // first real cell and appends nothing; it is never guessed a tail.
            if accepted == 1 {
                assert_eq!(accepted - plan.first_row, usize::from(drafts == 0));
            }
        }
    }
}

#[test]
fn full_input_preflight_rejects_bad_envelopes_and_every_unconsumed_carry() {
    let tokens = [11, 13, 17];
    let preceding = carries(1);
    let full = carries(3);
    for (old, start, stamp, count) in [
        (18, 17, true, 0),
        (18, 17, true, 4),
        (0, 0, true, 1),
        (15, 17, false, 1),
        (16, 17, true, 1),
        (19, 17, true, 3),
        (17, 17, false, 1),
        (17, 17, true, 3),
        (k::CONTEXT + 1, 17, true, 1),
        (k::CONTEXT, k::CONTEXT, true, 2),
        (1, usize::MAX, false, 1),
    ] {
        let t = vec![1; count];
        let c = carries(count);
        assert!(AcceptedCacheRepair::new(old, start, stamp, &t, &preceding, &c).is_err());
    }
    assert!(
        AcceptedCacheRepair::new(18, 17, true, &tokens, &preceding[..k::WIDE - 1], &full).is_err()
    );
    assert!(
        AcceptedCacheRepair::new(18, 17, true, &tokens, &preceding, &full[..full.len() - 1])
            .is_err()
    );
    for row in 0..3 {
        let mut bad = tokens;
        bad[row] = k::VOCAB as u32;
        assert!(AcceptedCacheRepair::new(18, 17, true, &bad, &preceding, &full).is_err());
        for bits in [
            f32::NAN.to_bits(),
            f32::INFINITY.to_bits(),
            f32::NEG_INFINITY.to_bits(),
        ] {
            let mut bad = full.clone();
            bad[row * k::WIDE + k::WIDE - 1] = f32::from_bits(bits);
            assert!(AcceptedCacheRepair::new(18, 17, true, &tokens, &preceding, &bad).is_err());
        }
    }
    let mut bad = preceding.clone();
    bad[0] = f32::NAN;
    assert!(AcceptedCacheRepair::new(18, 17, true, &tokens, &bad, &full).is_err());
    let last = carries(1);
    assert!(
        AcceptedCacheRepair::new(k::CONTEXT, k::CONTEXT, true, &[1], &preceding, &last).is_ok()
    );
    assert!(
        AcceptedCacheRepair::new(k::CONTEXT - 1, k::CONTEXT, false, &[1], &preceding, &last)
            .is_ok()
    );
}

#[test]
fn original_cell_positions_tokens_and_carry_bits_replay_every_accepted_prefix() {
    let start = 7;
    let tokens = [3, 5, 11];
    let preceding = carries(1);
    for drafts in 0..=2 {
        for accepted in 1..=drafts + 1 {
            let actual = carries(accepted);
            let old = start - 1 + drafts;
            let plan = AcceptedCacheRepair::new(
                old,
                start,
                drafts != 0,
                &tokens[..accepted],
                &preceding,
                &actual,
            )
            .unwrap();
            // Independent original scalar loop: retain the first completed
            // cell after drafting, otherwise append confirmed at old cursor.
            let expected: Vec<_> = if drafts == 0 {
                vec![(old, tokens[0], &preceding[..])]
            } else {
                (1..accepted)
                    .map(|row| {
                        (
                            start + row - 1,
                            tokens[row],
                            &actual[(row - 1) * k::WIDE..row * k::WIDE],
                        )
                    })
                    .collect()
            };
            let joined: Vec<_> = (plan.first_row..accepted)
                .map(|row| {
                    let c = if row == 0 {
                        &preceding[..]
                    } else {
                        &actual[(row - 1) * k::WIDE..row * k::WIDE]
                    };
                    (start + row - 1, tokens[row], c)
                })
                .collect();
            assert_eq!(joined.len(), expected.len());
            for ((p, t, c), (ep, et, ec)) in joined.iter().zip(&expected) {
                assert_eq!((p, t), (ep, et));
                assert!(c.iter().zip(*ec).all(|(a, b)| a.to_bits() == b.to_bits()));
            }
            // A whole original suffix rewind and exact scalar append stores
            // retain every prefix bit, including negative zero and spare KV.
            let original: Vec<u32> = (0..k::CONTEXT)
                .map(|i| {
                    if i % 19 == 0 {
                        (-0f32).to_bits()
                    } else {
                        (i as f32 / 71.).to_bits()
                    }
                })
                .collect();
            let mut scalar = original.clone();
            let mut joined_cache = original;
            if drafts != 0 {
                scalar[start..].fill(0);
            }
            if plan.rewind {
                joined_cache[start..].fill(0);
            }
            for &(p, t, c) in &expected {
                scalar[p] = c[t as usize].to_bits();
            }
            for &(p, t, c) in &joined {
                joined_cache[p] = c[t as usize].to_bits();
            }
            assert_eq!(scalar, joined_cache);
        }
    }
}

#[test]
fn repair_clears_all_original_scratch_and_never_loses_early_fault_or_padding() {
    let (roles, count) = clear_roles(true);
    let cleared: Vec<_> = roles[..count]
        .iter()
        .copied()
        .chain([B::Token as usize, B::Carry as usize])
        .collect();
    assert_eq!(cleared.len(), 39);
    for i in 0..N {
        assert_eq!(
            cleared.contains(&i),
            ![B::Keys as usize, B::Values as usize, B::Fault as usize].contains(&i)
        );
    }
    assert_eq!(
        cleared
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        39
    );
    // An earlier stage may flag invalid finite input or a nonzero physical
    // tail. Every subsequent scalar clear must retain all fault-owner words.
    for fault_word in [0, 1, BufferLayout::new(1).unwrap().owned_words - 1] {
        for bits in [1, 4, 5, u32::MAX] {
            let mut fault = vec![0u32; BufferLayout::new(1).unwrap().owned_words];
            fault[fault_word] = bits;
            for _ in 0..2 {
                for &role in &cleared {
                    assert_ne!(role, B::Fault as usize);
                }
                if fault.iter().any(|&x| x != 0) {
                    fault[0] |= 4;
                }
            }
            assert!(fault.iter().any(|&x| x != 0));
            assert_eq!(fault[fault_word] & bits, bits);
        }
    }
}

#[test]
fn combined_info_and_all_temporary_capacities_fit_the_existing_plan() {
    // Independent count from the actual original stage shapes: 16 embedding
    // tiles+finite, two three-call norms, concat/matmul/finite; 12 HC calls;
    // two two-call KV projections, three-call RoPE, append. No head/FFN.
    let condition = 16 + 1 + 2 * 3 + 1 + 1 + 1;
    let hc = 3 + 2 + 1 + 1 + 2 + 1 + 1 + 1;
    let kv = 2 + 2 + 3 + 1;
    assert_eq!(condition + hc + kv, 46);
    assert_eq!(MAX_BATCHED_REPAIR_DISPATCHES, 421);
    assert!(40 + 1 + 42 + 2 * (39 + 84 + condition + hc + kv) <= MAX_TOKEN_DISPATCHES);
    assert!(6 + 1 + 8 + 2 * (7 + 16 + condition + hc + kv) <= MAX_TOKEN_DISPATCHES);
    assert_eq!(N, 42);
    assert_eq!(private_plan().unwrap(), 11_698_176);
    assert_eq!(INFO_BYTES, 1_048_576);
    const { assert!(INFO_BYTES + 3 * 4 * 1024 * 1024 <= RUNTIME_BYTES) }
    let input = BufferLayout::new(k::WIDE).unwrap();
    let token = BufferLayout::new(1).unwrap();
    assert!(k::WIDE * 4 + input.owned_bytes + token.owned_bytes < HOST_BYTES);
    assert!(
        2 * token.owned_bytes + MAX_LIVE_OUTPUTS * OUTPUT_HOST_BYTES + 2 * 1024 * 1024
            <= HOST_BYTES
    );
}

#[test]
fn completed_counters_are_atomic_and_preserve_original_traffic() {
    let original = Traffic {
        host_read_calls: 7,
        host_read_bytes: 99,
        explicit_dispatches: 421,
        ..Traffic::default()
    };
    let completed = completed_batched_repair(original, 2).unwrap();
    assert_eq!(
        (
            completed.host_read_calls,
            completed.host_read_bytes,
            completed.explicit_dispatches
        ),
        (7, 99, 421)
    );
    assert_eq!(
        (
            completed.batched_repair_calls,
            completed.batched_repair_cells,
            completed.batched_repair_fault_gates
        ),
        (1, 2, 1)
    );
    let retained = completed_batched_repair(completed, 0).unwrap();
    assert_eq!(
        (
            retained.batched_repair_calls,
            retained.batched_repair_cells,
            retained.batched_repair_fault_gates
        ),
        (2, 2, 2)
    );
    for bad in [
        Traffic {
            batched_repair_calls: u64::MAX,
            ..original
        },
        Traffic {
            batched_repair_cells: u64::MAX,
            ..original
        },
        Traffic {
            batched_repair_fault_gates: u64::MAX,
            ..original
        },
    ] {
        assert!(completed_batched_repair(bad, 2).is_err());
        assert_eq!((bad.host_read_calls, bad.explicit_dispatches), (7, 421));
    }
}

#[test]
fn actual_joined_source_has_one_late_full_gate_and_original_scalar_defaults() {
    let source = include_str!("resident_mtp.rs");
    let joined = source
        .split_once("pub fn repair_accepted_cache(")
        .unwrap()
        .1
        .split_once("pub fn rewind(")
        .unwrap()
        .0;
    assert_eq!(joined.matches("self.check_fault()?").count(), 1);
    assert_eq!(joined.matches(".completed_fence(").count(), 1);
    assert_eq!(joined.matches("self.dispatches = 0").count(), 1);
    assert_eq!(joined.matches("self.position =").count(), 1);
    let guards = joined.rfind("self.guard_private(position + 1)?").unwrap();
    let gate = joined.find("self.check_fault()?").unwrap();
    let fence = joined.find(".completed_fence(").unwrap();
    let publication = joined.find("self.position = plan.end").unwrap();
    assert!(guards < gate && gate < fence && fence < publication);
    assert!(!joined.contains("self.complete_fault()") && !joined.contains("self.start("));
    assert_eq!(joined.matches("self.fail()").count(), 2); // returned error and panic
    assert!(joined.contains("self.key_value_at(position)?"));
    let append = source
        .split_once("pub fn append_kv(")
        .unwrap()
        .1
        .split_once("pub fn repair_accepted_cache(")
        .unwrap()
        .0;
    assert!(
        append.contains("self.start(token, carry)?;")
            && append.contains("self.key_value()?;")
            && append.contains("self.complete_fault()?;")
    );
    let rewind = source
        .split_once("pub fn rewind(")
        .unwrap()
        .1
        .split_once("pub fn close(")
        .unwrap()
        .0;
    assert!(
        rewind.contains("self.clear_scratch()?;") && rewind.contains("self.complete_fault()?;")
    );
    let clear = source
        .split_once("fn clear_repair_scratch(")
        .unwrap()
        .1
        .split_once("fn start(")
        .unwrap()
        .0;
    assert!(clear.contains("self.clear_scratch_preserving_inputs(true)?"));
    assert!(clear.contains("[B::Token as usize, B::Carry as usize]"));
    assert!(!clear.contains("B::Fault") && !clear.contains("self.position"));
}

#[test]
fn original_cache_shaders_keep_sticky_faults_in_checked_unchecked_runtime_spirv() {
    use cubecl::ir::{ElemType, UIntKind};
    use cubecl::{Compiler, WgpuCompilationOptions, server::ExecutionMode, wgpu::WgslCompiler};
    for rewind in [false, true] {
        for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
            let mut b = KernelBuilder::default();
            b.runtime_properties(WgpuRuntime::target_properties());
            AddressType::U32.register(&mut b.scope);
            let f = f32::as_type(&b.scope);
            let a = Atomic::<u32>::as_type(&b.scope);
            let keys = b.output_array(f).into();
            let values = b.output_array(f).into();
            let fault = b.output_array(a).into();
            let position = b
                .scalar(StorageType::Scalar(ElemType::UInt(UIntKind::U32)))
                .into();
            if rewind {
                let start = b
                    .scalar(StorageType::Scalar(ElemType::UInt(UIntKind::U32)))
                    .into();
                k::rewind_kv::expand(&mut b.scope, keys, values, fault, position, start);
            } else {
                let key = b.input_array(f).into();
                let value = b.input_array(f).into();
                k::append_kv::expand(&mut b.scope, key, value, keys, values, fault, position);
            }
            let source = WgslCompiler::default()
                .compile(
                    b.build(
                        KernelSettings::default()
                            .cube_dim(CubeDim::new_1d(64))
                            .kernel_name("mtp_batched_original_cache"),
                    ),
                    &WgpuCompilationOptions::default(),
                    mode,
                    AddressType::U32.unsigned_type(),
                )
                .unwrap()
                .to_string();
            let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
            let info = wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::SUBGROUP,
            )
            .validate(&module)
            .unwrap();
            assert!(source.contains("atomicOr"));
            assert!(!source.contains("atomicStore") && !source.contains("fma("));
            let options = wgpu::naga::back::spv::Options {
                zero_initialize_workgroup_memory:
                    wgpu::naga::back::spv::ZeroInitializeWorkgroupMemoryMode::None,
                ..Default::default()
            };
            let words = wgpu::naga::back::spv::write_vec(&module, &info, &options, None).unwrap();
            assert_eq!(words[0], 0x0723_0203);
        }
    }
}
