use super::*;

#[test]
fn predevice_limits_cover_complete_public_pages_and_seven_bindings() {
    let limits = wgpu::Limits::default();
    assert!(validate_limits(&limits).is_ok());
    let mut short = limits.clone();
    short.max_compute_workgroups_per_dimension = 65534;
    assert!(validate_limits(&short).is_err());
    short = limits.clone();
    short.max_storage_buffer_binding_size = 128 * 1024 * 1024 - 1;
    assert!(validate_limits(&short).is_err());
    short = limits;
    short.max_storage_buffers_per_shader_stage = 6;
    assert!(validate_limits(&short).is_err());
}

#[test]
fn exact_original_tensor_registry_and_full_page_plan() {
    // Independent original34-role inventory: eighteen ordinary matrices, one
    // token table, three expert families, eight F32 vectors, four unused
    // indexers. No optional family may silently disappear from residency.
    let mut names = std::collections::BTreeSet::new();
    for spec in MATRICES.into_iter().chain([HEAD_UP, EMBEDDING]) {
        assert!(names.insert(spec.name));
        assert!(matches!(spec.dtype, quant_matvec::Q8_0 | quant_matvec::F32));
        let pages = tile_plan(spec).unwrap();
        let row_bytes = if spec.dtype == quant_matvec::Q8_0 {
            spec.columns / 32 * 34
        } else {
            spec.columns * 4
        };
        assert_eq!(pages.iter().map(|p| p.1).sum::<usize>(), spec.rows);
        for &(first, rows, logical, owned) in &pages {
            assert_eq!(first % 16384, 0);
            assert_eq!(logical, rows * row_bytes);
            assert!(logical <= owned && owned <= PUBLIC_SOURCE_BYTES);
            assert!(owned.is_power_of_two() && owned >= 32768);
        }
    }
    for name in EXPERT_NAMES {
        assert!(names.insert(name));
    }
    for (name, _) in VECTORS {
        assert!(names.insert(name));
    }
    assert_eq!(names.len(), 30);
    assert!(!names.iter().any(|n| n.contains("indexer")));
    assert_eq!(
        MATRICES
            .iter()
            .filter(|s| s.dtype == quant_matvec::F32)
            .count(),
        1
    );
    assert_eq!(public_plan().unwrap(), 5_011_668_992);
    // Independent subplans from inspected original shapes and allocator bins.
    let embeddings = 15 * 64 * 1024 * 1024 + 8 * 1024 * 1024;
    let dense = 1_126_301_696;
    let banks = 3 * (7 * 128 * 1024 * 1024 + 16 * 1024 * 1024);
    let vectors = 4 * 65536 + 4 * 32768;
    assert_eq!(
        public_plan().unwrap(),
        embeddings + dense + banks + vectors + 2 * 524288
    );
}

#[test]
fn private_plan_owns_complete_two_kv_caches_and_separate_work_roles() {
    assert_eq!(N, 42);
    assert_eq!(private_plan().unwrap(), 11_698_176);
    assert_eq!(WORDS[B::Keys as usize], 2048 * 2 * 256);
    assert_eq!(WORDS[B::Values as usize], 2048 * 2 * 256);
    assert_eq!(
        BufferLayout::new(WORDS[B::Keys as usize])
            .unwrap()
            .owned_bytes,
        4 * 1024 * 1024
    );
    assert_eq!(
        BufferLayout::new(WORDS[B::Logits as usize])
            .unwrap()
            .owned_bytes,
        1024 * 1024
    );
    assert!(
        main_admission::check_capacity(
            MAIN_BYTES,
            &[
                PUBLIC_BYTES,
                PRIVATE_BYTES as u64,
                RUNTIME_BYTES,
                PUBLIC_SOURCE_BYTES as u64
            ]
        )
        .is_ok()
    );
    assert!(private_plan().unwrap() <= PRIVATE_BYTES as u64);
    const {
        assert!(INFO_BYTES + 3 * 4 * 1024 * 1024 <= RUNTIME_BYTES);
    }
    // Actual full-span read conversion, retained full output/carry and mapped
    // lease are precharged independently of logical lengths.
    for n in WORDS {
        let layout = BufferLayout::new(n).unwrap();
        assert!(
            2 * layout.owned_bytes + MAX_LIVE_OUTPUTS * OUTPUT_HOST_BYTES + 2 * 1024 * 1024
                <= HOST_BYTES
        );
    }
    assert!(BufferLayout::new(k::CONDITION_WORDS).unwrap().owned_words >= 4 * 5120);
}

#[test]
fn live_output_leases_bound_retention_and_return_credit_only_on_drop() {
    let live = Arc::new(AtomicUsize::new(0));
    let mut held = (0..MAX_LIVE_OUTPUTS)
        .map(|_| OutputLease::acquire(&live).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(live.load(Ordering::Acquire), 4);
    assert!(OutputLease::acquire(&live).is_err());
    assert_eq!(live.load(Ordering::Acquire), 4);
    drop(held.pop());
    assert_eq!(live.load(Ordering::Acquire), 3);
    let last = OutputLease::acquire(&live).unwrap();
    assert_eq!(live.load(Ordering::Acquire), 4);
    drop(held);
    drop(last);
    assert_eq!(live.load(Ordering::Acquire), 0);
}

#[test]
fn original_q8_source_checks_every_half_scale_without_decoding_weights() {
    let mut packed = vec![0_u8; 34 * 41];
    for block in packed.chunks_exact_mut(34) {
        block[..2].copy_from_slice(&0x0401_u16.to_le_bytes());
        for (i, q) in block[2..].iter_mut().enumerate() {
            *q = (i as i8 - 16) as u8;
        }
    }
    assert!(validate_source(&packed, quant_matvec::Q8_0).is_ok());
    for half in [0, 1, 0x03ff, 0x0400, 0x7bff, 0xfbff, 0x8000] {
        packed[34 * 40..34 * 40 + 2].copy_from_slice(&(half as u16).to_le_bytes());
        assert!(validate_source(&packed, quant_matvec::Q8_0).is_ok());
    }
    for half in [0x7c00, 0xfc00, 0x7c01, 0x7fff, 0xffff] {
        for block in [0, 1, 20, 40] {
            let mut bad = packed.clone();
            bad[34 * block..34 * block + 2].copy_from_slice(&(half as u16).to_le_bytes());
            assert!(validate_source(&bad, quant_matvec::Q8_0).is_err());
        }
    }
    assert!(validate_source(&packed[..packed.len() - 1], quant_matvec::Q8_0).is_err());
    assert!(validate_source(&packed, quant_matvec::Q4_K).is_err());
}

#[test]
fn f32_source_and_private_input_reject_nonfinite_or_wrong_shape_before_writes() {
    let valid = [
        f32::MIN,
        f32::MAX,
        0.0,
        -0.0,
        f32::MIN_POSITIVE,
        f32::from_bits(1),
    ];
    let bytes = valid
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect::<Vec<_>>();
    assert!(validate_source(&bytes, quant_matvec::F32).is_ok());
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let bytes = bad.to_le_bytes();
        assert!(validate_source(&bytes, quant_matvec::F32).is_err());
    }
    assert!(validate_source(&[0, 0, 0], quant_matvec::F32).is_err());
    let mut carry = vec![0.0; k::WIDE];
    assert!(validate_input(0, &carry, 0).is_ok());
    assert!(validate_input((k::VOCAB - 1) as u32, &carry, k::CONTEXT - 1).is_ok());
    assert!(validate_input(k::VOCAB as u32, &carry, 0).is_err());
    assert!(validate_input(0, &carry, k::CONTEXT).is_err());
    assert!(validate_input(0, &carry[..k::WIDE - 1], 0).is_err());
    carry[k::WIDE - 1] = f32::NAN;
    assert!(validate_input(0, &carry, 0).is_err());
}

#[test]
fn original_multi_rhs_eh_geometry_owns_four_distinct_conditions_and_results() {
    let geometry = matmul::MatrixGeometry::new(quant_matvec::Q8_0, 2560, 5120, 4).unwrap();
    assert_eq!(geometry.row_bytes(), 5120 / 32 * 34);
    let spans = matmul::BatchSpans::new(4, 0, 0, 5120, 0, 2560);
    assert!(
        geometry
            .validate_spans(&spans, 2560 * 5120 / 32 * 34 / 4, 20480, 10240)
            .is_ok()
    );
    assert!(
        geometry
            .validate_spans(&spans, 2560 * 5120 / 32 * 34 / 4 - 1, 20480, 10240)
            .is_err()
    );
    assert!(
        geometry
            .validate_spans(&spans, 2560 * 5120 / 32 * 34 / 4, 20479, 10240)
            .is_err()
    );
    assert!(
        geometry
            .validate_spans(&spans, 2560 * 5120 / 32 * 34 / 4, 20480, 10239)
            .is_err()
    );
    let overlap = matmul::BatchSpans::new(4, 0, 0, 5119, 0, 2560);
    assert!(
        geometry
            .validate_spans(&overlap, 2560 * 5120 / 32 * 34 / 4, 20480, 10240)
            .is_err()
    );
}

#[test]
fn conditioning_hidden_norm_is_global_while_hc_norm_remains_per_stream() {
    // Unequal stream magnitudes distinguish the publisher's global hnorm from
    // the old accidental per-stream operation, which makes all four streams1.
    let hidden = [1.0_f32, 2.0, 4.0, 8.0]
        .into_iter()
        .flat_map(|v| std::iter::repeat_n(v, 2560))
        .collect::<Vec<_>>();
    let gamma = vec![1.0; 10240];
    assert_eq!(HIDDEN_NORM_GROUPS * HIDDEN_NORM_WIDTH, 10240);
    assert_eq!(HIDDEN_NORM_WIDTH, hidden.len());
    let cpu = kuno_text::ops::rms_norm(&hidden, &gamma, 10240, EPS).unwrap();
    let lanes =
        crate::residual_gpu::reference_rms(&hidden, &gamma, HIDDEN_NORM_WIDTH, EPS).unwrap();
    for (&a, &b) in cpu.iter().zip(&lanes) {
        assert!((a - b).abs() < 2e-6);
    }
    let denominator = (21.25_f64 + f64::from(EPS)).sqrt();
    assert!((f64::from(lanes[0]) - 1.0 / denominator).abs() < 2e-6);
    assert!((f64::from(lanes[10239]) - 8.0 / denominator).abs() < 2e-6);
    let hc = crate::residual_gpu::reference_rms(&hidden, &gamma, 2560, EPS).unwrap();
    assert!((hc[0] - hc[10239]).abs() < 2e-6);
    assert!((lanes[0] - lanes[10239]).abs() > 1.0);
}

#[test]
fn greedy_proof_includes_final_row_and_lowest_id_ties() {
    let mut values = vec![-f32::MAX; k::VOCAB];
    assert_eq!(greedy_host(&values).unwrap(), 0);
    values[k::VOCAB - 1] = 0.;
    assert_eq!(greedy_host(&values).unwrap(), (k::VOCAB - 1) as u32);
    values[4096] = 0.;
    assert_eq!(greedy_host(&values).unwrap(), 4096);
    values[k::VOCAB - 2] = f32::NAN;
    assert!(greedy_host(&values).is_err());
    assert!(greedy_host(&values[..k::VOCAB - 1]).is_err());
}

#[test]
fn draft_packet_consumes_exact_gpu_fields_and_every_logical_gap_and_tail() {
    let owned = BufferLayout::new(WORDS[B::RawQuery as usize])
        .unwrap()
        .owned_words;
    let mut packet = vec![0_u32; owned];
    for (i, word) in packet[..k::WIDE].iter_mut().enumerate() {
        *word = ((i as f32 - 5000.0) / 8192.0).to_bits();
    }
    packet[0] = (-0.0_f32).to_bits();
    packet[DRAFT_TOKEN] = (k::VOCAB - 1) as u32;
    let (hidden, token) = decode_draft(&packet).unwrap();
    assert_eq!(hidden.len(), 10240);
    assert_eq!(hidden[0].to_bits(), (-0.0_f32).to_bits());
    assert_eq!(hidden[k::WIDE - 1].to_bits(), packet[k::WIDE - 1]);
    assert_eq!(token, 248319);
    for length in [0, DRAFT_WORDS - 1, DRAFT_WORDS, owned - 1, owned + 1] {
        let mut bad = packet.clone();
        bad.resize(length, 0);
        assert!(decode_draft(&bad).is_err());
    }
    for index in [DRAFT_FAULT, DRAFT_WORDS, 12287, 12288, owned - 1] {
        let mut bad = packet.clone();
        bad[index] = 1;
        assert!(decode_draft(&bad).is_err());
    }
    for index in [0, k::WIDE - 1] {
        let mut bad = packet.clone();
        bad[index] = f32::NAN.to_bits();
        assert!(decode_draft(&bad).is_err());
    }
    packet[DRAFT_TOKEN] = k::VOCAB as u32;
    assert!(decode_draft(&packet).is_err());
}

#[test]
fn committed_source_stamp_is_bound_to_exact_previous_position_and_consumed_once() {
    let mut source = Some(LastForward {
        binding: 19,
        start: 3,
    });
    assert!(source.as_ref().unwrap().valid(19, 4));
    assert!(!source.as_ref().unwrap().valid(18, 4));
    for position in [0, 1, 3, 5, k::CONTEXT] {
        assert!(!source.as_ref().unwrap().valid(19, position));
    }
    let lease = source.take().unwrap();
    assert!(source.take().is_none());
    assert!(lease.valid(19, 4));
    drop(lease);
}

#[test]
fn minimal_packet_and_followup_use_existing_complete_owners_without_new_main() {
    const { assert!(DRAFT_WORDS < WORDS[B::RawQuery as usize]) };
    assert_eq!(
        BufferLayout::new(WORDS[B::RawQuery as usize])
            .unwrap()
            .owned_bytes,
        65536
    );
    assert_eq!(
        BufferLayout::new(WORDS[B::Residual2 as usize]).unwrap(),
        BufferLayout::new(WORDS[B::Carry as usize]).unwrap()
    );
    assert_eq!(
        BufferLayout::new(WORDS[B::Greedy as usize]).unwrap(),
        BufferLayout::new(WORDS[B::Token as usize]).unwrap()
    );
    assert_eq!(private_plan().unwrap(), 11_698_176);
    assert_eq!(public_plan().unwrap(), 5_011_668_992);
    let full_read_bytes: usize = [B::Fault, B::Logits, B::Residual2, B::Greedy]
        .into_iter()
        .map(|b| BufferLayout::new(WORDS[b as usize]).unwrap().owned_bytes)
        .sum();
    assert_eq!(full_read_bytes, 1_179_648);
    let input_upload_bytes: usize = [B::Token, B::Carry]
        .into_iter()
        .map(|b| BufferLayout::new(WORDS[b as usize]).unwrap().owned_bytes)
        .sum();
    assert_eq!(input_upload_bytes, 98_304);
    assert_eq!(MAX_TOKEN_DISPATCHES, 512);
    // Original265 explicit dispatches; packet clear/producer/seal add3.
    // Followup source copy adds1 and preserves3 clear destinations:266.
    assert!(
        [268_usize, 266]
            .into_iter()
            .all(|n| n <= MAX_TOKEN_DISPATCHES)
    );
}

#[test]
fn draft_copy_producer_and_seal_lower_with_full_bindings_and_no_fma() {
    use cubecl::{Compiler, WgpuCompilationOptions, server::ExecutionMode, wgpu::WgslCompiler};
    for which in 0..3 {
        for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
            let mut builder = KernelBuilder::default();
            builder.runtime_properties(WgpuRuntime::target_properties());
            AddressType::U32.register(&mut builder.scope);
            let u = u32::as_type(&builder.scope);
            let a = Atomic::<u32>::as_type(&builder.scope);
            match which {
                0 => {
                    let hidden = builder.input_array(u).into();
                    let greedy = builder.input_array(u).into();
                    let carry = builder.output_array(u).into();
                    let token = builder.output_array(u).into();
                    let fault = builder.output_array(a).into();
                    copy_committed::expand(&mut builder.scope, hidden, greedy, carry, token, fault);
                }
                1 => {
                    let hidden = builder.input_array(u).into();
                    let greedy = builder.input_array(u).into();
                    let packet = builder.output_array(u).into();
                    let fault = builder.output_array(a).into();
                    pack_draft::expand(&mut builder.scope, hidden, greedy, packet, fault);
                }
                _ => {
                    let packet = builder.output_array(u).into();
                    let fault = builder.input_array(a).into();
                    seal_draft::expand(&mut builder.scope, packet, fault);
                }
            }
            let source = WgslCompiler::default()
                .compile(
                    builder.build(
                        KernelSettings::default()
                            .cube_dim(CubeDim::new_1d(64))
                            .kernel_name("fixed_mtp_draft_packet"),
                    ),
                    &WgpuCompilationOptions::default(),
                    mode,
                    AddressType::U32.unsigned_type(),
                )
                .unwrap()
                .to_string();
            let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
            wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::SUBGROUP,
            )
            .validate(&module)
            .unwrap();
            assert!(!source.contains("fma(") && !source.contains("f64"));
            if which < 2 {
                assert!(source.contains("atomicOr"));
            } else {
                assert!(source.contains("atomicLoad"));
            }
        }
    }
}

#[test]
fn full_span_guards_lower_checked_and_unchecked_for_all_public_roles() {
    use cubecl::{Compiler, WgpuCompilationOptions, server::ExecutionMode, wgpu::WgslCompiler};
    for which in 0..4 {
        for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
            let mut builder = KernelBuilder::default();
            builder.runtime_properties(WgpuRuntime::target_properties());
            AddressType::U32.register(&mut builder.scope);
            let words = u32::as_type(&builder.scope);
            let atomic = Atomic::<u32>::as_type(&builder.scope);
            let input = builder.input_array(words).into();
            let fault = builder.output_array(atomic).into();
            let logical = usize::__expand_new(&mut builder.scope, 10240);
            if which < 2 {
                let live = usize::__expand_new(&mut builder.scope, 7168);
                private_guard::expand(&mut builder.scope, input, fault, logical, live, which == 0);
            } else {
                public_guard::expand(
                    &mut builder.scope,
                    input,
                    fault,
                    logical,
                    if which == 2 {
                        quant_matvec::Q8_0
                    } else {
                        quant_matvec::F32
                    },
                );
            }
            let source = WgslCompiler::default()
                .compile(
                    builder.build(
                        KernelSettings::default()
                            .cube_dim(CubeDim::new_1d(64))
                            .kernel_name("fixed_mtp_owner_guard"),
                    ),
                    &WgpuCompilationOptions::default(),
                    mode,
                    AddressType::U32.unsigned_type(),
                )
                .unwrap()
                .to_string();
            let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
            wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::SUBGROUP,
            )
            .validate(&module)
            .unwrap();
            assert!(source.contains("atomicOr"));
            assert!(!source.contains("fma(") && !source.contains("f64"));
        }
    }
}
