//! CPU/source acceptance only. No GPU request, resident model or device creation.
use super::*;

// Independent scalar oracle uses a full vocabulary sort, not the device's
// four-way and61-way merges. Tests below exercise their index/selection geometry.
fn ordered(logits: &[f32], history: &[i32], p: &Params) -> Vec<(u32, f32)> {
    let h = &history[history.len().saturating_sub(p.penalty_last_n)..];
    let mut v: Vec<_> = logits
        .iter()
        .enumerate()
        .map(|(id, &x)| {
            let n = h.iter().filter(|&&t| t == id as i32).count();
            let y = if n == 0 {
                x
            } else {
                let repeated = if x <= 0. {
                    x * p.penalty_repeat
                } else {
                    x / p.penalty_repeat
                };
                repeated - (n as f32 * p.penalty_freq + p.penalty_present)
            };
            (id as u32, y)
        })
        .collect();
    v.sort_by(|a, b| {
        if a.1 == b.1 {
            a.0.cmp(&b.0)
        } else {
            b.1.partial_cmp(&a.1).unwrap()
        }
    });
    v
}
fn split(v: &[(u32, f32)], k: usize) -> Vec<(u32, f32)> {
    fn merge(lists: &[Vec<(u32, f32)>], k: usize) -> Vec<(u32, f32)> {
        let mut cursor = vec![0; lists.len()];
        let mut out = Vec::new();
        for _ in 0..k {
            let best = lists
                .iter()
                .enumerate()
                .filter_map(|(i, l)| l.get(cursor[i]).map(|p| (i, *p)))
                .max_by(|a, b| {
                    a.1.1
                        .partial_cmp(&b.1.1)
                        .unwrap()
                        .then_with(|| b.1.0.cmp(&a.1.0))
                });
            let Some((i, p)) = best else { break };
            out.push(p);
            cursor[i] += 1;
        }
        out
    }
    let mut by_id = v.to_vec();
    by_id.sort_by_key(|p| p.0);
    let blocks: Vec<_> = by_id
        .chunks(BLOCK)
        .map(|b| {
            let warps: Vec<_> = b
                .chunks(1024)
                .map(|w| {
                    let mut sorted = w.to_vec();
                    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
                    sorted.truncate(k);
                    sorted
                })
                .collect();
            merge(&warps, k)
        })
        .collect();
    merge(&blocks, k)
}
fn draw(seed: u64, counter: u64) -> f32 {
    //64-bit integer products rather than shader's sixteen-bit mulhi decomposition.
    let mut c = [
        counter as u32,
        (counter >> 32) as u32,
        seed as u32,
        (seed >> 32) as u32,
    ];
    for round in 0..10 {
        let a = 0x9e3779b9u64 * u64::from(c[0]);
        let b = 0xbb67ae85u64 * u64::from(c[2]);
        c = [
            (b >> 32) as u32 ^ c[1] ^ round,
            b as u32,
            (a >> 32) as u32 ^ c[3],
            a as u32,
        ];
    }
    (c[0] >> 8) as f32 / 16777216.
}
fn pick(sorted: &[(u32, f32)], p: &Params, counter: u64) -> (u32, f32) {
    if p.greedy() {
        return (sorted[0].0, 1.);
    }
    let k = p.k().min(sorted.len());
    let mut keep = k;
    if p.top_p < 1. {
        let ex: Vec<_> = sorted[..k]
            .iter()
            .map(|p| (p.1 - sorted[0].1).exp())
            .collect();
        let sum = ex.iter().copied().fold(0f32, |a, b| a + b);
        let mut cum = 0.;
        for (i, &v) in ex.iter().enumerate() {
            cum += v / sum;
            if cum >= p.top_p {
                keep = i + 1;
                break;
            }
        }
        keep = keep.max(p.min_keep.min(k));
    }
    if p.min_p > 0. {
        let cut = sorted[0].1 + p.min_p.ln();
        if let Some(i) = sorted[..keep].iter().position(|p| p.1 < cut) {
            keep = i;
        }
    }
    if keep == 0 {
        return (sorted[0].0, 1.);
    }
    let inv = 1. / p.temperature;
    let mx = sorted[0].1 * inv;
    let ex: Vec<_> = sorted[..keep]
        .iter()
        .map(|p| (p.1 * inv - mx).exp())
        .collect();
    let sum = ex.iter().copied().fold(0f32, |a, b| a + b);
    let u = draw(p.seed, counter);
    let mut cum = 0.;
    for (i, &v) in ex.iter().enumerate() {
        cum += v / sum;
        if u < cum {
            return (sorted[i].0, v / sum);
        }
    }
    (sorted[keep - 1].0, ex[keep - 1] / sum)
}
#[test]
fn native_sampler_selection_and_tail_preserve_order_and_probability() {
    let logits: Vec<_> = (0..VOCAB)
        .map(|i| {
            if i % 7 == 0 {
                0.
            } else {
                ((i * 65537 % 997) as f32 - 498.) / 113.
            }
        })
        .collect();
    let h = [-1, 0, 0, 1, 1, 1, 1024, 4095, 245760, 248319, VOCAB as i32];
    for k in [1, 20, 64] {
        let p = Params {
            top_k: k,
            top_p: 0.83,
            min_p: 0.02,
            temperature: 0.7,
            penalty_last_n: 9,
            penalty_repeat: 1.13,
            penalty_freq: 0.8,
            penalty_present: 0.3,
            seed: 5713,
            ..Default::default()
        };
        let sorted = ordered(&logits, &h, &p);
        let split = split(&sorted, p.k());
        assert_eq!(split, sorted[..p.k()]);
        for counter in [0, 1, 4095, u32::MAX as u64, u64::MAX] {
            assert_eq!(pick(&sorted, &p, counter), pick(&split, &p, counter));
        }
    }
    let p = Params {
        penalty_last_n: 3,
        penalty_repeat: 2.,
        penalty_freq: 0.5,
        penalty_present: 0.25,
        ..Default::default()
    };
    assert_eq!(
        ordered(&[-2., 2., 0., 2.], &[0, 1, 1], &p),
        vec![(3, 2.), (2, 0.), (1, -0.25), (0, -4.75)]
    );
}
#[test]
fn native_sampler_philox_pins_nonstandard_upstream_multipliers_and_counter_carry() {
    for (seed, counter, bits) in [
        (0, 0, 0xf559f9),
        (1, 0, 0x57fa11),
        (0, 1, 0x50c57a),
        (0x123456789abcdef0, 0x1ffffffff, 0xaa2be4),
        (u64::MAX, u64::MAX, 0xa4ac2a),
    ] {
        assert_eq!(draw(seed, counter), (bits as f32) / 16777216.);
    }
    fn shader_mulhi(a: u32, b: u32) -> u32 {
        let (al, ah, bl, bh) = (a & 65535, a >> 16, b & 65535, b >> 16);
        let w0 = al * bl;
        let t = ah * bl + (w0 >> 16);
        let w1 = (t & 65535) + al * bh;
        ah.wrapping_mul(bh)
            .wrapping_add(t >> 16)
            .wrapping_add(w1 >> 16)
    }
    let mut r = 0x12345678u32;
    for _ in 0..10000 {
        r = r.wrapping_mul(1664525).wrapping_add(1013904223);
        for a in [0, 1, 0x9e3779b9, 0xbb67ae85, u32::MAX] {
            assert_eq!(
                shader_mulhi(a, r),
                ((u64::from(a) * u64::from(r)) >> 32) as u32
            );
        }
    }
    let p = Params {
        counter: u32::MAX as u64,
        ..Default::default()
    };
    let w = p.words(2, 0, 0, 2).unwrap();
    assert_eq!(w[8], u32::MAX as u64);
}
#[test]
fn native_sampler_parameters_and_device_payload_do_not_change_kernel_identity() {
    let mut p = Params::default();
    assert_eq!(p.k(), 20);
    assert_eq!(p.words(1, 0, 0, 2).unwrap().len(), 16);
    for k in [0, 65, i32::MAX] {
        p.top_k = k;
        assert_eq!(p.k(), 64);
    }
    p.temperature = 0.;
    assert!(p.greedy());
    assert_eq!(p.k(), 1);
    p.temperature = f32::NAN;
    assert!(p.validate().is_err());
    p.temperature = 1.;
    p.top_k = -1;
    assert!(p.validate().is_err());
    p.top_k = 20;
    assert!(p.words(0, 0, 0, 2).is_err());
    assert!(p.words(1, 0, 0, 1).is_err());
    assert!(p.words(2, 0, u32::MAX as usize, 2).is_err());
    assert!(
        p.words(
            8.min(crate::strata_port::verify_capacity()),
            0,
            0,
            u32::MAX as usize
        )
        .is_err()
    );
    p.min_keep = 0;
    p.penalty_last_n = 10000;
    assert!(p.validate().is_ok());
    p.min_keep = 1;
    p.penalty_last_n = 0;
    assert!(p.words(1, 4097, 0, 2).is_err());
    p.penalty_last_n = 1;
    assert!(p.words(1, 0, 0, 2).is_err());
    assert!(scratch_bytes(0).is_err());
    assert_eq!(scratch_bytes(1).unwrap(), 31232);
    assert_eq!(Kernel::Part.id(), Kernel::Part.id());
    assert_ne!(Kernel::Greedy { penalties: false }.id(), Kernel::Part.id());
    assert_ne!(Kernel::Part.id(), Kernel::Merge.id());
}
#[test]
fn native_sampler_shader_set_is_valid_and_lowers_to_spirv_without_a_device() {
    use wgpu::naga as n;
    for s in sources() {
        let module = n::front::wgsl::parse_str(&s.source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&s.source)));
        let info = n::valid::Validator::new(
            n::valid::ValidationFlags::all(),
            n::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&s.source)));
        let globals: Vec<_> = module
            .global_variables
            .iter()
            .filter_map(|(_, g)| g.binding.as_ref().map(|b| (b.binding, g)))
            .collect();
        assert_eq!(globals.len(), 6);
        for (binding, g) in &globals {
            assert!(*binding < 6);
            assert!(matches!(g.space, n::AddressSpace::Storage { .. }));
        }
        let params = globals.iter().find(|(binding, _)| *binding == 5).unwrap().1;
        assert!(matches!(
            &module.types[params.ty].inner,
            n::TypeInner::Struct { span: 128, .. }
        ));
        assert_eq!(module.entry_points[0].workgroup_size, [s.cube_dim.x, 1, 1]);
        let bytes = n::back::spv::write_vec(
            &module,
            &info,
            &n::back::spv::Options {
                lang_version: (1, 3),
                zero_initialize_workgroup_memory:
                    n::back::spv::ZeroInitializeWorkgroupMemoryMode::None,
                ..Default::default()
            },
            Some(&n::back::spv::PipelineOptions {
                shader_stage: n::ShaderStage::Compute,
                entry_point: "main".into(),
            }),
        )
        .unwrap();
        assert!(bytes.len() > 100);
    }
}

#[test]
fn native_sampler_target_histories_and_draft_base_are_the_same_consumed_prefix() {
    assert_eq!(
        &*penalty_rows(&[10, 20], &[30, 40, 50], 4).unwrap(),
        &[-1, 10, 20, 30, 10, 20, 30, 40, 20, 30, 40, 50]
    );
    assert_eq!(
        &*penalty_rows(&[], &[0, 2], 3).unwrap(),
        &[-1, -1, 0, -1, 0, 2]
    );
    for n in [0, 1, 4095, 4096] {
        let tail: Vec<_> = (0..n as u32).collect();
        let window: Vec<_> = (0..crate::strata_port::verify_capacity() as u32)
            .map(|n| n + 5000)
            .collect();
        for width in [0, 1, 7, 4096] {
            let rows = penalty_rows(&tail, &window, width).unwrap();
            assert_eq!(
                &rows[..width.max(1)],
                &*coupled_base(&tail, window[0], width).unwrap()
            );
            for row in 0..window.len() {
                let mut all = vec![-1; width];
                all.extend(tail.iter().chain(&window[..=row]).map(|&id| id as i32));
                if width > 0 {
                    assert_eq!(
                        &rows[row * width..(row + 1) * width],
                        &all[all.len() - width..]
                    );
                }
            }
        }
    }
    assert!(penalty_rows(&[VOCAB as u32], &[0], 1).is_err());
    assert!(penalty_rows(&[], &[], 1).is_err());
    assert!(penalty_rows(&[], &[0], 4097).is_err());
    for cell in [0, 1, 7, 4095] {
        assert_eq!(coupled_counter(cell).unwrap(), cell as u64 + 1);
    }
    assert!(coupled_counter(usize::MAX).is_err());
}

#[test]
fn native_sampler_device_payload_has_exact_layout_and_no_graph_uniform() {
    let p = Params {
        top_k: 64,
        top_p: 0.8,
        min_p: 0.1,
        temperature: 0.4,
        min_keep: 3,
        penalty_last_n: 4096,
        penalty_repeat: 1.1,
        penalty_freq: 0.3,
        penalty_present: 0.2,
        seed: 0x123456789abcdef0,
        counter: 0x1ffffffff,
        greedy: false,
    };
    let words = p.words(1, 4096, 6, 10).unwrap();
    let bytes: Vec<_> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let read = |i: usize| u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
    assert_eq!(
        (read(0), read(1), read(2), read(3)),
        (1, VOCAB as u32, 64, 61)
    );
    assert_eq!((read(4), read(5), read(6), read(7)), (4096, 4096, 3, 0));
    assert_eq!(
        (read(8), read(9), read(10), read(11)),
        (
            0.8f32.to_bits(),
            0.1f32.to_bits(),
            0.4f32.to_bits(),
            1.1f32.to_bits()
        )
    );
    assert_eq!(
        (read(12), read(13), read(14), read(15)),
        (0.3f32.to_bits(), 0.2f32.to_bits(), 0x9abcdef0, 0x12345678)
    );
    assert_eq!(
        (read(16), read(17), read(18), read(19)),
        (u32::MAX, 1, 6, 10)
    );
    assert!((20..32).all(|i| read(i) == 0));
    assert!(p.words(1, 4096, u32::MAX as usize - 1, 2).is_err());
    assert!(p.words(1, 4096, u32::MAX as usize - 2, 2).is_ok());
    assert_ne!(
        Kernel::Greedy { penalties: true }.id(),
        Kernel::Greedy { penalties: false }.id()
    );
}

#[test]
fn native_sampler_coupled_draft_histories_match_every_target_row_for_each_window() {
    let tail = [3, 4, 5, 6, 7];
    for count in 1..=crate::strata_port::verify_capacity() {
        let window: Vec<_> = (10..10 + count as u32).collect();
        for h in [1, 4, 4096] {
            let rows = penalty_rows(&tail, &window, h).unwrap();
            let mut ring = coupled_base(&tail, window[0], h).unwrap().to_vec();
            for j in 0..count {
                // Draft j reads exactly this tail, then appends its ACTUAL pick
                // only for the next step. Target row j reads the same prefix.
                assert_eq!(&ring[ring.len() - h..], &rows[j * h..(j + 1) * h]);
                let cell = tail.len() - 1 + j;
                assert_eq!(coupled_counter(cell).unwrap(), (tail.len() + j) as u64);
                if let Some(&next) = window.get(j + 1) {
                    ring.push(next as i32);
                }
            }
        }
    }
}

#[test]
fn native_cjk_subset_keeps_original_tie_order_and_penalty_ids() {
    let raw = include_bytes!("../fixtures/strata-cjk-vocab/draft_vocab_cjk_original.bin");
    let ids: Vec<_> = raw
        .chunks_exact(4)
        .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    assert_eq!(ids.len(), 106299);
    assert_eq!(ids.len().div_ceil(BLOCK), 26);
    let mut inverse = vec![-1; VOCAB];
    for (compact, &original) in ids.iter().enumerate() {
        inverse[original as usize] = compact as i32;
    }
    let absent = inverse.iter().position(|&v| v == -1).unwrap() as u32;
    let history = [ids[0], absent, ids[0], ids[1]];
    let compact_history: Vec<_> = history.iter().map(|&id| inverse[id as usize]).collect();
    assert_eq!(compact_history, [0, -1, 0, 1]);
    let logits = vec![0.; ids.len()];
    let p = Params {
        top_k: 20,
        penalty_last_n: 4,
        penalty_freq: 1.,
        top_p: 1.,
        seed: 5713,
        ..Default::default()
    };
    let ranked = ordered(&logits, &compact_history, &p);
    // Absent original tokens never penalize compact token0; true repeats do.
    assert_eq!(ranked[0], (2, 0.));
    assert_eq!(ranked[ranked.len() - 1], (0, -2.));
    assert_eq!(ranked[ranked.len() - 2], (1, -1.));
    let split = split(&ranked, p.k());
    assert_eq!(split, ranked[..p.k()]);
    for counter in [0, 1, 4095] {
        let (compact, probability) = pick(&split, &p, counter);
        assert_eq!(pick(&ranked, &p, counter), (compact, probability));
        let original = ids[compact as usize];
        assert_eq!(inverse[original as usize], compact as i32);
        assert_eq!(probability, 1. / 20.);
    }
    assert_ne!(Kernel::CoupledPart { cjk: true }.id(), Kernel::Part.id());
    assert_ne!(Kernel::CoupledMerge { cjk: true }.id(), Kernel::Merge.id());
}

#[test]
fn long_request_context_keeps_only_the_native_penalty_window_in_target_and_draft_histories() {
    for start in [HISTORY_MAX - 1, HISTORY_MAX, 8192, CONTEXT_MAX - 1] {
        if start > CONTEXT_MAX {
            continue;
        }
        let tail = (0..start)
            .map(|i| ((i * 31 + 7) % VOCAB) as u32)
            .collect::<Vec<_>>();
        let window = [17, 29, 41, 53];
        for width in [0, 1, 7, HISTORY_MAX] {
            let actual = penalty_rows(&tail, &window, width).unwrap();
            assert_eq!(
                &actual[..width.max(1)],
                &*coupled_base(&tail, window[0], width).unwrap()
            );
            if width == 0 {
                assert_eq!(&*actual, &[-1]);
                continue;
            }
            for row in 0..window.len() {
                let full = tail
                    .iter()
                    .chain(&window[..=row])
                    .copied()
                    .collect::<Vec<_>>();
                let mut expected = vec![-1i32; width];
                let n = width.min(full.len());
                for (out, &token) in expected[width - n..]
                    .iter_mut()
                    .zip(&full[full.len() - n..])
                {
                    *out = token as i32;
                }
                assert_eq!(&actual[row * width..(row + 1) * width], &expected);
            }
        }
    }
    assert!(penalty_rows(&vec![1; CONTEXT_MAX + 1], &[2], 1).is_err());
    assert!(penalty_rows(&[1], &[2], HISTORY_MAX + 1).is_err());
}

#[test]
fn original_coupled_penalty_writes_and_fixed64_shortlists_match_independent_full_sort() {
    // Public synthetic oracle: full-sort penalties and selection do not use the
    // shader's bitmap winner or fixed64 per-warp/per-block merge architecture.
    for vocabulary in [VOCAB, crate::resident_mtp::cjk::ROWS] {
        let logits = (0..vocabulary)
            .map(|i| ((i * 65537 % 997) as f32 - 498.) / 113.)
            .collect::<Vec<_>>();
        let history = (0..3073)
            .map(|i| match i % 7 {
                0 => -1,
                1 => 0,
                2 => 1,
                3 => (vocabulary - 1) as i32,
                _ => ((i * 37) % vocabulary) as i32,
            })
            .collect::<Vec<_>>();
        let p = Params {
            top_k: 20,
            top_p: 0.83,
            min_p: 0.02,
            temperature: 0.7,
            penalty_last_n: 2049,
            penalty_repeat: 1.13,
            penalty_freq: 0.8,
            penalty_present: 0.3,
            seed: 5713,
            ..Default::default()
        };
        let expected = ordered(&logits, &history, &p);
        for permutation in [0, 1, 2] {
            let mut mutated = logits.clone();
            let mut seen = vec![false; vocabulary];
            let h = &history[history.len() - p.penalty_last_n..];
            for thread in 0..1024 {
                let lane = match permutation {
                    0 => thread,
                    1 => 1023 - thread,
                    _ => thread * 37 % 1024,
                };
                for at in (lane..h.len()).step_by(1024) {
                    let id = h[at];
                    if id < 0 || id as usize >= vocabulary {
                        continue;
                    }
                    let id = id as usize;
                    if std::mem::replace(&mut seen[id], true) {
                        continue;
                    }
                    let count = h.iter().filter(|&&v| v == id as i32).count();
                    let x = mutated[id];
                    mutated[id] = if x <= 0. {
                        x * p.penalty_repeat
                    } else {
                        x / p.penalty_repeat
                    } - (count as f32 * p.penalty_freq + p.penalty_present);
                }
            }
            let sorted = ordered(&mutated, &[], &p);
            assert_eq!(sorted, expected);
            // Actual coupled source always produces64 entries per part, even
            // when only1/20 survive the request's final top-k selection.
            let fixed = split(&sorted, KMAX);
            assert_eq!(fixed, expected[..KMAX]);
            for k in [1, 20, 64] {
                let q = Params {
                    top_k: k,
                    top_p: p.top_p,
                    min_p: p.min_p,
                    temperature: p.temperature,
                    seed: p.seed,
                    ..Default::default()
                };
                for counter in [1, 4096, u32::MAX as u64 + 1] {
                    assert_eq!(pick(&fixed, &q, counter), pick(&expected, &q, counter));
                }
            }
        }
    }
}

#[test]
fn actual_coupled_schedule_preserves_logits_write_fixed64_reads_and_mutable_tail_k() {
    use wgpu::naga as n;
    for cjk in [false, true] {
        let schedule = coupled_kernels(cjk);
        assert_eq!(
            schedule,
            [
                Kernel::CoupledPenalize { cjk },
                Kernel::CoupledPart { cjk },
                Kernel::CoupledMerge { cjk }
            ]
        );
        assert_eq!(schedule.map(Kernel::threads), [1024, 128, 32]);
        for (index, kernel) in schedule.into_iter().enumerate() {
            let source = kernel.source();
            assert!(source.contains("return p.dims.x==1u&&p.hist.w==0u"));
            let module = n::front::wgsl::parse_str(&source).unwrap();
            let logits = module
                .global_variables
                .iter()
                .find(|(_, g)| g.name.as_deref() == Some("logits"))
                .unwrap()
                .1;
            assert_eq!(
                logits.space,
                n::AddressSpace::Storage {
                    access: if index == 0 {
                        n::StorageAccess::LOAD | n::StorageAccess::STORE
                    } else {
                        n::StorageAccess::LOAD
                    }
                }
            );
            if index == 0 {
                assert!(source.contains("logits[v]=value"));
                assert!(source.contains("atomicOr(&seen[v>>5u],bit)"));
            } else if index == 1 {
                assert!(source.contains("let h=0u;"));
                assert!(source.contains("for(var r=0u;r<KM;r++)"));
                assert!(source.contains("candidates[(row*NB+group.x)*KM+r]"));
            } else {
                assert!(source.contains("let k=p.dims.z;"));
                assert!(source.contains("i<NB*KM"));
                assert!(source.contains("lists[list*KM]"));
                assert!(source.contains("cursor[m]<KM"));
                assert!(source.contains("for(var r=0u;r<k;r++)"));
            }
        }
        assert_ne!(Kernel::CoupledPart { cjk }.id(), Kernel::Part.id());
        assert_ne!(Kernel::CoupledMerge { cjk }.id(), Kernel::Merge.id());
    }
    // Target sampler still uses its own request-sized, read-only logits path.
    let part = Kernel::Part.source();
    assert!(part.contains("let h=min(p.hist.x,p.hist.y);"));
    assert!(part.contains("candidates[(row*NB+group.x)*p.dims.z+r]"));
}
