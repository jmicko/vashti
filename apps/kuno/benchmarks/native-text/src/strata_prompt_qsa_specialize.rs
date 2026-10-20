//! Prompt-only expansion of the selected Strata scalar QSA attention path.
//!
//! MIT Strata fb58e0, qsa_decode_attn.dp.cpp: load8_q8 and the explicitly
//! unrolled 12-head QK/PV loops. Keep each dot, XOR reduction and chronological
//! PV FMA in its original order. This changes neither geometry nor precision.
//! The base shader stays unchanged for decode/MTP and the public golden check.

fn replace_once(source: &mut String, old: &str, new: &str) {
    assert_eq!(
        source.matches(old).count(),
        1,
        "prompt attention specialization source drift"
    );
    *source = source.replacen(old, new, 1);
}

/// Specialize only the already-rebased prompt attention shader. This runs
/// after the original fault/status rewrite, preserving both closing contracts.
pub(super) fn source(mut scalar: String) -> String {
    assert!(scalar.contains("let query_row=t+p.reserved;"));
    assert!(scalar.contains("let n=cells(query_row,p.start);"));
    assert!(scalar.contains("fn kv(token:u32,offset:u32)->f32 {return q8_value(cache["));

    // Every one of these 12 stores is in range: tid is 0..255 and the shared
    // query row is exactly 12*256 elements. Descriptors/owning roots stay whole.
    let mut query_load = String::new();
    for h in 0..12 {
        query_load.push_str(&format!(
            "sq[tid+{}u]=query[query_row*6144u+kvh*3072u+tid+{}u];\n",
            h * 256,
            h * 256
        ));
    }
    replace_once(
        &mut scalar,
        "for(var i=tid;i<3072u;i+=256u) {sq[i]=query[query_row*6144u+kvh*3072u+i];}",
        &query_load,
    );

    // The eight K components always start at an eight-aligned lane offset.
    // Thus both packed code words share one original 64-value FP16 scale;
    // reuse it, without changing any signed byte, scale bit or multiplication.
    let mut k_load = String::from(
        "let koff=kvh*256u+lane*8u;\n\
         let kword0=cache[token*CACHE_WORDS+koff/4u];\n\
         let kword1=cache[token*CACHE_WORDS+koff/4u+1u];\n\
         let kscales=unpack2x16float(cache[token*CACHE_WORDS+256u+koff/128u]);\n\
         let kscale=select(kscales.x,kscales.y,((koff/64u)&1u)!=0u);\n",
    );
    for j in 0..8 {
        k_load.push_str(&format!(
            "let k{j}=f32(signed_code(kword{},{}u))*kscale;\nif !finite(k{j}) {{atomicOr(&broken,1u);}}\n",
            j / 4,
            j % 4
        ));
    }
    replace_once(
        &mut scalar,
        "var k:array<f32,8>;for(var j=0u;j<8u;j++) {k[j]=kv(token,kvh*256u+lane*8u+j);if !finite(k[j]) {atomicOr(&broken,1u);}}",
        &k_load,
    );

    // Keep the existing score/status/fault body and expand only its fixed
    // head/operand loops. Lexical blocks preserve all original local names.
    let begin = "for(var h=0u;h<12u;h++) {\n            let off=h*256u+lane*8u;";
    let end = "\n        }\n    }\n    workgroupBarrier();";
    let at = scalar.find(begin).expect("prompt QK head source drift");
    let body_at = at + "for(var h=0u;h<12u;h++) {".len();
    let body_end = body_at
        + scalar[body_at..]
            .find(end)
            .expect("prompt QK closing source drift");
    let mut body = scalar[body_at..body_end].to_owned();
    let mut dot = String::from("var dot=k0*sq[off];\n");
    for j in 1..8 {
        dot.push_str(&format!("dot+=k{j}*sq[off+{j}u];\n"));
    }
    replace_once(
        &mut body,
        "var dot=k[0]*sq[off];for(var j=1u;j<8u;j++) {dot+=k[j]*sq[off+j];}",
        &dot,
    );
    let mut heads = String::new();
    for h in 0..12 {
        heads.push_str(&format!("{{let h={h}u;{body}\n}}\n"));
    }
    // Consume only the head loop's closing brace. The outer cell loop and its
    // barrier remain byte-for-byte in their original chronological position.
    scalar.replace_range(at..body_end + "\n        }".len(), &heads);

    for (name, operation) in [
        ("dot", "dot+=subgroupShuffleXor(dot,DISTANCEu);"),
        (
            "maximum",
            "maximum=max(maximum,subgroupShuffleXor(maximum,DISTANCEu));",
        ),
        ("total", "total+=subgroupShuffleXor(total,DISTANCEu);"),
    ] {
        let old = match name {
            "dot" => "for(var d=16u;d>0u;d/=2u) {dot+=subgroupShuffleXor(dot,d);}",
            "maximum" => {
                "for(var d=16u;d>0u;d/=2u) {maximum=max(maximum,subgroupShuffleXor(maximum,d));}"
            }
            "total" => "for(var d=16u;d>0u;d/=2u) {total+=subgroupShuffleXor(total,d);}",
            _ => unreachable!(),
        };
        let new: String = [16, 8, 4, 2, 1]
            .into_iter()
            .map(|distance| operation.replace("DISTANCE", &distance.to_string()) + "\n")
            .collect();
        let expected = if name == "dot" { 12 } else { 1 };
        assert_eq!(scalar.matches(old).count(), expected);
        scalar = scalar.replace(old, &new);
    }

    let mut init = String::new();
    let mut pv = String::new();
    let mut store = String::new();
    for h in 0..12 {
        init.push_str(&format!("var accum{h}=0.0;\n"));
        pv.push_str(&format!("accum{h}=fma(sp[{h}u*64u+c],v,accum{h});\n"));
        store.push_str(&format!(
            "partial[t*202752u+(slot*12u+{h}u)*256u+tid]=accum{h};\nif !finite(accum{h}) {{atomicOr(&broken,1u);}}\n"
        ));
    }
    replace_once(
        &mut scalar,
        "var accum:array<f32,12>;for(var h=0u;h<12u;h++) {accum[h]=0.0;}",
        &init,
    );
    replace_once(
        &mut scalar,
        "for(var h=0u;h<12u;h++) {accum[h]=fma(sp[h*64u+c],v,accum[h]);}",
        &pv,
    );
    replace_once(
        &mut scalar,
        "for(var h=0u;h<12u;h++) {partial[t*202752u+(slot*12u+h)*256u+tid]=accum[h];if !finite(accum[h]) {atomicOr(&broken,1u);}}",
        &store,
    );
    scalar
}

#[cfg(test)]
mod tests {
    use super::super::Kernel;

    fn compiled(source: &str) -> (wgpu::naga::Module, Vec<u32>) {
        use wgpu::naga as n;
        let module = n::front::wgsl::parse_str(source)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(source)));
        let info = n::valid::Validator::new(
            n::valid::ValidationFlags::all(),
            n::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("actual prompt attention Naga validation");
        let options = n::back::spv::Options {
            lang_version: (1, 3),
            zero_initialize_workgroup_memory: n::back::spv::ZeroInitializeWorkgroupMemoryMode::None,
            bounds_check_policies: n::proc::BoundsCheckPolicies {
                index: n::proc::BoundsCheckPolicy::Unchecked,
                buffer: n::proc::BoundsCheckPolicy::Unchecked,
                image_load: n::proc::BoundsCheckPolicy::Unchecked,
                binding_array: n::proc::BoundsCheckPolicy::Unchecked,
            },
            ..Default::default()
        };
        let words = n::back::spv::write_vec(
            &module,
            &info,
            &options,
            Some(&n::back::spv::PipelineOptions {
                shader_stage: n::ShaderStage::Compute,
                entry_point: "main".into(),
            }),
        )
        .expect("actual prompt attention SPIR-V lowering");
        (module, words)
    }

    fn opcode_count(words: &[u32], opcode: u32) -> usize {
        let mut cursor = 5;
        let mut count = 0;
        while cursor < words.len() {
            let size = (words[cursor] >> 16) as usize;
            assert!(size != 0 && cursor + size <= words.len());
            count += usize::from(words[cursor] & 65535 == opcode);
            cursor += size;
        }
        count
    }

    fn function_array_count(module: &wgpu::naga::Module) -> usize {
        let arrays = |function: &wgpu::naga::Function| {
            function
                .local_variables
                .iter()
                .filter(|(_, local)| {
                    matches!(
                        module.types[local.ty].inner,
                        wgpu::naga::TypeInner::Array { .. }
                    )
                })
                .count()
        };
        arrays(&module.entry_points[0].function)
            + module
                .functions
                .iter()
                .map(|(_, f)| arrays(f))
                .sum::<usize>()
    }

    #[test]
    fn real_lowered_prompt_has_no_function_arrays_or_fixed_extent_loops() {
        for kernel in [Kernel::PromptAttention, Kernel::PromptAttentionFault] {
            let (old, old_spv) = compiled(&kernel.scalar_source());
            let (new, new_spv) = compiled(&kernel.source());
            assert_eq!(function_array_count(&old), 2);
            assert_eq!(function_array_count(&new), 0);
            // OpLoopMerge: cell/Q-softmax/PV loops remain dynamic, plus the
            // original status invalidation loop in the non-fault variant.
            assert_eq!(
                opcode_count(&new_spv, 246),
                if kernel == Kernel::PromptAttentionFault {
                    3
                } else {
                    4
                }
            );
            assert_eq!(
                opcode_count(&old_spv, 246),
                opcode_count(&new_spv, 246) + 10
            );
            // OpControlBarrier: original QK, softmax, PV and closing ordering.
            assert_eq!(opcode_count(&old_spv, 224), 4);
            assert_eq!(opcode_count(&new_spv, 224), 4);
        }
    }

    #[test]
    fn prompt_expansion_keeps_closing_contract_and_arithmetic_order() {
        for kernel in [Kernel::PromptAttention, Kernel::PromptAttentionFault] {
            let scalar = kernel.scalar_source();
            let expanded = kernel.source();
            assert_eq!(scalar.matches("workgroupBarrier();").count(), 4);
            assert_eq!(expanded.matches("workgroupBarrier();").count(), 4);
            assert_eq!(expanded.matches("let kword0=cache[").count(), 1);
            assert_eq!(expanded.matches("let kword1=cache[").count(), 1);
            assert_eq!(
                expanded
                    .matches("let kscales=unpack2x16float(cache[")
                    .count(),
                1
            );
            for old in ["var k:array", "var accum:array", "for(var j=", "for(var d="] {
                assert!(!expanded.contains(old));
            }
            assert!(expanded.contains("let query_row=t+p.reserved;let n=cells(query_row,p.start);"));
            assert!(expanded.contains("if token>=p.start+query_row+1u"));
            assert!(expanded.contains("let tok=selected[query_row*2051u+chunk*64u+tid]"));
            assert!(expanded.contains("if atomicLoad(&broken)!=0u"));
            let dot = "var dot=k0*sq[off];\ndot+=k1*sq[off+1u];\ndot+=k2*sq[off+2u];\ndot+=k3*sq[off+3u];\ndot+=k4*sq[off+4u];\ndot+=k5*sq[off+5u];\ndot+=k6*sq[off+6u];\ndot+=k7*sq[off+7u];";
            assert_eq!(expanded.matches(dot).count(), 12);
            for h in 0..12 {
                assert_eq!(
                    expanded
                        .matches(&format!("accum{h}=fma(sp[{h}u*64u+c],v,accum{h});"))
                        .count(),
                    1
                );
            }
            if kernel == Kernel::PromptAttentionFault {
                assert!(!expanded.contains("status["));
                assert!(expanded.contains("if tid==0u{atomicOr(&fault[0],1u);}"));
            } else {
                assert_eq!(expanded.matches("select(0u,1u,finite(score))").count(), 12);
                assert!(expanded.contains("for(var i=tid;i<here*12u;i+=256u)"));
            }
        }
    }

    #[test]
    fn other_qsa_paths_keep_original_generated_source() {
        for kernel in [
            Kernel::Attention,
            Kernel::MtpAttention,
            Kernel::PromptMerge,
            Kernel::PromptMergeFault,
            Kernel::Merge,
            Kernel::MtpMerge,
            Kernel::PromptScoreFault,
        ] {
            assert_eq!(kernel.source(), kernel.scalar_source());
        }
    }

    #[test]
    fn packed_eight_values_keep_signed_codes_and_original_scale_group() {
        // Every legal K head/lane window is 8-aligned and cannot cross a
        // 64-value group. Cover both packed halves, all signed byte patterns,
        // and all finite nonnegative FP16 scale representations.
        for offset in (0..512usize).step_by(8) {
            for j in 0..8 {
                assert_eq!((offset + j) / 64, offset / 64);
                assert_eq!((offset + j) / 128, offset / 128);
                for byte in 0..=255u32 {
                    let packed = byte << ((j & 3) * 8);
                    let code = ((packed >> ((j & 3) * 8)) << 24) as i32 >> 24;
                    assert_eq!(code, (byte as u8 as i8) as i32);
                }
            }
        }
        for bits in 0..0x7c00u16 {
            let scale = half::f16::from_bits(bits).to_f32();
            for code in [-128i32, -127, -1, 0, 1, 126, 127] {
                assert_eq!(
                    (code as f32 * scale).to_bits(),
                    ((code as i8) as f32 * scale).to_bits()
                );
            }
        }
    }
}
