//! Prompt-only scalar state expansion for the selected CB32 GDN recurrence.
//!
//! MIT Strata fb58e0 prefill/kernels.dp.cpp: gdn_rec_cols_pipe_kernel. The
//! original explicitly unrolls all four fixed RPG32 loops. Keep each row's
//! FMA/finite/atomic operations and the entire causal time loop in order.

fn replace_once(source: &mut String, old: &str, new: &str) {
    assert_eq!(
        source.matches(old).count(),
        1,
        "prompt GDN specialization source drift"
    );
    *source = source.replacen(old, new, 1);
}

pub(super) fn source(mut scalar: String) -> String {
    assert!(scalar.contains("let head=g.x/4u;let cb=g.x%4u;let rg=tid/32u;"));
    assert!(scalar.contains("for(var t=0u;t<p.count;t++)"));
    let mut initialize = String::new();
    let mut kv = String::new();
    let mut update = String::new();
    let mut writeback = String::new();
    for r in 0..32 {
        initialize.push_str(&format!(
            "var state{r}=old[((rg*32u+{r}u)*p.value_heads+head)*128u+col];\nif !finite(state{r}) {{atomicOr(&broken,1u);}}\n"
        ));
        kv.push_str(&format!("kv=fma(state{r},sk[rg*32u+{r}u],kv);\n"));
        update.push_str(&format!(
            "state{r}=fma(decay,state{r},sk[rg*32u+{r}u]*delta);\no=fma(state{r},sq[rg*32u+{r}u],o);\nif !finite(state{r}) {{atomicOr(&broken,1u);}}\n"
        ));
        writeback.push_str(&format!(
            "next[((rg*32u+{r}u)*p.value_heads+head)*128u+col]=select(0.0,state{r},good);\n"
        ));
    }
    replace_once(&mut scalar, "var state:array<f32,32>;", "");
    replace_once(
        &mut scalar,
        "for(var r=0u;r<32u;r++) {state[r]=old[((rg*32u+r)*p.value_heads+head)*128u+col];if !finite(state[r]) {atomicOr(&broken,1u);}}",
        &initialize,
    );
    replace_once(
        &mut scalar,
        "for(var r=0u;r<32u;r++) {kv=fma(state[r],sk[rg*32u+r],kv);}",
        &kv,
    );
    replace_once(
        &mut scalar,
        "for(var r=0u;r<32u;r++) {state[r]=fma(decay,state[r],sk[rg*32u+r]*delta);o=fma(state[r],sq[rg*32u+r],o);if !finite(state[r]) {atomicOr(&broken,1u);}}",
        &update,
    );
    replace_once(
        &mut scalar,
        "for(var r=0u;r<32u;r++) {next[((rg*32u+r)*p.value_heads+head)*128u+col]=select(0.0,state[r],good);}",
        &writeback,
    );
    scalar
}

#[cfg(test)]
mod tests {
    use super::super::Op;

    fn compiled(source: &str) -> (wgpu::naga::Module, Vec<u32>) {
        use wgpu::naga as n;
        let module = n::front::wgsl::parse_str(source)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(source)));
        let info = n::valid::Validator::new(
            n::valid::ValidationFlags::all(),
            n::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("actual prompt GDN Naga validation");
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
        .expect("actual prompt GDN SPIR-V lowering");
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
    fn real_lowered_prompt_keeps_only_causal_time_loop() {
        for op in [Op::Recurrence, Op::RecurrenceFault] {
            let (old, old_spv) = compiled(&op.scalar_source());
            let (new, new_spv) = compiled(&op.source());
            assert_eq!(function_array_count(&old), 1);
            assert_eq!(function_array_count(&new), 0);
            assert_eq!(opcode_count(&old_spv, 246), 5);
            assert_eq!(opcode_count(&new_spv, 246), 1);
            assert_eq!(opcode_count(&old_spv, 224), 6);
            assert_eq!(opcode_count(&new_spv, 224), 6);
        }
    }

    #[test]
    fn expansion_preserves_ordered_rows_and_closing_faults() {
        for op in [Op::Recurrence, Op::RecurrenceFault] {
            let expanded = op.source();
            assert!(!expanded.contains("state[r]"));
            assert!(!expanded.contains("for(var r="));
            assert_eq!(expanded.matches("for(var t=0u;t<p.count;t++)").count(), 1);
            assert_eq!(expanded.matches("workgroupBarrier();").count(), 6);
            let time = expanded.find("for(var t=0u;t<p.count;t++)").unwrap();
            let prefetch = expanded.find("if t+1u<p.count").unwrap();
            let mut previous = prefetch;
            for r in 0..32 {
                let initialize = expanded.find(&format!("var state{r}=old[((rg*32u+{r}u)*p.value_heads+head)*128u+col];\nif !finite(state{r}) {{atomicOr(&broken,1u);}}")).unwrap();
                assert!(initialize < time);
                let dot = expanded
                    .find(&format!("kv=fma(state{r},sk[rg*32u+{r}u],kv);"))
                    .unwrap();
                assert!(dot > previous);
                previous = dot;
            }
            let total = expanded.find("let total=red[c]+red[32u+c]+red[64u+c]+red[96u+c];let delta=(cv-decay*total)*cbt;var o=0.0;").unwrap();
            assert!(total > previous);
            previous = total;
            for r in 0..32 {
                let update = expanded.find(&format!("state{r}=fma(decay,state{r},sk[rg*32u+{r}u]*delta);\no=fma(state{r},sq[rg*32u+{r}u],o);\nif !finite(state{r}) {{atomicOr(&broken,1u);}}")).unwrap();
                assert!(update > previous);
                previous = update;
            }
            let close = expanded.find("let good=atomicLoad(&broken)==0u;").unwrap();
            assert!(close > previous);
            previous = close;
            for r in 0..32 {
                let write = expanded.find(&format!("next[((rg*32u+{r}u)*p.value_heads+head)*128u+col]=select(0.0,state{r},good);")).unwrap();
                assert!(write > previous);
                previous = write;
            }
            if op == Op::RecurrenceFault {
                assert!(!expanded.contains("status["));
                assert!(expanded.contains("if !good{atomicOr(&fault[0],1u);}"));
            } else {
                assert!(expanded.contains("status[at]=select(0u,1u,good);"));
            }
        }
    }

    #[test]
    fn other_prompt_gdn_stages_keep_original_source() {
        for op in [
            Op::Conv,
            Op::ConvFault,
            Op::History,
            Op::L2,
            Op::L2Fault,
            Op::Split,
            Op::Norm,
            Op::NormFault,
            Op::Retain,
            Op::Compatibility,
        ] {
            assert_eq!(op.source(), op.scalar_source());
        }
    }
}
