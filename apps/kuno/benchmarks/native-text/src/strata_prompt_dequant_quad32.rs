//! Optional coalesced publication of the original record32 decoder.
//! Four physical subgroup lanes share ONE authentic encoded record and publish
//! its four unchanged adjacent vector8 outputs. No private/source-owner change.
use super::*;

pub(super) const VALUES_PER_INVOCATION: usize = 8;

/// Hardware subgroup IDs, never assumed global/local-invocation lane ordering.
/// Both partially filled final subgroups and all real subgroups execute every
/// shuffle; only complete, in-range quads may access the original weight.
pub(super) fn source(mut source: String, down: bool) -> String {
    assert!(source.contains("// STRATA_DEQUANT_RECORD32_ORIGINAL_STORE8"));
    let signature = "fn main(@builtin(global_invocation_id) id:vec3<u32>) {";
    assert_eq!(source.matches(signature).count(), 1);
    source = source.replace(signature,
        "fn main(@builtin(global_invocation_id) id:vec3<u32>,@builtin(workgroup_id) wg:vec3<u32>,@builtin(subgroup_id) subgroup:u32,@builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_size) width:u32,@builtin(num_subgroups) subgroup_count:u32) {");
    let old = "    let record=id.x;if record>=p.rows*p.columns/32u {return;}\n    let row=record/(p.columns/32u);let col=(record%(p.columns/32u))*32u;\n    write_record32(row,col,record*4u);";
    assert_eq!(source.matches(old).count(), 1);
    source = source.replace(
        old,
        r#"    // Uniform hardware guard precedes every original quantized source read.
    if width<4u||width>128u||width%4u!=0u||256u%width!=0u||subgroup_count*width!=256u {atomicOr(&fault[0],128u);return;}
    let record=wg.x*64u+subgroup*(width/4u)+lane/4u;
    let live=record<p.rows*p.columns/32u;
    let row=record/(p.columns/32u);let col=(record%(p.columns/32u))*32u;
    write_quad32(row,col,record*4u,lane%4u,lane,live);"#,
    );
    let mut helper = String::from(
        "\nfn write_quad32(row:u32,col:u32,first:u32,quarter:u32,lane:u32,live:bool) {\n",
    );
    if down {
        helper.push_str("var e=DownEncoded(vec4<u32>(0u),0.0);\nif live&&quarter==0u {e=gate_prepare(row*p.row_bytes,col/32u);}\n");
    } else {
        helper.push_str("var e=gate_Encoded(vec4<u32>(0u),0u,0u,0u,0.0);\nif live&&quarter==0u {\nif row%2u==0u {e=gate_prepare((row/2u)*p.row_bytes,col/32u);}\nelse {let u=up_prepare((row/2u)*p.row_bytes,col/32u);e=gate_Encoded(u.codes,u.signs,u.high,u.scale,u.d);}\n}\n");
    }
    // All shuffles are outside both the leader and per-role branches. Sharing
    // d through u32 preserves NaN payloads and every original FP32 scale bit.
    helper.push_str("let leader=lane-(lane%4u);\ne.codes=subgroupShuffle(e.codes,leader);\ne.d=bitcast<f32>(subgroupShuffle(bitcast<u32>(e.d),leader));\n");
    if !down {
        helper.push_str("e.signs=subgroupShuffle(e.signs,leader);\ne.high=subgroupShuffle(e.high,leader);\ne.scale=subgroupShuffle(e.scale,leader);\n");
    }
    helper.push_str("if !live {return;}\nlet word=quarter*2u;\n");
    if down {
        helper.push_str("store_original8(first+quarter,decoded8(e.d,gate_packed(e,word),gate_packed(e,word+1u)));\n");
    } else {
        helper.push_str("if row%2u==0u {\nstore_original8(first+quarter,decoded8(gate_factor(e,quarter/2u),gate_packed(e,word),gate_packed(e,word+1u)));\n} else {\nlet u=up_Encoded(e.codes,e.signs,e.high,e.scale,e.d);\nstore_original8(first+quarter,decoded8(up_factor(u,quarter/2u),up_packed(u,word),up_packed(u,word+1u)));\n}\n");
    }
    helper.push_str("}\n// STRATA_DEQUANT_QUAD32_ORIGINAL_STORE8\n");
    let insert = source.find("\nfn store_original8(").unwrap();
    source.insert_str(insert, &helper);
    source
}

pub(super) fn hardware_supported(minimum: u32, maximum: u32) -> bool {
    minimum >= 4
        && maximum <= 128
        && minimum <= maximum
        && minimum.is_power_of_two()
        && maximum.is_power_of_two()
}

pub(super) fn check_hardware(client: &ComputeClient<WgpuRuntime>) -> Result<()> {
    let hardware = &client.properties().hardware;
    // strata-intel-core requires and actually enables the SUBGROUP feature on
    // the device. Geometry is read from this real client's queried properties.
    if !crate::strata_intel_core::enabled()
        || !hardware_supported(hardware.plane_size_min, hardware.plane_size_max)
        || hardware.max_units_per_cube < 256
    {
        return Err(crate::strata_prompt::invalid_buffer_length());
    }
    Ok(())
}

fn validate_one(source: &str, label: &str) -> std::result::Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;
    use wgpu::naga as n;
    let module = n::front::wgsl::parse_str(source)?;
    let _info = n::valid::Validator::new(
        n::valid::ValidationFlags::all(),
        n::valid::Capabilities::all(),
    )
    .validate(&module)?;
    // Same physical-source lowering as BOTH real native shader factories.
    // The original readonly source descriptors and pointer authority survive.
    let words = cubecl::wgpu::native_expert_shader_words(source)?;
    // No shared-memory/barrier or float arithmetic collective is introduced.
    let mut shuffle = 0;
    let mut at = 5;
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        if count == 0 || at + count > words.len() {
            return Err("quad32 SPIRV opcode extent".into());
        }
        match words[at] & 65535 {
            345 => shuffle += 1,
            224 | 225 | 346 | 349 | 350 | 351 | 352 | 353 | 354 | 355 => {
                return Err("quad32 unexpected barrier or collective".into());
            }
            _ => {}
        }
        at += count;
    }
    let expected = if source.contains("const DOWN_IMAGE:bool=true;") {
        2
    } else {
        5
    };
    if shuffle != expected {
        return Err(
            format!("quad32 original encoded field shuffles {shuffle} expected{expected}").into(),
        );
    }
    let mut validator = std::process::Command::new("spirv-val")
        .args(["--target-env", "vulkan1.3", "-"])
        .stdin(std::process::Stdio::piped())
        .spawn()?;
    validator
        .stdin
        .take()
        .ok_or("quad32 validator stdin")?
        .write_all(
            &words
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
    if !validator.wait()?.success() {
        return Err("quad32 SPIRV validation".into());
    }
    println!(
        "strata_dequant_quad32_shader_accepted layout={label} encoded_shuffles={shuffle} values_per_invocation=8"
    );
    Ok(())
}

/// Separate from and additional to the unchanged original shader population.
pub(crate) fn validate_sources() -> std::result::Result<(), Box<dyn std::error::Error>> {
    for op in super::dequant_operators() {
        let label = format!("{op:?}");
        validate_one(&op.selected_source(), &label)?;
    }
    #[cfg(feature = "strata-expert-cohorts")]
    for (down, types) in [
        (false, &[q::IQ2_S, q::IQ3_S, q::IQ3_XXS, q::IQ4_XS][..]),
        (true, &[q::IQ4_NL, q::Q2_0][..]),
    ] {
        for &dtype in types {
            let op = cohorts::CohortOp::Image { dtype, down };
            validate_one(&op.selected_source(), &format!("cohort-{dtype}-{down}"))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn quad32_physical_lane_mapping_is_total_unique_and_keeps_contiguous_original_vectors() {
        for width in [4usize, 8, 16, 32, 64, 128] {
            for (rows, columns) in [
                (1280usize, 2560usize),
                (2560, 640),
                (1, 32),
                (5, 32),
                (63, 32),
                (65, 32),
                (127, 32),
                (129, 32),
            ] {
                let records = rows * columns / 32;
                let vectors = records * 4;
                let mut seen = vec![0u8; vectors];
                let mut prepared = vec![0u8; records];
                for wg in 0..vectors.div_ceil(256) {
                    for subgroup in 0..256 / width {
                        for lane in 0..width {
                            let record = wg * 64 + subgroup * (width / 4) + lane / 4;
                            if record >= records {
                                continue;
                            }
                            let quarter = lane % 4;
                            let row = record / (columns / 32);
                            let col = (record % (columns / 32)) * 32;
                            let i = record * 4 + quarter;
                            assert_eq!(i, row * (columns / 8) + (col + quarter * 8) / 8);
                            assert_eq!(quarter / 2, (col + quarter * 8) % 32 / 16);
                            seen[i] += 1;
                            prepared[record] += u8::from(quarter == 0);
                            assert_eq!((lane - lane % 4) / 4, lane / 4);
                        }
                    }
                }
                assert!(seen.iter().all(|&n| n == 1));
                assert!(prepared.iter().all(|&n| n == 1));
            }
        }
        for (minimum, maximum) in [(0, 32), (2, 32), (8, 256), (32, 16), (6, 32), (8, 48)] {
            assert!(!hardware_supported(minimum, maximum));
        }
        assert!(hardware_supported(16, 32));
    }
    #[test]
    fn quad32_selected_sources_keep_original_guards_stores_scales_and_real_builtin_mapping() {
        for op in super::super::dequant_operators() {
            let old = op.source();
            let down = matches!(
                op,
                Op::Dequant { down: true, .. }
                    | Op::ResidentDequant { down: true, .. }
                    | Op::SlotDequant { down: true, .. }
            );
            let old = if old.contains("// STRATA_DEQUANT_RECORD32_ORIGINAL_STORE8") {
                old
            } else {
                super::super::record32::source(old, down)
            };
            let selected = if cfg!(feature = "strata-dequant-quad32") {
                op.selected_source()
            } else {
                source(old.clone(), down)
            };
            let module = wgpu::naga::front::wgsl::parse_str(&selected).unwrap();
            wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
            let oldstore = old
                .split_once("fn store_original8(")
                .unwrap()
                .1
                .split_once("\n}\nfn write_record32")
                .unwrap()
                .0;
            assert_eq!(
                selected
                    .split_once("fn store_original8(")
                    .unwrap()
                    .1
                    .split_once("\n}\nfn write_record32")
                    .unwrap()
                    .0,
                oldstore
            );
            let helper = selected
                .split_once("fn write_quad32(")
                .unwrap()
                .1
                .split_once("// STRATA_DEQUANT_QUAD32")
                .unwrap()
                .0;
            assert_eq!(
                helper.matches("subgroupShuffle(").count(),
                if down { 2 } else { 5 }
            );
            assert_eq!(helper.matches("gate_prepare(").count(), 1);
            assert_eq!(helper.matches("up_prepare(").count(), usize::from(!down));
            assert!(
                helper.find("if !live {return;}").unwrap()
                    > helper.rfind("subgroupShuffle(").unwrap()
            );
            assert!(selected.contains("@builtin(subgroup_id) subgroup:u32"));
            assert!(selected.contains("wg.x*64u+subgroup*(width/4u)+lane/4u"));
            assert!(selected.contains("arrayLength(&addresses)!=2048u"));
            assert!(selected.contains("arrayLength(&raw)!=p.r5"));
            assert!(selected.contains("src.z!=bytes/4u||src.z!=expected||src.w>1u"));
            assert!(selected.contains("p.columns%32u!=0u"));
            if !down {
                assert!(helper.contains("quarter/2u"));
            }
        }
        assert_eq!(
            Op::selected_dequant_values(),
            if cfg!(feature = "strata-dequant-quad32") {
                VALUES_PER_INVOCATION
            } else {
                Op::dequant_values()
            }
        );
    }
}
