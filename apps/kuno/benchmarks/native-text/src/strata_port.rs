//! Experimental Rust/Vulkan port of Strata's Qwen3.8 native execution policy.
//! Source: MIT Strata fb58e0dbc8399662c0e47c76578c6e878b14f6cf.
//! See strata-port-NOTICE.txt. No C++/SYCL inference library is linked.
use cubecl::prelude::CubeDim;
use std::sync::atomic::{AtomicU64, Ordering};

pub(crate) const fn enabled() -> bool {
    cfg!(feature = "strata-rust-port")
}
static GU: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
static DOWN: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
pub(crate) fn credit(down: bool, count: usize) {
    if enabled() && (1..=4).contains(&count) {
        (if down { &DOWN } else { &GU })[count - 1].fetch_add(1, Ordering::Relaxed);
    }
}
static LOOKUP_WINDOWS: AtomicU64 = AtomicU64::new(0);
static LOOKUP_PROPOSED: AtomicU64 = AtomicU64::new(0);
/// Port of Strata's in-request suffix drafter. Suggestions are always checked
/// by the actual target; no token oracle or persisted history is consulted.
pub(crate) fn lookup(
    prompt: &[u32],
    emitted: &[u32],
    maximum: usize,
) -> Option<zeroize::Zeroizing<Vec<u32>>> {
    if !enabled() || maximum == 0 {
        return None;
    }
    let mut history = zeroize::Zeroizing::new(Vec::with_capacity(prompt.len() + emitted.len()));
    history.extend_from_slice(prompt);
    history.extend_from_slice(emitted);
    for n in (2..=5.min(history.len())).rev() {
        let suffix = history.len() - n;
        if suffix < maximum {
            continue;
        }
        for at in (0..=suffix.saturating_sub(n + maximum)).rev() {
            if history[at..at + n] == history[suffix..] && at + n + maximum <= suffix {
                LOOKUP_WINDOWS.fetch_add(1, Ordering::Relaxed);
                LOOKUP_PROPOSED.fetch_add(maximum as u64, Ordering::Relaxed);
                return Some(zeroize::Zeroizing::new(
                    history[at + n..at + n + maximum].to_vec(),
                ));
            }
        }
    }
    None
}
pub(crate) fn public_preview(text: &str) {
    // Only the fixed public fixtures reach this lab; custom runner displays
    // this line live and omits it from the saved numeric diagnostic log.
    if enabled() {
        println!(
            "strata_public_preview={}",
            serde_json::to_string(text).unwrap()
        );
    }
}
pub(crate) fn receipt() -> serde_json::Value {
    serde_json::json!({"enabled": enabled(), "upstream_revision": "fb58e0dbc8399662c0e47c76578c6e878b14f6cf",
        "backend": "Rust/WGSL/Vulkan", "activation_planes": 1, "activation_scale": "fp16-q8_1",
        "projection_workgroup":256, "rows_per_workgroup":32, "group_stride":4,"suffix_lookup_windows":LOOKUP_WINDOWS.load(Ordering::Relaxed),"suffix_lookup_proposed":LOOKUP_PROPOSED.load(Ordering::Relaxed),
        "gu_dispatches_by_member_count":GU.each_ref().map(|v|v.load(Ordering::Relaxed)),
        "down_dispatches_by_member_count":DOWN.each_ref().map(|v|v.load(Ordering::Relaxed)),
        "original_cpu13_precision_contract": false, "installed_or_production": false})
}

fn replace(source: String, old: &str, new: &str) -> String {
    assert!(source.contains(old), "Strata port template drift: {old}");
    source.replace(old, new)
}
fn function_end(source: &str, first: usize) -> usize {
    let brace = first + source[first..].find('{').expect("function body");
    let mut depth = 0;
    for (i, b) in source.as_bytes().iter().enumerate().skip(brace) {
        match b {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return i + 1;
                }
            }
            _ => {}
        }
    }
    panic!("unclosed static shader function")
}

/// Single Q8_1 plane. Codes use the unrounded F32 scale; the dot's scale is
/// rounded to F16, exactly the representation used by Strata's native IQ path.
fn quantizer(mut source: String) -> String {
    let first = source
        .find("fn prepare_q8(block:u32)")
        .expect("quantizer entry");
    let end = function_end(&source, first);
    let original = &source[first..end];
    let bound = original
        .split_once("block>=")
        .unwrap()
        .1
        .split_once(" {return;}")
        .unwrap()
        .0;
    let body = format!(
        r#"fn prepare_q8(block:u32) {{
    if block>={bound} {{return;}}
    var amax=0.0;var valid=true;
    for(var i=0u;i<32u;i++) {{let x=cached_input[block*32u+i];valid=valid && abs(x)<3.4028234663852886e38;amax=max(amax,abs(x));}}
    if !valid {{clear_q8(block);return;}}
    let d=amax/127.0;
    scale_first[block]=unpack2x16float(pack2x16float(vec2<f32>(d,0.0))).x;
    scale_second[block]=0.0;
    for(var word=0u;word<8u;word++) {{
        var packed=0u;
        for(var byte=0u;byte<4u;byte++) {{let q=q8_code(cached_input[block*32u+word*4u+byte],d);packed|=(bitcast<u32>(q)&255u)<<(byte*8u);}}
        packed_first[block*8u+word]=packed;packed_second[block*8u+word]=0u;
    }}
}}"#
    );
    source.replace_range(first..end, &body);
    source
}

fn rows(source: String, down: bool) -> String {
    let source = replace(
        source,
        "@compute @workgroup_size(64)",
        "@compute @workgroup_size(256)",
    );
    let source = replace(
        source,
        "fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) local:u32) {",
        "fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) local:u32,@builtin(subgroup_id) sg:u32,@builtin(subgroup_invocation_id) sub_lane:u32,@builtin(subgroup_size) width:u32,@builtin(num_subgroups) subgroups:u32) {",
    );
    let source = replace(
        source,
        "if group.x>=",
        "if (width!=16u && width!=32u) || subgroups!=256u/width || group.x>=",
    );
    let source = replace(
        source,
        "let row=group.x*8u+local/8u;let lane=local%8u;",
        "let row=group.x*32u+sg*(width/8u)+sub_lane/8u;let lane=sub_lane%8u;",
    );
    if down {
        replace(source, "group.x>=320u", "group.x>=80u")
    } else {
        replace(source, "group.x>=80u", "group.x>=20u")
    }
}

fn reduction(mut source: String, down: bool, members: usize) -> String {
    let vec = format!("vec{members}<f32>");
    let (first, last) = if down {
        let first = source
            .find("    // Explicit initialization includes every unused RHS column.")
            .unwrap();
        let last = first + source[first..].find("    if lane==0u").unwrap();
        (first, last)
    } else {
        let marker = format!(
            "for(var j=0u;j<{members}u;j++) {{gate_sums[local*{members}u+j]=gates[j];up_sums[local*{members}u+j]=ups[j];}}"
        );
        let at = source.find(&marker).expect("GU tree");
        let first = source[..at].rfind('\n').unwrap() + 1;
        let last = at + source[at..].find("if lane==0u").unwrap();
        (first, last)
    };
    let tree = if down {
        format!(
            "    for(var stride=4u;stride>0u;stride/=2u) {{let other:{vec}=subgroupShuffleDown(values,stride);if lane<stride {{values+=other;}}}}\n"
        )
    } else {
        format!(
            "    for(var stride=4u;stride>0u;stride/=2u) {{let g:{vec}=subgroupShuffleDown(gates,stride);let u:{vec}=subgroupShuffleDown(ups,stride);if lane<stride {{gates+=g;ups+=u;}}}}\n    "
        )
    };
    source.replace_range(first..last, &tree);
    for (i, axis) in ["x", "y", "z", "w"].iter().take(members).enumerate() {
        for (old, new) in if down {
            vec![("sums", "values")]
        } else {
            vec![("gate_sums", "gates"), ("up_sums", "ups")]
        } {
            source = source.replace(
                &format!("{old}[local*{members}u+{i}u]"),
                &format!("{new}.{axis}"),
            );
        }
    }
    for name in if down {
        vec!["sums"]
    } else {
        vec!["gate_sums", "up_sums"]
    } {
        source = replace(
            source,
            &format!("var<workgroup> {name}:array<f32,{}>;\n", members * 64),
            "",
        );
    }
    if !down {
        source=source.replace("            // All lanes finish reading their reduction before the next group\n            // can overwrite any shared cell, including unused member cells.\n            workgroupBarrier();","");
    }
    source
}

fn gate_up(mut source: String, members: usize) -> String {
    source = rows(source, false);
    source = reduction(source, false, members);
    let words = members * 720;
    source = source.replacen(
        "@compute @workgroup_size(256)",
        &format!("var<workgroup> strata_input:array<u32,{words}>;\n@compute @workgroup_size(256)"),
        1,
    );
    let first = source
        .find("    workgroupBarrier();")
        .expect("immutable codebook barrier");
    let copy = format!(
        r#"    for(var word=local;word<{words}u;word+=256u) {{
        let token=word/720u;let at=word%720u;var value=0u;
        if token<p.h {{let frame=64u+token*1440u;if at<640u {{value=planes[frame+at];}}else {{value=planes[frame+1280u+at-640u];}}}}
        strata_input[word]=value;
    }}
"#
    );
    source.insert_str(first, &copy);
    for axis in ["x", "y", "z", "w"].iter().take(members) {
        let old = format!("frames.{axis}=64u+(");
        if let Some(first) = source.find(&old) {
            let end = first + source[first..].find(';').unwrap() + 1;
            let line = source[first..end].to_owned();
            source.replace_range(
                first..end,
                &line.replace("=64u+(", "=(").replace("*1440u", "*720u"),
            );
        }
        let dots = format!(
            "{{let first=planes[frames.{axis}+at];let second=planes[frames.{axis}+640u+at];gf.{axis}+=dot4I8Packed(gq,first);gs.{axis}+=dot4I8Packed(gq,second);uf.{axis}+=dot4I8Packed(uq,first);us.{axis}+=dot4I8Packed(uq,second);}}"
        );
        let primary = format!(
            "{{let first=strata_input[frames.{axis}+at];gf.{axis}+=dot4I8Packed(gq,first);uf.{axis}+=dot4I8Packed(uq,first);}}"
        );
        source = replace(source, &dots, &primary);
        let scales = format!(
            "{{let a=bitcast<f32>(planes[frames.{axis}+1280u+block]);let b=bitcast<f32>(planes[frames.{axis}+1360u+block]);let gp=a*f32(gf.{axis})+b*f32(gs.{axis});let up=a*f32(uf.{axis})+b*f32(us.{axis});gates.{axis}+=gd*gp;ups.{axis}+=ud*up;}}"
        );
        let primary = format!(
            "{{let a=bitcast<f32>(strata_input[frames.{axis}+640u+block]);gates.{axis}=fma(gd,a*f32(gf.{axis}),gates.{axis});ups.{axis}=fma(ud,a*f32(uf.{axis}),ups.{axis});}}"
        );
        source = replace(source, &scales, &primary);
    }
    source
}

fn down(mut source: String, members: usize) -> String {
    source = rows(source, true);
    source = reduction(source, true, members);
    let words = members * 180;
    source = source.replacen(
        "@compute @workgroup_size(256)",
        &format!("var<workgroup> strata_input:array<u32,{words}>;\n@compute @workgroup_size(256)"),
        1,
    );
    let copy = format!(
        r#"    for(var word=local;word<{words}u;word+=256u) {{
        let member=word/180u;let at=word%180u;var value=0u;
        if member<length {{let frame=frames[member];if at<160u {{value=planes[frame+at];}}else {{value=planes[frame+320u+at-160u];}}}}
        strata_input[word]=value;
    }}
    workgroupBarrier();
    frames=vec{members}<u32>({});
"#,
        (0..members)
            .map(|m| format!("{}u", m * 180))
            .collect::<Vec<_>>()
            .join(",")
    );
    let before = "    var values=";
    let at = source.find(before).unwrap();
    source.insert_str(at, &copy);
    for axis in ["x", "y", "z", "w"].iter().take(members) {
        let old = format!(
            "{{first.{axis}+=dot4I8Packed(q,planes[frames.{axis}+at]);second.{axis}+=dot4I8Packed(q,planes[frames.{axis}+160u+at]);}}"
        );
        let new = format!("{{first.{axis}+=dot4I8Packed(q,strata_input[frames.{axis}+at]);}}");
        source = replace(source, &old, &new);
        let old = format!(
            "{{let a=bitcast<f32>(planes[frames.{axis}+320u+block]);let b=bitcast<f32>(planes[frames.{axis}+340u+block]);let pair=a*f32(first.{axis})+b*f32(second.{axis});values.{axis}+=d*pair;}}"
        );
        let new = format!(
            "{{let a=bitcast<f32>(strata_input[frames.{axis}+160u+block]);values.{axis}=fma(d,a*f32(first.{axis}),values.{axis});}}"
        );
        source = replace(source, &old, &new);
    }
    source
}

pub(crate) fn shader(mut source: String, down_projection: bool, members: usize) -> String {
    if !enabled() {
        return source;
    }
    if source.contains("fn prepare_q8(block:u32)") {
        source = quantizer(source);
    }
    // Original record ownership/erasure is retained, with the actual new grid.
    source = source
        .replace("+2u]=80u;", "+2u]=20u;")
        .replace("+5u]=320u;", "+5u]=80u;")
        .replace("planes[base+2u]!=80u", "planes[base+2u]!=20u")
        .replace("planes[base+5u]!=320u", "planes[base+5u]!=80u");
    if source.contains("requires packed_4x8_integer_dot_product")
        && source.contains("let row=group.x*8u+local/8u;")
    {
        if down_projection {
            down(source, members)
        } else {
            gate_up(source, members)
        }
    } else {
        source
    }
}
pub(crate) fn dimension(original: CubeDim, project: bool) -> CubeDim {
    if enabled() && project {
        CubeDim::new_1d(256)
    } else {
        original
    }
}
pub(crate) fn validate_limits(limits: &wgpu::Limits) -> kuno_text::Result<()> {
    if enabled()
        && (limits.max_compute_invocations_per_workgroup < 256
            || limits.max_compute_workgroup_size_x < 256
            || limits.max_compute_workgroup_storage_size < 19712)
    {
        return Err(kuno_text::Error::InvalidFormat(
            "Strata Rust workgroup limits",
        ));
    }
    Ok(())
}

/// Final full-set WGSL validation and SPIR-V lowering, after the one-shot port.
pub(crate) fn validate_shader_set() -> Result<(), Box<dyn std::error::Error>> {
    if !enabled() {
        return Err("Strata port feature required".into());
    }
    use wgpu::naga as n;
    let mut shaders = crate::quant_matvec::frozen_gu_integer::strata_compiled_sources();
    shaders.extend(crate::quant_matvec::frozen_down_integer::strata_compiled_sources());
    for (index, compiled) in shaders.iter().enumerate() {
        let module = n::front::wgsl::parse_str(&compiled.source)
            .map_err(|e| e.emit_to_string(&compiled.source))?;
        let info = n::valid::Validator::new(
            n::valid::ValidationFlags::all(),
            n::valid::Capabilities::all(),
        )
        .validate(&module)
        .map_err(|e| e.emit_to_string(&compiled.source))?;
        if compiled.cube_dim != CubeDim::new_1d(module.entry_points[0].workgroup_size[0]) {
            return Err("port metadata/WGSL dimension disagreement".into());
        }
        n::back::spv::write_vec(
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
        )?;
        println!(
            "strata_shader_accepted index={index} workgroup={}",
            module.entry_points[0].workgroup_size[0]
        );
    }
    println!("strata_shader_set_complete shaders={}", shaders.len());
    Ok(())
}
