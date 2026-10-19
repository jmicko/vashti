//! MIT Strata fb58e0 Intel native_gu/native_down row_entries port.
//! Gate/up are distinct projections; GPU groups execute 4/2/1 entry chunks.
//! The existing PART owner is borrowed before DOWN, then erased by activation.
//! No route readback, new private root, or model copy is introduced.
use std::fmt::Write;

pub(super) fn source(
    mut source: String,
    down: bool,
    dtype: u32,
    width: usize,
    capacity: usize,
    role: u8,
    direct_planes: bool,
) -> String {
    assert!(matches!(width, 1 | 2 | 4) && width <= capacity);
    assert!(!direct_planes || capacity == 8);
    if direct_planes {
        // Selected global row_entries reads original quantized activations,
        // like native Multi::acts. No per-workgroup full-window copy is needed.
        let words = capacity * if down { 180 } else { 720 };
        let declaration = format!("var<workgroup> strata_input:array<u32,{words}>;\n");
        assert!(source.contains(&declaration));
        source = source.replace(&declaration, "");
    }
    assert!(if down {
        role == 3
    } else {
        matches!(role, 1 | 2)
    });
    if !down {
        // Two 640-value role planes share each dead 2560-value PART row.
        source = source
            .replace("p.e<640u", "p.e<1280u")
            .replace(
                "640u>arrayLength(&output)-p.d",
                "1280u>arrayLength(&output)-p.d",
            )
            .replace(
                "arrayLength(&output)-p.d-640u",
                "arrayLength(&output)-p.d-1280u",
            )
            .replace(
                "80u>arrayLength(&status)-p.f",
                "160u>arrayLength(&status)-p.f",
            );
    }
    let name = if down {
        "down"
    } else if role == 1 {
        "gate"
    } else {
        "up"
    };
    let rows = if down { 2560 } else { 640 };
    let blocks = if down { 20 } else { 80 };
    let lanes = crate::strata_port::expert_lanes();
    let tile = 256 / lanes;
    let float_ty = if width == 1 {
        "f32".into()
    } else {
        format!("vec{width}<f32>")
    };
    let int_ty = if width == 1 {
        "i32".into()
    } else {
        format!("vec{width}<i32>")
    };
    let axis = |i: usize| {
        if width == 1 {
            String::new()
        } else {
            format!(".{}", ["x", "y", "z", "w"][i])
        }
    };
    let mut body = format!(
        r#"@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group:vec3<u32>, @builtin(local_invocation_index) local:u32,
        @builtin(subgroup_id) sg:u32, @builtin(subgroup_invocation_id) sub_lane:u32,
        @builtin(subgroup_size) subgroup_width:u32, @builtin(num_subgroups) subgroups:u32) {{
    if subgroup_width!=32u || subgroups!=8u || group.x>={grid}u || group.z!=0u || !valid_group(group.y) {{return;}}
    let length=map[p.a+84u+group.y];
"#,
        grid = rows / tile
    );
    if capacity == 8 {
        body = body.replace(" || group.z!=0u", "");
        body.push_str(match width {
            4 => "    let member_base=group.z*4u;if member_base+4u>length {return;}\n",
            2 => "    let member_base=(length/4u)*4u;if group.z!=0u || length%4u<2u {return;}\n",
            1 => "    let member_base=(length/2u)*2u;if group.z!=0u || length%2u==0u {return;}\n",
            _ => unreachable!(),
        });
    } else {
        body.push_str(match width {
        4 => "    if length!=4u {return;}let member_base=0u;\n",
        2 => "    if length!=2u && length!=3u {return;}let member_base=0u;\n",
        1 => "    if length!=1u && length!=3u {return;}let member_base=select(0u,2u,length==3u);\n",
        _ => unreachable!(),
    });
    }
    writeln!(body, "    let row=group.x*{tile}u+sg*(32u/{lanes}u)+sub_lane/{lanes}u;let lane=sub_lane%{lanes}u;").unwrap();
    writeln!(
        body,
        "    let slot=map[p.a+4u+group.y]-strata_first;let offset=(slot*{rows}u+row)*ROW_BYTES;"
    )
    .unwrap();
    for i in 0..width {
        writeln!(
            body,
            "    let route{i}=map[p.a+164u+group.y*8u+member_base+{i}u];"
        )
        .unwrap();
    }
    if !direct_planes && down {
        // Copy just this chunk's route-indexed activation into shared memory.
        // All lanes in the workgroup agree on admission before the barrier.
        for i in 0..width {
            writeln!(body, "    for(var at=local;at<180u;at+=256u) {{let frame={header}u+route{i}*360u;var value=0u;if at<160u {{value=planes[frame+at];}}else {{value=planes[frame+320u+at-160u];}}strata_input[{}u+at]=value;}}", i*180, header=if capacity == 8 {128} else {64}).unwrap();
        }
    } else if !direct_planes {
        writeln!(body, "    for(var word=local;word<{}u;word+=256u) {{let token=word/720u;let at=word%720u;var value=0u;if token<p.h {{let frame=64u+token*1440u;if at<640u {{value=planes[frame+at];}}else {{value=planes[frame+1280u+at-640u];}}}}strata_input[word]=value;}}", capacity*720).unwrap();
    }
    if !direct_planes {
        body.push_str("    workgroupBarrier();\n");
    }
    for i in 0..width {
        writeln!(
            body,
            "    let frame{i}={};",
            if direct_planes && down {
                format!("128u+route{i}*360u")
            } else if direct_planes {
                format!("64u+(route{i}/10u)*1440u")
            } else if down {
                format!("{}u", i * 180)
            } else {
                format!("(route{i}/10u)*720u")
            }
        )
        .unwrap();
    }
    writeln!(body, "    var values={float_ty}(0.0);\n    for(var block=lane;block<{blocks}u;block+={lanes}u) {{let prepared={name}_prepare(offset,block);").unwrap();
    if capacity == 8 {
        // Native vec_dot/Multi finish keeps scale weighting and IQ2/IQ3
        // truncation in SIGNED integer arithmetic before converting to F32.
        // IQ4_NL visits two16-value calls per32-value block, matching Fmt<20>.
        if down && dtype == crate::quant_matvec::IQ4_NL {
            let start = body.rfind("    for(var block=lane;").unwrap();
            body.truncate(start);
            writeln!(body,"    for(var call=lane;call<40u;call+={lanes}u) {{let block=call/2u;let part=call%2u;let prepared={name}_prepare(offset,block);").unwrap();
        }
        writeln!(
            body,
            "        var first={int_ty}(0);var second={int_ty}(0);"
        )
        .unwrap();
        for q in 0..8 {
            let quartet = if down && dtype == crate::quant_matvec::IQ4_NL {
                if !matches!(q, 0 | 1 | 4 | 5) {
                    continue;
                }
                format!("part*2u+{q}u")
            } else {
                format!("{q}u")
            };
            writeln!(
                body,
                "        {{let q={quartet};let packed={name}_packed(prepared,q);let at=block*8u+q;"
            )
            .unwrap();
            for i in 0..width {
                let input = if direct_planes {
                    format!("strata_input_word(frame{i},at)")
                } else {
                    format!("strata_input[frame{i}+at]")
                };
                writeln!(
                    body,
                    "            {}{}+=dot4I8Packed(packed,{input});",
                    if q < 4 { "first" } else { "second" },
                    axis(i)
                )
                .unwrap();
            }
            body.push_str("        }\n");
        }
        for i in 0..width {
            let a = axis(i);
            let weighted = match dtype {
                crate::quant_matvec::IQ2_S => format!(
                    "(first{a}*i32(prepared.scale&15u)+second{a}*i32(prepared.scale>>4u)+(first{a}+second{a})/2)/4"
                ),
                crate::quant_matvec::IQ3_XXS => format!(
                    "(i32(prepared.signs>>28u)*(first{a}+second{a})+(first{a}+second{a})/2)/2"
                ),
                crate::quant_matvec::IQ3_S => {
                    format!("(first{a}+second{a})*i32(1u+2u*prepared.scale)")
                }
                crate::quant_matvec::IQ4_XS => {
                    format!("(first{a}+second{a})*(i32(prepared.scale)-32)")
                }
                _ => format!("first{a}+second{a}"),
            };
            let scale = if down { 160 } else { 640 };
            let input = if direct_planes {
                format!("strata_input_word(frame{i},{scale}u+block)")
            } else {
                format!("strata_input[frame{i}+{scale}u+block]")
            };
            writeln!(body,"        {{let weighted={weighted};let ds=bitcast<f32>({input});values{a}+=(prepared.d*ds)*f32(weighted);}}").unwrap();
        }
    } else {
        let halves = if !down && dtype == crate::quant_matvec::IQ2_S {
            2
        } else {
            1
        };
        let quartets = 8 / halves;
        for half in 0..halves {
            writeln!(body, "        {{var dots={int_ty}(0);").unwrap();
            for q in half * quartets..(half + 1) * quartets {
                writeln!(
                    body,
                    "            {{let packed={name}_packed(prepared,{q}u);let at=block*8u+{q}u;"
                )
                .unwrap();
                for i in 0..width {
                    writeln!(
                        body,
                        "                dots{}+=dot4I8Packed(packed,strata_input[frame{i}+at]);",
                        axis(i)
                    )
                    .unwrap();
                }
                body.push_str("            }\n");
            }
            writeln!(
                body,
                "            let scale={name}_factor(prepared{});",
                if down {
                    String::new()
                } else {
                    format!(",{half}u")
                }
            )
            .unwrap();
            for i in 0..width {
                let a = axis(i);
                writeln!(body, "            values{a}=fma(scale,bitcast<f32>(strata_input[frame{i}+{}u+block])*f32(dots{a}),values{a});", if down { 160 } else { 640 }).unwrap();
            }
            body.push_str("        }\n");
        }
    }
    body.push_str("    }\n");
    if capacity == 8 {
        writeln!(body, "    for(var stride={}u;stride>0u;stride/=2u) {{let other:{float_ty}=subgroupShuffleXor(values,stride);values+=other;}}", lanes/2).unwrap();
    } else {
        writeln!(body, "    for(var stride={}u;stride>0u;stride/=2u) {{let other:{float_ty}=subgroupShuffleDown(values,stride);if lane<stride {{values+=other;}}}}", lanes/2).unwrap();
    }
    body.push_str("    if lane==0u {\n");
    for i in 0..width {
        let offset = if role == 2 { 640 } else { 0 };
        let status_offset = if down || role == 2 { 80 } else { 0 };
        writeln!(body, "        output[p.d+route{i}*p.e+{offset}u+row]=values{};if row==0u {{status[p.f+{status_offset}u+route{i}]=1u;}}", axis(i)).unwrap();
    }
    body.push_str("    }\n}\n");
    let first = source
        .find("@compute @workgroup_size(256)")
        .expect("native expert entry");
    let end =
        crate::strata_port::function_end(&source, source[first..].find("fn main").unwrap() + first);
    source.replace_range(first..end, &body);
    if direct_planes {
        // Packed bytes and F32 scales occupy separate regions of the same
        // original admitted owner. Keep the frame explicit: no division or
        // remainder to recover a member from each dot-product input address.
        let packed = if down { 160 } else { 640 };
        writeln!(source, "\nfn strata_input_word(frame:u32,at:u32)->u32 {{if at<{packed}u {{return planes[frame+at];}}return planes[frame+{packed}u+at];}}").unwrap();
    }
    source
}

pub(super) const ACTIVATE: &str = r#"
struct Params { count:u32, first:u32, stride:u32, act_first:u32, act_stride:u32, status_first:u32, r0:u32, r1:u32 }
@group(0) @binding(0) var<storage,read_write> partial:array<f32>;
@group(0) @binding(1) var<storage,read_write> activated:array<f32>;
@group(0) @binding(2) var<storage,read_write> status:array<u32>;
@group(0) @binding(3) var<uniform> p:Params;
fn silu(x:f32)->f32 {
    if x>=0.0 {return x/(1.0+exp(-x));}
    let e=exp(x);return x*e/(1.0+e);
}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group:vec3<u32>, @builtin(local_invocation_index) lane:u32) {
    let route=group.x;
    if p.count<1u || p.count>4u || route>=p.count*10u || group.y!=0u || group.z!=0u || p.r0!=0u || p.r1!=0u {return;}
    if p.first>arrayLength(&partial) || p.stride<2560u || 1280u>arrayLength(&partial)-p.first {return;}
    if 79u>(arrayLength(&partial)-p.first-1280u)/p.stride {return;}
    if p.act_first>arrayLength(&activated) || p.act_stride<640u || 640u>arrayLength(&activated)-p.act_first {return;}
    if 79u>(arrayLength(&activated)-p.act_first-640u)/p.act_stride {return;}
    if p.status_first>arrayLength(&status) || 160u>arrayLength(&status)-p.status_first {return;}
    // Every lane checks both completed roles before any can clear the UP stamp.
    let completed=status[p.status_first+route]==1u && status[p.status_first+80u+route]==1u;
    workgroupBarrier();
    let first=p.first+route*p.stride;
    for(var row=lane;row<640u;row+=64u) {
        let g=partial[first+row];let u=partial[first+640u+row];
        if completed {activated[p.act_first+route*p.act_stride+row]=silu(g)*u;}
        partial[first+row]=0.0;partial[first+640u+row]=0.0;
    }
    workgroupBarrier();
    if lane==0u {status[p.status_first+route]=select(0u,1u,completed);status[p.status_first+80u+route]=0u;}
}
"#;
