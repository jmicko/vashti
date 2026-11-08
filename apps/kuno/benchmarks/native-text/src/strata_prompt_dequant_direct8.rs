//! Lane-local eight-value dequantization, following pinned MIT Strata's Intel
//! dq_dispatch/store_run schedule. No whole-record leader, cross-lane payload,
//! scratch, new allocation, or change to the original publication/owner rules.
use super::*;

fn decoder(name: &str, dtype: u32) -> String {
    let body = match dtype {
        q::IQ3_XXS => {
            r#"
let base=row+(block/8u)*98u;let group=block%8u;
let codes=NAME_direct_halfword(base+2u+group*8u+quarter*2u);
let aux=NAME_word(base+66u+group*4u);
let sign_code=(aux>>(quarter*7u))&127u;
let signs=sign_code|((countOneBits(sign_code)&1u)<<7u);
let d=NAME_half(base)*(0.5+f32(aux>>28u))*0.5;
return decoded8(d,iq3_signed_word(iq3_grid[codes&255u],signs),
    iq3_signed_word(iq3_grid[codes>>8u],signs>>4u));"#
        }
        q::IQ3_S => {
            r#"
let base=row+(block/8u)*110u;let group=block%8u;
let codes=NAME_direct_halfword(base+2u+group*8u+quarter*2u);
let high=NAME_byte(base+66u+group);
let first=(codes&255u)|(((high>>(quarter*2u))&1u)<<8u);
let second=(codes>>8u)|(((high>>(quarter*2u+1u))&1u)<<8u);
let signs=NAME_byte(base+74u+group*4u+quarter);
let scale=(NAME_byte(base+106u+group/2u)>>((group%2u)*4u))&15u;
let d=NAME_half(base)*f32(1u+2u*scale);
return decoded8(d,iq3_signed_word(iq3_grid[first],signs),
    iq3_signed_word(iq3_grid[second],signs>>4u));"#
        }
        q::IQ2_S => {
            r#"
let base=row+(block/8u)*82u;let group=block%8u;
let index=NAME_byte(base+2u+group*4u+quarter)
    |(((NAME_byte(base+66u+group)>>(quarter*2u))&3u)<<8u);
let signs=NAME_byte(base+34u+group*4u+quarter);
let scale=(NAME_byte(base+74u+group)>>((quarter/2u)*4u))&15u;
let d=NAME_half(base)*(0.5+f32(scale))*0.25;
return decoded8(d,iq3_signed_word(iq3_grid[index*2u],signs),
    iq3_signed_word(iq3_grid[index*2u+1u],signs>>4u));"#
        }
        q::IQ4_XS => {
            r#"
let base=row+(block/8u)*136u;let group=block%8u;
let header=NAME_word(base);
let low=(NAME_byte(base+4u+group/2u)>>((group%2u)*4u))&15u;
let high=(header>>(16u+group*2u))&3u;
let scale=low|(high<<4u);
let codes=NAME_load8(base+8u+group*16u+(quarter%2u)*8u);
let d=unpack2x16float(header&65535u).x*f32(i32(scale)-32);
return decoded8(d,iq4_word(codes.x,quarter>=2u),iq4_word(codes.y,quarter>=2u));"#
        }
        q::IQ4_NL => {
            r#"
let base=row+block*18u;
let codes=NAME_load8(base+2u+(quarter%2u)*8u);
return decoded8(NAME_half(base),iq4_word(codes.x,quarter>=2u),iq4_word(codes.y,quarter>=2u));"#
        }
        q::Q2_0 => {
            r#"
let base=row+(block/2u)*18u;
let codes=NAME_direct_halfword(base+2u+(block%2u)*8u+quarter*2u);
return decoded8(NAME_half(base),q2_word(codes&255u),q2_word(codes>>8u));"#
        }
        _ => unreachable!("validated original six direct8 formats"),
    };
    // Every original format's row/block/payload is two-byte aligned. Reading
    // only its aligned half-word cannot touch the following payload/owner.
    let helper = format!(
        "fn NAME_direct_halfword(at:u32)->u32 {{return (NAME[at/4u]>>((at%4u)*8u))&65535u;}}\nfn NAME_direct8(row:u32,block:u32,quarter:u32)->Decoded8 {{{body}}}\n"
    ).replace("NAME", name);
    crate::strata_expert_fused::replace_reads(helper, name)
}

pub(super) fn source(mut source: String, dtype: u32, down: bool) -> String {
    assert!(source.contains("// STRATA_DEQUANT_RECORD32_ORIGINAL_STORE8"));
    let old = "    let record=id.x;if record>=p.rows*p.columns/32u {return;}\n    let row=record/(p.columns/32u);let col=(record%(p.columns/32u))*32u;\n    write_record32(row,col,record*4u);";
    assert_eq!(source.matches(old).count(), 1);
    let dispatch = if down {
        "    let i=id.x;if i>=p.rows*p.columns/8u {return;}\n    let row=i/(p.columns/8u);let col=(i%(p.columns/8u))*8u;\n    store_original8(i,gate_direct8(row*p.row_bytes,col/32u,(col%32u)/8u));"
    } else {
        "    let i=id.x;if i>=p.rows*p.columns/8u {return;}\n    let row=i/(p.columns/8u);let col=(i%(p.columns/8u))*8u;\n    if row%2u==0u {store_original8(i,gate_direct8((row/2u)*p.row_bytes,col/32u,(col%32u)/8u));}\n    else {store_original8(i,up_direct8((row/2u)*p.row_bytes,col/32u,(col%32u)/8u));}"
    };
    source = source.replace(old, dispatch);
    assert_eq!(source.matches("@workgroup_size(256)").count(), 1);
    source = source.replace("@workgroup_size(256)", "@workgroup_size(32)");
    let mut helpers = decoder("gate", dtype);
    if !down {
        helpers.push_str(&decoder("up", dtype));
    }
    let offset = match dtype {
        q::IQ3_XXS => 0,
        q::IQ3_S => 256,
        q::IQ2_S => 768,
        _ => 0,
    };
    source.push_str(&helpers.replace("iq3_grid[", &format!("strata_grid[{offset}u+")));
    source.push_str("\n// STRATA_DEQUANT_DIRECT8_ORIGINAL_STORE8\n");
    source
}

fn validate_one(source: &str, label: &str) -> std::result::Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;
    let words = cubecl::wgpu::native_expert_shader_words(source)?;
    let mut at = 5;
    let mut actual_wg32 = 0;
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        if count == 0 || at + count > words.len() {
            return Err("direct8 SPIRV extent".into());
        }
        // No workgroup synchronization or cross-lane execution is needed.
        if matches!(words[at] & 65535, 224 | 225 | 333..=366) {
            return Err("direct8 unexpected collective/barrier".into());
        }
        // OpExecutionMode LocalSize on the actual physical-source module,
        // rather than trusting WGSL text, feature selection, or a fixture ID.
        if words[at] & 65535 == 16 && count >= 3 && words[at + 2] == 17 {
            if count != 6 || words[at + 3..at + 6] != [32, 1, 1] {
                return Err("direct8 actual SPIRV workgroup geometry".into());
            }
            actual_wg32 += 1;
        }
        at += count;
    }
    if actual_wg32 != 1 {
        return Err("direct8 missing actual WG32 entry".into());
    }
    let mut validator = std::process::Command::new("spirv-val")
        .args(["--target-env", "vulkan1.3", "-"])
        .stdin(std::process::Stdio::piped())
        .spawn()?;
    validator
        .stdin
        .take()
        .ok_or("direct8 validator stdin")?
        .write_all(
            &words
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
    if !validator.wait()?.success() {
        return Err("direct8 SPIRV validation".into());
    }
    println!(
        "strata_dequant_direct8_shader_accepted layout={label} values_per_invocation=8 encoded_shuffles=0 workgroup_invocations=32"
    );
    Ok(())
}
pub(crate) fn validate_sources() -> std::result::Result<(), Box<dyn std::error::Error>> {
    for op in super::dequant_operators() {
        let (dtype, down) = match op {
            Op::Dequant { dtype, down }
            | Op::ResidentDequant { dtype, down }
            | Op::SlotDequant { dtype, down, .. } => (dtype, down),
            _ => unreachable!(),
        };
        validate_one(&source(op.source(), dtype, down), &format!("{op:?}"))?;
    }
    #[cfg(feature = "strata-expert-cohorts")]
    for (down, types) in [
        (false, &[q::IQ2_S, q::IQ3_S, q::IQ3_XXS, q::IQ4_XS][..]),
        (true, &[q::IQ4_NL, q::Q2_0][..]),
    ] {
        for &dtype in types {
            let op = cohorts::CohortOp::Image { dtype, down };
            validate_one(
                &source(op.source(), dtype, down),
                &format!("cohort-{dtype}-{down}"),
            )?;
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "strata_prompt_dequant_direct8_tests.rs"]
mod tests;
