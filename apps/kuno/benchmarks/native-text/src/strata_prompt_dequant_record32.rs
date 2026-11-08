//! Decode one authentic quantized 32-value record per invocation, retaining
//! the exact original eight-value rounding, finite checks and publication.
//! No arithmetic, physical source, owner, role or completion rule changes.
use super::*;

pub(super) fn source(mut source: String, down: bool) -> String {
    let start = source
        .find("    let i=id.x;if i>=p.rows*p.columns/8u")
        .unwrap();
    let end = source[start..]
        .find("    let values=value8(row,col);\n")
        .unwrap()
        + start
        + "    let values=value8(row,col);\n".len();
    let body_end =
        crate::strata_port::function_end(&source, source[..start].rfind("fn main(").unwrap());
    let store = source[end..body_end - 1].to_owned();
    // Both original full-image and exact role-view guards remain verbatim.
    // The selected kernel requires a whole record before any weight access.
    let opening = "    if !valid() {return;}";
    assert_eq!(source.matches(opening).count(), 1);
    source.replace_range(
        start..body_end - 1,
        r#"    if p.columns%32u!=0u {atomicOr(&fault[0],128u);return;}
    let record=id.x;if record>=p.rows*p.columns/32u {return;}
    let row=record/(p.columns/32u);let col=(record%(p.columns/32u))*32u;
    write_record32(row,col,record*4u);
"#,
    );
    source.push_str("\nfn store_original8(i:u32,values:Decoded8) {\n");
    source.push_str(&store);
    source.push_str("}\nfn write_record32(row:u32,col:u32,first:u32) {\n");
    // Prepare exactly once in each mutually exclusive real projection branch.
    // IQ2_S uses its original distinct 16-value subscales. The other formats'
    // factor helpers ignore the half index, just as in the original value8.
    let branches: &[(&str, &str)] = if down {
        &[("", "gate")]
    } else {
        &[("if (row%2u)==0u {", "gate"), ("} else {", "up")]
    };
    for &(open, role) in branches {
        source.push_str(open);
        let source_row = if down { "row" } else { "(row/2u)" };
        source.push_str(&format!(
            "\nlet e={role}_prepare({source_row}*p.row_bytes,col/32u);\n"
        ));
        for quarter in 0..4 {
            let factor = if down {
                "e.d".into()
            } else {
                format!("{role}_factor(e,{}u)", quarter / 2)
            };
            source.push_str(&format!("store_original8(first+{quarter}u,decoded8({factor},{role}_packed(e,{}u),{role}_packed(e,{}u)));\n",quarter*2,quarter*2+1));
        }
    }
    if !down {
        source.push_str("}\n");
    }
    source.push_str("}\n// STRATA_DEQUANT_RECORD32_ORIGINAL_STORE8\n");
    source
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_record32_keeps_exact_gu_down_rounding_fault_and_owner_body() {
        for (down, types) in [
            (false, &[q::IQ2_S, q::IQ3_S, q::IQ3_XXS, q::IQ4_XS][..]),
            (true, &[q::IQ4_NL, q::Q2_0][..]),
        ] {
            for &dtype in types {
                // Generate the unselected source too, even when feature active.
                let actual = Op::Dequant { dtype, down }.source();
                let selected = if cfg!(feature = "strata-dequant-record32") {
                    actual
                } else {
                    source(actual, down)
                };
                let module = wgpu::naga::front::wgsl::parse_str(&selected).unwrap();
                wgpu::naga::valid::Validator::new(
                    wgpu::naga::valid::ValidationFlags::all(),
                    wgpu::naga::valid::Capabilities::all(),
                )
                .validate(&module)
                .unwrap();
                let prepare = selected.split_once("fn write_record32(").unwrap().1;
                assert_eq!(prepare.matches("gate_prepare(").count(), 1);
                assert_eq!(prepare.matches("up_prepare(").count(), usize::from(!down));
                assert_eq!(
                    prepare.matches("store_original8(").count(),
                    if down { 4 } else { 8 }
                );
                let original = include_str!("strata_prompt_experts.wgsl");
                let store = original
                    .split_once("    let values=value8(row,col);\n")
                    .unwrap()
                    .1
                    .split_once("\n}\n// DEQUANT_END")
                    .unwrap()
                    .0;
                assert_eq!(
                    selected
                        .split_once("fn store_original8(i:u32,values:Decoded8) {\n")
                        .unwrap()
                        .1
                        .split_once("\n}\nfn write_record32")
                        .unwrap()
                        .0,
                    store
                );
                assert!(selected.contains("p.columns%32u!=0u"));
                assert!(selected.contains("arrayLength(&addresses)!=2048u"));
                assert!(selected.contains("arrayLength(&raw)!=p.r5"));
                assert!(selected.contains("record>=p.rows*p.columns/32u"));
                if !down {
                    assert_eq!(prepare.matches("_factor(e,0u)").count(), 4);
                    assert_eq!(prepare.matches("_factor(e,1u)").count(), 4);
                }
            }
        }
    }

    #[test]
    fn actual_record_grids_cover_every_original_vector_once_and_reject_no_owner() {
        for (rows, cols) in [(1280usize, 2560usize), (2560, 640)] {
            let records = rows * cols / 32;
            let mut vectors = vec![0u8; rows * cols / 8];
            for record in 0..records.div_ceil(256) * 256 {
                if record >= records {
                    continue;
                }
                let row = record / (cols / 32);
                let col = (record % (cols / 32)) * 32;
                for quarter in 0..4 {
                    let i = record * 4 + quarter;
                    assert_eq!(i, row * (cols / 8) + (col + quarter * 8) / 8);
                    vectors[i] += 1;
                }
            }
            assert!(vectors.iter().all(|&n| n == 1));
            assert_eq!(records * 32, rows * cols);
        }
        assert_eq!(
            Op::dequant_values(),
            if cfg!(feature = "strata-dequant-record32") {
                32
            } else {
                8
            }
        );
    }
}
