//! Count4 compact projections with the original eight-lane F32 tree.
//! Only lane placement and reduction transport change. All decoders, two Q8
//! planes, per-lane accumulation, routing, bounds and output identities derive
//! from the accepted original compact source. Selection is laboratory-only.

// Immutable public experiment selector, reported by both protected probes.
pub(crate) const PUBLIC_PROBE: bool = true;

fn replace_once(source: String, before: &str, after: &str) -> String {
    // These are fixed source templates, never model or request content. Refuse
    // template drift rather than silently compiling a half-transformed kernel.
    assert_eq!(source.matches(before).count(), 1);
    source.replacen(before, after, 1)
}

fn subgroup_rows(source: String) -> String {
    let source = replace_once(
        source,
        "fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) local:u32) {",
        "fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) local:u32,\n    @builtin(subgroup_id) sg:u32,@builtin(subgroup_invocation_id) sub_lane:u32,\n    @builtin(subgroup_size) width:u32,@builtin(num_subgroups) subgroups:u32) {",
    );
    let source = replace_once(
        source,
        "if group.x>=",
        "if (width!=16u && width!=32u) || subgroups!=64u/width || group.x>=",
    );
    replace_once(
        source,
        "let row=group.x*8u+local/8u;let lane=local%8u;",
        "// Logical rows do not assume any subgroup/local invocation mapping.\n    let row=group.x*8u+sg*(width/8u)+sub_lane/8u;let lane=sub_lane%8u;",
    )
}

pub(crate) fn gate_up(source: String) -> String {
    let source = subgroup_rows(source);
    let source = replace_once(
        source,
        "var<workgroup> gate_sums:array<f32,256>;\nvar<workgroup> up_sums:array<f32,256>;\n",
        "",
    );
    let begin =
        "    for(var j=0u;j<4u;j++) {gate_sums[local*4u+j]=gates[j];up_sums[local*4u+j]=ups[j];}";
    let first = source.find(begin).expect("original GU reduction");
    let last = first + source[first..].find("    if lane==0u {").unwrap();
    let mut source = source;
    source.replace_range(first..last, include_str!("frozen_compact_shuffle_gu.wgsl"));
    // All eight logical row lanes execute each collective. Only the same lower
    // half as the original shared-memory tree consumes its result.
    for (member, axis) in ["x", "y", "z", "w"].into_iter().enumerate() {
        source = replace_once(
            source,
            &format!("gate_sums[local*4u+{member}u]"),
            &format!("gates.{axis}"),
        );
        source = replace_once(
            source,
            &format!("up_sums[local*4u+{member}u]"),
            &format!("ups.{axis}"),
        );
    }
    replace_once(
        source,
        "            // All lanes finish reading their reduction before the next group\n            // can overwrite any shared cell, including unused member cells.\n            workgroupBarrier();",
        "            // Reductions are private registers; the codebook stays immutable.",
    )
}

pub(crate) fn down(source: String) -> String {
    let source = subgroup_rows(source);
    let source = replace_once(source, "var<workgroup> sums:array<f32,256>;\n", "");
    let begin = "    // Explicit initialization includes every unused RHS column. No WG zeroing.";
    let first = source.find(begin).expect("original DOWN reduction");
    let last = first + source[first..].find("    if lane==0u {").unwrap();
    let mut source = source;
    source.replace_range(
        first..last,
        include_str!("frozen_compact_shuffle_down.wgsl"),
    );
    for (member, axis) in ["x", "y", "z", "w"].into_iter().enumerate() {
        source = replace_once(
            source,
            &format!("sums[local*4u+{member}u]"),
            &format!("values.{axis}"),
        );
    }
    source
}

#[cfg(test)]
mod tests {
    #[test]
    fn eight_lane_shuffle_covers_every_row_and_keeps_all_four_member_bits() {
        for width in [16usize, 32] {
            let mut seen = [[false; 8]; 8];
            for sg in 0..64 / width {
                for sub_lane in 0..width {
                    let row = sg * (width / 8) + sub_lane / 8;
                    let lane = sub_lane % 8;
                    assert!(!seen[row][lane]);
                    seen[row][lane] = true;
                }
            }
            assert!(seen.into_iter().flatten().all(|v| v));
            for seed in 0..257usize {
                let mut input = [[[0f32; 4]; 8]; 8];
                for (row, lanes) in input.iter_mut().enumerate() {
                    for (lane, members) in lanes.iter_mut().enumerate() {
                        for (member, value) in members.iter_mut().enumerate() {
                            let index = row * 32 + lane * 4 + member;
                            // Cancellation, mixed scales, both signs and zero:
                            // unlike monotone sums this exposes tree changes.
                            let mantissa = ((index * 29 + seed * 17) % 101) as f32 - 50.;
                            *value =
                                mantissa * 2f32.powi(((index * 11 + seed * 7) % 81) as i32 - 40);
                        }
                    }
                }
                let mut original = input;
                for stride in [4, 2, 1] {
                    for lanes in &mut original {
                        for lane in 0..stride {
                            let other = lanes[lane + stride];
                            for (value, other) in lanes[lane].iter_mut().zip(other) {
                                *value += other;
                            }
                        }
                    }
                }
                // Store invocations by subgroup numbering, independent of
                // local invocation order. Each shuffle samples a full snapshot.
                let mut invocations = [[0f32; 4]; 64];
                for sg in 0..64 / width {
                    for sub_lane in 0..width {
                        let row = sg * (width / 8) + sub_lane / 8;
                        invocations[sg * width + sub_lane] = input[row][sub_lane % 8];
                    }
                }
                for stride in [4, 2, 1] {
                    let before = invocations;
                    for sg in 0..64 / width {
                        for sub_lane in 0..width {
                            if sub_lane % 8 < stride {
                                assert!(sub_lane + stride < width);
                                for member in 0..4 {
                                    invocations[sg * width + sub_lane][member] +=
                                        before[sg * width + sub_lane + stride][member];
                                }
                            }
                        }
                    }
                }
                for sg in 0..64 / width {
                    for sub_lane in (0..width).step_by(8) {
                        let row = sg * (width / 8) + sub_lane / 8;
                        for member in 0..4 {
                            assert_eq!(
                                invocations[sg * width + sub_lane][member].to_bits(),
                                original[row][0][member].to_bits()
                            );
                        }
                    }
                }
            }
        }
    }
}
