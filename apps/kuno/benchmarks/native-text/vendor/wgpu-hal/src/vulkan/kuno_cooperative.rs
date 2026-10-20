//! Fixed subgroup selection for the port's Intel cooperative matrix shaders.
//! This only supplies compiler-owned pipeline metadata, never device authority.

const SUBGROUP: u32 = 16;
// Original blocks and the opt-in CubeK16×N block have at most sixteen tiles.
// A separately recognized native512×1 block must prove genuine subgroup
// topology below. Physical workgroup/subgroup caps still precede execution.
const MAX_BLOCK_LANES: u32 = 256;

/// Select SIMD16 for the port's subgroup-scoped8x16x16 matrix instructions.
/// A block may contain several independent16-lane subgroups. Other shaders,
/// incomplete modules and unsupported geometry do not select this override.
/// Full SPIR-V validation and physical device limits remain separate gates.
pub fn kuno_cooperative_subgroup(words: &[u32]) -> Option<u32> {
    if words.len() < 5 || words[0] != 0x07230203 || words[3] == 0 || words[4] != 0 {
        return None;
    }
    let mut entry = None;
    let mut local = None;
    let mut explicit_subgroup = None;
    let mut capability = false;
    let mut matrices = 0usize;
    let mut subgroup_id = false;
    let mut subgroup_lane = false;
    let mut at = 5;
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        let opcode = words[at] & 65535;
        if count == 0 || at.checked_add(count).is_none_or(|end| end > words.len()) {
            return None;
        }
        let instruction = &words[at..at + count];
        match opcode {
            // OpCapability CooperativeMatrixKHR.
            17 if count == 2 && instruction[1] == 6022 => capability = true,
            // Exactly one GLCompute entry. Associate its literal LocalSize
            // with this entry instead of accepting a different entry's mode.
            15 => {
                if count < 4
                    || instruction[1] != 5
                    || instruction[2] == 0
                    || instruction[2] >= words[3]
                    || !instruction[3..]
                        .iter()
                        .any(|word| word.to_le_bytes().contains(&0))
                    || entry.replace(instruction[2]).is_some()
                {
                    return None;
                }
            }
            // OpExecutionMode LocalSize.
            16 if count >= 3 && instruction[2] == 17 => {
                if count != 6
                    || local
                        .replace((
                            instruction[1],
                            instruction[3],
                            instruction[4],
                            instruction[5],
                        ))
                        .is_some()
                {
                    return None;
                }
            }
            // An explicit conflicting SubgroupSize must not be overridden.
            16 if count >= 3 && instruction[2] == 35 => {
                if count != 4
                    || explicit_subgroup
                        .replace((instruction[1], instruction[3]))
                        .is_some()
                {
                    return None;
                }
            }
            // OpExecutionModeId LocalSizeId is not the literal block geometry
            // this compiler contract proves. Never guess specialization IDs.
            331 if count >= 3 && instruction[2] == 38 => return None,
            // OpTypeCooperativeMatrixKHR has six operands, all IDs. Only the
            // port's known subgroup-scoped A/B/accumulator shapes select16.
            4456 => {
                if count != 7 || !known_matrix_type(words, instruction) {
                    return None;
                }
                matrices += 1;
            }
            // The bounded512-lane native entry uses subgroup builtins for
            // actual32-subgroup identity/lane mapping, not local-ID division.
            71 if count == 4 && instruction[2] == 11 => {
                subgroup_id |= instruction[3] == 40;
                subgroup_lane |= instruction[3] == 41;
            }
            _ => {}
        }
        at += count;
    }
    let entry = entry?;
    let (owner, x, y, z) = local?;
    if !capability
        || matrices == 0
        || owner != entry
        || z != 1
        || x == 0
        || (x > MAX_BLOCK_LANES
            && !(x == 512
                && y == 1
                && words[1] == 0x00010600
                && subgroup_id
                && subgroup_lane))
        || !x.is_multiple_of(SUBGROUP)
        // CubeK keeps its genuine2D geometry. Its selected compiler entry
        // maps logical planes/lanes to subgroup builtins, never assumes that
        // LocalInvocationId.y equals SubgroupId. SPIR-V1.6 plus X=required16
        // provides full-subgroup completeness without an unenabled pipeline
        // REQUIRE_FULL_SUBGROUPS feature. Device geometry checks remain real.
        || (y != 1
            && (words[1] != 0x00010600 || x != SUBGROUP || y == 0 || y > 16))
        || explicit_subgroup.is_some_and(|mode| mode != (entry, SUBGROUP))
    {
        return None;
    }
    Some(SUBGROUP)
}

fn known_matrix_type(words: &[u32], instruction: &[u32]) -> bool {
    let scope = constant(words, instruction[3]);
    let rows = constant(words, instruction[4]);
    let columns = constant(words, instruction[5]);
    let usage = constant(words, instruction[6]);
    let expected_width = match (rows, columns, usage) {
        (Some(8), Some(16), Some(0)) | (Some(16), Some(16), Some(1)) => 16,
        (Some(8), Some(16), Some(2)) => 32,
        _ => return false,
    };
    scope == Some(3) && float_width(words, instruction[2]) == Some(expected_width)
}

fn constant(words: &[u32], id: u32) -> Option<u32> {
    let instruction = defining_instruction(words, id, 43)?;
    if instruction.len() != 4 {
        return None;
    }
    let ty = defining_instruction(words, instruction[1], 21)?;
    (ty.len() == 4 && ty[2] == 32 && ty[3] == 0).then_some(instruction[3])
}

fn float_width(words: &[u32], id: u32) -> Option<u32> {
    let instruction = defining_instruction(words, id, 22)?;
    // BF16 carries the optional BFloat16KHR encoding operand. No other input
    // encoding is part of this native port's admitted matrix contract.
    match instruction {
        [_, _, width] => Some(*width),
        [_, _, 16, 0] => Some(16),
        _ => None,
    }
}

fn defining_instruction(words: &[u32], id: u32, opcode: u32) -> Option<&[u32]> {
    if id == 0 || id >= words[3] {
        return None;
    }
    let mut found = None;
    let mut at = 5;
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        if count == 0 || at.checked_add(count).is_none_or(|end| end > words.len()) {
            return None;
        }
        let instruction = &words[at..at + count];
        let result = if opcode == 43 { 2 } else { 1 };
        if words[at] & 65535 == opcode && count > result && instruction[result] == id {
            if found.replace(instruction).is_some() {
                return None;
            }
        }
        at += count;
    }
    found
}

#[cfg(test)]
mod tests {
    use super::kuno_cooperative_subgroup as subgroup;

    fn module(lanes: u32) -> Vec<u32> {
        let mut words = vec![0x07230203, 0x00010600, 0, 64, 0];
        for instruction in [
            vec![2 << 16 | 17, 1], // Shader capability.
            vec![2 << 16 | 17, 6022],
            vec![5 << 16 | 15, 5, 30, 0x6e69616d, 0],
            vec![6 << 16 | 16, 30, 17, lanes, 1, 1],
            vec![4 << 16 | 21, 1, 32, 0],
            vec![3 << 16 | 22, 2, 16],
            vec![3 << 16 | 22, 3, 32],
            vec![4 << 16 | 43, 1, 4, 3], // Scope Subgroup.
            vec![4 << 16 | 43, 1, 5, 8],
            vec![4 << 16 | 43, 1, 6, 16],
            vec![4 << 16 | 43, 1, 7, 0], // Matrix A.
            vec![4 << 16 | 43, 1, 8, 1], // Matrix B.
            vec![4 << 16 | 43, 1, 9, 2], // Accumulator.
            vec![7 << 16 | 4456, 10, 2, 4, 5, 6, 7],
            vec![7 << 16 | 4456, 11, 2, 4, 6, 6, 8],
            vec![7 << 16 | 4456, 12, 3, 4, 5, 6, 9],
        ] {
            words.extend(instruction);
        }
        words
    }

    fn instruction(words: &mut [u32], opcode: u32, nth: usize) -> &mut [u32] {
        let mut at = 5;
        let mut seen = 0;
        while at < words.len() {
            let count = (words[at] >> 16) as usize;
            if words[at] & 65535 == opcode {
                if seen == nth {
                    return &mut words[at..at + count];
                }
                seen += 1;
            }
            at += count;
        }
        panic!("fixture instruction absent")
    }

    #[test]
    fn one_or_multiple_complete_subgroups_select_the_same_fixed16_metadata() {
        for lanes in [16, 32, 48, 64, 128, 256] {
            assert_eq!(subgroup(&module(lanes)), Some(16));
        }
    }

    #[test]
    fn partial_unbounded_or_multidimensional_blocks_do_not_select_an_override() {
        for lanes in [0, 1, 8, 17, 31, 127, 257, 512, u32::MAX] {
            assert_eq!(subgroup(&module(lanes)), None);
        }
        for axis in [4, 5] {
            let mut words = module(128);
            instruction(&mut words, 16, 0)[axis] = 2;
            assert_eq!(subgroup(&words), None);
        }
    }

    #[test]
    fn cubek_keeps_bounded_original_two_dimensional_geometry_and_spirv16() {
        for planes in [2, 4, 8, 16] {
            let mut words = module(16);
            instruction(&mut words, 16, 0)[4] = planes;
            assert_eq!(subgroup(&words), Some(16));
        }
        for (x, y, version) in [
            (16, 0, 0x00010600),
            (16, 17, 0x00010600),
            (32, 4, 0x00010600),
            (16, 4, 0x00010500),
        ] {
            let mut words = module(x);
            words[1] = version;
            instruction(&mut words, 16, 0)[4] = y;
            assert_eq!(subgroup(&words), None);
        }
    }

    #[test]
    fn native_512_block_requires_literal_spirv16_and_both_real_subgroup_builtins() {
        let both = |words: &mut Vec<u32>| {
            words.extend([4 << 16 | 71, 40, 11, 40]);
            words.extend([4 << 16 | 71, 41, 11, 41]);
        };
        let mut words = module(512);
        assert_eq!(subgroup(&words), None);
        words.extend([4 << 16 | 71, 40, 11, 40]);
        assert_eq!(subgroup(&words), None);
        words.extend([4 << 16 | 71, 41, 11, 41]);
        assert_eq!(subgroup(&words), Some(16));
        for (x, y, z, version) in [
            (272, 1, 1, 0x00010600),
            (496, 1, 1, 0x00010600),
            (528, 1, 1, 0x00010600),
            (512, 2, 1, 0x00010600),
            (512, 1, 2, 0x00010600),
            (512, 1, 1, 0x00010500),
        ] {
            let mut words = module(x);
            both(&mut words);
            words[1] = version;
            let local = instruction(&mut words, 16, 0);
            local[4] = y;
            local[5] = z;
            assert_eq!(subgroup(&words), None);
        }
    }

    #[test]
    fn bf16_inputs_preserve_the_fixed16_selection() {
        let mut words = module(128);
        let at = words
            .windows(3)
            .position(|i| i == [3 << 16 | 22, 2, 16])
            .unwrap();
        words.splice(at..at + 3, [4 << 16 | 22, 2, 16, 0]);
        assert_eq!(subgroup(&words), Some(16));
        instruction(&mut words, 22, 0)[3] = 1;
        assert_eq!(subgroup(&words), None);
    }

    #[test]
    fn ordinary32_shaders_never_gain_a_cooperative_override() {
        let mut words = module(32);
        while let Some(at) = words.iter().position(|&v| v == 7 << 16 | 4456) {
            words.drain(at..at + 7);
        }
        assert_eq!(subgroup(&words), None);
    }

    #[test]
    fn only_known_subgroup_scoped_matrix_types_select16() {
        for (constant_index, value) in [(0, 2), (1, 16), (2, 32), (3, 2)] {
            let mut words = module(128);
            instruction(&mut words, 43, constant_index)[3] = value;
            assert_eq!(subgroup(&words), None);
        }
        let mut words = module(128);
        instruction(&mut words, 4456, 0)[2] = 3; // F32 input is outside the port.
        assert_eq!(subgroup(&words), None);
        let mut words = module(128);
        instruction(&mut words, 17, 1)[1] = 1; // Cooperative capability absent.
        assert_eq!(subgroup(&words), None);
    }

    #[test]
    fn exactly_one_matching_compute_entry_and_literal_geometry_are_required() {
        for model in [0, 4, 6] {
            let mut words = module(128);
            instruction(&mut words, 15, 0)[1] = model;
            assert_eq!(subgroup(&words), None);
        }
        let mut words = module(128);
        instruction(&mut words, 16, 0)[1] = 31;
        assert_eq!(subgroup(&words), None);
        let mut words = module(128);
        words.extend([5 << 16 | 15, 5, 31, 0x6e69616d, 0]);
        assert_eq!(subgroup(&words), None);
        let mut words = module(128);
        words.extend([6 << 16 | 16, 30, 17, 128, 1, 1]);
        assert_eq!(subgroup(&words), None);
        let mut words = module(128);
        words.extend([6 << 16 | 331, 30, 38, 5, 6, 7]);
        assert_eq!(subgroup(&words), None);
    }

    #[test]
    fn explicit_subgroup_modes_must_match_both_width_and_entry() {
        let mut words = module(128);
        words.extend([4 << 16 | 16, 30, 35, 16]);
        assert_eq!(subgroup(&words), Some(16));
        for mode in [[4 << 16 | 16, 30, 35, 32], [4 << 16 | 16, 31, 35, 16]] {
            let mut words = module(128);
            words.extend(mode);
            assert_eq!(subgroup(&words), None);
        }
    }

    #[test]
    fn malformed_headers_instructions_ids_and_types_are_rejected() {
        assert_eq!(subgroup(&[]), None);
        assert_eq!(subgroup(&[0; 5]), None);
        for trailer in [vec![0], vec![4 << 16 | 71, 2, 6]] {
            let mut words = module(128);
            words.extend(trailer);
            assert_eq!(subgroup(&words), None);
        }
        let mut words = module(128);
        instruction(&mut words, 4456, 0)[3] = 64;
        assert_eq!(subgroup(&words), None);
        let mut words = module(128);
        words.extend([4 << 16 | 43, 1, 4, 3]); // Duplicate defining constant.
        assert_eq!(subgroup(&words), None);
    }
}
