//! Exact GLSL half packing and retained inner product loops for quantized CMMA.
//! Typed CubeCL two-lane casts are compiler input, never the executed rounding
//! instruction: both real native module paths apply this same checked lowering.
use alloc::{
    collections::{BTreeMap, BTreeSet},
    vec::Vec,
};

/// Keep the genuine inner matrix-product loop rolled. Outer staging loops
/// contain the original fault atomics; they and all non-product loops retain
/// their original controls. Only the already-authenticated quant entry calls
/// this helper. No arithmetic, operand, edge or allocation is changed.
fn retain_product_loops(mut words: Vec<u32>, quant_entry: bool) -> Result<Vec<u32>, &'static str> {
    if !quant_entry {
        return Ok(words);
    }
    #[derive(Default)]
    struct Block {
        successors: Vec<u32>,
        product: bool,
        atomic: bool,
        loop_info: Option<(u32, u32, usize)>,
    }
    let mut blocks = BTreeMap::<u32, Block>::new();
    let mut current = None;
    let mut at = 5;
    while at < words.len() {
        let n = (words[at] >> 16) as usize;
        if n == 0 || at.checked_add(n).is_none_or(|end| end > words.len()) {
            return Err("quant loop instruction extent");
        }
        let op = words[at] & 65535;
        if op == 248 && n == 2 {
            current = Some(words[at + 1]);
            if blocks.insert(words[at + 1], Block::default()).is_some() {
                return Err("quant loop duplicate block");
            }
        } else if op == 56 {
            current = None;
        } else if let Some(block) = current.and_then(|id| blocks.get_mut(&id)) {
            match op {
                246 => {
                    if n != 4
                        || block
                            .loop_info
                            .replace((words[at + 1], words[at + 2], at + 3))
                            .is_some()
                    {
                        return Err("quant loop unsupported merge control");
                    }
                }
                249 if n == 2 => block.successors.push(words[at + 1]),
                250 if n == 4 || n == 6 => {
                    block.successors.extend_from_slice(&words[at + 2..at + 4])
                }
                251 => {
                    if n < 3 || (n - 3) % 2 != 0 {
                        return Err("quant loop unsupported switch width");
                    }
                    block.successors.push(words[at + 2]);
                    for target in (4..n).step_by(2) {
                        block.successors.push(words[at + target]);
                    }
                }
                4459 => block.product = true,
                241 => block.atomic = true,
                _ => {}
            }
        }
        at += n;
    }
    let mut controls = Vec::new();
    for (&header, block) in &blocks {
        let Some((merge, continuation, control)) = block.loop_info else {
            continue;
        };
        if !blocks.contains_key(&merge) || !blocks.contains_key(&continuation) {
            return Err("quant loop missing real merge/continue block");
        }
        let mut work = Vec::from([header]);
        let mut seen = BTreeSet::new();
        let mut product = false;
        let mut atomic = false;
        while let Some(id) = work.pop() {
            if id == merge || !seen.insert(id) {
                continue;
            }
            let block = blocks.get(&id).ok_or("quant loop unknown CFG target")?;
            product |= block.product;
            atomic |= block.atomic;
            work.extend_from_slice(&block.successors);
        }
        if product && !atomic {
            if !seen.contains(&continuation) || !matches!(words[control], 0 | 2) {
                return Err("quant product loop conflicting control or continuation");
            }
            controls.push(control);
        }
    }
    for control in controls {
        words[control] = 2;
    } // SPIR-V DontUnroll.
    Ok(words)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Scalar {
    Uint,
    Half,
    Float,
}

fn literal(i: &[u32]) -> Option<Vec<u8>> {
    let mut s = Vec::new();
    for w in i {
        for byte in w.to_le_bytes() {
            if byte == 0 {
                return Some(s);
            }
            s.push(byte);
        }
    }
    None
}

/// Preserve every original descriptor, owner, instruction result and arithmetic
/// operation while replacing genuine vec2 F32/F16 casts in the authenticated
/// quantized-cohort module (or its separately compiled public edge control).
/// The genuine quant entry also retains its atomic-free inner CMMA loop with
/// DontUnroll; every arithmetic instruction, CFG edge and outer loop is preserved.
/// Unrelated shaders return None without modification. Selected malformed or
/// unsupported F16 conversion shapes fail closed rather than execute FConvert.
pub fn kuno_quant_half_pack(words: &[u32]) -> Result<Option<Vec<u32>>, &'static str> {
    if words.len() < 5 || words[0] != 0x07230203 || words[3] == 0 || words[4] != 0 {
        return Err("quant half SPIRV header");
    }
    // Unrelated native kernels keep their exact compiler policy and do not
    // incur typed-SSA reflection or a changed acceptance rule.
    let mut probe = 5;
    let mut selected = false;
    while probe < words.len() {
        let n = (words[probe] >> 16) as usize;
        if n == 0 || probe.checked_add(n).is_none_or(|end| end > words.len()) {
            return Err("quant half instruction extent");
        }
        if words[probe] & 65535 == 15 && n >= 4 {
            selected |= matches!(
                literal(&words[probe + 3..probe + n]).as_deref(),
                Some(b"expert_quant_cohort" | b"quant_half_edges")
            );
        }
        probe += n;
    }
    if !selected {
        return Ok(None);
    }
    let mut instructions = Vec::new();
    let mut entries = Vec::new();
    let mut local = None;
    let mut types = BTreeMap::new();
    let mut u32_type = None;
    let mut bindings = BTreeMap::new();
    let mut sets = BTreeMap::new();
    let mut variables = BTreeMap::new();
    let mut glsl = None;
    let mut at = 5;
    while at < words.len() {
        let n = (words[at] >> 16) as usize;
        if n == 0 || at.checked_add(n).is_none_or(|end| end > words.len()) {
            return Err("quant half instruction extent");
        }
        let i = &words[at..at + n];
        let op = i[0] & 65535;
        match op {
            11 if n >= 3 && literal(&i[2..]).as_deref() == Some(b"GLSL.std.450") => {
                if glsl.replace(i[1]).is_some() {
                    return Err("quant half duplicate GLSL import");
                }
            }
            15 if n >= 4 => entries.push((i[1], i[2], literal(&i[3..]))),
            16 if n == 6 && i[2] == 17 => {
                if local.replace((i[1], [i[3], i[4], i[5]])).is_some() {
                    return Err("quant half duplicate local size");
                }
            }
            21 if n == 4 && i[2..4] == [32, 0] => {
                types.insert(i[1], (Scalar::Uint, 1));
                u32_type = Some(i[1]);
            }
            22 if n == 3 && matches!(i[2], 16 | 32) => {
                types.insert(
                    i[1],
                    (
                        if i[2] == 16 {
                            Scalar::Half
                        } else {
                            Scalar::Float
                        },
                        1,
                    ),
                );
            }
            23 if n == 4 => {
                if let Some(&(scalar, 1)) = types.get(&i[2]) {
                    types.insert(i[1], (scalar, i[3]));
                }
            }
            59 if n >= 4 => {
                variables.insert(i[2], i[3]);
            }
            71 if n == 4 && i[2] == 33 => {
                if bindings.insert(i[1], i[3]).is_some() {
                    return Err("quant half duplicate descriptor binding");
                }
            }
            71 if n == 4 && i[2] == 34 => {
                if sets.insert(i[1], i[3]).is_some() {
                    return Err("quant half duplicate descriptor set");
                }
            }
            _ => {}
        }
        instructions.push((at, n, op));
        at += n;
    }
    if entries.len() != 1 || entries[0].0 != 5 {
        return Err("quant half single real compute entry");
    }
    let (_, entry, name) = &entries[0];
    if local != Some((*entry, [64, 1, 1])) || super::kuno_cooperative_subgroup(words) != Some(16) {
        return Err("quant half genuine cooperative SG16/WG64 geometry");
    }
    let data_bindings = if name.as_deref() == Some(b"expert_quant_cohort") {
        7
    } else {
        3
    };
    if bindings.len() != data_bindings + 1 || sets.len() != bindings.len() {
        return Err("quant half original complete descriptor inventory");
    }
    let mut seen = Vec::new();
    for (&id, &binding) in &bindings {
        if binding as usize > data_bindings
            || seen.contains(&binding)
            || sets.get(&id) != Some(&0)
            || ((binding as usize) < data_bindings && variables.get(&id) != Some(&12))
            || (binding as usize == data_bindings && !matches!(variables.get(&id), Some(2 | 12)))
        {
            return Err("quant half original descriptor storage/order/set");
        }
        seen.push(binding);
    }
    let uint = u32_type.ok_or("quant half actual U32 type")?;
    let mut value_types = BTreeMap::new();
    // These are real typed SSA-producing instructions. In particular no
    // decoration/literal is interpreted as an instruction result definition.
    for &(at, n, op) in &instructions {
        let i = &words[at..at + n];
        if n >= 3
            && (matches!(op, 12 | 41..=52 | 55 | 59 | 61 | 65..=70 | 77..=83 | 245)
                || (109..=215).contains(&op))
            && types.contains_key(&i[1])
        {
            value_types.insert(i[2], i[1]);
        }
    }
    let mut conversions = BTreeMap::new();
    let mut original_pack = 0;
    let mut original_unpack = 0;
    for &(at, n, op) in &instructions {
        let i = &words[at..at + n];
        if op == 12 && n >= 6 && Some(i[3]) == glsl {
            original_pack += usize::from(i[4] == 58);
            original_unpack += usize::from(i[4] == 62);
        }
        if op != 115 {
            continue;
        }
        if n != 4 {
            return Err("quant half FConvert operand shape");
        }
        let out = types.get(&i[1]).copied();
        let input = value_types.get(&i[3]).and_then(|id| types.get(id)).copied();
        if input.is_none() || out.is_none() {
            return Err("quant half unproven FConvert input/result type");
        }
        if !matches!(out, Some((Scalar::Half, _))) && !matches!(input, Some((Scalar::Half, _))) {
            continue;
        }
        let narrow = match (input, out) {
            (Some((Scalar::Float, 2)), Some((Scalar::Half, 2))) => true,
            (Some((Scalar::Half, 2)), Some((Scalar::Float, 2))) => false,
            _ => return Err("quant half unsupported F16 conversion type/width"),
        };
        conversions.insert(at, narrow);
    }
    if conversions.is_empty() {
        // Original independent software controls have no FConvert. The pass
        // is also idempotent for an actual transformed module/cache entry.
        if original_pack != 0 || original_unpack != 0 {
            if original_pack == 0 || original_unpack == 0 {
                return Err("quant half incomplete native pack/unpack pair");
            }
            return Ok(Some(retain_product_loops(
                words.to_vec(),
                name.as_deref() == Some(b"expert_quant_cohort"),
            )?));
        }
        return Ok(None);
    }
    if !conversions.values().any(|v| *v) || !conversions.values().any(|v| !*v) {
        return Err("quant half requires actual native pack and unpack conversions");
    }
    let mut bound = words[3];
    let mut next_id = || {
        let id = bound;
        bound = bound
            .checked_add(1)
            .ok_or("quant half result id overflow")?;
        Ok::<_, &'static str>(id)
    };
    let new_import = glsl.is_none();
    let glsl = glsl.map(Ok).unwrap_or_else(&mut next_id)?;
    let mut output = Vec::with_capacity(words.len() + conversions.len() * 7 + 6);
    output.extend_from_slice(&words[..5]);
    for &(at, n, op) in &instructions {
        let i = &words[at..at + n];
        if op == 14 && new_import {
            output.extend_from_slice(&[
                (6 << 16) | 11,
                glsl,
                0x4c534c47,
                0x6474732e,
                0x3035342e,
                0,
            ]);
        }
        if let Some(&narrow) = conversions.get(&at) {
            let temporary = next_id()?;
            if narrow {
                output.extend_from_slice(&[(6 << 16) | 12, uint, temporary, glsl, 58, i[3]]);
                output.extend_from_slice(&[(4 << 16) | 124, i[1], i[2], temporary]);
            } else {
                output.extend_from_slice(&[(4 << 16) | 124, uint, temporary, i[3]]);
                output.extend_from_slice(&[(6 << 16) | 12, i[1], i[2], glsl, 62, temporary]);
            }
        } else {
            output.extend_from_slice(i);
        }
    }
    output[3] = bound;
    Ok(Some(retain_product_loops(
        output,
        name.as_deref() == Some(b"expert_quant_cohort"),
    )?))
}
