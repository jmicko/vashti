//! Exact constant XOR1/2 participant lookup through native four-lane swaps.
//! Mesa lowers relative shuffles to indexed lookups, but preserves QuadSwap.
//! No reassociation, operation-count, ownership or fault-policy change.
use super::transient_io::fault;
use cubecl_core::server::ServerError;
use rspirv::{
    binary::Assemble,
    dr::{Instruction, Operand, load_words},
    spirv::{Capability, Op},
};
use std::collections::{HashMap, HashSet};

/// Vulkan core subgroup bits: COMPUTE stage=0x20, QUAD operation=0x80.
/// All-stage quads must be explicitly advertised before selecting this path.
pub fn native_compute_quads_supported(stages: u32, operations: u32, all_stages: bool) -> bool {
    stages & 0x20 != 0 && operations & 0x80 != 0 && all_stages
}

/// Transform only scalar F32, subgroup-scope, literal-u32 XOR1/2 operations.
/// Other types, dynamic masks/scopes and larger exchanges remain untouched.
/// Existing zero/one IDs are reused: no new IDs, bounds or arithmetic appear.
pub fn lower_native_subgroup_quads(words: &[u32]) -> Result<Vec<u32>, ServerError> {
    let mut m = load_words(words).map_err(|_| fault("native quad SPIRV parse"))?;
    if m.header.is_none() {
        return Err(fault("native quad SPIRV header"));
    }
    let mut f32_types = HashSet::new();
    let mut u32_types = HashSet::new();
    for i in &m.types_global_values {
        if i.class.opcode == Op::TypeFloat && i.operands == [Operand::LiteralBit32(32)] {
            f32_types.extend(i.result_id);
        }
        if i.class.opcode == Op::TypeInt
            && i.operands == [Operand::LiteralBit32(32), Operand::LiteralBit32(0)]
        {
            u32_types.extend(i.result_id);
        }
    }
    let mut constants = HashMap::new();
    let mut by_value = HashMap::new();
    for i in &m.types_global_values {
        if i.class.opcode == Op::Constant && i.result_type.is_some_and(|t| u32_types.contains(&t)) {
            if let (Some(id), Some(ty), [Operand::LiteralBit32(value)]) =
                (i.result_id, i.result_type, i.operands.as_slice())
            {
                constants.insert(id, (ty, *value));
                by_value.entry((ty, *value)).or_insert(id);
            }
        }
    }
    let mut changed = false;
    for i in m
        .functions
        .iter_mut()
        .flat_map(|f| &mut f.blocks)
        .flat_map(|b| &mut b.instructions)
    {
        if i.class.opcode != Op::GroupNonUniformShuffleXor
            || !i.result_type.is_some_and(|t| f32_types.contains(&t))
        {
            continue;
        }
        let [
            Operand::IdRef(scope),
            Operand::IdRef(value),
            Operand::IdRef(mask),
        ] = i.operands.as_slice()
        else {
            continue;
        };
        if !constants.get(scope).is_some_and(|(_, v)| *v == 3) {
            continue;
        }
        let Some(&(ty, mask)) = constants.get(mask) else {
            continue;
        };
        let direction = match mask {
            1 => 0,
            2 => 1,
            _ => continue,
        };
        let Some(&direction_id) = by_value.get(&(ty, direction)) else {
            continue;
        };
        *i = Instruction::new(
            Op::GroupNonUniformQuadSwap,
            i.result_type,
            i.result_id,
            vec![
                Operand::IdRef(*scope),
                Operand::IdRef(*value),
                Operand::IdRef(direction_id),
            ],
        );
        changed = true;
    }
    if !changed {
        return Ok(words.to_vec());
    }
    if !m
        .capabilities
        .iter()
        .any(|i| i.operands == [Operand::Capability(Capability::GroupNonUniformQuad)])
    {
        m.capabilities.push(Instruction::new(
            Op::Capability,
            None,
            None,
            vec![Operand::Capability(Capability::GroupNonUniformQuad)],
        ));
    }
    let mut result = m.assemble();
    result[..5].copy_from_slice(&words[..5]);
    Ok(result)
}
