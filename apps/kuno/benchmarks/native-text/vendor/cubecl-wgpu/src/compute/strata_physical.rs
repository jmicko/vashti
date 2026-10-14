//! Rust lowering of original packed expert reads to Vulkan physical storage
//! pointers, matching Strata's single resolved USM pointer per role/group.
//! Naga retains the complete logical shader for Wgpu initialization tracking;
//! only the native executor uses this transformed module. No weights change.
use super::transient_io::fault;
use cubecl_core::server::ServerError;
use rspirv::{
    binary::Assemble,
    dr::{Instruction, Operand, load_words},
    spirv::{AddressingModel, Capability, MemoryAccess, MemoryModel, Op, StorageClass},
};
use std::collections::HashMap;
fn ins(op: Op, ty: Option<u32>, id: Option<u32>, args: Vec<Operand>) -> Instruction {
    Instruction::new(op, ty, id, args)
}
fn id(o: &Operand) -> Option<u32> {
    if let Operand::IdRef(v) = o {
        Some(*v)
    } else {
        None
    }
}
fn fresh(bound: &mut u32) -> u32 {
    let v = *bound;
    *bound += 1;
    v
}
fn ptr_type(
    inner: u32,
    types: &mut HashMap<u32, u32>,
    out: &mut Vec<Instruction>,
    bound: &mut u32,
) -> u32 {
    if let Some(v) = types.get(&inner) {
        return *v;
    }
    let v = fresh(bound);
    out.push(ins(
        Op::TypePointer,
        None,
        Some(v),
        vec![
            Operand::StorageClass(StorageClass::PhysicalStorageBuffer),
            Operand::IdRef(inner),
        ],
    ));
    types.insert(inner, v);
    v
}
/// Same checked transformation used by the actual HAL shader constructor.
pub(crate) fn canonical_workgroup_layout(words: &[u32]) -> Result<Vec<u32>, ServerError> {
    wgpu::hal::vulkan::kuno_native_workgroup_layout(words).map_err(fault)
}

pub(crate) fn lower(words: &[u32]) -> Result<Vec<u32>, ServerError> {
    let mut m = load_words(words).map_err(|_| fault("physical expert SPIRV parse"))?;
    let mut bound = m
        .header
        .as_ref()
        .ok_or_else(|| fault("physical expert SPIRV header"))?
        .bound;
    let mut names = HashMap::<String, u32>::new();
    for i in &m.debug_names {
        if i.class.opcode == Op::Name {
            if let [Operand::IdRef(id), Operand::LiteralString(name)] = i.operands.as_slice() {
                names.insert(name.clone(), *id);
            }
        }
    }
    let mut pointer_inner = HashMap::<u32, u32>::new();
    let mut variables = HashMap::<u32, u32>::new();
    let mut scalar32 = None;
    let mut vec2 = None;
    for i in &m.types_global_values {
        match i.class.opcode {
            Op::TypePointer => {
                if let Some(inner) = i.operands.get(1).and_then(id) {
                    pointer_inner.insert(i.result_id.unwrap(), inner);
                }
            }
            Op::Variable => {
                variables.insert(i.result_id.unwrap(), i.result_type.unwrap());
            }
            Op::TypeInt => {
                if i.operands == [Operand::LiteralBit32(32), Operand::LiteralBit32(0)] {
                    scalar32 = i.result_id;
                }
            }
            _ => {}
        }
    }
    let scalar32 = scalar32.ok_or_else(|| fault("physical expert uint32 type"))?;
    for i in &m.types_global_values {
        if i.class.opcode == Op::TypeVector
            && i.operands == [Operand::IdRef(scalar32), Operand::LiteralBit32(2)]
        {
            vec2 = i.result_id;
        }
    }
    let vec2 = vec2.ok_or_else(|| fault("physical expert address vector"))?;
    let mut extra = Vec::new();
    let u64_ty = fresh(&mut bound);
    extra.push(ins(
        Op::TypeInt,
        None,
        Some(u64_ty),
        vec![Operand::LiteralBit32(64), Operand::LiteralBit32(0)],
    ));
    let shift = fresh(&mut bound);
    extra.push(ins(
        Op::Constant,
        Some(u64_ty),
        Some(shift),
        vec![Operand::LiteralBit64(32)],
    ));
    let mut physical_types = HashMap::new();
    let mut roots = HashMap::<u32, (u32, u32)>::new();
    for role in ["gate", "up", "down"] {
        let Some(&root) = names.get(&format!("{role}0")) else {
            continue;
        };
        let address = *names
            .get(&format!("strata_{role}_address"))
            .ok_or_else(|| fault("physical expert address variable"))?;
        let inner = *pointer_inner
            .get(
                variables
                    .get(&root)
                    .ok_or_else(|| fault("physical expert source variable"))?,
            )
            .ok_or_else(|| fault("physical expert source pointer"))?;
        roots.insert(
            root,
            (
                address,
                ptr_type(inner, &mut physical_types, &mut extra, &mut bound),
            ),
        );
    }
    if roots.is_empty() {
        return Err(fault("physical expert source roots missing"));
    }
    let mut physical_values = HashMap::<u32, u32>::new();
    for f in &mut m.functions {
        for block in &mut f.blocks {
            let original = std::mem::take(&mut block.instructions);
            let mut rewritten = Vec::new();
            for mut i in original {
                if [Op::AccessChain, Op::InBoundsAccessChain].contains(&i.class.opcode) {
                    let base = i
                        .operands
                        .first()
                        .and_then(id)
                        .ok_or_else(|| fault("physical expert access base"))?;
                    let mut physical = false;
                    if let Some(&(address, physical_root_ty)) = roots.get(&base) {
                        let pair = fresh(&mut bound);
                        let low = fresh(&mut bound);
                        let high = fresh(&mut bound);
                        let lo64 = fresh(&mut bound);
                        let hi64 = fresh(&mut bound);
                        let shifted = fresh(&mut bound);
                        let combined = fresh(&mut bound);
                        let pointer = fresh(&mut bound);
                        rewritten.extend([
                            ins(
                                Op::Load,
                                Some(vec2),
                                Some(pair),
                                vec![Operand::IdRef(address)],
                            ),
                            ins(
                                Op::CompositeExtract,
                                Some(scalar32),
                                Some(low),
                                vec![Operand::IdRef(pair), Operand::LiteralBit32(0)],
                            ),
                            ins(
                                Op::CompositeExtract,
                                Some(scalar32),
                                Some(high),
                                vec![Operand::IdRef(pair), Operand::LiteralBit32(1)],
                            ),
                            ins(
                                Op::UConvert,
                                Some(u64_ty),
                                Some(lo64),
                                vec![Operand::IdRef(low)],
                            ),
                            ins(
                                Op::UConvert,
                                Some(u64_ty),
                                Some(hi64),
                                vec![Operand::IdRef(high)],
                            ),
                            ins(
                                Op::ShiftLeftLogical,
                                Some(u64_ty),
                                Some(shifted),
                                vec![Operand::IdRef(hi64), Operand::IdRef(shift)],
                            ),
                            ins(
                                Op::BitwiseOr,
                                Some(u64_ty),
                                Some(combined),
                                vec![Operand::IdRef(lo64), Operand::IdRef(shifted)],
                            ),
                            ins(
                                Op::ConvertUToPtr,
                                Some(physical_root_ty),
                                Some(pointer),
                                vec![Operand::IdRef(combined)],
                            ),
                        ]);
                        i.operands[0] = Operand::IdRef(pointer);
                        physical = true;
                    } else if physical_values.contains_key(&base) {
                        physical = true;
                    }
                    if physical {
                        let inner = *pointer_inner
                            .get(
                                &i.result_type
                                    .ok_or_else(|| fault("physical expert access type"))?,
                            )
                            .ok_or_else(|| fault("physical expert access pointee"))?;
                        let ty = ptr_type(inner, &mut physical_types, &mut extra, &mut bound);
                        i.result_type = Some(ty);
                        physical_values.insert(i.result_id.unwrap(), inner);
                    }
                } else if i.class.opcode == Op::Load {
                    if i.operands
                        .first()
                        .and_then(id)
                        .is_some_and(|v| physical_values.contains_key(&v))
                    {
                        // Every expert root is at least16-byte aligned; scalar
                        // loads remain4-byte aligned and vec4 reads16-byte aligned.
                        let ty = i
                            .result_type
                            .ok_or_else(|| fault("physical expert load type"))?;
                        let align = if ty == scalar32 { 4 } else { 16 };
                        if i.operands.len() != 1 {
                            return Err(fault("physical expert unexpected load operands"));
                        }
                        i.operands.extend([
                            Operand::MemoryAccess(MemoryAccess::ALIGNED),
                            Operand::LiteralBit32(align),
                        ]);
                    }
                }
                rewritten.push(i);
            }
            block.instructions = rewritten;
        }
    }
    m.types_global_values.extend(extra);
    for capability in [
        Capability::Int64,
        Capability::PhysicalStorageBufferAddresses,
    ] {
        let operand = Operand::Capability(capability);
        if !m
            .capabilities
            .iter()
            .any(|i| i.operands == [operand.clone()])
        {
            m.capabilities
                .push(ins(Op::Capability, None, None, vec![operand]));
        }
    }
    m.extensions.push(ins(
        Op::Extension,
        None,
        None,
        vec![Operand::LiteralString(
            "SPV_KHR_physical_storage_buffer".into(),
        )],
    ));
    m.memory_model = Some(ins(
        Op::MemoryModel,
        None,
        None,
        vec![
            Operand::AddressingModel(AddressingModel::PhysicalStorageBuffer64),
            Operand::MemoryModel(MemoryModel::GLSL450),
        ],
    ));
    m.header.as_mut().unwrap().bound = bound;
    Ok(m.assemble())
}

/// CPU-only public compiler inspection, shared with actual native registration.
pub fn native_expert_shader_words(source: &str) -> Result<Vec<u32>, ServerError> {
    use wgpu::naga as n;
    let m = n::front::wgsl::parse_str(source).map_err(|_| fault("physical expert WGSL parse"))?;
    let info = n::valid::Validator::new(
        n::valid::ValidationFlags::all(),
        n::valid::Capabilities::all(),
    )
    .validate(&m)
    .map_err(|_| fault("physical expert WGSL validation"))?;
    for (_, root) in m.global_variables.iter() {
        if matches!(root.name.as_deref(), Some("gate0" | "up0" | "down0"))
            && root.space
                != (n::AddressSpace::Storage {
                    access: n::StorageAccess::LOAD,
                })
        {
            return Err(fault("physical expert roots must be readonly"));
        }
    }
    let words = n::back::spv::write_vec(
        &m,
        &info,
        &n::back::spv::Options {
            lang_version: (1, 3),
            flags: n::back::spv::WriterFlags::DEBUG,
            zero_initialize_workgroup_memory: n::back::spv::ZeroInitializeWorkgroupMemoryMode::None,
            bounds_check_policies: n::proc::BoundsCheckPolicies {
                index: n::proc::BoundsCheckPolicy::Unchecked,
                buffer: n::proc::BoundsCheckPolicy::Unchecked,
                image_load: n::proc::BoundsCheckPolicy::Unchecked,
                binding_array: n::proc::BoundsCheckPolicy::Unchecked,
            },
            ..Default::default()
        },
        Some(&n::back::spv::PipelineOptions {
            shader_stage: n::ShaderStage::Compute,
            entry_point: "main".into(),
        }),
    )
    .map_err(|_| fault("physical expert SPIRV emission"))?;
    lower(&canonical_workgroup_layout(&words)?)
}
