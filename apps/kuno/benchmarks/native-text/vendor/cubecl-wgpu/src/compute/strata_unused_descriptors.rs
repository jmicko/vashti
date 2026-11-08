//! Zero-allocation proof over the actual, already validated final SPIR-V.
//! No function use is allowed, even an unused pointer copy or metadata query.
//! Declared admission/signature accesses remain separate and unchanged.
use rspirv::spirv::{AddressingModel, Capability, Decoration, Op, StorageClass};
const WORDS: usize = 262144;
fn each(words: &[u32], mut visit: impl FnMut(&[u32], bool) -> bool) -> bool {
    if words.len() < 5
        || words.len() > WORDS
        || words[0] != 0x07230203
        || words[3] == 0
        || words[3] > WORDS as u32
    {
        return false;
    }
    let mut at = 5;
    let mut body = false;
    while at < words.len() {
        let n = (words[at] >> 16) as usize;
        if n == 0 || at.checked_add(n).is_none_or(|end| end > words.len()) {
            return false;
        }
        let i = &words[at..at + n];
        let Some(op) = Op::from_u32(i[0] & 65535) else {
            return false;
        };
        if op == Op::Function {
            if body {
                return false;
            }
            body = true;
        }
        if op == Op::FunctionEnd && !body {
            return false;
        }
        if !visit(i, body) {
            return false;
        }
        if op == Op::FunctionEnd {
            body = false;
        }
        at += n;
    }
    !body
}
fn code(i: &[u32], op: Op) -> bool {
    i[0] & 65535 == op as u32
}
fn descriptor(words: &[u32], id: u32) -> Option<u32> {
    let mut set = None;
    let mut binding = None;
    if !each(words, |i, _| {
        if code(i, Op::Decorate) && i.get(1) == Some(&id) {
            let field = match i.get(2).copied() {
                Some(v) if v == Decoration::DescriptorSet as u32 => Some(&mut set),
                Some(v) if v == Decoration::Binding as u32 => Some(&mut binding),
                _ => None,
            };
            if let Some(field) = field {
                if i.len() != 4 || field.replace(i[3]).is_some() {
                    return false;
                }
            }
        }
        true
    }) || set != Some(0)
    {
        return None;
    }
    binding.filter(|&b| b < 32)
}
fn storage_pointer(words: &[u32], ty: u32) -> bool {
    let mut pointee = None;
    if !each(words, |i, _| {
        if code(i, Op::TypePointer) && i.get(1) == Some(&ty) {
            if i.len() != 4
                || i[2] != StorageClass::StorageBuffer as u32
                || pointee.replace(i[3]).is_some()
            {
                return false;
            }
        }
        true
    }) {
        return false;
    }
    let Some(pointee) = pointee else {
        return false;
    };
    let mut structure = false;
    let mut block = false;
    each(words, |i, _| {
        if code(i, Op::TypeStruct) && i.get(1) == Some(&pointee) {
            structure = true;
        }
        if code(i, Op::Decorate)
            && i.get(1) == Some(&pointee)
            && i.get(2) == Some(&(Decoration::Block as u32))
        {
            if i.len() != 3 || block {
                return false;
            }
            block = true;
        }
        true
    }) && structure
        && block
}
fn unreferenced(words: &[u32], id: u32) -> bool {
    let mut definitions = 0;
    each(words, |i, body| {
        if body {
            return !i[1..].contains(&id);
        }
        if code(i, Op::Variable) && i.get(2) == Some(&id) {
            definitions += 1;
            return i.len() == 4 && i[3] == StorageClass::StorageBuffer as u32;
        }
        if (code(i, Op::Name) || code(i, Op::Decorate)) && i.get(1) == Some(&id) {
            return true;
        }
        if code(i, Op::EntryPoint) {
            return true;
        } // interfaces are not memory operations
        // Any unknown alias/escape outside functions, including an initializer,
        // declines. Literal-ID collisions only keep the conservative access.
        !i[1..].contains(&id)
    }) && definitions == 1
}
/// storage_mask is derived from the original compiled descriptor TYPES. It
/// cannot grant access, completion or ownership;0 means no optimization.
pub(super) fn mask(words: &[u32], storage_mask: u32) -> u32 {
    let mut models = 0;
    let mut entries = 0;
    if !each(words, |i, _| {
        if code(i, Op::MemoryModel) {
            models += 1;
            if i.len() != 3 || i[1] != AddressingModel::Logical as u32 {
                return false;
            }
        }
        if code(i, Op::EntryPoint) {
            entries += 1;
            if i.len() < 4 || i[1] != 5 {
                return false;
            }
        }
        if code(i, Op::Capability) {
            if i.len() != 2
                || matches!(i[1],v if v==Capability::PhysicalStorageBufferAddresses as u32||v==Capability::VariablePointers as u32||v==Capability::VariablePointersStorageBuffer as u32)
            {
                return false;
            }
        }
        true
    }) || models != 1
        || entries != 1
    {
        return 0;
    }
    let mut occupied = 0u32;
    let mut result = 0;
    if !each(words, |i, body| {
        if code(i, Op::Variable)
            && !body
            && i.len() >= 4
            && matches!(i[3],v if v==StorageClass::StorageBuffer as u32||v==StorageClass::Uniform as u32||v==StorageClass::UniformConstant as u32)
        {
            if i[2] == 0 || i[2] >= words[3] {
                return false;
            }
            let Some(b) = descriptor(words, i[2]) else {
                return false;
            };
            let bit = 1u32 << b;
            if occupied & bit != 0 {
                return false;
            }
            occupied |= bit;
            if storage_mask & bit != 0
                && i[3] == StorageClass::StorageBuffer as u32
                && storage_pointer(words, i[1])
                && unreferenced(words, i[2])
            {
                result |= bit;
            }
        }
        true
    }) {
        return 0;
    }
    result
}
pub(super) fn execution_access(
    declared: ash::vk::AccessFlags,
    index: usize,
    unused: u32,
) -> ash::vk::AccessFlags {
    if index < 32 && unused & (1u32 << index) != 0 {
        ash::vk::AccessFlags::empty()
    } else {
        declared
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn compile(body: &str) -> Vec<u32> {
        let s = format!(
            "@group(0) @binding(0) var<storage,read_write> output:array<u32>; @group(0) @binding(1) var<storage,read_write> flags:array<atomic<u32>>; @group(0) @binding(2) var<storage,read> spare:array<u32>; @compute @workgroup_size(1) fn main() {{output[0]=7u;{body}}}"
        );
        let m = wgpu::naga::front::wgsl::parse_str(&s).unwrap();
        let info = wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&m)
        .unwrap();
        let mut w = wgpu::naga::back::spv::write_vec(
            &m,
            &info,
            &Default::default(),
            Some(&wgpu::naga::back::spv::PipelineOptions {
                shader_stage: wgpu::naga::ShaderStage::Compute,
                entry_point: "main".into(),
            }),
        )
        .unwrap();
        // Naga elides unused globals. Add only genuine unused declarations using
        // its real Block/storage-pointer type, then validate the resulting module.
        // This is a CPU shader fixture, never production request/batch authority.
        let mut vars = Vec::new();
        each(&w, |i, _| {
            if code(i, Op::Variable) && i.len() == 4 && i[3] == 12 {
                vars.push((i[2], i[1]));
            }
            true
        });
        let ty = vars
            .iter()
            .find(|&&(id, _)| descriptor(&w, id) == Some(0))
            .unwrap()
            .1;
        for bind in [1u32, 2] {
            if vars.iter().any(|&(id, _)| descriptor(&w, id) == Some(bind)) {
                continue;
            }
            let id = w[3];
            w[3] += 1;
            let at = instruction(&w, Op::TypeVoid);
            w.splice(
                at..at,
                [
                    (4 << 16) | Op::Decorate as u32,
                    id,
                    Decoration::DescriptorSet as u32,
                    0,
                    (4 << 16) | Op::Decorate as u32,
                    id,
                    Decoration::Binding as u32,
                    bind,
                ],
            );
            let at = instruction(&w, Op::Function);
            w.splice(
                at..at,
                [
                    (4 << 16) | Op::Variable as u32,
                    ty,
                    id,
                    StorageClass::StorageBuffer as u32,
                ],
            );
            if w[1] >= 0x00010400 {
                let at = instruction(&w, Op::EntryPoint);
                let n = (w[at] >> 16) as usize;
                w[at] = (((n + 1) as u32) << 16) | Op::EntryPoint as u32;
                w.insert(at + n, id);
            }
        }
        use std::io::Write;
        let mut val = std::process::Command::new("spirv-val")
            .args(["--target-env", "vulkan1.3", "-"])
            .stdin(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        val.stdin
            .take()
            .unwrap()
            .write_all(&w.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<_>>())
            .unwrap();
        assert!(val.wait().unwrap().success());
        w
    }

    fn instruction(words: &[u32], op: Op) -> usize {
        let mut at = 5;
        loop {
            if code(&words[at..], op) {
                return at;
            }
            at += (words[at] >> 16) as usize;
            assert!(at < words.len());
        }
    }
    #[test]
    fn actual_unused_descriptors_do_not_classify_any_real_load_store_atomic_or_query() {
        assert_eq!(mask(&compile(""), 7), 6);
        assert_eq!(mask(&compile(""), 1), 0);
        for body in [
            "atomicOr(&flags[0],1u);",
            "output[0]=atomicLoad(&flags[0]);",
            "atomicStore(&flags[0],3u);",
            "output[0]=arrayLength(&flags);",
            "let p=&flags[0];atomicOr(p,1u);",
        ] {
            assert_eq!(mask(&compile(body), 7) & 2, 0, "{body}");
        }
        for body in [
            "output[0]=spare[0];",
            "output[0]=arrayLength(&spare);",
            "let p=&spare[0];output[0]=*p;",
        ] {
            assert_eq!(mask(&compile(body), 7) & 4, 0, "{body}");
        }
    }
    #[test]
    fn malformed_unknown_complex_or_physical_modules_never_grant_a_proof() {
        let w = compile("");
        for n in [0, 1, 4, 5, w.len() - 1] {
            assert_eq!(mask(&w[..n], 7), 0);
        }
        let mut b = w.clone();
        b[0] = 0;
        assert_eq!(mask(&b, 7), 0);
        let mut b = w.clone();
        b[3] = 0;
        assert_eq!(mask(&b, 7), 0);
        let mut b = w.clone();
        b[3] = WORDS as u32 + 1;
        assert_eq!(mask(&b, 7), 0);
        let mut b = w.clone();
        b[5] = 0;
        assert_eq!(mask(&b, 7), 0);
        let mut b = w.clone();
        b[5] = (1 << 16) | 65535;
        assert_eq!(mask(&b, 7), 0);
        let mut b = w.clone();
        let i = instruction(&b, Op::MemoryModel);
        b[i + 1] = AddressingModel::PhysicalStorageBuffer64 as u32;
        assert_eq!(mask(&b, 7), 0);
        let mut b = w.clone();
        let i = instruction(&b, Op::MemoryModel);
        let copy = b[i..i + 3].to_vec();
        b.splice(i..i, copy);
        assert_eq!(mask(&b, 7), 0);
        let mut b = w.clone();
        b.splice(
            5..5,
            [
                (2 << 16) | Op::Capability as u32,
                Capability::VariablePointers as u32,
            ],
        );
        assert_eq!(mask(&b, 7), 0);
        let mut b = w.clone();
        let i = instruction(&b, Op::Function);
        let n = (b[i] >> 16) as usize;
        let copy = b[i..i + n].to_vec();
        b.splice(i..i, copy);
        assert_eq!(mask(&b, 7), 0);
        let mut b = w.clone();
        let i = instruction(&b, Op::FunctionEnd);
        b.drain(i..i + 1);
        assert_eq!(mask(&b, 7), 0);
    }
    #[test]
    fn genuine_binding_set_type_identity_and_no_alias_or_initializer_are_required() {
        let w = compile("");
        let mut vars = Vec::new();
        each(&w, |i, _| {
            if code(i, Op::Variable) && i.len() >= 4 && i[3] == 12 {
                vars.push((i[2], i[1]));
            }
            true
        });
        let (id, ty) = vars
            .iter()
            .copied()
            .find(|&(id, _)| descriptor(&w, id) == Some(1))
            .unwrap();
        let mut b = w.clone();
        b.splice(
            5..5,
            [
                (4 << 16) | Op::Decorate as u32,
                id,
                Decoration::Binding as u32,
                1,
            ],
        );
        assert_eq!(mask(&b, 7), 0);
        let mut b = w.clone();
        let mut at = 5;
        while at < b.len() {
            let n = (b[at] >> 16) as usize;
            if code(&b[at..], Op::Decorate)
                && b[at + 1] == id
                && b[at + 2] == Decoration::DescriptorSet as u32
            {
                b[at + 3] = 1;
                break;
            }
            at += n;
        }
        assert_eq!(mask(&b, 7), 0);
        let mut b = w.clone();
        let mut at = 5;
        while at < b.len() {
            let n = (b[at] >> 16) as usize;
            if code(&b[at..], Op::TypePointer) && b[at + 1] == ty {
                b[at + 2] = StorageClass::Uniform as u32;
                break;
            }
            at += n;
        }
        assert_eq!(mask(&b, 7) & 2, 0);
        let mut b = w.clone();
        let i = instruction(&b, Op::Function);
        b.splice(i..i, [(4 << 16) | Op::CopyObject as u32, ty, w[3], id]);
        b[3] += 1;
        assert_eq!(mask(&b, 7) & 2, 0);
        let mut b = w.clone();
        let i = instruction(&b, Op::Function);
        let n = (b[i] >> 16) as usize;
        b.splice(
            i + n..i + n,
            [(4 << 16) | Op::CopyObject as u32, ty, w[3], id],
        );
        b[3] += 1;
        assert_eq!(mask(&b, 7) & 2, 0);
    }
    #[test]
    fn three_queue_plan_drops_only_unused_status_dependency_and_keeps_used_aliases_and_future_writers()
     {
        use super::super::strata_schedule::{Access, Planner};
        let rw = ash::vk::AccessFlags::SHADER_READ | ash::vk::AccessFlags::SHADER_WRITE;
        for alias in [false, true] {
            let mut p = Planner::new(4, false).unwrap();
            let status = || Access {
                buffer: 9,
                first: 0,
                bytes: 64,
                write: true,
            };
            p.push_accesses(false, true, false, &[status()]).unwrap();
            let gu = vec![
                status(),
                Access {
                    buffer: if alias { 9 } else { 11 },
                    first: 0,
                    bytes: 64,
                    write: false,
                },
            ];
            p.push_accesses_with_memory(false, false, false, &gu, |a| !a.write)
                .unwrap();
            p.push_accesses(
                true,
                false,
                false,
                &[Access {
                    buffer: 19,
                    first: 0,
                    bytes: 64,
                    write: true,
                }],
            )
            .unwrap();
            p.push_accesses(
                false,
                false,
                false,
                &[
                    status(),
                    Access {
                        buffer: 19,
                        first: 0,
                        bytes: 64,
                        write: false,
                    },
                ],
            )
            .unwrap();
            let p = p.finish().unwrap();
            let producer = &p.segments[p.node_segments[0]];
            let gu = &p.segments[p.node_segments[1]];
            assert_eq!(
                gu.waits[2] > 0,
                alias,
                "a used alias cannot disappear with the unused view"
            );
            let last = &p.segments[p.node_segments[3]];
            assert!(
                last.waits[2] >= producer.signal,
                "future real status writer still joins producer"
            );
            assert!(
                last.waits[1] > 0,
                "copy dependency still joins original third queue"
            );
        }
        for i in 0..40 {
            assert_eq!(execution_access(rw, i, 1 << 4).is_empty(), i == 4);
        }
        assert_eq!(execution_access(rw, 4, 0), rw);
    }
}
