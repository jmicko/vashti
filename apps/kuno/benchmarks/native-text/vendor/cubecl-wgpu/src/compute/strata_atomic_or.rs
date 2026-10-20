//! Conservative proof over the ACTUAL final SPIR-V, never shader names/hints.
//! Only unused-result unsigned32 AtomicOr may touch a classified descriptor.
//! Any unsupported pointer use, escape, read, store or used result falls back.
use rspirv::spirv::{Decoration, Op, StorageClass};
const WORDS: usize = 262144;
const POINTERS: usize = 256;
fn each(words: &[u32], mut visit: impl FnMut(usize, &[u32]) -> bool) -> bool {
    if words.len() < 5 || words.len() > WORDS || words[0] != 0x07230203 || words[3] > WORDS as u32 {
        return false;
    }
    let mut at = 5;
    let mut function_open = false;
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        if count == 0 || at.checked_add(count).is_none_or(|end| end > words.len()) {
            return false;
        }
        let instruction = &words[at..at + count];
        if op(instruction, Op::Function) {
            if function_open {
                return false;
            }
            function_open = true;
        }
        if op(instruction, Op::FunctionEnd) {
            if !function_open {
                return false;
            }
            function_open = false;
        }
        if !visit(at, instruction) {
            return false;
        }
        at += count;
    }
    !function_open
}
fn op(i: &[u32], code: Op) -> bool {
    i[0] & 65535 == code as u32
}
fn binding(words: &[u32], variable: u32) -> Option<u32> {
    let mut set = None;
    let mut binding = None;
    let mut valid = true;
    if !each(words, |_, i| {
        if op(i, Op::Decorate) && i.len() >= 4 && i[1] == variable {
            let field = if i[2] == Decoration::DescriptorSet as u32 {
                Some(&mut set)
            } else if i[2] == Decoration::Binding as u32 {
                Some(&mut binding)
            } else {
                None
            };
            if let Some(field) = field {
                if field.replace(i[3]).is_some() {
                    valid = false;
                }
            }
        }
        true
    }) || !valid
        || set != Some(0)
    {
        return None;
    }
    binding.filter(|b| *b < 32)
}
fn u32_type(words: &[u32], ty: u32) -> bool {
    let mut found = false;
    each(words, |_, i| {
        if op(i, Op::TypeInt) && i.len() == 4 && i[1] == ty {
            found = i[2] == 32 && i[3] == 0;
        }
        true
    }) && found
}
fn int32_type(words: &[u32], ty: u32) -> bool {
    let mut found = false;
    each(words, |_, i| {
        if op(i, Op::TypeInt) && i.len() == 4 && i[1] == ty {
            found = i[2] == 32 && i[3] <= 1;
        }
        true
    }) && found
}
fn device_scope(words: &[u32], id: u32) -> bool {
    let mut found = false;
    each(words, |_, i| {
        if op(i, Op::Constant) && i.len() == 4 && i[2] == id {
            found = i[3] == rspirv::spirv::Scope::Device as u32 && int32_type(words, i[1]);
        }
        true
    }) && found
}
fn result_used(words: &[u32], id: u32, definition: usize) -> bool {
    let mut body = false;
    let mut used = false;
    if !each(words, |at, i| {
        if op(i, Op::Function) {
            body = true;
        }
        if body && at != definition && !op(i, Op::Line) && !op(i, Op::NoLine) {
            // A literal collision only declines the optimization; it can never
            // establish an unused result. Ignore ExtInst's literal opcode.
            let refs = if op(i, Op::ExtInst) && i.len() >= 5 {
                &i[5..]
            } else {
                &i[1..]
            };
            used |= refs.contains(&id);
        }
        if op(i, Op::FunctionEnd) {
            body = false;
        }
        !used
    }) {
        return true;
    }
    used
}
fn only_or(words: &[u32], variable: u32) -> bool {
    let mut pointers = [0u32; POINTERS];
    pointers[0] = variable;
    let mut count = 1;
    let mut next = 0;
    let mut seen = false;
    while next < count {
        let pointer = pointers[next];
        next += 1;
        let mut body = false;
        let mut good = true;
        if !each(words, |at, i| {
            if op(i, Op::Function) {
                body = true;
            }
            if body {
                if op(i, Op::AccessChain)
                    || op(i, Op::InBoundsAccessChain)
                    || op(i, Op::PtrAccessChain)
                    || op(i, Op::InBoundsPtrAccessChain)
                {
                    if i.len() < 4 {
                        good = false;
                    } else if i[3] == pointer {
                        if i[4..].contains(&pointer) {
                            good = false;
                        } else if !pointers[..count].contains(&i[2]) {
                            if count == POINTERS {
                                good = false;
                            } else {
                                pointers[count] = i[2];
                                count += 1;
                            }
                        }
                    } else if i[4..].contains(&pointer) {
                        good = false;
                    }
                } else if op(i, Op::CopyObject) {
                    if i.len() != 4 {
                        good = false;
                    } else if i[3] == pointer && !pointers[..count].contains(&i[2]) {
                        if count == POINTERS {
                            good = false;
                        } else {
                            pointers[count] = i[2];
                            count += 1;
                        }
                    }
                } else if op(i, Op::ArrayLength) {
                    if i.len() != 5 {
                        good = false;
                    } // metadata query only
                } else if op(i, Op::AtomicOr) {
                    if i.len() != 7 {
                        good = false;
                    } else if i[3] == pointer {
                        if i[4..].contains(&pointer)
                            || !device_scope(words, i[4])
                            || !u32_type(words, i[1])
                            || result_used(words, i[2], at)
                        {
                            good = false;
                        } else {
                            seen = true;
                        }
                    } else if i[3..].contains(&pointer) {
                        good = false;
                    }
                } else if op(i, Op::Load) {
                    if i.len() < 4 || i[3] == pointer {
                        good = false;
                    }
                } else if op(i, Op::ExtInst) {
                    if i.len() < 5 || i[5..].contains(&pointer) {
                        good = false;
                    }
                } else if !op(i, Op::Line) && !op(i, Op::NoLine) && i[1..].contains(&pointer) {
                    // Includes atomic load/store/add, normal stores/copies,
                    // pointer casts/phi/select, calls/returns and all escapes.
                    good = false;
                }
            }
            if op(i, Op::FunctionEnd) {
                body = false;
            }
            good
        }) {
            return false;
        }
    }
    seen
}
/// storage_mask comes from the ORIGINAL actual descriptor types, not a caller
/// claim. Ambiguity or complexity rejects proof, never rejects valid inference.
pub(super) fn mask(words: &[u32], storage_mask: u32) -> u32 {
    let mut result = 0;
    let mut occupied = 0;
    let mut duplicate = false;
    if !each(words, |_, i| {
        if op(i, Op::Variable)
            && i.len() >= 4
            && (i[3] == StorageClass::StorageBuffer as u32 || i[3] == StorageClass::Uniform as u32)
        {
            if let Some(binding) = binding(words, i[2]) {
                let bit = 1u32 << binding;
                if occupied & bit != 0 {
                    duplicate = true;
                    return false;
                }
                occupied |= bit;
                if storage_mask & bit != 0
                    && (i[3] == StorageClass::StorageBuffer as u32
                        || i[3] == StorageClass::Uniform as u32)
                    && only_or(words, i[2])
                {
                    result |= bit;
                }
            }
        }
        true
    }) || duplicate
    {
        return 0;
    }
    result
}
#[cfg(test)]
mod tests {
    use super::*;
    fn compile(body: &str) -> Vec<u32> {
        let text = format!(
            "@group(0) @binding(0) var<storage,read_write> flags:array<atomic<u32>>; @group(0) @binding(1) var<storage,read_write> output:array<u32>; @compute @workgroup_size(1) fn main() {{{body}}}"
        );
        let m = wgpu::naga::front::wgsl::parse_str(&text).unwrap();
        let info = wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&m)
        .unwrap();
        wgpu::naga::back::spv::write_vec(
            &m,
            &info,
            &Default::default(),
            Some(&wgpu::naga::back::spv::PipelineOptions {
                shader_stage: wgpu::naga::ShaderStage::Compute,
                entry_point: "main".into(),
            }),
        )
        .unwrap()
    }
    #[test]
    fn actual_lowered_or_allows_only_unused_results_and_exact_storage_descriptor() {
        for body in [
            "atomicOr(&flags[0], 128u);",
            "atomicOr(&flags[0],1u); atomicOr(&flags[1],2u);",
            "let unused=atomicOr(&flags[0],1u);",
        ] {
            let words = compile(body);
            assert_eq!(mask(&words, 3), 1, "{body}");
            assert_eq!(mask(&words, 0), 0);
        }
    }
    #[test]
    fn actual_reads_clear_other_atomics_and_consumed_results_decline_proof() {
        for body in [
            "output[0]=atomicLoad(&flags[0]); atomicOr(&flags[0],1u);",
            "atomicStore(&flags[0],0u);",
            "atomicAdd(&flags[0],1u);",
            "let old=atomicOr(&flags[0],1u);output[0]=old;",
            "output[0]=1u;",
        ] {
            assert_eq!(mask(&compile(body), 3), 0, "{body}");
        }
    }
    #[test]
    fn malformed_truncated_duplicate_and_non_u32_modules_never_get_a_proof() {
        let words = compile("atomicOr(&flags[0],1u);");
        for n in 0..words.len() {
            assert_eq!(mask(&words[..n], 3), 0);
        } // bounded parser, no panic
        let mut bad = words.clone();
        bad[5] = 0;
        assert_eq!(mask(&bad, 3), 0);
        let mut duplicate = words.clone();
        let mut binding_instruction = None;
        each(&words, |at, i| {
            if op(i, Op::Decorate) && i.len() == 4 && i[2] == Decoration::Binding as u32 {
                binding_instruction = Some(words[at..at + 4].to_vec());
            }
            true
        });
        duplicate.splice(5..5, binding_instruction.unwrap());
        assert_eq!(mask(&duplicate, 3), 0);
        let mut scope = None;
        each(&words, |_, i| {
            if op(i, Op::AtomicOr) {
                scope = Some(i[4]);
            }
            true
        });
        let mut local_scope = words.clone();
        let mut at = 5;
        while at < local_scope.len() {
            let n = (local_scope[at] >> 16) as usize;
            if local_scope[at] & 65535 == Op::Constant as u32
                && n == 4
                && Some(local_scope[at + 2]) == scope
            {
                local_scope[at + 3] = rspirv::spirv::Scope::Workgroup as u32;
            }
            at += n;
        }
        assert_eq!(mask(&local_scope, 3), 0);
        let mut signed = words.clone();
        let mut at = 5;
        while at < signed.len() {
            let n = (signed[at] >> 16) as usize;
            if signed[at] & 65535 == Op::TypeInt as u32 && n == 4 && signed[at + 2] == 32 {
                signed[at + 3] = 1;
            }
            at += n;
        }
        assert_eq!(mask(&signed, 3), 0);
    }
}
