use super::*;
pub(in crate::strata_matrix) fn validate_program(
    words: &[u32],
    expert: bool,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    use std::collections::{BTreeMap, BTreeSet};
    let (mut parents, mut classes, mut bindings, mut defs) = (
        BTreeMap::new(),
        BTreeMap::new(),
        BTreeMap::new(),
        BTreeMap::<u32, Vec<u32>>::new(),
    );
    let (mut blocks, mut loops) = (BTreeMap::<u32, Vec<(usize, u32)>>::new(), Vec::new());
    let (mut loads, mut stores, mut products) = (Vec::new(), Vec::new(), Vec::new());
    let mut matrix_uses = BTreeMap::new();
    let mut operand_roles = Vec::new();
    let mut store_layouts = Vec::new();
    let (mut at, mut block, mut wg, mut barriers, mut atomics) = (5, None, false, 0, 0);
    while at < words.len() {
        let n = (words[at] >> 16) as usize;
        let op = words[at] & 65535;
        if n == 0 || at + n > words.len() {
            return Err("balanced wide32 opcode extent".into());
        }
        match op {
            16 if n == 6 && words[at + 2] == 17 => wg = words[at + 3..at + 6] == [512, 1, 1],
            71 if n == 4 && words[at + 2] == 33 => {
                bindings.insert(words[at + 1], words[at + 3]);
            }
            59 => {
                classes.insert(words[at + 2], words[at + 3]);
            }
            65 | 66 | 67 | 83 | 124 => {
                parents.insert(words[at + 2], words[at + 3]);
            }
            248 => {
                block = Some(words[at + 1]);
                blocks.insert(words[at + 1], Vec::new());
            }
            246 => loops.push((block.ok_or("balanced wide32 loop header")?, words[at + 1])),
            4456 => {
                matrix_uses.insert(words[at + 1], words[at + 6]);
            }
            4457 => {
                operand_roles.push((words[at + 1], words[at + 3], words[at + 4]));
                loads.push((
                    block.ok_or("balanced wide32 load block")?,
                    at,
                    words[at + 3],
                    words[at + 5],
                ));
            }
            4458 => {
                stores.push(words[at + 1]);
                store_layouts.push(words[at + 3]);
            }
            227 => {
                defs.insert(words[at + 2], vec![words[at + 3]]);
                atomics += 1;
            }
            4459 => products.push((block.ok_or("balanced wide32 product block")?, at)),
            224 => barriers += 1,
            227..=242 => atomics += 1,
            _ => {}
        }
        // Only value-producing instructions relevant to authentic dynamic strides.
        if matches!(
            op,
            43 | 164
                ..=191
                    | 227
                    | 334
                    | 61
                    | 65
                    | 66
                    | 67
                    | 79
                    | 80
                    | 81
                    | 83
                    | 113
                    | 114
                    | 124
                    | 128
                    | 130
                    | 132
                    | 134
                    | 135
                    | 194
                    | 196
                    | 197
                    | 198
                    | 199
                    | 245
        ) {
            defs.insert(words[at + 2], words[at + 3..at + n].to_vec());
        }
        if let Some(b) = block {
            blocks.get_mut(&b).unwrap().push((at, op));
        }
        at += n;
    }
    if !wg
        || (!expert && (barriers != 0 || atomics != 0))
        || loads.len() != if expert { 24 } else { 15 }
        || stores.len() != if expert { 8 } else { 4 }
        || products.len() != if expert { 12 } else { 8 }
        || (!expert && classes.values().any(|&c| c == 4))
    {
        return Err(format!("balanced wide32 actual module WG={wg} barriers={barriers} atomics={atomics} loads={} stores={} products={}",loads.len(),stores.len(),products.len()).into());
    }
    let root = |mut v: u32| -> std::result::Result<u32, Box<dyn std::error::Error>> {
        let mut seen = BTreeSet::new();
        while let Some(&p) = parents.get(&v) {
            if !seen.insert(v) {
                return Err("balanced wide32 pointer cycle".into());
            }
            v = p;
        }
        Ok(v)
    };
    for (ty, ptr, layout) in operand_roles {
        let binding = bindings.get(&root(ptr)?).copied();
        let use_id = matrix_uses.get(&ty).ok_or("transpose actual CMMA type")?;
        let actual_use = defs.get(use_id).and_then(|v| v.first()).copied();
        let actual_layout = defs.get(&layout).and_then(|v| v.first()).copied();
        if !matches!(
            (binding, actual_use, actual_layout),
            (Some(0), Some(1), Some(1)) | (Some(1), Some(0), Some(0))
        ) {
            return Err("transpose actual weight-A rowmajor/input-B colmajor operands".into());
        }
    }
    if store_layouts
        .iter()
        .any(|id| defs.get(id).and_then(|v| v.first()) != Some(&1))
    {
        return Err("transpose actual column-major output/fallback stores".into());
    }
    let mut input_loads = 0;
    let mut weight_loads = 0;
    for &(_, _, ptr, stride) in &loads {
        let r = root(ptr)?;
        if classes.get(&r) != Some(&12) {
            return Err("balanced wide32 actual StorageBuffer operand".into());
        }
        match bindings.get(&r) {
            Some(0) => input_loads += 1,
            Some(1) => weight_loads += 1,
            _ => return Err("balanced wide32 original input/weight binding".into()),
        }
        let (mut seen, mut todo, mut uniform) = (BTreeSet::new(), vec![stride], false);
        while let Some(v) = todo.pop() {
            if !seen.insert(v) {
                continue;
            }
            if matches!(classes.get(&v), Some(2 | 12))
                && bindings.get(&v) == Some(&(if expert { 6 } else { 3 }))
            {
                uniform = true;
            }
            if let Some(d) = defs.get(&v) {
                todo.extend(
                    d.iter()
                        .copied()
                        .filter(|v| defs.contains_key(v) || classes.contains_key(v)),
                );
            }
        }
        if !uniform {
            return Err("balanced wide32 stride is not authentic dynamic scalar INFO".into());
        }
    }
    if input_loads != if expert { 6 } else { 3 } || weight_loads != if expert { 18 } else { 12 } {
        return Err("balanced wide32 original weight-A/input-B fragment populations".into());
    }
    let (mut global_stores, mut fallback_stores) = (0, 0);
    for ptr in stores {
        let r = root(ptr)?;
        if classes.get(&r) == Some(&12) && bindings.get(&r) == Some(&2) {
            global_stores += 1;
        } else if expert && classes.get(&r) == Some(&4) {
            fallback_stores += 1;
        } else {
            return Err("balanced wide32 original global output or exact fallback owner".into());
        }
    }
    if global_stores != if expert { 6 } else { 4 } || fallback_stores != if expert { 2 } else { 0 }
    {
        return Err("balanced wide32 exact publication population".into());
    }
    let mut edges = BTreeMap::new();
    for (&b, ins) in &blocks {
        let &(a, op) = ins.last().ok_or("balanced wide32 block terminator")?;
        let n = (words[a] >> 16) as usize;
        edges.insert(
            b,
            match op {
                249 => vec![words[a + 1]],
                250 => vec![words[a + 2], words[a + 3]],
                251 => std::iter::once(words[a + 2])
                    .chain((a + 4..a + n).step_by(2).map(|i| words[i]))
                    .collect(),
                _ => Vec::new(),
            },
        );
    }
    let mut candidates = Vec::new();
    for (header, merge) in loops {
        let (mut region, mut todo) = (BTreeSet::new(), vec![header]);
        while let Some(b) = todo.pop() {
            if b == merge || !region.insert(b) {
                continue;
            }
            todo.extend(edges.get(&b).ok_or("wide32 CFG edge")?.iter().copied());
        }
        let count = products.iter().filter(|p| region.contains(&p.0)).count();
        if count > 0 {
            candidates.push((header, region, count));
        }
    }
    let mut innermost = Vec::new();
    for (i, c) in candidates.iter().enumerate() {
        if !candidates
            .iter()
            .enumerate()
            .any(|(j, d)| i != j && d.1.is_subset(&c.1) && d.1 != c.1)
        {
            innermost.push(c);
        }
    }
    let counts: BTreeSet<usize> = innermost.iter().map(|c| c.2).collect();
    if counts
        != if expert {
            BTreeSet::from([4, 8])
        } else {
            BTreeSet::from([8])
        }
        || innermost.len() != if expert { 2 } else { 1 }
    {
        return Err("wide32 genuine raw/checked product-loop partition".into());
    }
    let normal = innermost.iter().find(|c| c.2 == 8).unwrap();
    let normal_loads = loads
        .iter()
        .filter(|p| normal.1.contains(&p.0))
        .collect::<Vec<_>>();
    if normal_loads.len() != 10 {
        return Err("balanced wide32 two genuine5-operand banks in K loop".into());
    }
    if normal_loads
        .iter()
        .filter(|p| bindings.get(&root(p.2).unwrap()) == Some(&0))
        .count()
        != 2
    {
        return Err("balanced wide32 authentic two input-B loop operands".into());
    }
    // SPIR-V products have a common matrix type and unique result IDs; follow
    // actual c operand rather than trusting source names or physical code order.
    let ps = products
        .iter()
        .filter(|p| normal.1.contains(&p.0))
        .map(|p| p.1)
        .collect::<Vec<_>>();
    let first = ps[..4]
        .iter()
        .map(|&a| words[a + 2])
        .collect::<BTreeSet<_>>();
    let second = ps[4..]
        .iter()
        .map(|&a| words[a + 5])
        .collect::<BTreeSet<_>>();
    if first.len() != 4 || second != first {
        return Err("balanced wide32 four exact chronological accumulator chains".into());
    }
    let mut in_loop = BTreeMap::new();
    for &(block, at, _, _) in &loads {
        in_loop.insert(words[at + 2], normal.1.contains(&block));
    }
    // Bank1's four A and one B are loaded this iteration. Bank0 has loop-carried
    // phi operands, and cannot be replaced by a second use of bank1.
    for &at in &ps[4..] {
        if in_loop.get(&words[at + 3]) != Some(&true) || in_loop.get(&words[at + 4]) != Some(&true)
        {
            return Err("balanced wide32 actual current bank1 product operands".into());
        }
    }
    for &at in &ps[..4] {
        if in_loop.contains_key(&words[at + 3]) || in_loop.contains_key(&words[at + 4]) {
            return Err("balanced wide32 actual carried bank0 operands".into());
        }
    }
    if expert && cfg!(feature = "strata-wide32-local-routes") {
        // Invoke the production classifier on the actual final module. A name,
        // branch prediction, source hint or expected fault value is no proof.
        if cubecl_wgpu::native_strata_atomic_or_descriptor_mask(words, 0x3f) != 1 << 5 {
            return Err("local32 original fault descriptor is not proven AtomicOr-only".into());
        }
        let checked = innermost.iter().find(|c| c.2 == 4).unwrap();
        let reachable = |start: u32, stop: u32| -> BTreeSet<u32> {
            let (mut seen, mut todo) = (BTreeSet::new(), vec![start]);
            while let Some(b) = todo.pop() {
                if b == stop || !seen.insert(b) {
                    continue;
                }
                if let Some(next) = edges.get(&b) {
                    todo.extend(next.iter().copied());
                }
            }
            seen
        };
        let mut guarded = false;
        for ins in blocks.values() {
            let Some(&(branch, 250)) = ins.last() else {
                continue;
            };
            let Some(&(merge, _)) = ins.iter().rev().find(|(_, op)| *op == 247) else {
                continue;
            };
            let condition = words[branch + 1];
            // GroupNonUniformAll must be the condition actually separating the
            // eight-product and four-product loops, not an unrelated header All.
            let Some(&all_at) = ins
                .iter()
                .find(|(at, op)| *op == 334 && words[*at + 2] == condition)
                .map(|(at, _)| at)
            else {
                continue;
            };
            let yes = reachable(words[branch + 2], words[merge + 1]);
            let no = reachable(words[branch + 3], words[merge + 1]);
            if !(yes.contains(&normal.0)
                && !yes.contains(&checked.0)
                && no.contains(&checked.0)
                && !no.contains(&normal.0))
            {
                continue;
            }
            let (mut seen, mut todo, mut map_reads, mut comparisons) = (
                BTreeSet::new(),
                vec![words[all_at + 2]],
                BTreeSet::new(),
                0usize,
            );
            while let Some(v) = todo.pop() {
                if !seen.insert(v) {
                    continue;
                }
                if let Some(d) = defs.get(&v) {
                    // Locate the real definition, rather than guessing an ID's
                    // semantic role from a source name or integer value.
                    let mut at = 5;
                    while at < words.len() {
                        let n = (words[at] >> 16) as usize;
                        let op = words[at] & 65535;
                        if n >= 4 && words[at + 2] == v {
                            if op == 176 {
                                comparisons += 1;
                            }
                            if op == 61 && bindings.get(&root(words[at + 3])?) == Some(&3) {
                                map_reads.insert(v);
                            }
                        }
                        at += n;
                    }
                    todo.extend(d.iter().copied().filter(|v| defs.contains_key(v)));
                }
            }
            guarded |= map_reads.len() >= 2 && comparisons >= 1;
        }
        if !guarded {
            return Err("local32 actual subgroup-All two-route guard and raw/checked CFG".into());
        }
    } else if expert {
        if cubecl_wgpu::native_strata_atomic_or_descriptor_mask(words, 0x3f) != 0 {
            return Err(
                "wide32 preceding fault-load descriptor unexpectedly classified AtomicOr-only"
                    .into(),
            );
        }
        let mut fault_reads = BTreeSet::new();
        let mut all_values = BTreeSet::new();
        let mut branches = BTreeSet::new();
        let mut at = 5;
        while at < words.len() {
            let n = (words[at] >> 16) as usize;
            if words[at] & 65535 == 227 && bindings.get(&root(words[at + 3])?) == Some(&5) {
                fault_reads.insert(words[at + 2]);
            }
            if words[at] & 65535 == 334 {
                all_values.insert(words[at + 2]);
            }
            if words[at] & 65535 == 250 {
                branches.insert(words[at + 1]);
            }
            at += n;
        }
        let mut guarded = false;
        for all in all_values.intersection(&branches) {
            let mut seen = BTreeSet::new();
            let mut todo = vec![*all];
            while let Some(v) = todo.pop() {
                if !seen.insert(v) {
                    continue;
                }
                if fault_reads.contains(&v) {
                    guarded = true;
                }
                if let Some(d) = defs.get(&v) {
                    todo.extend(d.iter().copied().filter(|v| defs.contains_key(v)));
                }
            }
        }
        if !guarded {
            return Err("wide32 subgroup-uniform original sticky-fault branch".into());
        }
    }
    Ok(())
}
pub(in crate::strata_matrix) fn validate_sources()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    use cubecl::wgpu::{AutoCompiler, AutoRepresentation};
    use cubecl::{Compiler, ExecutionMode, WgpuCompilationOptions};
    use std::io::Write;
    for mask in 0..8u8 {
        for (name, definition) in [
            ("fp16", balanced_wide32_definition::<f16>(false)),
            ("bf16", balanced_wide32_definition::<bf16>(false)),
            ("gr_bf16", balanced_wide32_definition::<bf16>(true)),
        ] {
            let mut options = WgpuCompilationOptions {
                supports_vulkan: true,
                ..Default::default()
            };
            options.vulkan.max_spirv_version = (1, 6);
            options.vulkan.supports_uniform_standard_layout = mask & 1 != 0;
            options.vulkan.supports_uniform_unsized_array = mask & 2 != 0;
            options.vulkan.supports_explicit_smem = mask & 4 != 0;
            let AutoRepresentation::SpirV(k) = Compiler::compile(
                &mut AutoCompiler::SpirV(Default::default()),
                definition,
                &options,
                ExecutionMode::Unchecked,
                cubecl::ir::StorageType::Scalar(ElemType::UInt(cubecl::ir::UIntKind::U32)),
            )?
            else {
                return Err("balanced wide32 actual SPIRV".into());
            };
            if k.shared_size != 0 {
                return Err("balanced wide32 actual zero shared bytes".into());
            }
            let words = wgpu::hal::vulkan::kuno_native_workgroup_layout(&k.assembled_module)?;
            let words = wgpu::hal::vulkan::kuno_cooperative_fp16_preserve(&words)?.unwrap_or(words);
            if wgpu::hal::vulkan::kuno_cooperative_subgroup(&words) != Some(16) {
                return Err("balanced wide32 actual SG16".into());
            }
            validate_program(&words, false)?;
            #[cfg(test)]
            validate_negatives(&words, false)?;

            let mut cmd = std::process::Command::new("spirv-val");
            cmd.args(["--target-env", "vulkan1.3"]);
            if mask & 1 != 0 {
                cmd.arg("--uniform-buffer-standard-layout");
            }
            let mut child = cmd.arg("-").stdin(std::process::Stdio::piped()).spawn()?;
            child
                .stdin
                .take()
                .ok_or("balanced wide32 validator stdin")?
                .write_all(
                    &words
                        .iter()
                        .flat_map(|w| w.to_le_bytes())
                        .collect::<Vec<_>>(),
                )?;
            if !child.wait()?.success() {
                return Err("balanced wide32 SPIRV validation".into());
            }
            println!(
                "strata_balanced_wide32_shader_accepted dtype={name} layout_mask={mask} shared_bytes=0 workgroup=512 subgroup=16 m=128 n=128 k=32 products=8 global_operand_loads=15 global_stores=4 whole_workgroup_handoffs=0"
            );
        }
    }
    Ok(())
}

#[cfg(test)]
pub(in crate::strata_matrix) fn validate_negatives(
    words: &[u32],
    expert: bool,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let (mut at, mut changes, mut zero) = (5, Vec::new(), None);
    while at < words.len() {
        let n = (words[at] >> 16) as usize;
        let op = words[at] & 65535;
        if op == 43 && n == 4 && words[at + 3] == 0 {
            zero = Some(words[at + 2]);
        }
        if op == 16 && n == 6 && words[at + 2] == 17 {
            changes.push((at + 3, 1024));
        }
        if op == 71 && n == 4 && words[at + 2] == 33 && words[at + 3] == 0 {
            changes.push((at + 3, 9));
        }
        if op == 71 && n == 4 && words[at + 2] == 33 && words[at + 3] == if expert { 6 } else { 3 }
        {
            changes.push((at + 3, 10));
        }

        at += n;
    }
    at = 5;
    while at < words.len() {
        let n = (words[at] >> 16) as usize;
        if words[at] & 65535 == 4458 {
            changes.push((at + 3, zero.ok_or("wide32 actual zero constant")?));
            break;
        }
        at += n;
    }
    if changes.len() < 4 {
        return Err("wide32 actual negative inventory".into());
    }
    if expert {
        let mut bad = words.to_vec();
        let mut at = 5;
        while at < bad.len() {
            let n = (bad[at] >> 16) as usize;
            if bad[at] & 65535 == 334 {
                bad[at] = (n as u32) << 16 | 335;
            }
            at += n;
        }
        if validate_program(&bad, true).is_ok() {
            return Err("wide32 accepted non-all fault selection".into());
        }
    }
    if expert && cfg!(feature = "strata-wide32-local-routes") {
        let mut bad = words.to_vec();
        let mut at = 5;
        let mut changed = false;
        while at < bad.len() {
            let n = (bad[at] >> 16) as usize;
            if bad[at] & 65535 == 241 && n == 7 {
                bad[at] = (6u32 << 16) | 227;
                bad.remove(at + 6);
                changed = true;
                break;
            }
            at += n;
        }
        if !changed
            || cubecl_wgpu::native_strata_atomic_or_descriptor_mask(&bad, 0x3f) != 0
            || validate_program(&bad, true).is_ok()
        {
            return Err("local32 accepted actual fault read in an AtomicOr-only module".into());
        }
        let mut bad = words.to_vec();
        let mut at = 5;
        let mut changed = false;
        while at < bad.len() {
            let n = (bad[at] >> 16) as usize;
            if bad[at] & 65535 == 71 && n == 4 && bad[at + 2] == 33 && bad[at + 3] == 3 {
                bad[at + 3] = 9;
                changed = true;
            }
            at += n;
        }
        if !changed || validate_program(&bad, true).is_ok() {
            return Err("local32 accepted wrong actual route-map binding".into());
        }
    }
    for (i, value) in changes {
        let mut bad = words.to_vec();
        bad[i] = value;
        if validate_program(&bad, expert).is_ok() {
            return Err("wide32 accepted wrong grid/binding/INFO/store/fault-uniformity".into());
        }
    }
    Ok(())
}
