// Actual native module and CFG evidence for the direct-global operand banks.
fn validate_global_n64_program(
    words: &[u32],
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    validate_global_n64_pipeline(words, false)
}
#[cfg(feature = "strata-global-image-experts")]
pub(in crate::strata_matrix) fn validate_global_n64_expert_program(
    words: &[u32],
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    validate_global_n64_pipeline(words, true)
}
fn validate_global_n64_pipeline(
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
    let (mut at, mut block, mut wg, mut barriers, mut atomics) = (5, None, false, 0, 0);
    while at < words.len() {
        let n = (words[at] >> 16) as usize;
        let op = words[at] & 65535;
        if n == 0 || at + n > words.len() {
            return Err("global n64 opcode extent".into());
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
            246 => loops.push((block.ok_or("global n64 loop header")?, words[at + 1])),
            4457 => loads.push((
                block.ok_or("global n64 load block")?,
                at,
                words[at + 3],
                words[at + 5],
            )),
            4458 => stores.push(words[at + 1]),
            4459 => products.push((block.ok_or("global n64 product block")?, at)),
            224 => barriers += 1,
            227..=242 => atomics += 1,
            _ => {}
        }
        // Only value-producing instructions relevant to authentic dynamic strides.
        if matches!(
            op,
            43 | 61
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
        || loads.len() != 9
        || stores.len() != if expert { 4 } else { 2 }
        || products.len() != 4
        || (!expert && classes.values().any(|&c| c == 4))
    {
        return Err(format!("global n64 actual module WG={wg} barriers={barriers} atomics={atomics} loads={} stores={} products={}",loads.len(),stores.len(),products.len()).into());
    }
    let root = |mut v: u32| -> std::result::Result<u32, Box<dyn std::error::Error>> {
        let mut seen = BTreeSet::new();
        while let Some(&p) = parents.get(&v) {
            if !seen.insert(v) {
                return Err("global n64 pointer cycle".into());
            }
            v = p;
        }
        Ok(v)
    };
    let mut input_loads = 0;
    let mut weight_loads = 0;
    for &(_, _, ptr, stride) in &loads {
        let r = root(ptr)?;
        if classes.get(&r) != Some(&12) {
            return Err("global n64 actual StorageBuffer operand".into());
        }
        match bindings.get(&r) {
            Some(0) => input_loads += 1,
            Some(1) => weight_loads += 1,
            _ => return Err("global n64 original input/weight binding".into()),
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
            return Err("global n64 stride is not authentic dynamic scalar INFO".into());
        }
    }
    if input_loads != 6 || weight_loads != 3 {
        return Err("global n64 actual six A and three B loads".into());
    }
    let (mut global_stores, mut fallback_stores) = (0, 0);
    for ptr in stores {
        let r = root(ptr)?;
        if classes.get(&r) == Some(&12) && bindings.get(&r) == Some(&2) {
            global_stores += 1;
        } else if expert && classes.get(&r) == Some(&4) {
            fallback_stores += 1;
        } else {
            return Err("global n64 original global output or exact fallback owner".into());
        }
    }
    if global_stores != 2 || fallback_stores != if expert { 2 } else { 0 } {
        return Err("global n64 exact publication population".into());
    }
    let mut edges = BTreeMap::new();
    for (&b, ins) in &blocks {
        let &(a, op) = ins.last().ok_or("global n64 block terminator")?;
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
    let mut product_loops = Vec::new();
    for (header, merge) in loops {
        let (mut region, mut todo) = (BTreeSet::new(), vec![header]);
        while let Some(b) = todo.pop() {
            if b == merge || !region.insert(b) {
                continue;
            }
            todo.extend(edges.get(&b).ok_or("global n64 CFG edge")?.iter().copied());
        }
        if products.iter().any(|p| region.contains(&p.0)) {
            product_loops.push((header, region));
        }
    }
    if product_loops.len() != 1 {
        return Err("global n64 one complete K32 product loop".into());
    }
    let (header, region) = product_loops.pop().unwrap();
    if products.iter().any(|p| !region.contains(&p.0))
        || loads.iter().filter(|p| region.contains(&p.0)).count() != 6
    {
        return Err("global n64 actual initial3 plus in-loop6 operand loads".into());
    }
    if region
        .iter()
        .flat_map(|b| &blocks[b])
        .any(|(_, op)| *op == 224 || (227..=242).contains(op))
    {
        return Err("global n64 product loop must not contain publication barriers/atomics".into());
    }
    let reachable = |from: u32, to: u32| {
        let (mut todo, mut seen) = (vec![from], BTreeSet::new());
        while let Some(b) = todo.pop() {
            if !seen.insert(b) {
                continue;
            }
            if b == to {
                return true;
            }
            todo.extend(
                edges
                    .get(&b)
                    .into_iter()
                    .flatten()
                    .copied()
                    .filter(|n| *n != header && region.contains(n)),
            );
        }
        false
    };
    let before = |a: (u32, usize), b: (u32, usize)| {
        if a.0 == b.0 {
            a.1 < b.1
        } else {
            reachable(a.0, b.0)
        }
    };
    // First K16 quartet, then the next bank prefetch, then second quartet.
    products.sort_by_key(|p| p.1);
    let first = products[0];
    let middle = products[1];
    let second = products[2];
    let last = products[3];
    if products[..2].iter().any(|p| p.0 != first.0)
        || products[2..].iter().any(|p| p.0 != second.0)
        || !before(middle, second)
    {
        return Err("global n64 two chronological two-product banks".into());
    }
    let early = loads
        .iter()
        .filter(|p| region.contains(&p.0) && before((p.0, p.1), first))
        .count();
    let later = loads
        .iter()
        .filter(|p| {
            region.contains(&p.0) && before(middle, (p.0, p.1)) && before((p.0, p.1), second)
        })
        .count();
    if early != 3 || later != 3 || !before(first, last) {
        return Err(format!("global n64 actual prefetch chronology {early}/{later}").into());
    }
    if expert {
        println!(
            "strata_global_n64_expert_pipeline_accepted initial_global_loads=3 first_bank_prefetch=3 second_bank_prefetch=3 products=4 iteration_backedge_excluded=true original_fallback_publication=true"
        );
    } else {
        println!(
            "strata_global_n64_pipeline_accepted initial_global_loads=3 first_bank_prefetch=3 second_bank_prefetch=3 products=4 iteration_backedge_excluded=true"
        );
    }
    Ok(())
}
pub(in crate::strata_matrix) fn validate_global_n64_sources() -> std::result::Result<(), Box<dyn std::error::Error>> {
    use cubecl::wgpu::{AutoCompiler, AutoRepresentation};
    use cubecl::{Compiler, ExecutionMode, WgpuCompilationOptions};
    use std::io::Write;
    for mask in 0..8u8 {
        for (name, definition) in [
            ("fp16", global_n64_definition::<f16>(false)),
            ("bf16", global_n64_definition::<bf16>(false)),
            ("gr_bf16", global_n64_definition::<bf16>(true)),
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
                return Err("global n64 actual SPIRV".into());
            };
            if k.shared_size != 0 {
                return Err("global n64 actual zero shared bytes".into());
            }
            let words = wgpu::hal::vulkan::kuno_native_workgroup_layout(&k.assembled_module)?;
            let words = wgpu::hal::vulkan::kuno_cooperative_fp16_preserve(&words)?.unwrap_or(words);
            if wgpu::hal::vulkan::kuno_cooperative_subgroup(&words) != Some(16) {
                return Err("global n64 actual SG16".into());
            }
            validate_global_n64_program(&words)?;
            #[cfg(test)]
            validate_global_n64_native_negatives(&words)?;

            let mut cmd = std::process::Command::new("spirv-val");
            cmd.args(["--target-env", "vulkan1.3"]);
            if mask & 1 != 0 {
                cmd.arg("--uniform-buffer-standard-layout");
            }
            let mut child = cmd.arg("-").stdin(std::process::Stdio::piped()).spawn()?;
            child
                .stdin
                .take()
                .ok_or("global n64 validator stdin")?
                .write_all(
                    &words
                        .iter()
                        .flat_map(|w| w.to_le_bytes())
                        .collect::<Vec<_>>(),
                )?;
            if !child.wait()?.success() {
                return Err("global n64 SPIRV validation".into());
            }
            println!(
                "strata_global_n64_shader_accepted dtype={name} layout_mask={mask} shared_bytes=0 workgroup=512 subgroup=16 m=128 n=64 k=32 products=4 global_operand_loads=9 global_stores=2 whole_workgroup_handoffs=0"
            );
        }
    }
    Ok(())
}

#[cfg(test)]
fn validate_global_n64_native_negatives(
    words: &[u32],
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let (mut entry, mut grid, mut subgroup, mut input_binding, mut input_var) =
        (None, None, None, None, None);
    let mut at = 5;
    let mut input_id = None;
    while at < words.len() {
        let n = (words[at] >> 16) as usize;
        let op = words[at] & 65535;
        if op == 15 {
            entry = Some(at + 3);
        }
        if op == 16 && n == 6 && words[at + 2] == 17 {
            grid = Some(at + 3);
        }
        if op == 71 && n == 4 && words[at + 2] == 11 && words[at + 3] == 40 {
            subgroup = Some(at + 3);
        }
        if op == 71 && n == 4 && words[at + 2] == 33 && words[at + 3] == 0 {
            input_binding = Some(at + 3);
            input_id = Some(words[at + 1]);
        }
        at += n;
    }
    at = 5;
    while at < words.len() {
        let n = (words[at] >> 16) as usize;
        if words[at] & 65535 == 59 && Some(words[at + 2]) == input_id {
            input_var = Some(at + 3);
        }
        at += n;
    }
    for (pos, value) in [
        (
            entry.ok_or("global n64 real entry negative")?,
            words[entry.unwrap()] ^ 1,
        ),
        (grid.ok_or("global n64 real grid negative")?, 1024),
        (subgroup.ok_or("global n64 real subgroup negative")?, 41),
        (input_binding.ok_or("global n64 real binding negative")?, 4),
        (input_var.ok_or("global n64 real storage negative")?, 4),
    ] {
        let mut bad = words.to_vec();
        bad[pos] = value;
        if wgpu::hal::vulkan::kuno_cooperative_subgroup(&bad).is_some() {
            return Err("global n64 native gate accepted wrong original entry/grid/subgroup/binding/storage".into());
        }
    }
    Ok(())
}
