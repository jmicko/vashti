// Actual emitted shader proofs for the whole new small/medium bundle.
use super::*;
use std::collections::{BTreeMap, BTreeSet};
fn program(
    words: &[u32],
    cohort: bool,
    tile: usize,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let (mut at, mut wg, mut block) = (5, false, 0u32);
    let (mut parents, mut classes, mut bindings, mut constants, mut uses) = (
        BTreeMap::new(),
        BTreeMap::new(),
        BTreeMap::new(),
        BTreeMap::new(),
        BTreeMap::new(),
    );
    let (mut loads, mut stores, mut products, mut barriers, mut loops) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut values: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    let mut matrix_results = BTreeMap::new();
    let mut locations = BTreeMap::new();
    let mut actual_definitions: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    let mut product_inputs = BTreeMap::new();
    let mut shared_writes = Vec::new();
    let mut barrier_sites = Vec::new();
    let mut blocks: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    while at < words.len() {
        let n = (words[at] >> 16) as usize;
        let op = words[at] & 65535;
        if n == 0 || at + n > words.len() {
            return Err("small full walk opcode extent".into());
        }
        if n >= 3
            && matches!(
                op,
                43 | 44 | 45 | 46 | 61 | 80 | 81 | 83 | 124 | 245 | 4457 | 4459
            )
        {
            actual_definitions.insert(words[at + 2], words[at..at + n].to_vec());
            locations.insert(words[at + 2], (block, at));
        }
        match op {
            16 if n == 6 && words[at + 2] == 17 => {
                wg = words[at + 3..at + 6] == [if tile == 64 { 64 } else { 32 }, 1, 1]
            }
            71 if n == 4 && words[at + 2] == 33 => {
                bindings.insert(words[at + 1], words[at + 3]);
            }
            59 => {
                classes.insert(words[at + 2], words[at + 3]);
            }
            43 if n == 4 => {
                constants.insert(words[at + 2], words[at + 3]);
            }
            65 | 66 | 67 | 124 => {
                parents.insert(words[at + 2], words[at + 3]);
            }
            245 => {
                values.insert(
                    words[at + 2],
                    (at + 3..at + n).step_by(2).map(|i| words[i]).collect(),
                );
            }
            83 => {
                values.insert(words[at + 2], vec![words[at + 3]]);
                parents.insert(words[at + 2], words[at + 3]);
            }
            62 => shared_writes.push((block, at, words[at + 1])),
            248 => {
                block = words[at + 1];
                blocks.insert(block, Vec::new());
            }
            246 => loops.push((block, words[at + 1])),
            4456 => {
                uses.insert(words[at + 1], words[at + 6]);
            }
            4457 => {
                loads.push((block, at, words[at + 1], words[at + 3], words[at + 4]));
                matrix_results.insert(words[at + 2], at);
            }
            4458 => stores.push((words[at + 1], words[at + 3])),
            4459 => {
                products.push((block, at));
                product_inputs.insert(at, words[at + 4]);
            }
            224 => {
                barriers.push(words[at + 1]);
                barrier_sites.push((block, at));
            }
            227 => return Err("small full walk fault read is forbidden".into()),
            _ => {}
        }
        if block != 0 {
            blocks.get_mut(&block).unwrap().push(at);
        }
        at += n;
    }
    let root = |mut v: u32| -> std::result::Result<u32, Box<dyn std::error::Error>> {
        let mut seen = BTreeSet::new();
        while let Some(&p) = parents.get(&v) {
            if !seen.insert(v) {
                return Err("small full walk pointer cycle".into());
            }
            v = p;
        }
        Ok(v)
    };
    if !wg || cubecl_wgpu::native_strata_atomic_or_descriptor_mask(words, 0x3f) != 1 << 5 {
        return Err("small full walk actual WG/original AtomicOr-only fault".into());
    }
    if barriers.iter().any(|s| constants.get(s) != Some(&3)) {
        return Err("small full walk workgroup barrier under subgroup branches".into());
    }
    for &(_, _, ty, ptr, layout) in &loads {
        let r = root(ptr)?;
        let use_ = constants
            .get(uses.get(&ty).ok_or("small full walk actual matrix type")?)
            .copied();
        let layout = constants.get(&layout).copied();
        if !matches!(
            (
                classes.get(&r).copied(),
                bindings.get(&r).copied(),
                use_,
                layout
            ),
            (Some(12), Some(1), Some(0), Some(0))
                | (Some(12), Some(0), Some(1), Some(1))
                | (Some(4), None, Some(1), Some(1))
        ) {
            return Err(
                "small full walk weight-A/global-or-masked-input-B provenance/layout".into(),
            );
        }
    }
    for &(ptr, layout) in &stores {
        let r = root(ptr)?;
        if constants.get(&layout) != Some(&1)
            || !((classes.get(&r) == Some(&12) && bindings.get(&r) == Some(&2))
                || classes.get(&r) == Some(&4))
        {
            return Err("small full walk original column-major output/publication owner".into());
        }
    }
    let mut edges = BTreeMap::new();
    for (&b, ins) in &blocks {
        let a = *ins.last().ok_or("small full walk empty block")?;
        let n = (words[a] >> 16) as usize;
        edges.insert(
            b,
            match words[a] & 65535 {
                249 => vec![words[a + 1]],
                250 => vec![words[a + 2], words[a + 3]],
                251 => std::iter::once(words[a + 2])
                    .chain((a + 4..a + n).step_by(2).map(|i| words[i]))
                    .collect(),
                _ => Vec::new(),
            },
        );
    }
    let mut regions = Vec::new();
    for (header, merge) in loops {
        let (mut seen, mut todo) = (BTreeSet::new(), vec![header]);
        while let Some(b) = todo.pop() {
            if b == merge || !seen.insert(b) {
                continue;
            }
            todo.extend(
                edges
                    .get(&b)
                    .ok_or("small full walk missing CFG edge")?
                    .iter()
                    .copied(),
            );
        }
        let count = products.iter().filter(|p| seen.contains(&p.0)).count();
        if count > 0 {
            regions.push((seen, count));
        }
    }
    let innermost = regions
        .iter()
        .enumerate()
        .filter(|(i, c)| {
            !regions
                .iter()
                .enumerate()
                .any(|(j, d)| *i != j && d.0 != c.0 && d.0.is_subset(&c.0))
        })
        .map(|(_, c)| c)
        .collect::<Vec<_>>();
    let raw_count = if cohort { 4 } else { 8 };
    if innermost.len() != if cohort { 1 } else { 2 } || innermost.iter().any(|r| r.1 != raw_count) {
        return Err(
            "full walk must retain ALL C fragments through each chronological K loop".into(),
        );
    }
    struct BDefinitions<'a> {
        words: &'a [u32],
        blocks: &'a BTreeMap<u32, Vec<usize>>,
        edges: &'a BTreeMap<u32, Vec<u32>>,
        region: &'a BTreeSet<u32>,
        defs: &'a BTreeMap<u32, Vec<u32>>,
        matrices: &'a BTreeMap<u32, usize>,
        aliases: &'a BTreeMap<u32, Vec<u32>>,
        classes: &'a BTreeMap<u32, u32>,
        parents: &'a BTreeMap<u32, u32>,
        locations: &'a BTreeMap<u32, (u32, usize)>,
    }
    impl BDefinitions<'_> {
        fn root(&self, mut id: u32) -> std::result::Result<u32, Box<dyn std::error::Error>> {
            let mut seen = BTreeSet::new();
            while let Some(&p) = self.parents.get(&id) {
                if !seen.insert(id) {
                    return Err("full walk matrix pointer cycle".into());
                }
                id = p;
            }
            Ok(id)
        }
        fn reaching(
            &self,
            ptr: u32,
            block: u32,
            before: usize,
            mut seen: BTreeSet<u32>,
        ) -> std::result::Result<BTreeSet<u32>, Box<dyn std::error::Error>> {
            // Reject a previous iteration's store: every B load must be assigned
            // on every path in THIS K iteration, including the first iteration.
            if !seen.insert(block) {
                return Err("full walk B load depends on a previous iteration".into());
            }
            for &a in self
                .blocks
                .get(&block)
                .ok_or("full walk B missing load block")?
                .iter()
                .rev()
            {
                if a < before && self.words[a] & 65535 == 62 && self.root(self.words[a + 1])? == ptr
                {
                    return Ok(BTreeSet::from([self.words[a + 2]]));
                }
            }
            let predecessors = self
                .edges
                .iter()
                .filter(|(_, es)| es.contains(&block))
                .map(|(&b, _)| b)
                .collect::<Vec<_>>();
            if predecessors.is_empty() || predecessors.iter().any(|b| !self.region.contains(b)) {
                return Err("full walk B has an uninitialized current-iteration path".into());
            }
            let mut stores = BTreeSet::new();
            for b in predecessors {
                stores.extend(self.reaching(ptr, b, usize::MAX, seen.clone())?);
            }
            Ok(stores)
        }
        fn origins(
            &self,
            id: u32,
            mut seen: BTreeSet<u32>,
        ) -> std::result::Result<BTreeSet<u32>, Box<dyn std::error::Error>> {
            if self.matrices.contains_key(&id) {
                return Ok(BTreeSet::from([id]));
            }
            if !seen.insert(id) {
                return Err("full walk B matrix value cycle".into());
            }
            let mut result = BTreeSet::new();
            if let Some(inputs) = self.aliases.get(&id) {
                for &i in inputs {
                    result.extend(self.origins(i, seen.clone())?);
                }
                return Ok(result);
            }
            let d = self
                .defs
                .get(&id)
                .ok_or_else(|| format!("full walk B missing definition id={id}"))?;
            if d[0] & 65535 != 61 || d.len() != 4 {
                return Err(
                    format!("full walk B unsupported definition id={id} words={d:?}").into(),
                );
            }
            let ptr = self.root(d[3])?;
            if self.classes.get(&ptr) != Some(&7) {
                return Err(
                    "full walk B load is not original function-local matrix storage".into(),
                );
            }
            let &(block, at) = self
                .locations
                .get(&id)
                .ok_or("full walk B missing actual load location")?;
            for value in self.reaching(ptr, block, at, BTreeSet::new())? {
                result.extend(self.origins(value, seen.clone())?);
            }
            Ok(result)
        }
    }
    let mut pure = 0;
    let mut partial = 0;
    for region in innermost {
        let ls = loads
            .iter()
            .filter(|p| region.0.contains(&p.0))
            .collect::<Vec<_>>();
        let ps = products
            .iter()
            .filter(|p| region.0.contains(&p.0))
            .map(|p| p.1)
            .collect::<Vec<_>>();
        let weights = ls
            .iter()
            .filter(|p| bindings.get(&root(p.3).unwrap()) == Some(&1))
            .map(|p| p.1)
            .collect::<Vec<_>>();
        let globals = ls
            .iter()
            .filter(|p| bindings.get(&root(p.3).unwrap()) == Some(&0))
            .count();
        let shared = ls
            .iter()
            .filter(|p| classes.get(&root(p.3).unwrap()) == Some(&4))
            .count();
        if weights.len() != 4
            || globals != if cohort { 1 } else { 2 }
            || shared != 0 && (cohort || shared != 2)
            || ls.len() != 4 + globals + shared
        {
            return Err("full walk actual global weights and complete/masked B loads".into());
        }
        for i in 0..4 {
            let n = if cohort { 1 } else { 2 };
            if weights[i] >= ps[i * n] || (i > 0 && weights[i] <= ps[i * n - 1]) {
                return Err("full walk one A reused across genuine B values".into());
            }
        }
        let b_definitions = BDefinitions {
            words,
            blocks: &blocks,
            edges: &edges,
            region: &region.0,
            defs: &actual_definitions,
            matrices: &matrix_results,
            aliases: &values,
            classes: &classes,
            parents: &parents,
            locations: &locations,
        };
        let mut b_values = BTreeMap::<BTreeSet<u32>, usize>::new();
        for &p in &ps {
            let b = b_definitions.origins(product_inputs[&p], BTreeSet::new())?;
            *b_values.entry(b).or_default() += 1;
        }
        if b_values.len() != if cohort { 1 } else { 2 } || b_values.values().any(|n| *n != 4) {
            return Err("full walk independent B SSA values and four products each".into());
        }
        if !cohort {
            let keys = b_values.keys().collect::<Vec<_>>();
            if !keys[0].is_disjoint(keys[1]) {
                return Err("full walk B stage overwrite aliases earlier matrix value".into());
            }
        }
        let bs = barrier_sites
            .iter()
            .filter(|p| region.0.contains(&p.0))
            .collect::<Vec<_>>();
        if shared == 0 {
            pure += 1;
            if !bs.is_empty() {
                return Err("full walk pure global K loop has a barrier".into());
            }
            if b_values.keys().any(|ids| ids.len() != 1) {
                return Err("full walk pure B load identity".into());
            }
        } else {
            partial += 1;
            if bs.len() != 4 || b_values.keys().any(|ids| ids.len() != 2) {
                return Err("full walk two masked B loads need independent global/shared SSA and four SG handoffs".into());
            }
            let sl = ls
                .iter()
                .filter(|p| classes.get(&root(p.3).unwrap()) == Some(&4))
                .map(|p| p.1)
                .collect::<Vec<_>>();
            let ws = shared_writes
                .iter()
                .filter(|p| region.0.contains(&p.0) && classes.get(&root(p.2).unwrap()) == Some(&4))
                .map(|p| p.1)
                .collect::<Vec<_>>();
            if ws.len() != 8
                || !(ws[3] < bs[0].1
                    && bs[0].1 < sl[0]
                    && sl[0] < bs[1].1
                    && bs[1].1 < ws[4]
                    && ws[7] < bs[2].1
                    && bs[2].1 < sl[1]
                    && sl[1] < bs[3].1
                    && bs[3].1 < ps[0])
            {
                return Err("full walk actual fully unrolled stage initialization/load-before-overwrite chronology".into());
            }
        }
        for b in &region.0 {
            for &a in &blocks[b] {
                let op = words[a] & 65535;
                if (227..=242).contains(&op) {
                    return Err("full walk arithmetic K loop contains atomics".into());
                }
            }
        }
    }
    if pure != 1 || partial != if cohort { 0 } else { 1 } {
        return Err("full walk genuine complete and masked product paths".into());
    }

    Ok(())
}
#[cfg(test)]
fn negatives(
    words: &[u32],
    cohort: bool,
    tile: usize,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut changes = Vec::new();
    let mut at = 5;
    while at < words.len() {
        let n = (words[at] >> 16) as usize;
        let op = words[at] & 65535;
        if op == 16 && n == 6 && words[at + 2] == 17 {
            changes.push((at + 3, 256));
        }
        if op == 71 && n == 4 && words[at + 2] == 33 && matches!(words[at + 3], 0 | 1 | 2 | 5) {
            changes.push((at + 3, 29));
        }
        at += n;
    }
    if changes.len() < 5 {
        return Err("small full walk genuine negative inventory".into());
    }
    for (i, v) in changes {
        let mut w = words.to_vec();
        w[i] = v;
        if program(&w, cohort, tile).is_ok() {
            return Err("small full walk accepted wrong actual grid/owner/role".into());
        }
    }
    if !cohort {
        let mut at = 5;
        let mut products = Vec::new();
        while at < words.len() {
            let n = (words[at] >> 16) as usize;
            if words[at] & 65535 == 4459 {
                products.push(at);
            }
            at += n;
        }
        let first = words[products[0] + 4];
        let second = words[products[1] + 4];
        if first == second {
            return Err("full walk negative needs genuine independent B values".into());
        }
        let mut changed = words.to_vec();
        let mut edits = 0;
        for a in products {
            if changed[a + 4] == second {
                changed[a + 4] = first;
                edits += 1;
            }
        }
        if edits != 4 || program(&changed, cohort, tile).is_ok() {
            return Err("full walk accepted real B1-to-B0 matrix alias mutation".into());
        }
    }

    Ok(())
}
pub(crate) fn validate_sources() -> std::result::Result<(), Box<dyn std::error::Error>> {
    use cubecl::wgpu::{AutoCompiler, AutoRepresentation};
    use cubecl::{Compiler, ExecutionMode, WgpuCompilationOptions};
    use std::io::Write;
    for mask in 0..8u8 {
        let defs = definitions()
            .into_iter()
            .map(|(n, d)| {
                (
                    n,
                    d,
                    false,
                    if n.contains("macro64") { 64usize } else { 32 },
                )
            })
            .chain(
                crate::strata_matrix::cohorts::small_full_walk::definitions()
                    .into_iter()
                    .map(|(n, d)| (n, d, true, 16usize)),
            );
        for (name, def, cohort, tile) in defs {
            let mut o = WgpuCompilationOptions {
                supports_vulkan: true,
                ..Default::default()
            };
            o.vulkan.max_spirv_version = (1, 6);
            o.vulkan.supports_uniform_standard_layout = mask & 1 != 0;
            o.vulkan.supports_uniform_unsized_array = mask & 2 != 0;
            o.vulkan.supports_explicit_smem = mask & 4 != 0;
            let AutoRepresentation::SpirV(k) = Compiler::compile(
                &mut AutoCompiler::SpirV(Default::default()),
                def,
                &o,
                ExecutionMode::Unchecked,
                cubecl::ir::StorageType::Scalar(ElemType::UInt(cubecl::ir::UIntKind::U32)),
            )?
            else {
                return Err("small full walk actual SPIRV".into());
            };
            // Final exact per-layout resource populations are frozen after these real
            // modules compile; never assume larger scratch than the preceding kernels.
            if k.shared_size
                != if cohort {
                    1024
                } else if tile == 64 {
                    4096
                } else {
                    2048
                }
            {
                return Err(
                    format!("small full walk bounded real scratch {}", k.shared_size).into(),
                );
            }
            let words = wgpu::hal::vulkan::kuno_native_workgroup_layout(&k.assembled_module)?;
            let words = wgpu::hal::vulkan::kuno_cooperative_fp16_preserve(&words)?.unwrap_or(words);
            if wgpu::hal::vulkan::kuno_cooperative_subgroup(&words) != Some(16) {
                return Err("small full walk authentic SG16".into());
            }
            #[cfg(test)]
            if let Some(dir) = std::env::var_os("KUNO_PUBLIC_FULL_WALK_MODULE_DIR") {
                let path = std::path::PathBuf::from(dir);
                // Fixed compiler definitions only: no request, activation,
                // weight contents or GPU data is present in these modules.
                let file = path.join(format!("{name}-layout{mask}.spv"));
                let bytes = words
                    .iter()
                    .flat_map(|w| w.to_le_bytes())
                    .collect::<Vec<_>>();
                if file.exists() {
                    if std::fs::read(&file)? != bytes {
                        return Err(
                            "full walk fixed compiler module changed within frozen test".into()
                        );
                    }
                } else {
                    std::fs::write(file, bytes)?;
                }
            }
            program(&words, cohort, tile)?;
            #[cfg(test)]
            negatives(&words, cohort, tile)?;
            let mut cmd = std::process::Command::new("spirv-val");
            cmd.args(["--target-env", "vulkan1.3"]);
            if mask & 1 != 0 {
                cmd.arg("--uniform-buffer-standard-layout");
            }
            let mut child = cmd.arg("-").stdin(std::process::Stdio::piped()).spawn()?;
            child
                .stdin
                .take()
                .ok_or("small full walk validator input")?
                .write_all(
                    &words
                        .iter()
                        .flat_map(|w| w.to_le_bytes())
                        .collect::<Vec<_>>(),
                )?;
            if !child.wait()?.success() {
                return Err("small full walk validated actual module".into());
            }
            println!(
                "strata_small_full_walk_shader_accepted name={name} layout={mask} shared_bytes={} workgroup={} subgroup=16 weights_first=true checked_original_halves=true",
                k.shared_size,
                if tile == 64 { 64 } else { 32 }
            );
        }
    }
    Ok(())
}
