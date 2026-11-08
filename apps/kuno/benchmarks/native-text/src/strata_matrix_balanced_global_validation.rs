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
    let mut matrix_shapes=BTreeMap::new();let mut float_widths=BTreeMap::new();
    let mut locations = BTreeMap::new();
    let mut actual_definitions: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    let mut product_inputs = BTreeMap::new();
    let mut product_a_inputs = BTreeMap::new();
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
            22 if n==3 => {float_widths.insert(words[at+1],words[at+2]);}
            16 if n == 6 && words[at + 2] == 17 => {
                wg = words[at + 3..at + 6] == [(tile*4) as u32, 1, 1]
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
                matrix_shapes.insert(words[at+1],(words[at+2],words[at+3],words[at+4],words[at+5],words[at+6]));
            }
            4457 => {
                loads.push((block, at, words[at + 1], words[at + 3], words[at + 4]));
                matrix_results.insert(words[at + 2], at);
            }
            4458 => stores.push((words[at + 1], words[at + 3])),
            4459 => {
                products.push((block, at));
                product_inputs.insert(at, words[at + 4]);
                product_a_inputs.insert(at,words[at+3]);
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
    for &(component,scope,m,n,use_)in matrix_shapes.values(){
        let u=constants.get(&use_).copied().ok_or("balanced literal matrix use")?;
        if !matches!(u,0..=2)||constants.get(&scope)!=Some(&3)||constants.get(&m)!=Some(&if u==1{16}else{8})||constants.get(&n)!=Some(&16)||float_widths.get(&component)!=Some(&if u==2{32}else{16}) {
            return Err("balanced actual SG16 A8x16/B16x16/C8x16 FP16/FP32 shape".into());
        }
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
    let mut region_headers = BTreeMap::new();
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
            region_headers.insert(seen.clone(), header);
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
    let raw_count = if PRELOAD {4}else{2};
    let expected_k_loops = if cohort || PADDED_GLOBAL || cfg!(feature = "strata-single-k-walk") { 1 } else { 2 };
    if innermost.len() != expected_k_loops || innermost.iter().any(|r| r.1 != raw_count) {
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
                if CARRY {return Ok(BTreeSet::new());}
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
            if predecessors.is_empty() || (!CARRY && predecessors.iter().any(|b| !self.region.contains(b))) {
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
                if CARRY {return Ok(BTreeSet::new());}
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
        if weights.len() != 2*if PRELOAD {2}else{1}
            || globals != if PRELOAD {2}else{1}
            || shared != 0
            || ls.len() != 2*if PRELOAD {2}else{1} + globals + shared
        {
            return Err("full walk actual global weights and complete/masked B loads".into());
        }
        if PRELOAD && !CARRY {
            if ls.iter().any(|l| l.1>=ps[0]) {return Err("preload all six real global operands must precede first current-bank product".into());}
        }else if !CARRY {
        for i in 0..2 {
            let n = 1;
            if weights[i] >= ps[i * n] || (i > 0 && weights[i] <= ps[i * n - 1]) {
                return Err("full walk one A reused across genuine B values".into());
            }
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
        if b_values.len() != if PRELOAD {2}else{1} || b_values.values().any(|n| *n != 2) {
            return Err("full walk independent B SSA values and four products each".into());
        }
        if CARRY {
            let initial=loads.iter().filter(|l|!region.0.contains(&l.0)).collect::<Vec<_>>();
            let initial_a=initial.iter().filter(|l|bindings.get(&root(l.3).unwrap())==Some(&1)).collect::<Vec<_>>();
            let initial_b=initial.iter().filter(|l|bindings.get(&root(l.3).unwrap())==Some(&0)).collect::<Vec<_>>();
            if loads.len()!=9 || initial_a.len()!=2 || initial_b.len()!=1 {
                return Err("carry requires authentic initial A0/A1/B0 and six in-loop loads".into());
            }
            for(i,&p)in ps.iter().enumerate(){
                let expected=if i<2 {BTreeSet::from([words[initial_a[i].1+2],words[weights[i+2]+2]])}else{BTreeSet::from([words[weights[i-2]+2]])};
                if b_definitions.origins(product_a_inputs[&p],BTreeSet::new())?!=expected {
                    return Err("carry A must reach initial/next-iteration bank0 or current bank1, never a crossed or fabricated operand".into());
                }
            }
            let initial_b=words[initial_b[0].1+2];
            let in_b=ls.iter().filter(|l|bindings.get(&root(l.3).unwrap())==Some(&0)).collect::<Vec<_>>();
            let b0=BTreeSet::from([initial_b,words[in_b[1].1+2]]);
            let b1=BTreeSet::from([words[in_b[0].1+2]]);
            if b_definitions.origins(product_inputs[&ps[0]],BTreeSet::new())?!=b0 || b_definitions.origins(product_inputs[&ps[2]],BTreeSet::new())?!=b1 {
                return Err("carry authentic B0 initial/backedge and B1 current origins".into());
            }
        }else{
            for(i,&p)in ps.iter().enumerate(){
                let a=b_definitions.origins(product_a_inputs[&p],BTreeSet::new())?;
                if a!=BTreeSet::from([words[weights[i]+2]]) {return Err("balanced matrix A must be the corresponding distinct chronological weight load".into());}
            }
        }
        if PRELOAD {
            let origin=|at:usize|b_definitions.origins(product_inputs[&at],BTreeSet::new());
            if origin(ps[0])?!=origin(ps[1])? || origin(ps[2])?!=origin(ps[3])? || origin(ps[0])?==origin(ps[2])? {
                return Err("preload first/next bank input fragments must be distinct and each reused by its own two C cells".into());
            }
            let mut accumulator_results=matrix_results.clone();
            for &p in &ps {accumulator_results.insert(words[p+2],p);}
            let accumulator_definitions=BDefinitions{matrices:&accumulator_results,..b_definitions};
            for i in 0..2 {
                if accumulator_definitions.origins(words[ps[i+2]+5],BTreeSet::new())?!=BTreeSet::from([words[ps[i]+2]]) {
                    return Err("preload next K16 must consume the corresponding first-bank C result, without accumulator crosswire".into());
                }
            }
        }
        let header = region_headers[&region.0];
        let mut flow: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for b in &region.0 {
            let ins = &blocks[b];
            for pair in ins.windows(2) {
                flow.insert(pair[0], vec![pair[1]]);
            }
            flow.insert(
                *ins.last().ok_or("full walk empty matrix block")?,
                edges[b]
                    .iter()
                    .filter(|n| region.0.contains(n) && **n != header)
                    .map(|n| blocks[n][0])
                    .collect(),
            );
        }
        let reachable = |from: usize, to: usize, skip: Option<usize>| {
            let (mut seen, mut todo) = (BTreeSet::new(), vec![from]);
            while let Some(a) = todo.pop() {
                if Some(a) == skip || !seen.insert(a) {
                    continue;
                }
                if a == to {
                    return true;
                }
                if let Some(ns) = flow.get(&a) {
                    todo.extend(ns.iter().copied());
                }
            }
            false
        };
        // Actual instruction CFG, not physical SPIR-V block placement. Helpers'
        // branch blocks may be serialized after their consuming product block.
        let must_pass = |from: usize, to: usize, guard: usize| {
            reachable(from, to, None) && !reachable(from, to, Some(guard))
        };
        let entry = blocks[&header][0];
        if CARRY {
            // Evaluate instruction CFG with the backedge cut. Conditional reload
            // blocks can be physically after their consuming product block.
            let bank1=ls.iter().filter(|l|l.1<ps[0]).collect::<Vec<_>>();
            let bank0=ls.iter().filter(|l|l.1>=ps[0]).collect::<Vec<_>>();
            if bank1.len()!=3 || bank0.len()!=3 || bank1.iter().any(|l|!must_pass(entry,ps[0],l.1)) || bank0.iter().any(|l|!reachable(ps[1],l.1,None)||!reachable(l.1,ps[2],None)) || !must_pass(entry,ps[2],ps[1]) {
                return Err("carry requires bank1 prefetch before current products and conditional bank0 refill between product pairs".into());
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
            if (!CARRY && b_values.keys().any(|ids| ids.len() != 1)) || (CARRY && b_values.keys().map(|ids|ids.len()).collect::<BTreeSet<_>>()!=BTreeSet::from([1,2])) {
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
            let initialized = ws.len() == 8
                && (0..2).all(|i| {
                    let before = bs[i * 2].1;
                    let after = bs[i * 2 + 1].1;
                    ws[i * 4..i * 4 + 4]
                        .iter()
                        .all(|&w| must_pass(entry, before, w))
                        && must_pass(entry, sl[i], before)
                        && must_pass(entry, after, sl[i])
                        && must_pass(sl[i], ps[0], after)
                });
            // Physical block order does not identify which B stage executes
            // first. Require one actual iteration order and a handoff before
            // EVERY subsequent store into the aliased stage.
            let forward = ws.len() == 8 && reachable(sl[0], ws[4], None);
            let reverse = ws.len() == 8 && reachable(sl[1], ws[0], None);
            let reuse_safe = forward != reverse
                && if forward {
                    ws[4..].iter().all(|&w| must_pass(sl[0], w, bs[1].1))
                } else {
                    ws[..4].iter().all(|&w| must_pass(sl[1], w, bs[3].1))
                };
            if !initialized || !reuse_safe {
                return Err(format!("full walk actual CFG stage initialization/load-before-overwrite chronology stores={ws:?} barriers={bs:?} shared_loads={sl:?} first_product={:?}",ps.first()).into());
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
    let expected_pure = usize::from(cohort || PADDED_GLOBAL || !cfg!(feature = "strata-single-k-walk"));
    if pure != expected_pure || partial != if cohort || PADDED_GLOBAL { 0 } else { 1 } {
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
            changes.push((at + 3, words[at+3]+16));
        }
        if op == 71 && n == 4 && words[at + 2] == 33 && matches!(words[at + 3], 0 | 1 | 2 | 5) {
            changes.push((at + 3, 29));
        }
        at += n;
    }
    if changes.len() < 5 {
        return Err("small full walk genuine negative inventory".into());
    }
    let mut at=5;
    while at<words.len(){let n=(words[at]>>16)as usize;if words[at]&65535==22 && words[at+2]==16{changes.push((at+2,32));}at+=n;}
    for (i, v) in changes {
        let mut w = words.to_vec();
        w[i] = v;
        if program(&w, cohort, tile).is_ok() {
            return Err("small full walk accepted wrong actual grid/owner/role".into());
        }
    }
    let mut ps=Vec::new();let mut at=5;
    while at<words.len(){let n=(words[at]>>16)as usize;if words[at]&65535==4459{ps.push((at,n));}at+=n;}
    if ps.len()!=if PRELOAD {4}else{2} {return Err("balanced negatives need every real product".into());}
    let mut w=words.to_vec();w[ps[1].0+3]=w[ps[0].0+3];
    if program(&w,cohort,tile).is_ok(){return Err("balanced matrix accepted A1 to A0 alias".into());}
    if PRELOAD {
        let mut w=words.to_vec();w[ps[2].0+4]=w[ps[0].0+4];
        if program(&w,cohort,tile).is_ok(){return Err("preload accepted next-bank B aliased to current bank".into());}
        let mut w=words.to_vec();w[ps[2].0+5]=w[ps[3].0+5];
        if program(&w,cohort,tile).is_ok(){return Err("preload accepted next-bank accumulator crosswire".into());}
        let mut w=words.to_vec();w[ps[2].0+3]=w[ps[0].0+3];
        if program(&w,cohort,tile).is_ok(){return Err("preload accepted next-bank A aliased to previous K".into());}
    }
    let mut w=words.to_vec();w.drain(ps[1].0..ps[1].0+ps[1].1);
    if program(&w,cohort,tile).is_ok(){return Err("balanced matrix accepted missing second product".into());}

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
                crate::strata_matrix::cohorts::balanced_global::definitions()
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
            if k.shared_size as usize != tile*128 {return Err(format!("balanced actual publication scratch {}",k.shared_size).into());}
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
                "{}_shader_accepted name={name} layout={mask} shared_bytes={} workgroup={} subgroup=16 weights_first=true checked_original_halves=true",
                "strata_balanced_global",
                k.shared_size,
                tile*4
            );
        }
    }
    Ok(())
}
