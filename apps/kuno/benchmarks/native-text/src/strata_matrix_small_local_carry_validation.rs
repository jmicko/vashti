use super::*;
pub(crate) fn validate_program(
    words: &[u32],
    expert: bool,
    small: Option<(usize, bool, u32)>,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    use std::collections::{BTreeMap, BTreeSet};
    assert!(expert && small.is_some(), "small carried validator scope");
    let wg_expected = small.map_or(256, |(tile, _, _)| tile * 4) as u32;
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
    let mut value_ops = BTreeMap::new();
    let (mut float_widths, mut vectors, mut pointer_types, mut value_types) = (
        BTreeMap::new(),
        BTreeMap::new(),
        BTreeMap::new(),
        BTreeMap::new(),
    );
    let mut store_layouts = Vec::new();
    let (mut at, mut block, mut wg, mut barriers, mut atomics) = (5, None, false, 0, 0);
    while at < words.len() {
        let n = (words[at] >> 16) as usize;
        let op = words[at] & 65535;
        if n == 0 || at + n > words.len() {
            return Err("global wide32 opcode extent".into());
        }
        if matches!(op, 65 | 66 | 67 | 83 | 124) {
            value_types.insert(words[at + 2], words[at + 1]);
        }
        match op {
            22 if n == 3 => {
                float_widths.insert(words[at + 1], words[at + 2]);
            }
            23 if n == 4 => {
                vectors.insert(words[at + 1], (words[at + 2], words[at + 3]));
            }
            32 if n == 4 => {
                pointer_types.insert(words[at + 1], (words[at + 2], words[at + 3]));
            }
            16 if n == 6 && words[at + 2] == 17 => {
                wg = words[at + 3..at + 6] == [wg_expected, 1, 1]
            }
            71 if n == 4 && words[at + 2] == 33 => {
                bindings.insert(words[at + 1], words[at + 3]);
            }
            56 => {
                block = None;
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
            246 => loops.push((block.ok_or("global wide32 loop header")?, words[at + 1])),
            4456 => {
                matrix_uses.insert(words[at + 1], words[at + 6]);
            }
            4457 => {
                operand_roles.push((words[at + 1], words[at + 3], words[at + 4]));
                loads.push((
                    block.ok_or("global wide32 load block")?,
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
            4459 => products.push((block.ok_or("global wide32 product block")?, at)),
            224 => barriers += 1,
            227..=242 => atomics += 1,
            _ => {}
        }
        if matches!(op, 43 | 83 | 113 | 114 | 124 | 194) {
            value_ops.insert(words[at + 2], op);
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
        || loads.len()
            != if small.is_some() {
                15
            } else if expert {
                15
            } else {
                6
            }
        || stores.len()
            != if small.is_some() {
                12
            } else if expert {
                12
            } else {
                8
            }
        || products.len()
            != if small.is_some() {
                8
            } else if expert {
                12
            } else {
                8
            }
        || (!expert && classes.values().any(|&c| c == 4))
    {
        return Err(format!("global wide32 actual module WG={wg} barriers={barriers} atomics={atomics} loads={} stores={} products={}",loads.len(),stores.len(),products.len()).into());
    }
    let root = |mut v: u32| -> std::result::Result<u32, Box<dyn std::error::Error>> {
        let mut seen = BTreeSet::new();
        while let Some(&p) = parents.get(&v) {
            if !seen.insert(v) {
                return Err("global wide32 pointer cycle".into());
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
    fn constant(
        id: u32,
        defs: &BTreeMap<u32, Vec<u32>>,
        ops: &BTreeMap<u32, u32>,
        depth: usize,
    ) -> Option<u32> {
        if depth > 16 {
            return None;
        }
        let d = defs.get(&id)?;
        match ops.get(&id)? {
            43 if d.len() == 1 => Some(d[0]),
            83 | 113 | 114 | 124 if d.len() == 1 => constant(d[0], defs, ops, depth + 1),
            194 if d.len() == 2 => constant(d[0], defs, ops, depth + 1)?.checked_shr(constant(
                d[1],
                defs,
                ops,
                depth + 1,
            )?),
            _ => None,
        }
    }
    let mut input_loads = 0;
    let mut weight_loads = 0;
    for &(_, _, ptr, stride) in &loads {
        let r = root(ptr)?;
        if classes.get(&r) != Some(&12) {
            return Err("global wide32 actual StorageBuffer operand".into());
        }
        match bindings.get(&r) {
            Some(0) => input_loads += 1,
            Some(1) => weight_loads += 1,
            _ => return Err("global wide32 original input/weight binding".into()),
        }
        let (mut seen, mut todo, mut uniform) = (BTreeSet::new(), vec![stride], false);
        while let Some(v) = todo.pop() {
            if !seen.insert(v) {
                continue;
            }
            // OpConstant operands are literal bits, never value-ID edges.
            if value_ops.get(&v) == Some(&43) {
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
        let constant_cohort_stride = small.is_some_and(|(_, cohort, columns)| {
            if !cohort || constant(stride, &defs, &value_ops, 0) != Some(columns / 4) {
                return false;
            }
            value_types
                .get(&ptr)
                .and_then(|ty| pointer_types.get(ty))
                .filter(|(class, _)| *class == 12)
                .and_then(|(_, ty)| vectors.get(ty))
                .is_some_and(|(scalar, width)| *width == 4 && float_widths.get(scalar) == Some(&16))
        });
        if if small.is_some_and(|(_, cohort, _)| cohort) {
            !constant_cohort_stride
        } else {
            !uniform
        } {
            return Err("local prefetch authentic dynamic INFO or exact vec4 cohort stride".into());
        }
    }
    if input_loads
        != if small.is_some() {
            3
        } else if expert {
            5
        } else {
            2
        }
        || weight_loads
            != if small.is_some() {
                12
            } else if expert {
                10
            } else {
                4
            }
    {
        return Err("global wide32 original weight-A/input-B fragment populations".into());
    }
    let (mut global_stores, mut fallback_stores) = (0, 0);
    for ptr in stores {
        let r = root(ptr)?;
        if classes.get(&r) == Some(&12) && bindings.get(&r) == Some(&2) {
            global_stores += 1;
        } else if expert && classes.get(&r) == Some(&4) {
            fallback_stores += 1;
        } else {
            return Err("global wide32 original global output or exact fallback owner".into());
        }
    }
    if global_stores
        != if small.is_some() {
            8
        } else if expert {
            10
        } else {
            8
        }
        || fallback_stores
            != if small.is_some() {
                4
            } else if expert {
                2
            } else {
                0
            }
    {
        return Err("global wide32 exact publication population".into());
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
    if counts != BTreeSet::from([8]) || innermost.len() != 1 {
        return Err("small local carry exact eight-product loop".into());
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
    // Follow genuine Function-storage reaching definitions, including initial
    // loads and loop backedges. CubeCL does not keep CMMA handles in scalar SSA.
    let mut full_defs = BTreeMap::new();
    let mut aliases = BTreeMap::new();
    let mut locations = BTreeMap::new();
    let mut matrix_values = BTreeMap::new();
    let mut at = 5;
    let mut block = 0;
    while at < words.len() {
        let n = (words[at] >> 16) as usize;
        let op = words[at] & 65535;
        if op == 248 {
            block = words[at + 1];
        }
        if matches!(op, 61 | 83 | 124 | 245 | 4457 | 4459) {
            full_defs.insert(words[at + 2], words[at..at + n].to_vec());
            locations.insert(words[at + 2], (block, at));
        }
        if op == 4457 {
            matrix_values.insert(words[at + 2], at);
        }
        if op == 83 {
            aliases.insert(words[at + 2], vec![words[at + 3]]);
        }
        if op == 245 {
            aliases.insert(
                words[at + 2],
                (at + 3..at + n).step_by(2).map(|i| words[i]).collect(),
            );
        }
        at += n;
    }
    let actual_blocks = blocks
        .iter()
        .map(|(&b, ins)| (b, ins.iter().map(|i| i.0).collect::<Vec<_>>()))
        .collect::<BTreeMap<_, _>>();
    struct BDefinitions<'a> {
        words: &'a [u32],
        blocks: &'a BTreeMap<u32, Vec<usize>>,
        edges: &'a BTreeMap<u32, Vec<u32>>,
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
            // A revisited loop edge contributes no new store. The callers still
            // require the exact initial/current/backedge origin sets.
            if !seen.insert(block) {
                return Ok(BTreeSet::new());
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
            if predecessors.is_empty() {
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
                return Ok(BTreeSet::new());
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
    let proof = BDefinitions {
        words,
        blocks: &actual_blocks,
        edges: &edges,
        defs: &full_defs,
        matrices: &matrix_values,
        aliases: &aliases,
        classes: &classes,
        parents: &parents,
        locations: &locations,
    };
    let ps = products
        .iter()
        .filter(|p| normal.1.contains(&p.0))
        .map(|p| p.1)
        .collect::<Vec<_>>();
    let mut first_weights = BTreeSet::new();
    let mut next_weights = BTreeSet::new();
    let mut first_b = None;
    let mut next_b = None;
    for i in 0..4 {
        let a0 = proof.origins(words[ps[i] + 3], BTreeSet::new())?;
        let a1 = proof.origins(words[ps[i + 4] + 3], BTreeSet::new())?;
        let b0 = proof.origins(words[ps[i] + 4], BTreeSet::new())?;
        let b1 = proof.origins(words[ps[i + 4] + 4], BTreeSet::new())?;
        if a0.len() != 2
            || a1.len() != 1
            || b0.len() != 2
            || b1.len() != 1
            || !a0.is_disjoint(&a1)
            || !b0.is_disjoint(&b1)
        {
            return Err(format!("balanced wide32 authentic carried/current operand sets i={i} a0={a0:?} a1={a1:?} b0={b0:?} b1={b1:?}").into());
        }
        for origins in [&a0, &a1] {
            for id in origins {
                let a = matrix_values[id];
                if bindings.get(&root(words[a + 3])?) != Some(&1) {
                    return Err("balanced wide32 weight-A origin".into());
                }
            }
        }
        for origins in [&b0, &b1] {
            for id in origins {
                let a = matrix_values[id];
                if bindings.get(&root(words[a + 3])?) != Some(&0) {
                    return Err("balanced wide32 input-B origin".into());
                }
            }
        }
        if !first_weights.is_disjoint(&a0) || !next_weights.is_disjoint(&a1) {
            return Err("balanced wide32 distinct output rows cannot alias A".into());
        }
        for origins in [&a0, &b0] {
            if origins
                .iter()
                .filter(|&&id| normal.1.contains(&locations[&id].0))
                .count()
                != 1
            {
                return Err(
                    "balanced wide32 bank0 must include one initial and one backedge refill".into(),
                );
            }
        }
        for origins in [&a1, &b1] {
            if origins
                .iter()
                .any(|id| !normal.1.contains(&locations[id].0) || matrix_values[id] >= ps[0])
            {
                return Err(
                    "balanced wide32 genuine next-bank loads precede current products".into(),
                );
            }
        }
        first_weights.extend(a0);
        next_weights.extend(a1);
        if first_b.as_ref().is_some_and(|old| old != &b0)
            || next_b.as_ref().is_some_and(|old| old != &b1)
        {
            return Err("balanced wide32 four rows share each genuine B bank".into());
        }
        first_b = Some(b0);
        next_b = Some(b1);
    }
    let mut c_values = matrix_values.clone();
    for &p in &ps {
        c_values.insert(words[p + 2], p);
    }
    let c_proof = BDefinitions {
        matrices: &c_values,
        ..proof
    };
    for i in 0..4 {
        if c_proof.origins(words[ps[i + 4] + 5], BTreeSet::new())?
            != BTreeSet::from([words[ps[i] + 2]])
        {
            return Err("balanced wide32 exact chronological C dependency, no crosswire".into());
        }
    }
    if cubecl_wgpu::native_strata_atomic_or_descriptor_mask(words, 0x3f) != 1 << 5 {
        return Err("small local carry original AtomicOr-only fault".into());
    }
    Ok(())
}
#[cfg(test)]
pub(crate) fn negatives(
    words: &[u32],
    expert: bool,
    small: Option<(usize, bool, u32)>,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut mutations = Vec::new();
    let mut at = 5;
    let mut zero = None;
    let mut ps = Vec::new();
    let mut stores = Vec::new();
    while at < words.len() {
        let n = (words[at] >> 16) as usize;
        let op = words[at] & 65535;
        if op == 43 && n == 4 && words[at + 3] == 0 {
            zero = Some(words[at + 2]);
        }
        if op == 16 && n == 6 && words[at + 2] == 17 {
            mutations.push((at + 3, 1024));
        }
        if op == 71 && n == 4 && words[at + 2] == 33 && matches!(words[at + 3], 0 | 1 | 2) {
            mutations.push((at + 3, 9));
        }
        if op == 4459 {
            ps.push(at);
        }
        if op == 62 {
            stores.push(at);
        }
        at += n;
    }
    at = 5;
    while at < words.len() {
        let n = (words[at] >> 16) as usize;
        if words[at] & 65535 == 4458 {
            mutations.push((at + 3, zero.ok_or("local prefetch actual zero")?));
            break;
        }
        at += n;
    }
    if small.is_some_and(|(_, cohort, _)| cohort) {
        let mut at = 5;
        let mut floats = std::collections::BTreeSet::new();
        let mut actual = None;
        while at < words.len() {
            let n = (words[at] >> 16) as usize;
            let op = words[at] & 65535;
            if op == 22 && n == 3 && words[at + 2] == 16 {
                floats.insert(words[at + 1]);
            }
            if op == 23 && n == 4 && words[at + 3] == 4 && floats.contains(&words[at + 2]) {
                actual = Some(at + 3);
            }
            if op == 4457 {
                mutations.push((at + 5, zero.ok_or("local prefetch actual zero stride")?));
                break;
            }
            at += n;
        }
        mutations.push((
            actual.ok_or("local prefetch actual vec4 half pointer unit")?,
            3,
        ));
    }
    for (i, v) in mutations {
        let mut bad = words.to_vec();
        bad[i] = v;
        if validate_program(&bad, expert, small).is_ok() {
            return Err("local prefetch accepted geometry/binding/layout corruption".into());
        }
    }
    let row = if small.is_some() { 1 } else { 2 };
    let mut bad = words.to_vec();
    bad[ps[row] + 3] = words[ps[0] + 3];
    if validate_program(&bad, expert, small).is_ok() {
        return Err("local prefetch accepted weight-row crosswire".into());
    }
    let a = stores
        .iter()
        .copied()
        .find(|&a| words[a + 2] == words[ps[0] + 2])
        .ok_or("local prefetch first C store")?;
    let b = stores
        .iter()
        .copied()
        .find(|&a| words[a + 2] == words[ps[row] + 2])
        .ok_or("local prefetch other C store")?;
    let mut bad = words.to_vec();
    bad[a + 1] = words[b + 1];
    if validate_program(&bad, expert, small).is_ok() {
        return Err("local prefetch accepted C crosswire".into());
    }
    Ok(())
}
