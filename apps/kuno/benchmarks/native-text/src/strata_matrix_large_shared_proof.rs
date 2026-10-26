//! Closed proof for actual shared-panel LARGE prefix SPIR-V, never an alias
//! to the old global-load proof. All old validators remain unchanged.
use super::*;
pub(super) fn validate(words:&[u32],columns:u32)->std::result::Result<(),Box<dyn std::error::Error>> {
    if ![640,2560].contains(&columns) {return Err("shared LARGE fixed axes".into());}
    let expert=true;
    let small:Option<(usize,bool,u32)>=None;
    use std::collections::{BTreeMap, BTreeSet};
    let raw = if small.is_some() { 4 } else { 8 };
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
    let mut instruction_defs=BTreeMap::new();
    let mut builtins=BTreeMap::new();
    let mut array_types=BTreeMap::new();
    let mut struct_types=BTreeMap::new();
    let (mut at, mut block, mut wg, mut barriers, mut atomics) = (5, None, false, 0, 0);
    while at < words.len() {
        let n = (words[at] >> 16) as usize;
        let op = words[at] & 65535;
        if n == 0 || at + n > words.len() {
            return Err("global wide32 opcode extent".into());
        }
        if matches!(op, 59 | 61 | 65 | 66 | 67 | 83 | 124) {
            value_types.insert(words[at + 2], words[at + 1]);
        }
        if matches!(op,43|61|65|66|67|77|79|80|81|83|113|114|124|128|130|132|134|135|137|164..=199|245|334) {
            instruction_defs.insert(words[at+2],(op,words[at+3..at+n].to_vec()));
        }
        match op {
            28 if n==4 => {array_types.insert(words[at+1],(words[at+2],words[at+3]));}
            30 => {struct_types.insert(words[at+1],words[at+2..at+n].to_vec());}
            71 if n==4 && words[at+2]==11 => {builtins.insert(words[at+1],words[at+3]);}
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
                5
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
                4
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
    let role_of = |ty:u32| -> Option<u32> {
        defs.get(matrix_uses.get(&ty)?).and_then(|v|v.first()).copied()
    };
    let load_role = |at:usize| -> Option<u32> {role_of(words[at+1])};
    for (ty, ptr, layout) in &operand_roles {
        let (ty,ptr,layout)=(*ty,*ptr,*layout);
        let binding = bindings.get(&root(ptr)?).copied();
        let use_id = matrix_uses.get(&ty).ok_or("transpose actual CMMA type")?;
        let actual_use = defs.get(use_id).and_then(|v| v.first()).copied();
        let actual_layout = defs.get(&layout).and_then(|v| v.first()).copied();
        let shared=classes.get(&root(ptr)?)==Some(&4);
        if !if shared {
            matches!((actual_use,actual_layout),(Some(0),Some(0))|(Some(1),Some(1))) && binding.is_none()
        } else {
            matches!((binding,actual_use,actual_layout),(Some(0),Some(1),Some(1))|(Some(1),Some(0),Some(0)))
        } {
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
    let mut input_loads=0;
    let mut weight_loads=0;
    let mut shared_a=0;
    let mut shared_b=0;
    let mut shared_root=None;
    for &(_,at,ptr,stride) in &loads {
        let r=root(ptr)?;
        if classes.get(&r)==Some(&4) {
            if shared_root.replace(r).is_some_and(|old|old!=r) {return Err("shared panel multiple operand roots".into());}
            match load_role(at) {Some(0)=>shared_a+=1,Some(1)=>shared_b+=1,_=>return Err("shared panel matrix role".into())}
            if constant(stride,&defs,&value_ops,0)!=Some(8) {return Err("shared panel exact8 f32 pointee stride".into());}
            let genuine=value_types.get(&ptr).and_then(|ty|pointer_types.get(ty))
                .is_some_and(|(class,scalar)|*class==4 && float_widths.get(scalar)==Some(&32));
            if !genuine {return Err("shared panel actual scalar f32 pointer".into());}
        } else {
            if classes.get(&r)!=Some(&12) {return Err("shared fallback original StorageBuffer operand".into());}
            match bindings.get(&r) {Some(0)=>input_loads+=1,Some(1)=>weight_loads+=1,_=>return Err("shared fallback original operand binding".into())}
            let genuine=constant(stride,&defs,&value_ops,0)==Some(4)
                && value_types.get(&ptr).and_then(|ty|pointer_types.get(ty))
                .filter(|(class,_)|*class==12).and_then(|(_,ty)|vectors.get(ty))
                .is_some_and(|(scalar,width)|*width==4 && float_widths.get(scalar)==Some(&16));
            if !genuine {return Err("shared fallback exact packed vec4/f16 stride".into());}
        }
    }
    if (shared_a,shared_b,input_loads,weight_loads)!=(4,2,3,6) {return Err("shared fast6/fallback9 exact operand population".into());}
    let shared_root=shared_root.ok_or("shared panel missing actual slab")?;
    let physical_roots=classes.iter().filter(|(_,class)|**class==4).map(|(&id,_)|id).collect::<Vec<_>>();
    if physical_roots!=[shared_root] {return Err("shared panel must have ONE physical Workgroup slab".into());}
    let mut slab_type=pointer_types.get(value_types.get(&shared_root).ok_or("shared slab pointer type")?)
        .ok_or("shared slab variable pointer")?.1;
    if let Some(fields)=struct_types.get(&slab_type) {
        if fields.len()!=1 {return Err("shared slab actual single member layout block".into());}
        slab_type=fields[0];
    }
    let &(scalar,length)=array_types.get(&slab_type).ok_or("shared slab actual fixed array")?;
    if float_widths.get(&scalar)!=Some(&32) || constant(length,&defs,&value_ops,0)!=Some(2048) {
        return Err("shared slab exact2048 scalar f32 /8192bytes".into());
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
        let &(a, op) = ins.last().ok_or("global wide32 block terminator")?;
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
        != if expert && small.is_none() {
            BTreeSet::from([4, 8])
        } else {
            BTreeSet::from([raw])
        }
        || innermost.len() != if expert && small.is_none() { 2 } else { 1 }
    {
        return Err("local prefetch authentic K-loop population".into());
    }
    let normal = innermost
        .iter()
        .find(|c| c.2 == raw)
        .ok_or("local prefetch actual product loop")?;
    let ls = loads
        .iter()
        .filter(|p| normal.1.contains(&p.0))
        .collect::<Vec<_>>();
    let ps = products
        .iter()
        .filter(|p| normal.1.contains(&p.0))
        .map(|p| p.1)
        .collect::<Vec<_>>();
    let ws = ls
        .iter()
        .filter(|l| load_role(l.1)==Some(0))
        .map(|l| l.1)
        .collect::<Vec<_>>();
    let bs = ls
        .iter()
        .filter(|l| load_role(l.1)==Some(1))
        .map(|l| l.1)
        .collect::<Vec<_>>();
    if ws.len() != 4 || bs.len() != if small.is_some() { 1 } else { 2 } || ls.len() != 4 + bs.len()
    {
        return Err("local prefetch genuine current weight/input populations".into());
    }
    let pairs = if small.is_some() { 1 } else { 2 };
    if !(ws[0] < ws[1]
        && ws[1] < ps[0]
        && ps[pairs - 1] < ws[2]
        && ws[2] < ps[pairs]
        && ps[2 * pairs - 1] < ws[3]
        && ws[3] < ps[2 * pairs])
    {
        return Err("local prefetch authentic alternating two-A schedule".into());
    }
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
                (at + 3..at + n).step_by(2).map(|a| words[a]).collect(),
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
            // A revisited loop edge contributes no new store. The callers still
            // require the exact initial/current/backedge origin sets.
            if !seen.insert(block) {
                return Err("local prefetch operand reached an older iteration".into());
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
        region: &normal.1,
        defs: &full_defs,
        matrices: &matrix_values,
        aliases: &aliases,
        classes: &classes,
        parents: &parents,
        locations: &locations,
    };
    let mut cptrs = BTreeSet::new();
    for (i, &p) in ps.iter().enumerate() {
        if proof.origins(words[p + 3], BTreeSet::new())?
            != BTreeSet::from([words[ws[i / pairs] + 2]])
            || proof.origins(words[p + 4], BTreeSet::new())?
                != BTreeSet::from([words[bs[i % pairs] + 2]])
        {
            return Err(
                "local prefetch actual current A/B origins, no stale/crossed operands".into(),
            );
        }
        let d = full_defs
            .get(&words[p + 5])
            .ok_or("local prefetch actual C load")?;
        if d[0] & 65535 != 61 {
            return Err("local prefetch original mutable C storage".into());
        }
        let ptr = root(d[3])?;
        if classes.get(&ptr) != Some(&7) || !cptrs.insert(ptr) {
            return Err("local prefetch independent accumulator owners".into());
        }
        let stores = normal
            .1
            .iter()
            .flat_map(|b| blocks[b].iter())
            .filter(|&&(a, op)| op == 62 && words[a + 2] == words[p + 2])
            .map(|&(a, _)| words[a + 1])
            .collect::<Vec<_>>();
        if stores.len() != 1 || root(stores[0])? != ptr {
            return Err("local prefetch exact chronological C writeback".into());
        }
    }
    validate_panel_program(words,normal.0,&normal.1,&blocks,&edges,&instruction_defs,
        &parents,&classes,&bindings,&builtins,shared_root,&loads,&products)?;
    if expert && cubecl_wgpu::native_strata_atomic_or_descriptor_mask(words, 0x3f) != 1 << 5 {
        return Err("local prefetch actual AtomicOr-only fault descriptor".into());
    }
    Ok(())
}

type Definitions=std::collections::BTreeMap<u32,(u32,Vec<u32>)>;
type Linear=(i64,std::collections::BTreeMap<u32,i64>);
fn shared_root(mut id:u32,parents:&std::collections::BTreeMap<u32,u32>)->Option<u32> {
    let mut seen=std::collections::BTreeSet::new();
    while let Some(&parent)=parents.get(&id) {if !seen.insert(id){return None;}id=parent;}
    Some(id)
}
fn linear(id:u32,defs:&Definitions,parents:&std::collections::BTreeMap<u32,u32>,builtins:&std::collections::BTreeMap<u32,u32>,depth:usize)->Option<Linear> {
    use std::collections::BTreeMap;
    if depth>64 {return None;}
    let atom=||Some((0,BTreeMap::from([(id,1)])));
    let Some((op,d))=defs.get(&id) else {return atom();};
    let recurse=|id|linear(id,defs,parents,builtins,depth+1);
    match *op {
        43 if d.len()==1=>Some((i64::from(d[0]),BTreeMap::new())),
        83|113|114|124 if d.len()==1=>recurse(d[0]),
        61 if d.len()==1=>{
            let root=shared_root(d[0],parents)?;
            if let Some(builtin)=builtins.get(&root) {
                Some((0,BTreeMap::from([(0xf000_0000+*builtin,1)])))
            } else {atom()}
        }
        128|130 if d.len()==2=>{
            let (a,mut terms)=recurse(d[0])?;let(b,right)=recurse(d[1])?;
            let sign=if *op==128 {1}else{-1};
            for (key,value) in right {*terms.entry(key).or_insert(0)+=sign*value;}
            terms.retain(|_,v|*v!=0);Some((a+sign*b,terms))
        }
        196 if d.len()==2=>{
            let (c,terms)=recurse(d[0])?;
            let shift=constant_value(d[1],defs)?;
            let factor=1i64.checked_shl(shift)?;
            Some((c*factor,terms.into_iter().map(|(key,value)|(key,value*factor)).collect()))
        }
        194 if d.len()==2=>{
            let (c,terms)=recurse(d[0])?;
            let shift=constant_value(d[1],defs)?;
            let divisor=1i64.checked_shl(shift)?;
            if c%divisor==0 && terms.values().all(|v|*v%divisor==0) {
                Some((c/divisor,terms.into_iter().map(|(key,value)|(key,value/divisor)).collect()))
            } else {atom()}
        }
        132 if d.len()==2=>{
            let (a,left)=recurse(d[0])?;let (b,right)=recurse(d[1])?;
            if left.is_empty() {Some((a*b,right.into_iter().map(|(k,v)|(k,v*a)).collect()))}
            else if right.is_empty() {Some((a*b,left.into_iter().map(|(k,v)|(k,v*b)).collect()))}
            else {atom()}
        }
        _=>atom(),
    }
}
fn constant_value(id:u32,defs:&Definitions)->Option<u32> {
    match defs.get(&id)? { (43,d) if d.len()==1=>Some(d[0]),
        (83|113|114|124,d) if d.len()==1=>constant_value(d[0],defs),_=>None }
}
fn dependencies(id:u32,defs:&Definitions)->std::collections::BTreeSet<u32> {
    let(mut seen,mut todo)=(std::collections::BTreeSet::new(),vec![id]);
    while let Some(id)=todo.pop() {
        if !seen.insert(id) {continue;}
        if let Some((op,d))=defs.get(&id) {
            // Never treat constant bits, component literals, or scope IDs as
            // arbitrary value edges. Preserve real access/index dependencies.
            match *op {
                43=>{},81=>todo.extend(d.first().copied()),
                79=>todo.extend(d.iter().take(2).copied()),
                _=>todo.extend(d.iter().copied().filter(|id|defs.contains_key(id))),
            }
        }
    }
    seen
}
fn address_index(id:u32,defs:&Definitions)->Option<u32> {
    let (op,d)=defs.get(&id)?;
    match *op {65|66|67=>d.last().copied(),83|124=>address_index(*d.first()?,defs),_=>None}
}
#[allow(clippy::too_many_arguments)]
fn validate_panel_program(
    words:&[u32],header:u32,region:&std::collections::BTreeSet<u32>,
    blocks:&std::collections::BTreeMap<u32,Vec<(usize,u32)>>,
    edges:&std::collections::BTreeMap<u32,Vec<u32>>,
    defs:&Definitions,parents:&std::collections::BTreeMap<u32,u32>,
    classes:&std::collections::BTreeMap<u32,u32>,bindings:&std::collections::BTreeMap<u32,u32>,
    builtins:&std::collections::BTreeMap<u32,u32>,slab:u32,
    loads:&[(u32,usize,u32,u32)],products:&[(u32,usize)],
)->std::result::Result<(),Box<dyn std::error::Error>> {
    use std::collections::{BTreeMap,BTreeSet};
    let root=|id|shared_root(id,parents).ok_or("shared panel actual root cycle");
    let body=region.iter().flat_map(|b|blocks[b].iter().copied()).collect::<Vec<_>>();
    let mut barriers=body.iter().filter(|(_,op)|*op==224).map(|(at,_)|*at).collect::<Vec<_>>();
    barriers.sort();
    if barriers.len()!=2 {return Err("shared panel exact TWO WG barriers per K16".into());}
    for &at in &barriers {
        if constant_value(words[at+1],defs)!=Some(2) || constant_value(words[at+2],defs)!=Some(2)
            || constant_value(words[at+3],defs).is_none_or(|bits|bits&0x100==0 || bits&0x8==0) {
            return Err("shared panel Workgroup execution/memory scope and acquire-release shared semantics".into());
        }
    }
    let mut stage_loads=BTreeMap::new();
    let mut stage_stores=Vec::new();
    let mut casts=BTreeMap::new();
    for &(at,op) in &body {
        match op {
            61 if [Some(&0),Some(&1)].contains(&bindings.get(&root(words[at+3])?)) => {
                stage_loads.insert(words[at+2],(at,*bindings.get(&root(words[at+3])?).unwrap()));
            }
            62 if root(words[at+1])?==slab=>stage_stores.push(at),
            124 => {casts.insert(words[at+2],(at,words[at+3]));},
            // Packed FP16 bits must never be numerically converted or used by
            // floating arithmetic during staging. Matrix math is separate4459.
            109|110|111|112|115|116|127|129|131|133|136|140|141=>{
                return Err("shared panel fast loop added numeric conversion/arithmetic".into());
            }
            _=>{}
        }
    }
    if stage_loads.len()!=4 || stage_stores.len()!=8 || casts.len()!=4 {
        return Err(format!("shared panel exact4 vec4 loads/4 bitcasts/8 pair writes: {}/{}/{}",stage_loads.len(),casts.len(),stage_stores.len()).into());
    }
    if stage_loads.values().filter(|(_,binding)|*binding==0).count()!=2
        || stage_loads.values().filter(|(_,binding)|*binding==1).count()!=2 {
        return Err("shared panel exactly two reads per original input/weight owner".into());
    }
    let mut types=BTreeMap::new();let mut floats=BTreeMap::new();let mut vectors=BTreeMap::new();let mut matrix_uses=BTreeMap::new();
    let mut at=5;
    while at<words.len() {
        let n=(words[at]>>16) as usize;let op=words[at]&65535;
        if matches!(op,61|124) {types.insert(words[at+2],words[at+1]);}
        if op==22 {floats.insert(words[at+1],words[at+2]);}
        if op==23 {vectors.insert(words[at+1],(words[at+2],words[at+3]));}
        if op==4456 {matrix_uses.insert(words[at+1],words[at+6]);}
        at+=n;
    }
    let mut stage_casts=BTreeSet::new();
    let mut offsets=BTreeSet::new();
    let mut per_cast=BTreeMap::<u32,BTreeSet<u32>>::new();
    let mut cast_bank=BTreeMap::new();
    for &store in &stage_stores {
        if store>=barriers[0] {return Err("shared panel write after consumer barrier".into());}
        let value=words[store+2];
        let (op,d)=defs.get(&value).ok_or("shared pair scalar extraction definition")?;
        let (cast,component)=match *op {
            81 if d.len()==2=>(d[0],d[1]),
            77 if d.len()==2=>(d[0],constant_value(d[1],defs).ok_or("shared pair constant component")?),
            _=>return Err("shared pair stores must be literal component extracts".into()),
        };
        let &(cast_at,source)=casts.get(&cast).ok_or("shared pair must originate in OpBitcast")?;
        let &(load_at,binding)=stage_loads.get(&source).ok_or("shared bitcast must directly consume actual owner vec4 load")?;
        if !(load_at<cast_at && cast_at<store) || component>1 {return Err("shared pair load/bitcast/store order".into());}
        let from=types.get(&source).and_then(|ty|vectors.get(ty));
        let to=types.get(&cast).and_then(|ty|vectors.get(ty));
        if !from.is_some_and(|(scalar,n)|*n==4 && floats.get(scalar)==Some(&16))
            || !to.is_some_and(|(scalar,n)|*n==2 && floats.get(scalar)==Some(&32)) {
            return Err("shared pair actual64bit vec4f16 to vec2f32 OpBitcast".into());
        }
        let index=address_index(words[store+1],defs).ok_or("shared pair actual array address")?;
        let (offset,terms)=linear(index,defs,parents,builtins,0).ok_or("shared pair affine address")?;
        if terms!=BTreeMap::from([(0xf000_0000+29,2)])
            || ![0,1,512,513,1024,1025,1536,1537].contains(&offset)
            || (binding==1)!=(offset<1024) || offset%2!=i64::from(component)
            || !offsets.insert(offset) {
            return Err("shared pair exact unique2*UNIT_POS + bank/component partition".into());
        }
        if cast_bank.insert(cast,offset-i64::from(component)).is_some_and(|old|old!=offset-i64::from(component)) {
            return Err("shared pair two components must retain the SAME bank/source vector".into());
        }
        stage_casts.insert(cast);per_cast.entry(cast).or_default().insert(component);
    }
    if stage_casts.len()!=4 || per_cast.values().any(|parts|*parts!=BTreeSet::from([0,1])) {
        return Err("shared pair every actual source bitcast written once completely".into());
    }
    // Every subgroup must finish stage writes before its genuine shared operand
    // loads; the second WG barrier follows the last DPAS and precedes reuse.
    let fast_loads=loads.iter().filter(|(b,_,_,_)|region.contains(b)).collect::<Vec<_>>();
    let fast_products=products.iter().filter(|(b,_)|region.contains(b)).collect::<Vec<_>>();
    if fast_loads.len()!=6 || fast_products.len()!=8
        || fast_loads.iter().any(|(_,at,ptr,_)|root(*ptr).ok()!=Some(slab)||*at<=barriers[0]||*at>=barriers[1])
        || fast_products.iter().any(|(_,at)|*at<=barriers[0]||*at>=barriers[1]) {
        return Err("shared panel authentic staged6 CMMA loads /8 chronological products between WG barriers".into());
    }
    // All six genuine shared matrix starts retain the exact4A/2B SG grid.
    // Pointer width changes cannot silently substitute a different panel.
    let mut a_offsets=Vec::new();let mut b_offsets=Vec::new();
    for &&(_,at,ptr,_) in &fast_loads {
        let index=address_index(ptr,defs).ok_or("shared CMMA actual panel offset")?;
        let (offset,terms)=linear(index,defs,parents,builtins,0).ok_or("shared CMMA panel affine offset")?;
        if terms.len()!=1 || terms.values().next()!=Some(&256) {return Err("shared CMMA exact subgroup panel coefficient".into());}
        let atom=*terms.keys().next().unwrap();
        let (operation,args)=defs.get(&atom).ok_or("shared CMMA actual subgroup quotient/remainder")?;
        let divisor=if matches!(*operation,137|134) {4}else if *operation==199 {3}else if *operation==194 {2}else {u32::MAX};
        if args.len()!=2 || constant_value(args[1],defs)!=Some(divisor) {return Err("shared CMMA four subgroup rows/columns".into());}
        let sg=linear(args[0],defs,parents,builtins,0).ok_or("shared CMMA actual subgroup source")?;
        if sg!=(0,BTreeMap::from([(0xf000_0000+40,1)])) {return Err("shared CMMA genuine SubgroupId".into());}
        let role=matrix_uses.get(&words[at+1]).and_then(|id|constant_value(*id,defs));
        match (*operation,role) {(137|199,Some(0))=>a_offsets.push(offset),(134|194,Some(1))=>b_offsets.push(offset),_=>return Err("shared CMMA exact A/remainder and B/quotient panels".into())}
        if at<=barriers[0]||at>=barriers[1] {return Err("shared CMMA lifetime".into());}
    }
    a_offsets.sort();b_offsets.sort();
    if a_offsets!=[0,64,128,192] || b_offsets!=[1024,1152] {return Err("shared CMMA exact unique fourA/twoB offsets".into());}
    // Both barriers must be unconditional along the fast K16 path. Conditional
    // branches inside the loop can only be the uniform loop-control branch.
    let body_branches=body.iter().filter(|(_,op)|*op==250).map(|(at,_)|*at).collect::<Vec<_>>();
    for branch in body_branches {
        let deps=dependencies(words[branch+1],defs);
        for id in deps {if let Some((61,d))=defs.get(&id) {
            if builtins.get(&root(d[0])?).is_some_and(|b|matches!(*b,27|28|29|40|41)) {
                return Err("shared K16 loop branch depends on invocation/subgroup".into());
            }
        }}
    }
    let reachable=|start:u32,stop:u32| {
        let(mut seen,mut todo)=(BTreeSet::new(),vec![start]);
        while let Some(b)=todo.pop() {if b==stop||!seen.insert(b){continue;}if let Some(next)=edges.get(&b){todo.extend(next);}}
        seen
    };
    let mut eligibility=false;
    for ins in blocks.values() {
        let Some(&(branch,250))=ins.last() else {continue;};
        let Some(&(merge,_))=ins.iter().rev().find(|(_,op)|*op==247) else {continue;};
        let condition=words[branch+1];
        let Some((334,all))=defs.get(&condition) else {continue;};
        if all.len()!=2 || constant_value(all[0],defs)!=Some(3) {continue;}
        let yes=reachable(words[branch+2],words[merge+1]);
        let no=reachable(words[branch+3],words[merge+1]);
        if !yes.contains(&header)||no.contains(&header) {continue;}
        if fast_products.iter().any(|(b,_)|!yes.contains(b)) {continue;}
        let deps=dependencies(all[1],defs);
        let mut route_indices=Vec::new();
        let mut comparisons=0;
        let mut nonuniform=false;
        for id in deps {
            let Some((op,d))=defs.get(&id) else {continue;};
            if *op==176 {comparisons+=1;}
            if *op==61 {
                let r=root(d[0])?;
                if builtins.get(&r).is_some_and(|b|matches!(*b,27|28|29|40)) {nonuniform=true;}
                if bindings.get(&r)==Some(&3) {
                    let index=address_index(d[0],defs).ok_or("shared route actual map address")?;
                    let expression=linear(index,defs,parents,builtins,0).ok_or("shared route affine map address")?;
                    if expression.1.get(&(0xf000_0000+41))==Some(&1) {route_indices.push(expression);}
                }
            }
        }
        route_indices.sort();route_indices.dedup();
        if nonuniform || comparisons<8 || route_indices.len()!=8 {continue;}
        let base_terms=&route_indices[0].1;
        let base_constant=route_indices[0].0;
        if route_indices.iter().enumerate().any(|(i,(offset,terms))|terms!=base_terms||*offset!=base_constant+(i*16) as i64) {continue;}
        // map[route_prefix+WGtile128+1540+lane+round16], with exactly
        // eight routes per lane, never a subgroup-position-dependent base.
        if base_constant!=1540 {continue;}
        eligibility=true;
    }
    if !eligibility {return Err("shared panel exact same128 immutable routes and subgroup-All fast/fallback CFG".into());}
    if classes.get(&slab)!=Some(&4) {return Err("shared panel slab storage class".into());}
    Ok(())
}

#[cfg(test)]
pub(super) fn negatives(words:&[u32],columns:u32)->std::result::Result<(),Box<dyn std::error::Error>> {
    let mut offsets=Vec::new();let mut products=Vec::new();let mut stores=Vec::new();
    let mut zero=None;let mut four=None;let mut subgroup=None;let mut at=5;
    while at<words.len() {
        let n=(words[at]>>16) as usize;let op=words[at]&65535;
        if op==43 && n==4 {match words[at+3] {0=>zero=Some(words[at+2]),3=>subgroup=Some(words[at+2]),4=>four=Some(words[at+2]),_=>{}}}
        if op==4459 {products.push(at);}
        if op==62 {stores.push(at);}
        at+=n;
    }
    let zero=zero.ok_or("shared negative actualzero")?;
    let four=four.ok_or("shared negative actualfour")?;
    let subgroup=subgroup.ok_or("shared negative actualsubgroup")?;
    at=5;
    while at<words.len() {
        let n=(words[at]>>16) as usize;let op=words[at]&65535;
        match op {
            16 if n==6 && words[at+2]==17=>offsets.push((at+3,64)),
            71 if n==4 && words[at+2]==33 && words[at+3]<=2=>offsets.push((at+3,9)),
            71 if n==4 && words[at+2]==11 && words[at+3]==41=>offsets.push((at+3,40)),
            43 if n==4 && words[at+3]==2048=>offsets.push((at+3,1024)),
            224=>offsets.push((at+1,subgroup)),
            124=>offsets.push((at,(n as u32)<<16|115)),
            4457=>offsets.push((at+5,four)),
            4458=>offsets.push((at+3,zero)),
            _=>{}
        }
        at+=n;
    }
    let mut rejected=0;
    for (offset,value) in offsets {
        if words[offset]==value {continue;}
        let mut bad=words.to_vec();bad[offset]=value;
        if validate(&bad,columns).is_ok() {return Err(format!("shared proof accepted corruption at{offset} value{value}").into());}
        rejected+=1;
    }
    if products.len()!=12 {return Err("shared negative genuine12 products".into());}
    let mut bad=words.to_vec();bad[products[2]+3]=words[products[0]+3];
    if validate(&bad,columns).is_ok() {return Err("shared proof accepted crossed A origins".into());}
    let mut bad=words.to_vec();bad[products[1]+4]=words[products[0]+4];
    if validate(&bad,columns).is_ok() {return Err("shared proof accepted crossed B origins".into());}
    let cstore=|p:usize|stores.iter().copied().find(|&at|words[at+2]==words[products[p]+2]).ok_or("shared negative original C store");
    let (a,b)=(cstore(0)?,cstore(2)?);let mut bad=words.to_vec();bad[a+1]=words[b+1];
    if validate(&bad,columns).is_ok() {return Err("shared proof accepted crossed C owner".into());}
    if rejected<20 {return Err("shared proof required actual mutation population".into());}
    Ok(())
}
