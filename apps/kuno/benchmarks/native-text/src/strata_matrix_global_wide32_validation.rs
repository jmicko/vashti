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
            return Err("global wide32 opcode extent".into());
        }
        match op {
            16 if n == 6 && words[at + 2] == 17 => wg = words[at + 3..at + 6] == [256, 1, 1],
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
            246 => loops.push((block.ok_or("global wide32 loop header")?, words[at + 1])),
            4456 => {matrix_uses.insert(words[at+1], words[at+6]);}
            4457 => {operand_roles.push((words[at+1],words[at+3],words[at+4])); loads.push((
                block.ok_or("global wide32 load block")?,
                at,
                words[at + 3],
                words[at + 5],
            ));},
            4458 => {stores.push(words[at + 1]); store_layouts.push(words[at+3]);},
            227 => {defs.insert(words[at+2],vec![words[at+3]]); atomics+=1;},
            4459 => products.push((block.ok_or("global wide32 product block")?, at)),
            224 => barriers += 1,
            227..=242 => atomics += 1,
            _ => {}
        }
        // Only value-producing instructions relevant to authentic dynamic strides.
        if matches!(
            op,
            43 | 170 | 171 | 227 | 334 | 61
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
        || loads.len() != if expert {15} else {6}
        || stores.len() != if expert {12} else {8}
        || products.len() != if expert {12} else {8}
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
    for (ty,ptr,layout) in operand_roles {
        let binding=bindings.get(&root(ptr)?).copied();
        let use_id=matrix_uses.get(&ty).ok_or("transpose actual CMMA type")?;
        let actual_use=defs.get(use_id).and_then(|v|v.first()).copied();
        let actual_layout=defs.get(&layout).and_then(|v|v.first()).copied();
        if !matches!((binding,actual_use,actual_layout),(Some(0),Some(1),Some(1))|(Some(1),Some(0),Some(0))) {
            return Err("transpose actual weight-A rowmajor/input-B colmajor operands".into());
        }
    }
    if store_layouts.iter().any(|id|defs.get(id).and_then(|v|v.first())!=Some(&1)) {
        return Err("transpose actual column-major output/fallback stores".into());
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
            return Err("global wide32 stride is not authentic dynamic scalar INFO".into());
        }
    }
    if input_loads != if expert {5}else{2} || weight_loads != if expert{10}else{4} {
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
    if global_stores != if expert{10}else{8} || fallback_stores != if expert { 2 } else { 0 } {
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
    let mut candidates=Vec::new();
    for (header,merge) in loops {
        let (mut region,mut todo)=(BTreeSet::new(),vec![header]);
        while let Some(b)=todo.pop(){if b==merge||!region.insert(b){continue;}todo.extend(edges.get(&b).ok_or("wide32 CFG edge")?.iter().copied());}
        let count=products.iter().filter(|p|region.contains(&p.0)).count();
        if count>0{candidates.push((header,region,count));}
    }
    let mut innermost=Vec::new();
    for (i,c) in candidates.iter().enumerate(){if !candidates.iter().enumerate().any(|(j,d)|i!=j&&d.1.is_subset(&c.1)&&d.1!=c.1){innermost.push(c);}}
    let counts: BTreeSet<usize>=innermost.iter().map(|c|c.2).collect();
    if counts!=if expert{BTreeSet::from([4,8])}else{BTreeSet::from([8])}||innermost.len()!=if expert{2}else{1}{return Err("wide32 genuine raw/checked product-loop partition".into());}
    let normal=innermost.iter().find(|c|c.2==8).unwrap();
    if products.iter().filter(|p|normal.1.contains(&p.0)).map(|p|p.0).collect::<BTreeSet<_>>().len()!=1{return Err("wide32 chronological eight products in one actual loop body".into());}
    let mut ops=Vec::new();
    for b in &normal.1{ops.extend(blocks[b].iter().copied());}
    if ops.iter().any(|(_,op)|*op==224||(227..=242).contains(op)) {return Err("wide32 raw product loop publication/atomic".into());}
    let normal_loads=loads.iter().filter(|p|normal.1.contains(&p.0)).collect::<Vec<_>>();
    if normal_loads.len()!=6{return Err("wide32 six global raw-loop loads".into());}
    let in_loads=normal_loads.iter().filter(|p|bindings.get(&root(p.2).unwrap())==Some(&0)).count();
    if in_loads!=2{return Err("wide32 two input B and four weight A fragments".into());}
    let ps=products.iter().filter(|p|normal.1.contains(&p.0)).map(|p|p.1).collect::<Vec<_>>();
    let ws=normal_loads.iter().filter(|p|bindings.get(&root(p.2).unwrap())==Some(&1)).map(|p|p.1).collect::<Vec<_>>();
    for i in 0..4 {if ws[i]>=ps[2*i] || (i>0&&ws[i]<=ps[2*i-1]){return Err("wide32 actual one-A/two-product lifetime order".into());}}
    if expert {
        let mut fault_reads=BTreeSet::new();let mut all_values=BTreeSet::new();let mut branches=BTreeSet::new();
        let mut at=5;
        while at<words.len(){let n=(words[at]>>16)as usize;
            if words[at]&65535==227 && bindings.get(&root(words[at+3])?)==Some(&5){fault_reads.insert(words[at+2]);}
            if words[at]&65535==334{all_values.insert(words[at+2]);}
            if words[at]&65535==250{branches.insert(words[at+1]);}
            at+=n;
        }
        let mut guarded=false;
        for all in all_values.intersection(&branches){let mut seen=BTreeSet::new();let mut todo=vec![*all];while let Some(v)=todo.pop(){if !seen.insert(v){continue;}if fault_reads.contains(&v){guarded=true;}if let Some(d)=defs.get(&v){todo.extend(d.iter().copied().filter(|v|defs.contains_key(v)));}}}
        if !guarded{return Err("wide32 subgroup-uniform original sticky-fault branch".into());}
    }
    Ok(())
}
pub(in crate::strata_matrix) fn validate_sources() -> std::result::Result<(), Box<dyn std::error::Error>> {
    use cubecl::wgpu::{AutoCompiler, AutoRepresentation};
    use cubecl::{Compiler, ExecutionMode, WgpuCompilationOptions};
    use std::io::Write;
    for mask in 0..8u8 {
        for (name, definition) in [
            ("fp16", global_wide32_definition::<f16>(false)),
            ("bf16", global_wide32_definition::<bf16>(false)),
            ("gr_bf16", global_wide32_definition::<bf16>(true)),
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
                return Err("global wide32 actual SPIRV".into());
            };
            if k.shared_size != 0 {
                return Err("global wide32 actual zero shared bytes".into());
            }
            let words = wgpu::hal::vulkan::kuno_native_workgroup_layout(&k.assembled_module)?;
            let words = wgpu::hal::vulkan::kuno_cooperative_fp16_preserve(&words)?.unwrap_or(words);
            if wgpu::hal::vulkan::kuno_cooperative_subgroup(&words) != Some(16) {
                return Err("global wide32 actual SG16".into());
            }
            validate_program(&words,false)?;
            #[cfg(test)]
            validate_negatives(&words,false)?;


            let mut cmd = std::process::Command::new("spirv-val");
            cmd.args(["--target-env", "vulkan1.3"]);
            if mask & 1 != 0 {
                cmd.arg("--uniform-buffer-standard-layout");
            }
            let mut child = cmd.arg("-").stdin(std::process::Stdio::piped()).spawn()?;
            child
                .stdin
                .take()
                .ok_or("global wide32 validator stdin")?
                .write_all(
                    &words
                        .iter()
                        .flat_map(|w| w.to_le_bytes())
                        .collect::<Vec<_>>(),
                )?;
            if !child.wait()?.success() {
                return Err("global wide32 SPIRV validation".into());
            }
            println!(
                "strata_global_wide32_shader_accepted dtype={name} layout_mask={mask} shared_bytes=0 workgroup=256 subgroup=16 m=128 n=128 k=16 products=8 global_operand_loads=6 global_stores=8 whole_workgroup_handoffs=0"
            );
        }
    }
    Ok(())
}

#[cfg(test)]
pub(in crate::strata_matrix) fn validate_negatives(words:&[u32],expert:bool)->std::result::Result<(),Box<dyn std::error::Error>>{
 let (mut at,mut changes,mut zero)=(5,Vec::new(),None);
 while at<words.len(){let n=(words[at]>>16)as usize;let op=words[at]&65535;
  if op==43&&n==4&&words[at+3]==0{zero=Some(words[at+2]);}
  if op==16&&n==6&&words[at+2]==17{changes.push((at+3,1024));}
  if op==71&&n==4&&words[at+2]==33&&words[at+3]==0{changes.push((at+3,9));}
  if op==71&&n==4&&words[at+2]==33&&words[at+3]==if expert{6}else{3}{changes.push((at+3,10));}

  at+=n;
 }
 at=5;while at<words.len(){let n=(words[at]>>16)as usize;if words[at]&65535==4458{changes.push((at+3,zero.ok_or("wide32 actual zero constant")?));break;}at+=n;}
 if changes.len()<4{return Err("wide32 actual negative inventory".into());}
 if expert{let mut bad=words.to_vec();let mut at=5;while at<bad.len(){let n=(bad[at]>>16)as usize;if bad[at]&65535==334{bad[at]=(n as u32)<<16|335;}at+=n;}if validate_program(&bad,true).is_ok(){return Err("wide32 accepted non-all fault selection".into());}}
 for(i,value)in changes{let mut bad=words.to_vec();bad[i]=value;if validate_program(&bad,expert).is_ok(){return Err("wide32 accepted wrong grid/binding/INFO/store/fault-uniformity".into());}}
 Ok(())
}
