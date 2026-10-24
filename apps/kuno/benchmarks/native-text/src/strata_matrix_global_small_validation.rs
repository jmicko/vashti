// Actual emitted shader proofs for the whole new small/medium bundle.
use super::*;
use std::collections::{BTreeMap,BTreeSet};
fn program(words:&[u32],cohort:bool,tile:usize)->std::result::Result<(),Box<dyn std::error::Error>>{
 let(mut at,mut wg,mut block)=(5,false,0u32);
 let(mut parents,mut classes,mut bindings,mut constants,mut uses)=(BTreeMap::new(),BTreeMap::new(),BTreeMap::new(),BTreeMap::new(),BTreeMap::new());
 let(mut loads,mut stores,mut products,mut barriers,mut loops)=(Vec::new(),Vec::new(),Vec::new(),Vec::new(),Vec::new());
 let mut blocks:BTreeMap<u32,Vec<usize>>=BTreeMap::new();
 while at<words.len(){let n=(words[at]>>16)as usize;let op=words[at]&65535;
  if n==0||at+n>words.len(){return Err("global small opcode extent".into());}
  match op {
   16 if n==6&&words[at+2]==17=>wg=words[at+3..at+6]==[if tile==64{64}else{32},1,1],
   71 if n==4&&words[at+2]==33=>{bindings.insert(words[at+1],words[at+3]);},
   59=>{classes.insert(words[at+2],words[at+3]);},
   43 if n==4=>{constants.insert(words[at+2],words[at+3]);},
   65|66|67|83|124=>{parents.insert(words[at+2],words[at+3]);},
   248=>{block=words[at+1];blocks.insert(block,Vec::new());},
   246=>loops.push((block,words[at+1])),
   4456=>{uses.insert(words[at+1],words[at+6]);},
   4457=>loads.push((block,at,words[at+1],words[at+3],words[at+4])),
   4458=>stores.push((words[at+1],words[at+3])),
   4459=>products.push((block,at)),
   224=>barriers.push(words[at+1]),
   227=>return Err("global small fault read is forbidden".into()),
   _=>{}
  }
  if block!=0{blocks.get_mut(&block).unwrap().push(at);}at+=n;
 }
 let root=|mut v:u32|->std::result::Result<u32,Box<dyn std::error::Error>>{
  let mut seen=BTreeSet::new();while let Some(&p)=parents.get(&v){if !seen.insert(v){return Err("global small pointer cycle".into());}v=p;}Ok(v)
 };
 if !wg||cubecl_wgpu::native_strata_atomic_or_descriptor_mask(words,0x3f)!=1<<5{return Err("global small actual WG/original AtomicOr-only fault".into());}
 if barriers.iter().any(|s|constants.get(s)!=Some(&3)){return Err("global small workgroup barrier under subgroup branches".into());}
 for &(_,_,ty,ptr,layout)in &loads{
  let r=root(ptr)?;let use_=constants.get(uses.get(&ty).ok_or("global small actual matrix type")?).copied();let layout=constants.get(&layout).copied();
  if !matches!((classes.get(&r).copied(),bindings.get(&r).copied(),use_,layout),(Some(12),Some(1),Some(0),Some(0))|(Some(12),Some(0),Some(1),Some(1))|(Some(4),None,Some(1),Some(1))){return Err("global small weight-A/global-or-masked-input-B provenance/layout".into());}
 }
 for &(ptr,layout)in &stores{let r=root(ptr)?;
  if constants.get(&layout)!=Some(&1)||!((classes.get(&r)==Some(&12)&&bindings.get(&r)==Some(&2))||classes.get(&r)==Some(&4)){return Err("global small original column-major output/publication owner".into());}
 }
 let mut edges=BTreeMap::new();
 for(&b,ins)in &blocks{let a=*ins.last().ok_or("global small empty block")?;let n=(words[a]>>16)as usize;
  edges.insert(b,match words[a]&65535{249=>vec![words[a+1]],250=>vec![words[a+2],words[a+3]],251=>std::iter::once(words[a+2]).chain((a+4..a+n).step_by(2).map(|i|words[i])).collect(),_=>Vec::new()});
 }
 let mut regions=Vec::new();
 for(header,merge)in loops{let(mut seen,mut todo)=(BTreeSet::new(),vec![header]);while let Some(b)=todo.pop(){if b==merge||!seen.insert(b){continue;}todo.extend(edges.get(&b).ok_or("global small missing CFG edge")?.iter().copied());}
  let count=products.iter().filter(|p|seen.contains(&p.0)).count();if count>0{regions.push((seen,count));}
 }
 let innermost=regions.iter().enumerate().filter(|(i,c)|!regions.iter().enumerate().any(|(j,d)|*i!=j&&d.0!=c.0&&d.0.is_subset(&c.0))).map(|(_,c)|c).collect::<Vec<_>>();
 let raw_count=if cohort{4}else{8};
 if innermost.len()!=2||innermost.iter().map(|r|r.1).collect::<BTreeSet<_>>()!=BTreeSet::from([2,raw_count]){return Err("global small raw/checked chronological product loops".into());}
 let raw=innermost.iter().find(|r|r.1==raw_count).unwrap();
 let raw_loads=loads.iter().filter(|p|raw.0.contains(&p.0)).collect::<Vec<_>>();
 let weight=raw_loads.iter().filter(|p|bindings.get(&root(p.3).unwrap())==Some(&1)).map(|p|p.1).collect::<Vec<_>>();
 let input=raw_loads.iter().filter(|p|bindings.get(&root(p.3).unwrap())==Some(&0)).count();
 let ps=products.iter().filter(|p|raw.0.contains(&p.0)).map(|p|p.1).collect::<Vec<_>>();
 if weight.len()!=4||input!=if cohort{1}else{2}||raw_loads.len()!=4+input||ps.len()!=raw_count{return Err("global small real normal operand population".into());}
 for i in 0..4{let n=if cohort{1}else{2};if weight[i]>=ps[i*n]||(i>0&&weight[i]<=ps[i*n-1]){return Err("global small bounded one-A reuse chronology".into());}}
 for b in &raw.0{for &a in &blocks[b]{let op=words[a]&65535;if op==224||(227..=242).contains(&op){return Err("global small raw product-loop publication/atomic".into());}}}
 if !cohort&&!loads.iter().any(|p|classes.get(&root(p.3).unwrap())==Some(&4)){return Err("global small partial input matrix load missing".into());}
 Ok(())
}
#[cfg(test)]
fn negatives(words:&[u32],cohort:bool,tile:usize)->std::result::Result<(),Box<dyn std::error::Error>>{
 let mut changes=Vec::new();let mut at=5;
 while at<words.len(){let n=(words[at]>>16)as usize;let op=words[at]&65535;
  if op==16&&n==6&&words[at+2]==17{changes.push((at+3,256));}
  if op==71&&n==4&&words[at+2]==33&&matches!(words[at+3],0|1|2|5){changes.push((at+3,29));}
  at+=n;
 }
 if changes.len()<5{return Err("global small genuine negative inventory".into());}
 for(i,v)in changes{let mut w=words.to_vec();w[i]=v;if program(&w,cohort,tile).is_ok(){return Err("global small accepted wrong actual grid/owner/role".into());}}
 Ok(())
}
pub(crate) fn validate_sources()->std::result::Result<(),Box<dyn std::error::Error>>{
 use cubecl::wgpu::{AutoCompiler,AutoRepresentation};use cubecl::{Compiler,ExecutionMode,WgpuCompilationOptions};use std::io::Write;
 for mask in 0..8u8{
  let defs=definitions().into_iter().map(|(n,d)|(n,d,false,if n.contains("macro64"){64usize}else{32})).chain(crate::strata_matrix::cohorts::global_small::definitions().into_iter().map(|(n,d)|(n,d,true,16usize)));
  for(name,def,cohort,tile)in defs{
   let mut o=WgpuCompilationOptions{supports_vulkan:true,..Default::default()};o.vulkan.max_spirv_version=(1,6);o.vulkan.supports_uniform_standard_layout=mask&1!=0;o.vulkan.supports_uniform_unsized_array=mask&2!=0;o.vulkan.supports_explicit_smem=mask&4!=0;
   let AutoRepresentation::SpirV(k)=Compiler::compile(&mut AutoCompiler::SpirV(Default::default()),def,&o,ExecutionMode::Unchecked,cubecl::ir::StorageType::Scalar(ElemType::UInt(cubecl::ir::UIntKind::U32)))?else{return Err("global small actual SPIRV".into());};
   // Final exact per-layout resource populations are frozen after these real
   // modules compile; never assume larger scratch than the preceding kernels.
   if k.shared_size>if tile==64{4096}else{2048}{return Err(format!("global small bounded real scratch {}",k.shared_size).into());}
   let words=wgpu::hal::vulkan::kuno_native_workgroup_layout(&k.assembled_module)?;let words=wgpu::hal::vulkan::kuno_cooperative_fp16_preserve(&words)?.unwrap_or(words);
   if wgpu::hal::vulkan::kuno_cooperative_subgroup(&words)!=Some(16){return Err("global small authentic SG16".into());}
   program(&words,cohort,tile)?;
   #[cfg(test)]negatives(&words,cohort,tile)?;
   let mut cmd=std::process::Command::new("spirv-val");cmd.args(["--target-env","vulkan1.3"]);if mask&1!=0{cmd.arg("--uniform-buffer-standard-layout");}
   let mut child=cmd.arg("-").stdin(std::process::Stdio::piped()).spawn()?;child.stdin.take().ok_or("global small validator input")?.write_all(&words.iter().flat_map(|w|w.to_le_bytes()).collect::<Vec<_>>())?;
   if !child.wait()?.success(){return Err("global small validated actual module".into());}
   println!("strata_global_small_shader_accepted name={name} layout={mask} shared_bytes={} workgroup={} subgroup=16 weights_first=true checked_original_halves=true",k.shared_size,if tile==64{64}else{32});
  }
 }
 Ok(())
}
