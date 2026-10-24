//! Compiled-public invalid-row sentinel initialization; no production caller.
//! Retains the original full owned output, without offset private uploads or
//! an output-sized host staging allocation. Independent full-owner scan follows.
use cubecl::prelude::*;
use cubecl::ir::ElemType;
use cubecl::server::Handle;
use cubecl::wgpu::WgpuRuntime;
type R<T> = std::result::Result<T,Box<dyn std::error::Error>>;
#[cube(launch_unchecked)]
fn seed_rows(output:&mut Array<u32>,first:usize,rows:usize,bits:u32){
 let i=usize::cast_from(ABSOLUTE_POS_X);
 if i<rows {output[first+i]=bits;}
}
fn extent(owned:usize,logical:usize,first:usize,rows:usize)->bool{
 rows>0&&logical<=owned&&first.checked_add(rows).is_some_and(|end|end<=logical)
}
pub(super) fn seed(client:&ComputeClient<WgpuRuntime>,owner:Handle,logical:usize,first:usize,rows:usize,bits:u32)->R<()> {
 if !extent(owner.size_in_used()as usize/4,logical,first,rows){return Err("wide32 sentinel original active-row extent".into());}
 unsafe {seed_rows::launch_unchecked::<WgpuRuntime>(client,CubeCount::Static(rows.div_ceil(256)as u32,1,1),CubeDim::new_1d(256),crate::resident_util::raw_arg(owner,logical)?,first,rows,bits);}
 Ok(())
}
pub(in crate::strata_matrix) fn validate_sources()->R<()> {
 use cubecl::wgpu::{AutoCompiler,AutoRepresentation};use cubecl::{Compiler,ExecutionMode,WgpuCompilationOptions};use std::io::Write;
 for mask in 0..8u8{
  let mut b=KernelBuilder::default();cubecl::ir::AddressType::U32.register(&mut b.scope);
  let output=b.output_array(u32::as_type(&b.scope));let first=b.scalar(u32::as_type(&b.scope).storage_type());let rows=b.scalar(u32::as_type(&b.scope).storage_type());let bits=b.scalar(u32::as_type(&b.scope).storage_type());
  seed_rows::expand(&mut b.scope,output.into(),first.into(),rows.into(),bits.into());assert!(b.scope.pop_errors().is_empty());
  let def=b.build(KernelSettings::default().cube_dim(CubeDim::new_1d(256)));
  let mut o=WgpuCompilationOptions{supports_vulkan:true,..Default::default()};o.vulkan.max_spirv_version=(1,6);o.vulkan.supports_uniform_standard_layout=mask&1!=0;o.vulkan.supports_uniform_unsized_array=mask&2!=0;o.vulkan.supports_explicit_smem=mask&4!=0;
  let AutoRepresentation::SpirV(k)=Compiler::compile(&mut AutoCompiler::SpirV(Default::default()),def,&o,ExecutionMode::Unchecked,cubecl::ir::StorageType::Scalar(ElemType::UInt(cubecl::ir::UIntKind::U32)))? else{return Err("wide32 sentinel actual SPIRV".into())};
  if k.shared_size!=0{return Err("wide32 sentinel actual zero scratch".into());}
  let words=&k.assembled_module;let(mut at,mut grid,mut matrices)=(5,false,0);
  while at<words.len(){let n=(words[at]>>16)as usize;if n==0||at+n>words.len(){return Err("wide32 sentinel opcode extent".into());}
   if words[at]&65535==16&&n==6&&words[at+2]==17{grid=words[at+3..at+6]==[256,1,1];}
   if(4456..=4459).contains(&(words[at]&65535)){matrices+=1;}at+=n;
  }
  if !grid||matrices!=0{return Err("wide32 sentinel bounded scalar shader".into());}
  let mut cmd=std::process::Command::new("spirv-val");cmd.args(["--target-env","vulkan1.3"]);if mask&1!=0{cmd.arg("--uniform-buffer-standard-layout");}
  let mut child=cmd.arg("-").stdin(std::process::Stdio::piped()).spawn()?;child.stdin.take().unwrap().write_all(&words.iter().flat_map(|w|w.to_le_bytes()).collect::<Vec<_>>())?;
  if !child.wait()?.success(){return Err("wide32 sentinel SPIRV validation".into());}
  println!("strata_global_wide32_fixture_shader_accepted layout={mask} shared_bytes=0 workgroup=256 no_offset_host_upload=true");
 }
 Ok(())
}
#[cfg(test)]
mod tests{
 use super::*;
 #[test]
 fn sentinel_gpu_seed_keeps_original_logical_rows_without_offset_upload_or_new_owner(){
  for rows in [1280usize,2560]{for first in [0usize,rows*8,40959*rows]{let logical=40960*rows;assert!(extent(logical+64,logical,first,rows));let mut touched=vec![0u8;rows+64];for lane in 0..rows.div_ceil(256)*256{if lane<rows{touched[lane]+=1;}}assert!(touched[..rows].iter().all(|&v|v==1)&&touched[rows..].iter().all(|&v|v==0));}}
  assert!(!extent(1024,1024,1023,2));assert!(!extent(1024,1025,0,1));assert!(!extent(1024,1024,0,0));assert!(!extent(usize::MAX,usize::MAX,usize::MAX,1));
  validate_sources().unwrap();
 }
}
