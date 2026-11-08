//! Compile the actual new prefix/tail programs under all eight original layouts.
use super::*;
pub(crate) fn validate_sources() -> std::result::Result<(),Box<dyn std::error::Error>> {
    use cubecl::wgpu::{AutoCompiler,AutoRepresentation};
    use cubecl::{Compiler,WgpuCompilationOptions};
    use std::io::Write;
    for mask in 0..8u8 {for (name,definition) in definitions() {
        let prefix=name.starts_with("large_cohort128_");
        let mut options=WgpuCompilationOptions{supports_vulkan:true,..Default::default()};
        options.vulkan.max_spirv_version=(1,6);
        options.vulkan.supports_uniform_standard_layout=mask&1!=0;
        options.vulkan.supports_uniform_unsized_array=mask&2!=0;
        options.vulkan.supports_explicit_smem=mask&4!=0;
        let AutoRepresentation::SpirV(kernel)=Compiler::compile(&mut AutoCompiler::SpirV(Default::default()),
            definition,&options,ExecutionMode::Unchecked,
            StorageType::Scalar(ElemType::UInt(UIntKind::U32)))? else {return Err("large cohort actual compiler representation".into());};
        let original=if prefix {8192}else if name.starts_with("large_cohort32_") {
            if mask&4!=0 {24576}else{26624}
        }else if name.starts_with("large_cohort8_") {
            if mask&4!=0 {18432}else{20480}
        }else{if mask&4!=0 {20480}else{22528}};
        if kernel.shared_size!=original {return Err(format!("large cohort original SLM envelope {name}: {} != {original}",kernel.shared_size).into());}
        let words=wgpu::hal::vulkan::kuno_native_workgroup_layout(&kernel.assembled_module)?;
        let words=wgpu::hal::vulkan::kuno_cooperative_subgroup_shared_barriers(&words)?.unwrap_or(words);
        let words=wgpu::hal::vulkan::kuno_cooperative_fp16_preserve(&words)?.unwrap_or(words);
        if wgpu::hal::vulkan::kuno_cooperative_subgroup(&words)!=Some(16) {return Err("large cohort original SG16".into());}
        if prefix {
            let columns=if name.ends_with("_down"){640}else{2560};
            crate::strata_matrix::local_validation::validate_large_cohort_prefix(&words,columns)?;
            #[cfg(feature="strata-large-packed-tiles")]
            if crate::strata_matrix::local_validation::validate_large_packed_cohort_prefix(&words,columns).is_ok() {
                return Err("original row-major prefix accepted as packed operands".into());
            }
            #[cfg(test)]
            crate::strata_matrix::local_validation::large_prefix_negatives(&words,columns)?;
        }
        let mut at=5;let(mut products,mut fault,mut stores)=(0,0,0);
        while at<words.len() {
            let n=(words[at]>>16) as usize;if n==0||at+n>words.len(){return Err("large cohort SPIRV extent".into());}
            match words[at]&65535 {4459=>products+=1,241=>fault+=1,4458=>stores+=1,_=>{}}
            at+=n;
        }
        if products==0||fault==0||stores==0 {return Err("large cohort missing genuine matrix/publication/fault program".into());}
        let mut cmd=std::process::Command::new("spirv-val");cmd.args(["--target-env","vulkan1.3"]);
        if mask&1!=0 {cmd.arg("--uniform-buffer-standard-layout");}
        let mut child=cmd.arg("-").stdin(std::process::Stdio::piped()).spawn()?;
        child.stdin.take().ok_or("large cohort validator stdin")?.write_all(&words.iter().flat_map(|w|w.to_le_bytes()).collect::<Vec<_>>())?;
        if !child.wait()?.success(){return Err("large cohort actual SPIRV validation".into());}
        println!("strata_large_cohort_shader_accepted name={name} layout={mask} shared_bytes={original} subgroup=16");
    }}
    Ok(())
}
#[cfg(test)]
#[test]
fn large_cohort_all_actual_prefix_tail_programs_compile() {validate_sources().unwrap();}
