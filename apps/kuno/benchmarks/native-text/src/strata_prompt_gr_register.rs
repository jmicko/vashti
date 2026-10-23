//! Coupled original scalar R registers and native-SG32 pair packing.
use super::*;
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct RegisterNorm(bool);
impl KernelMetadata for RegisterNorm {
    fn name(&self) -> &'static str {
        if self.0 {
            "StrataPromptGrRegisterNormalizeInPlace"
        } else {
            "StrataPromptGrRegisterNormalize"
        }
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
            .info(self.0)
            .cube_dim(CubeDim::new_1d(256))
    }
    fn address_type(&self) -> StorageType {
        StorageType::Scalar(ElemType::UInt(UIntKind::U32))
    }
}
impl CubeTask<AutoCompiler> for RegisterNorm {
    fn compile(
        &self,
        _: &mut AutoCompiler,
        _: &cubecl::WgpuCompilationOptions,
        _: ExecutionMode,
        _: StorageType,
    ) -> std::result::Result<CompiledKernel<AutoCompiler>, CompilationError> {
        Ok(CompiledKernel {
            entrypoint_name: "main".into(),
            debug_name: None,
            source: source(self.0),
            repr: None,
            cube_dim: CubeDim::new_1d(256),
            debug_info: None,
        })
    }
}
fn source(in_place: bool) -> String {
    let old = Kernel::GrNorm.source();
    let first = "    for(var col=tid;col<2560u;col+=256u)";
    let mut s = old
        .split_once(first)
        .expect("original GR scalar map")
        .0
        .to_owned();
    // Explicit complete-SG32 logical topology, never an assumed local-ID/lane map.
    s = s
        .replace(
            "@builtin(local_invocation_index) tid:u32",
            "@builtin(local_invocation_index) physical_tid:u32",
        )
        .replace("    let t=g.y;", "    let tid=sg*32u+lane;let t=g.y;");
    for k in 0..10 {
        let d = k * 256;
        s.push_str(&format!("    var r{k}=residual[base+tid+{d}u];if p.apply!=0u {{r{k}=fma(block[t*2560u+tid+{d}u],injection[t*4u+h],previous[base+tid+{d}u]);residual[base+tid+{d}u]=r{k};}}sum+=r{k}*r{k};\n"));
    }
    let reduction = old
        .split_once("    for(var d=16u;d>0u;d/=2u)")
        .unwrap()
        .1
        .split_once("    for(var pair=tid;")
        .unwrap()
        .0;
    s.push_str("    for(var d=16u;d>0u;d/=2u)");
    s.push_str(reduction);
    for k in 0..10 {
        let d = k * 256;
        s.push_str(&format!("    let a{k}=(r{k}*rs)*gamma[h*2560u+tid+{d}u];let b{k}=subgroupShuffleXor(a{k},1u);if (lane&1u)==0u {{let bits=image(a{k},b{k},1u);let good=finite_value(a{k})&&finite_value(b{k})&&finite_value(restored(bits,false,1u))&&finite_value(restored(bits,true,1u));normalized[t*5120u+h*1280u+(tid+{d}u)/2u]=select(0u,bits,good);if !good {{atomicOr(&fault[0],1u);}}}}\n"));
    }
    s.push_str("}\n");
    if in_place {
        let declaration = "@group(0) @binding(0) var<storage,read> previous:array<f32>;\n";
        assert!(s.contains(declaration));
        s = s.replace(declaration, "").replace("previous[", "residual[");
        for i in 1..=8 {
            s = s.replace(&format!("@binding({i})"), &format!("@binding({})", i - 1));
        }
    }
    s
}
fn dispatch(
    client: &ComputeClient<WgpuRuntime>,
    new: bool,
    in_place: bool,
    handles: Vec<Handle>,
    info: [u32; 16],
) -> Result<()> {
    let grid = dispatch_grid(Kernel::GrNorm, [4, info[0] as usize, 1])?;
    if info[0] == 0
        || handles.len() != if in_place { 7 } else { 8 }
        || handles.iter().any(|h| h.stream != handles[0].stream)
        || client.properties().hardware.max_units_per_cube < 256
    {
        return Err(invalid_buffer_length());
    }
    let mut args = KernelArguments::new();
    for h in handles {
        args = args.with_buffer(h.binding());
    }
    args = args.with_info(MetadataBindingInfo::custom(
        info.chunks_exact(2)
            .map(|v| v[0] as u64 | ((v[1] as u64) << 32))
            .collect(),
    ));
    let task: Box<dyn CubeTask<AutoCompiler>> = if new {
        Box::new(RegisterNorm(in_place))
    } else {
        Box::new(if in_place {
            Kernel::GrNormInPlace
        } else {
            Kernel::GrNorm
        })
    };
    // SAFETY: production caller retains original validated GR owners and scope;
    // the separate fixed-public comparator validates each authentic arena owner.
    unsafe {
        client.launch_unchecked(
            task,
            CubeCount::Static(grid[0] as u32, grid[1] as u32, 1),
            args,
        );
    }
    Ok(())
}
pub(super) fn launch(
    client: &ComputeClient<WgpuRuntime>,
    in_place: bool,
    h: Vec<Handle>,
    info: [u32; 16],
) -> Result<()> {
    if !active() {
        return Err(invalid_buffer_length());
    }
    dispatch(client, true, in_place, h, info)?;
    crate::strata_prefill_profile::count("gr_register_norm_products", 1);
    Ok(())
}
fn validate_one(in_place: bool) -> std::result::Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;
    use wgpu::naga as n;
    let src = source(in_place);
    let old = if in_place {
        Kernel::GrNormInPlace
    } else {
        Kernel::GrNorm
    }
    .source();
    let module = n::front::wgsl::parse_str(&src).map_err(|e| e.emit_to_string(&src))?;
    let previous = n::front::wgsl::parse_str(&old).map_err(|e| e.emit_to_string(&old))?;
    let info = n::valid::Validator::new(
        n::valid::ValidationFlags::all(),
        n::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| e.emit_to_string(&src))?;
    let abi = |m: &n::Module| {
        m.global_variables
            .iter()
            .filter_map(|(_, v)| {
                v.binding.as_ref().map(|b| {
                    (
                        b.group,
                        b.binding,
                        v.name.clone(),
                        format!("{:?}", v.space),
                        format!("{:?}", m.types[v.ty].inner),
                    )
                })
            })
            .collect::<Vec<_>>()
    };
    if abi(&module) != abi(&previous)
        || module.entry_points.len() != 1
        || module.entry_points[0].workgroup_size != [256, 1, 1]
    {
        return Err("GR register original ABI/workgroup drift".into());
    }
    let entry = &module.entry_points[0].function;
    for k in 0..10 {
        let var = entry
            .local_variables
            .iter()
            .find(|(_, v)| v.name.as_deref() == Some(&format!("r{k}")))
            .ok_or("GR ten explicit residual registers")?
            .1;
        if !matches!(
            module.types[var.ty].inner,
            n::TypeInner::Scalar(n::Scalar {
                kind: n::ScalarKind::Float,
                width: 4
            })
        ) {
            return Err("GR residual must remain scalar F32".into());
        }
    }
    let rs = entry
        .named_expressions
        .iter()
        .find(|(_, s)| s.as_str() == "rs")
        .ok_or("GR original row scale")?
        .0
        .index();
    fn global(
        f: &n::Function,
        p: n::Handle<n::Expression>,
    ) -> Option<n::Handle<n::GlobalVariable>> {
        match f.expressions[p] {
            n::Expression::GlobalVariable(g) => Some(g),
            n::Expression::Access { base, .. } | n::Expression::AccessIndex { base, .. } => {
                global(f, base)
            }
            _ => None,
        }
    }
    for (i, e) in entry.expressions.iter() {
        if let n::Expression::Load { pointer } = *e {
            if let Some(g) = global(entry, pointer) {
                if i.index() > rs
                    && matches!(
                        module.global_variables[g].name.as_deref(),
                        Some("previous" | "residual")
                    )
                {
                    return Err("GR original R reread after reduction".into());
                }
            }
        }
    }
    let words = n::back::spv::write_vec(
        &module,
        &info,
        &n::back::spv::Options {
            lang_version: (1, 3),
            zero_initialize_workgroup_memory: n::back::spv::ZeroInitializeWorkgroupMemoryMode::None,
            ..Default::default()
        },
        Some(&n::back::spv::PipelineOptions {
            shader_stage: n::ShaderStage::Compute,
            entry_point: "main".into(),
        }),
    )?;
    let words = wgpu::hal::vulkan::kuno_native_workgroup_layout(&words)?;
    let mut constants = std::collections::BTreeMap::new();
    let mut barriers = Vec::new();
    let mut shuffle = 0;
    let mut at = 5;
    while at < words.len() {
        let len = (words[at] >> 16) as usize;
        let op = words[at] & 65535;
        if len == 0 || at + len > words.len() {
            return Err("GR register opcode extent".into());
        }
        if op == 43 && len == 4 {
            constants.insert(words[at + 2], words[at + 3]);
        }
        if op == 224 {
            barriers.push(words[at + 3]);
        }
        if op == 225 {
            return Err("GR register unexpected memory barrier".into());
        }
        if op == 346 {
            shuffle += 1;
        }
        at += len;
    }
    if barriers.len() != 1
        || constants
            .get(&barriers[0])
            .is_none_or(|s| s & 0x100 == 0 || s & 0x40 != 0)
        || shuffle < 12
    {
        return Err(format!(
            "GR register actual local barrier/shuffle contract {barriers:?} shuffle={shuffle}"
        )
        .into());
    }
    let mut c = std::process::Command::new("spirv-val")
        .args(["--target-env", "vulkan1.3", "-"])
        .stdin(std::process::Stdio::piped())
        .spawn()?;
    c.stdin.take().ok_or("GR validator stdin")?.write_all(
        &words
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    if !c.wait()?.success() {
        return Err("GR register SPIRV validation".into());
    }
    println!(
        "strata_gr_register_norm_shader_accepted inplace={in_place} scalar_registers=10 local_barriers=1 storage_barriers=0"
    );
    Ok(())
}
pub(crate) fn validate_sources() -> std::result::Result<(), Box<dyn std::error::Error>> {
    for b in [false, true] {
        validate_one(b)?;
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    #[test]
    fn complete_scalar_registers_and_neighbor_pairs_preserve_original_256_stride_map() {
        let mut reads = vec![0u8; 2560];
        let mut pairs = vec![0u8; 1280];
        for sg in 0..8 {
            for lane in 0..32 {
                let tid = sg * 32 + lane;
                for k in 0..10 {
                    let col = tid + k * 256;
                    reads[col] += 1;
                    if lane % 2 == 0 {
                        let next = sg * 32 + (lane ^ 1) + k * 256;
                        assert_eq!(next, col + 1);
                        pairs[col / 2] += 1;
                        let bits = [0u32, 0x80000000, 0x7fc00001, 0x00000001][k % 4];
                        assert_eq!(f32::from_bits(bits).to_bits(), bits);
                    }
                }
            }
        }
        assert!(reads.iter().all(|&n| n == 1) && pairs.iter().all(|&n| n == 1));
    }
    #[test]
    fn actual_modules_keep_original_abi_ten_f32_registers_and_real_local_barrier() {
        super::validate_sources().unwrap();
    }
}
#[path = "strata_prompt_gr_register_probe.rs"]
mod probe;
pub(crate) use probe::cases as public_numeric;
