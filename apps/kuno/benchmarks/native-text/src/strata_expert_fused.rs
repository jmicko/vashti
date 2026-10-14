//! Native Intel separate gate/up, activation and DOWN over the five-tier map.
//! Non-core historical mode retains one combined GU and one DOWN launch.
//! All original packed public pages remain pinned; no model repack or private
//! allocation occurs. The parent authenticates the source generation and every
//! physical root before lending these handles. MIT decoder notices unchanged.
use crate::quant_matvec::grouped_moe as g;
use cubecl::ir::{ElemType, StorageType, UIntKind};
use cubecl::prelude::{ComputeClient, CubeCount, CubeDim};
use cubecl::wgpu::{AutoCompiler, WgpuRuntime};
use cubecl_runtime::compiler::{CompilationError, CubeTask};
use cubecl_runtime::id::KernelId;
use cubecl_runtime::kernel::{CompiledKernel, KernelMetadata};
use cubecl_runtime::server::{ExecutionMode, Handle, KernelArguments, MetadataBindingInfo};
use kuno_text::{Error, Result};
#[path = "strata_global_expert_kernels.rs"]
#[allow(dead_code)] // Global coordinator consumes this complete operator family.
pub(crate) mod global;
#[path = "strata_expert_roles.rs"]
mod roles;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Kernel {
    down: bool,
    dtype: u32,
    row_bytes: usize,
    four: bool,
    member_width: u8,
    native_role: u8,
}
/// All public widths launch; immutable GPU MAP admits exactly one per group.
/// Strata fb58e0 row_entries / row_dot_multi use cardinalities4/2/1 rather than
/// reserving four accumulators for every group. Per-member dot/FMA and reduction
/// orders remain unchanged here, including the existing combined GU activation.
fn specialize_members(mut source: String, down: bool, capacity: usize, width: usize) -> String {
    assert!((1..=capacity).contains(&width));
    let header = "let length=map[p.a+84u+group.y];";
    assert_eq!(
        source.matches(header).count(),
        1,
        "owning expert group header"
    );
    source = source.replace(header, &format!("{header}if length!={width}u {{return;}}"));
    for index in (1..capacity).rev() {
        let needle = format!("if length>{index}u {{");
        while let Some(first) = source.find(&needle) {
            let end = crate::strata_port::function_end(&source, first);
            let replacement = if index < width {
                source[first + needle.len() - 1..end].to_owned()
            } else {
                String::new()
            };
            source.replace_range(first..end, &replacement);
        }
    }
    let float_ty = if width == 1 {
        "f32".to_owned()
    } else {
        format!("vec{width}<f32>")
    };
    let int_ty = if width == 1 {
        "i32".to_owned()
    } else {
        format!("vec{width}<i32>")
    };
    source = source
        .replace(&format!("vec{capacity}<f32>"), &float_ty)
        .replace(&format!("vec{capacity}<i32>"), &int_ty);
    if width == 1 {
        for name in if down {
            vec!["values", "first", "second"]
        } else {
            vec!["gates", "ups", "gf", "gs", "uf", "us"]
        } {
            source = source.replace(&format!("{name}.x"), name);
        }
    }
    source.push_str(&format!("\n// STRATA_CARDINALITY_MEMBERS={width}\n"));
    source
}
pub(crate) fn replace_reads(mut source: String, name: &str) -> String {
    let needle = format!("{name}[");
    while let Some(first) = source.find(&needle) {
        let end = first
            + source[first..]
                .find(']')
                .expect("packed scalar weight read");
        let expr = &source[first + needle.len()..end];
        let read = format!("strata_{name}({expr})");
        source.replace_range(first..end + 1, &read);
    }
    source
}
impl Kernel {
    fn source(self) -> String {
        let count = if self.four { 4 } else { 3 };
        let mut source = if self.down {
            crate::quant_matvec::frozen_down_integer::strata_fused_source(
                self.dtype,
                self.row_bytes,
                count,
            )
        } else {
            crate::quant_matvec::frozen_gu_integer::strata_fused_source(
                self.dtype,
                self.row_bytes,
                count,
            )
        };
        let first = source
            .find("@group(0) @binding(0)")
            .expect("packed bindings");
        let end = first
            + source[first..]
                .find("var<uniform> p:Params;")
                .expect("packed bindings end")
            + "var<uniform> p:Params;".len();
        let names: &[&str] = if self.down {
            &["down"]
        } else {
            &["gate", "up"]
        };
        let mut bindings = String::new();
        let mut helpers = String::new();
        let mut binding = 0;
        for name in names {
            for shard in 0..5 {
                let element = if crate::strata_intel_core::enabled() {
                    "vec4<u32>"
                } else {
                    "u32"
                };
                bindings.push_str(&format!(
                    "@group(0) @binding({binding}) var<storage,read> {name}{shard}:array<{element}>;\n"
                ));
                binding += 1;
            }
            helpers.push_str(&format!(
                "fn strata_{name}(index:u32)->u32 {{switch strata_shard {{\n"
            ));
            let index = if crate::strata_intel_core::enabled() {
                "[index/4u][index%4u]"
            } else {
                "[index]"
            };
            for shard in 0..4 {
                helpers.push_str(&format!(
                    "case {shard}u: {{return {name}{shard}{index};}}\n"
                ));
            }
            helpers.push_str(&format!("default: {{return {name}4{index};}} }} }}\n"));
        }
        for (name, access, ty) in [
            ("map", "read", "u32"),
            ("output", "read_write", "f32"),
            ("status", "read_write", "u32"),
            ("planes", "read", "u32"),
        ] {
            bindings.push_str(&format!(
                "@group(0) @binding({binding}) var<storage,{access}> {name}:array<{ty}>;\n"
            ));
            binding += 1;
        }
        let table_binding = binding;
        if crate::strata_intel_core::enabled() && !self.down {
            binding += 1;
        }
        if crate::strata_intel_core::enabled() {
            for name in names {
                bindings.push_str(&format!("@group(0) @binding({binding}) var<storage,read> strata_staged_{name}:array<u32>;\n"));
                binding += 1;
            }
        }
        let address_binding = binding;
        if crate::strata_intel_core::enabled() {
            bindings.push_str(&format!("// STRATA_PHYSICAL_EXPERTS\n@group(0) @binding({address_binding}) var<storage,read> strata_addresses:array<vec4<u32>>;\n"));
            for name in names {
                bindings.push_str(&format!("var<private> strata_{name}_address:vec2<u32>;\n"));
            }
            binding += 1;
        }
        bindings.push_str(&format!("@group(0) @binding({binding}) var<uniform> p:Params;\nvar<private> strata_shard:u32;var<private> strata_first:u32;\n"));
        source.replace_range(first..end, &bindings);
        let words = if self.down { "2560u" } else { "640u" };
        let old = if self.down {
            "    if arrayLength(&down)!=p.c*2560u*(ROW_BYTES/4u) {return false;}".to_owned()
        } else {
            "    if arrayLength(&gate)!=p.c*640u*(ROW_BYTES/4u) || arrayLength(&up)!=arrayLength(&gate) {return false;}".to_owned()
        };
        assert!(source.contains(&old), "fused source shape guard drift");
        let mut check = String::new();
        for (shard, n) in crate::daslab_profile::SOURCE_PARTS.into_iter().enumerate() {
            for name in names {
                let factor = if crate::strata_intel_core::enabled() {
                    "*4u"
                } else {
                    ""
                };
                check.push_str(&format!("    if arrayLength(&{name}{shard}){factor}!={n}u*{words}*(ROW_BYTES/4u) {{return false;}}\n"));
            }
        }
        source = source.replace(&old, &check);
        let old = "let slot=map[p.a+4u+group];if slot<p.b || slot-p.b>=p.c {return false;}";
        assert!(source.contains(old), "fused logical source guard drift");
        let mut select = String::from(
            "let slot=map[p.a+4u+group];if slot>=512u || p.b!=0u || p.c!=512u {return false;}\n",
        );
        select.push_str(&format!(
            "strata_shard=4u;strata_first={}u;\n",
            crate::daslab_profile::SOURCE_FIRST[4]
        ));
        for shard in (0..4).rev() {
            let first = crate::daslab_profile::SOURCE_FIRST[shard];
            let end = first + crate::daslab_profile::SOURCE_PARTS[shard];
            select.push_str(&format!(
                "if slot<{end}u {{strata_shard={shard}u;strata_first={first}u;}}\n"
            ));
        }
        if crate::strata_intel_core::enabled() {
            select.push_str("if arrayLength(&strata_addresses)!=2048u {return false;}\n");
            for name in names {
                let role = match *name {
                    "gate" => 0,
                    "up" => 1,
                    "down" => 2,
                    _ => unreachable!(),
                };
                select.push_str(&format!(
                    "strata_{name}_address=strata_addresses[{role}u*5u+strata_shard].xy;\nif arrayLength(&strata_staged_{name})!=16777216u {{return false;}}\nif strata_shard>=2u {{strata_{name}_address=strata_addresses[15u+{role}u].xy;}}\n"
                ));
            }
        }
        if crate::strata_intel_core::enabled() {
            select.push_str("if strata_shard>=2u {strata_first=slot-group;}\n");
        }
        source = source.replace(old, &select).replace(
            "let slot=map[p.a+4u+group.y]-p.b;",
            "let slot=map[p.a+4u+group.y]-strata_first;",
        );
        for name in names {
            source = replace_reads(source, name);
        }
        if crate::strata_intel_core::enabled() {
            helpers.clear();
            for name in names {
                helpers.push_str(&format!(
                    "fn strata_{name}(index:u32)->u32 {{return {name}0[index/4u][index%4u];}}\n"
                ));
            }
        }
        source.push_str(&helpers);
        if crate::strata_intel_core::enabled() && !self.down {
            source = crate::strata_intel_core::shared_table(source, table_binding, self.dtype);
        }
        if self.native_role != 0 {
            return roles::source(
                source,
                self.down,
                self.dtype,
                self.member_width as usize,
                count,
                self.native_role,
            );
        }
        if self.member_width != 0 {
            source = specialize_members(source, self.down, count, self.member_width as usize);
        }
        source
    }
}
impl KernelMetadata for Kernel {
    fn name(&self) -> &'static str {
        if self.native_role == 1 {
            "StrataNativeGateChunks"
        } else if self.native_role == 2 {
            "StrataNativeUpChunks"
        } else if self.native_role == 3 {
            "StrataNativeDownChunks"
        } else if self.down {
            "StrataFusedFiveTierDown"
        } else {
            "StrataFusedFiveTierGu"
        }
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
            .info((*self, crate::strata_port::expert_lanes()))
            .cube_dim(CubeDim::new_1d(256))
    }
    fn address_type(&self) -> StorageType {
        StorageType::Scalar(ElemType::UInt(UIntKind::U32))
    }
}
impl CubeTask<AutoCompiler> for Kernel {
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
            source: self.source(),
            repr: None,
            cube_dim: CubeDim::new_1d(256),
            debug_info: None,
        })
    }
}

/// Parent holds the graph/source/arena ownership proof through final completion.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch(
    client: &ComputeClient<WgpuRuntime>,
    geometry: g::Geometry,
    spans: &g::Spans,
    pages: &[Handle],
    owners: &[Handle; 7],
    scratch: &Handle,
    count: usize,
    groups: usize,
    down: bool,
) -> Result<()> {
    if crate::strata_intel_core::enabled() {
        // One complete native expert sequence: fetch, separate roles, activation,
        // then route-indexed preparation/DOWN. Never speed-gate its ingredients.
        if !down {
            crate::strata_stage::launch(
                client,
                geometry,
                pages,
                &owners[0],
                spans.map_first(),
                count,
            )?;
        }
        for native_role in if down { 3..=3 } else { 1..=2 } {
            for member_width in if count == 4 {
                &[4, 2, 1][..]
            } else {
                &[2, 1][..]
            } {
                launch_width(
                    client,
                    geometry,
                    spans,
                    pages,
                    owners,
                    scratch,
                    count,
                    groups,
                    down,
                    *member_width,
                    native_role,
                )?;
            }
        }
        if !down {
            activate(client, spans, owners, count)?;
        }
        Ok(())
    } else if crate::strata_port::specialized_members() {
        for member_width in 1..=if count == 4 { 4 } else { 3 } {
            launch_width(
                client,
                geometry,
                spans,
                pages,
                owners,
                scratch,
                count,
                groups,
                down,
                member_width,
                0,
            )?;
        }
        Ok(())
    } else {
        launch_width(
            client, geometry, spans, pages, owners, scratch, count, groups, down, 0, 0,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_width(
    client: &ComputeClient<WgpuRuntime>,
    geometry: g::Geometry,
    spans: &g::Spans,
    pages: &[Handle],
    owners: &[Handle; 7],
    scratch: &Handle,
    count: usize,
    groups: usize,
    down: bool,
    member_width: u8,
    native_role: u8,
) -> Result<()> {
    if !crate::strata_port::enabled()
        || pages.len() != 15
        || !(1..=4).contains(&count)
        || groups != if count == 4 { 40 } else { 30 }
    {
        return Err(Error::InvalidFormat("Strata fused complete source scope"));
    }
    let projections = geometry.projections();
    let mut args = KernelArguments::new();
    for role in if down { 2..3 } else { 0..2 } {
        for (shard, &slots) in crate::daslab_profile::SOURCE_PARTS.iter().enumerate() {
            let page = &pages[role * 5 + shard];
            if page.size_in_used()
                != projections[role]
                    .segment_words(slots)
                    .map_err(Error::InvalidFormat)? as u64
                    * 4
            {
                return Err(Error::InvalidBufferLength);
            }
            args = args.with_buffer(page.clone().binding());
        }
    }
    for handle in [
        &owners[0],
        &owners[if down || native_role != 0 { 3 } else { 2 }],
        &owners[6],
        scratch,
    ] {
        args = args.with_buffer(handle.clone().binding());
    }
    if pages.iter().chain(owners).chain([scratch]).any(|h| {
        h.stream != owners[0].stream
            || h.offset_start.unwrap_or(0) != 0
            || h.offset_end.unwrap_or(0) != 0
            || h.size() != h.size_in_used()
    }) {
        return Err(Error::InvalidBufferLength);
    }
    if crate::strata_intel_core::enabled() && !down {
        args = crate::strata_intel_core::bind(args, owners[0].stream)?;
    }
    if crate::strata_intel_core::enabled() {
        let stages = crate::strata_stage::handles()?;
        for role in if down { 2..3 } else { 0..2 } {
            args = args.with_buffer(stages[role].clone().binding());
        }
        // Copies all original selected HOST words exactly once per layer.
        if !down && native_role == 0 && member_width <= 1 {
            crate::strata_stage::launch(
                client,
                geometry,
                pages,
                &owners[0],
                spans.map_first(),
                count,
            )?;
        }
        args = crate::strata_intel_core::bind_addresses(client, pages, args)?;
    }
    let values = [
        spans.map_first(),
        0,
        512,
        if down || native_role != 0 {
            spans.partial_first()
        } else {
            spans.act_first()
        },
        if down || native_role != 0 {
            spans.partial_stride()
        } else {
            spans.act_stride()
        },
        spans.status_first(),
        if down { groups } else { 0 },
        count,
    ];
    args = args.with_info(MetadataBindingInfo::custom(
        values
            .chunks_exact(2)
            .map(|v| v[0] as u64 | ((v[1] as u64) << 32))
            .collect(),
    ));
    let projection = projections[if down { 2 } else { 0 }];
    // SAFETY: admitted parent authenticates each distinct complete role root;
    // shape guards above and in WGSL cover all five original source partitions.
    // One uniform source selection is made before every weight or shared access.
    unsafe {
        client.launch_unchecked(
            Box::new(Kernel {
                down,
                dtype: projection.dtype(),
                row_bytes: projection.row_bytes(),
                four: count == 4,
                member_width,
                native_role,
            }),
            CubeCount::Static(
                if down {
                    2560 / (256 / crate::strata_port::expert_lanes())
                } else {
                    640 / (256 / crate::strata_port::expert_lanes())
                } as u32,
                groups as u32,
                1,
            ),
            args,
        );
    }
    Ok(())
}
pub(crate) fn compiled_sources() -> Vec<CompiledKernel<AutoCompiler>> {
    let mut result = Vec::new();
    for (down, dtypes, columns) in [
        (
            false,
            vec![
                crate::quant_matvec::IQ2_S,
                crate::quant_matvec::IQ3_XXS,
                crate::quant_matvec::IQ3_S,
                crate::quant_matvec::IQ4_XS,
            ],
            2560,
        ),
        (
            true,
            vec![crate::quant_matvec::Q2_0, crate::quant_matvec::IQ4_NL],
            640,
        ),
    ] {
        for dtype in dtypes {
            for four in [false, true] {
                let widths = if crate::strata_intel_core::enabled() {
                    (0..=if four { 4 } else { 3 }).collect::<Vec<_>>()
                } else {
                    vec![0]
                };
                for member_width in widths {
                    let row_bytes = kuno_text::TensorType::from_gguf_id(dtype)
                        .unwrap()
                        .row_bytes(columns)
                        .unwrap() as usize;
                    result.push(
                        Kernel {
                            down,
                            dtype,
                            row_bytes,
                            four,
                            member_width,
                            native_role: 0,
                        }
                        .compile(
                            &mut AutoCompiler::Wgsl(Default::default()),
                            &Default::default(),
                            ExecutionMode::Unchecked,
                            StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
                        )
                        .unwrap(),
                    );
                }
            }
        }
    }
    if crate::strata_intel_core::enabled() {
        for native_role in 1..=3 {
            let down = native_role == 3;
            let dtypes: &[u32] = if down {
                &[crate::quant_matvec::Q2_0, crate::quant_matvec::IQ4_NL]
            } else {
                &[
                    crate::quant_matvec::IQ2_S,
                    crate::quant_matvec::IQ3_XXS,
                    crate::quant_matvec::IQ3_S,
                    crate::quant_matvec::IQ4_XS,
                ]
            };
            for &dtype in dtypes {
                for four in [false, true] {
                    for &member_width in if four { &[4, 2, 1][..] } else { &[2, 1][..] } {
                        result.push(
                            Kernel {
                                down,
                                dtype,
                                row_bytes: kuno_text::TensorType::from_gguf_id(dtype)
                                    .unwrap()
                                    .row_bytes(if down { 640 } else { 2560 })
                                    .unwrap() as usize,
                                four,
                                member_width,
                                native_role,
                            }
                            .compile(
                                &mut AutoCompiler::Wgsl(Default::default()),
                                &Default::default(),
                                ExecutionMode::Unchecked,
                                StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
                            )
                            .unwrap(),
                        );
                    }
                }
            }
        }
        result.push(
            Activate
                .compile(
                    &mut AutoCompiler::Wgsl(Default::default()),
                    &Default::default(),
                    ExecutionMode::Unchecked,
                    StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
                )
                .unwrap(),
        );
    }
    result
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Activate;
impl KernelMetadata for Activate {
    fn name(&self) -> &'static str {
        "StrataNativeGuActivate"
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>().cube_dim(CubeDim::new_1d(64))
    }
    fn address_type(&self) -> StorageType {
        StorageType::Scalar(ElemType::UInt(UIntKind::U32))
    }
}
impl CubeTask<AutoCompiler> for Activate {
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
            source: roles::ACTIVATE.into(),
            repr: None,
            cube_dim: CubeDim::new_1d(64),
            debug_info: None,
        })
    }
}
fn activate(
    client: &ComputeClient<WgpuRuntime>,
    spans: &g::Spans,
    owners: &[Handle; 7],
    count: usize,
) -> Result<()> {
    // The caller's Spans/arena proof covers all MAX80 rows. PART is dead until
    // DOWN: activation clears exactly its borrowed two role planes after use.
    let values = [
        count,
        spans.partial_first(),
        spans.partial_stride(),
        spans.act_first(),
        spans.act_stride(),
        spans.status_first(),
        0,
        0,
    ];
    if values.iter().any(|&v| v > u32::MAX as usize) {
        return Err(Error::InvalidBufferLength);
    }
    let mut args = KernelArguments::new();
    for i in [3, 2, 6] {
        args = args.with_buffer(owners[i].clone().binding());
    }
    args = args.with_info(MetadataBindingInfo::custom(
        values
            .chunks_exact(2)
            .map(|v| v[0] as u64 | ((v[1] as u64) << 32))
            .collect(),
    ));
    // SAFETY: authentic full private roots, nonoverlapping rows, same queue;
    // native role writes precede activation and DOWN cannot run before it.
    unsafe {
        client.launch_unchecked(
            Box::new(Activate),
            CubeCount::Static((count * 10) as u32, 1, 1),
            args,
        );
    }
    Ok(())
}
