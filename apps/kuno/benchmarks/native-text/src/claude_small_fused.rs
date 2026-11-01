//! Fused small-expert projection: decode quantized weights straight into the
//! product instead of materializing a whole f16 image per expert.
//!
//! Small cohort experts (<32 routed tokens) used to dequantize every weight to
//! an 80 MiB f16 working image and then run a cooperative-matrix product over
//! at most 31 tokens. Here a unit of 16 (GU) or 4 (DOWN) lanes owns one output
//! row pair. Its lanes split K into 32-weight blocks, decode each block once
//! with the original record32 decoder (rounded to f16 exactly like the image),
//! and multiply it by every token row. A shuffle-xor reduction joins the lanes.
//! The native graph pins SIMD16 so the accumulators fit in registers.
//! The guards, owners, outputs and route-status rules of the image and matrix
//! phases are kept unchanged.
use super::*;

pub(crate) const LANES: u32 = 128;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(crate) struct Fused {
    pub(crate) dtype: u32,
    pub(crate) down: bool,
}

pub(crate) fn disabled() -> bool {
    std::env::var_os("KUNO_CLAUDE_SMALL_OFF").is_some()
}

impl Fused {
    /// Lanes splitting K per unit: 80 GU blocks over 16, 20 DOWN blocks over 4.
    fn width(self) -> u32 {
        if self.down { 4 } else { 16 }
    }
    pub(crate) fn grid_x(self) -> u32 {
        let pairs = if self.down { 1280 } else { 640 };
        pairs / (LANES / self.width())
    }
    pub(crate) fn source(self) -> String {
        let Self { dtype, down } = self;
        let mut s = super::super::direct8::source(CohortOp::Image { dtype, down }.source(), dtype, down);
        // Main entry: add the builtins the block mapping needs.
        let entry = "fn main(@builtin(global_invocation_id) id:vec3<u32>) {";
        assert_eq!(s.matches(entry).count(), 1);
        s = s.replace(entry, "fn main(@builtin(global_invocation_id) id:vec3<u32>,@builtin(workgroup_id) wg:vec3<u32>,@builtin(local_invocation_index) li:u32,@builtin(subgroup_size) ss:u32,@builtin(subgroup_invocation_id) sl:u32) {");
        assert_eq!(s.matches("@workgroup_size(32)").count(), 1);
        s = s.replace("@workgroup_size(32)", &format!("@workgroup_size({LANES})"));
        // Three more owners precede the INFO uniform.
        let uniform = "var<uniform> cohort:CohortInfo;";
        let at = s.find(uniform).expect("cohort uniform");
        let head = &s[..at];
        let open = head.rfind("@binding(").expect("uniform binding") + "@binding(".len();
        let close = open + head[open..].find(')').unwrap();
        let index: u32 = head[open..close].parse().unwrap();
        let bindings = format!(
            "@group(0) @binding({index}) var<storage,read> fused_in:array<vec4<u32>>;\n\
             @group(0) @binding({}) var<storage,read_write> fused_out:array<f32>;\n\
             @group(0) @binding({}) var<storage,read_write> fused_status:array<u32>;\n\
             @group(0) @binding({}) ",
            index + 1,
            index + 2,
            index + 3
        );
        let line = head[..open].rfind("@group(0)").unwrap();
        s.replace_range(line..close + 2, &bindings);
        // Replace the image store dispatch with the fused product.
        let dispatch = if down {
            "    let i=id.x;if i>=p.rows*p.columns/8u {return;}\n    let row=i/(p.columns/8u);let col=(i%(p.columns/8u))*8u;\n    store_original8(i,gate_direct8(row*p.row_bytes,col/32u,(col%32u)/8u));"
        } else {
            "    let i=id.x;if i>=p.rows*p.columns/8u {return;}\n    let row=i/(p.columns/8u);let col=(i%(p.columns/8u))*8u;\n    if row%2u==0u {store_original8(i,gate_direct8((row/2u)*p.row_bytes,col/32u,(col%32u)/8u));}\n    else {store_original8(i,up_direct8((row/2u)*p.row_bytes,col/32u,(col%32u)/8u));}"
        };
        assert_eq!(s.matches(dispatch).count(), 1, "direct8 dispatch");
        s = s.replace(dispatch, &self.body());
        s.push_str(HELPERS);
        // Written for (and checking) 16-wide subgroups: the native graph pins
        // SIMD16, which leaves the accumulators room in 128 registers.
        s.push_str("// KUNO_SIMD16\n");
        s
    }
    fn body(self) -> String {
        let down = self.down;
        let width = self.width();
        let units = LANES / width;
        let per_row = if down { 80 } else { 320 }; // vec4<u32> (8 halves) per input row
        let mut b = String::new();
        b.push_str(&format!(
            "    if ss<16u || ss%16u!=0u || p.rows!={} || p.columns!={} {{atomicOr(&fault[0],128u);return;}}\n",
            if down { 2560 } else { 1280 },
            if down { 640 } else { 2560 }
        ));
        b.push_str("    let members=map[516u+p.group];\n");
        b.push_str("    if sl!=li%ss {atomicOr(&fault[0],128u);}\n");
        b.push_str(&format!("    let lane=li%{width}u;let pair0=wg.x*{units}u+li/{width}u;\n"));
        b.push_str("    let route_first=map[1028u+p.group];\n");
        if down {
            b.push_str("    let xrow=cohort_prefix;\n");
        } else {
            // Layer-gathered X rows: identical checks to the matrix consumer.
            b.push_str(
                "    var xrow=cohort_prefix;\n    if p.r7==1u {\n        let global=map[43008u+p.group];let total=map[43520u];let extent=(members+31u)/32u*32u;\n        var next=total;if p.group+1u<map[3] {next=map[43008u+p.group+1u];}\n        if total>56832u || global%32u!=0u || global>total || extent>total-global || (p.group==0u && global!=0u) || global+extent!=next {atomicOr(&fault[0],128u);return;}\n        xrow=global;\n    }\n",
            );
        }
        // Padded token rows: 8 (<=8 members), 16 (<=16) or 32 (<32).
        b.push_str("    let tile=select(select(32u,16u,members<=16u),8u,members<=8u);\n");
        b.push_str(&format!(
            "    if (xrow+tile)*{per_row}u>arrayLength(&fused_in) {{atomicOr(&fault[0],128u);return;}}\n"
        ));
        if down {
            b.push_str("    if (route_first+members)*2560u>arrayLength(&fused_out) || arrayLength(&fused_out)!=PROMPT_PART_WORDS || arrayLength(&fused_status)!=PROMPT_STATUS_WORDS {atomicOr(&fault[0],128u);return;}\n");
        } else {
            b.push_str("    if (cohort_prefix+tile)*1280u>arrayLength(&fused_out) || arrayLength(&fused_status)!=PROMPT_STATUS_WORDS {atomicOr(&fault[0],128u);return;}\n");
        }
        b.push_str("    var bad=false;\n");
        // One shader, a uniform per-slot branch per tile size.
        for (i, tile) in [8u32, 16, 32].into_iter().enumerate() {
            b.push_str(if i == 0 { "    if tile==8u {\n" } else if tile == 16 { "    } else if tile==16u {\n" } else { "    } else {\n" });
            b.push_str(&self.product(tile));
        }
        b.push_str("    }\n    if bad {atomicOr(&fault[0],128u);}\n");
        b
    }
    fn product(self, tile: u32) -> String {
        let down = self.down;
        let rows = 2;
        let width = self.width();
        let per_row = if down { 80 } else { 320 };
        let blocks = per_row / 4; // 32-weight quant blocks per row
        let mut b = String::new();
        for q in 0..rows {
            for t in 0..tile {
                b.push_str(&format!("    var a{q}_{t}=0.0;"));
            }
            b.push('\n');
        }
        // Each lane owns whole 32-weight blocks: block metadata is loaded once
        // (the original record32 prepare), then its four 8-weight quarters.
        b.push_str(&format!("    for(var block=lane;block<{blocks}u;block+={width}u) {{\n"));
        for q in 0..rows {
            let (role, row) = if down {
                ("gate", format!("(2u*pair0+{q}u)*p.row_bytes"))
            } else {
                (if q % 2 == 0 { "gate" } else { "up" }, "pair0*p.row_bytes".to_string())
            };
            b.push_str(&format!("        let e{q}={role}_prepare({row},block);\n"));
        }
        // A runtime quarter loop (p.r9 is validated zero) keeps Mesa from
        // unrolling it and hoisting four quarters of live values (spills).
        b.push_str("        for(var quarter=0u;quarter<4u+p.r9;quarter++) {\n");
        for q in 0..rows {
            let role = if down || q % 2 == 0 { "gate" } else { "up" };
            let factor = if down { format!("e{q}.d") } else { format!("{role}_factor(e{q},quarter/2u)") };
            b.push_str(&format!(
                "        let w{q}=fused_round(decoded8({factor},{role}_packed(e{q},quarter*2u),{role}_packed(e{q},quarter*2u+1u)));bad=bad||w{q}.bad;\n"
            ));

        }
        b.push_str("        let chunk=block*4u+quarter;\n");
        for t in 0..tile {
            b.push_str(&format!(
                "        {{let v=fused_in[(xrow+{t}u)*{per_row}u+chunk];let xl=vec4<f32>(unpack2x16float(v.x),unpack2x16float(v.y));let xh=vec4<f32>(unpack2x16float(v.z),unpack2x16float(v.w));"
            ));
            for q in 0..rows {
                b.push_str(&format!("a{q}_{t}+=dot(w{q}.a,xl)+dot(w{q}.b,xh);"));
            }
            b.push_str("}\n");
        }
        b.push_str("        }\n    }\n");
        // Join the lanes of each unit (aligned within the subgroup).
        for q in 0..rows {
            for t in 0..tile {
                b.push_str(&format!("    a{q}_{t}=fused_join{width}(a{q}_{t});"));
            }
            b.push('\n');
        }
        // Lane l publishes tokens l, l+width, ...: one store per row per group.
        for group in 0..tile.div_ceil(width) {
            let base = group * width;
            let span = (tile - base).min(width);
            b.push_str(&format!("    {{let t={base}u+lane;\n"));
            b.push_str(&format!("    if lane<{span}u && t<members {{\n"));
            b.push_str("        let route=map[1540u+route_first+t];\n");
            b.push_str("        if route>=p.count*10u {atomicOr(&fault[0],128u);} else {\n");
            let first = if down { "(route_first+t)*2560u" } else { "(cohort_prefix+t)*1280u" };
            for q in 0..rows {
                b.push_str(&format!("        var o{q}=a{q}_{base};"));
                for t in 1..span {
                    b.push_str(&format!("o{q}=select(o{q},a{q}_{},lane=={t}u);", base + t));
                }
                b.push_str(&format!("\n        fused_out[{first}+2u*pair0+{q}u]=o{q};\n"));
            }
            if down {
                b.push_str("        if pair0==0u {fused_status[PROMPT_ROUTES+route]=2u;}\n");
            }
            b.push_str("        }\n    }}\n");
        }
        b
    }
}

const HELPERS: &str = r#"
struct FusedWeights {a:vec4<f32>,b:vec4<f32>,bad:bool}
// Exactly the image's f16 rounding and finiteness rule; faulted chunks are zero.
fn fused_round(v:Decoded8)->FusedWeights {
    let bits=vec4<u32>(pack2x16float(v.a.xy),pack2x16float(v.a.zw),pack2x16float(v.b.xy),pack2x16float(v.b.zw));
    let lo=vec4<f32>(unpack2x16float(bits.x),unpack2x16float(bits.y));
    let hi=vec4<f32>(unpack2x16float(bits.z),unpack2x16float(bits.w));
    let maximum=vec4<f32>(3.4028234663852886e38);
    let good=all(abs(v.a)<=maximum)&&all(abs(v.b)<=maximum)&&all(abs(lo)<=maximum)&&all(abs(hi)<=maximum);
    return FusedWeights(select(vec4<f32>(0.0),lo,good),select(vec4<f32>(0.0),hi,good),!good);
}
fn fused_join16(x:f32)->f32 {
    var v=x;v+=subgroupShuffleXor(v,8u);v+=subgroupShuffleXor(v,4u);v+=subgroupShuffleXor(v,2u);v+=subgroupShuffleXor(v,1u);
    return v;
}
fn fused_join4(x:f32)->f32 {
    var v=x;v+=subgroupShuffleXor(v,2u);v+=subgroupShuffleXor(v,1u);
    return v;
}
"#;

impl KernelMetadata for Fused {
    fn name(&self) -> &'static str {
        if self.down { "ClaudeSmallFusedDown" } else { "ClaudeSmallFusedGu" }
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info(*self).cube_dim(CubeDim::new_1d(LANES))
    }
    fn address_type(&self) -> StorageType {
        StorageType::Scalar(ElemType::UInt(UIntKind::U32))
    }
    /// The unit's final launch is wrapped by the batch's terminal release.
    fn native_release_after(&self) -> bool {
        false
    }
}
impl CubeTask<AutoCompiler> for Fused {
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
            cube_dim: CubeDim::new_1d(LANES),
            debug_info: None,
        })
    }
}

/// Every fused variant, for warm-up and source validation.
pub(crate) fn all() -> Vec<Fused> {
    let mut out = Vec::new();
    for dtype in [q::IQ2_S, q::IQ3_S, q::IQ3_XXS, q::IQ4_XS] {
        out.push(Fused { dtype, down: false });
    }
    for dtype in [q::IQ4_NL, q::Q2_0] {
        out.push(Fused { dtype, down: true });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fused_sources_validate() {
        for op in all() {
            let s = op.source();
            let module = wgpu::naga::front::wgsl::parse_str(&s)
                .unwrap_or_else(|e| panic!("{op:?}: {}", e.emit_to_string(&s)));
            wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::all(),
            )
            .subgroup_stages(wgpu::naga::valid::ShaderStages::COMPUTE)
            .subgroup_operations(wgpu::naga::valid::SubgroupOperationSet::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{op:?}: {e:?}"));
            assert!(s.contains("cohort_begin(id.z)"));
            assert!(!s.contains("store_original8(i,"));
            if let Ok(dir) = std::env::var("KUNO_DUMP_FUSED") {
                std::fs::write(format!("{dir}/fused-{}-{}.wgsl", op.dtype, op.down), &s).unwrap();
            }
        }
    }
}

/// Fixed-public numerics: the fused kernel against a CPU f64 reference built
/// from real model expert weights (8 slots, mixed host/device residency, every
/// tile size), plus its output layout, route status and fault rules.
pub(crate) fn public_cases(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut crate::private_arena::PrivateArena,
    file: &kuno_text::GgufFile,
) -> std::result::Result<Vec<serde_json::Value>, Box<dyn std::error::Error>> {
    let mut table = Vec::new();
    for dtype in [q::IQ3_XXS, q::IQ3_S, q::IQ2_S] {
        table.extend(q::iq_moe_integer::original_grid_words(dtype));
    }
    table.resize(8192, 0);
    let table = super::super::public_numeric::public(client, table.into_iter().flat_map(u32::to_le_bytes).collect())?;
    let mut results = Vec::new();
    for dtype in [q::IQ3_XXS, q::IQ3_S, q::IQ2_S, q::IQ4_XS, q::IQ4_NL, q::Q2_0] {
        let result = public_one(client, arena, file, dtype, &table);
        arena.reset()?;
        results.push(result?);
    }
    Ok(results)
}

fn public_one(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut crate::private_arena::PrivateArena,
    file: &kuno_text::GgufFile,
    dtype: u32,
    table: &Handle,
) -> std::result::Result<serde_json::Value, Box<dyn std::error::Error>> {
    use crate::resident_util as u;
    const COUNT: usize = 32; // tokens (>= every group size); 320 routes
    const MEMBERS: [usize; 8] = [1, 3, 7, 8, 9, 16, 17, 31];
    let down = matches!(dtype, q::IQ4_NL | q::Q2_0);
    let rows = if down { 2560usize } else { 640 };
    let cols = if down { 640usize } else { 2560 };
    let suffix = if down { "ffn_down_exps.weight" } else { "ffn_gate_exps.weight" };
    let (name, tensor) = file
        .tensors()
        .iter()
        .find(|(name, t)| name.ends_with(suffix) && t.dtype().gguf_id() == dtype)
        .ok_or("public fused original format missing")?;
    let row_bytes = tensor.dtype().row_bytes(cols as u64)? as usize;
    let bytes = rows * row_bytes;
    let gu_bytes = if down { 256 } else { bytes };
    let down_bytes = if down { bytes } else { 256 };
    let stride = 2 * gu_bytes + down_bytes;
    let mut raw = Zeroizing::new(vec![0u8; allocation::RAW_RING_BYTES as usize]);
    let mut handles = Vec::new();
    let mut weights: Vec<[Vec<f32>; 2]> = Vec::new();
    for slot in 0..8 {
        let mut gate = vec![0u8; bytes];
        file.tensor(name).unwrap().read_range((slot * bytes) as u64, &mut gate)?;
        let mut up = gate.clone();
        if !down {
            file.tensor(&name.replace(suffix, "ffn_up_exps.weight"))
                .ok_or("public fused original up missing")?
                .read_range((slot * bytes) as u64, &mut up)?;
        }
        if slot % 2 == 0 {
            let at = slot / 2 * stride;
            if down {
                raw[at + 2 * gu_bytes..at + stride].copy_from_slice(&gate);
            } else {
                raw[at..at + gu_bytes].copy_from_slice(&gate);
                raw[at + gu_bytes..at + 2 * gu_bytes].copy_from_slice(&up);
            }
        }
        // Reference weights rounded to f16 exactly like the image.
        let mut a = vec![0f32; rows * cols];
        let mut b = vec![0f32; rows * cols];
        tensor.dtype().decode(&gate, &mut a)?;
        tensor.dtype().decode(&up, &mut b)?;
        for v in a.iter_mut().chain(b.iter_mut()) {
            *v = half::f16::from_f32(*v).to_f32();
        }
        weights.push([a, b]);
        handles.push(super::super::public_numeric::public(client, gate)?);
        handles.push(super::super::public_numeric::public(client, up)?);
    }
    let physical = client.native_strata_source_addresses(handles.clone()).map_err(u::fault)?;
    let mut addresses = vec![0u32; 8192];
    for slot in 0..8 {
        for role in if down { &[2usize][..] } else { &[0usize, 1][..] } {
            let address = physical[slot * 2 + usize::from(*role == 1)];
            let at = (*role * 512 + slot) * 4;
            let host = slot % 2 == 0 && (down || slot % 4 == 0 || *role == 1);
            addresses[at] = address[0] as u32;
            addresses[at + 1] = (address[0] >> 32) as u32;
            addresses[at + 2] = (bytes / 4) as u32;
            addresses[at + 3] = u32::from(host);
        }
    }
    let addresses = super::super::public_numeric::public(client, addresses.into_iter().flat_map(u32::to_le_bytes).collect())?;
    // Eight groups; group g's members use routes route_first..+members.
    let mut map = Zeroizing::new(vec![0u32; groups::MAP_WORDS]);
    map[..4].copy_from_slice(&[COUNT as u32, 0, COUNT as u32, 8]);
    let mut route_first = 0;
    let mut words = Zeroizing::new([0u32; WORDS]);
    let mut prefix = 0;
    let mut prefixes = [0usize; 8];
    let mut route_firsts = [0usize; 8];
    for (g, &members) in MEMBERS.iter().enumerate() {
        map[4 + g] = g as u32;
        map[516 + g] = members as u32;
        map[1028 + g] = route_first as u32;
        for t in 0..members {
            map[1540 + route_first + t] = (route_first + t) as u32;
        }
        let record = if g % 2 == 0 { g as u32 / 2 } else { 8 };
        words[16 + g * 4..20 + g * 4].copy_from_slice(&[g as u32, members as u32, prefix as u32, record]);
        prefixes[g] = prefix;
        route_firsts[g] = route_first;
        let tile = if members <= 8 { 8 } else { 16 };
        prefix += members.div_ceil(tile) * tile;
        route_first += members;
    }
    words[0] = COUNT as u32;
    words[1] = cols as u32;
    words[2] = if down { rows as u32 } else { 2 * rows as u32 };
    words[3] = row_bytes as u32;
    words[5] = if down { 2 } else { 0 };
    words[6] = (gu_bytes / 4) as u32;
    words[7] = (gu_bytes / 4) as u32;
    words[8] = (stride / 4) as u32;
    words[9] = 8;
    words[10] = 31;
    validate_words(&words)?;
    // Public f16 activations, row-major by cohort prefix row.
    let input_rows = prefix + 32;
    let x: Vec<f32> = (0..input_rows * cols)
        .map(|i| half::f16::from_f32((((i * 2654435761usize) >> 9) % 97) as f32 / 97.0 - 0.5).to_f32())
        .collect();
    let x_words: Vec<u32> = x
        .chunks_exact(2)
        .map(|v| u32::from(half::f16::from_f32(v[0]).to_bits()) | (u32::from(half::f16::from_f32(v[1]).to_bits()) << 16))
        .collect();
    let raw_words = Zeroizing::new(raw.chunks_exact(4).map(|v| u32::from_le_bytes(v.try_into().unwrap())).collect::<Vec<_>>());
    let map_id = arena.upload_u32(&map)?;
    // Public weights ring and the (unread) image owner live outside the arena.
    let raw_handle = super::super::public_numeric::public(client, raw_words.iter().flat_map(|w| w.to_le_bytes()).collect())?;
    let x_id = arena.upload_u32(&x_words)?;
    let image = client.empty(storage::WORKING_BYTES as usize);
    let out_words = if down { groups::PART_BYTES } else { groups::GU_BYTES } as usize / 4;
    let out = arena.allocate_zeroed(out_words)?;
    let status = arena.allocate_zeroed(groups::STATUS_WORDS)?;
    let fault = arena.allocate_zeroed(1)?;
    let mut args = KernelArguments::new().with_buffer(handles[0].clone().binding());
    if !down {
        args = args.with_buffer(handles[1].clone().binding());
    }
    args = args
        .with_buffers(vec![
            u::handle(arena, map_id)?.binding(),
            raw_handle.binding(),
            image.binding(),
            u::handle(arena, fault)?.binding(),
            table.clone().binding(),
            addresses.binding(),
            u::handle(arena, x_id)?.binding(),
            u::handle(arena, out)?.binding(),
            u::handle(arena, status)?.binding(),
        ])
        .with_read_dependencies(handles.iter().cloned().map(Handle::binding).collect())
        .with_info(MetadataBindingInfo::custom(
            words.chunks_exact(2).map(|v| u64::from(v[0]) | (u64::from(v[1]) << 32)).collect(),
        ));
    let op = Fused { dtype, down };
    unsafe {
        client.launch_unchecked(Box::new(op), CubeCount::Static(op.grid_x(), 1, 8), args);
    }
    cubecl::future::block_on(client.sync()).map_err(u::fault)?;
    let read = |arena: &crate::private_arena::PrivateArena, id, n| -> std::result::Result<Zeroizing<Vec<u32>>, Box<dyn std::error::Error>> {
        let data = u::read_handle_prefix_u32(arena, arena.handle(id)?.clone(), n)?;
        super::probe::sanitize(arena)?;
        Ok(data)
    };
    let stride_out = if down { 2560 } else { 1280 };
    let used_rows = if down { route_first } else { prefix };
    let gpu = read(arena, out, used_rows * stride_out)?;
    let status_words = read(arena, status, groups::STATUS_WORDS)?;
    let fault_word = read(arena, fault, 1)?[0];
    // CPU reference (f64 accumulation of f16 weights x f16 activations).
    let mut worst = 0f64;
    let mut checked = 0usize;
    for (g, &members) in MEMBERS.iter().enumerate() {
        let [a, b] = &weights[g];
        for t in 0..members {
            let xr = &x[(prefixes[g] + t) * cols..(prefixes[g] + t + 1) * cols];
            for out_row in 0..stride_out {
                let (w, r) = if down { (a, out_row) } else { (if out_row % 2 == 0 { a } else { b }, out_row / 2) };
                let wr = &w[r * cols..(r + 1) * cols];
                let (mut sum, mut scale) = (0f64, 0f64);
                for k in 0..cols {
                    let p = f64::from(wr[k]) * f64::from(xr[k]);
                    sum += p;
                    scale += p.abs();
                }
                let at = if down { (route_firsts[g] + t) * 2560 } else { (prefixes[g] + t) * 1280 } + out_row;
                let actual = f64::from(f32::from_bits(gpu[at]));
                let error = (actual - sum).abs() / (scale + 1e-6);
                if !(error <= worst) {
                    worst = if error.is_nan() { f64::INFINITY } else { error };
                }
                checked += 1;
            }
            if down && status_words[groups::ROUTES + route_firsts[g] + t] != 2 {
                return Err("public fused down route status".into());
            }
        }
    }
    if fault_word != 0 {
        return Err(format!("public fused fault {fault_word}").into());
    }
    // f32 accumulation over <=2560 terms: relative to sum|w x|, well under 1e-5.
    if worst > 1e-5 {
        return Err(format!("public fused numerics dtype={dtype} worst_relative={worst:e}").into());
    }
    Ok(serde_json::json!({"family":"claude-small-fused","dtype":dtype,"down":down,"groups":8,"members":MEMBERS,
        "outputs_checked":checked,"worst_relative_error":worst,"mixed_host_gpu_residency":true}))
}
