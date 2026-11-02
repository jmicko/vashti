//! Fused small-expert projection with Strata's int8 arithmetic.
//!
//! Activations are int8 per 32 values with an f32 scale (`ClaudeQuantX` per
//! layer for X, `ClaudeActivateQ` for the hidden H). A weight block of 32 is
//! d_w * q with q the format's signed int8 codes (the original record32
//! decoder's packed words), so sum w x = d_x * (d_w0 dot0 + d_w1 dot1) with
//! exact int32 dots over 16 values each (dot4I8Packed).
//!
//! Small cohort experts (<32 routed tokens): a unit of 16 (GU) or 4 (DOWN)
//! lanes owns one output row pair; lanes split K into 32-weight blocks, decode
//! each block once and multiply it by every token; a shuffle-xor reduction
//! joins the lanes. The native graph pins SIMD16 so the accumulators fit.
//! Outputs, route status and fault rules match the original matrix phases.
use super::*;

pub(crate) const LANES: u32 = 128;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(crate) struct Fused {
    pub(crate) dtype: u32,
    pub(crate) down: bool,
}

/// Fixed-public quality check selects the original f16 path per request.
pub(crate) static FORCE_OFF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub(crate) fn disabled() -> bool {
    FORCE_OFF.load(std::sync::atomic::Ordering::Relaxed)
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
        let stride4 = if down { 45 } else { 180 }; // vec4<u32> per int8 activation row
        let mut b = String::new();
        b.push_str(&format!(
            "    if ss<16u || ss%16u!=0u || p.rows!={} || p.columns!={} || arrayLength(&fused_in)!={}u {{atomicOr(&fault[0],128u);return;}}\n",
            if down { 2560 } else { 1280 },
            if down { 640 } else { 2560 },
            groups::CAPACITY * stride4
        ));
        b.push_str("    let members=map[516u+p.group];\n");
        b.push_str("    if sl!=li%ss {atomicOr(&fault[0],128u);}\n");
        b.push_str(&format!("    let lane=li%{width}u;let pair0=wg.x*{units}u+li/{width}u;\n"));
        b.push_str("    let route_first=map[1028u+p.group];\n");
        // Padded token rows: 8 (<=8 members), 16 (<=16) or 32 (<32).
        b.push_str("    let tile=select(select(32u,16u,members<=16u),8u,members<=8u);\n");
        if down {
            b.push_str("    if cohort_prefix+tile>PROMPT_MAX_ROWS || (route_first+members)*2560u>arrayLength(&fused_out) || arrayLength(&fused_out)!=PROMPT_PART_WORDS || arrayLength(&fused_status)!=PROMPT_STATUS_WORDS {atomicOr(&fault[0],128u);return;}\n");
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
        let blocks = if down { 20 } else { 80 }; // 32-weight quant blocks per row
        let stride4 = if down { 45 } else { 180 };
        let scales4 = if down { 40 } else { 160 };
        let mut b = String::new();
        for q in 0..rows {
            for t in 0..tile {
                b.push_str(&format!("    var a{q}_{t}=0.0;"));
            }
            b.push('\n');
        }
        // Each lane owns whole 32-weight blocks: the original record32 prepare
        // yields the block's signed int8 codes (packed words) and scales.
        b.push_str(&format!("    for(var block=lane;block<{blocks}u;block+={width}u) {{\n"));
        for q in 0..rows {
            let (role, row) = if down {
                ("gate", format!("(2u*pair0+{q}u)*p.row_bytes"))
            } else {
                (if q % 2 == 0 { "gate" } else { "up" }, "pair0*p.row_bytes".to_string())
            };
            b.push_str(&format!("        let e{q}={role}_prepare({row},block);\n"));
            if down {
                b.push_str(&format!("        let f{q}_0=e{q}.d;let f{q}_1=e{q}.d;\n"));
            } else {
                b.push_str(&format!("        let f{q}_0={role}_factor(e{q},0u);let f{q}_1={role}_factor(e{q},1u);\n"));
            }
            b.push_str(&format!("        bad=bad||!finite(f{q}_0)||!finite(f{q}_1);\n"));
            for j in 0..8 {
                b.push_str(&format!("        let w{q}_{j}={role}_packed(e{q},{j}u);"));
            }
            b.push('\n');
        }
        for t in 0..tile {
            if down {
                b.push_str(&format!("        {{let r=cohort_prefix+{t}u;"));
            } else {
                b.push_str(&format!(
                    "        {{let r=min(map[1540u+route_first+min({t}u,members-1u)]/10u,p.count-1u);"
                ));
            }
            b.push_str(&format!(
                "let xa=fused_in[r*{stride4}u+block*2u];let xb=fused_in[r*{stride4}u+block*2u+1u];let dx=bitcast<f32>(fused_in[r*{stride4}u+{scales4}u+block/4u][block%4u]);\n        "
            ));
            for q in 0..rows {
                b.push_str(&format!(
                    "a{q}_{t}=fma(dx,f{q}_0*f32(dot4I8Packed(w{q}_0,xa.x)+dot4I8Packed(w{q}_1,xa.y)+dot4I8Packed(w{q}_2,xa.z)+dot4I8Packed(w{q}_3,xa.w))+f{q}_1*f32(dot4I8Packed(w{q}_4,xb.x)+dot4I8Packed(w{q}_5,xb.y)+dot4I8Packed(w{q}_6,xb.z)+dot4I8Packed(w{q}_7,xb.w)),a{q}_{t});"
                ));
            }
            b.push_str("}\n");
        }
        b.push_str("    }\n");
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

/// Per-layer int8 activations (Strata's quant_act_nat): each token's 2560
/// inputs in blocks of 32, code = round_half_even(x * 127 / amax), scale
/// amax / 127, into Xq rows of 640 code words then 80 scales.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(crate) struct QuantX;
const QUANT_X: &str = r#"
struct Info {count:u32,half:u32,r0:u32,r1:u32,r2:u32,r3:u32,r4:u32,r5:u32}
@group(0) @binding(0) var<storage,read> input:array<u32>;
@group(0) @binding(1) var<storage,read_write> xq:array<u32>;
@group(0) @binding(2) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(3) var<uniform> p:Info;
fn finite(x:f32)->bool {return abs(x)<=3.4028234663852886e38;}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    if p.count<1u || p.count>PROMPT_MAX_ROWS || p.half>1u || p.r0!=0u || p.r1!=0u || p.r2!=0u || p.r3!=0u || p.r4!=0u || p.r5!=0u
        || arrayLength(&xq)!=PROMPT_MAX_ROWS*900u || arrayLength(&input)<p.count*select(2560u,1280u,p.half==1u) {atomicOr(&fault[0],128u);return;}
    let i=id.x;if i>=p.count*80u {return;}
    let token=i/80u;let block=i%80u;
    var v:array<f32,32>;
    var amax=0.0;var good=true;
    for(var j=0u;j<32u;j++) {
        var x=0.0;
        if p.half==1u {let pair=unpack2x16float(input[token*1280u+block*16u+j/2u]);x=select(pair.x,pair.y,(j%2u)==1u);}
        else {x=bitcast<f32>(input[token*2560u+block*32u+j]);}
        good=good&&finite(x);v[j]=x;amax=max(amax,abs(x));
    }
    let inv=select(0.0,127.0/amax,amax>0.0);
    for(var w=0u;w<8u;w++) {
        var word=0u;
        for(var b=0u;b<4u;b++) {
            let c=clamp(i32(round(v[w*4u+b]*inv)),-127,127);
            word|=(u32(c)&255u)<<(b*8u);
        }
        xq[token*720u+block*8u+w]=select(0u,word,good);
    }
    xq[token*720u+640u+block]=select(0u,bitcast<u32>(amax/127.0),good);
    if !good {atomicOr(&fault[0],128u);}
}
"#;
impl KernelMetadata for QuantX {
    fn name(&self) -> &'static str { "ClaudeQuantX" }
    fn id(&self) -> KernelId { KernelId::new::<Self>().cube_dim(CubeDim::new_1d(256)) }
    fn address_type(&self) -> StorageType { StorageType::Scalar(ElemType::UInt(UIntKind::U32)) }
}
impl CubeTask<AutoCompiler> for QuantX {
    fn compile(&self, _: &mut AutoCompiler, _: &cubecl::WgpuCompilationOptions, _: ExecutionMode, _: StorageType)
        -> std::result::Result<CompiledKernel<AutoCompiler>, CompilationError> {
        Ok(CompiledKernel { entrypoint_name: "main".into(), debug_name: None, source: (groups::shader_prefix() + QUANT_X).into(), repr: None,
            cube_dim: CubeDim::new_1d(256), debug_info: None })
    }
}
/// Quantize the layer's expert input (f32, or the f16 mixed image) into Xq.
pub(crate) fn quant_x(client: &ComputeClient<WgpuRuntime>, input: Handle, q: Handle, fault: Handle, count: usize, half: bool) -> Result<()> {
    if !(1..=groups::CAPACITY).contains(&count) || q.size_in_used() != groups::LAYER_Q_BYTES {
        return Err(Error::InvalidBufferLength);
    }
    let words = [count as u32, u32::from(half), 0, 0, 0, 0, 0, 0];
    let args = KernelArguments::new()
        .with_buffers(vec![input.binding(), q.binding(), fault.binding()])
        .with_info(MetadataBindingInfo::custom(words.chunks_exact(2).map(|v| u64::from(v[0]) | (u64::from(v[1]) << 32)).collect()));
    unsafe {
        client.launch_unchecked(Box::new(QuantX), CubeCount::Static((count * 80).div_ceil(256) as u32, 1, 1), args);
    }
    crate::strata_prefill_profile::count("expert_int8_quant_x_launches", 1);
    Ok(())
}

/// SwiGLU of the fused GU output and H to int8 per 32 features (Strata's GU
/// epilogue): one invocation per (member, 32-feature block). Original
/// non-finite rules: a bad gate/up value faults; status[route] follows block 0.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(crate) struct ActivateQ;
impl ActivateQ {
    fn source() -> String {
        let all = include_str!("strata_prompt_experts.wgsl");
        let common = all.split_once("// COMMON_BEGIN\n").unwrap().1.split_once("// COMMON_END").unwrap().0;
        let mut s = groups::shader_prefix() + common;
        s.push_str(
            "@group(0) @binding(0) var<storage,read> map:array<u32>;\n\
             @group(0) @binding(1) var<storage,read> gu:array<f32>;\n\
             @group(0) @binding(2) var<storage,read_write> hq:array<u32>;\n\
             @group(0) @binding(3) var<storage,read_write> status:array<u32>;\n\
             @group(0) @binding(4) var<storage,read_write> fault:array<atomic<u32>>;\n\
             @group(0) @binding(5) var<uniform> cohort:CohortInfo;\n\
             var<private> p:Params;\nvar<private> cohort_prefix:u32;\nvar<private> cohort_slot:u32;\n",
        );
        s.push_str(include_str!("strata_prompt_expert_cohort_header.wgsl"));
        s.push_str(r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    if !cohort_begin(id.z) {return;}
    if !valid() {return;}
    if arrayLength(&hq)!=PROMPT_MAX_ROWS*180u || arrayLength(&gu)!=PROMPT_MAX_ROWS*1280u || arrayLength(&status)!=PROMPT_STATUS_WORDS {atomicOr(&fault[0],128u);return;}
    let members=map[516u+p.group];let i=id.x;if i>=members*20u {return;}
    let member=i/20u;let block=i%20u;
    if cohort_prefix+member>=PROMPT_MAX_ROWS {atomicOr(&fault[0],128u);return;}
    let route=map[1540u+map[1028u+p.group]+member];if route>=p.count*10u {atomicOr(&fault[0],128u);return;}
    let base=(cohort_prefix+member)*1280u+block*64u;
    var y:array<f32,32>;var amax=0.0;var good=true;
    for(var j=0u;j<32u;j++) {
        var g=gu[base+2u*j];var u=gu[base+2u*j+1u];
        if !finite(g) {g=0.0;good=false;}
        if !finite(u) {u=0.0;good=false;}
        let h=g/(1.0+exp(-g))*u;
        let ok=finite(h);good=good&&ok;
        y[j]=select(0.0,h,ok);amax=max(amax,abs(y[j]));
    }
    let inv=select(0.0,127.0/amax,amax>0.0);
    let row=(cohort_prefix+member)*180u;
    for(var w=0u;w<8u;w++) {
        var word=0u;
        for(var b=0u;b<4u;b++) {
            let c=clamp(i32(round(y[w*4u+b]*inv)),-127,127);
            word|=(u32(c)&255u)<<(b*8u);
        }
        hq[row+block*8u+w]=word;
    }
    hq[row+160u+block]=bitcast<u32>(amax/127.0);
    if block==0u {status[route]=select(0u,1u,good);}
    if !good {atomicOr(&fault[0],128u);}
}
"#);
        s
    }
}
impl KernelMetadata for ActivateQ {
    fn name(&self) -> &'static str { "ClaudeActivateQ" }
    fn id(&self) -> KernelId { KernelId::new::<Self>().cube_dim(CubeDim::new_1d(256)) }
    fn address_type(&self) -> StorageType { StorageType::Scalar(ElemType::UInt(UIntKind::U32)) }
}
impl CubeTask<AutoCompiler> for ActivateQ {
    fn compile(&self, _: &mut AutoCompiler, _: &cubecl::WgpuCompilationOptions, _: ExecutionMode, _: StorageType)
        -> std::result::Result<CompiledKernel<AutoCompiler>, CompilationError> {
        Ok(CompiledKernel { entrypoint_name: "main".into(), debug_name: None, source: Self::source(), repr: None,
            cube_dim: CubeDim::new_1d(256), debug_info: None })
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
    fn int8_activation_sources_validate() {
        for s in [groups::shader_prefix() + QUANT_X, ActivateQ::source()] {
            let module = wgpu::naga::front::wgsl::parse_str(&s).unwrap_or_else(|e| panic!("{}", e.emit_to_string(&s)));
            wgpu::naga::valid::Validator::new(wgpu::naga::valid::ValidationFlags::all(), wgpu::naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap();
        }
    }
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
            assert!(s.contains("dot4I8Packed"));
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
        // Reference weights: the format's exact d_w * q (no f16 image rounding).
        let mut a = vec![0f32; rows * cols];
        let mut b = vec![0f32; rows * cols];
        tensor.dtype().decode(&gate, &mut a)?;
        tensor.dtype().decode(&up, &mut b)?;
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
    // Int8 activation region [Xq | Hq]; public values.
    let q_words = groups::LAYER_Q_BYTES as usize / 4;
    let xq_words = groups::XQ_BYTES as usize / 4;
    let mut q_init = vec![0u32; q_words];
    let x: Vec<f32> = (0..COUNT * 2560)
        .map(|i| (((i * 2654435761usize) >> 9) % 97) as f32 / 97.0 - 0.5 + if i % 509 == 0 { 3.0 } else { 0.0 })
        .collect();
    if down {
        // Synthetic int8 hidden rows (codes -127..127, positive scales).
        for row in 0..prefix + 32 {
            let at = xq_words + row * 180;
            for w in 0..160 {
                let c = |k: usize| (((row * 640 + w * 4 + k) * 40503 >> 5) % 255) as u32;
                q_init[at + w] = (0..4).map(|k| c(k) << (8 * k)).fold(0, |a, b| a | b);
            }
            for blk in 0..20 {
                q_init[at + 160 + blk] = (0.001 + ((row + blk) % 13) as f32 * 0.0007).to_bits();
            }
        }
    }
    let q_handle = super::super::public_numeric::public(client, q_init.iter().flat_map(|w| w.to_le_bytes()).collect())?;
    let sub = |first: u64, bytes: u64| q_handle.clone().offset_start(first).offset_end(groups::LAYER_Q_BYTES - first - bytes);
    let xq = sub(0, groups::XQ_BYTES);
    let hq = sub(groups::XQ_BYTES, groups::HQ_BYTES);
    let raw_words = Zeroizing::new(raw.chunks_exact(4).map(|v| u32::from_le_bytes(v.try_into().unwrap())).collect::<Vec<_>>());
    let map_id = arena.upload_u32(&map)?;
    // Public weights ring and the (unread) image owner live outside the arena.
    let raw_handle = super::super::public_numeric::public(client, raw_words.iter().flat_map(|w| w.to_le_bytes()).collect())?;
    let image = client.empty(storage::WORKING_BYTES as usize);
    let out_words = if down { groups::PART_BYTES } else { groups::GU_BYTES } as usize / 4;
    let out = arena.allocate_zeroed(out_words)?;
    let status = arena.allocate_zeroed(groups::STATUS_WORDS)?;
    let fault = arena.allocate_zeroed(1)?;
    let info = |w: &[u32; WORDS]| MetadataBindingInfo::custom(w.chunks_exact(2).map(|v| u64::from(v[0]) | (u64::from(v[1]) << 32)).collect());
    if !down {
        let x_handle = super::super::public_numeric::public(client, x.iter().flat_map(|v| v.to_le_bytes()).collect())?;
        quant_x(client, x_handle, q_handle.clone(), u::handle(arena, fault)?, COUNT, false)?;
    }
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
            if down { hq.clone() } else { xq.clone() }.binding(),
            u::handle(arena, out)?.binding(),
            u::handle(arena, status)?.binding(),
        ])
        .with_read_dependencies(handles.iter().cloned().map(Handle::binding).collect())
        .with_info(info(&words));
    let op = Fused { dtype, down };
    unsafe {
        client.launch_unchecked(Box::new(op), CubeCount::Static(op.grid_x(), 1, 8), args);
    }
    if !down {
        let mut aw = *words;
        aw[3] = 0;
        let args = KernelArguments::new()
            .with_buffers(vec![
                u::handle(arena, map_id)?.binding(),
                u::handle(arena, out)?.binding(),
                hq.clone().binding(),
                u::handle(arena, status)?.binding(),
                u::handle(arena, fault)?.binding(),
            ])
            .with_info(info(&aw));
        unsafe {
            client.launch_unchecked(Box::new(ActivateQ), CubeCount::Static((31 * 20u32).div_ceil(256), 1, 8), args);
        }
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
    let q_now = Zeroizing::new(u::read_handle_prefix_u32(arena, q_handle.clone(), q_words)?);
    super::probe::sanitize(arena)?;
    if fault_word != 0 {
        return Err(format!("public fused fault {fault_word}").into());
    }
    let code = |word: u32, k: usize| f64::from(((word >> (8 * k)) & 255) as u8 as i8);
    // Dequantized activation rows (GPU codes): GU from Xq by token, DOWN from Hq.
    let act = |row: usize, k: usize| -> f64 {
        let (at, codes) = if down { (xq_words + row * 180, 160) } else { (row * 720, 640) };
        let scale = f64::from(f32::from_bits(q_now[at + codes + k / 32]));
        scale * code(q_now[at + k / 4], k % 4)
    };
    let mut worst_quant = 0f64;
    if !down {
        // Quantization: |x - d q| <= d/2 (+ rounding slack) for every value.
        for t in 0..COUNT {
            for k in 0..2560 {
                let d = f64::from(f32::from_bits(q_now[t * 720 + 640 + k / 32]));
                let e = (act(t, k) - f64::from(x[t * 2560 + k])).abs() / (d.max(1e-30));
                worst_quant = worst_quant.max(e);
            }
        }
        if worst_quant > 0.5001 {
            return Err(format!("public fused int8 quantization error {worst_quant}").into());
        }
    }
    let mut worst = 0f64;
    let mut checked = 0usize;
    let mut hq_mismatch = 0usize;
    for (g, &members) in MEMBERS.iter().enumerate() {
        let [a, b] = &weights[g];
        for t in 0..members {
            let route = route_firsts[g] + t;
            let arow: Vec<f64> = (0..cols).map(|k| if down { act(prefixes[g] + t, k) } else { act(route / 10, k) }).collect();
            let mut gu_row = vec![0f64; if down { 0 } else { 1280 }];
            for out_row in 0..stride_out {
                let (w, r) = if down { (a, out_row) } else { (if out_row % 2 == 0 { a } else { b }, out_row / 2) };
                let wr = &w[r * cols..(r + 1) * cols];
                let (mut sum, mut scale) = (0f64, 0f64);
                for k in 0..cols {
                    let p = f64::from(wr[k]) * arow[k];
                    sum += p;
                    scale += p.abs();
                }
                let at = if down { route * 2560 } else { (prefixes[g] + t) * 1280 } + out_row;
                let actual = f64::from(f32::from_bits(gpu[at]));
                if !down {
                    gu_row[out_row] = actual;
                }
                let error = (actual - sum).abs() / (scale + 1e-12);
                if !(error <= worst) {
                    worst = if error.is_nan() { f64::INFINITY } else { error };
                }
                checked += 1;
            }
            if down {
                if status_words[groups::ROUTES + route] != 2 {
                    return Err("public fused down route status".into());
                }
            } else {
                if status_words[route] != 1 {
                    return Err("public fused activation route status".into());
                }
                // ActivateQ: SwiGLU of the GPU GU row, int8 per 32 features (+-1 code).
                let row = xq_words + (prefixes[g] + t) * 180;
                for blk in 0..20 {
                    let y: Vec<f64> = (0..32).map(|j| {
                        let (gt, up) = (gu_row[2 * (blk * 32 + j)], gu_row[2 * (blk * 32 + j) + 1]);
                        gt / (1.0 + (-gt).exp()) * up
                    }).collect();
                    let amax = y.iter().fold(0f64, |m, v| m.max(v.abs()));
                    let d = f64::from(f32::from_bits(q_now[row + 160 + blk]));
                    if (d - amax / 127.0).abs() > 1e-5 * amax / 127.0 + 1e-30 {
                        hq_mismatch += 1;
                    }
                    for (j, v) in y.iter().enumerate() {
                        let k = blk * 32 + j;
                        let c = code(q_now[row + k / 4], k % 4);
                        let expect = if amax > 0.0 { (v * 127.0 / amax).round_ties_even() } else { 0.0 };
                        if (c - expect).abs() > 1.0 {
                            hq_mismatch += 1;
                        }
                    }
                }
            }
        }
    }
    if hq_mismatch != 0 {
        return Err(format!("public fused int8 activation mismatches {hq_mismatch}").into());
    }
    // Exact int32 dots; f32 scale products and accumulation over <=80 blocks.
    if worst > 1e-5 {
        return Err(format!("public fused numerics dtype={dtype} worst_relative={worst:e}").into());
    }
    Ok(serde_json::json!({"family":"claude-small-fused-int8","dtype":dtype,"down":down,"groups":8,"members":MEMBERS,
        "outputs_checked":checked,"worst_relative_error":worst,"worst_quantization_steps":worst_quant,
        "activation_checked":!down,"mixed_host_gpu_residency":true}))
}
