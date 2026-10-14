// MIT Strata fb58e0 Intel prefill.cpp native FP16 experts. Original packed
// format helpers retain ggml/Strata notices from their source modules.
// COMMON_BEGIN
struct Params {count:u32,columns:u32,rows:u32,row_bytes:u32,group:u32,role:u32,r0:u32,r1:u32,r2:u32,r3:u32,r4:u32,r5:u32,r6:u32,r7:u32,r8:u32,r9:u32}
fn finite(x:f32)->bool {return abs(x)<=3.4028234663852886e38;}
fn signed_byte(x:u32)->f32 {return f32(i32(x)-select(0,256,x>=128u));}
fn valid()->bool {
    return p.count>=1u&&p.count<=256u&&p.role<=2u&&p.group<p.count*10u
        && arrayLength(&map)==8192u&&map[0]==p.count&&map[1]==0u&&map[2]==p.count
        && map[3]>=1u&&map[3]<=512u&&map[3]<=p.count*10u&&p.group<map[3]
        && map[4u+p.group]<512u&&map[516u+p.group]>=1u&&map[516u+p.group]<=p.count&&map[1028u+p.group]+map[516u+p.group]<=p.count*10u;
}
// COMMON_END
// GATHER_BEGIN
@group(0) @binding(0) var<storage,read> map:array<u32>;
@group(0) @binding(1) var<storage,read> input:array<f32>;
@group(0) @binding(2) var<storage,read_write> packed:array<u32>;
@group(0) @binding(3) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(4) var<uniform> p:Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    if !valid() {return;}let i=id.x;let length=map[516u+p.group];
    if i>=length*p.columns/2u {return;}let member=i/(p.columns/2u);let col=(i%(p.columns/2u))*2u;
    let route=map[1540u+map[1028u+p.group]+member];if route>=p.count*10u {atomicOr(&fault[0],128u);return;}
    let at=(route/10u)*2560u+col;
    if at+1u>=arrayLength(&input)||i>=arrayLength(&packed) {atomicOr(&fault[0],128u);return;}
    let a=input[at];let b=input[at+1u];let bits=pack2x16float(vec2<f32>(a,b));let v=unpack2x16float(bits);
    let good=finite(a)&&finite(b)&&finite(v.x)&&finite(v.y);packed[i]=select(0u,bits,good);if !good {atomicOr(&fault[0],128u);}
}
// GATHER_END
// DEQUANT_BEGIN
BINDINGS
fn strata_gate(index:u32)->u32 {
    if host_gate {return raw[p.r0+select(0u,p.r2,p.role==2u)+index];}
    return gate0[index/4u][index%4u];
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    if !valid() {return;}
    if arrayLength(&addresses)!=2048u||arrayLength(&stage_addresses)!=2048u||arrayLength(&strata_grid)!=8192u
        || arrayLength(&raw)!=16777216u||arrayLength(&packed)!=33554432u {atomicOr(&fault[0],128u);return;}
    let src=addresses[p.role*512u+map[4u+p.group]];
    let original_rows=select(p.rows/2u,p.rows,p.role==2u);let bytes=original_rows*p.row_bytes;
    if src.z!=bytes/4u||src.w>1u||p.r0+p.r2+bytes/4u>arrayLength(&raw) {atomicOr(&fault[0],128u);return;}
    strata_gate_address=src.xy;host_gate=src.w==1u;
    CHOOSE_UP
    let i=id.x;if i>=p.rows*p.columns/2u {return;}let row=i/(p.columns/2u);let col=(i%(p.columns/2u))*2u;
    let a=value(row,col);let b=value(row,col+1u);let bits=pack2x16float(vec2<f32>(a,b));let v=unpack2x16float(bits);
    let good=finite(a)&&finite(b)&&finite(v.x)&&finite(v.y);let first=select(p.r3,p.r4,p.role==2u);
    packed[first+i]=select(0u,bits,good);if !good {atomicOr(&fault[0],128u);}
}
// DEQUANT_END
// GEMM_BEGIN
@group(0) @binding(0) var<storage,read> map:array<u32>;
@group(0) @binding(1) var<storage,read> input:array<u32>;
@group(0) @binding(2) var<storage,read> weight:array<u32>;
@group(0) @binding(3) var<storage,read_write> partial:array<f32>;
@group(0) @binding(4) var<storage,read_write> status:array<u32>;
@group(0) @binding(5) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(6) var<uniform> p:Params;
var<workgroup> a:array<f32,256>;
var<workgroup> b:array<f32,256>;
fn half(bits:u32,odd:bool)->f32 {let v=unpack2x16float(bits);return select(v.x,v.y,odd);}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) tid:u32) {
    if !valid() {return;}let length=map[516u+p.group];let first=g.y*8u;let member=first+tid/8u;let row=g.x*8u+tid%8u;var sum=0.0;
    for(var k=0u;k<p.columns;k+=32u) {
        for(var i=tid;i<256u;i+=64u) {let m=first+i/32u;let wr=g.x*8u+i/32u;let col=k+i%32u;var av=0.0;var bv=0.0;
            if m<length {av=half(input[m*(p.columns/2u)+col/2u],(col%2u)!=0u);}
            if wr<p.rows {bv=half(weight[select(p.r3,p.r4,p.role==2u)+wr*(p.columns/2u)+col/2u],(col%2u)!=0u);}
            a[i]=av;b[i]=bv;
        }workgroupBarrier();
        for(var j=0u;j<32u;j++) {sum=fma(a[(tid/8u)*32u+j],b[(tid%8u)*32u+j],sum);}workgroupBarrier();
    }
    if member<length&&row<p.rows {
        let route=map[1540u+map[1028u+p.group]+member];if route>=p.count*10u {atomicOr(&fault[0],128u);return;}
        let base=select(member*1280u,route*2560u,p.role==2u);let good=finite(sum);
        partial[base+row]=select(0.0,sum,good);if !good {atomicOr(&fault[0],128u);}
        if p.role==2u&&row==0u {status[2560u+route]=select(0u,1u,good&&status[route]==1u);}
    }
}
// GEMM_END
// ACTIVATE_BEGIN
@group(0) @binding(0) var<storage,read> map:array<u32>;
@group(0) @binding(1) var<storage,read> gu:array<f32>;
@group(0) @binding(2) var<storage,read_write> hidden:array<u32>;
@group(0) @binding(3) var<storage,read_write> status:array<u32>;
@group(0) @binding(4) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(5) var<uniform> p:Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    if !valid() {return;}let i=id.x;if i>=map[516u+p.group]*320u {return;}
    let member=i/320u;let col=(i%320u)*2u;let base=member*1280u+col*2u;
    let route=map[1540u+map[1028u+p.group]+member];if route>=p.count*10u {atomicOr(&fault[0],128u);return;}
    let g=vec2<f32>(gu[base],gu[base+2u]);let u=vec2<f32>(gu[base+1u],gu[base+3u]);
    let y=(g/(vec2<f32>(1.0)+exp(-g)))*u;
    let bits=pack2x16float(clamp(y,vec2<f32>(-65504.0),vec2<f32>(65504.0)));let rounded=unpack2x16float(bits);
    // Native hf_sat clamps infinities and leaves NaNs visible as faults.
    let good=(bitcast<u32>(y.x)&0x7fffffffu)<=0x7f800000u
        &&(bitcast<u32>(y.y)&0x7fffffffu)<=0x7f800000u&&finite(rounded.x)&&finite(rounded.y);
    hidden[i]=select(0u,bits,good);
    if col==0u {status[route]=select(0u,1u,good);}if !good {atomicOr(&fault[0],128u);}
}
// ACTIVATE_END
