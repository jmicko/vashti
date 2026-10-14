// MIT Strata fb58e0 record_front/capture_prefill and native_qsa norm.
// COMMON_BEGIN
struct Params {count:u32,start:u32,columns:u32,rows:u32,first:u32,stride:u32,epsilon:f32,reserved:u32}
fn finite(x:f32)->bool {return abs(x)<=3.4028234663852886e38;}
// COMMON_END
// EMBED_BEGIN
@group(0) @binding(0) var<storage,read> weight:array<u32>;
@group(0) @binding(1) var<storage,read> tokens:array<u32>;
@group(0) @binding(2) var<storage,read_write> output:array<f32>;
@group(0) @binding(3) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(4) var<uniform> p:Params;
fn byte_at(at:u32)->u32 {return (weight[at/4u]>>((at%4u)*8u))&255u;}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    let d=id.x;let t=id.y;if t>=p.count || d>=2560u {return;}
    let token=tokens[t];if token>=p.columns {atomicOr(&fault[0],1u);return;}if token<p.first || token-p.first>=p.rows {return;}
    let b=(token-p.first)*1360u+(d/256u)*136u;let sub=(d%256u)/32u;
    let low=(byte_at(b+4u+sub/2u)>>((sub%2u)*4u))&15u;
    let high=((byte_at(b+2u)|(byte_at(b+3u)<<8u))>>(sub*2u))&3u;
    let scale=unpack2x16float(byte_at(b)|(byte_at(b+1u)<<8u)).x*f32(i32(low|(high<<4u))-32);
    let qs=byte_at(b+8u+sub*16u+d%16u);let nib=(qs>>select(0u,4u,d%32u>=16u))&15u;
    let table=array<u32,4>(0xbfad9881u,0xf6eaddcfu,0x26190d01u,0x71594535u);
    let v=(table[nib/4u]>>((nib%4u)*8u))&255u;let signed=i32(v)-select(0,256,v>=128u);
    let value=scale*f32(signed);output[t*2560u+d]=select(0.0,value,finite(value));if !finite(value) {atomicOr(&fault[0],1u);}
}
// EMBED_END
// NORM_BEGIN
@group(0) @binding(0) var<storage,read> input:array<f32>;
@group(0) @binding(1) var<storage,read> gamma:array<f32>;
@group(0) @binding(2) var<storage,read_write> output:array<f32>;
@group(0) @binding(3) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(4) var<uniform> p:Params;
var<workgroup> partial:array<f32,32>;
@compute @workgroup_size(1024)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) tid:u32,@builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_id) warp:u32,@builtin(subgroup_size) sw:u32) {
    let t=group.x;if sw!=32u || t>=p.count {return;}
    var sum=0.0;for(var d=tid;d<p.columns;d+=1024u) {let x=input[t*p.columns+d];sum+=x*x;}
    for(var off=16u;off>0u;off/=2u) {sum+=subgroupShuffleXor(sum,off);}
    if lane==0u {partial[warp]=sum;}workgroupBarrier();
    sum=partial[lane];for(var off=16u;off>0u;off/=2u) {sum+=subgroupShuffleXor(sum,off);}
    let scale=inverseSqrt(sum/f32(p.columns)+p.epsilon);
    for(var d=tid;d<p.columns;d+=1024u) {let x=input[t*p.columns+d];let y=scale*x*gamma[d];let ok=finite(x)&&finite(y)&&finite(scale)&&finite(gamma[d]);output[t*p.columns+d]=select(0.0,y,ok);if !ok {atomicOr(&fault[0],1u);}}
}
// NORM_END
// BROADCAST_BEGIN
@group(0) @binding(0) var<storage,read> embedding:array<f32>;
@group(0) @binding(1) var<storage,read_write> residual:array<f32>;
@group(0) @binding(2) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(3) var<uniform> p:Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {let i=id.x;let t=id.y;if t>=p.count || i>=10240u {return;}let at=t*10240u+i;let x=residual[at]+embedding[t*2560u+i%2560u];residual[at]=select(0.0,x,finite(x));if !finite(x) {atomicOr(&fault[0],1u);}}
// BROADCAST_END
// POSITIONS_BEGIN
@group(0) @binding(0) var<storage,read_write> positions:array<u32>;
@group(0) @binding(1) var<uniform> p:Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {let t=id.x;if t<p.count {positions[p.start+t]=p.start+t;}}
// POSITIONS_END
// APPEND_BEGIN
@group(0) @binding(0) var<storage,read> key:array<f32>;
@group(0) @binding(1) var<storage,read> value:array<f32>;
@group(0) @binding(2) var<storage,read_write> keys:array<f32>;
@group(0) @binding(3) var<storage,read_write> values:array<f32>;
@group(0) @binding(4) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(5) var<uniform> p:Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    let i=id.x;let t=id.y;if t>=p.count || i>=512u {return;}
    let k=unpack2x16float(pack2x16float(vec2<f32>(key[t*512u+i],0.0))).x;
    let v=unpack2x16float(pack2x16float(vec2<f32>(value[t*512u+i],0.0))).x;
    let ok=finite(k)&&finite(v);keys[(p.start+t)*512u+i]=select(0.0,k,ok);values[(p.start+t)*512u+i]=select(0.0,v,ok);if !ok {atomicOr(&fault[0],1u);}
}
// APPEND_END
