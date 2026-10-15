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
    let t=group.x;if sw!=32u {if tid==0u {atomicOr(&fault[0],1u);}return;}if t>=p.count {return;}
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
// TARGET_BROADCAST_BEGIN
@group(0) @binding(0) var<storage,read> embedding:array<f32>;
@group(0) @binding(1) var<storage,read_write> residual:array<f32>;
@group(0) @binding(2) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(3) var<uniform> p:Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {let i=id.x;let t=id.y;if t>=p.count || i>=10240u {return;}let at=t*10240u+i;let x=embedding[t*2560u+i%2560u];residual[at]=select(0.0,x,finite(x));if !finite(x) {atomicOr(&fault[0],1u);}}
// TARGET_BROADCAST_END
// POSITIONS_BEGIN
@group(0) @binding(0) var<storage,read_write> positions:array<u32>;
@group(0) @binding(1) var<uniform> p:Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {let t=id.x;if t<p.count {positions[p.start+t]=p.start+t;}}
// POSITIONS_END
// APPEND_BEGIN
// STRATA_DENSE_PACKED: this original packed KV kernel requires subgroup32.
// kv_q8.dp.cpp: max over64, FP16-stored scale, nearest-even signed codes.
// Two64-value groups share a128-thread workgroup so one lane owns each packed
// pair of scales. Four-cell/head page order stays unchanged.
@group(0) @binding(0) var<storage,read> key:array<f32>;
@group(0) @binding(1) var<storage,read> value:array<f32>;
@group(0) @binding(2) var<storage,read_write> keys:array<u32>;
@group(0) @binding(3) var<storage,read_write> values:array<u32>;
@group(0) @binding(4) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(5) var<uniform> p:Params;
var<workgroup> kmax:array<f32,4>;
var<workgroup> vmax:array<f32,4>;
var<workgroup> valid:array<u32,4>;
var<workgroup> kscale:array<f32,2>;
var<workgroup> vscale:array<f32,2>;
var<workgroup> kcode:array<u32,128>;
var<workgroup> vcode:array<u32,128>;
fn nearest_even(x:f32)->i32 {let lo=floor(x);let d=x-lo;return i32(lo)+select(0,1,d>0.5||(d==0.5&&(i32(lo)&1)!=0));}
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) tid:u32,@builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_id) warp:u32,@builtin(subgroup_size) sw:u32) {
    let t=g.y;let group=g.x;if t>=p.count||group>=4u{return;}
    if sw!=32u||p.start>=4096u||t>=4096u-p.start {if tid==0u{atomicOr(&fault[0],1u);}return;}
    let at=t*512u+group*128u+tid;let k=key[at];let v=value[at];
    var ok=select(0u,1u,finite(k)&&finite(v));var ka=abs(select(0.0,k,finite(k)));var va=abs(select(0.0,v,finite(v)));
    for(var d=16u;d>0u;d/=2u){ka=max(ka,subgroupShuffleXor(ka,d));va=max(va,subgroupShuffleXor(va,d));ok&=subgroupShuffleXor(ok,d);}
    if lane==0u{kmax[warp]=ka;vmax[warp]=va;valid[warp]=ok;}workgroupBarrier();
    let pair=(tid/64u)*2u;
    let ks=unpack2x16float(pack2x16float(vec2<f32>(max(kmax[pair],kmax[pair+1u])/127.0,0.0))).x;
    let vs=unpack2x16float(pack2x16float(vec2<f32>(max(vmax[pair],vmax[pair+1u])/127.0,0.0))).x;
    let good=(valid[pair]&valid[pair+1u])==1u&&finite(ks)&&finite(vs)&&ks>=0.0&&vs>=0.0;
    var kc=0;var vc=0;if good&&ks>0.0{kc=clamp(nearest_even(k/ks),-127,127);}if good&&vs>0.0{vc=clamp(nearest_even(v/vs),-127,127);}
    kcode[tid]=bitcast<u32>(kc)&255u;vcode[tid]=bitcast<u32>(vc)&255u;
    if tid%64u==0u{kscale[tid/64u]=select(0.0,ks,good);vscale[tid/64u]=select(0.0,vs,good);}
    if !good{atomicOr(&fault[0],1u);}workgroupBarrier();
    let token=p.start+t;let row=((token/4u)*2u+group/2u)*4u+token%4u;
    if tid<32u{let d=tid*4u;let dst=row*66u+(group%2u)*32u+tid;keys[dst]=kcode[d]|(kcode[d+1u]<<8u)|(kcode[d+2u]<<16u)|(kcode[d+3u]<<24u);values[dst]=vcode[d]|(vcode[d+1u]<<8u)|(vcode[d+2u]<<16u)|(vcode[d+3u]<<24u);}
    if tid==0u{let dst=row*66u+64u+group%2u;keys[dst]=pack2x16float(vec2<f32>(kscale[0],kscale[1]));values[dst]=pack2x16float(vec2<f32>(vscale[0],vscale[1]));}
}
// APPEND_END
// GUARD_BEGIN
@group(0) @binding(0) var<storage,read> keys:array<u32>;
@group(0) @binding(1) var<storage,read> values:array<u32>;
@group(0) @binding(2) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(3) var<uniform> p:Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    let i=id.x;if i>=540672u {return;}
    let k=keys[i];let v=values[i];
    let row=i/66u;let token=(row/8u)*4u+row%4u;
    if token>=p.start {if k!=0u || v!=0u {atomicOr(&fault[0],4u);}}
    else if i%66u>=64u{let a=unpack2x16float(k);let b=unpack2x16float(v);if !finite(a.x)||!finite(a.y)||!finite(b.x)||!finite(b.y)||a.x<0.0||a.y<0.0||b.x<0.0||b.y<0.0{atomicOr(&fault[0],1u);}}
    else {for(var j=0u;j<4u;j++){if ((k>>(j*8u))&255u)==128u||((v>>(j*8u))&255u)==128u{atomicOr(&fault[0],1u);}}}
}
// GUARD_END
// REWIND_BEGIN
@group(0) @binding(0) var<storage,read_write> keys:array<u32>;
@group(0) @binding(1) var<storage,read_write> values:array<u32>;
@group(0) @binding(2) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(3) var<uniform> p:Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    let i=id.x;if p.rows>4096u || p.start>p.rows {if i==0u {atomicOr(&fault[0],1u);}return;}
    if i<540672u {let row=i/66u;let token=(row/8u)*4u+row%4u;if token>=p.start {keys[i]=0u;values[i]=0u;}}
}
// REWIND_END
// IDENTITY_BEGIN
@group(0) @binding(0) var<storage,read_write> ids:array<u32>;
@group(0) @binding(1) var<uniform> p:Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {let i=id.x;if i<=p.start && i<4096u {ids[i]=i;}}
// IDENTITY_END
// SELECT_BEGIN
// capture_round: copy_row_to_first selects R, injection and mixed TOGETHER.
// All selected source rows are disjoint from row0; each destination has one writer.
@group(0) @binding(0) var<storage,read_write> residual:array<f32>;
@group(0) @binding(1) var<storage,read_write> injection:array<f32>;
@group(0) @binding(2) var<storage,read_write> mixed:array<f32>;
@group(0) @binding(3) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(4) var<uniform> p:Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    let i=id.x;let row=p.columns;
    if p.count<2u || p.count>8u || row!=p.count-1u ||
       arrayLength(&residual)<p.count*10240u || arrayLength(&injection)<p.count*4u ||
       arrayLength(&mixed)<p.count*2560u {
        if i==0u {atomicOr(&fault[0],8u);}return;
    }
    if i<10240u {let x=residual[row*10240u+i];residual[i]=select(0.0,x,finite(x));if !finite(x) {atomicOr(&fault[0],1u);}}
    if i<4u {let x=injection[row*4u+i];injection[i]=select(0.0,x,finite(x));if !finite(x) {atomicOr(&fault[0],1u);}}
    if i<2560u {let x=mixed[row*2560u+i];mixed[i]=select(0.0,x,finite(x));if !finite(x) {atomicOr(&fault[0],1u);}}
}
// SELECT_END
