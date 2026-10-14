// MIT Strata native_bf16/native_router/s2_expert_grouped/native_moe fb58e0.
// COMMON_BEGIN
requires packed_4x8_integer_dot_product;
struct Params {count:u32,first:u32,rows:u32,r0:u32,r1:u32,r2:u32,r3:u32,r4:u32}
const ROUTE_FIRST:u32=3073u;
fn finite(x:f32)->bool {return abs(x)<=3.4028234663852886e38;}
// COMMON_END
// QUANT_BEGIN
@group(0) @binding(0) var<storage,read> input:array<f32>;
@group(0) @binding(1) var<storage,read_write> planes:array<u32>;
@group(0) @binding(2) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(3) var<uniform> p:Params;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(subgroup_id) sg:u32,
 @builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_size) sw:u32) {
    let b=g.x*8u+sg;let blocks=select(80u,200u,ACTIVATE);
    if sw!=32u || b>=blocks {return;}
    let i=b*32u+lane;var x=input[i];
    if ACTIVATE {let u=input[6400u+i];x=(x/(1.0+exp(-x)))*u;if !finite(u) {atomicOr(&fault[0],1u);}}
    let good=subgroupAll(finite(x));let maximum=subgroupMax(select(0.0,abs(x),finite(x)));
    let scale=maximum/127.0;var inv=0.0;if scale>0.0 {inv=1.0/scale;}
    let t=x*inv;var q=0;
    if good {q=clamp(i32(t+select(-0.5,0.5,t>=0.0)),-127,127);}else{atomicOr(&fault[0],1u);}
    let bits=bitcast<u32>(q)&255u;
    // Native load_x_chunk transpose: word 4h+f contains x[16h+4b+f].
    let word=lane%8u;let start=(word/4u)*16u+word%4u;
    let packed=subgroupShuffle(bits,start)|(subgroupShuffle(bits,start+4u)<<8u)
        |(subgroupShuffle(bits,start+8u)<<16u)|(subgroupShuffle(bits,start+12u)<<24u);
    if lane<8u {planes[b*10u+lane]=packed;}
    var hx=q;for(var d=16u;d>0u;d/=2u) {hx+=subgroupShuffleXor(hx,d);}
    if lane==0u {planes[b*10u+8u]=bitcast<u32>(select(0.0,scale,good));planes[b*10u+9u]=bitcast<u32>(hx);}
}
// QUANT_END
// EXPERT_BEGIN
@group(0) @binding(0) var<storage,read> weight:array<u32>;
@group(0) @binding(1) var<storage,read> planes:array<u32>;
@group(0) @binding(2) var<storage,read> packet:array<u32>;
@group(0) @binding(3) var<storage,read_write> output:array<f32>;
@group(0) @binding(4) var<storage,read_write> readiness:array<u32>;
@group(0) @binding(5) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(6) var<uniform> p:Params;
fn byte_at(at:u32)->u32 {return (weight[at/4u]>>((at%4u)*8u))&255u;}
fn word_at(at:u32)->u32 {let shift=(at%4u)*8u;let lo=weight[at/4u];if shift==0u {return lo;}return (lo>>shift)|(weight[at/4u+1u]<<(32u-shift));}
fn half_at(at:u32)->f32 {return unpack2x16float(byte_at(at)|(byte_at(at+1u)<<8u)).x;}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(subgroup_id) sg:u32,
 @builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_size) sw:u32) {
    let pairs=select(640u,1280u,DOWN);let pair=g.x*8u+sg;let rank=pair/pairs;
    if sw!=32u || rank>=10u || p.first>=512u || p.first%128u!=0u {return;}
    let expert=packet[ROUTE_FIRST+rank*2u];
    if expert<p.first || expert>=p.first+128u {return;}
    if expert>=512u {if lane==0u {atomicOr(&fault[0],8u);}return;}
    let row=pair%pairs;let n=select(2560u,640u,DOWN);let bytes=n/64u*18u;
    let r0=select(row,row*2u,DOWN);let r1=select(row+640u,row*2u+1u,DOWN);
    let base=(expert-p.first)*select(1280u,2560u,DOWN)*bytes;
    let a=base+r0*bytes;let b=base+r1*bytes;var sum0=0.0;var sum1=0.0;
    for(var chunk=lane;chunk<n/32u;chunk+=32u) {
        let at=select(chunk,(rank*20u+chunk),DOWN)*10u;
        let dx=bitcast<f32>(planes[at+8u]);let hx=bitcast<i32>(planes[at+9u]);
        let wa=a+(chunk/2u)*18u;let wb=b+(chunk/2u)*18u;var dot0=0;var dot1=0;
        let ca=wa+2u+(chunk%2u)*8u;let cb=wb+2u+(chunk%2u)*8u;
        for(var j=0u;j<8u;j++) {
            let shift=(j%4u)*2u;let q0=(word_at(ca+(j/4u)*4u)>>shift)&0x03030303u;
            let q1=(word_at(cb+(j/4u)*4u)>>shift)&0x03030303u;
            let x=planes[at+j];dot0+=dot4I8Packed(q0,x);dot1+=dot4I8Packed(q1,x);
        }
        // Keep separate product/add, matching S2's ordered accumulation.
        sum0+=(half_at(wa)*dx)*f32(dot0-hx);sum1+=(half_at(wb)*dx)*f32(dot1-hx);
    }
    for(var d=16u;d>0u;d/=2u) {
        let a0=subgroupShuffleDown(sum0,d);let a1=subgroupShuffleDown(sum1,d);
        if lane+d<32u {sum0+=a0;sum1+=a1;}
    }
    if lane==0u {
        if !finite(sum0) || !finite(sum1) {atomicOr(&fault[0],1u);}
        if DOWN {output[rank*2560u+row*2u]=sum0;output[rank*2560u+row*2u+1u]=sum1;if row==0u {readiness[10u+rank]=1u;}}
        else{output[rank*640u+row]=sum0;output[6400u+rank*640u+row]=sum1;if row==0u {readiness[rank]=1u;}}
    }
}
// EXPERT_END
// DOT_BEGIN
@group(0) @binding(0) var<storage,read> weight:array<u32>;
@group(0) @binding(1) var<storage,read> input:array<f32>;
@group(0) @binding(2) var<storage,read_write> output:array<f32>;
@group(0) @binding(3) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(4) var<uniform> p:Params;
var<workgroup> partial:array<f32,32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) tid:u32,
 @builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_id) sg:u32,@builtin(subgroup_size) sw:u32) {
    if sw!=32u || g.x>=p.rows || (p.rows!=512u && p.rows!=1u) {return;}
    if tid<32u {partial[tid]=0.0;}workgroupBarrier();var sum=0.0;
    for(var pair=tid;pair<1280u;pair+=256u) {let w=weight[g.x*1280u+pair];
        sum=fma(bitcast<f32>((w&65535u)<<16u),input[pair*2u],sum);
        sum=fma(bitcast<f32>(w&0xffff0000u),input[pair*2u+1u],sum);
    }
    for(var d=16u;d>0u;d/=2u) {sum+=subgroupShuffleXor(sum,d);}
    if lane==0u {partial[sg]=sum;}workgroupBarrier();
    if sg==0u {sum=partial[lane];for(var d=16u;d>0u;d/=2u) {sum+=subgroupShuffleXor(sum,d);}if lane==0u {output[g.x]=sum;if !finite(sum){atomicOr(&fault[0],1u);}}}
}
// DOT_END
// ROUTE_BEGIN
@group(0) @binding(0) var<storage,read> logits:array<f32>;
@group(0) @binding(1) var<storage,read_write> packet:array<u32>;
@group(0) @binding(2) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(3) var<uniform> p:Params;
@compute @workgroup_size(256)
fn main(@builtin(subgroup_id) sg:u32,@builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_size) sw:u32) {
    if sw!=32u || sg!=0u {return;}
    var x:array<f32,16>;var maximum=-3.4028234663852886e38;
    for(var i=0u;i<16u;i++) {x[i]=logits[lane+i*32u];if !finite(x[i]){atomicOr(&fault[0],1u);}maximum=max(maximum,x[i]);}
    for(var d=16u;d>0u;d/=2u) {maximum=max(maximum,subgroupShuffleXor(maximum,d));}
    var sum=0.0;for(var i=0u;i<16u;i++) {x[i]=exp(x[i]-maximum);sum+=x[i];}
    for(var d=16u;d>0u;d/=2u) {sum+=subgroupShuffleXor(sum,d);}
    let inv=1.0/sum;for(var i=0u;i<16u;i++) {x[i]*=inv;}
    var selected=0.0;var chosen_sum=0.0;
    for(var rank=0u;rank<10u;rank++) {
        var best=x[0];var expert=lane;
        for(var i=1u;i<16u;i++) {if x[i]>best {best=x[i];expert=lane+i*32u;}}
        for(var d=16u;d>0u;d/=2u) {let other=subgroupShuffleXor(best,d);let id=subgroupShuffleXor(expert,d);if other>best || (other==best && id<expert){best=other;expert=id;}}
        if expert%32u==lane {x[expert/32u]=-3.4028234663852886e38;packet[ROUTE_FIRST+rank*2u]=expert;chosen_sum+=best;}
        if rank==lane {selected=best;}
    }
    for(var d=16u;d>0u;d/=2u) {chosen_sum+=subgroupShuffleXor(chosen_sum,d);}
    let inverse=1.0/max(chosen_sum,0.00006103515625);
    if lane<10u {let w=selected*inverse;packet[ROUTE_FIRST+lane*2u+1u]=bitcast<u32>(w);if !finite(w)||w<0.0 {atomicOr(&fault[0],1u);}}
}
// ROUTE_END
// SHARED_BEGIN
@group(0) @binding(0) var<storage,read> gate:array<f32>;
@group(0) @binding(1) var<storage,read> up:array<f32>;
@group(0) @binding(2) var<storage,read_write> h:array<f32>;
@group(0) @binding(3) var<storage,read_write> planes:array<u32>;
@group(0) @binding(4) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(5) var<uniform> p:Params;
fn q8_code(x:f32,d:f32)->i32 {
    if d==0.0 {return 0;}
    let v=clamp(x/d,-127.0,127.0);
    return i32(abs(v)+0.5)*select(-1,1,v>=0.0);
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(subgroup_id) sg:u32,
 @builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_size) sw:u32) {
    let block=g.x*8u+sg;if sw!=32u || block>=20u {return;}
    let i=block*32u+lane;let x=gate[i];let y=(x/(1.0+exp(-x)))*up[i];h[i]=y;
    let good=subgroupAll(finite(y));let maximum=subgroupMax(select(0.0,abs(y),finite(y)));let d=min(maximum/127.0,65504.0);
    var sum=select(0.0,y,finite(y));for(var distance=16u;distance>0u;distance/=2u) {sum+=subgroupShuffleXor(sum,distance);}
    var q=0u;if good {q=bitcast<u32>(q8_code(y,d))&255u;}else{atomicOr(&fault[0],1u);}
    let first=lane&~3u;let packed=subgroupShuffle(q,first)|(subgroupShuffle(q,first+1u)<<8u)
        |(subgroupShuffle(q,first+2u)<<16u)|(subgroupShuffle(q,first+3u)<<24u);
    if lane%4u==0u {planes[block*8u+lane/4u]=packed;planes[160u+block*8u+lane/4u]=0u;}
    if lane==0u {let scale=unpack2x16float(pack2x16float(vec2<f32>(d,0.0))).x;
        planes[320u+block]=bitcast<u32>(select(0.0,scale,good&&finite(scale)));
        let half_sum=unpack2x16float(pack2x16float(vec2<f32>(clamp(sum,-65504.0,65504.0),0.0))).x;
        planes[340u+block]=bitcast<u32>(half_sum);
        if !finite(scale){atomicOr(&fault[0],1u);}}
}
// SHARED_END
// COMBINE_BEGIN
@group(0) @binding(0) var<storage,read> parts:array<f32>;
@group(0) @binding(1) var<storage,read> shared_values:array<f32>;
@group(0) @binding(2) var<storage,read> scalar:array<f32>;
@group(0) @binding(3) var<storage,read> packet:array<u32>;
@group(0) @binding(4) var<storage,read> readiness:array<u32>;
@group(0) @binding(5) var<storage,read_write> output:array<f32>;
@group(0) @binding(6) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(7) var<uniform> p:Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    let i=id.x;if i>=2560u {return;}
    var sum=parts[i]*bitcast<f32>(packet[ROUTE_FIRST+1u]);
    for(var rank=1u;rank<10u;rank++) {sum+=parts[rank*2560u+i]*bitcast<f32>(packet[ROUTE_FIRST+rank*2u+1u]);}
    let sh=shared_values[i]*(1.0/(1.0+exp(-scalar[0])));sum+=sh;var good=finite(sum)&&finite(sh);
    for(var rank=0u;rank<10u;rank++) {good=good&&readiness[rank]==1u&&readiness[10u+rank]==1u;}
    output[i]=select(0.0,sum,good);if !good {atomicOr(&fault[0],1u);}
}
// COMBINE_END
