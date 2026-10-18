// MIT Strata fb58e0 Intel prefill/kernels.dp.cpp: split, RMS, RoPE, FP16 gate.
// COMMON_BEGIN
struct Params {count:u32,start:u32,capacity:u32,input_stride:u32,output_stride:u32,status_stride:u32,heads:u32,width:u32,block_positions:u32,epsilon:f32,theta_scale:f32,reserved:u32}
fn finite(x:f32)->bool {return abs(x)<=3.4028234663852886e38;}
// COMMON_END
// POSITIONS_BEGIN
@group(0) @binding(0) var<storage,read_write> positions:array<u32>;
@group(0) @binding(1) var<uniform> p:Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>){if id.x<p.count{positions[p.start+id.x]=p.start+id.x;}}
// POSITIONS_END
// JOIN_FLAGS_BEGIN
@group(0) @binding(0) var<storage,read> flags:array<u32>;
@group(0) @binding(1) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(2) var<uniform> p:Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>){
    let i=id.x;if i>=p.count*p.status_stride{return;}
    let row=i/p.status_stride;let col=i%p.status_stride;
    if flags[p.input_stride+row*p.output_stride+col]!=1u{atomicOr(&fault[0],1u);}
}
// JOIN_FLAGS_END
// COMPATIBILITY_BEGIN
@group(0) @binding(0) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(1) var<storage,read_write> flags:array<u32>;
@group(0) @binding(2) var<uniform> p:Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>){
    let t=id.y;let i=id.x;if t>=p.count{return;}
    let n=p.start+t+1u;var width=p.input_stride;
    if p.output_stride==1u{width=n/4u;}
    if p.output_stride==2u{width=n/4u+select(0u,1u,n%4u!=0u);}
    if p.output_stride==3u{width=24u*min(n,2051u);}
    if i<width{flags[t*p.status_stride+i]=select(0u,1u,atomicLoad(&fault[0])==0u);}
}
// COMPATIBILITY_END
// POOL_MEAN_BEGIN
@group(0) @binding(0) var<storage,read> tail_state:array<f32>;
@group(0) @binding(1) var<storage,read> raw:array<f32>;
@group(0) @binding(2) var<storage,read_write> means:array<f32>;
@group(0) @binding(3) var<storage,read_write> status:array<u32>;
@group(0) @binding(4) var<uniform> p:Params;
var<workgroup> good:array<u32,128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) d:u32) {
    let block=g.x;if block<p.start/4u||block>=(p.start+p.count)/4u{return;}
    var sum=0.0;var valid=1u;
    for(var j=0u;j<4u;j++) {
        let token=block*4u+j;var x=0.0;
        if token<p.start {x=tail_state[j*128u+d];}else{x=unpack2x16float(pack2x16float(vec2<f32>(raw[(token-p.start)*128u+d],0.0))).x;}
        if j==0u {sum=x;}else{sum+=x;}if !finite(x){valid=0u;}
    }
    let mean=fma(0.25,sum,0.0);if !finite(mean){valid=0u;}
    means[block*128u+d]=select(0.0,mean,valid==1u);good[d]=valid;workgroupBarrier();
    for(var n=64u;n>0u;n/=2u){if d<n{good[d]&=good[d+n];}workgroupBarrier();}
    if d==0u{status[block]=good[0];}
}
// POOL_MEAN_END
// POOL_FLAGS_BEGIN
@group(0) @binding(0) var<storage,read> valid:array<u32>;
@group(0) @binding(1) var<storage,read_write> means:array<u32>;
@group(0) @binding(2) var<storage,read_write> norms:array<u32>;
@group(0) @binding(3) var<uniform> p:Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>){
    let b=id.x;let t=id.y;if t>=p.count||b>=(p.start+t+1u)/4u{return;}
    var ok=1u;if b>=p.start/4u{ok=valid[b];}
    means[t*p.status_stride+b]=ok;norms[t*p.status_stride+b]=ok;
}
// POOL_FLAGS_END
// SPLIT_BEGIN
@group(0) @binding(0) var<storage,read> raw:array<f32>;
@group(0) @binding(1) var<storage,read_write> packed:array<f32>;
@group(0) @binding(2) var<storage,read_write> status:array<u32>;
@group(0) @binding(3) var<uniform> p:Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    let i=id.x;let t=id.y;if t>=p.count||i>=12288u{return;}
    let d=i%6144u;let extra=select(0u,256u,i>=6144u);
    let x=raw[t*12288u+(d/256u)*512u+d%256u+extra];let good=finite(x);
    if i<6144u {packed[t*6144u+i]=select(0.0,x,good);}status[t*12288u+i]=select(0u,1u,good);
}
// SPLIT_END
// NORM_BEGIN
@group(0) @binding(0) var<storage,read> input:array<f32>;
@group(0) @binding(1) var<storage,read> gamma:array<f32>;
@group(0) @binding(2) var<storage,read_write> output:array<f32>;
@group(0) @binding(3) var<storage,read_write> status:array<u32>;
@group(0) @binding(4) var<uniform> p:Params;
var<workgroup> sums:array<f32,8>;
var<workgroup> good:array<u32,8>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) tid:u32,@builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_id) warp:u32,@builtin(subgroup_size) sw:u32) {
    let t=g.y;let head=g.x;let n=select(p.heads,(p.start+t+1u)/4u,p.block_positions!=0u);
    if sw!=32u{if tid==0u&&t<p.count&&head<n&&(p.block_positions==0u||head>=p.start/4u){status[t*p.status_stride+head]=0u;}return;}
    if t>=p.count||head>=n||(p.block_positions!=0u&&head<p.start/4u){return;}
    let at=t*p.input_stride+head*p.width;var x=0.0;var ss=0.0;var ok=1u;
    if tid<p.width{x=input[at+tid];ss=x*x;if !finite(x)||!finite(gamma[tid]){ok=0u;}}
    for(var d=16u;d>0u;d/=2u){ss+=subgroupShuffleXor(ss,d);ok&=subgroupShuffleXor(ok,d);}
    if lane==0u{sums[warp]=ss;good[warp]=ok;}workgroupBarrier();
    ss=0.0;ok=1u;if lane<8u{ss=sums[lane];ok=good[lane];}
    for(var d=16u;d>0u;d/=2u){ss+=subgroupShuffleXor(ss,d);ok&=subgroupShuffleXor(ok,d);}
    let scale=inverseSqrt(ss/f32(p.width)+p.epsilon);
    if tid<p.width{let y=(scale*x)*gamma[tid];let valid=ok==1u&&finite(scale)&&finite(y);output[t*p.output_stride+head*p.width+tid]=select(0.0,y,valid);if !valid{ok=0u;}}
    for(var d=16u;d>0u;d/=2u){ok&=subgroupShuffleXor(ok,d);}if lane==0u{good[warp]=ok;}workgroupBarrier();
    if tid==0u{var valid=1u;for(var w=0u;w<8u;w++){valid&=good[w];}status[t*p.status_stride+head]=valid;}
}
// NORM_END
// ROPE_BEGIN
@group(0) @binding(0) var<storage,read> positions:array<u32>;
@group(0) @binding(1) var<storage,read_write> output:array<f32>;
@group(0) @binding(2) var<storage,read_write> status:array<u32>;
@group(0) @binding(3) var<uniform> p:Params;
var<workgroup> good:array<u32,64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) tid:u32) {
    let t=g.y;let head=g.x;let n=select(p.heads,(p.start+t+1u)/4u,p.block_positions!=0u);
    if t>=p.count||head>=n||(p.block_positions!=0u&&head<p.start/4u){return;}
    let position=positions[select(p.start+t,head*4u,p.block_positions!=0u)];
    let at=t*p.output_stride+head*p.width;var ok=select(0u,1u,position<p.capacity&&status[t*p.status_stride+head]==1u);
    if tid<32u{let a=output[at+tid];let b=output[at+tid+32u];let angle=f32(position)*pow(p.theta_scale,f32(tid));let c=cos(angle);let s=sin(angle);
        let low=a*c-b*s;let high=a*s+b*c;if !finite(low)||!finite(high){ok=0u;}output[at+tid]=select(0.0,low,ok==1u);output[at+tid+32u]=select(0.0,high,ok==1u);}
    good[tid]=ok;workgroupBarrier();for(var d=32u;d>0u;d/=2u){if tid<d{good[tid]&=good[tid+d];}workgroupBarrier();}
    if tid==0u{status[t*p.status_stride+head]=good[0];}
}
// ROPE_END
// GATE_BEGIN
@group(0) @binding(0) var<storage,read> raw:array<f32>;
@group(0) @binding(1) var<storage,read> attention:array<f32>;
@group(0) @binding(2) var<storage,read_write> packed:array<u32>;
@group(0) @binding(3) var<storage,read_write> status:array<u32>;
@group(0) @binding(4) var<uniform> p:Params;
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    let t=id.y;let pair=id.x;if t>=p.count||pair>=3072u{return;}
    let i=pair*2u;let gate=t*12288u+(i/256u)*512u+256u+i%256u;
    let a=attention[t*6144u+i];let b=attention[t*6144u+i+1u];let ga=raw[gate];let gb=raw[gate+1u];
    let x=a*(1.0/(1.0+exp(-ga)));let y=b*(1.0/(1.0+exp(-gb)));let bits=pack2x16float(vec2<f32>(x,y));let back=unpack2x16float(bits);
    let oka=finite(a)&&finite(ga)&&finite(x)&&finite(back.x);let okb=finite(b)&&finite(gb)&&finite(y)&&finite(back.y);
    packed[t*3072u+pair]=pack2x16float(vec2<f32>(select(0.0,back.x,oka),select(0.0,back.y,okb)));
    status[t*p.status_stride+i]=select(0u,1u,oka);status[t*p.status_stride+i+1u]=select(0u,1u,okb);
}
// GATE_END
