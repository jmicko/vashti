// MIT Strata fb58e0 Intel verify_kernels.dp.cpp. Runtime T=1..8, S=128.
// COMMON_BEGIN
struct Params {count:u32,start:u32,channels:u32,key_heads:u32,value_heads:u32,gamma_shared:u32,epsilon:f32,reserved:u32}
fn finite(x:f32)->bool {return abs(x)<=3.4028234663852886e38;}
// COMMON_END
// CONV_BEGIN
@group(0) @binding(0) var<storage,read> raw:array<f32>;
@group(0) @binding(1) var<storage,read> weights:array<f32>;
@group(0) @binding(2) var<storage,read> old:array<f32>;
@group(0) @binding(3) var<storage,read_write> next:array<f32>;
@group(0) @binding(4) var<storage,read_write> query:array<f32>;
@group(0) @binding(5) var<storage,read_write> key:array<f32>;
@group(0) @binding(6) var<storage,read_write> value:array<f32>;
@group(0) @binding(7) var<storage,read_write> status:array<u32>;
@group(0) @binding(8) var<storage,read_write> prepared_status:array<u32>;
@group(0) @binding(9) var<uniform> p:Params;
var<workgroup> sums:array<f32,4>;
var<workgroup> valid:array<u32,4>;
fn history(t:u32,c:u32,j:u32)->f32 {let at=t+j;if at<3u {return old[c*3u+at];}return raw[(at-3u)*p.channels+c];}
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) col:u32,@builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_id) warp:u32,@builtin(subgroup_size) sw:u32) {
    let head=group.x;let t=group.y;if sw!=32u || t>=p.count || head>=p.channels/128u {return;}
    let c=head*128u+col;let a=history(t,c,0u);let b=history(t,c,1u);let d=history(t,c,2u);let x=raw[t*p.channels+c];
    let sum=a*weights[c*4u]+b*weights[c*4u+1u]+d*weights[c*4u+2u]+x*weights[c*4u+3u];
    var y=sum/(1.0+exp(-sum));var ok=select(0u,1u,finite(a)&&finite(b)&&finite(d)&&finite(x)&&finite(sum)&&finite(y));
    var sq=y*y;
    for(var distance=16u;distance>0u;distance/=2u) {sq+=subgroupShuffleXor(sq,distance);ok&=subgroupShuffleXor(ok,distance);}
    if lane==0u {sums[warp]=sq;valid[warp]=ok;}workgroupBarrier();
    let total=sums[0]+sums[1]+sums[2]+sums[3];let all_good=valid[0]&valid[1]&valid[2]&valid[3];
    if head<2u*p.key_heads {y*=inverseSqrt(total+p.epsilon);}
    ok=all_good;if !finite(y) {ok=0u;}
    let emit=select(0.0,y,ok==1u);status[t*p.channels+c]=ok;prepared_status[t*p.channels+c]=ok;
    if head<p.key_heads {query[t*p.key_heads*128u+c]=emit;}
    else if head<2u*p.key_heads {key[t*p.key_heads*128u+(head-p.key_heads)*128u+col]=emit;}
    else {value[t*p.value_heads*128u+(head-2u*p.key_heads)*128u+col]=emit;}
    if t==p.count-1u {
        // The last three RAW inputs, never the activated/normalized values.
        next[c*3u]=history(p.count,c,0u);next[c*3u+1u]=history(p.count,c,1u);next[c*3u+2u]=history(p.count,c,2u);
    }
}
// CONV_END
// CONTROLS_BEGIN
@group(0) @binding(0) var<storage,read> alpha:array<f32>;
@group(0) @binding(1) var<storage,read> beta:array<f32>;
@group(0) @binding(2) var<storage,read> params:array<f32>;
@group(0) @binding(3) var<storage,read_write> controls:array<f32>;
@group(0) @binding(4) var<storage,read_write> status:array<u32>;
@group(0) @binding(5) var<uniform> p:Params;
fn positive_log1p(x:f32)->f32 {let y=1.0+x;if y==1.0 {return x;}return log(y)*(x/(y-1.0));}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    let h=id.x;let t=id.y;if h>=p.value_heads || t>=p.count {return;}
    let a=alpha[t*p.value_heads+h];let b=beta[t*p.value_heads+h];let dt=params[h];let sa=params[p.value_heads+h];let biased=a+dt;
    var sp=biased;if biased<=20.0 {sp=positive_log1p(exp(biased));}
    let gate=sp*sa;let probability=1.0/(1.0+exp(-b));
    let ok=finite(a)&&finite(b)&&finite(dt)&&finite(sa)&&finite(gate)&&finite(probability)&&gate<=0.0&&probability>=0.0&&probability<=1.0;
    controls[t*p.value_heads*2u+h]=select(0.0,gate,ok);controls[t*p.value_heads*2u+p.value_heads+h]=select(0.0,probability,ok);status[t*p.value_heads+h]=select(0u,1u,ok);
}
// CONTROLS_END
// STEP_BEGIN
const STATE_ONLY:bool=false;
@group(0) @binding(0) var<storage,read> query:array<f32>;
@group(0) @binding(1) var<storage,read> key:array<f32>;
@group(0) @binding(2) var<storage,read> value:array<f32>;
@group(0) @binding(3) var<storage,read> controls:array<f32>;
@group(0) @binding(4) var<storage,read> old:array<f32>;
@group(0) @binding(5) var<storage,read_write> next:array<f32>;
@group(0) @binding(6) var<storage,read> z:array<f32>;
@group(0) @binding(7) var<storage,read> gamma:array<f32>;
@group(0) @binding(8) var<storage,read_write> output:array<f32>;
@group(0) @binding(9) var<storage,read_write> status:array<u32>;
@group(0) @binding(10) var<storage,read_write> closing_status:array<u32>;
@group(0) @binding(11) var<uniform> p:Params;
var<workgroup> sk:array<f32,128>;
var<workgroup> sq:array<f32,128>;
var<workgroup> kv_partial:array<f32,512>;
var<workgroup> out_partial:array<f32,512>;
var<workgroup> sums:array<f32,4>;
var<workgroup> broken:atomic<u32>;
@compute @workgroup_size(512)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) tid:u32,@builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_size) sw:u32) {
    let head=group.x;if sw!=32u || head>=p.value_heads || group.y!=0u || group.z!=0u {return;}
    let col=tid%128u;let rg=tid/128u;let qh=head%p.key_heads;
    if tid==0u {atomicStore(&broken,0u);}workgroupBarrier();
    var state:array<f32,32>;
    for(var r=0u;r<32u;r++) {let at=head*16384u+(rg*32u+r)*128u+col;state[r]=old[at];if !finite(state[r]) {atomicOr(&broken,1u);}}
    for(var t=0u;t<p.count;t++) {
        if rg==0u {sk[col]=key[t*p.key_heads*128u+qh*128u+col];if !STATE_ONLY {sq[col]=query[t*p.key_heads*128u+qh*128u+col];}}
        workgroupBarrier();
        let decay=exp(controls[t*p.value_heads*2u+head]);let beta=controls[t*p.value_heads*2u+p.value_heads+head];
        var predicted=0.0;
        for(var r=0u;r<32u;r++) {predicted=fma(state[r],sk[rg*32u+r],predicted);}
        kv_partial[tid]=predicted;workgroupBarrier();
        let combined=kv_partial[col]+kv_partial[128u+col]+kv_partial[256u+col]+kv_partial[384u+col];
        let delta=(value[t*p.value_heads*128u+head*128u+col]-decay*combined)*beta;
        if !finite(decay)||!finite(beta)||!finite(delta) {atomicOr(&broken,1u);}
        var readout=0.0;
        for(var r=0u;r<32u;r++) {state[r]=fma(decay,state[r],sk[rg*32u+r]*delta);if !finite(state[r]) {atomicOr(&broken,1u);}if !STATE_ONLY {readout=fma(state[r],sq[rg*32u+r],readout);}}
        out_partial[tid]=readout;workgroupBarrier();
        var oc=0.0;var ss=0.0;
        if rg==0u && !STATE_ONLY {oc=(out_partial[col]+out_partial[128u+col]+out_partial[256u+col]+out_partial[384u+col])*inverseSqrt(128.0);ss=oc*oc;}
        for(var distance=16u;distance>0u;distance/=2u) {ss+=subgroupShuffleXor(ss,distance);}
        if rg==0u && lane==0u {sums[col/32u]=ss;}workgroupBarrier();
        if rg==0u {
            var y=0.0;
            if !STATE_ONLY {
                let scale=inverseSqrt((sums[0]+sums[1]+sums[2]+sums[3])/128.0+p.epsilon);
                let ga=gamma[select(head*128u+col,col,p.gamma_shared!=0u)];let zz=z[t*p.value_heads*128u+head*128u+col];
                y=oc*scale*ga*(1.0/(1.0+exp(-zz)));if !finite(y)||!finite(ga)||!finite(zz) {atomicOr(&broken,1u);}
                output[t*p.value_heads*128u+head*128u+col]=y;
            }
        }
        workgroupBarrier();
        if rg==0u {let ok=select(0u,1u,atomicLoad(&broken)==0u);status[t*p.value_heads*128u+head*128u+col]=ok;closing_status[t*p.value_heads*128u+head*128u+col]=ok;}
        workgroupBarrier();
    }
    let good=atomicLoad(&broken)==0u;
    for(var r=0u;r<32u;r++) {next[head*16384u+(rg*32u+r)*128u+col]=select(0.0,state[r],good);}
    if rg==0u && !good {for(var t=0u;t<p.count;t++) {status[t*p.value_heads*128u+head*128u+col]=0u;closing_status[t*p.value_heads*128u+head*128u+col]=0u;if !STATE_ONLY {output[t*p.value_heads*128u+head*128u+col]=0.0;}}}
}
// STEP_END
