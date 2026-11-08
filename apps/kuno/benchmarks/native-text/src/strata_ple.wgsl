// MIT Strata native_ple_postops/native_bf16 fb58e0, fixed real Qwen PLE.
// COMMON_BEGIN
struct Params {count:u32,ib:u32,is:u32,ob:u32,os:u32,ss:u32,rows:u32,epsilon:f32,r0:u32,r1:u32,r2:u32,r3:u32}
fn finite(x:f32)->bool {return abs(x)<=3.4028234663852886e38;}
// COMMON_END
// NORM_BEGIN
@group(0) @binding(0) var<storage,read> input:array<f32>;
@group(0) @binding(1) var<storage,read> gamma:array<f32>;
@group(0) @binding(2) var<storage,read_write> output:array<f32>;
@group(0) @binding(3) var<storage,read_write> status:array<u32>;
@group(0) @binding(4) var<uniform> p:Params;
var<workgroup> sums:array<f32,32>;
var<workgroup> valid:array<u32,32>;
@compute @workgroup_size(1024)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) tid:u32,
 @builtin(subgroup_id) sg:u32,@builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_size) sw:u32) {
    let t=g.y;let h=g.x;if sw!=32u || t>=p.count || h>=4u {return;}
    let base=p.ib+t*p.is+h*2560u;var sum=0.0;var good=1u;
    for(var col=tid;col<2560u;col+=1024u) {let x=input[base+col];sum+=x*x;if !finite(x)||!finite(gamma[h*2560u+col]) {good=0u;}}
    for(var d=16u;d>0u;d/=2u) {sum+=subgroupShuffleXor(sum,d);good&=subgroupShuffleXor(good,d);}
    if lane==0u {sums[sg]=sum;valid[sg]=good;}workgroupBarrier();
    sum=sums[lane];good=valid[lane];for(var d=16u;d>0u;d/=2u) {sum+=subgroupShuffleXor(sum,d);good&=subgroupShuffleXor(good,d);}
    let scale=inverseSqrt(sum/2560.0+p.epsilon);if !finite(scale)||scale<=0.0 {good=0u;}
    for(var col=tid;col<2560u;col+=1024u) {let y=(scale*input[base+col])*gamma[h*2560u+col];output[p.ob+t*p.os+h*2560u+col]=select(0.0,y,good==1u&&finite(y));if !finite(y){good=0u;}}
    for(var d=16u;d>0u;d/=2u) {good&=subgroupShuffleXor(good,d);}if lane==0u {valid[sg]=good;}workgroupBarrier();
    if tid==0u {var ok=1u;for(var i=0u;i<32u;i++){ok&=valid[i];}status[t*p.ss+h]=ok;}
}
// NORM_END
// GATE_BEGIN
@group(0) @binding(0) var<storage,read> key:array<f32>;
@group(0) @binding(1) var<storage,read> query:array<f32>;
@group(0) @binding(2) var<storage,read_write> gates:array<f32>;
@group(0) @binding(3) var<storage,read_write> status:array<u32>;
@group(0) @binding(4) var<uniform> p:Params;
var<workgroup> sums:array<f32,16>;
var<workgroup> valid:array<u32,16>;
@compute @workgroup_size(512)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) tid:u32,
 @builtin(subgroup_id) sg:u32,@builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_size) sw:u32) {
    let h=g.x;let t=g.y;if sw!=32u || h>=4u || t>=p.count {return;}var sum=0.0;var good=1u;
    for(var j=0u;j<8u;j++) {let col=tid+j*512u;var product=0.0;if col<2560u {let at=t*10240u+h*2560u+col;product=key[at]*query[at];if !finite(product){good=0u;}}sum+=product;}
    for(var d=16u;d>0u;d/=2u){sum+=subgroupShuffleXor(sum,d);good&=subgroupShuffleXor(good,d);}
    if lane==0u {sums[sg]=sum;valid[sg]=good;}workgroupBarrier();sum=0.0;good=1u;
    if lane<16u {sum=sums[lane];good=valid[lane];}
    for(var d=16u;d>0u;d/=2u){sum+=subgroupShuffleXor(sum,d);good&=subgroupShuffleXor(good,d);}
    if tid==0u {let s=fma(p.epsilon,sum,0.0);let sign=select(select(0.0,1.0,s>0.0),-1.0,s<0.0);
        let y=1.0/(1.0+exp(-sign*sqrt(max(abs(s),0.000001))));
        gates[t*4u+h]=select(0.0,y,good==1u&&finite(y));status[t*4u+h]=select(0u,1u,good==1u&&finite(y));}
}
// GATE_END
// BROADCAST_BEGIN
@group(0) @binding(0) var<storage,read> value:array<f32>;
@group(0) @binding(1) var<storage,read> gate:array<f32>;
@group(0) @binding(2) var<storage,read_write> output:array<f32>;
@group(0) @binding(3) var<storage,read_write> status:array<u32>;
@group(0) @binding(4) var<uniform> p:Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {let c=id.x;let t=id.y;if c>=10240u||t>=p.count {return;}let y=value[t*2560u+c%2560u]*gate[t*4u+c/2560u];output[t*10240u+c]=select(0.0,y,finite(y));status[t*10240u+c]=select(0u,1u,finite(y));}
// BROADCAST_END
// CONV_BEGIN
@group(0) @binding(0) var<storage,read> history:array<f32>;
@group(0) @binding(1) var<storage,read> normalized:array<f32>;
@group(0) @binding(2) var<storage,read> weight:array<f32>;
@group(0) @binding(3) var<storage,read> hidden:array<f32>;
@group(0) @binding(4) var<storage,read> gated:array<f32>;
@group(0) @binding(5) var<storage,read_write> conv:array<f32>;
@group(0) @binding(6) var<storage,read_write> result:array<f32>;
@group(0) @binding(7) var<storage,read_write> conv_status:array<u32>;
@group(0) @binding(8) var<storage,read_write> result_status:array<u32>;
@group(0) @binding(9) var<uniform> p:Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    let c=id.x;let t=id.y;if c>=10240u||t>=p.count {return;}var sum=0.0;var ok=true;
    for(var k=0u;k<4u;k++) {let source=i32(t)-9+i32(k)*3;var x=0.0;
        if source>=0 {x=normalized[u32(source)*10240u+c];}else{x=history[c*9u+u32(9+source)];}
        let w=unpack2x16float(pack2x16float(vec2<f32>(weight[c*4u+k],0.0))).x;let term=x*w;
        if k==0u {sum=term;}else{sum+=term;}ok=ok&&finite(x)&&finite(w)&&finite(term);
    }
    let activation=sum/(1.0+exp(-sum));let y=(hidden[p.ib+t*p.is+c]+gated[t*10240u+c])+activation;
    conv[t*10240u+c]=select(0.0,activation,ok&&finite(activation));result[p.ob+t*p.os+c]=select(0.0,y,ok&&finite(y));
    conv_status[t*10240u+c]=select(0u,1u,ok&&finite(activation));result_status[t*10240u+c]=select(0u,1u,ok&&finite(y));
}
// CONV_END
// HISTORY_BEGIN
@group(0) @binding(0) var<storage,read> history:array<f32>;
@group(0) @binding(1) var<storage,read> normalized:array<f32>;
@group(0) @binding(2) var<storage,read_write> next:array<f32>;
@group(0) @binding(3) var<storage,read_write> status:array<u32>;
@group(0) @binding(4) var<uniform> p:Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {let c=id.x;if c>=10240u {return;}var ok=true;
    for(var r=0u;r<9u;r++) {let source=i32(p.count)-9+i32(r);var x=0.0;if source>=0 {x=normalized[u32(source)*10240u+c];}else{x=history[c*9u+p.count+r];}
        next[c*9u+r]=select(0.0,x,finite(x));ok=ok&&finite(x);}
    status[c]=select(0u,1u,ok);
}
// HISTORY_END
// PROJECTION_BEGIN
@group(0) @binding(0) var<storage,read> weight:array<u32>;
@group(0) @binding(1) var<storage,read> input:array<f32>;
@group(0) @binding(2) var<storage,read_write> output:array<f32>;
@group(0) @binding(3) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(4) var<uniform> p:Params;
var<workgroup> partial:array<f32,256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) tid:u32,
 @builtin(subgroup_id) sg:u32,@builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_size) sw:u32) {
    if sw!=32u||g.x>=p.rows {return;}
    if tid<32u {for(var t=0u;t<p.count;t++){partial[t*32u+tid]=0.0;}}workgroupBarrier();var acc:array<f32,8>;
    for(var pair=tid;pair<1280u;pair+=256u) {let w=weight[g.x*1280u+pair];let w0=bitcast<f32>((w&65535u)<<16u);let w1=bitcast<f32>(w&0xffff0000u);
        for(var t=0u;t<p.count;t++) {let at=p.ib+t*p.is+pair*2u;acc[t]=fma(w0,input[at],acc[t]);acc[t]=fma(w1,input[at+1u],acc[t]);}}
    for(var t=0u;t<p.count;t++){var sum=acc[t];for(var d=16u;d>0u;d/=2u){sum+=subgroupShuffleXor(sum,d);}if lane==0u {partial[t*32u+sg]=sum;}}
    workgroupBarrier();if sg==0u {for(var t=0u;t<p.count;t++){var sum=partial[t*32u+lane];for(var d=16u;d>0u;d/=2u){sum+=subgroupShuffleXor(sum,d);}
        if lane==0u {output[p.ob+t*p.os+g.x]=sum;if !finite(sum){atomicOr(&fault[0],1u);}}}}
}
// PROJECTION_END
