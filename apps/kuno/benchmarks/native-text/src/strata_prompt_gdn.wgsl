// MIT Strata fb58e0 Intel prompt GDN. No content-dependent source generation.
// COMMON_BEGIN
struct Params {count:u32,start:u32,channels:u32,key_heads:u32,value_heads:u32,gamma_shared:u32,epsilon:f32,reserved:u32}
fn finite(x:f32)->bool {return abs(x)<=3.4028234663852886e38;}
// COMMON_END
// CONV_BEGIN
@group(0) @binding(0) var<storage,read> raw:array<f32>;
@group(0) @binding(1) var<storage,read> weights:array<f32>;
@group(0) @binding(2) var<storage,read> old:array<f32>;
@group(0) @binding(3) var<storage,read_write> h:array<f32>;
@group(0) @binding(4) var<storage,read_write> status:array<u32>;
@group(0) @binding(5) var<storage,read_write> prepared:array<u32>;
@group(0) @binding(6) var<uniform> p:Params;
fn input(t:u32,c:u32)->f32 {if t<3u {return old[c*3u+t];}return raw[(t-3u)*p.channels+c];}
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) tid:u32) {
    let c=g.x*128u+tid;let t0=g.y*64u;let t1=min(p.count,t0+64u);if c>=p.channels||t0>=p.count {return;}
    var v0=input(t0,c);var v1=input(t0+1u,c);var v2=input(t0+2u,c);
    let w0=weights[c*4u];let w1=weights[c*4u+1u];let w2=weights[c*4u+2u];let w3=weights[c*4u+3u];
    for(var t=t0;t<t1;t++) {
        let x=raw[t*p.channels+c];let s=v0*w0+v1*w1+v2*w2+x*w3;let y=s/(1.0+exp(-s));
        let good=finite(v0)&&finite(v1)&&finite(v2)&&finite(x)&&finite(s)&&finite(y);
        h[t*p.channels+c]=select(0.0,y,good);status[t*p.channels+c]=select(0u,1u,good);prepared[t*p.channels+c]=select(0u,1u,good);
        v0=v1;v1=v2;v2=x;
    }
}
// CONV_END
// HISTORY_BEGIN
@group(0) @binding(0) var<storage,read> raw:array<f32>;
@group(0) @binding(1) var<storage,read> old:array<f32>;
@group(0) @binding(2) var<storage,read_write> next:array<f32>;
@group(0) @binding(3) var<uniform> p:Params;
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    let c=id.x;if c>=p.channels {return;}
    for(var j=0u;j<3u;j++) {let at=p.count+j;var x=0.0;if at<3u {x=old[c*3u+at];}else {x=raw[(at-3u)*p.channels+c];}next[c*3u+j]=x;}
}
// HISTORY_END
// L2_BEGIN
@group(0) @binding(0) var<storage,read_write> h:array<f32>;
@group(0) @binding(1) var<storage,read_write> prepared:array<u32>;
@group(0) @binding(2) var<uniform> p:Params;
var<workgroup> parts:array<f32,4>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) col:u32,@builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_size) sw:u32) {
    let head=g.x;let t=g.y;if t>=p.count||head>=p.key_heads*2u {return;}let at=t*p.channels+head*128u+col;
    if sw!=32u {prepared[at]=0u;return;}let x=h[at];var sum=x*x;
    for(var d=16u;d>0u;d/=2u) {sum+=subgroupShuffleXor(sum,d);}if lane==0u {parts[col/32u]=sum;}workgroupBarrier();
    let ss=parts[0]+parts[1]+parts[2]+parts[3];let y=x*inverseSqrt(ss+p.epsilon);let good=finite(y)&&prepared[at]==1u;
    h[at]=select(0.0,y,good);prepared[at]=select(0u,1u,good);
}
// L2_END
// SPLIT_BEGIN
@group(0) @binding(0) var<storage,read> h:array<f32>;
@group(0) @binding(1) var<storage,read_write> query:array<f32>;
@group(0) @binding(2) var<storage,read_write> key:array<f32>;
@group(0) @binding(3) var<storage,read_write> value:array<f32>;
@group(0) @binding(4) var<uniform> p:Params;
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    let i=id.x;if i>=p.count*p.channels {return;}let t=i/p.channels;let c=i%p.channels;let k=p.key_heads*128u;
    if c<k {query[t*k+c]=h[i];}else if c<2u*k {key[t*k+c-k]=h[i];}else {value[t*p.value_heads*128u+c-2u*k]=h[i];}
}
// SPLIT_END
// RECURRENCE_BEGIN
@group(0) @binding(0) var<storage,read> h:array<f32>;
@group(0) @binding(1) var<storage,read> controls:array<f32>;
@group(0) @binding(2) var<storage,read> old:array<f32>;
@group(0) @binding(3) var<storage,read_write> next:array<f32>;
@group(0) @binding(4) var<storage,read_write> output:array<f32>;
@group(0) @binding(5) var<storage,read_write> status:array<u32>;
@group(0) @binding(6) var<uniform> p:Params;
var<workgroup> sq:array<f32,128>;
var<workgroup> sk:array<f32,128>;
var<workgroup> red:array<f32,128>;
var<workgroup> broken:atomic<u32>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) tid:u32) {
    let head=g.x/4u;let cb=g.x%4u;let rg=tid/32u;let c=tid%32u;let col=cb*32u+c;let qh=head%p.key_heads;
    if head>=p.value_heads {return;}if tid==0u {atomicStore(&broken,0u);}workgroupBarrier();
    var state:array<f32,32>;
    // Existing head-major state is retained as an explicit owner-layout bridge.
    for(var r=0u;r<32u;r++) {state[r]=old[head*16384u+(rg*32u+r)*128u+col];if !finite(state[r]) {atomicOr(&broken,1u);}}
    var nq=h[qh*128u+tid];var nk=h[p.key_heads*128u+qh*128u+tid];
    var nv=h[p.key_heads*256u+head*128u+col];var ng=controls[head];var nb=controls[p.value_heads+head];
    for(var t=0u;t<p.count;t++) {
        let cq=nq;let ck=nk;let cv=nv;let cg=ng;let cbt=nb;workgroupBarrier();sq[tid]=cq;sk[tid]=ck;workgroupBarrier();
        if t+1u<p.count {let ht=(t+1u)*p.channels;nq=h[ht+qh*128u+tid];nk=h[ht+p.key_heads*128u+qh*128u+tid];nv=h[ht+p.key_heads*256u+head*128u+col];ng=controls[(t+1u)*p.value_heads*2u+head];nb=controls[(t+1u)*p.value_heads*2u+p.value_heads+head];}
        let decay=exp(cg);var kv=0.0;
        for(var r=0u;r<32u;r++) {kv=fma(state[r],sk[rg*32u+r],kv);}red[tid]=kv;workgroupBarrier();
        let total=red[c]+red[32u+c]+red[64u+c]+red[96u+c];let delta=(cv-decay*total)*cbt;var o=0.0;
        for(var r=0u;r<32u;r++) {state[r]=fma(decay,state[r],sk[rg*32u+r]*delta);o=fma(state[r],sq[rg*32u+r],o);if !finite(state[r]) {atomicOr(&broken,1u);}}
        workgroupBarrier();red[tid]=o;workgroupBarrier();
        if rg==0u {let oc=(red[c]+red[32u+c]+red[64u+c]+red[96u+c])*inverseSqrt(128.0);let at=t*p.value_heads*128u+head*128u+col;
            let good=atomicLoad(&broken)==0u&&finite(oc)&&finite(cv)&&finite(cg)&&finite(cbt);output[at]=select(0.0,oc,good);status[at]=select(0u,1u,good);}
    }
    let good=atomicLoad(&broken)==0u;
    for(var r=0u;r<32u;r++) {next[head*16384u+(rg*32u+r)*128u+col]=select(0.0,state[r],good);}
}
// RECURRENCE_END
// NORM_BEGIN
@group(0) @binding(0) var<storage,read> raw:array<f32>;
@group(0) @binding(1) var<storage,read> z:array<f32>;
@group(0) @binding(2) var<storage,read> gamma:array<f32>;
@group(0) @binding(3) var<storage,read_write> packed:array<u32>;
@group(0) @binding(4) var<storage,read> status:array<u32>;
@group(0) @binding(5) var<storage,read_write> closing:array<u32>;
@group(0) @binding(6) var<uniform> p:Params;
var<workgroup> parts:array<f32,4>;
var<workgroup> values:array<f32,128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) col:u32,@builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_size) sw:u32) {
    let head=g.x;let t=g.y;if head>=p.value_heads||t>=p.count {return;}let at=t*p.value_heads*128u+head*128u+col;
    if sw!=32u {closing[at]=0u;return;}let oc=raw[at];var sum=oc*oc;
    for(var d=16u;d>0u;d/=2u) {sum+=subgroupShuffleXor(sum,d);}if lane==0u {parts[col/32u]=sum;}workgroupBarrier();
    let ss=parts[0]+parts[1]+parts[2]+parts[3];let ga=gamma[select(head*128u+col,col,p.gamma_shared!=0u)];let zz=z[at];
    let y=oc*inverseSqrt(ss/128.0+p.epsilon)*ga*(1.0/(1.0+exp(-zz)));values[col]=y;
    closing[at]=select(0u,1u,finite(y)&&finite(ga)&&finite(zz)&&status[at]==1u);workgroupBarrier();
    if col<64u {let bits=pack2x16float(vec2<f32>(values[col*2u],values[col*2u+1u]));let v=unpack2x16float(bits);
        let good=finite(v.x)&&finite(v.y);packed[(t*p.value_heads*128u+head*128u)/2u+col]=select(0u,bits,good);
        if !good {closing[at-col+col*2u]=0u;closing[at-col+col*2u+1u]=0u;}}
}
// NORM_END
