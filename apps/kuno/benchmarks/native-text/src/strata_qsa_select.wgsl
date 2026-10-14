// MIT Strata fb58e0: native_qsa_indexer.dp.cpp and qsa_select.dp.cpp.
// COMMON_BEGIN
struct Params {count:u32,start:u32,blocks:u32,stride:u32,score_stride:u32,status_stride:u32,dead_offset:u32,reserved:u32}
fn finite(x:f32)->bool{return abs(x)<=3.4028234663852886e38;}
// COMMON_END
// DEAD_BEGIN
@group(0) @binding(0) var<storage,read> candidate:array<u32>;
@group(0) @binding(1) var<storage,read> gamma:array<f32>;
@group(0) @binding(2) var<storage,read_write> pooled:array<f32>;
@group(0) @binding(3) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(4) var<uniform> p:Params;
var<workgroup> sums:array<f32,8>;
var<workgroup> valid:array<u32,8>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) tid:u32,@builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_id) warp:u32,@builtin(subgroup_size) sw:u32){
    if sw!=32u {if tid==0u{atomicOr(&fault[0],1u);}return;}
    var mean=0.0;var ss=0.0;var ok=1u;
    if tid<128u {
        let incoming=bitcast<f32>(candidate[512u+tid]);
        var sum=incoming;for(var j=1u;j<4u;j++){sum+=incoming;}
        mean=fma(0.25,sum,0.0);ss=mean*mean;
        if !finite(incoming)||!finite(mean)||!finite(gamma[tid]){ok=0u;}
    }
    for(var d=16u;d>0u;d/=2u){ss+=subgroupShuffleXor(ss,d);ok&=subgroupShuffleXor(ok,d);}
    if lane==0u{sums[warp]=ss;valid[warp]=ok;}workgroupBarrier();
    ss=0.0;ok=1u;if lane<8u{ss=sums[lane];ok=valid[lane];}
    for(var d=16u;d>0u;d/=2u){ss+=subgroupShuffleXor(ss,d);ok&=subgroupShuffleXor(ok,d);}
    let scale=inverseSqrt(ss/128.0+0.000001);
    if tid<128u {
        let y=(scale*mean)*gamma[tid];if !finite(scale)||!finite(y){ok=0u;}
        pooled[p.dead_offset+tid]=select(0.0,y,ok==1u);
    }
    for(var d=16u;d>0u;d/=2u){ok&=subgroupShuffleXor(ok,d);}
    if lane==0u{valid[warp]=ok;}workgroupBarrier();
    if tid==0u{var good=1u;for(var i=0u;i<8u;i++){good&=valid[i];}if good!=1u{atomicOr(&fault[0],1u);}}
}
// DEAD_END
// VALIDATE_BEGIN
@group(0) @binding(0) var<storage,read> scores:array<f32>;
@group(0) @binding(1) var<storage,read> status:array<u32>;
@group(0) @binding(2) var<storage,read> selected:array<u32>;
@group(0) @binding(3) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(4) var<uniform> p:Params;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) tid:u32){
    let t=g.x;let n=p.start+t+1u;let visible_blocks=n/4u+select(0u,1u,n%4u!=0u);
    for(var b=tid;b<visible_blocks;b+=256u){
        let v=scores[t*p.score_stride+b];
        if status[t*p.status_stride+b]!=1u||!finite(v)||v<0.0{atomicOr(&fault[0],1u);}
    }
}
// VALIDATE_END
// SELECT_BEGIN
@group(0) @binding(0) var<storage,read> scores:array<f32>;
@group(0) @binding(1) var<storage,read> status:array<u32>;
@group(0) @binding(2) var<storage,read_write> selected:array<u32>;
@group(0) @binding(3) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(4) var<uniform> p:Params;
var<workgroup> hist:array<atomic<u32>,8192>;
var<workgroup> warp_scan:array<u32,33>;
var<workgroup> digit:u32;
var<workgroup> above_shared:u32;
var<workgroup> enabled:u32;
fn key(v:f32)->u32{let f=v+0.0;let b=bitcast<u32>(f);if !(f==f){return 0u;}return select(b|0x80000000u,~b,(b&0x80000000u)!=0u);}
fn weight(b:u32,nbid:u32,n:u32)->u32{return select(n-nbid*4u,4u,b<nbid);}
fn scan(v:u32,lane:u32,warp:u32)->vec2<u32>{
    var x=v;for(var d=1u;d<32u;d*=2u){let y=subgroupShuffleUp(x,d);if lane>=d{x+=y;}}
    if lane==31u{warp_scan[warp]=x;}workgroupBarrier();
    if warp==0u{
        let w=warp_scan[lane];var z=w;
        for(var d=1u;d<32u;d*=2u){let y=subgroupShuffleUp(z,d);if lane>=d{z+=y;}}
        warp_scan[lane]=z-w;if lane==31u{warp_scan[32]=z;}
    }
    workgroupBarrier();let result=vec2<u32>(warp_scan[warp]+x-v,warp_scan[32]);workgroupBarrier();return result;
}
@compute @workgroup_size(1024)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) tid:u32,@builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_id) warp:u32,@builtin(subgroup_size) sw:u32){
    let t=g.x;let n=p.start+t+1u;let width=min(n,2051u);let nbid=n/4u;
    if sw!=32u {if tid==0u{atomicOr(&fault[0],1u);}return;}
    if tid==0u{enabled=select(0u,1u,atomicLoad(&fault[0])==0u);}
    if workgroupUniformLoad(&enabled)==0u{return;}
    if n<=width {for(var j=tid;j<n;j+=1024u){selected[t*p.stride+j]=j;}return;}
    let nb=nbid+1u;let per=(nb+1023u)/1024u;let b0=tid*per;let b1=min(b0+per,nb);
    var keys:array<u32,33>;for(var j=0u;j<per;j++){let b=b0+j;keys[j]=0u;if b<b1&&weight(b,nbid,n)>0u{keys[j]=key(scores[t*p.score_stride+b]);}}
    var prefix=0u;var above=0u;
    for(var round=0u;round<4u;round++){
        let shift=24u-round*8u;for(var i=lane;i<256u;i+=32u){atomicStore(&hist[warp*256u+i],0u);}workgroupBarrier();
        var high=0u;if shift<24u{high=0xffffffffu<<(shift+8u);}
        for(var b=b0;b<b1;b++){
            let w=weight(b,nbid,n);if w==0u{continue;}
            let k=keys[b-b0];
            if (k&high)==(prefix&high){atomicAdd(&hist[warp*256u+((k>>shift)&255u)],w);}
        }
        workgroupBarrier();
        if tid<256u{var sum=0u;for(var w=0u;w<32u;w++){sum+=atomicLoad(&hist[w*256u+tid]);}atomicStore(&hist[tid],sum);}
        workgroupBarrier();
        if tid==0u {
            var cum=above;var d=255u;
            loop{if d==0u||cum+atomicLoad(&hist[d])>=width{break;}cum+=atomicLoad(&hist[d]);d--;}
            digit=d;above_shared=cum;
        }
        workgroupBarrier();prefix|=digit<<shift;above=above_shared;workgroupBarrier();
    }
    let threshold=prefix;let eq_budget=width-above;var gt=0u;var eq=0u;
    for(var b=b0;b<b1;b++){
        let w=weight(b,nbid,n);if w==0u{continue;}
        let k=keys[b-b0];if k>threshold{gt+=w;}else if k==threshold{eq+=w;}
    }
    let eq_before=scan(eq,lane,warp).x;var my_eq=0u;if eq_budget>eq_before{my_eq=min(eq_budget-eq_before,eq);}
    let selected_scan=scan(gt+my_eq,lane,warp);
    if tid==0u&&selected_scan.y!=width{atomicOr(&fault[0],1u);}
    var out=selected_scan.x;var eq_left=my_eq;
    for(var b=b0;b<b1;b++){
        let w=weight(b,nbid,n);if w==0u{continue;}
        let k=keys[b-b0];
        if k>threshold{for(var c=0u;c<w;c++){selected[t*p.stride+out]=b*4u+c;out++;}}
        else if k==threshold{for(var c=0u;c<w&&eq_left>0u;c++){selected[t*p.stride+out]=b*4u+c;out++;eq_left--;}}
    }
}
// SELECT_END
