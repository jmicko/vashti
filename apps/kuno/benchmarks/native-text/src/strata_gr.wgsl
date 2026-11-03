// MIT Strata fb58e0, Intel fused_gr.dp.cpp: norm_split, sliced down,
// fixed slice reduction, up/mix, and explicit writes at non-GR boundaries.
// N=2560, HC=4, LR=320, MAX_T=8. All request values are runtime INFO.
// COMMON_BEGIN
struct Params {count:u32,first:u32,apply:u32,inject:u32,eps:f32,r0:u32,r1:u32,r2:u32}
fn finite(x:f32)->bool {return abs(x)<=3.4028234663852886e38;}
fn sigmoid(x:f32)->f32 {return 1.0/(1.0+exp(-x));}
// COMMON_END
// NORM_BEGIN
@group(0) @binding(0) var<storage,read> residual:array<f32>;
@group(0) @binding(1) var<storage,read> block:array<f32>;
@group(0) @binding(2) var<storage,read> injection:array<f32>;
@group(0) @binding(3) var<storage,read> gamma:array<f32>;
@group(0) @binding(4) var<storage,read_write> normalized:array<f32>;
@group(0) @binding(5) var<storage,read_write> scales:array<f32>;
@group(0) @binding(6) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(7) var<uniform> p:Params;
var<workgroup> partial:array<f32,8>;
var<workgroup> rs:f32;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) local:u32,
        @builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_id) warp:u32,@builtin(subgroup_size) width:u32) {
    if width!=32u || p.count<1u || p.count>8u || p.apply>1u || p.inject>1u || group.x>=p.count*4u || group.y!=0u || group.z!=0u {return;}
    if arrayLength(&fault)!=1u || arrayLength(&gamma)!=10240u || arrayLength(&scales)!=64u || arrayLength(&normalized)!=81920u {return;}
    if p.first>arrayLength(&residual) || p.count*10240u>arrayLength(&residual)-p.first {return;}
    if p.apply!=0u && (arrayLength(&block)!=20480u || arrayLength(&injection)!=32u) {return;}
    let token=group.x/4u;let stream=group.x%4u;
    var gate=0.0;if p.apply!=0u {gate=2.0*sigmoid(injection[token*4u+stream]/4.0);}
    var keep:array<vec4<f32>,3>;var kept=0u;var sum=0.0;
    for(var i=local*4u;i<10240u;i+=1024u) {
        if i/2560u!=stream {continue;}
        let at=p.first+token*10240u+i;
        var r=vec4<f32>(residual[at],residual[at+1u],residual[at+2u],residual[at+3u]);
        if p.apply!=0u {
            let b=token*2560u+i%2560u;
            r=vec4<f32>(fma(block[b],gate,r.x),fma(block[b+1u],gate,r.y),fma(block[b+2u],gate,r.z),fma(block[b+3u],gate,r.w));
        }
        if !finite(r.x) || !finite(r.y) || !finite(r.z) || !finite(r.w) {atomicOr(&fault[0],1u);}
        sum+=r.x*r.x+r.y*r.y+r.z*r.z+r.w*r.w;
        keep[kept]=r*vec4<f32>(gamma[i],gamma[i+1u],gamma[i+2u],gamma[i+3u]);kept+=1u;
    }
    for(var distance=16u;distance>0u;distance/=2u) {sum+=subgroupShuffleXor(sum,distance);}
    if lane==0u {partial[warp]=sum;}
    workgroupBarrier();
    if local==0u {
        var total=0.0;for(var w=0u;w<8u;w++) {total+=partial[w];}
        rs=inverseSqrt(total/2560.0+p.eps);
        scales[token*4u+stream]=rs;scales[32u+token*4u+stream]=gate;
        if !finite(rs) {atomicOr(&fault[0],1u);}
    }
    workgroupBarrier();
    var k=0u;
    for(var i=local*4u;i<10240u;i+=1024u) {
        if i/2560u!=stream {continue;}
        let x=keep[k]*rs;k+=1u;let at=token*10240u+i;
        normalized[at]=x.x;normalized[at+1u]=x.y;normalized[at+2u]=x.z;normalized[at+3u]=x.w;
    }
}
// NORM_END
// DOT_BEGIN
fn dot8(w:vec4<u32>,first:u32)->f32 {
    var value=0.0;
    value=fma(bitcast<f32>(w.x<<16u),stage[first],value);
    value=fma(bitcast<f32>(w.x&0xffff0000u),stage[first+1u],value);
    value=fma(bitcast<f32>(w.y<<16u),stage[first+2u],value);
    value=fma(bitcast<f32>(w.y&0xffff0000u),stage[first+3u],value);
    value=fma(bitcast<f32>(w.z<<16u),stage[first+4u],value);
    value=fma(bitcast<f32>(w.z&0xffff0000u),stage[first+5u],value);
    value=fma(bitcast<f32>(w.w<<16u),stage[first+6u],value);
    value=fma(bitcast<f32>(w.w&0xffff0000u),stage[first+7u],value);
    return value;
}
// DOT_END
// SLICES_BEGIN
@group(0) @binding(0) var<storage,read> normalized:array<f32>;
@group(0) @binding(1) var<storage,read> down:array<vec4<u32>>;
@group(0) @binding(2) var<storage,read> injection:array<vec4<u32>>;
@group(0) @binding(3) var<storage,read_write> partial:array<f32>;
@group(0) @binding(4) var<uniform> p:Params;
var<workgroup> stage:array<f32,1024>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) local:u32,
        @builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_id) warp:u32,@builtin(subgroup_size) width:u32) {
    if width!=32u || p.count<1u || p.count>8u || p.inject>1u || group.x>=80u || group.y!=0u || group.z!=0u {return;}
    if arrayLength(&normalized)!=81920u || arrayLength(&down)!=409600u || arrayLength(&partial)!=207360u {return;}
    if p.inject!=0u && arrayLength(&injection)!=5120u {return;}
    let slice=group.x;let first=slice*128u;
    for(var i=local;i<p.count*128u;i+=256u) {let token=i/128u;let column=i%128u;stage[i]=normalized[token*10240u+first+column];}
    workgroupBarrier();
    let subrow=lane/4u;let sub=lane%4u;
    for(var r0=0u;r0<324u;r0+=64u) {
        let row=r0+warp*8u+subrow;
        var w0=vec4<u32>(0u);var w1=w0;var w2=w0;var w3=w0;
        if row<320u {
            let at=row*1280u+first/8u+sub;
            w0=down[at];w1=down[at+4u];w2=down[at+8u];w3=down[at+12u];
        } else if row<324u && p.inject!=0u {
            let at=(row-320u)*1280u+first/8u+sub;
            w0=injection[at];w1=injection[at+4u];w2=injection[at+8u];w3=injection[at+12u];
        }
        for(var token=0u;token<p.count;token++) {
            let at=token*128u+sub*8u;
            var sum=dot8(w0,at);sum+=dot8(w1,at+32u);sum+=dot8(w2,at+64u);sum+=dot8(w3,at+96u);
            sum+=subgroupShuffleXor(sum,1u);sum+=subgroupShuffleXor(sum,2u);
            if sub==0u && row<324u {partial[(slice*8u+token)*324u+row]=sum;}
        }
    }
}
// SLICES_END
// NORMSLICES_BEGIN
// NORM fused into SLICES: each slice workgroup recomputes its stream's RMS
// scale with NORM's exact per-thread order, subgroup reduction and warp sum,
// writes its own 128 normalized columns and the slice-0 stream scales.
@group(0) @binding(0) var<storage,read> residual:array<f32>;
@group(0) @binding(1) var<storage,read> block:array<f32>;
@group(0) @binding(2) var<storage,read> gatein:array<f32>;
@group(0) @binding(3) var<storage,read> gamma:array<f32>;
@group(0) @binding(4) var<storage,read_write> normalized:array<f32>;
@group(0) @binding(5) var<storage,read_write> scales:array<f32>;
@group(0) @binding(6) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(7) var<storage,read> down:array<vec4<u32>>;
@group(0) @binding(8) var<storage,read> injection:array<vec4<u32>>;
@group(0) @binding(9) var<storage,read_write> partial:array<f32>;
@group(0) @binding(10) var<uniform> p:Params;
var<workgroup> wsum:array<f32,8>;
var<workgroup> rs_t:array<f32,8>;
var<workgroup> gate_t:array<f32,8>;
var<workgroup> stage:array<f32,1024>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) local:u32,
        @builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_id) warp:u32,@builtin(subgroup_size) width:u32) {
    if width!=32u || p.count<1u || p.count>8u || p.apply>1u || p.inject>1u || group.x>=160u || group.y!=0u || group.z!=0u {return;}
    if arrayLength(&fault)!=1u || arrayLength(&gamma)!=10240u || arrayLength(&scales)!=64u || arrayLength(&normalized)!=81920u {return;}
    if arrayLength(&down)!=409600u || arrayLength(&partial)!=207360u {return;}
    if p.inject!=0u && arrayLength(&injection)!=5120u {return;}
    if p.first>arrayLength(&residual) || p.count*10240u>arrayLength(&residual)-p.first {return;}
    if p.apply!=0u && (arrayLength(&block)!=20480u || arrayLength(&gatein)!=32u) {return;}
    let slice=group.x%80u;let half=group.x/80u;let first=slice*128u;let stream=first/2560u;
    for(var token=0u;token<p.count;token++) {
        var gate=0.0;if p.apply!=0u {gate=2.0*sigmoid(gatein[token*4u+stream]/4.0);}
        var sum=0.0;
        for(var i=local*4u;i<10240u;i+=1024u) {
            if i/2560u!=stream {continue;}
            let at=p.first+token*10240u+i;
            var r=vec4<f32>(residual[at],residual[at+1u],residual[at+2u],residual[at+3u]);
            if p.apply!=0u {
                let b=token*2560u+i%2560u;
                r=vec4<f32>(fma(block[b],gate,r.x),fma(block[b+1u],gate,r.y),fma(block[b+2u],gate,r.z),fma(block[b+3u],gate,r.w));
            }
            if !finite(r.x) || !finite(r.y) || !finite(r.z) || !finite(r.w) {atomicOr(&fault[0],1u);}
            sum+=r.x*r.x+r.y*r.y+r.z*r.z+r.w*r.w;
        }
        for(var distance=16u;distance>0u;distance/=2u) {sum+=subgroupShuffleXor(sum,distance);}
        if lane==0u {wsum[warp]=sum;}
        workgroupBarrier();
        if local==0u {
            var total=0.0;for(var w=0u;w<8u;w++) {total+=wsum[w];}
            let rs=inverseSqrt(total/2560.0+p.eps);
            rs_t[token]=rs;gate_t[token]=gate;
            if slice%20u==0u && half==0u {scales[token*4u+stream]=rs;scales[32u+token*4u+stream]=gate;}
            if !finite(rs) {atomicOr(&fault[0],1u);}
        }
        workgroupBarrier();
    }
    for(var i=local;i<p.count*128u;i+=256u) {
        let token=i/128u;let column=first+i%128u;
        var r=residual[p.first+token*10240u+column];
        if p.apply!=0u {r=fma(block[token*2560u+column%2560u],gate_t[token],r);}
        let x=(r*gamma[column])*rs_t[token];
        stage[i]=x;if half==0u {normalized[token*10240u+column]=x;}
    }
    workgroupBarrier();
    let subrow=lane/4u;let sub=lane%4u;
    for(var r0=half*192u;r0<min(324u,half*192u+192u);r0+=64u) {
        let row=r0+warp*8u+subrow;
        var w0=vec4<u32>(0u);var w1=w0;var w2=w0;var w3=w0;
        if row<320u {
            let at=row*1280u+first/8u+sub;
            w0=down[at];w1=down[at+4u];w2=down[at+8u];w3=down[at+12u];
        } else if row<324u && p.inject!=0u {
            let at=(row-320u)*1280u+first/8u+sub;
            w0=injection[at];w1=injection[at+4u];w2=injection[at+8u];w3=injection[at+12u];
        }
        for(var token=0u;token<p.count;token++) {
            let at=token*128u+sub*8u;
            var sum=dot8(w0,at);sum+=dot8(w1,at+32u);sum+=dot8(w2,at+64u);sum+=dot8(w3,at+96u);
            sum+=subgroupShuffleXor(sum,1u);sum+=subgroupShuffleXor(sum,2u);
            if sub==0u && row<324u {partial[(slice*8u+token)*324u+row]=sum;}
        }
    }
}
// NORMSLICES_END
// REDUCE_BEGIN
@group(0) @binding(0) var<storage,read> partial:array<f32>;
@group(0) @binding(1) var<storage,read_write> down:array<f32>;
@group(0) @binding(2) var<storage,read_write> injection:array<f32>;
@group(0) @binding(3) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(4) var<uniform> p:Params;
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global:vec3<u32>) {
    if p.count<1u || p.count>8u || p.inject>1u || arrayLength(&fault)!=1u || arrayLength(&partial)!=207360u || arrayLength(&down)!=2560u || arrayLength(&injection)!=32u {return;}
    let i=global.x;if i>=p.count*324u || global.y!=0u || global.z!=0u {return;}
    let token=i/324u;let row=i%324u;
    var sum=0.0;for(var slice=0u;slice<80u;slice++) {sum+=partial[(slice*8u+token)*324u+row];}
    if !finite(sum) {atomicOr(&fault[0],1u);}
    if row<320u {down[token*320u+row]=sum;}
    else if p.inject!=0u {injection[token*4u+row-320u]=sum;}
}
// REDUCE_END
// UP_BEGIN
@group(0) @binding(0) var<storage,read> weights:array<vec4<u32>>;
@group(0) @binding(1) var<storage,read> down:array<f32>;
@group(0) @binding(2) var<storage,read> normalized:array<f32>;
@group(0) @binding(3) var<storage,read> scales:array<f32>;
@group(0) @binding(4) var<storage,read> residual:array<f32>;
@group(0) @binding(5) var<storage,read> block:array<f32>;
@group(0) @binding(6) var<storage,read_write> residual_out:array<f32>;
@group(0) @binding(7) var<storage,read_write> low:array<f32>;
@group(0) @binding(8) var<storage,read_write> mixed:array<f32>;
@group(0) @binding(9) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(10) var<uniform> p:Params;
var<workgroup> stage:array<f32,2560>;
var<workgroup> gates:array<f32,512>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) local:u32,
        @builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_id) warp:u32,@builtin(subgroup_size) width:u32) {
    if width!=32u || p.count<1u || p.count>8u || p.apply>1u || group.x>=320u || group.y!=0u || group.z!=0u {return;}
    if arrayLength(&weights)!=409600u || arrayLength(&down)!=2560u || arrayLength(&normalized)!=81920u || arrayLength(&scales)!=64u || arrayLength(&low)!=2560u || arrayLength(&mixed)!=20480u || arrayLength(&fault)!=1u {return;}
    if p.first>arrayLength(&residual) || p.count*10240u>arrayLength(&residual)-p.first {return;}
    if p.first>arrayLength(&residual_out) || p.count*10240u>arrayLength(&residual_out)-p.first {return;}
    if p.apply!=0u && arrayLength(&block)!=20480u {return;}
    for(var i=local;i<p.count*320u;i+=256u) {
        let x=(0.0+down[i])/4.0;let v=x/(1.0+exp(-x));stage[i]=v;
        // Each cell has one real global writer; other blocks stage the same
        // values locally instead of racing identical stores to LOW.
        if group.x==0u {low[i]=v;}
    }
    workgroupBarrier();
    let d0=group.x*8u;
    for(var r=warp;r<32u;r+=8u) {
        let stream=r/8u;let d=r%8u;let column=stream*2560u+d0+d;
        let wa=weights[column*40u+lane];var wb=vec4<u32>(0u);
        if lane<8u {wb=weights[column*40u+32u+lane];}
        var mine=0.0;
        for(var token=0u;token<p.count;token++) {
            var sum=dot8(wa,token*320u+lane*8u);
            if lane<8u {sum+=dot8(wb,token*320u+(32u+lane)*8u);}
            for(var distance=16u;distance>0u;distance/=2u) {sum+=subgroupShuffleXor(sum,distance);}
            if lane==token {mine=sum;}
        }
        if lane<p.count {
            let value=normalized[lane*10240u+column]*sigmoid(mine);
            gates[(lane*4u+stream)*16u+d]=value;
            if !finite(value) || !finite(mine) {atomicOr(&fault[0],1u);}
            if p.apply!=0u {
                let at=p.first+lane*10240u+column;
                let value=fma(block[lane*2560u+d0+d],scales[32u+lane*4u+stream],residual[at]);
                residual_out[at]=value;
                if !finite(value) {atomicOr(&fault[0],1u);}
            }
        }
    }
    workgroupBarrier();
    for(var i=local;i<p.count*8u;i+=256u) {
        let token=i/8u;let d=i%8u;var value=0.0;
        for(var stream=0u;stream<4u;stream++) {value+=gates[(token*4u+stream)*16u+d];}
        value/=4.0;mixed[token*2560u+d0+d]=value;
        if !finite(value) {atomicOr(&fault[0],1u);}
    }
}
// UP_END
// WRITE_BEGIN
@group(0) @binding(0) var<storage,read> residual:array<f32>;
@group(0) @binding(1) var<storage,read> block:array<f32>;
@group(0) @binding(2) var<storage,read> injection:array<f32>;
@group(0) @binding(3) var<storage,read_write> output:array<f32>;
@group(0) @binding(4) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(5) var<uniform> p:Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global:vec3<u32>) {
    if p.count<1u || p.count>8u || arrayLength(&residual)!=81920u || arrayLength(&block)!=20480u || arrayLength(&injection)!=32u || arrayLength(&output)!=81920u || arrayLength(&fault)!=1u {return;}
    let i=global.x;if i>=p.count*10240u || global.y!=0u || global.z!=0u {return;}
    let token=i/10240u;let column=i%10240u;let stream=column/2560u;
    let value=fma(block[token*2560u+column%2560u],2.0*sigmoid(injection[token*4u+stream]/4.0),residual[i]);
    output[i]=value;if !finite(value) {atomicOr(&fault[0],1u);}
}
// WRITE_END
