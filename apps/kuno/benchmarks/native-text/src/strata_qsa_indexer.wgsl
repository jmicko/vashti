// MIT Strata fb58e0 native_qsa_indexer.dp.cpp::append<false>.
// INDEXER_BEGIN
// STRATA_ROUND_PRODUCTS: original materialized F32 pooled values and products.
@group(0) @binding(0) var<storage,read> cache:array<u32>;
@group(0) @binding(1) var<storage,read> candidate:array<u32>;
@group(0) @binding(2) var<storage,read> gamma:array<f32>;
@group(0) @binding(3) var<storage,read_write> pooled:array<f32>;
@group(0) @binding(4) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(5) var<uniform> p:Params;
var<workgroup> tail:array<f32,384>;
var<workgroup> values:array<f32,128>;
var<workgroup> partial:array<f32,8>;
var<workgroup> broken:atomic<u32>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) d:u32,@builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_id) warp:u32,@builtin(subgroup_size) sw:u32) {
    if sw!=32u {if d==0u {atomicOr(&fault[0],1u);}return;}
    if d==0u {atomicStore(&broken,0u);}
    workgroupBarrier();
    // Only the current incomplete block needs its preceding RAW FP16 keys.
    // Cache is the authenticated pre-window prefix; candidate owns new rows.
    if d<128u {
        for(var j=0u;j<3u;j++) {
            var x=0.0;
            if j<p.start%4u {
                x=bitcast<f32>(cache[((p.start/4u)*4u+j)*CACHE_WORDS+INDEX_OFFSET+d]);
                if !finite(x) {atomicOr(&broken,1u);}
            }
            tail[j*128u+d]=x;
        }
    }
    workgroupBarrier();
    for(var t=0u;t<p.count;t++) {
        let pos=p.start+t;let slot=pos%4u;
        var incoming=0.0;
        if d<128u {
            incoming=bitcast<f32>(candidate[t*CACHE_WORDS+INDEX_OFFSET+d]);
            if !finite(incoming) {atomicOr(&broken,1u);}
            if slot<3u {tail[slot*128u+d]=incoming;}
        }
        if pos==0u || slot==3u {
            var mean=0.0;
            if d<128u {
                var sum=select(tail[d],incoming,pos==0u);
                for(var j=1u;j<4u;j++) {
                    var x=incoming;if pos!=0u && j<3u {x=tail[j*128u+d];}
                    sum+=x;
                }
                mean=fma(0.25,sum,0.0);
            }
            var ss=mean*mean;
            for(var off=16u;off>0u;off/=2u) {ss+=subgroupShuffleXor(ss,off);}
            if lane==0u {partial[warp]=ss;}
            workgroupBarrier();
            ss=0.0;if lane<8u {ss=partial[lane];}
            for(var off=16u;off>0u;off/=2u) {ss+=subgroupShuffleXor(ss,off);}
            let scale=inverseSqrt(ss/128.0+p.epsilon);
            if d<128u {
                values[d]=(scale*mean)*gamma[d];
                if !finite(mean)||!finite(scale)||!finite(gamma[d])||!finite(values[d]) {atomicOr(&broken,1u);}
            }
            workgroupBarrier();
            if d<128u {
                var y=values[d];
                if d<64u {
                    let pair=d%32u;let angle=f32((pos/4u)*4u)*pow(p.theta_scale,f32(pair));
                    let c=cos(angle);let s=sin(angle);let a=values[pair];let z=values[pair+32u];
                    y=select(fma(a,s,z*c),fma(a,c,-z*s),d<32u);
                }
                if !finite(y) {atomicOr(&broken,1u);}
                pooled[(pos/4u)*128u+d]=select(0.0,y,finite(y));
                if pos==0u {pooled[(p.capacity/4u)*128u+d]=select(0.0,y,finite(y));}
            }
        }
        workgroupBarrier();
    }
    if d==0u && atomicLoad(&broken)!=0u {atomicOr(&fault[0],1u);}
}
// INDEXER_END
// INDEXER_FLAGS_BEGIN
// Bridge the original receipt's scoped flags only after real indexer/fault work.
@group(0) @binding(0) var<storage,read_write> prepared:array<u32>;
@group(0) @binding(1) var<storage,read_write> normalized:array<u32>;
@group(0) @binding(2) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(3) var<uniform> p:Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
    let block=gid.x;let t=gid.y;
    if t>=p.count || block>=(p.start+t+1u)/4u {return;}
    let ok=select(0u,1u,atomicLoad(&fault[0])==0u);
    prepared[t*p.status_stride+block]=ok;normalized[t*p.status_stride+block]=ok;
}
// INDEXER_FLAGS_END
