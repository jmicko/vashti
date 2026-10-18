// MIT Strata fb58e0 native_qsa_indexer.dp.cpp::append<false>.
// INDEXER_BEGIN
// STRATA_ROUND_PRODUCTS: original materialized F32 pooled values and products.
@group(0) @binding(0) var<storage,read_write> tail_state:array<f32>;
@group(0) @binding(1) var<storage,read> raw:array<f32>;
@group(0) @binding(2) var<storage,read> gamma:array<f32>;
@group(0) @binding(3) var<storage,read_write> pooled:array<f32>;
@group(0) @binding(4) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(5) var<storage,read_write> snapshot:array<f32>;
@group(0) @binding(6) var<storage,read_write> window:array<f32>;
@group(0) @binding(7) var<uniform> p:Params;
var<workgroup> tail:array<f32,384>;
var<workgroup> values:array<f32,128>;
var<workgroup> partial:array<f32,8>;
var<workgroup> broken:atomic<u32>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) d:u32,@builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_id) warp:u32,@builtin(subgroup_size) sw:u32) {
    if sw!=32u {if d==0u {atomicOr(&fault[0],1u);}return;}
    if d==0u {atomicStore(&broken,0u);}
    workgroupBarrier();
    // Save the actual prewindow tail, including inactive slots, before updates.
    if d<128u {
        for(var j=0u;j<3u;j++) {
            let x=tail_state[j*128u+d];
            snapshot[j*128u+d]=x;tail[j*128u+d]=x;
            if !finite(x) {atomicOr(&broken,1u);}
        }
    }
    workgroupBarrier();
    for(var t=0u;t<p.count;t++) {
        let pos=p.start+t;let slot=pos%4u;
        var incoming=0.0;
        if d<128u {
            let original=raw[t*128u+d];
            incoming=unpack2x16float(pack2x16float(vec2<f32>(original,0.0))).x;
            window[t*128u+d]=original;
            if !finite(original) {atomicOr(&broken,1u);}
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
    if d<128u {for(var j=0u;j<3u;j++) {tail_state[j*128u+d]=tail[j*128u+d];}}
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

// PROMPT_TAIL_BEGIN
// Parallel append_tail from native_qsa_indexer: retain only the final cell of
// each slot. Earlier prompt pooling reads the old tail before this dispatch.
@group(0) @binding(0) var<storage,read> raw:array<f32>;
@group(0) @binding(1) var<storage,read_write> tail_state:array<f32>;
@group(0) @binding(2) var<storage,read_write> snapshot:array<f32>;
@group(0) @binding(3) var<storage,read_write> window:array<f32>;
@group(0) @binding(4) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(5) var<uniform> p:Params;
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
    let i=gid.x;if i>=384u {return;}
    let old=tail_state[i];snapshot[i]=old;if !finite(old){atomicOr(&fault[0],1u);}
    let slot=i/128u;let d=i%128u;let last=p.start+p.count-1u;
    if last>=slot {
        let cell=last-(last-slot)%4u;
        if cell>=p.start {
            let x=raw[(cell-p.start)*128u+d];
            let y=unpack2x16float(pack2x16float(vec2<f32>(x,0.0))).x;
            if !finite(x)||!finite(y){atomicOr(&fault[0],1u);}
            tail_state[i]=select(0.0,y,finite(y));
        }
    }
    // Short prompt transactions may also use MAX8 checkpoint/repair consumers.
    // Every slot is written once; a large prompt has no MAX8 restore permit.
    for(var t=slot;t<8u;t+=3u) {
        if p.count<=8u && t<p.count {
            let x=raw[t*128u+d];
            window[t*128u+d]=x;
        }else{window[t*128u+d]=0.0;}
    }
}
// PROMPT_TAIL_END
