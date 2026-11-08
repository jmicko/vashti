// MIT Strata fb58e0 native_router.dp.cpp route<512>: one32x8 block per
// token, only subgroup row zero active. The selected Intel launch deliberately
// avoids the older multi-token-per-block path that hung Alchemist.
// Output logical IDs occupy the last80 words of the SAME private map root.
struct Params { count:u32,layer:u32,epoch:u32,r0:u32 }
@group(0) @binding(0) var<storage,read> logits:array<f32>;
@group(0) @binding(1) var<storage,read> remap:array<u32>;
@group(0) @binding(2) var<storage,read_write> map:array<u32>;
@group(0) @binding(3) var<storage,read_write> probability:array<f32>;
@group(0) @binding(4) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(5) var<storage,read_write> usage:array<f32>;
@group(0) @binding(6) var<uniform> p:Params;
fn finite(x:f32)->bool { return abs(x)<=bitcast<f32>(0x7f7fffffu); }
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) block:vec3<u32>,@builtin(subgroup_id) sg:u32,
        @builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_size) width:u32,
        @builtin(num_subgroups) nsg:u32) {
    if p.count<1u || p.count>8u || p.layer>=48u || p.epoch==0u || p.r0>1u
        || width!=32u || nsg!=8u || block.x>=p.count || arrayLength(&logits)<p.count*512u
        || arrayLength(&remap)!=32768u || arrayLength(&map)!=8192u
        || arrayLength(&probability)!=8192u || arrayLength(&usage)!=32768u || arrayLength(&fault)<1u {return;}
    // No workgroup barrier after this subgroup-uniform exit. All surviving
    // arithmetic and shuffle collectives belong solely to subgroup row zero.
    if sg!=0u {return;}
    var candidates:array<f32,16>;
    for(var i=0u;i<16u;i++) {
        let value=logits[block.x*512u+lane+i*32u];
        if !finite(value) {atomicOr(&fault[0],128u);}
        candidates[i]=value;
    }
    subgroupBarrier();
    var maximum=-3.4028234663852886e38;
    for(var i=0u;i<16u;i++) {maximum=max(maximum,candidates[i]);}
    for(var d=16u;d>0u;d/=2u) {maximum=max(maximum,subgroupShuffleXor(maximum,d));}
    var denominator=0.0;
    for(var i=0u;i<16u;i++) {candidates[i]=exp(candidates[i]-maximum);denominator+=candidates[i];}
    for(var d=16u;d>0u;d/=2u) {denominator+=subgroupShuffleXor(denominator,d);}
    let reciprocal=1.0/denominator;
    for(var i=0u;i<16u;i++) {candidates[i]*=reciprocal;}
    var selected_weight=0.0;var winning_lane_sum=0.0;
    for(var rank=0u;rank<10u;rank++) {
        var value=candidates[0];var id=lane;
        for(var i=1u;i<16u;i++) {
            if candidates[i]>value {value=candidates[i];id=lane+i*32u;}
        }
        for(var d=16u;d>0u;d/=2u) {
            let other=subgroupShuffleXor(value,d);let other_id=subgroupShuffleXor(id,d);
            if other>value || (other==value && other_id<id) {value=other;id=other_id;}
        }
        if id%32u==lane {
            candidates[id/32u]=bitcast<f32>(0xff800000u);
            map[8112u+block.x*10u+rank]=id;
            winning_lane_sum+=value;
        }
        if lane==rank {selected_weight=value;}
    }
    for(var d=16u;d>0u;d/=2u) {winning_lane_sum+=subgroupShuffleXor(winning_lane_sum,d);}
    if lane<10u {
        let value=selected_weight*(1.0/max(winning_lane_sum,0.00006103515625));
        if !finite(value)||value<0.0||value>1.0 {atomicOr(&fault[0],128u);}
        probability[block.x*10u+lane]=value;
    }
}
