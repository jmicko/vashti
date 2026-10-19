// MIT Strata Intel prompt router: eight independent tokens per workgroup,
// matching prefill/kernels.dp.cpp::route_kernel<16> and ceil(T/8) launch.
// CPU grouping consumes logical+physical route pairs once per layer.
// Logical expert -> physical cell remap is request-private and bijective.
// The following finalize dispatch publishes the header only after all groups.
struct Params { count:u32,layer:u32,epoch:u32,r0:u32 }
@group(0) @binding(0) var<storage,read> logits:array<f32>;
@group(0) @binding(1) var<storage,read> remap:array<u32>;
@group(0) @binding(2) var<storage,read_write> map:array<u32>;
@group(0) @binding(3) var<storage,read_write> probability:array<f32>;
@group(0) @binding(4) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(5) var<storage,read_write> usage:array<f32>;
@group(0) @binding(6) var<uniform> p:Params;
var<workgroup> bad:atomic<u32>;
fn finite(x:f32)->bool { return abs(x)<=bitcast<f32>(0x7f7fffffu); }
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) block:vec3<u32>,@builtin(local_invocation_index) local:u32,@builtin(subgroup_id) sg:u32,
        @builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_size) width:u32,
        @builtin(num_subgroups) nsg:u32) {
    if p.count<1u || p.count>PROMPT_MAX_ROWS || p.layer>=48u || p.epoch==0u || p.r0>3u
        || width!=32u || nsg!=8u || arrayLength(&logits)<p.count*512u
        || arrayLength(&remap)!=32768u || arrayLength(&map)!=PROMPT_RAW_WORDS
        || arrayLength(&probability)!=PROMPT_PROB_WORDS || arrayLength(&usage)!=32768u || arrayLength(&fault)<1u
        || block.x>=(p.count+7u)/8u || block.y!=0u || block.z!=0u {return;}
    if local==0u {atomicStore(&bad,0u);}
    workgroupBarrier();
    let token=block.x*8u+sg;let live=token<p.count;
    var candidates:array<f32,16>;
    for(var i=0u;i<16u;i++) {
        var value=bitcast<f32>(0xff7fffffu);
        if live {value=logits[token*512u+lane+i*32u];if !finite(value) {atomicOr(&bad,1u);}}
        candidates[i]=value;
    }
    var maximum=-3.4028234663852886e38;
    for(var i=0u;i<16u;i++) {maximum=max(maximum,candidates[i]);}
    for(var d=16u;d>0u;d/=2u) {maximum=max(maximum,subgroupShuffleXor(maximum,d));}
    var denominator=0.0;
    for(var i=0u;i<16u;i++) {candidates[i]=exp(candidates[i]-maximum);denominator+=candidates[i];}
    for(var d=16u;d>0u;d/=2u) {denominator+=subgroupShuffleXor(denominator,d);}
    let reciprocal=1.0/denominator;
    for(var i=0u;i<16u;i++) {candidates[i]*=reciprocal;}
    var remaining=65535u;var selected_weight=0.0;var winning_lane_sum=0.0;
    for(var rank=0u;rank<10u;rank++) {
        var value=bitcast<f32>(0xff7fffffu);var id=512u;
        for(var i=0u;i<16u;i++) {
            let candidate=lane+i*32u;
            if (remaining&(1u<<i))!=0u && (candidates[i]>value || (candidates[i]==value && candidate<id)) {
                value=candidates[i];id=candidate;
            }
        }
        let maximum=subgroupMax(value);
        let selected=subgroupMin(select(512u,id,value==maximum));
        if selected<512u && selected%32u==lane {remaining&=~(1u<<(selected/32u));winning_lane_sum+=maximum;}
        if lane==rank {selected_weight=maximum;}
        if live && lane==0u {
            if selected>=512u {atomicOr(&bad,2u);map[1540u+token*10u+rank]=512u;}
            let cell=remap[p.layer*512u+min(selected,511u)];
            if cell>=512u {atomicOr(&bad,2u);}
            map[1540u+token*10u+rank]=cell;map[PROMPT_LOGICAL+token*10u+rank]=selected;
        }
    }
    for(var d=16u;d>0u;d/=2u) {winning_lane_sum+=subgroupShuffleXor(winning_lane_sum,d);}
    if live && lane<10u {
        let value=selected_weight*(1.0/max(winning_lane_sum,0.00006103515625));
        if !finite(value)||value<0.0||value>1.0 {atomicOr(&bad,4u);}
        probability[token*10u+lane]=value;
    }
    workgroupBarrier();
    if local==0u {
        if atomicLoad(&bad)!=0u {atomicOr(&fault[0],128u);}
    }
}
