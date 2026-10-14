// MIT Strata native MoE route/plan:8 subgroup rows, real top10 and stable ties.
// Logical expert -> physical cell remap is request-private and bijective.
struct Params { count:u32,layer:u32,epoch:u32,r0:u32 }
@group(0) @binding(0) var<storage,read> logits:array<f32>;
@group(0) @binding(1) var<storage,read> remap:array<u32>;
@group(0) @binding(2) var<storage,read_write> map:array<u32>;
@group(0) @binding(3) var<storage,read_write> probability:array<f32>;
@group(0) @binding(4) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(5) var<storage,read_write> usage:array<f32>;
@group(0) @binding(6) var<uniform> p:Params;
var<workgroup> chosen:array<u32,80>;
var<workgroup> logical:array<u32,80>;
var<workgroup> group_logical:array<u32,80>;
var<workgroup> bad:atomic<u32>;
fn finite(x:f32)->bool { return abs(x)<=bitcast<f32>(0x7f7fffffu); }
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) local:u32,@builtin(subgroup_id) sg:u32,
        @builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_size) width:u32,
        @builtin(num_subgroups) nsg:u32) {
    if p.count<1u || p.count>8u || p.layer>=48u || p.epoch==0u || p.r0>3u
        || width!=32u || nsg!=8u || arrayLength(&logits)<p.count*512u
        || arrayLength(&remap)!=32768u || arrayLength(&map)!=8192u
        || arrayLength(&probability)!=8192u || arrayLength(&usage)!=32768u || arrayLength(&fault)<1u {return;}
    if local==0u {atomicStore(&bad,0u);}
    workgroupBarrier();
    let live=sg<p.count;
    var candidates:array<f32,16>;
    for(var i=0u;i<16u;i++) {
        var value=bitcast<f32>(0xff7fffffu);
        if live {value=logits[sg*512u+lane+i*32u];if !finite(value) {atomicOr(&bad,1u);}}
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
            if selected>=512u {atomicOr(&bad,2u);chosen[sg*10u+rank]=512u;}
            let cell=remap[p.layer*512u+min(selected,511u)];
            if cell>=512u {atomicOr(&bad,2u);}
            chosen[sg*10u+rank]=cell;logical[sg*10u+rank]=selected;
        }
    }
    for(var d=16u;d>0u;d/=2u) {winning_lane_sum+=subgroupShuffleXor(winning_lane_sum,d);}
    if live && lane<10u {
        let value=selected_weight*(1.0/max(winning_lane_sum,0.00006103515625));
        if !finite(value)||value<0.0||value>1.0 {atomicOr(&bad,4u);}
        probability[sg*10u+lane]=value;
    }
    workgroupBarrier();
    if local==0u {
        if atomicLoad(&bad)!=0u {atomicOr(&fault[0],128u);return;}
        var groups=0u;
        for(var route=0u;route<p.count*10u;route++) {
            let cell=chosen[route];var group=groups;
            for(var g=0u;g<groups;g++) {if map[4u+g]==cell {group=g;break;}}
            if group==groups {map[4u+group]=cell;group_logical[group]=logical[route];groups++;}
            let length=map[84u+group];
            if length>=p.count {atomicOr(&fault[0],128u);return;}
            for(var i=0u;i<length;i++) {if map[164u+group*8u+i]/10u==route/10u {atomicOr(&fault[0],128u);return;}}
            map[164u+group*8u+length]=route;map[84u+group]=length+1u;
        }
        // Count actual logical routes, not physical-cell IDs. Only one lane
        // writes this layer; ordered windows share no concurrent usage writer.
        if (p.r0&1u)!=0u {for(var route=0u;route<p.count*10u;route++) {usage[p.layer*512u+logical[route]]+=1.0;}}
        // Intel FP16 prefill traverses experts in logical ID order; group
        // members remain in token/rank order and output ranks remain unchanged.
        // Physical cells can change after adaptation, so never sort by cell.
        if (p.r0&2u)!=0u {
            for(var g=1u;g<groups;g++) {
                var at=g;
                while at>0u && group_logical[at]<group_logical[at-1u] {
                    let key=group_logical[at];group_logical[at]=group_logical[at-1u];group_logical[at-1u]=key;
                    let cell=map[4u+at];map[4u+at]=map[4u+at-1u];map[4u+at-1u]=cell;
                    let length=map[84u+at];map[84u+at]=map[84u+at-1u];map[84u+at-1u]=length;
                    for(var i=0u;i<8u;i++) {let route=map[164u+at*8u+i];map[164u+at*8u+i]=map[164u+(at-1u)*8u+i];map[164u+(at-1u)*8u+i]=route;}
                    at--;
                }
            }
        }
        map[0]=p.count;map[1]=0u;map[2]=p.count;map[3]=groups;
    }
}
