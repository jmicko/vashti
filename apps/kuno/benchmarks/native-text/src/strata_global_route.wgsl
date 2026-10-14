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
var<workgroup> bad:atomic<u32>;
fn finite(x:f32)->bool { return abs(x)<=bitcast<f32>(0x7f7fffffu); }
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) local:u32,@builtin(subgroup_id) sg:u32,
        @builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_size) width:u32,
        @builtin(num_subgroups) nsg:u32) {
    if p.count<1u || p.count>8u || p.layer>=48u || p.epoch==0u || p.r0>1u
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
    var remaining=65535u;
    var row_scores:array<f32,10>;
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
        if selected<512u && selected%32u==lane {remaining&=~(1u<<(selected/32u));}
        if live && lane==0u {
            if selected>=512u {atomicOr(&bad,2u);chosen[sg*10u+rank]=512u;}
            let cell=remap[p.layer*512u+min(selected,511u)];
            if cell>=512u {atomicOr(&bad,2u);}
            chosen[sg*10u+rank]=cell;logical[sg*10u+rank]=selected;row_scores[rank]=maximum;
        }
    }
    if live && lane==0u {
        var sum=0.0;
        for(var rank=0u;rank<10u;rank++) {let value=exp(row_scores[rank]-row_scores[0]);row_scores[rank]=value;sum+=value;}
        if !finite(sum) || sum<=0.0 {atomicOr(&bad,4u);}
        else {for(var rank=0u;rank<10u;rank++) {probability[sg*10u+rank]=row_scores[rank]/sum;}}
    }
    workgroupBarrier();
    if local==0u {
        if atomicLoad(&bad)!=0u {atomicOr(&fault[0],128u);return;}
        var groups=0u;
        for(var route=0u;route<p.count*10u;route++) {
            let cell=chosen[route];var group=groups;
            for(var g=0u;g<groups;g++) {if map[4u+g]==cell {group=g;break;}}
            if group==groups {map[4u+group]=cell;groups++;}
            let length=map[84u+group];
            if length>=p.count {atomicOr(&fault[0],128u);return;}
            for(var i=0u;i<length;i++) {if map[164u+group*8u+i]/10u==route/10u {atomicOr(&fault[0],128u);return;}}
            map[164u+group*8u+length]=route;map[84u+group]=length+1u;
        }
        // Count actual logical routes, not physical-cell IDs. Only one lane
        // writes this layer; ordered windows share no concurrent usage writer.
        if p.r0==1u {for(var route=0u;route<p.count*10u;route++) {usage[p.layer*512u+logical[route]]+=1.0;}}
        map[0]=p.count;map[1]=0u;map[2]=p.count;map[3]=groups;
    }
}
