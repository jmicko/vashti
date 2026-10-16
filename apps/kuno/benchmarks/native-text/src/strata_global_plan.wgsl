// MIT Strata fb58e0 verify_kernels.dp.cpp resident_plan_kernel: one128-thread
// workgroup, entry-per-thread, original packed (entries<<16 | groups) scan.
// Vulkan substitution: authenticated physical cell IDs address the SAME role
// tables in the expert consumer instead of storing raw pointer pairs in plan.
// Last80 map words are immutable logical IDs from the preceding route dispatch;
// existing fixed8 group member planes and private roots remain unchanged.
struct Params { count:u32,layer:u32,epoch:u32,r0:u32 }
@group(0) @binding(0) var<storage,read> logits:array<f32>;
@group(0) @binding(1) var<storage,read> remap:array<u32>;
@group(0) @binding(2) var<storage,read_write> map:array<u32>;
@group(0) @binding(3) var<storage,read_write> probability:array<f32>;
@group(0) @binding(4) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(5) var<storage,read_write> usage:array<f32>;
@group(0) @binding(6) var<uniform> p:Params;
var<workgroup> ids:array<u32,128>;
var<workgroup> exclusive:array<u32,128>;
var<workgroup> sums:array<u32,4>;
var<workgroup> bad:atomic<u32>;
@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) tid:u32,@builtin(subgroup_id) sg:u32,
        @builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_size) width:u32,
        @builtin(num_subgroups) nsg:u32) {
    if p.count<1u || p.count>8u || p.layer>=48u || p.epoch==0u || p.r0>1u
        || width!=32u || nsg!=4u || arrayLength(&remap)!=32768u || arrayLength(&map)!=8192u
        || arrayLength(&usage)!=32768u || arrayLength(&fault)<1u {return;}
    let n=p.count*10u;
    if tid==0u {atomicStore(&bad,atomicLoad(&fault[0]));}
    workgroupBarrier();
    var eid=512u;var cell=512u;
    if tid<n {
        eid=map[8112u+tid];ids[tid]=eid;
        if eid<512u {cell=remap[p.layer*512u+eid];}
        if eid>=512u || cell>=512u {atomicOr(&bad,1u);}
    }
    workgroupBarrier();
    if atomicLoad(&bad)!=0u {if tid==0u {atomicOr(&fault[0],128u);}return;}
    var first=tid;var rank=0u;var count_same=0u;
    if tid<n {
        for(var j=0u;j<n;j++) {
            if ids[j]==eid {
                first=min(first,j);count_same++;
                if j<tid {
                    rank++;
                    if j/10u==tid/10u {atomicOr(&bad,2u);}
                }
            }
        }
        if count_same>p.count {atomicOr(&bad,2u);}
    }
    let is_first=tid<n && first==tid;
    let packed=select(0u,(count_same<<16u)|1u,is_first);
    var prefix=packed;
    for(var d=1u;d<32u;d*=2u) {
        let up=subgroupShuffleUp(prefix,d);if lane>=d {prefix+=up;}
    }
    exclusive[tid]=prefix-packed;
    if lane==31u {sums[sg]=prefix;}
    workgroupBarrier();
    if atomicLoad(&bad)!=0u {if tid==0u {atomicOr(&fault[0],128u);}return;}
    if tid<n {
        var total=exclusive[first];
        for(var w=0u;w<4u;w++) {if w<first/32u {total+=sums[w];}}
        let group=total&65535u;
        // Every member writes one disjoint original route slot. Only the first
        // occurrence publishes the cell/count; no serial planner or atomics.
        map[164u+group*8u+rank]=tid;
        if is_first {
            map[4u+group]=cell;map[84u+group]=count_same;
            if (p.r0&1u)!=0u {
                for(var j=0u;j<count_same;j++) {usage[p.layer*512u+eid]+=1.0;}
            }
        }
    }
    if tid==0u {
        let total=sums[0]+sums[1]+sums[2]+sums[3];
        map[0]=p.count;map[1]=0u;map[2]=p.count;map[3]=total&65535u;
    }
}
