// Cache one logit per lane; deterministic GPU top8 without serial scans.
var<workgroup> group_values:array<f32,32>;
var<workgroup> group_ids:array<u32,32>;
var<workgroup> winner:u32;
var<workgroup> selected:array<f32,RANKS>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) e:u32,@builtin(subgroup_id) sg:u32,
        @builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_size) width:u32) {
    var value=arena[p.a+e];var other=bitcast<f32>(0xff800000u);
    if EXPERTS==512u {other=arena[p.a+e+256u];}
    if !(abs(value)<=bitcast<f32>(0x7f7fffffu)) {value=bitcast<f32>(0xff800000u);}
    if !(abs(other)<=bitcast<f32>(0x7f7fffffu)) {other=bitcast<f32>(0xff800000u);}
    for(var rank=0u;rank<RANKS;rank++) {
        let local_maximum=max(value,other);
        let local_id=select(e+256u,e,value>=other);
        let maximum=subgroupMax(local_maximum);
        let id=subgroupMin(select(0xffffffffu,local_id,local_maximum==maximum));
        if lane==0u {group_values[sg]=maximum;group_ids[sg]=id;}
        workgroupBarrier();
        if sg==0u {
            let v=select(bitcast<f32>(0xff800000u),group_values[lane],lane<256u/width);
            let best=subgroupMax(v);
            let candidate=select(0xffffffffu,group_ids[lane],lane<256u/width && v==best);
            let best_id=subgroupMin(candidate);
            if lane==0u {winner=best_id;route[ROUTE_FIRST+select(rank,rank*2u,PAIRED)]=best_id;selected[rank]=best;}
        }
        workgroupBarrier();
        if e==winner {value=bitcast<f32>(0xff800000u);}
        if e+256u==winner {other=bitcast<f32>(0xff800000u);}
    }
    if e==0u {
        var sum=0.0;let maximum=selected[0];
        for(var rank=0u;rank<RANKS;rank++) {selected[rank]=exp(selected[rank]-maximum);sum+=selected[rank];}
        for(var rank=0u;rank<RANKS;rank++) {route[ROUTE_FIRST+select(RANKS+rank,rank*2u+1u,PAIRED)]=bitcast<u32>(selected[rank]/sum);}
    }
}
