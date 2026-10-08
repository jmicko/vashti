// Cache one logit per lane; deterministic GPU top8 without serial scans.
var<workgroup> group_values:array<f32,32>;
var<workgroup> group_ids:array<u32,32>;
var<workgroup> winner:u32;
var<workgroup> selected:array<f32,8>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) e:u32,@builtin(subgroup_id) sg:u32,
        @builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_size) width:u32) {
    var value=arena[p.a+e];
    for(var rank=0u;rank<8u;rank++) {
        let maximum=subgroupMax(value);
        let id=subgroupMin(select(0xffffffffu,e,value==maximum));
        if lane==0u {group_values[sg]=maximum;group_ids[sg]=id;}
        workgroupBarrier();
        if sg==0u {
            let v=select(-3.402823e38,group_values[lane],lane<256u/width);
            let best=subgroupMax(v);
            let candidate=select(0xffffffffu,group_ids[lane],lane<256u/width && v==best);
            let best_id=subgroupMin(candidate);
            if lane==0u {winner=best_id;route[rank]=best_id;selected[rank]=best;}
        }
        workgroupBarrier();
        if e==winner {value=-3.402823e38;}
    }
    if e==0u {
        var sum=0.0;let maximum=selected[0];
        for(var rank=0u;rank<8u;rank++) {selected[rank]=exp(selected[rank]-maximum);sum+=selected[rank];}
        for(var rank=0u;rank<8u;rank++) {route[8u+rank]=bitcast<u32>(selected[rank]/sum);}
    }
}
