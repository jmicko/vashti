@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) group:vec3<u32>, @builtin(subgroup_id) sg:u32,
        @builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_size) width:u32) {
    let row=group.x*(128u/width)+sg;
    let rank=group.y;
    var base=row*ROW_BYTES;
    var input=p.a;
    if ROUTED { base+=(route[rank]*ROWS)*ROW_BYTES; }
    if RANKED_INPUT { input+=rank*COLUMNS; }
    var sum=0.0;
    if row<ROWS {
        for(var c=lane;c<COLUMNS;c+=width){ sum+=packed(base,c)*arena[input+c]; }
    }
    let total=subgroupAdd(sum);
    if row<ROWS && lane==0u { arena[p.b+rank*ROWS+row]=total; }
}
