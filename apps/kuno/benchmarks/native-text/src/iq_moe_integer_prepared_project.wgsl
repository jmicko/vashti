// Same original-packed integer projections; planes are prepared once per
// input/stage in a distinct private owner phase, not recomputed per row cube.
requires packed_4x8_integer_dot_product;
const COLUMNS:u32=COLUMNS_VALUEu;
const ROWS:u32=ROWS_VALUEu;
const ROW_BYTES:u32=ROW_BYTES_VALUEu;
const DTYPE:u32=DTYPE_VALUEu;
const FIRST_SLOT:u32=FIRST_SLOT_VALUEu;
const SLOTS:u32=SLOTS_VALUEu;
struct Params {a:u32,b:u32,c:u32,d:u32,e:u32,f:u32,g:u32,h:u32}
BINDINGS_VALUE
var<workgroup> packed_first:array<u32,PACKED_VALUE>;
var<workgroup> packed_second:array<u32,PACKED_VALUE>;
var<workgroup> scale_first:array<f32,BLOCKS_VALUE>;
var<workgroup> scale_second:array<f32,BLOCKS_VALUE>;
GRID_DECLARATION_VALUE
fn valid_group(group:u32)->bool {
    let n=arrayLength(&map);
    if p.a>n {return false;}
    if 804u>n-p.a {return false;}
    if map[p.a]!=1u || map[p.a+1u]!=0u || map[p.a+2u]!=1u {return false;}
    let groups=map[p.a+3u];
    if groups==0u || groups>10u || group>=groups {return false;}
    let slot=map[p.a+4u+group];
    if slot<FIRST_SLOT || slot-FIRST_SLOT>=SLOTS {return false;}
    return map[p.a+84u+group]==1u && map[p.a+164u+group*8u]<10u;
}
DECODE_HELPERS_VALUE
PROJECTION_HELPERS_VALUE
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) local:u32,
    @builtin(subgroup_id) sg:u32,@builtin(subgroup_invocation_id) lane:u32,
    @builtin(subgroup_size) width:u32) {
    let row=g.x*(128u/width)+sg;var valid=valid_group(g.y);
    var slot=0u;var route=0u;
    if valid {slot=map[p.a+4u+g.y]-FIRST_SLOT;route=map[p.a+164u+g.y*8u];}
    INPUT_READY_VALUE
    var flag=p.g;var frame=p.g+32u;
    if PREPARED_DOWN_VALUE {flag+=1u+g.y;frame=p.g+1472u+g.y*360u;}
    let ready=valid && planes[flag]==1u;
    // Complete bounded caches initialized even for inactive rows/groups.
    for(var i=local;i<PACKED_VALUEu;i+=128u) {
        var first=0u;var second=0u;
        if ready {first=planes[frame+i];second=planes[frame+PACKED_VALUEu+i];}
        packed_first[i]=first;packed_second[i]=second;
    }
    for(var i=local;i<BLOCKS_VALUEu;i+=128u) {
        var d=0.0;var rd=0.0;
        if ready {d=bitcast<f32>(planes[frame+2u*PACKED_VALUEu+i]);rd=bitcast<f32>(planes[frame+2u*PACKED_VALUEu+BLOCKS_VALUEu+i]);}
        scale_first[i]=d;scale_second[i]=rd;
    }
    GRID_INITIALIZE_VALUE
    workgroupBarrier();
    EPILOGUE_VALUE
}
