// Original packed IQ weights and unquantized F32 activations. Each physical
// lane owns the same low/high logical64 accumulators as fast::dot. The only
// codebook storage is initialized workgroup memory; no Function array copies.
const COLUMNS:u32=COLUMNS_VALUEu;
const ROWS:u32=ROWS_VALUEu;
const ROW_BYTES:u32=ROW_BYTES_VALUEu;
const FIRST_SLOT:u32=FIRST_SLOT_VALUEu;
const SLOTS:u32=SLOTS_VALUEu;
struct Params {a:u32,b:u32,c:u32,d:u32,e:u32,f:u32,g:u32,h:u32}
BINDINGS_VALUE
var<workgroup> cached_input:array<f32,COLUMNS_VALUE>;
// Separate gate/up cells prevent a subgroup starting the second projection
// from overwriting another subgroup's still-live stride16 read. The original
// arithmetic and single barrier inside each reduction remain unchanged.
var<workgroup> reduction_cells:array<f32,REDUCTION_CELLS_VALUE>;
var<workgroup> input_fault:atomic<u32>;
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
fn binary16(bits:u32)->f32 {
    let sign=(bits&32768u)<<16u;
    let exponent=(bits>>10u)&31u;
    var mantissa=bits&1023u;
    var result=sign;
    if exponent==0u {
        if mantissa!=0u {
            var adjusted=113u;
            while (mantissa&1024u)==0u {mantissa<<=1u;adjusted-=1u;}
            result=sign|(adjusted<<23u)|((mantissa&1023u)<<13u);
        }
    } else if exponent==31u {
        result=sign|2139095040u|(mantissa<<13u);
    } else {
        result=sign|((exponent+112u)<<23u)|(mantissa<<13u);
    }
    return bitcast<f32>(result);
}
fn signed_byte(value:u32)->i32 {
    var result=i32(value);
    if value>=128u {result-=256;}
    return result;
}
fn iq4_value(code:u32)->f32 {
    var word=0xbfad9881u;
    if code>=4u {word=0xf6eaddcfu;}
    if code>=8u {word=0x26190d01u;}
    if code>=12u {word=0x71594535u;}
    return f32(signed_byte((word>>((code&3u)*8u))&255u));
}
fn reduce_original64(low:f32,high:f32,row:u32,lane:u32,sub_lane:u32,width:u32,first:u32)->f32 {
    var value=low+high;
    reduction_cells[first+row*32u+lane]=value;
    workgroupBarrier();
    if width==16u {
        if lane<16u {value+=reduction_cells[first+row*32u+lane+16u];}
    }
    var stride=width/2u;
    while stride>0u {
        let other=subgroupShuffleDown(value,stride);
        if sub_lane<stride {value+=other;}
        stride/=2u;
    }
    return value;
}
PROJECTION_HELPERS_VALUE

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) local:u32,
    @builtin(subgroup_id) sg:u32,@builtin(subgroup_invocation_id) sub_lane:u32,
    @builtin(subgroup_size) width:u32) {
    // Caller must admit only16/32 subgroups. Defensive numbering also avoids
    // out-of-range scratch writes if a rejected width reaches this shader.
    let partitions=max(1u,32u/width);
    let local_row=sg/partitions;
    let lane=((sg%partitions)*width+sub_lane)%32u;
    let row=g.x*4u+local_row;
    var valid=valid_group(g.y) && (width==16u || width==32u);
    var slot=0u;var route=0u;
    if valid {slot=map[p.a+4u+g.y]-FIRST_SLOT;route=map[p.a+164u+g.y*8u];}
    INPUT_READY_VALUE
    if local==0u {atomicStore(&input_fault,0u);}
    for(var i=local;i<COLUMNS;i+=128u) {
        var x=0.0;if valid {x=input[p.b+INPUT_OFFSET_VALUE+i];}
        cached_input[i]=x;
    }
    GRID_INITIALIZE_VALUE
    workgroupBarrier();
    // Initialize the fault word before any concurrent OR. All original finite
    // F32 values, including MAX and signed/subnormal zeros, stay unchanged.
    for(var i=local;i<COLUMNS;i+=128u) {
        if !(abs(cached_input[i])<=3.4028234663852886e38) {atomicOr(&input_fault,1u);}
    }
    workgroupBarrier();
    let ready=valid && atomicLoad(&input_fault)==0u && row<ROWS;
    EPILOGUE_VALUE
}
