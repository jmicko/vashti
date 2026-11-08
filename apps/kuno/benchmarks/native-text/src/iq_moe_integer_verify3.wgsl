// Fixed-public Verify3 two-residual-Q8 activation arithmetic.
// Packed weights/quantizer provenance: iq_moe_integer-NOTICE.txt.
// VERIFY3_PREPARE_BEGIN
const COLUMNS:u32=COLUMNS_VALUEu;
struct Params {a:u32,b:u32,c:u32,d:u32,e:u32,f:u32,g:u32,h:u32}
PREP_BINDINGS_VALUE
var<workgroup> cached_input:array<f32,COLUMNS_VALUE>;
var<workgroup> packed_first:array<u32,PACKED_VALUE>;
var<workgroup> packed_second:array<u32,PACKED_VALUE>;
var<workgroup> scale_first:array<f32,BLOCKS_VALUE>;
var<workgroup> scale_second:array<f32,BLOCKS_VALUE>;
var<workgroup> quant_fault:atomic<u32>;
Q8_PREPARE_VALUE
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) local:u32) {
    // Count/complete-owner guards precede every access. A failed producer does
    // not reset any old header; the caller clears the whole owner once/layer.
    if p.h<1u || p.h>3u || arrayLength(&planes)!=16384u {return;}
    var valid=true;var offset=0u;var frame=0u;var flag=0u;
    FRAME_SELECT_VALUE
    if p.b>arrayLength(&input) || COLUMNS>arrayLength(&input)-p.b {valid=false;}
    if valid && (p.c<COLUMNS || offset>arrayLength(&input)-p.b-COLUMNS) {valid=false;}
    if local==0u {atomicStore(&quant_fault,0u);}
    for(var i=local;i<COLUMNS;i+=128u) {
        var value=0.0;if valid {value=input[p.b+offset+i];}
        cached_input[i]=value;
    }
    workgroupBarrier();prepare_q8(local);workgroupBarrier();
    // Invalid input leaves the cleared frame/flag unpublishable.
    if valid && atomicLoad(&quant_fault)==0u {
        for(var i=local;i<PACKED_VALUEu;i+=128u) {
            planes[frame+i]=packed_first[i];planes[frame+PACKED_VALUEu+i]=packed_second[i];
        }
        for(var i=local;i<BLOCKS_VALUEu;i+=128u) {
            planes[frame+2u*PACKED_VALUEu+i]=bitcast<u32>(scale_first[i]);
            planes[frame+2u*PACKED_VALUEu+BLOCKS_VALUEu+i]=bitcast<u32>(scale_second[i]);
        }
        if local==0u {planes[flag]=1u;}
    }
}
// VERIFY3_PREPARE_END
// VERIFY3_PROJECT_BEGIN
requires packed_4x8_integer_dot_product;
const COLUMNS:u32=COLUMNS_VALUEu;const ROWS:u32=ROWS_VALUEu;
const ROW_BYTES:u32=ROW_BYTES_VALUEu;const FIRST_SLOT:u32=FIRST_SLOT_VALUEu;
struct Params {a:u32,b:u32,c:u32,d:u32,e:u32,f:u32,g:u32,h:u32}
BINDINGS_VALUE
var<workgroup> packed_first:array<u32,THREE_PACKED_VALUE>;
var<workgroup> packed_second:array<u32,THREE_PACKED_VALUE>;
var<workgroup> scale_first:array<f32,THREE_BLOCKS_VALUE>;
var<workgroup> scale_second:array<f32,THREE_BLOCKS_VALUE>;
GRID_DECLARATION_VALUE
fn valid_group(group:u32)->bool {
    let n=arrayLength(&map);
    if p.a>n || 804u>n-p.a || p.h<1u || p.h>3u {return false;}
    if map[p.a]!=p.h || map[p.a+1u]!=0u || map[p.a+2u]!=p.h {return false;}
    let groups=map[p.a+3u];if groups<1u || groups>p.h*10u || group>=groups {return false;}
    let slot=map[p.a+4u+group];if slot<FIRST_SLOT || slot-FIRST_SLOT>=72u {return false;}
    let length=map[p.a+84u+group];if length<1u || length>p.h {return false;}
    for(var j=0u;j<length;j++) {
        let route=map[p.a+164u+group*8u+j];if route>=p.h*10u {return false;}
        for(var k=0u;k<j;k++) {
            if route/10u==map[p.a+164u+group*8u+k]/10u {return false;}
        }
    }
    return true;
}
DECODE_HELPERS_VALUE
PROJECTION_HELPERS_VALUE
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) local:u32,
    @builtin(subgroup_id) sg:u32,@builtin(subgroup_invocation_id) lane:u32,
    @builtin(subgroup_size) width:u32) {
    let row=group.x*(128u/width)+sg;
    // Full physical sizes were authenticated by the caller; no subview owners.
    var valid=(width==16u || width==32u) && arrayLength(&planes)==16384u && valid_group(group.y);
    var length=0u;var slot=0u;
    if valid {length=map[p.a+84u+group.y];slot=map[p.a+4u+group.y]-FIRST_SLOT;}
    var ready=valid;
    if valid {
        for(var j=0u;j<length;j++) {
            let route=map[p.a+164u+group.y*8u+j];
            READY_CHECK_VALUE
        }
    }
    // Three independently bounded prepared frames; inactive components are
    // initialized to exact0 and never acquire status/output publication.
    for(var i=local;i<THREE_PACKED_VALUEu;i+=128u) {
        let column=i/PACKED_VALUEu;let at=i%PACKED_VALUEu;
        var first=0u;var second=0u;
        if ready && column<length {
            let route=map[p.a+164u+group.y*8u+column];FRAME_VALUE
            first=planes[frame+at];second=planes[frame+PACKED_VALUEu+at];
        }
        packed_first[i]=first;packed_second[i]=second;
    }
    for(var i=local;i<THREE_BLOCKS_VALUEu;i+=128u) {
        let column=i/BLOCKS_VALUEu;let at=i%BLOCKS_VALUEu;var a=0.0;var b=0.0;
        if ready && column<length {
            let route=map[p.a+164u+group.y*8u+column];FRAME_VALUE
            a=bitcast<f32>(planes[frame+2u*PACKED_VALUEu+at]);
            b=bitcast<f32>(planes[frame+2u*PACKED_VALUEu+BLOCKS_VALUEu+at]);
        }
        scale_first[i]=a;scale_second[i]=b;
    }
    GRID_INITIALIZE_VALUE
    workgroupBarrier();
    EPILOGUE_VALUE
}
// VERIFY3_PROJECT_END
// VERIFY3_AUDIT_BEGIN
struct Params {count:u32,reserved0:u32,reserved1:u32,reserved2:u32}
@group(0) @binding(0) var<storage,read> planes:array<u32>;
@group(0) @binding(1) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(2) var<uniform> p:Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global:vec3<u32>) {
    if arrayLength(&fault)!=8192u {return;}
    if p.count<1u || p.count>3u || p.reserved0!=0u || p.reserved1!=0u || p.reserved2!=0u
        || arrayLength(&planes)!=16384u {if global.x==0u {atomicOr(&fault[0],8u);}return;}
    let at=global.x;if at>=16384u {return;}let word=planes[at];var valid=true;
    if at<64u {
        var wanted=0u;
        if at<p.count || (at>=3u && at<3u+p.count*10u) {wanted=1u;}
        valid=word==wanted;
    }else if at<4384u {
        let frame=(at-64u)/1440u;let offset=(at-64u)%1440u;
        if frame>=p.count {valid=word==0u;}
        else if offset>=1280u {let scale=bitcast<f32>(word);valid=scale>=0.0 && scale<=3.4028234663852886e38;}
    }else if at<15184u {
        let frame=(at-4384u)/360u;let offset=(at-4384u)%360u;
        if frame>=p.count*10u {valid=word==0u;}
        else if offset>=320u {let scale=bitcast<f32>(word);valid=scale>=0.0 && scale<=3.4028234663852886e38;}
    }else {valid=word==0u;}
    if !valid {atomicOr(&fault[0],8u);}
}
// VERIFY3_AUDIT_END
