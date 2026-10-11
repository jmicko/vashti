// Two-residual-Q8 arithmetic, NOT original64 F32 parity. Original packing,
// quantizer and MIT notices: iq_moe_integer-NOTICE.txt.
// FROZEN_GU_PREPARE_BEGIN
const COLUMNS:u32=2560u;
struct Params {a:u32,b:u32,c:u32,d:u32,e:u32,f:u32,g:u32,h:u32}
@group(0) @binding(0) var<storage,read> input:array<f32>;
@group(0) @binding(1) var<storage,read_write> planes:array<u32>;
@group(0) @binding(2) var<uniform> p:Params;
var<workgroup> cached_input:array<f32,2560>;
var<workgroup> packed_first:array<u32,640>;
var<workgroup> packed_second:array<u32,640>;
var<workgroup> scale_first:array<f32,80>;
var<workgroup> scale_second:array<f32,80>;
var<workgroup> quant_fault:atomic<u32>;
Q8_PREPARE_VALUE
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) local:u32) {
    // Every count/owner/grid predicate is uniform and precedes any access.
    if p.h<1u || p.h>3u || arrayLength(&planes)!=16384u || group.x!=0u || group.y>=3u || group.z!=0u {return;}
    let token=group.y;let frame=64u+token*1440u;
    // Disjoint frame stores and a single header/tail writer: no cross-WG race.
    if local==0u {planes[token]=0u;}
    for(var i=local;i<1440u;i+=128u) {planes[frame+i]=0u;}
    if token==0u {
        if local>=3u && local<64u {planes[local]=0u;}
        for(var i=4384u+local;i<16384u;i+=128u) {planes[i]=0u;}
    }
    var valid=token<p.h && p.c>=2560u;
    if p.b>arrayLength(&input) {valid=false;}
    else if 2560u>arrayLength(&input)-p.b {valid=false;}
    else if valid && token>(arrayLength(&input)-p.b-2560u)/p.c {valid=false;}
    if local==0u {atomicStore(&quant_fault,0u);}
    for(var i=local;i<2560u;i+=128u) {
        var x=0.0;if valid {x=input[p.b+token*p.c+i];}cached_input[i]=x;
    }
    workgroupBarrier();prepare_q8(local);workgroupBarrier();
    if valid && atomicLoad(&quant_fault)==0u {
        for(var i=local;i<640u;i+=128u) {planes[frame+i]=packed_first[i];planes[frame+640u+i]=packed_second[i];}
        for(var i=local;i<80u;i+=128u) {
            planes[frame+1280u+i]=bitcast<u32>(scale_first[i]);
            planes[frame+1360u+i]=bitcast<u32>(scale_second[i]);
        }
        if local==0u {planes[token]=1u;}
    }
}
// FROZEN_GU_PREPARE_END
// FROZEN_GU_PROJECT_BEGIN
requires packed_4x8_integer_dot_product;
const ROW_BYTES:u32=ROW_BYTES_VALUEu;
struct Params {a:u32,b:u32,c:u32,d:u32,e:u32,f:u32,g:u32,h:u32}
@group(0) @binding(0) var<storage,read> gate:array<u32>;
@group(0) @binding(1) var<storage,read> up:array<u32>;
@group(0) @binding(2) var<storage,read> map:array<u32>;
@group(0) @binding(3) var<storage,read_write> output:array<f32>;
@group(0) @binding(4) var<storage,read_write> status:array<u32>;
@group(0) @binding(5) var<storage,read> planes:array<u32>;
@group(0) @binding(6) var<uniform> p:Params;
var<workgroup> gate_sums:array<f32,192>;
var<workgroup> up_sums:array<f32,192>;
GRID_DECLARATION_VALUE
fn valid_group(group:u32)->bool {
    if p.h<1u || p.h>3u || arrayLength(&planes)!=16384u || p.g!=0u {return false;}
    if p.a>arrayLength(&map) {return false;}
    if 804u>arrayLength(&map)-p.a {return false;}
    if map[p.a]!=p.h || map[p.a+1u]!=0u || map[p.a+2u]!=p.h {return false;}
    let groups=map[p.a+3u];if groups<1u || groups>p.h*10u || group>=groups {return false;}
    if p.b>=512u || p.c<1u || p.c>512u-p.b {return false;}
    if arrayLength(&gate)!=p.c*640u*(ROW_BYTES/4u) || arrayLength(&up)!=arrayLength(&gate) {return false;}
    let slot=map[p.a+4u+group];if slot<p.b || slot-p.b>=p.c {return false;}
    let length=map[p.a+84u+group];if length<1u || length>p.h {return false;}
    if p.d>arrayLength(&output) || p.e<640u {return false;}
    if 640u>arrayLength(&output)-p.d {return false;}
    if 79u>(arrayLength(&output)-p.d-640u)/p.e {return false;}
    if p.f>arrayLength(&status) {return false;}
    if 80u>arrayLength(&status)-p.f {return false;}
    for(var j=0u;j<length;j++) {
        let route=map[p.a+164u+group*8u+j];if route>=p.h*10u {return false;}
        if planes[route/10u]!=1u {return false;}
        for(var k=0u;k<j;k++) {if route/10u==map[p.a+164u+group*8u+k]/10u {return false;}}
    }
    return true;
}
DECODE_HELPERS_VALUE
PROJECTION_HELPERS_VALUE
fn silu(value:f32)->f32 {
    var sigmoid=0.0;if value>=0.0 {sigmoid=1.0/(1.0+exp(-value));}
    else {let exponential=exp(value);sigmoid=exponential/(1.0+exponential);}
    return value*sigmoid;
}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) local:u32) {
    // Every lane exits together before codebook writes, source reads or fences.
    if group.x>=80u || group.z!=0u || !valid_group(group.y) {return;}
    let row=group.x*8u+local/8u;let lane=local%8u;
    let length=map[p.a+84u+group.y];let slot=map[p.a+4u+group.y]-p.b;
    let offset=(slot*640u+row)*ROW_BYTES;
    var frames=vec3<u32>(0u);
    FRAME_MEMBERS_VALUE
    GRID_INITIALIZE_VALUE
    workgroupBarrier();
    var gates=vec3<f32>(0.0);var ups=vec3<f32>(0.0);
    for(var block=lane;block<80u;block+=8u) {
        for(var half=0u;half<HALVES_VALUEu;half++) {
            var gf=vec3<i32>(0);var gs=vec3<i32>(0);var uf=vec3<i32>(0);var us=vec3<i32>(0);
            for(var quartet=half*QUARTETS_VALUEu;quartet<(half+1u)*QUARTETS_VALUEu;quartet++) {
                let gq=gate_packed(offset,block,quartet);let uq=up_packed(offset,block,quartet);
                let at=block*8u+quartet;
                DOT_MEMBERS_VALUE
            }
            let gd=gate_factor(offset,block,half);let ud=up_factor(offset,block,half);
            SCALE_MEMBERS_VALUE
        }
    }
    for(var j=0u;j<3u;j++) {gate_sums[local*3u+j]=gates[j];up_sums[local*3u+j]=ups[j];}
    workgroupBarrier();
    // Fixed8-lane tree4/2/1. Separate rows never share arithmetic cells.
    for(var stride=4u;stride>0u;stride/=2u) {
        if lane<stride {
            for(var j=0u;j<3u;j++) {gate_sums[local*3u+j]+=gate_sums[(local+stride)*3u+j];up_sums[local*3u+j]+=up_sums[(local+stride)*3u+j];}
        }
        workgroupBarrier();
    }
    if lane==0u {
        OUTPUT_MEMBERS_VALUE
    }
}
// FROZEN_GU_PROJECT_END
// FROZEN_GU_AUDIT_BEGIN
struct Params {count:u32,r0:u32,r1:u32,r2:u32,r3:u32,r4:u32,r5:u32,r6:u32}
@group(0) @binding(0) var<storage,read> planes:array<u32>;
@group(0) @binding(1) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(2) var<uniform> p:Params;
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global:vec3<u32>) {
    if arrayLength(&fault)!=8192u {return;}
    if p.count<1u || p.count>3u || p.r0!=0u || p.r1!=0u || p.r2!=0u || p.r3!=0u || p.r4!=0u || p.r5!=0u || p.r6!=0u
        || arrayLength(&planes)!=16384u {if global.x==0u {atomicOr(&fault[0],8u);}return;}
    let at=global.x;if at>=16384u {return;}let word=planes[at];var valid=true;
    if at<64u {valid=word==select(0u,1u,at<p.count);}
    else if at<4384u {
        let frame=(at-64u)/1440u;let offset=(at-64u)%1440u;
        if frame>=p.count {valid=word==0u;}
        else if offset>=1280u {let scale=bitcast<f32>(word);valid=scale>=0.0 && scale<=3.4028234663852886e38;}
    }else {valid=word==0u;}
    if !valid {atomicOr(&fault[0],8u);}
}
// FROZEN_GU_AUDIT_END
