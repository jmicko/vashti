// Separate exactcount4 member capacity; original count1..3 template unchanged.
// Explicit alternate two-residual-Q8 DOWN arithmetic. Original packed weights
// and quantizer provenance: iq_moe_integer-NOTICE.txt. Never original64 parity.
// FROZEN_DOWN_PREPARE_BEGIN
const COLUMNS:u32=640u;
struct Params {a:u32,b:u32,c:u32,d:u32,e:u32,f:u32,g:u32,h:u32}
@group(0) @binding(0) var<storage,read> input:array<f32>;
@group(0) @binding(1) var<storage,read> status:array<u32>;
@group(0) @binding(2) var<storage,read_write> planes:array<u32>;
@group(0) @binding(3) var<uniform> p:Params;
var<workgroup> cached_input:array<f32,640>;
var<workgroup> packed_first:array<u32,160>;
var<workgroup> packed_second:array<u32,160>;
var<workgroup> scale_first:array<f32,20>;
var<workgroup> scale_second:array<f32,20>;
var<workgroup> quant_fault:atomic<u32>;
Q8_PREPARE_VALUE
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) local:u32) {
    // All40 fixed route frames clear even for count1/2. No group/member product
    // or private MAP-dependent owner indexing is needed to prepare original ACT.
    if (p.h<5u || p.h>8u) || arrayLength(&planes)!=32768u || group.x!=0u || group.y>=80u || group.z!=0u
        || p.d!=0u || p.e!=0u || p.f!=0u || p.g!=0u {return;}
    let route=group.y;let frame=64u+route*360u;
    if local==0u {planes[route]=0u;}
    for(var i=local;i<360u;i+=64u) {planes[frame+i]=0u;}
    if route==0u {
        if local>=80u {planes[local]=0u;}
        for(var i=28864u+local;i<32768u;i+=64u) {planes[i]=0u;}
    }
    var valid=route<p.h*10u && p.b>=640u;
    if p.a>arrayLength(&input) {valid=false;}
    else if 640u>arrayLength(&input)-p.a {valid=false;}
    else if valid && route>(arrayLength(&input)-p.a-640u)/p.b {valid=false;}
    if p.c>arrayLength(&status) {valid=false;}
    else if 160u>arrayLength(&status)-p.c {valid=false;}
    else if valid && status[p.c+route]!=1u {valid=false;}
    if local==0u {atomicStore(&quant_fault,0u);}
    for(var i=local;i<640u;i+=64u) {
        var x=0.0;if valid {x=input[p.a+route*p.b+i];}cached_input[i]=x;
    }
    workgroupBarrier();prepare_q8(local);workgroupBarrier();
    if valid && atomicLoad(&quant_fault)==0u {
        for(var i=local;i<160u;i+=64u) {planes[frame+i]=packed_first[i];planes[frame+160u+i]=packed_second[i];}
        if local<20u {
            planes[frame+320u+local]=bitcast<u32>(scale_first[local]);
            planes[frame+340u+local]=bitcast<u32>(scale_second[local]);
        }
        if local==0u {planes[route]=1u;}
    }
}
// FROZEN_DOWN_PREPARE_END
// FROZEN_DOWN_PROJECT_BEGIN
requires packed_4x8_integer_dot_product;
const ROW_BYTES:u32=ROW_BYTES_VALUEu;
struct Params {a:u32,b:u32,c:u32,d:u32,e:u32,f:u32,g:u32,h:u32}
@group(0) @binding(0) var<storage,read> down:array<u32>;
@group(0) @binding(1) var<storage,read> map:array<u32>;
@group(0) @binding(2) var<storage,read_write> output:array<f32>;
@group(0) @binding(3) var<storage,read_write> status:array<u32>;
@group(0) @binding(4) var<storage,read> planes:array<u32>;
@group(0) @binding(5) var<uniform> p:Params;
var<workgroup> sums:array<f32,512>;
fn valid_group(group:u32)->bool {
    if (p.h<5u || p.h>8u) || arrayLength(&planes)!=32768u || p.g<1u || p.g>80u || group>=p.g {return false;}
    if p.a>arrayLength(&map) {return false;}
    if 804u>arrayLength(&map)-p.a {return false;}
    if map[p.a]!=p.h || map[p.a+1u]!=0u || map[p.a+2u]!=p.h {return false;}
    let groups=map[p.a+3u];if groups<1u || groups>p.h*10u || group>=groups {return false;}
    if p.b>=512u || p.c<1u || p.c>512u-p.b {return false;}
    if arrayLength(&down)!=p.c*2560u*(ROW_BYTES/4u) {return false;}
    let slot=map[p.a+4u+group];if slot<p.b || slot-p.b>=p.c {return false;}
    let length=map[p.a+84u+group];if length<1u || length>p.h {return false;}
    if p.d>arrayLength(&output) || p.e<2560u {return false;}
    if 2560u>arrayLength(&output)-p.d {return false;}
    if 79u>(arrayLength(&output)-p.d-2560u)/p.e {return false;}
    if p.f>arrayLength(&status) {return false;}
    if 160u>arrayLength(&status)-p.f {return false;}
    for(var j=0u;j<length;j++) {
        let route=map[p.a+164u+group*8u+j];if route>=p.h*10u {return false;}
        if planes[route]!=1u || status[p.f+route]!=1u {return false;}
        for(var k=0u;k<j;k++) {if route/10u==map[p.a+164u+group*8u+k]/10u {return false;}}
    }
    return true;
}
DECODE_HELPERS_VALUE
PROJECTION_HELPERS_VALUE
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) local:u32) {
    // Every lane exits uniformly before any public source payload/shared write.
    if group.x>=320u || group.z!=0u || !valid_group(group.y) {return;}
    let row=group.x*8u+local/8u;let lane=local%8u;
    let length=map[p.a+84u+group.y];let slot=map[p.a+4u+group.y]-p.b;
    let offset=(slot*2560u+row)*ROW_BYTES;
    var frames=array<u32,8>();
    FRAME_MEMBERS_VALUE
    var values=array<f32,8>();
    for(var block=lane;block<20u;block+=8u) {
        let encoded=down_prepare(offset,block);
        var first=array<i32,8>();var second=array<i32,8>();
        for(var quartet=0u;quartet<8u;quartet++) {
            let q=down_packed(encoded,quartet);let at=block*8u+quartet;
            DOT_MEMBERS_VALUE
        }
        let d=down_factor(encoded);
        SCALE_MEMBERS_VALUE
    }
    // Explicit initialization includes every unused RHS column. No WG zeroing.
    for(var j=0u;j<8u;j++) {sums[local*8u+j]=values[j];}
    workgroupBarrier();
    for(var stride=4u;stride>0u;stride/=2u) {
        if lane<stride {for(var j=0u;j<8u;j++) {sums[local*8u+j]+=sums[(local+stride)*4u+j];}}
        workgroupBarrier();
    }
    if lane==0u {OUTPUT_MEMBERS_VALUE}
}
// FROZEN_DOWN_PROJECT_END
// FROZEN_DOWN_AUDIT_BEGIN
struct Params {count:u32,r0:u32,r1:u32,r2:u32,r3:u32,r4:u32,r5:u32,r6:u32}
@group(0) @binding(0) var<storage,read> planes:array<u32>;
@group(0) @binding(1) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(2) var<uniform> p:Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global:vec3<u32>) {
    if arrayLength(&fault)!=8192u {return;}
    if (p.count<5u || p.count>8u) || p.r0!=0u || p.r1!=0u || p.r2!=0u || p.r3!=0u || p.r4!=0u || p.r5!=0u || p.r6!=0u
        || arrayLength(&planes)!=32768u {if global.x==0u {atomicOr(&fault[0],8u);}return;}
    let at=global.x;if at>=32768u {return;}let word=planes[at];var valid=true;
    if at<64u {valid=word==select(0u,1u,at<p.count*10u);}
    else if at<28864u {
        let route=(at-64u)/360u;let offset=(at-64u)%360u;
        if route>=p.count*10u {valid=word==0u;}
        else if offset>=320u {let scale=bitcast<f32>(word);valid=scale>=0.0 && scale<=3.4028234663852886e38;}
    }else {valid=word==0u;}
    if !valid {atomicOr(&fault[0],8u);}
}
// FROZEN_DOWN_AUDIT_END
// FROZEN_DOWN_PROJECT_COMPACT_BEGIN
// Compaction variant: the immutable GU shard record drives the dispatch and
// the group selection. Adds the full GU owner as a read-only binding; the
// DOWN arithmetic, member order and all original payload checks are unchanged.
requires packed_4x8_integer_dot_product;
const ROW_BYTES:u32=ROW_BYTES_VALUEu;
struct Params {a:u32,b:u32,c:u32,d:u32,e:u32,f:u32,g:u32,h:u32}
@group(0) @binding(0) var<storage,read> down:array<u32>;
@group(0) @binding(1) var<storage,read> map:array<u32>;
@group(0) @binding(2) var<storage,read_write> output:array<f32>;
@group(0) @binding(3) var<storage,read_write> status:array<u32>;
@group(0) @binding(4) var<storage,read> planes:array<u32>;
@group(0) @binding(5) var<storage,read> gu_planes:array<u32>;
@group(0) @binding(6) var<uniform> p:Params;
var<workgroup> sums:array<f32,512>;
fn valid_group(group:u32)->bool {
    if (p.h<5u || p.h>8u) || arrayLength(&planes)!=32768u || arrayLength(&gu_planes)!=16384u || p.g>4u {return false;}
    if p.a>arrayLength(&map) {return false;}
    if 804u>arrayLength(&map)-p.a {return false;}
    if map[p.a]!=p.h || map[p.a+1u]!=0u || map[p.a+2u]!=p.h {return false;}
    let groups=map[p.a+3u];if groups<1u || groups>p.h*10u || group>=groups {return false;}
    if p.b>=512u || p.c<1u || p.c>512u-p.b {return false;}
    if arrayLength(&down)!=p.c*2560u*(ROW_BYTES/4u) {return false;}
    let slot=map[p.a+4u+group];if slot<p.b || slot-p.b>=p.c {return false;}
    let length=map[p.a+84u+group];if length<1u || length>p.h {return false;}
    if p.d>arrayLength(&output) || p.e<2560u {return false;}
    if 2560u>arrayLength(&output)-p.d {return false;}
    if 79u>(arrayLength(&output)-p.d-2560u)/p.e {return false;}
    if p.f>arrayLength(&status) {return false;}
    if 160u>arrayLength(&status)-p.f {return false;}
    for(var j=0u;j<length;j++) {
        let route=map[p.a+164u+group*8u+j];if route>=p.h*10u {return false;}
        if planes[route]!=1u || status[p.f+route]!=1u {return false;}
        for(var k=0u;k<j;k++) {if route/10u==map[p.a+164u+group*8u+k]/10u {return false;}}
    }
    return true;
}
DECODE_HELPERS_VALUE
PROJECTION_HELPERS_VALUE
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) local:u32) {
    // p.g is this launch's shard; the indirect dispatch already bounded the
    // workgroups to the record's count. Every lane exits together otherwise.
    if group.x>=320u || group.z!=0u || (p.h<5u || p.h>8u) || p.g>4u {return;}
    let base=12288u+88u*p.g;
    if gu_planes[base]!=1u {return;}
    let shard_count=gu_planes[base+1u];
    if group.y>=shard_count {return;}
    let routed_group=gu_planes[base+8u+group.y];
    if !valid_group(routed_group) {return;}
    let row=group.x*8u+local/8u;let lane=local%8u;
    let length=map[p.a+84u+routed_group];let slot=map[p.a+4u+routed_group]-p.b;
    let offset=(slot*2560u+row)*ROW_BYTES;
    var frames=array<u32,8>();
    FRAME_MEMBERS_VALUE
    var values=array<f32,8>();
    for(var block=lane;block<20u;block+=8u) {
        let encoded=down_prepare(offset,block);
        var first=array<i32,8>();var second=array<i32,8>();
        for(var quartet=0u;quartet<8u;quartet++) {
            let q=down_packed(encoded,quartet);let at=block*8u+quartet;
            DOT_MEMBERS_VALUE
        }
        let d=down_factor(encoded);
        SCALE_MEMBERS_VALUE
    }
    // Explicit initialization includes every unused RHS column. No WG zeroing.
    for(var j=0u;j<8u;j++) {sums[local*8u+j]=values[j];}
    workgroupBarrier();
    for(var stride=4u;stride>0u;stride/=2u) {
        if lane<stride {for(var j=0u;j<8u;j++) {sums[local*8u+j]+=sums[(local+stride)*4u+j];}}
        workgroupBarrier();
    }
    if lane==0u {OUTPUT_MEMBERS_VALUE}
}
// FROZEN_DOWN_PROJECT_COMPACT_END
