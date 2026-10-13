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
// FROZEN_GU_PROJECT_BODY_BEGIN
    var gates=vec3<f32>(0.0);var ups=vec3<f32>(0.0);
    for(var block=lane;block<80u;block+=8u) {
        let gprepared=gate_prepare(offset,block);let uprepared=up_prepare(offset,block);
        for(var half=0u;half<HALVES_VALUEu;half++) {
            var gf=vec3<i32>(0);var gs=vec3<i32>(0);var uf=vec3<i32>(0);var us=vec3<i32>(0);
            for(var quartet=half*QUARTETS_VALUEu;quartet<(half+1u)*QUARTETS_VALUEu;quartet++) {
                let gq=gate_packed(gprepared,quartet);let uq=up_packed(uprepared,quartet);
                let at=block*8u+quartet;
                DOT_MEMBERS_VALUE
            }
            let gd=gate_factor(gprepared,half);let ud=up_factor(uprepared,half);
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
// FROZEN_GU_PROJECT_BODY_END
}
// FROZEN_GU_PROJECT_END
// FROZEN_GU_PROJECT_STRIDE4_MAIN_BEGIN
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) local:u32) {
    // The bounded envelope and every candidate predicate are uniform. A shard
    // with no valid assigned group exits before codebook writes or fences.
    if group.x>=80u || group.y>=4u || group.z!=0u || p.h<1u || p.h>3u || p.g<1u || p.g>80u {return;}
    var any_valid=false;
    for(var routed_group=group.y;routed_group<p.g;routed_group+=4u) {
        if valid_group(routed_group) {any_valid=true;}
    }
    if !any_valid {return;}
    let row=group.x*8u+local/8u;let lane=local%8u;
    GRID_INITIALIZE_VALUE
    workgroupBarrier();
    for(var routed_group=group.y;routed_group<p.g;routed_group+=4u) {
        if valid_group(routed_group) {
            let length=map[p.a+84u+routed_group];let slot=map[p.a+4u+routed_group]-p.b;
            let offset=(slot*640u+row)*ROW_BYTES;
            var frames=vec3<u32>(0u);
            FRAME_MEMBERS_VALUE
            // Each group's scalar/vector accumulators and every reduction
            // cell reset in the shared body; arithmetic order is unchanged.
            PROJECT_BODY_VALUE
            // All lanes finish reading their reduction before the next group
            // can overwrite any shared cell, including unused member cells.
            workgroupBarrier();
        }
    }
}
// FROZEN_GU_PROJECT_STRIDE4_MAIN_END
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

// FROZEN_GU_PREPARE_COMPACT_BEGIN
// Compaction variant: identical frame preparation plus five disjoint shard
// records in the zeroed owning tail. Token0 lanes0..4 derive each record from
// the immutable MAP using the original partition ranges; the records drive
// indirect GU/DOWN dispatch and are audited and erased after all consumers.
const COLUMNS:u32=2560u;
struct Params {a:u32,b:u32,c:u32,d:u32,e:u32,f:u32,g:u32,h:u32}
@group(0) @binding(0) var<storage,read> input:array<f32>;
@group(0) @binding(1) var<storage,read_write> planes:array<u32>;
@group(0) @binding(2) var<storage,read> map:array<u32>;
@group(0) @binding(3) var<uniform> p:Params;
var<workgroup> cached_input:array<f32,2560>;
var<workgroup> packed_first:array<u32,640>;
var<workgroup> packed_second:array<u32,640>;
var<workgroup> scale_first:array<f32,80>;
var<workgroup> scale_second:array<f32,80>;
var<workgroup> quant_fault:atomic<u32>;
Q8_PREPARE_VALUE
fn shard_first(shard:u32)->u32 {
    var first=0u;
    if shard==1u {first=72u;}
    if shard==2u {first=144u;}
    if shard==3u {first=288u;}
    if shard==4u {first=432u;}
    return first;
}
fn shard_parts(shard:u32)->u32 {
    var parts=72u;
    if shard==2u {parts=144u;}
    if shard==3u {parts=144u;}
    if shard==4u {parts=80u;}
    return parts;
}
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
    // Finish every lane's storage-tail clear before record lanes overwrite it.
    storageBarrier();
    // Records build only from a valid immutable MAP header; a failed build
    // leaves the cleared tail zeros, and every consumer exits on ready!=1.
    if token==0u && local<5u {
        let shard=local;
        let first=shard_first(shard);let parts=shard_parts(shard);
        var map_valid=p.d<=arrayLength(&map) && 804u<=arrayLength(&map)-p.d;
        var groups=0u;
        if map_valid {
            if map[p.d]!=p.h || map[p.d+1u]!=0u || map[p.d+2u]!=p.h {map_valid=false;}
            else {groups=map[p.d+3u];}
        }
        // p.e is the caller's capacity bound; the MAP header carries the
        // exact routed group count for this source.
        if map_valid && (p.e<1u || p.e>30u || groups<1u || groups>p.e) {map_valid=false;}
        if map_valid {
            var count=0u;
            for(var g=0u;g<groups;g+=1u) {
                let slot=map[p.d+4u+g];
                if slot>=first && slot-first<parts {
                    planes[6144u+48u*shard+8u+count]=g;
                    count+=1u;
                }
            }
            var gu_y=count;if gu_y>4u {gu_y=4u;}
            planes[6144u+48u*shard+0u]=1u;
            planes[6144u+48u*shard+1u]=count;
            planes[6144u+48u*shard+2u]=80u;
            planes[6144u+48u*shard+3u]=gu_y;
            planes[6144u+48u*shard+4u]=1u;
            planes[6144u+48u*shard+5u]=320u;
            planes[6144u+48u*shard+6u]=count;
            planes[6144u+48u*shard+7u]=1u;
        }
    }
}
// FROZEN_GU_PREPARE_COMPACT_END
// FROZEN_GU_PROJECT_COMPACT_MAIN_BEGIN
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) local:u32) {
    // p.g is this launch's shard; the record bounds every dispatch and list
    // access. A shard whose record never built exits before any codebook write.
    if group.x>=80u || group.y>=4u || group.z!=0u || p.h<1u || p.h>3u || p.g>4u {return;}
    let base=6144u+48u*p.g;
    if planes[base]!=1u {return;}
    let shard_count=planes[base+1u];
    if group.y>=shard_count {return;}
    let row=group.x*8u+local/8u;let lane=local%8u;
    GRID_INITIALIZE_VALUE
    workgroupBarrier();
    // One indirect dispatch already bounded the shards; the ordinal stride
    // keeps the original codebook reuse and per-group arithmetic order.
    for(var ordinal=group.y;ordinal<shard_count;ordinal+=4u) {
        let routed_group=planes[base+8u+ordinal];
        if valid_group(routed_group) {
            let length=map[p.a+84u+routed_group];let slot=map[p.a+4u+routed_group]-p.b;
            let offset=(slot*640u+row)*ROW_BYTES;
            var frames=vec4<u32>(0u);
            FRAME_MEMBERS_VALUE
            // Each group's scalar/vector accumulators and every reduction
            // cell reset in the shared body; arithmetic order is unchanged.
            PROJECT_BODY_VALUE
            // All lanes finish reading their reduction before the next group
            // can overwrite any shared cell, including unused member cells.
            workgroupBarrier();
        }
    }
}
// FROZEN_GU_PROJECT_COMPACT_MAIN_END
// FROZEN_GU_COMPACT_AUDIT_BEGIN
struct Params {count:u32,r0:u32,r1:u32,r2:u32,r3:u32,r4:u32,r5:u32,r6:u32}
@group(0) @binding(0) var<storage,read_write> planes:array<u32>;
@group(0) @binding(1) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(2) var<storage,read> map:array<u32>;
@group(0) @binding(3) var<uniform> p:Params;
var<workgroup> records_valid:atomic<u32>;
fn audit_shard_first(shard:u32)->u32 {
    var first=0u;
    if shard==1u {first=72u;}
    if shard==2u {first=144u;}
    if shard==3u {first=288u;}
    if shard==4u {first=432u;}
    return first;
}
fn audit_shard_parts(shard:u32)->u32 {
    var parts=72u;
    if shard==2u {parts=144u;}
    if shard==3u {parts=144u;}
    if shard==4u {parts=80u;}
    return parts;
}
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global:vec3<u32>,@builtin(workgroup_id) workgroup:vec3<u32>,
    @builtin(local_invocation_index) local:u32) {
    if arrayLength(&fault)!=8192u {return;}
    if p.count<1u || p.count>3u || p.r2!=0u || p.r3!=0u || p.r4!=0u || p.r5!=0u || p.r6!=0u
        || p.r1<1u || p.r1>30u || arrayLength(&planes)!=16384u {
        if global.x==0u {atomicOr(&fault[0],8u);}
        return;
    }
    if workgroup.x<128u {
        // Original per-word validation; the five shard records are validated
        // and erased only by the dedicated workgroup below.
        let at=global.x;if at>=16384u {return;}var valid=true;
        if at<64u {valid=planes[at]==select(0u,1u,at<p.count);}
        else if at<4384u {
            let frame=(at-64u)/1440u;let offset=(at-64u)%1440u;
            if frame>=p.count {valid=planes[at]==0u;}
            else if offset>=1280u {
                let scale=bitcast<f32>(planes[at]);
                valid=scale>=0.0 && scale<=3.4028234663852886e38;
            }
        } else if at<6144u {valid=planes[at]==0u;}
        else if at<6384u {return;}
        else {valid=planes[at]==0u;}
        if !valid {atomicOr(&fault[0],8u);}
        return;
    }
    if global.x>=16512u {return;}
    if local==0u {atomicStore(&records_valid,1u);}
    workgroupBarrier();
    if local<5u {
        let shard=local;
        let base=6144u+48u*shard;
        let first=audit_shard_first(shard);let parts=audit_shard_parts(shard);
        // Reconstruct with the builder's exact rule from the unchanged MAP.
        var map_valid=p.r0<=arrayLength(&map) && 804u<=arrayLength(&map)-p.r0;
        var groups=0u;
        if map_valid {
            if map[p.r0]!=p.count || map[p.r0+1u]!=0u || map[p.r0+2u]!=p.count {map_valid=false;}
            else {groups=map[p.r0+3u];}
        }
        if map_valid && (groups<1u || groups>p.r1) {map_valid=false;}
        if !map_valid {
            for(var i=0u;i<48u;i+=1u) {
                if planes[base+i]!=0u {atomicStore(&records_valid,0u);atomicOr(&fault[0],8u);}
            }
        } else {
            var count=0u;
            for(var g=0u;g<groups;g+=1u) {
                let slot=map[p.r0+4u+g];
                if slot>=first && slot-first<parts {
                    if planes[base+8u+count]!=g {atomicStore(&records_valid,0u);atomicOr(&fault[0],8u);}
                    count+=1u;
                }
            }
            var gu_y=count;if gu_y>4u {gu_y=4u;}
            if planes[base+0u]!=1u || planes[base+1u]!=count
                || planes[base+2u]!=80u || planes[base+3u]!=gu_y || planes[base+4u]!=1u
                || planes[base+5u]!=320u || planes[base+6u]!=count || planes[base+7u]!=1u
            {atomicStore(&records_valid,0u);atomicOr(&fault[0],8u);}
            for(var i=count;i<40u;i+=1u) {
                if planes[base+8u+i]!=0u {atomicStore(&records_valid,0u);atomicOr(&fault[0],8u);}
            }
        }
    }
    workgroupBarrier();
    // Erase each disjoint record only after every shard validated; a failed
    // validation keeps the records as evidence with the sticky fault set.
    if atomicLoad(&records_valid)==1u && local<5u {
        let shard=local;
        for(var i=0u;i<48u;i+=1u) {planes[6144u+48u*shard+i]=0u;}
    }
}
// FROZEN_GU_COMPACT_AUDIT_END
