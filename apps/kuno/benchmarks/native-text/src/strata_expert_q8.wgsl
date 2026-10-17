// MIT Strata fb58e0 native q8_1_store and swiglu_q8_1_entries_kernel.
// COMMON_BEGIN
struct Params {count:u32,first:u32,stride:u32,status_first:u32,r0:u32,r1:u32,r2:u32,r3:u32}
var<workgroup> quant_fault:atomic<u32>;
fn finite(x:f32)->bool {return abs(x)<=3.4028234663852886e38;}
fn q8_code(x:f32,d:f32)->i32 {
    if d==0.0 {return 0;}
    let v=clamp(x/d,-127.0,127.0);let whole=trunc(v);
    let extra=select(0.0,1.0,abs(v-whole)>=0.5);
    return i32(whole+select(-extra,extra,v>=0.0));
}
fn write_block(x:f32,lane:u32,block:u32,frame:u32,blocks:u32) {
    let good=subgroupAll(finite(x));
    let maximum=subgroupMax(select(0.0,abs(x),finite(x)));
    let d=min(maximum/127.0,65504.0);
    var sum=select(0.0,x,finite(x));
    for(var distance=16u;distance>0u;distance/=2u) {sum+=subgroupShuffleXor(sum,distance);}
    let q=bitcast<u32>(q8_code(select(0.0,x,good),d))&255u;
    let first=lane&~3u;
    let packed=subgroupShuffle(q,first)|(subgroupShuffle(q,first+1u)<<8u)
        |(subgroupShuffle(q,first+2u)<<16u)|(subgroupShuffle(q,first+3u)<<24u);
    if lane%4u==0u {planes[frame+block*8u+lane/4u]=packed;}
    if lane==0u {
        let ds=unpack2x16float(pack2x16float(vec2<f32>(d,clamp(sum,-65504.0,65504.0))));
        planes[frame+blocks*16u+block]=bitcast<u32>(ds.x);
        planes[frame+blocks*17u+block]=bitcast<u32>(ds.y);
        if !good || !finite(ds.x) || !finite(ds.y) {atomicOr(&quant_fault,1u);}
    }
}
// COMMON_END
// GU_BEGIN
@group(0) @binding(0) var<storage,read> input:array<f32>;
@group(0) @binding(1) var<storage,read_write> planes:array<u32>;
@group(0) @binding(2) var<uniform> p:Params;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) local:u32,
 @builtin(subgroup_id) sg:u32,@builtin(subgroup_invocation_id) lane:u32,
 @builtin(subgroup_size) width:u32,@builtin(num_subgroups) subgroups:u32) {
    if p.count<1u || p.count>8u || p.stride<COLUMNS || p.status_first!=0u
        || p.r0!=0u || p.r1!=0u || p.r2!=0u || p.r3!=0u
        || arrayLength(&planes)!=TOTAL_WORDS || width!=32u || subgroups!=8u
        || group.x!=0u || group.y>=p.count || group.z!=0u || p.first>arrayLength(&input) {return;}
    let available=arrayLength(&input)-p.first;
    if COLUMNS>available || (p.count-1u)>(available-COLUMNS)/p.stride {return;}
    let token=group.y;let frame=HEADER_WORDS+token*FRAME_WORDS;
    if local==0u {atomicStore(&quant_fault,0u);planes[token]=0u;}
    for(var at=local;at<FRAME_WORDS;at+=256u) {planes[frame+at]=0u;}
    if token==0u {
        for(var at=p.count+local;at<HEADER_WORDS;at+=256u) {planes[at]=0u;}
        for(var at=HEADER_WORDS+p.count*FRAME_WORDS+local;at<TOTAL_WORDS;at+=256u) {planes[at]=0u;}
    }
    workgroupBarrier();storageBarrier(); // Different lanes clear and pack each storage word.
    for(var step=0u;step<BLOCKS/8u;step++) {
        let block=step*8u+sg;
        let x=input[p.first+token*p.stride+block*32u+lane];
        write_block(x,lane,block,frame,BLOCKS);
    }
    workgroupBarrier();storageBarrier();
    if local==0u {planes[token]=select(0u,1u,atomicLoad(&quant_fault)==0u);}
}
// GU_END
// DOWN_BEGIN
@group(0) @binding(0) var<storage,read_write> partial:array<f32>;
@group(0) @binding(1) var<storage,read_write> status:array<u32>;
@group(0) @binding(2) var<storage,read_write> planes:array<u32>;
@group(0) @binding(3) var<uniform> p:Params;
// The materialized F32 product and workgroup barrier preserve the source's
// contract(off) product before q8_1_store's first sum. No ACT device root copy.
var<workgroup> activated:array<f32,640>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) local:u32,
 @builtin(subgroup_id) sg:u32,@builtin(subgroup_invocation_id) lane:u32,
 @builtin(subgroup_size) width:u32,@builtin(num_subgroups) subgroups:u32) {
    if p.count<1u || p.count>8u || p.stride<2560u
        || p.r0!=0u || p.r1!=0u || p.r2!=0u || p.r3!=0u
        || arrayLength(&planes)!=TOTAL_WORDS || width!=32u || subgroups!=8u
        || group.x!=0u || group.y>=p.count*ROWS_PER_INPUT || group.z!=0u
        || p.first>arrayLength(&partial) || p.status_first>arrayLength(&status) {return;}
    let available=arrayLength(&partial)-p.first;
    if 1280u>available || (p.count*ROWS_PER_INPUT-1u)>(available-1280u)/p.stride
        || 160u>arrayLength(&status)-p.status_first {return;}
    let route=group.y;let frame=HEADER_WORDS+route*FRAME_WORDS;
    let completed=status[p.status_first+route]==1u && status[p.status_first+80u+route]==1u;
    if local==0u {atomicStore(&quant_fault,0u);planes[route]=0u;}
    for(var at=local;at<FRAME_WORDS;at+=256u) {planes[frame+at]=0u;}
    if route==0u {
        for(var at=p.count*ROWS_PER_INPUT+local;at<HEADER_WORDS;at+=256u) {planes[at]=0u;}
        for(var at=HEADER_WORDS+p.count*ROWS_PER_INPUT*FRAME_WORDS+local;at<TOTAL_WORDS;at+=256u) {planes[at]=0u;}
    }
    workgroupBarrier();storageBarrier(); // Initialize fault and order plane clears before packing.
    let base=p.first+route*p.stride;
    for(var at=local;at<640u;at+=256u) {
        var y=0.0;
        if completed {
            let g=partial[base+at];let up=partial[base+640u+at];
            y=(g/(1.0+exp(-g)))*up;
            if !finite(g) || !finite(up) {atomicOr(&quant_fault,1u);}
        }
        activated[at]=y;
        partial[base+at]=0.0;partial[base+640u+at]=0.0;
    }
    workgroupBarrier();
    for(var step=0u;step<(BLOCKS+7u)/8u;step++) {
        let block=step*8u+sg;
        if block<BLOCKS {write_block(activated[block*32u+lane],lane,block,frame,BLOCKS);}
    }
    workgroupBarrier();storageBarrier();
    for(var at=local;at<640u;at+=256u) {activated[at]=0.0;}
    if local==0u {
        let good=completed && atomicLoad(&quant_fault)==0u;
        planes[route]=select(0u,1u,good);
        status[p.status_first+route]=select(0u,1u,good);
        status[p.status_first+80u+route]=0u;
    }
}
// DOWN_END
