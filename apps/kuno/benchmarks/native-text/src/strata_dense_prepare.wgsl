// MIT Strata fb58e0 native_mmvq Q8_1 subgroup quantization, Rust/WGSL port.
// Native q8_1_quant rounds half away from zero and clamps its half scale/sum.
// Preserve that contract; this producer is selected only in the Strata core.
// The second
// plane remains explicitly zero in the original fully owned scratch envelope.
struct Params {input_first:u32,input_stride:u32,planes_stride:u32,count:u32}
@group(0) @binding(0) var<storage,read> input:array<f32>;
@group(0) @binding(1) var<storage,read_write> planes:array<u32>;
@group(0) @binding(2) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(3) var<uniform> p:Params;
fn q8_code(x:f32,d:f32)->i32 {
    if d==0.0 {return 0;}
    let value=clamp(x/d,-127.0,127.0);
    return i32(abs(value)+0.5)*select(-1,1,value>=0.0);
}
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(subgroup_id) sg:u32,
        @builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_size) width:u32,
        @builtin(local_invocation_index) local:u32) {
    if arrayLength(&fault)!=1u {return;}
    var valid=width==32u && p.count>=1u && p.count<=8u && p.input_stride>=COLUMNS
        && p.planes_stride>=PLANE_WORDS && p.input_first<=arrayLength(&input);
    if valid {
        let available=arrayLength(&input)-p.input_first;
        if COLUMNS>available || PLANE_WORDS>arrayLength(&planes) {valid=false;}
        else if (p.count-1u)>(available-COLUMNS)/p.input_stride
            || (p.count-1u)>(arrayLength(&planes)-PLANE_WORDS)/p.planes_stride {valid=false;}
    }
    if !valid {if local==0u {atomicOr(&fault[0],8u);}return;}
    if g.x>=BLOCKS/4u || g.y>=p.count || g.z!=0u {return;}
    let block=g.x*4u+sg;let base=g.y*p.planes_stride;
    let x=input[p.input_first+g.y*p.input_stride+block*32u+lane];
    let finite=abs(x)<3.4028234663852886e38;
    let all_finite=subgroupAll(finite);
    let maximum=subgroupMax(select(0.0,abs(x),finite));
    let d=min(maximum/127.0,65504.0);
    var sum=select(0.0,x,finite);for(var distance=16u;distance>0u;distance/=2u) {sum+=subgroupShuffleXor(sum,distance);}
    var q=0u;
    if all_finite {q=bitcast<u32>(q8_code(x,d))&255u;}
    let first=lane&~3u;
    let packed=subgroupShuffle(q,first)|(subgroupShuffle(q,first+1u)<<8u)
        |(subgroupShuffle(q,first+2u)<<16u)|(subgroupShuffle(q,first+3u)<<24u);
    if lane%4u==0u {
        planes[base+block*8u+lane/4u]=packed;
        planes[base+PACKED_WORDS+block*8u+lane/4u]=0u;
    }
    if lane==0u {
        var scale=0.0;
        if all_finite {scale=unpack2x16float(pack2x16float(vec2<f32>(d,0.0))).x;}
        else {atomicOr(&fault[0],1u);}
        planes[base+2u*PACKED_WORDS+block]=bitcast<u32>(scale);
        let half_sum=unpack2x16float(pack2x16float(vec2<f32>(clamp(sum,-65504.0,65504.0),0.0))).x;
        planes[base+2u*PACKED_WORDS+BLOCKS+block]=bitcast<u32>(half_sum);
    }
}
