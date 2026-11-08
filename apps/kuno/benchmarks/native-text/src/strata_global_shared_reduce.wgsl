// MIT Strata fb58e0 native_moe combine_k10_vec4; authentic joined shared row.
struct Params {count:u32,layer:u32,epoch:u32,input_first:u32,output_first:u32,output_stride:u32,r0:u32,r1:u32}
@group(0) @binding(0) var<storage,read> partial:array<vec4<f32>>;
@group(0) @binding(1) var<storage,read> probability:array<f32>;
@group(0) @binding(2) var<storage,read> status:array<u32>;
@group(0) @binding(3) var<storage,read> map:array<u32>;
@group(0) @binding(4) var<storage,read_write> output:array<vec4<f32>>;
@group(0) @binding(5) var<storage,read_write> audit:array<u32>;
@group(0) @binding(6) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(7) var<storage,read> shared_result:array<vec4<f32>>;
@group(0) @binding(8) var<uniform> p:Params;
fn finite(x:f32)->bool {return abs(x)<=bitcast<f32>(0x7f7fffffu);}
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    if p.count<1u || p.count>8u || p.layer>=48u || p.epoch==0u || p.input_first!=0u || p.r0!=0u || p.r1!=0u
        || arrayLength(&partial)!=65536u || arrayLength(&shared_result)!=5120u || arrayLength(&probability)!=8192u
        || arrayLength(&status)!=8192u || arrayLength(&map)!=8192u
        || arrayLength(&audit)!=8192u || arrayLength(&fault)<1u {return;}
    if p.output_first%4u!=0u || p.output_stride%4u!=0u || p.output_stride<2560u || p.output_first>(arrayLength(&output)*4u)
        || 2560u>(arrayLength(&output)*4u)-p.output_first
        || p.count-1u>((arrayLength(&output)*4u)-p.output_first-2560u)/p.output_stride {return;}
    let row=id.y;let column=id.x*4u;
    if row>=p.count || column>=2560u || id.z!=0u {return;}
    var valid=map[0]==p.count && map[1]==0u && map[2]==p.count && map[3]<=p.count*10u;
    var sum=vec4<f32>(0.0);var weight_sum=0.0;
    for(var rank=0u;rank<10u;rank++) {
        let route=row*10u+rank;
        let x=partial[p.input_first/4u+route*640u+id.x];let weight=probability[route];
        valid=valid && status[route]==1u && status[80u+route]==1u && all(abs(x)<=vec4<f32>(bitcast<f32>(0x7f7fffffu))) && finite(weight) && weight>=0.0;
        if rank==0u {sum=x*weight;}else{sum=fma(x,vec4<f32>(weight),sum);}
        weight_sum+=weight;
    }
    let sh=shared_result[row*640u+id.x];sum+=sh;
    valid=valid && all(abs(sh)<=vec4<f32>(bitcast<f32>(0x7f7fffffu))) && all(abs(sum)<=vec4<f32>(bitcast<f32>(0x7f7fffffu))) && abs(weight_sum-1.0)<0.00002;
    if valid {output[(p.output_first+row*p.output_stride)/4u+id.x]=sum;}
    else {atomicOr(&fault[0],256u);}
    if row==0u && column==0u {
        // The full-model fault fence precedes CPU acceptance of this stamp.
        audit[p.layer*4u]=p.count;audit[p.layer*4u+1u]=p.epoch;
        audit[p.layer*4u+2u]=127u;audit[p.layer*4u+3u]=p.layer+1u;
    }
}
