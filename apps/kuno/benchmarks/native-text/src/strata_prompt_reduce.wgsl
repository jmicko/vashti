// Prompt-only native4096 layout; decode keeps its original MAX8 contract.
struct Params {count:u32,layer:u32,epoch:u32,input_first:u32,output_first:u32,output_stride:u32,r0:u32,r1:u32}
@group(0) @binding(0) var<storage,read> partial:array<f32>;
@group(0) @binding(1) var<storage,read> probability:array<f32>;
@group(0) @binding(2) var<storage,read> status:array<u32>;
@group(0) @binding(3) var<storage,read> map:array<u32>;
@group(0) @binding(4) var<storage,read_write> output:array<f32>;
@group(0) @binding(5) var<storage,read_write> audit:array<u32>;
@group(0) @binding(6) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(7) var<uniform> p:Params;
fn finite(x:f32)->bool {return abs(x)<=bitcast<f32>(0x7f7fffffu);}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    if p.count<1u || p.count>PROMPT_MAX_ROWS || p.layer>=48u || p.epoch==0u || p.input_first!=0u || p.r0!=0u || p.r1!=0u
        || arrayLength(&partial)!=PROMPT_PART_WORDS || arrayLength(&probability)!=PROMPT_PROB_WORDS
        || arrayLength(&status)!=PROMPT_STATUS_WORDS || arrayLength(&map)!=PROMPT_MAP_WORDS
        || arrayLength(&audit)!=8192u || arrayLength(&fault)<1u {return;}
    if p.output_stride<2560u || p.output_first>arrayLength(&output)
        || 2560u>arrayLength(&output)-p.output_first
        || p.count-1u>(arrayLength(&output)-p.output_first-2560u)/p.output_stride {return;}
    let row=id.y;let column=id.x;
    if row>=p.count || column>=2560u || id.z!=0u {return;}
    var valid=map[0]==p.count && map[1]==0u && map[2]==p.count && map[3]<=p.count*10u;
    var sum=0.0;var weight_sum=0.0;
    for(var rank=0u;rank<10u;rank++) {
        let route=row*10u+rank;
        let x=partial[p.input_first+route*2560u+column];let weight=probability[route];
        valid=valid && status[route]==1u && status[PROMPT_ROUTES+route]==1u && finite(x) && finite(weight) && weight>=0.0;
        sum=fma(x,weight,sum);weight_sum+=weight;
    }
    valid=valid && finite(sum) && abs(weight_sum-1.0)<0.00002;
    if valid {output[p.output_first+row*p.output_stride+column]=sum;}
    else {atomicOr(&fault[0],256u);}
    if row==0u && column==0u {
        // The full-model fault fence precedes CPU acceptance of this stamp.
        audit[p.layer*4u]=p.count;audit[p.layer*4u+1u]=p.epoch;
        audit[p.layer*4u+2u]=127u;audit[p.layer*4u+3u]=p.layer+1u;
    }
}
