// Preserve original SharedMerge's separate multiply/add source and its actual
// unannotated compiler policy. The paired original-kernel fixture checks bits.
// One workgroup owns one token and all ten of its unique original routes.
struct Params {count:u32,layer:u32,epoch:u32,input_first:u32,output_first:u32,output_stride:u32,r0:u32,r1:u32}
@group(0) @binding(0) var<storage,read_write> partial:array<vec4<f32>>;
@group(0) @binding(1) var<storage,read> probability:array<f32>;
@group(0) @binding(2) var<storage,read_write> status:array<u32>;
@group(0) @binding(3) var<storage,read> map:array<u32>;
@group(0) @binding(4) var<storage,read_write> routed:array<vec4<f32>>;
@group(0) @binding(5) var<storage,read_write> audit:array<u32>;
@group(0) @binding(6) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(7) var<storage,read> inverse:array<u32>;
@group(0) @binding(8) var<storage,read> shared_out:array<vec4<f32>>;
@group(0) @binding(9) var<storage,read> scalar:array<f32>;
@group(0) @binding(10) var<storage,read_write> combined:array<vec4<f32>>;
@group(0) @binding(11) var<uniform> p:Params;
var<workgroup> ordinals:array<u32,10>;
var<workgroup> mappings:array<u32,10>;
var<workgroup> markers:array<u32,10>;
var<workgroup> route_valid:array<u32,10>;
var<workgroup> weights:array<f32,10>;
var<workgroup> shared_gate:f32;
var<workgroup> shared_scalar_finite:u32;
fn finite(x:f32)->bool {return abs(x)<=bitcast<f32>(0x7f7fffffu);}
fn finite4(x:vec4<f32>)->vec4<bool> {return abs(x)<=vec4<f32>(bitcast<f32>(0x7f7fffffu));}
fn and4(a:vec4<bool>,b:vec4<bool>)->vec4<bool> {return vec4<bool>(a.x&&b.x,a.y&&b.y,a.z&&b.z,a.w&&b.w);}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) block:vec3<u32>,@builtin(local_invocation_id) local:vec3<u32>) {
    if p.count<1u || p.count>PROMPT_MAX_ROWS || p.layer>=48u || p.epoch==0u || p.input_first!=0u || p.output_first!=0u || p.output_stride!=2560u || p.r0!=0u || p.r1!=0u
        || arrayLength(&partial)!=PROMPT_PART_WORDS/4u || arrayLength(&probability)!=PROMPT_PROB_WORDS
        || arrayLength(&status)!=PROMPT_STATUS_WORDS || arrayLength(&map)!=PROMPT_MAP_WORDS
        || arrayLength(&inverse)!=PROMPT_RAW_WORDS || arrayLength(&audit)!=8192u || arrayLength(&fault)<1u
        || arrayLength(&routed)<p.count*640u || arrayLength(&shared_out)<p.count*640u
        || arrayLength(&scalar)<p.count || arrayLength(&combined)<p.count*640u {return;}
    let row=block.x;let lane=local.x;
    if row>=p.count || block.y!=0u || block.z!=0u {return;}
    // No status normalization or PART sanitization precedes this snapshot.
    // The map roundtrip also prevents two accepted routes sharing an ordinal.
    if lane<10u {
        let route=row*10u+lane;let ordinal=inverse[route];
        let gu=status[route];let down=status[PROMPT_ROUTES+route];
        var mapped=ordinal<p.count*10u;
        if mapped {mapped=map[1540u+ordinal]==route;}
        var valid=mapped && map[0]==p.count && map[1]==0u && map[2]==p.count && map[3]<=p.count*10u;
        mappings[lane]=select(0u,1u,mapped);
        ordinals[lane]=ordinal;markers[lane]=down;weights[lane]=probability[route];
        if valid && down==2u {
            // Original DOWN's route stamp depends specifically on raw col0.
            valid=finite(partial[ordinal*640u].x);
        }
        route_valid[lane]=select(0u,1u,valid && gu==1u && (down==1u || down==2u));
    }
    if lane==0u {
        let s=scalar[row];shared_scalar_finite=select(0u,1u,finite(s));
        shared_gate=1.0/(1.0+exp(-s));
    }
    workgroupBarrier();
    storageBarrier();
    if lane<10u && markers[lane]==2u {
        status[PROMPT_ROUTES+row*10u+lane]=route_valid[lane];
    }
    var common_valid=true;var weight_sum=0.0;
    for(var rank=0u;rank<10u;rank++) {
        let weight=weights[rank];
        common_valid=common_valid && route_valid[rank]==1u && finite(weight) && weight>=0.0;
        weight_sum+=weight;
    }
    common_valid=common_valid && abs(weight_sum-1.0)<0.00002;
    for(var column=lane;column<640u;column+=256u) {
        var sum=vec4<f32>(0.0);var valid=vec4<bool>(common_valid);
        for(var rank=0u;rank<10u;rank++) {
            let ordinal=ordinals[rank];let weight=weights[rank];var x=vec4<f32>(0.0);
            // Missing marker0 never authorizes any stale PART load.
            let mapped=mappings[rank]==1u;
            if mapped && (markers[rank]==1u || markers[rank]==2u) {
                x=partial[ordinal*640u+column];
                if markers[rank]==2u {
                    let good=finite4(x);
                    x=select(vec4<f32>(0.0),x,good);
                    partial[ordinal*640u+column]=x;
                    if !all(good) {atomicOr(&fault[0],128u);}
                }
            }
            valid=and4(valid,finite4(x));
            // Include rank0's FMA with positive zero, as the original prompt.
            sum=fma(x,vec4<f32>(weight),sum);
        }
        valid=and4(valid,finite4(sum));
        let at=row*640u+column;var retained=routed[at];
        if valid.x {routed[at].x=sum.x;retained.x=sum.x;}
        if valid.y {routed[at].y=sum.y;retained.y=sum.y;}
        if valid.z {routed[at].z=sum.z;retained.z=sum.z;}
        if valid.w {routed[at].w=sum.w;retained.w=sum.w;}
        if !all(valid) {atomicOr(&fault[0],256u);}
        let product=shared_out[at]*vec4<f32>(shared_gate);
        let value=retained+product;
        let good=and4(finite4(value),vec4<bool>(shared_scalar_finite==1u));
        combined[at]=select(vec4<f32>(0.0),value,good);
        if !all(good) {atomicOr(&fault[0],1u);}
    }
    if row==0u && lane==0u {
        audit[p.layer*4u]=p.count;audit[p.layer*4u+1u]=p.epoch;
        audit[p.layer*4u+2u]=127u;audit[p.layer*4u+3u]=p.layer+1u;
    }
}
