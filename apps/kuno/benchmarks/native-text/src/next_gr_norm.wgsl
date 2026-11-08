// Original max-scaled GR RMS, vector loads and subgroup reductions.
struct Params { count:u32, input_first:u32, input_stride:u32, output_first:u32,
    output_stride:u32, status_first:u32, status_stride:u32, epsilon:f32 }
@group(0) @binding(0) var<storage,read> input4:array<vec4<f32>>;
@group(0) @binding(1) var<storage,read> gamma4:array<vec4<f32>>;
@group(0) @binding(2) var<storage,read_write> output4:array<vec4<f32>>;
@group(0) @binding(3) var<storage,read_write> status:array<u32>;
@group(0) @binding(4) var<uniform> p:Params;
var<workgroup> maxima:array<f32,64>;
var<workgroup> goods:array<u32,64>;
var<workgroup> sums:array<f32,64>;
var<workgroup> output_goods:array<u32,64>;
fn finite4(x:vec4<f32>)->bool {return all(abs(x)<=vec4<f32>(3.4028234663852886e38));}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) local:u32,
    @builtin(subgroup_id) sg:u32,@builtin(subgroup_invocation_id) lane:u32,
    @builtin(subgroup_size) subgroup_width:u32) {
    let token=group.y;let stream=group.x;
    let input=(p.input_first+token*p.input_stride+stream*WIDTH)/4u;
    let out=(p.output_first+token*p.output_stride+stream*WIDTH)/4u;
    let gamma=stream*WIDTH/4u;
    var maximum=1.0;var good=1u;
    for(var d=local;d<WIDTH/4u;d+=64u) {
        let x=input4[input+d];let w=gamma4[gamma+d];
        if finite4(x) && finite4(w) {
            let a=abs(x);maximum=max(maximum,max(max(a.x,a.y),max(a.z,a.w)));
        } else {good=0u;}
    }
    let sg_max=subgroupMax(maximum);let sg_good=subgroupMin(good);
    if lane==0u {maxima[sg]=sg_max;goods[sg]=sg_good;}
    workgroupBarrier();
    var scale=1.0;var all_good=1u;
    for(var j=0u;j<64u/subgroup_width;j++) {scale=max(scale,maxima[j]);all_good=min(all_good,goods[j]);}
    var sum=0.0;
    if all_good==1u {
        for(var d=local;d<WIDTH/4u;d+=64u) {
            let x=input4[input+d]/vec4<f32>(scale);sum+=dot(x,x);
        }
    }
    let sg_sum=subgroupAdd(sum);
    if lane==0u {sums[sg]=sg_sum;}
    workgroupBarrier();
    var total=0.0;
    for(var j=0u;j<64u/subgroup_width;j++) {total+=sums[j];}
    let denominator=sqrt(total/f32(WIDTH)+(p.epsilon/scale)/scale);
    var output_good=1u;
    for(var d=local;d<WIDTH/4u;d+=64u) {
        let y=(input4[input+d]/vec4<f32>(scale))/vec4<f32>(denominator)*gamma4[gamma+d];
        // Preserve per-element output clearing even when one output overflows.
        let valid=abs(y)<=vec4<f32>(3.4028234663852886e38);
        if all_good==1u && abs(p.epsilon)<=3.4028234663852886e38 && p.epsilon>0.0 {
            output4[out+d]=select(vec4<f32>(0.0),y,valid);
            if !all(valid) {output_good=0u;}
        } else {output4[out+d]=vec4<f32>(0.0);output_good=0u;}
    }
    let sg_out=subgroupMin(output_good);
    if lane==0u {output_goods[sg]=sg_out;}
    workgroupBarrier();
    var result=1u;
    for(var j=0u;j<64u/subgroup_width;j++) {result=min(result,output_goods[j]);}
    if local==0u {status[p.status_first+token*p.status_stride+stream]=result;}
}
