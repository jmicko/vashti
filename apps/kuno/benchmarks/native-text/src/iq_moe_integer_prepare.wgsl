// Quantization body is injected verbatim from the accepted integer operator.
const COLUMNS:u32=COLUMNS_VALUEu;
struct Params {a:u32,b:u32,c:u32,d:u32,e:u32,f:u32,g:u32,h:u32}
@group(0) @binding(0) var<storage,read> input:array<f32>;
@group(0) @binding(1) var<storage,read> map:array<u32>;
@group(0) @binding(2) var<storage,read> status:array<u32>;
@group(0) @binding(3) var<storage,read_write> planes:array<u32>;
@group(0) @binding(4) var<uniform> p:Params;
var<workgroup> cached_input:array<f32,COLUMNS_VALUE>;
var<workgroup> packed_first:array<u32,PACKED_VALUE>;
var<workgroup> packed_second:array<u32,PACKED_VALUE>;
var<workgroup> scale_first:array<f32,BLOCKS_VALUE>;
var<workgroup> scale_second:array<f32,BLOCKS_VALUE>;
var<workgroup> quant_fault:atomic<u32>;
fn valid_group(group:u32)->bool {
    let n=arrayLength(&map);
    if p.a>n {return false;}
    if 804u>n-p.a {return false;}
    if map[p.a]!=1u || map[p.a+1u]!=0u || map[p.a+2u]!=1u {return false;}
    let groups=map[p.a+3u];
    if groups==0u || groups>10u || group>=groups {return false;}
    return map[p.a+4u+group]<144u && map[p.a+84u+group]==1u && map[p.a+164u+group*8u]<10u;
}
Q8_PREPARE_VALUE
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) local:u32) {
    var valid=valid_group(g.y);var route=0u;
    if valid {route=map[p.a+164u+g.y*8u];}
    var frame=p.d+32u;var flag=p.d;var offset=0u;
    if DOWN_VALUE {
        frame=p.d+1472u+g.y*360u;flag=p.d+1u+g.y;offset=route*p.c;
        if valid && status[p.e+route]!=1u {valid=false;}
    } else {
        // Every down flag starts invalid. A later separate dispatch publishes
        // readiness only after its distinct ACT input has been quantized.
        if local<11u {planes[p.d+local]=0u;}
    }
    if local==0u {atomicStore(&quant_fault,0u);}
    for(var i=local;i<COLUMNS;i+=128u) {
        var x=0.0;if valid {x=input[p.b+offset+i];}
        cached_input[i]=x;
    }
    workgroupBarrier();
    prepare_q8(local);
    workgroupBarrier();
    for(var i=local;i<PACKED_VALUEu;i+=128u) {
        planes[frame+i]=packed_first[i];
        planes[frame+PACKED_VALUEu+i]=packed_second[i];
    }
    for(var i=local;i<BLOCKS_VALUEu;i+=128u) {
        planes[frame+2u*PACKED_VALUEu+i]=bitcast<u32>(scale_first[i]);
        planes[frame+2u*PACKED_VALUEu+BLOCKS_VALUEu+i]=bitcast<u32>(scale_second[i]);
    }
    if local==0u {planes[flag]=select(0u,1u,valid && atomicLoad(&quant_fault)==0u);}
}
