// MIT Strata fb58e0 sampler_greedy_kernel's no-penalty Intel path.
@group(0) @binding(0) var<storage,read> logits:array<f32>;
@group(0) @binding(1) var<storage,read_write> selected:array<u32>;
@group(0) @binding(2) var<storage,read_write> fault:array<atomic<u32>>;
var<workgroup> winners:array<vec2<u32>,32>;
fn reduce(value:vec2<u32>,lane:u32)->vec2<u32>{var best=value;for(var off=16u;off>0u;off/=2u){let x=subgroupShuffleDown(bitcast<f32>(best.y),off);let id=subgroupShuffleDown(best.x,off);if lane+off<32u&&(x>bitcast<f32>(best.y)||(x==bitcast<f32>(best.y)&&id<best.x)){best=vec2<u32>(id,bitcast<u32>(x));}}return best;}
@compute @workgroup_size(1024)
fn main(@builtin(subgroup_id) sg:u32,@builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_size) width:u32){
 if arrayLength(&fault)<1u{return;}
 if width!=32u||arrayLength(&logits)<248320u||arrayLength(&selected)<1u{atomicOr(&fault[0],0x10000000u);return;}
 var best=vec2<u32>(248320u,0xff800000u);
 for(var i=sg*32u+lane;i<248320u;i+=1024u){let x=logits[i];if !(abs(x)<=bitcast<f32>(0x7f7fffffu)){atomicOr(&fault[0],0x10000000u);}else if x>bitcast<f32>(best.y)||(x==bitcast<f32>(best.y)&&i<best.x){best=vec2<u32>(i,bitcast<u32>(x));}}
 let winner=reduce(best,lane);if lane==0u{winners[sg]=winner;}workgroupBarrier();storageBarrier();
 if sg==0u{let answer=reduce(winners[lane],lane);if lane==0u{selected[0]=select(248320u,answer.x,atomicLoad(&fault[0])==0u);}}
}
