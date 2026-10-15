// MIT Strata target selection/carry boundary. The preceding sampler owns the
// logits scan and sticky finite fault; this separate dispatch sees its join.
struct Params{count:u32,start:u32,zero0:u32,zero1:u32}
@group(0) @binding(0) var<storage,read> picks:array<vec2<u32>>;
@group(0) @binding(1) var<storage,read> carry:array<u32>;
@group(0) @binding(2) var<storage,read> fault:array<u32>;
@group(0) @binding(3) var<storage,read_write> output:array<u32>;
@group(0) @binding(4) var<uniform> p:Params;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32){
 let row=group.x;
 if row>=VERIFY_ROWSu||p.count<1u||p.count>VERIFY_ROWSu||p.start+p.count>4096u||arrayLength(&picks)<p.count||arrayLength(&carry)<VERIFY_ROWSu*10240u||arrayLength(&fault)<1u||arrayLength(&output)!=PACKET_OWNER_WORDSu||p.zero0!=0u||p.zero1!=0u{return;}
 if lane==0u{var id=0u;if row<p.count{let pair=picks[row];let probability=bitcast<f32>(pair.y);id=pair.x;if id>=248320u||!(probability>=0.0&&probability<=1.0)||fault[0]!=0u{id=248320u;}}output[8u+row]=id;}
 if row==0u&&lane<8u{var stamp=0u;switch lane{case 0u:{stamp=2u;}case 1u:{stamp=p.count;}case 2u:{stamp=p.start;}case 3u:{stamp=248320u;}case 4u:{stamp=10240u;}case 5u:{stamp=VERIFY_ROWSu;}case 6u:{stamp=fault[0];}default:{}}output[lane]=stamp;}
 for(var col=lane;col<10240u;col+=128u){var v=0u;if row<p.count{v=carry[row*10240u+col];}output[CARRY_FIRSTu+row*10240u+col]=v;}
}
