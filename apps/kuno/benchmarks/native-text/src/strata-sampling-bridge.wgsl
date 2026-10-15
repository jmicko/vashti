// MIT Strata fb58e0 coupled sampler staging/publish; Rust private owners.
// PREPARE_BEGIN
@group(0) @binding(0) var<storage,read_write> params:array<u32>;
@group(0) @binding(1) var<storage,read_write> history:array<i32>;
@group(0) @binding(2) var<storage,read_write> ring:array<i32>;
@group(0) @binding(3) var<storage,read> base:array<u32>;
@group(0) @binding(4) var<storage,read> positions:array<u32>;
@group(0) @binding(5) var<storage,read> control:array<u32>;
@group(0) @binding(6) var<storage,read> input_token:array<u32>;
@group(0) @binding(7) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(8) var<storage,read_write> picks:array<vec2<u32>>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) lane:u32){
 if arrayLength(&fault)<1u{return;}
 if lane==0u&&arrayLength(&picks)>0u{picks[0]=vec2<u32>(248320u,0x7fc00000u);}
 if arrayLength(&params)<32u||arrayLength(&history)<4096u||arrayLength(&ring)<4112u||arrayLength(&base)<4160u||arrayLength(&control)<3u||arrayLength(&input_token)<1u||arrayLength(&picks)<1u{atomicOr(&fault[0],0x10000000u);return;}
 let cell=control[0];let j=control[1];let start=control[2];let h=min(base[5],4096u);
 if cell>=4096u||cell>=arrayLength(&positions)||positions[cell]!=cell||j>=7u||cell+1u!=start+j||base[0]!=1u||base[4]!=h||base[7]!=0u{atomicOr(&fault[0],0x10000000u);return;}
 if lane<32u&&lane!=16u&&lane!=17u{params[lane]=base[lane];}if lane==0u{params[16]=positions[cell]+1u;params[17]=0u;}
 if j==0u{for(var i=lane;i<4112u;i+=256u){var v=-1;if i<4096u{v=bitcast<i32>(base[64u+i]);}ring[i]=v;}}
 storageBarrier();workgroupBarrier();
 if input_token[0]>=248320u||u32(ring[4095u+j])!=input_token[0]{atomicOr(&fault[0],0x10000000u);}
 for(var i=lane;i<h;i+=256u){history[i]=ring[4096u+j-h+i];}
 if h==0u&&lane==0u{history[0]=-1;}
}
// END
// PUBLISH_BEGIN
@group(0) @binding(0) var<storage,read> picks:array<vec2<u32>>;
@group(0) @binding(1) var<storage,read_write> ring:array<i32>;
@group(0) @binding(2) var<storage,read_write> token:array<u32>;
@group(0) @binding(3) var<storage,read> control:array<u32>;
@group(0) @binding(4) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(5) var<storage,read_write> confidence:array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) lane:u32){
 if lane!=0u||arrayLength(&fault)<1u{return;}
 if arrayLength(&picks)<1u||arrayLength(&ring)<4112u||arrayLength(&token)<1u||arrayLength(&control)<3u||arrayLength(&confidence)<65u{atomicOr(&fault[0],0x10000000u);return;}
 let j=control[1];let pick=picks[0];let probability=bitcast<f32>(pick.y);
 if j>=7u||pick.x>=248320u||!(probability>=0.0&&probability<=1.0)||atomicLoad(&fault[0])!=0u{atomicOr(&fault[0],0x10000000u);token[0]=248320u;return;}
 confidence[64]=pick.y;ring[4096u+j]=i32(pick.x);ring[4104u+j]=bitcast<i32>(pick.y);token[0]=pick.x;
}
// END
