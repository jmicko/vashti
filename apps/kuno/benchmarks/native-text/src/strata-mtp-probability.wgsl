// MIT Strata fb58e0 top probability. Warp parts use atomic bit storage instead
// of SYCL volatile float storage to give explicit device-wide publication.
@group(0) @binding(0) var<storage,read> logits:array<f32>;
@group(0) @binding(1) var<storage,read> selected:array<u32>;
@group(0) @binding(2) var<storage,read_write> scratch:array<atomic<u32>>;
@group(0) @binding(3) var<storage,read_write> probability:array<u32>;
@group(0) @binding(4) var<storage,read_write> fault:array<atomic<u32>>;
var<workgroup> last:u32;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) local:u32,@builtin(subgroup_id) sg:u32,@builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_size) width:u32){
 if arrayLength(&fault)<1u{return;}
 if group.x>=8u||width!=32u||arrayLength(&logits)<248320u||arrayLength(&selected)<1u||arrayLength(&scratch)<36u||arrayLength(&probability)<1u{atomicOr(&fault[0],0x10000000u);return;}
 if group.x==0u&&local==0u{probability[0]=0x7fc00000u;}
 let pick=selected[0];if pick>=248320u{atomicOr(&fault[0],0x10000000u);return;}
 let maximum=logits[pick];if !(abs(maximum)<=bitcast<f32>(0x7f7fffffu)){atomicOr(&fault[0],0x10000000u);return;}
 let vt=group.x*128u+sg*32u+lane;var sum=0.0;
 for(var i=vt;i<248320u;i+=1024u){let x=logits[i];if !(abs(x)<=bitcast<f32>(0x7f7fffffu)){atomicOr(&fault[0],0x10000000u);}else{sum+=exp(x-maximum);}}
 for(var offset=16u;offset>0u;offset/=2u){let other=subgroupShuffleXor(sum,offset);sum+=other;}
 if lane==0u{atomicStore(&scratch[4u+group.x*4u+sg],bitcast<u32>(sum));}
 // Naga lowers storageBarrier to Workgroup execution + DEVICE memory scope,
 // AcquireRelease|UniformMemory. Release parts before counter publication;
 // acquire after the last RMW before reading all32 independently stored parts.
 storageBarrier();
 if local==0u{last=select(0u,1u,atomicAdd(&scratch[0],1u)==7u);}
 workgroupBarrier();storageBarrier();
 if last!=0u&&local==0u{var total=0.0;for(var w=0u;w<32u;w++){total+=bitcast<f32>(atomicLoad(&scratch[4u+w]));}let p=1.0/total;
  if !(p>0.0&&p<=1.0)||atomicLoad(&fault[0])!=0u{atomicOr(&fault[0],0x10000000u);}else{probability[0]=bitcast<u32>(p);}atomicStore(&scratch[0],0u);
 }
}
