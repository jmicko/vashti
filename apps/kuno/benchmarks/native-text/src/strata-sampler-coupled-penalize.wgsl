// MIT Strata fb58e0 sampler.dp.cpp::coupled_penalize_kernel.
// The original logits owner is mutated ONCE per distinct history token, before
// fixed64 split selection. The inverse-ID bridge already selected this domain.
var<workgroup> seen:array<atomic<u32>,BITMAP_WORDS>;
@compute @workgroup_size(1024)
fn main(@builtin(local_invocation_index) local:u32,@builtin(subgroup_size) width:u32){
 if arrayLength(&fault)<1u{return;}
 if !valid()||width!=32u{atomicOr(&fault[0],0x10000000u);return;}
 let h=min(p.hist.x,p.hist.y);if h==0u{return;}
 for(var w=local;w<(NV+31u)/32u;w+=1024u){atomicStore(&seen[w],0u);}
 workgroupBarrier();
 for(var i=local;i<h;i+=1024u){
  let id=history[p.hist.x-h+i];if id<0||id>=i32(NV){continue;}
  let v=u32(id);let bit=1u<<(v&31u);
  if (atomicOr(&seen[v>>5u],bit)&bit)!=0u{continue;}
  let value=penalize(logits[v],count_history(0u,v));
  if !finite(value){atomicOr(&fault[0],0x10000000u);}else{logits[v]=value;}
 }
}
