var<workgroup> bits:array<atomic<u32>,BITMAP_WORDS>;
var<workgroup> winners:array<vec2<u32>,32>;
@compute @workgroup_size(1024)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32,@builtin(subgroup_invocation_id) sl:u32,@builtin(subgroup_size) ss:u32,@builtin(subgroup_id) sg:u32){
 if arrayLength(&fault)<1u{return;}if !valid()||ss!=32u{atomicOr(&fault[0],0x10000000u);return;}if group.x>=p.dims.x{return;}let row=group.x;let h=min(p.hist.x,p.hist.y);
 if BITMAP_WORDS==1u&&h>0u {atomicOr(&fault[0],0x10000000u);return;}
 if h>0u {
  for(var i=lane;i<7760u;i+=1024u){atomicStore(&bits[i],0u);}workgroupBarrier();
  for(var i=lane;i<h;i+=1024u){let id=history[row*p.hist.x+p.hist.x-h+i];if id>=0&&id<i32(NV){atomicOr(&bits[u32(id)>>5u],1u<<(u32(id)&31u));}}workgroupBarrier();
 }
 var best=sentinel();for(var col=lane;col<NV;col+=1024u){var x=logits[row*NV+col];if h>0u&&(atomicLoad(&bits[col>>5u])&(1u<<(col&31u)))!=0u{x=penalize(x,count_history(row,col));}if !finite(x){atomicOr(&fault[0],0x10000000u);}else{best=better(best,vec2<u32>(col,bitcast<u32>(x)));}}
 let win=reduce_pair(best);if sl==0u{winners[sg]=win;}workgroupBarrier();storageBarrier();
 if sg==0u{let answer=reduce_pair(winners[sl]);if sl==0u{output[p.destination.x+row*p.destination.y]=select(NV,answer.x,atomicLoad(&fault[0])==0u);output[p.destination.x+row*p.destination.y+1u]=bitcast<u32>(1.0);}}
}
