var<workgroup> bits:array<atomic<u32>,128>;
var<workgroup> lists:array<vec2<u32>,256>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32,@builtin(subgroup_invocation_id) sl:u32,@builtin(subgroup_size) ss:u32,@builtin(subgroup_id) warp:u32){
 if arrayLength(&fault)<1u{return;}if !valid()||ss!=32u{atomicOr(&fault[0],0x10000000u);return;}if group.x>=NB||group.y>=p.dims.x{return;}let row=group.y;let block=group.x*4096u;let h=min(p.hist.x,p.hist.y);
 if h>0u{atomicStore(&bits[lane],0u);workgroupBarrier();for(var i=lane;i<h;i+=128u){let id=history[row*p.hist.x+p.hist.x-h+i];if id>=i32(block)&&id<min(i32(block+4096u),i32(NV)){let b=u32(id)-block;atomicOr(&bits[b>>5u],1u<<(b&31u));}}workgroupBarrier();}
 var values:array<f32,32>;for(var j=0u;j<32u;j++){let col=block+warp*1024u+j*32u+sl;var x=bitcast<f32>(0xff800000u);if col<NV{x=logits[row*NV+col];let b=col-block;if h>0u&&(atomicLoad(&bits[b>>5u])&(1u<<(b&31u)))!=0u{x=penalize(x,count_history(row,col));}if !finite(x){atomicOr(&fault[0],0x10000000u);x=bitcast<f32>(0xff800000u);}}values[j]=x;}
 var prev=vec2<u32>(0xffffffffu,0x7f800000u);
 for(var r=0u;r<p.dims.z;r++) {var even=sentinel();var odd=sentinel();for(var j=0u;j<32u;j+=2u){
  let id=block+warp*1024u+j*32u+sl;let x=values[j];let y=values[j+1u];let id1=id+32u;
  if id<NV&&(x<bitcast<f32>(prev.y)||(x==bitcast<f32>(prev.y)&&id>prev.x)){even=better(even,vec2<u32>(id,bitcast<u32>(x)));}
  if id1<NV&&(y<bitcast<f32>(prev.y)||(y==bitcast<f32>(prev.y)&&id1>prev.x)){odd=better(odd,vec2<u32>(id1,bitcast<u32>(y)));}
 }prev=reduce_pair(better(even,odd));if sl==0u{lists[warp*64u+r]=prev;}}
 workgroupBarrier();if warp==0u{var cursor=0u;
 // All128 threads reached the preceding barrier. Each lane keeps its own
 // list cursor; warp0 merges without shared cursor writes or partial-WG barriers.
 for(var r=0u;r<p.dims.z;r++){var entry=sentinel();var which=0xffffffffu;if sl<4u&&cursor<p.dims.z{entry=lists[sl*64u+cursor];which=sl;}let win=reduce_pair(entry);let list=subgroupMin(select(0xffffffffu,which,entry.x==win.x));if sl==0u{candidates[(row*NB+group.x)*p.dims.z+r]=win;}if sl==list{cursor++;}}
 }
}
