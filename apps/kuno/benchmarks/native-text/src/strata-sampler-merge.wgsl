//61*64 pairs plus selected/ex/cursors: <32KiB per row, independent of rows.
var<workgroup> lists:array<vec2<u32>,3904>;
var<workgroup> selected:array<vec2<u32>,64>;
var<workgroup> ex:array<f32,64>;
var<workgroup> ordered_sum:f32;
var<workgroup> ordered_cut:u32;
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_size) ss:u32){
 if !valid()||group.x>=p.dims.x||ss!=32u{return;}let row=group.x;let k=p.dims.z;
 for(var i=lane;i<NB*k;i+=32u){lists[i]=candidates[row*NB*k+i];}workgroupBarrier();
 // Exactly two list cursors per lane, kept in registers as upstream's
 // warp_merge_lists. No per-candidate shared cursor or workgroup barrier.
 var head:array<vec2<u32>,2>;var cursor:array<u32,2>;
 for(var m=0u;m<2u;m++){cursor[m]=0u;head[m]=sentinel();let list=lane+32u*m;if list<NB{head[m]=lists[list*k];}}
 for(var r=0u;r<k;r++){
  let win=reduce_pair(better(head[0],head[1]));if lane==0u{selected[r]=win;}
  if win.x<NV{for(var m=0u;m<2u;m++){if head[m].x==win.x{cursor[m]++;head[m]=sentinel();if cursor[m]<k{head[m]=lists[(lane+32u*m)*k+cursor[m]];}}}}
 }
 workgroupBarrier();
 // One full32-lane workgroup is the native tail warp. Entries/quotients are
 // shared among lanes; lane0 alone sums/scans in ascending selection order.
 var keep=k;
 var mx=bitcast<f32>(selected[0].y);
 for(var i=1u;i<k;i++){mx=max(mx,bitcast<f32>(selected[i].y));}
 if p.filters.x<1.0 {
  for(var i=lane;i<k;i+=32u){ex[i]=exp(bitcast<f32>(selected[i].y)-mx);}
  workgroupBarrier();
  if lane==0u{var sum=0.0;for(var i=0u;i<k;i++){sum+=ex[i];}ordered_sum=sum;}
  workgroupBarrier();
  for(var i=lane;i<k;i+=32u){ex[i]=ex[i]/ordered_sum;}
  workgroupBarrier();
  if lane==0u{var cut=k;var cum=0.0;for(var i=0u;i<k;i++){cum+=ex[i];if cum>=p.filters.x{cut=i+1u;break;}}ordered_cut=cut;}
  workgroupBarrier();keep=max(ordered_cut,min(p.hist.z,k));
  workgroupBarrier();
 }
 if p.filters.y>0.0{let threshold=bitcast<f32>(selected[0].y)+log(p.filters.y);for(var i=0u;i<keep;i++){if bitcast<f32>(selected[i].y)<threshold{keep=i;break;}}}
 let inv=1.0/p.filters.z;
 var smx=bitcast<f32>(selected[0].y)*inv;
 for(var i=1u;i<keep;i++){smx=max(smx,bitcast<f32>(selected[i].y)*inv);}
 for(var i=lane;i<keep;i+=32u){ex[i]=exp(bitcast<f32>(selected[i].y)*inv-smx);}
 workgroupBarrier();
 if lane==0u{var sum=0.0;for(var i=0u;i<keep;i++){sum+=ex[i];}ordered_sum=sum;}
 workgroupBarrier();
 for(var i=lane;i<keep;i+=32u){ex[i]=ex[i]/ordered_sum;}
 workgroupBarrier();
 if lane==0u{
  let u=draw(row);var cum=0.0;var pi=select(0u,keep-1u,keep>0u);
  for(var i=0u;i<keep;i++){cum+=ex[i];if u<cum{pi=i;break;}}
  let pick=selected[pi].x;var probability=1.0;if keep>0u{probability=ex[pi];}
  if pick>=NV||keep>0u&&(!finite(ordered_sum)||ordered_sum<=0.0){atomicOr(&fault[0],0x10000000u);}
  output[p.destination.x+row*p.destination.y]=select(NV,pick,atomicLoad(&fault[0])==0u);
  output[p.destination.x+row*p.destination.y+1u]=bitcast<u32>(probability);
 }
}
