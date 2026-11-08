// Weights-first global small/medium arithmetic with authentic partial input.
// No new owners, route cache, precision, phase or capacity. Original consumers
// and exact8-member failure/status semantics remain mandatory.
use super::*;
#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn publish(
    acc: &cmma::Matrix<f32>, scratch: &mut SharedMemory<f32>,
    output: &mut Array<f32>, map: &Array<u32>, status: &mut Array<u32>,
    fault: &mut Array<Atomic<u32>>, member0: usize, row0: usize,
    count: usize, gu_prefix: usize, group: usize, #[comptime] down: bool,
) {
    let sg = usize::cast_from(PLANE_POS);
    let lane = usize::cast_from(UNIT_POS_PLANE);
    let prefix = usize::cast_from(map[1028 + group]);
    let rows = comptime!(if down {2560usize} else {1280usize});
    // Each original8-member half is independent, including partial tiles.
    let mut good = false;
    if member0 + lane < count && row0 + 8 <= rows {
        let route = usize::cast_from(map[1540 + prefix + member0 + lane]);
        good = route < usize::cast_from(map[0]) * 10;
    }
    let low_good = plane_all(lane >= 8 || good);
    let high_good = plane_all(lane < 8 || good);
    if low_good && high_good {
        let mut first = (gu_prefix + member0) * 1280 + row0;
        let mut stride = 1280u32;
        if comptime!(down) {first = (prefix + member0) * 2560 + row0; stride = 2560u32;}
        cmma::store(&mut output.slice_mut(first, output.len()), acc, stride, cmma::MatrixLayout::ColMajor);
        if comptime!(down) && row0 == 0 {
            let route = usize::cast_from(map[1540 + prefix + member0 + lane]);
            status[comptime!(crate::strata_prompt::groups::ROUTES) + route] = 2;
        }
    } else {
        cmma::store(&mut scratch.slice_mut(sg * 128, sg * 128 + 128), acc, 8, cmma::MatrixLayout::ColMajor);
        sync_plane();
        #[unroll]
        for round in 0..8usize {
            let i = lane + round * 16;
            let member = member0 + i / 8;
            let row = row0 + i % 8;
            if member < count && row < rows {
                let route = usize::cast_from(map[1540 + prefix + member]);
                if route < usize::cast_from(map[0]) * 10 {
                    let value = scratch[sg * 128 + i];
                    let mut first = (gu_prefix + member) * 1280;
                    if comptime!(down) {first = (prefix + member) * 2560;}
                    let raw_half = (i / 8 < 8 && low_good) || (i / 8 >= 8 && high_good);
                    if raw_half {
                        output[first + row] = value;
                        if comptime!(down) && row == 0 {
                            status[comptime!(crate::strata_prompt::groups::ROUTES) + route] = 2;
                        }
                    } else {
                        let finite = value.abs() <= f32::MAX;
                        if finite {output[first + row] = value;} else {output[first + row] = 0.0; fault[0].fetch_or(128);}
                        if comptime!(down) && row == 0 {
                            let mut valid = 0u32;
                            if finite && status[route] == 1 {valid = 1;}
                            status[comptime!(crate::strata_prompt::groups::ROUTES) + route] = valid;
                        }
                    }
                } else {fault[0].fetch_or(128);}
            }
        }
        sync_plane();
    }
}


#[cube]
#[allow(clippy::too_many_arguments)]
fn checked_product(
 input:&Array<Vector<f16,Const<4>>>, weight:&Array<Vector<f16,Const<4>>>,
 stage:&mut SharedMemory<Vector<f16,Const<4>>>, c0:&cmma::Matrix<f32>, c1:&cmma::Matrix<f32>,
 token:usize, row:usize, count:usize, columns:usize, weight_first:usize,
 input_member_base:usize, #[comptime] padded_input:bool,
){
 let a=unsafe{cmma::Matrix::<f16>::uninitialized(cmma::MatrixIdent::A,8usize,16usize,16usize,cmma::MatrixLayout::RowMajor)};
 let b=unsafe{cmma::Matrix::<f16>::uninitialized(cmma::MatrixIdent::B,8usize,16usize,16usize,cmma::MatrixLayout::ColMajor)};
 let sg=usize::cast_from(PLANE_POS); let lane=usize::cast_from(UNIT_POS_PLANE);
 let mut k=0usize;
 while k<columns {
  if comptime!(padded_input) || token+16<=count {
   let first=((input_member_base+token)*columns+k)/4;
   cmma::load(&b,&input.to_slice().slice(first,input.len()),u32::cast_from(columns));
  } else {
   // Real original input only. Initialize EVERY word of this SG's private
   //16-member K16 tile before reading it, including inactive padded cells.
   #[unroll]
   for round in 0..4usize {
    let v=lane+round*16;let member=token+v/4;
    let mut x=Vector::<f16,Const<4>>::new(f16::new(0.0));
    if member<count {x=input[((input_member_base+member)*columns+k)/4+v%4];}
    stage[sg*64+v]=x;
   }
   sync_plane();
   cmma::load(&b,&stage.to_slice().slice(sg*64,sg*64+64),16u32);
  }
  let af=(weight_first+row*columns+k)/4;
  cmma::load(&a,&weight.to_slice().slice(af,weight.len()),u32::cast_from(columns));
  cmma::execute::<f16,f16,f32,f32>(&a,&b,c0,c0);
  cmma::load(&a,&weight.to_slice().slice(af+8*columns/4,weight.len()),u32::cast_from(columns));
  cmma::execute::<f16,f16,f32,f32>(&a,&b,c1,c1);
  if !comptime!(padded_input) && token+16>count {sync_plane();}
  k+=16;
 }
}

#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn product(
 input:&Array<Vector<f16,Const<4>>>,weight:&Array<Vector<f16,Const<4>>>,output:&mut Array<f32>,
 map:&Array<u32>,status:&mut Array<u32>,fault:&mut Array<Atomic<u32>>,
 columns:usize,rows:usize,count:usize,weight_first:usize,group:usize,
 input_member_base:usize,gu_prefix:usize,#[comptime] down:bool,#[comptime] tile:usize,
){
 let sg=usize::cast_from(PLANE_POS);let lane=usize::cast_from(UNIT_POS_PLANE);
 let token=usize::cast_from(CUBE_POS_Y)*comptime!(tile)+(sg/2)*comptime!(if tile==16{16usize}else{32usize});
 let row=usize::cast_from(CUBE_POS_X)*64+(sg%2)*32;
 // There are NO workgroup barriers. A fully inactive SG owns no real cell.
 if token>=count {terminate!();}
 let prefix=usize::cast_from(map[1028+group]);let max_route=usize::cast_from(map[0])*10;
 let mut good=false;
 if token+comptime!(if tile==16{16usize}else{32usize})<=count {
  let low=usize::cast_from(map[1540+prefix+token+lane]);
  good=low<max_route;
  if comptime!(tile!=16) {let high=usize::cast_from(map[1540+prefix+token+16+lane]);good=good&&high<max_route;}
 }
 if plane_all(good) {
  let c00=cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator,8usize,16usize,16usize,cmma::MatrixLayout::Undefined,0.0);
  let c01=if comptime!(tile!=16){cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator,8usize,16usize,16usize,cmma::MatrixLayout::Undefined,0.0)}else{c00};
  let c10=cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator,8usize,16usize,16usize,cmma::MatrixLayout::Undefined,0.0);
  let c11=if comptime!(tile!=16){cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator,8usize,16usize,16usize,cmma::MatrixLayout::Undefined,0.0)}else{c10};
  let c20=cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator,8usize,16usize,16usize,cmma::MatrixLayout::Undefined,0.0);
  let c21=if comptime!(tile!=16){cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator,8usize,16usize,16usize,cmma::MatrixLayout::Undefined,0.0)}else{c20};
  let c30=cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator,8usize,16usize,16usize,cmma::MatrixLayout::Undefined,0.0);
  let c31=if comptime!(tile!=16){cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator,8usize,16usize,16usize,cmma::MatrixLayout::Undefined,0.0)}else{c30};
  if comptime!(tile!=16) {
   crate::strata_matrix::native_wide::global_wide32::compute::<f16>(input,weight,&c00,&c01,&c10,&c11,&c20,&c21,&c30,&c31,token,row,columns,weight_first);
  } else {
   let a=unsafe{cmma::Matrix::<f16>::uninitialized(cmma::MatrixIdent::A,8usize,16usize,16usize,cmma::MatrixLayout::RowMajor)};
   let b=unsafe{cmma::Matrix::<f16>::uninitialized(cmma::MatrixIdent::B,8usize,16usize,16usize,cmma::MatrixLayout::ColMajor)};
   let mut k=0usize;
   while k<columns {
    let bf=((input_member_base+token)*columns+k)/4;
    cmma::load(&b,&input.to_slice().slice(bf,input.len()),u32::cast_from(columns));
    cmma::load(&a,&weight.to_slice().slice((weight_first+(row+0)*columns+k)/4,weight.len()),u32::cast_from(columns));
    cmma::execute::<f16,f16,f32,f32>(&a,&b,&c00,&c00);
    cmma::load(&a,&weight.to_slice().slice((weight_first+(row+8)*columns+k)/4,weight.len()),u32::cast_from(columns));
    cmma::execute::<f16,f16,f32,f32>(&a,&b,&c10,&c10);
    cmma::load(&a,&weight.to_slice().slice((weight_first+(row+16)*columns+k)/4,weight.len()),u32::cast_from(columns));
    cmma::execute::<f16,f16,f32,f32>(&a,&b,&c20,&c20);
    cmma::load(&a,&weight.to_slice().slice((weight_first+(row+24)*columns+k)/4,weight.len()),u32::cast_from(columns));
    cmma::execute::<f16,f16,f32,f32>(&a,&b,&c30,&c30);
    k+=16;
   }
  }
  let mut first=(gu_prefix+token)*rows+row;
  if comptime!(down){first=(prefix+token)*rows+row;}
  cmma::store(&mut output.slice_mut(first+0*rows+0,output.len()),&c00,u32::cast_from(rows),cmma::MatrixLayout::ColMajor);
  if comptime!(tile!=16){cmma::store(&mut output.slice_mut(first+16*rows+0,output.len()),&c01,u32::cast_from(rows),cmma::MatrixLayout::ColMajor);}
  cmma::store(&mut output.slice_mut(first+0*rows+8,output.len()),&c10,u32::cast_from(rows),cmma::MatrixLayout::ColMajor);
  if comptime!(tile!=16){cmma::store(&mut output.slice_mut(first+16*rows+8,output.len()),&c11,u32::cast_from(rows),cmma::MatrixLayout::ColMajor);}
  cmma::store(&mut output.slice_mut(first+0*rows+16,output.len()),&c20,u32::cast_from(rows),cmma::MatrixLayout::ColMajor);
  if comptime!(tile!=16){cmma::store(&mut output.slice_mut(first+16*rows+16,output.len()),&c21,u32::cast_from(rows),cmma::MatrixLayout::ColMajor);}
  cmma::store(&mut output.slice_mut(first+0*rows+24,output.len()),&c30,u32::cast_from(rows),cmma::MatrixLayout::ColMajor);
  if comptime!(tile!=16){cmma::store(&mut output.slice_mut(first+16*rows+24,output.len()),&c31,u32::cast_from(rows),cmma::MatrixLayout::ColMajor);}
  // C fragments are dead before original pending-status publication.
  if comptime!(down) && row==0 {
   let r=usize::cast_from(map[1540+prefix+token+lane]);
   status[comptime!(crate::strata_prompt::groups::ROUTES)+r]=2;
   if comptime!(tile!=16) {let r=usize::cast_from(map[1540+prefix+token+16+lane]);status[comptime!(crate::strata_prompt::groups::ROUTES)+r]=2;}
  }
 } else {
  let mut input_stage=SharedMemory::<Vector<f16,Const<4>>>::new(comptime!(if tile==64{256usize}else{128usize}));
  let mut scratch=SharedMemory::<f32>::new(comptime!(if tile==64{512usize}else{256usize}));
  // Every checked subtile is owned by the SAME subgroup as its raw32x32
  // (or16x32) cell. Never use old whole-workgroup64x64 fallback coordinates.
  for tm in 0..comptime!(if tile==16{1usize}else{2usize}) {
   for tn in 0..2usize {
    let t=token+tm*16;let r=row+tn*16;
    if t<count {
     let c0=cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator,8usize,16usize,16usize,cmma::MatrixLayout::Undefined,0.0);
     let c1=cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator,8usize,16usize,16usize,cmma::MatrixLayout::Undefined,0.0);
     checked_product(input,weight,&mut input_stage,&c0,&c1,t,r,count,columns,weight_first,input_member_base,comptime!(tile==16));
     publish(&c0,&mut scratch,output,map,status,fault,t,r,count,gu_prefix,group,down);
     publish(&c1,&mut scratch,output,map,status,fault,t,r+8,count,gu_prefix,group,down);
    }
   }
  }
 }
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn expert_global_macro32_gu(input:&Array<Vector<f16,Const<4>>>,weight:&Array<Vector<f16,Const<4>>>,output:&mut Array<f32>,map:&Array<u32>,status:&mut Array<u32>,fault:&mut Array<Atomic<u32>>,columns:usize,rows:usize,count:usize,weight_first:usize,group:usize){
    // Workgroup-uniform guards precede every map-derived index and barrier.
    let mut header_good = false;
    if group < 512 && map.len() == 65536 {
        let total = usize::cast_from(map[0]);
        if total >= 1 && total <= comptime!(crate::strata_prompt::groups::CAPACITY) && group < usize::cast_from(map[3]) {
            let prefix = usize::cast_from(map[1028 + group]);
            header_good = map[1] == 0
                && usize::cast_from(map[2]) == total
                && map[3] <= 512
                && map[4 + group] < 512
                && usize::cast_from(map[516 + group]) == count
                && count >= 32
                && count < 64
                && count <= total
                && prefix <= total * 10
                && count <= total * 10 - prefix;
        }
    }
    if !header_good
        || columns != 2560
        || rows != 1280
    {
        if UNIT_POS == 0 {
            fault[0].fetch_or(128);
        }
        terminate!();
    }
    if CUBE_DIM_X!=32u32 || CUBE_DIM_Y!=1u32 || CUBE_DIM_Z!=1u32 {if UNIT_POS==0{fault[0].fetch_or(128);}terminate!();}
    if usize::cast_from(CUBE_POS_X)>=rows/64 || usize::cast_from(CUBE_POS_Y)*32>=count {terminate!();}
    product(input,weight,output,map,status,fault,columns,rows,count,weight_first,group,0usize,0usize,false,32usize);
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn expert_global_macro32_down(input:&Array<Vector<f16,Const<4>>>,weight:&Array<Vector<f16,Const<4>>>,output:&mut Array<f32>,map:&Array<u32>,status:&mut Array<u32>,fault:&mut Array<Atomic<u32>>,columns:usize,rows:usize,count:usize,weight_first:usize,group:usize){
    // Workgroup-uniform guards precede every map-derived index and barrier.
    let mut header_good = false;
    if group < 512 && map.len() == 65536 {
        let total = usize::cast_from(map[0]);
        if total >= 1 && total <= comptime!(crate::strata_prompt::groups::CAPACITY) && group < usize::cast_from(map[3]) {
            let prefix = usize::cast_from(map[1028 + group]);
            header_good = map[1] == 0
                && usize::cast_from(map[2]) == total
                && map[3] <= 512
                && map[4 + group] < 512
                && usize::cast_from(map[516 + group]) == count
                && count >= 32
                && count < 64
                && count <= total
                && prefix <= total * 10
                && count <= total * 10 - prefix;
        }
    }
    if !header_good
        || columns != 640
        || rows != 2560
    {
        if UNIT_POS == 0 {
            fault[0].fetch_or(128);
        }
        terminate!();
    }
    if CUBE_DIM_X!=32u32 || CUBE_DIM_Y!=1u32 || CUBE_DIM_Z!=1u32 {if UNIT_POS==0{fault[0].fetch_or(128);}terminate!();}
    if usize::cast_from(CUBE_POS_X)>=rows/64 || usize::cast_from(CUBE_POS_Y)*32>=count {terminate!();}
    product(input,weight,output,map,status,fault,columns,rows,count,weight_first,group,0usize,0usize,true,32usize);
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn expert_global_macro64_gu(input:&Array<Vector<f16,Const<4>>>,weight:&Array<Vector<f16,Const<4>>>,output:&mut Array<f32>,map:&Array<u32>,status:&mut Array<u32>,fault:&mut Array<Atomic<u32>>,columns:usize,rows:usize,count:usize,weight_first:usize,group:usize){
    // Workgroup-uniform guards precede every map-derived index and barrier.
    let mut header_good = false;
    if group < 512 && map.len() == 65536 {
        let total = usize::cast_from(map[0]);
        if total >= 1 && total <= comptime!(crate::strata_prompt::groups::CAPACITY) && group < usize::cast_from(map[3]) {
            let prefix = usize::cast_from(map[1028 + group]);
            header_good = map[1] == 0
                && usize::cast_from(map[2]) == total
                && map[3] <= 512
                && map[4 + group] < 512
                && usize::cast_from(map[516 + group]) == count
                && count >= 64
                && count < 256
                && count <= total
                && prefix <= total * 10
                && count <= total * 10 - prefix;
        }
    }
    if !header_good
        || columns != 2560
        || rows != 1280
    {
        if UNIT_POS == 0 {
            fault[0].fetch_or(128);
        }
        terminate!();
    }
    if CUBE_DIM_X!=64u32 || CUBE_DIM_Y!=1u32 || CUBE_DIM_Z!=1u32 {if UNIT_POS==0{fault[0].fetch_or(128);}terminate!();}
    if usize::cast_from(CUBE_POS_X)>=rows/64 || usize::cast_from(CUBE_POS_Y)*64>=count {terminate!();}
    product(input,weight,output,map,status,fault,columns,rows,count,weight_first,group,0usize,0usize,false,64usize);
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn expert_global_macro64_down(input:&Array<Vector<f16,Const<4>>>,weight:&Array<Vector<f16,Const<4>>>,output:&mut Array<f32>,map:&Array<u32>,status:&mut Array<u32>,fault:&mut Array<Atomic<u32>>,columns:usize,rows:usize,count:usize,weight_first:usize,group:usize){
    // Workgroup-uniform guards precede every map-derived index and barrier.
    let mut header_good = false;
    if group < 512 && map.len() == 65536 {
        let total = usize::cast_from(map[0]);
        if total >= 1 && total <= comptime!(crate::strata_prompt::groups::CAPACITY) && group < usize::cast_from(map[3]) {
            let prefix = usize::cast_from(map[1028 + group]);
            header_good = map[1] == 0
                && usize::cast_from(map[2]) == total
                && map[3] <= 512
                && map[4 + group] < 512
                && usize::cast_from(map[516 + group]) == count
                && count >= 64
                && count < 256
                && count <= total
                && prefix <= total * 10
                && count <= total * 10 - prefix;
        }
    }
    if !header_good
        || columns != 640
        || rows != 2560
    {
        if UNIT_POS == 0 {
            fault[0].fetch_or(128);
        }
        terminate!();
    }
    if CUBE_DIM_X!=64u32 || CUBE_DIM_Y!=1u32 || CUBE_DIM_Z!=1u32 {if UNIT_POS==0{fault[0].fetch_or(128);}terminate!();}
    if usize::cast_from(CUBE_POS_X)>=rows/64 || usize::cast_from(CUBE_POS_Y)*64>=count {terminate!();}
    product(input,weight,output,map,status,fault,columns,rows,count,weight_first,group,0usize,0usize,true,64usize);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_expert(client:&ComputeClient<WgpuRuntime>,h:[Handle;6],count:usize,columns:usize,rows:usize,weight_first:usize,group:usize,down:bool)->Result<bool>{
 if !(32..256).contains(&count){return Ok(false);}
 compatible(client,false)?;
 if group>=512 || u32::try_from(weight_first).is_err() || columns!=if down{640}else{2560} || rows!=if down{2560}else{1280}
  || !macro_compatible(client,&h[0],&h[1],count,rows,columns,weight_first)
  || h.iter().any(|v|v.stream!=h[0].stream)
  || (count*columns*2)as u64>h[0].size_in_used()
  || weight_first.checked_add(rows*columns).is_none_or(|n|(n as u64)*2>h[1].size_in_used())
  || h[2].size_in_used()<(if down{crate::strata_prompt::groups::PART_BYTES}else{(count*1280*4)as u64})
  || h[3].size_in_used()!=(crate::strata_prompt::groups::MAP_WORDS*4)as u64
  || h[4].size_in_used()!=(crate::strata_prompt::groups::STATUS_WORDS*4)as u64 || h[5].size_in_used()!=4
 {return Err(Error::InvalidBufferLength);}
 let tile=if count<64{32usize}else{64usize};
 let hw=&client.properties().hardware;
 if rows/64>hw.max_cube_count.0 as usize || count.div_ceil(tile)>hw.max_cube_count.1 as usize || hw.max_cube_count.2<1{return Err(Error::LimitExceeded("global small grid"));}
 let [i,w,o,m,s,f]=h;
 let ia=macro_half_arg(i)?;let wa=macro_half_arg(w)?;
 let raw=|v:Handle|crate::resident_util::raw_arg(v.clone(),v.size_in_used()as usize/4);
 let oa=raw(o)?;let ma=raw(m)?;let sa=raw(s)?;let fa=raw(f)?;
 let grid=CubeCount::Static((rows/64)as u32,count.div_ceil(tile)as u32,1);
 let dim=CubeDim::new_1d(if tile==64{64}else{32});
 unsafe {match(tile,down){
 (32,false)=>expert_global_macro32_gu::launch_unchecked::<WgpuRuntime>(client,grid,dim,ia,wa,oa,ma,sa,fa,columns,rows,count,weight_first,group),
 (32,true)=>expert_global_macro32_down::launch_unchecked::<WgpuRuntime>(client,grid,dim,ia,wa,oa,ma,sa,fa,columns,rows,count,weight_first,group),
 (64,false)=>expert_global_macro64_gu::launch_unchecked::<WgpuRuntime>(client,grid,dim,ia,wa,oa,ma,sa,fa,columns,rows,count,weight_first,group),
 (64,true)=>expert_global_macro64_down::launch_unchecked::<WgpuRuntime>(client,grid,dim,ia,wa,oa,ma,sa,fa,columns,rows,count,weight_first,group),
 _=>unreachable!(),}}
 crate::strata_prefill_profile::count(if tile==64{"expert_global_small_macro64_products"}else{"expert_global_small_macro32_products"},1);
 if tile==64{crate::strata_prefill_profile::count("expert_macro64_products",1);}
 crate::strata_prefill_profile::count("expert_global_small_macro_workgroups",(rows/64*count.div_ceil(tile))as u64);
 Ok(true)
}

pub(crate) fn definitions()->Vec<(&'static str,KernelDefinition)>{
 let mut out=Vec::new();
 for tile in [32usize,64] {for down in [false,true] {
  let mut b=KernelBuilder::default();cubecl::ir::AddressType::U32.register(&mut b.scope);
  let t=Vector::<f16,Const<4>>::as_type(&b.scope);
  let i=b.input_array(t);let w=b.input_array(t);let o=b.output_array(f32::as_type(&b.scope));let m=b.input_array(u32::as_type(&b.scope));let s=b.output_array(u32::as_type(&b.scope));let f=b.output_array(cubecl::ir::StorageType::Atomic(ElemType::UInt(cubecl::ir::UIntKind::U32)).into());
  let a:[cubecl::ir::ManagedVariable;5]=std::array::from_fn(|_|b.scalar(u32::as_type(&b.scope).storage_type()));
  match(tile,down){
  (32,false)=>expert_global_macro32_gu::expand(&mut b.scope,i.into(),w.into(),o.into(),m.into(),s.into(),f.into(),a[0].clone().into(),a[1].clone().into(),a[2].clone().into(),a[3].clone().into(),a[4].clone().into()),
  (32,true)=>expert_global_macro32_down::expand(&mut b.scope,i.into(),w.into(),o.into(),m.into(),s.into(),f.into(),a[0].clone().into(),a[1].clone().into(),a[2].clone().into(),a[3].clone().into(),a[4].clone().into()),
  (64,false)=>expert_global_macro64_gu::expand(&mut b.scope,i.into(),w.into(),o.into(),m.into(),s.into(),f.into(),a[0].clone().into(),a[1].clone().into(),a[2].clone().into(),a[3].clone().into(),a[4].clone().into()),
  (64,true)=>expert_global_macro64_down::expand(&mut b.scope,i.into(),w.into(),o.into(),m.into(),s.into(),f.into(),a[0].clone().into(),a[1].clone().into(),a[2].clone().into(),a[3].clone().into(),a[4].clone().into()),
  _=>unreachable!(),}
  let name=match(tile,down){(32,false)=>"global_small_macro32_gu",(32,true)=>"global_small_macro32_down",(64,false)=>"global_small_macro64_gu",_=>"global_small_macro64_down"};
  assert!(b.scope.pop_errors().is_empty());out.push((name,b.build(KernelSettings::default().cube_dim(CubeDim::new_1d(if tile==64{64}else{32})))));
 }}
 out
}

mod validation{include!("strata_matrix_global_small_validation.rs");}
pub(crate) use validation::validate_sources;
#[cfg(test)]mod tests{include!("strata_matrix_global_small_tests.rs");}

pub(crate) fn profile_names()->[&'static str;5]{[
core::any::type_name::<expert_global_macro32_gu::ExpertGlobalMacro32Gu<WgpuRuntime>>(),
core::any::type_name::<expert_global_macro32_down::ExpertGlobalMacro32Down<WgpuRuntime>>(),
core::any::type_name::<expert_global_macro64_gu::ExpertGlobalMacro64Gu<WgpuRuntime>>(),
core::any::type_name::<expert_global_macro64_down::ExpertGlobalMacro64Down<WgpuRuntime>>(),
core::any::type_name::<crate::strata_matrix::cohorts::global_small::expert_global_cohort::ExpertGlobalCohort<WgpuRuntime>>()
]}

mod probe{include!("strata_matrix_global_small_probe.rs");}
pub(crate) use probe::cases as public_numeric;
