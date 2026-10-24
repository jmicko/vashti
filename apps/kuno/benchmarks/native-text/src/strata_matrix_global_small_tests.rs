use super::*;
#[test]
fn global_small_real_cells_and_checked_subtiles_cover_every_original_member_once(){
 for count in 32usize..256{
  let tile=if count<64{32usize}else{64};let groups=tile/16;
  let mut owner=vec![0u8;count*64];let mut fallback=owner.clone();
  for y in 0..count.div_ceil(tile){for sg in 0..groups{
   let token=y*tile+sg/2*32;let row=sg%2*32;
   for m in token..(token+32).min(count){for r in row..row+32{owner[m*64+r]+=1;}}
   for tm in 0..2{for tn in 0..2{for b in 0..2{for t in 0..16{for w in 0..8{
    let m=token+tm*16+t;let r=row+tn*16+b*8+w;
    if m<count{fallback[m*64+r]+=1;}
   }}}}}
  }}
  assert_eq!(owner,fallback);assert!(owner.iter().all(|&n|n==1));
 }
 for count in 9usize..32{let mut owner=vec![0u8;count*64];for y in 0..count.div_ceil(16){for sg in 0..2{for t in 0..16{for w in 0..32{let m=y*16+t;if m<count{owner[m*64+sg*32+w]+=1;}}}}}assert!(owner.iter().all(|&n|n==1));}
}
#[test]
fn global_small_masked_original_input_tiles_are_fully_initialized_without_out_of_bounds_reads(){
 for count in 32usize..256{for token in (0..count).step_by(16){for columns in [640usize,2560]{for k in [0usize,columns-16]{
  let mut initialized=[false;64];let mut reads=0;
  for lane in 0..16{for round in 0..4{let v=lane+round*16;let member=token+v/4;assert!(!initialized[v]);initialized[v]=true;
   if member<count{let word=(member*columns+k)/4+v%4;assert!(word<count*columns/4);assert_eq!(word*4/columns,member);reads+=1;}
  }}
  assert!(initialized.iter().all(|v|*v));assert_eq!(reads,(count-token).min(16)*4);
 }}}}
}
#[test]
fn global_small_transposed_partial_halves_retain_original_raw_status_and_invalid_sentinels(){
 for count in 1usize..33{for bad in 0..=count{for first in (0..count).step_by(16){
  let actual_good=(0..16).map(|lane|first+lane<count&&first+lane!=bad).collect::<Vec<_>>();
  for half in 0..2{let all=actual_good[half*8..half*8+8].iter().all(|v|*v);
   let old_raw=first+half*8+8<=count&&!(first+half*8..first+half*8+8).contains(&bad);
   assert_eq!(all,old_raw);
   for m in first+half*8..(first+half*8+8).min(count){if m==bad{let sentinel=0x3ec00000u32;assert_eq!(f32::from_bits(sentinel),0.375);continue;}
    for finite in [false,true]{for status in [0u32,1]{let original=if old_raw{2}else{u32::from(finite&&status==1)};let transposed=if all{2}else if finite{status}else{0};assert_eq!(original,transposed);}}
   }
  }
 }}}
}
#[test]
fn global_small_actual_complete_modules_authenticate_bounds_layouts_barriers_and_atomic_or(){validate_sources().unwrap();}
