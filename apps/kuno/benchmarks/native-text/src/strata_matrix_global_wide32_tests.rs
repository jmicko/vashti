use super::*;
#[test]
fn wide32_genuine_subgroup_outputs_and_source_spans_cover_original_full_tiles(){
 for rows in [128usize,256,1280,2560]{let mut seen=vec![0u8;128*rows];
 for x in 0..rows/128{for sg in 0..16{for wb in 0..4{for tb in 0..2{for w in 0..8{for t in 0..16{
 let token=sg/4*32+tb*16+t;let row=x*128+sg%4*32+wb*8+w;seen[token*rows+row]+=1;
 }}}}}}assert!(seen.iter().all(|&v|v==1));}
 for cols in [64usize,320,2560,10240]{for sg in 0..16{for k in (0..cols).step_by(16){for t in 0..32{assert!((sg/4*32+t)*cols+k+15<128*cols);}for w in 0..32{assert!((sg%4*32+w)*cols+k+15<128*cols);}}}}
}
#[test]
fn wide32_chronological_single_a_lifetime_keeps_original_prefix_and_capabilities(){
 for cols in [64usize,320,640,2560,10240]{let mut cs=[Vec::new(),Vec::new(),Vec::new(),Vec::new(),Vec::new(),Vec::new(),Vec::new(),Vec::new()];for k in (0..cols).step_by(16){for w in 0..4{cs[w*2].push(k);cs[w*2+1].push(k);}}for c in cs{assert_eq!(c,(0..cols).step_by(16).collect::<Vec<_>>());}}
 for count in [128usize,256,512,4096]{assert!(global_wide32_geometry(count,2560,128,128,49152,512,(512,1,1),16,32));}
 for count in [0usize,127,129,257]{assert!(!global_wide32_geometry(count,2560,128,128,49152,512,(512,1,1),16,32));}
 assert!(gr_down_shape(true,10240,320,0,0,320));
}
#[test]
fn wide32_actual_dense_modules_prove_global_operands_column_stores_and_lifetime_order(){validate_global_wide32_sources().unwrap();}
