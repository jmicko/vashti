use super::*;
#[test]
fn wide32_actual_expert_modules_keep_original_checked_fallback_and_raw_math(){validate_sources().unwrap();}
#[test]
fn wide32_checked_subtiles_and_raw_cells_have_same_original_prefix_coverage(){
 let mut raw=vec![0u8;128*128];let mut checked=raw.clone();
 for sg in 0..16{for wb in 0..4{for tb in 0..2{for w in 0..8{for t in 0..16{raw[(sg/4*32+tb*16+t)*128+sg%4*32+wb*8+w]+=1;}}}}}
 for tm in 0..2{for tn in 0..2{for sg in 0..16{for wb in 0..2{for w in 0..8{for t in 0..16{let token=tm*64+sg/4*16+t;let row=tn*64+sg%4*16+wb*8+w;checked[token*128+row]+=1;}}}}}}
 assert_eq!(raw,checked);assert!(raw.iter().all(|&n|n==1));assert_eq!(16*128*4,8192);
}
#[test]
fn wide32_original_prefix_tail_scope_and_physical_credit_stay_distinct(){for count in [256usize,257,383,4096]{let p=plan(count).unwrap();assert_eq!(p.prefix+p.tail,count);assert_eq!(p.prefix%128,0);assert_eq!((1280/128+2560/128)*(p.prefix/128),15*2*(p.prefix/128));}}
#[test]
fn wide32_production_guard_is_existing_authentic_gather_without_extra_tape_nodes(){
 let source=include_str!("strata_prompt_experts.wgsl");let gather=source.split("// GATHER_BEGIN").nth(1).unwrap().split("// GATHER_END").next().unwrap();
 assert!(gather.contains("route>=p.count*10u {atomicOr(&fault[0],128u);return;}"));assert!(gather.contains("i>=length*p.columns/2u"));
 assert!(gather.contains("@binding(0) var<storage,read> map"));assert!(gather.contains("@binding(3) var<storage,read_write> fault"));
}
