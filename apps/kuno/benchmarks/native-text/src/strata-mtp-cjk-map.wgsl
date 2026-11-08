// MIT Strata mtp.cpp: map_ids after compact argmax and row_top_prob.
@group(0) @binding(0) var<storage,read> ids:array<u32>;
@group(0) @binding(1) var<storage,read_write> selected:array<u32>;
@group(0) @binding(2) var<storage,read_write> fault:array<atomic<u32>>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) lane:u32) {
 if lane!=0u || arrayLength(&fault)<1u {return;}
 if arrayLength(&ids)<106299u || arrayLength(&selected)<1u {atomicOr(&fault[0],0x10000000u);return;}
 let compact=selected[0];
 if compact>=106299u || atomicLoad(&fault[0])!=0u {atomicOr(&fault[0],0x10000000u);selected[0]=248320u;return;}
 let token=ids[compact];
 if token>=248320u {atomicOr(&fault[0],0x10000000u);selected[0]=248320u;return;}
 selected[0]=token;
}
