// Same immutable INFO owner, erased by the original transient uniform lifecycle.
struct CohortInfo {base:Params, records:array<vec4<u32>,8>}
fn cohort_begin(slot:u32)->bool {
    if cohort.base.r3<1u || cohort.base.r3>8u || slot>=cohort.base.r3 {return false;}
    if cohort.base.r0<1u || cohort.base.r0>1048576u || cohort.base.r1!=cohort.base.r0 || cohort.base.r2<=2u*cohort.base.r0 || cohort.base.r2>1048576u || cohort.base.group>512u-cohort.base.r3 {atomicOr(&fault[0],128u);return false;}
    let v=cohort.records[slot];let tile=16u;let padded=((v.y+tile-1u)/tile)*tile;
    if v.x!=cohort.base.group+slot || v.y<32u || v.y>=256u || v.z%16u!=0u || v.z>4096u || padded>4096u-min(v.z,4096u) || v.w>8u {atomicOr(&fault[0],128u);return false;}
    p=cohort.base;p.group=v.x;cohort_prefix=v.z;cohort_slot=v.w;
    p.r0=select(v.w,0u,v.w==8u)*cohort.base.r2;p.r1=cohort.base.r0;p.r2=2u*cohort.base.r0;
    p.r3=slot*2457600u;p.r4=p.r3+1638400u;p.r5=cohort.base.r2;
    if !valid() || map[516u+p.group]!=v.y {atomicOr(&fault[0],128u);return false;}
    return true;
}
