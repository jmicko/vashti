    // The compact slot is the logical cohort ordinal. The RAW slot retained in
    // cohort_slot is independently authenticated and may be resident sentinel8.
    if p.r1%4u!=0u || p.r5%4u!=0u || p.r5>arrayLength(&packed)*4u/cohort.base.r3 {
        atomicOr(&fault[0],128u);return;
    }
    let i=id.x;if i>=p.r1/4u {return;}
    let at=i*4u;let first=id.z*p.r5/4u;
    packed[first+i]=vec4<u32>(strata_gate(at),strata_gate(at+1u),strata_gate(at+2u),strata_gate(at+3u));
    packed[first+p.r1/4u+i]=vec4<u32>(strata_up(at),strata_up(at+1u),strata_up(at+2u),strata_up(at+3u));
