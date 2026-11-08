    if p.r1%4u!=0u || p.r5%4u!=0u || p.r5>arrayLength(&packed)*4u/cohort.base.r3 {
        atomicOr(&fault[0],128u);return;
    }
    let i=id.x;if i>=(p.r5-p.r2)/4u {return;}
    let at=i*4u;let first=(id.z*p.r5+p.r2)/4u;
    packed[first+i]=vec4<u32>(strata_gate(at),strata_gate(at+1u),strata_gate(at+2u),strata_gate(at+3u));
