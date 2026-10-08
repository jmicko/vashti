@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) group:vec3<u32>, @builtin(subgroup_id) sg:u32,
        @builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_size) width:u32) {
    let row=group.x*(128u/width)+sg;
    let rank=group.y;
    var base=row*ROW_BYTES;
    var input=p.a;
    if ROUTED { base+=(route[rank]*ROWS)*ROW_BYTES; }
    if RANKED_INPUT { input+=rank*COLUMNS; }
    var sum=0.0;
    if row<ROWS {
        if BLOCK_DOT && DTYPE==12u {
            for(var block=0u;block<COLUMNS/256u;block++) {
                let b=base/4u+block*36u;
                let factors=unpack2x16float(weight[b]);
                let a=weight[b+1u];let z=weight[b+2u];let scales=weight[b+3u];
                for(var pair=0u;pair<4u;pair++) {
                    let d0=factors.x*q4_scale(a,scales,pair*2u);
                    let d1=factors.x*q4_scale(a,scales,pair*2u+1u);
                    let m0=factors.y*q4_minimum(z,scales,pair*2u);
                    let m1=factors.y*q4_minimum(z,scales,pair*2u+1u);
                    for(var i=lane;i<32u;i+=width) {
                        let q=word_byte(weight[b+4u+pair*8u+i/4u],i%4u);
                        let col=block*256u+pair*64u+i;
                        sum+=(d0*f32(q&15u)-m0)*arena[input+col];
                        sum+=(d1*f32(q>>4u)-m1)*arena[input+col+32u];
                    }
                }
            }
        } else if BLOCK_DOT && DTYPE==14u {
            for(var block=0u;block<COLUMNS/256u;block++) {
                let b=base+block*210u;let d=wh(b+208u);
                for(var half=0u;half<2u;half++) {
                    for(var i=lane;i<32u;i+=width) {
                        let l0=wb(b+half*64u+i);let l1=wb(b+half*64u+32u+i);
                        let h=wb(b+128u+half*32u+i);let sc=b+192u+half*8u+i/16u;
                        let col=block*256u+half*128u+i;
                        let q0=i32((l0&15u)|((h&3u)<<4u))-32;
                        let q1=i32((l1&15u)|(((h>>2u)&3u)<<4u))-32;
                        let q2=i32((l0>>4u)|(((h>>4u)&3u)<<4u))-32;
                        let q3=i32((l1>>4u)|(((h>>6u)&3u)<<4u))-32;
                        sum+=(d*signed(wb(sc)))*f32(q0)*arena[input+col];
                        sum+=(d*signed(wb(sc+2u)))*f32(q1)*arena[input+col+32u];
                        sum+=(d*signed(wb(sc+4u)))*f32(q2)*arena[input+col+64u];
                        sum+=(d*signed(wb(sc+6u)))*f32(q3)*arena[input+col+96u];
                    }
                }
            }
        } else {
            for(var c=lane;c<COLUMNS;c+=width){ sum+=packed(base,c)*arena[input+c]; }
        }
    }
    let total=subgroupAdd(sum);
    if row<ROWS && lane==0u { arena[p.b+rank*ROWS+row]=total; }
}
