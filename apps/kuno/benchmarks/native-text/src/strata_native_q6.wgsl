// MIT Strata fb58e0 iq_kernels.dp.cpp Fmt<14>/vec_dot_q6_K_q8_1.
// Original Q6_K rows + one native Q8_1 plane, 32-lane iqs integer chain.
// All active RHS rows share each original code load; no decoded weight copy.
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(subgroup_id) sg:u32,
        @builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_size) width:u32,
        @builtin(local_invocation_index) local:u32) {
    if arrayLength(&fault)!=1u {return;}
    if DTYPE!=14u || width!=32u || p.count<1u || p.count>MEMBERS || MEMBERS>8u
        || COLUMNS%256u!=0u || p.stride<PLANE_WORDS
        || arrayLength(&weight)*4u<ROWS*ROW_BYTES
        || arrayLength(&planes)<p.count*p.stride || p.output_stride<ROWS
        || p.output_first>arrayLength(&output) || ROWS>arrayLength(&output)-p.output_first
        || (p.count-1u)*p.output_stride>arrayLength(&output)-p.output_first-ROWS {
        if local==0u {atomicOr(&fault[0],8u);}return;
    }
    let row=g.x*4u+sg;
    var sums:array<f32,8>;
    if row<ROWS {
        // QI6_K=32, QR6_K=2, QI8_1=8; exactly one iqs per lane.
        let q8_offset=4u*(lane/16u)+(lane%16u)/8u;
        let scale_offset=8u*(lane/16u)+(lane%16u)/4u;
        let high_shift=2u*((lane%16u)/8u);
        for(var block=0u;block<COLUMNS/256u;block++) {
            let b=row*ROW_BYTES+block*210u;
            let low=word_at(b+lane*4u);
            let high=word_at(b+128u+(8u*(lane/16u)+lane%8u)*4u)>>high_shift;
            let d=half_at(b+208u);
            var contribution:array<f32,8>;
            for(var i=0u;i<2u;i++) {
                let code=((low>>(4u*i))&0x0f0f0f0fu)|(((high>>(4u*i))<<4u)&0x30303030u);
                let signed=(code+0x60606060u)^0x80808080u;
                let scale=signed_byte(byte_at(b+192u+scale_offset+4u*i));
                let q8=block*8u+q8_offset+2u*i;
                let at=q8*8u+lane%8u;
                for(var member=0u;member<MEMBERS;member++) {
                    if member<p.count {
                        let base=member*p.stride;
                        let product=dot4I8Packed(signed,planes[base+at])*scale;
                        let d8=bitcast<f32>(planes[base+2u*PACKED_WORDS+q8]);
                        contribution[member]=fma(d8,f32(product),contribution[member]);
                    }
                }
            }
            for(var member=0u;member<MEMBERS;member++) {
                if member<p.count {sums[member]=fma(d,contribution[member],sums[member]);}
            }
        }
    }
    for(var member=0u;member<MEMBERS;member++) {
        var total=sums[member];
        for(var offset=16u;offset>0u;offset>>=1u) {total+=subgroupShuffleXor(total,offset);}
        if row<ROWS && lane==0u && member<p.count {
            if !finite(total) {atomicOr(&fault[0],1u);}
            output[p.output_first+member*p.output_stride+row]=total;
        }
    }
}
