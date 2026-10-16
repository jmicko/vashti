// MIT Strata fb58e0 native_mmvq.dp.cpp::native_mmvq_q6k_wide_kernel.
// Selected A2/RPW1/subgroup32 path. Original210-byte blocks, aligned int4
// loads,8 lanes per block and4 blocks per subgroup. No packed weight copy.
fn load16_a2(at:u32)->vec4<u32> {
    let low=weight[at/16u];var high=low;
    if at%16u!=0u {high=weight[at/16u+1u];}
    let k=(at/4u)%4u;var w=low;var next=high.x;
    if k==1u {w=vec4<u32>(low.y,low.z,low.w,high.x);next=high.y;}
    else if k==2u {w=vec4<u32>(low.z,low.w,high.x,high.y);next=high.z;}
    else if k==3u {w=vec4<u32>(low.w,high.x,high.y,high.z);next=high.w;}
    if (at&2u)!=0u {return (w>>vec4<u32>(16u))|(vec4<u32>(w.y,w.z,w.w,next)<<vec4<u32>(16u));}
    return w;
}
fn signed_q6(q:vec4<u32>)->vec4<u32> {
    return (q+vec4<u32>(0x60606060u))^vec4<u32>(0x80808080u);
}
fn dp4a4(a:vec4<u32>,b:vec4<u32>)->i32 {
    var sum=dot4I8Packed(a.x,b.x);
    sum+=dot4I8Packed(a.y,b.y);sum+=dot4I8Packed(a.z,b.z);sum+=dot4I8Packed(a.w,b.w);
    return sum;
}
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(subgroup_id) sg:u32,
        @builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_size) width:u32,
        @builtin(local_invocation_index) local:u32) {
    if arrayLength(&fault)!=1u {return;}
    if DTYPE!=14u || width!=32u || p.count!=MEMBERS || MEMBERS<1u || MEMBERS>8u
        || COLUMNS%256u!=0u || p.stride<PLANE_WORDS
        || arrayLength(&weight)*16u<ROWS*ROW_BYTES
        || arrayLength(&planes)<p.count*p.stride || p.output_stride<ROWS
        || p.output_first>arrayLength(&output) || ROWS>arrayLength(&output)-p.output_first
        || (p.count-1u)*p.output_stride>arrayLength(&output)-p.output_first-ROWS {
        if local==0u {atomicOr(&fault[0],8u);}return;
    }
    let row=group.x*4u+sg;
    let g=lane&7u;let sub=lane>>3u;
    let vh_shift=2u*((g&3u)>>1u);
    let qh4_index=2u*(g>>2u)+(g&1u);
    let scale_offset=8u*(g>>2u)+(g&3u);
    let q8_offset=4u*(g>>2u)+((g&3u)>>1u);
    let u_int4=g&1u;
    var sums:array<f32,MEMBERS>;
    if row<ROWS {
        for(var block=sub;block<COLUMNS/256u;block+=4u) {
            let b=row*ROW_BYTES+block*210u;
            let low=load16_a2(b+16u*g);
            let high=load16_a2(b+128u+16u*qh4_index)>>vec4<u32>(vh_shift);
            let d=half_at(b+208u);
            let ds0=d*f32(signed_byte(byte_at(b+192u+scale_offset)));
            let ds1=d*f32(signed_byte(byte_at(b+196u+scale_offset)));
            let vi0=signed_q6((low&vec4<u32>(0x0f0f0f0fu))|((high<<vec4<u32>(4u))&vec4<u32>(0x30303030u)));
            let vi1=signed_q6(((low>>vec4<u32>(4u))&vec4<u32>(0x0f0f0f0fu))|(((high>>vec4<u32>(4u))<<vec4<u32>(4u))&vec4<u32>(0x30303030u)));
            // Original weights are shared across every real RHS. The original
            // Q8_1 codes/scales live in the existing transposed private plane.
            for(var member=0u;member<MEMBERS;member++) {
                let base=member*p.stride;let q8=block*8u+q8_offset;
                let at0=base+q8*8u+4u*u_int4;let at1=at0+16u;
                let u0=vec4<u32>(planes[at0],planes[at0+1u],planes[at0+2u],planes[at0+3u]);
                let u1=vec4<u32>(planes[at1],planes[at1+1u],planes[at1+2u],planes[at1+3u]);
                let d0=bitcast<f32>(planes[base+2u*PACKED_WORDS+q8]);
                let d1=bitcast<f32>(planes[base+2u*PACKED_WORDS+q8+2u]);
                let s0=dp4a4(vi0,u0);let s1=dp4a4(vi1,u1);
                sums[member]+=d0*ds0*f32(s0)+d1*ds1*f32(s1);
            }
        }
    }
    for(var member=0u;member<MEMBERS;member++) {
        var total=sums[member];
        for(var offset=16u;offset>0u;offset>>=1u) {total+=subgroupShuffleXor(total,offset);}
        if row<ROWS && lane==0u {
            if !finite(total) {atomicOr(&fault[0],1u);}
            output[p.output_first+member*p.output_stride+row]=total;
        }
    }
}
