// MIT Strata fb58e0 native_mmvq.dp.cpp::native_mmvq_wide{,32}_kernel.
// Original packed weights, one row/subgroup and every real RHS. Q4/Q5 use
// eight lanes/block, IQ4_XS one subblock/lane; Q8_0 two lanes/block and
// IQ4_NL one lane/block. The original padded owner permits aligned A2 loads.
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
fn dp4a4(a:vec4<u32>,b:vec4<u32>)->i32 {
    var sum=dot4I8Packed(a.x,b.x);
    sum+=dot4I8Packed(a.y,b.y);sum+=dot4I8Packed(a.z,b.z);sum+=dot4I8Packed(a.w,b.w);
    return sum;
}
fn input16(base:u32,block:u32,half:u32)->vec4<u32> {
    let at=base+block*8u+half*4u;
    return vec4<u32>(planes[at],planes[at+1u],planes[at+2u],planes[at+3u]);
}
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(subgroup_id) sg:u32,
        @builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_size) width:u32,
        @builtin(local_invocation_index) local:u32) {
    if arrayLength(&fault)!=1u {return;}
    if width!=32u || p.count!=MEMBERS || MEMBERS<1u || MEMBERS>8u
        || p.stride<PLANE_WORDS || arrayLength(&weight)*16u<ROWS*ROW_BYTES
        || arrayLength(&planes)<p.count*p.stride || p.output_stride<ROWS
        || p.output_first>arrayLength(&output) || ROWS>arrayLength(&output)-p.output_first
        || (p.count-1u)*p.output_stride>arrayLength(&output)-p.output_first-ROWS {
        if local==0u {atomicOr(&fault[0],8u);}return;
    }
    let row=group.x*4u+sg;
    var sums:array<f32,MEMBERS>;
    if row<ROWS {
        let is32=DTYPE==8u || DTYPE==20u;
        let lanes=select(8u,select(1u,2u,DTYPE==8u),is32);
        let values=select(256u,32u,is32);
        let g=lane%lanes;let sub=lane/lanes;
        for(var block=sub;block<COLUMNS/values;block+=32u/lanes) {
            let b=row*ROW_BYTES+block*BLOCK_BYTES;
            var lo=vec4<u32>(0u);var hi=lo;
            var dl=0.0;var dh=0.0;var ml=0.0;var mh=0.0;
            var sb=0u;var half=0u;
            if DTYPE==8u {
                lo=load16_a2(b+2u+16u*g);dl=half_at(b);half=g;
            } else if DTYPE==20u || DTYPE==23u {
                if DTYPE==23u {sb=g;}
                let q=load16_a2(b+select(2u,8u+16u*g,DTYPE==23u));
                lo=vec4<u32>(nonlinear(q.x&0x0f0f0f0fu),nonlinear(q.y&0x0f0f0f0fu),nonlinear(q.z&0x0f0f0f0fu),nonlinear(q.w&0x0f0f0f0fu));
                hi=vec4<u32>(nonlinear((q.x>>4u)&0x0f0f0f0fu),nonlinear((q.y>>4u)&0x0f0f0f0fu),nonlinear((q.z>>4u)&0x0f0f0f0fu),nonlinear((q.w>>4u)&0x0f0f0f0fu));
                dl=half_at(b);
                if DTYPE==23u {
                    let low=(byte_at(b+4u+g/2u)>>((g&1u)*4u))&15u;
                    let high=((byte_at(b+2u)|(byte_at(b+3u)<<8u))>>(2u*g))&3u;
                    dl*=f32(i32(low|(high<<4u))-32);
                }
            } else {
                sb=2u*(g>>1u);half=g&1u;
                let q=load16_a2(b+select(16u,48u,DTYPE==13u)+16u*g);
                lo=q&vec4<u32>(0x0f0f0f0fu);hi=(q>>vec4<u32>(4u))&vec4<u32>(0x0f0f0f0fu);
                if DTYPE==13u {
                    let h=load16_a2(b+16u+16u*half);
                    lo|=((h>>vec4<u32>(sb))&vec4<u32>(0x01010101u))<<vec4<u32>(4u);
                    hi|=((h>>vec4<u32>(sb+1u))&vec4<u32>(0x01010101u))<<vec4<u32>(4u);
                }
                let d=half_at(b);let m=half_at(b+2u);
                dl=d*scale_at(b,sb,false);dh=d*scale_at(b,sb+1u,false);
                ml=m*scale_at(b,sb,true);mh=m*scale_at(b,sb+1u,true);
            }
            for(var member=0u;member<MEMBERS;member++) {
                let base=member*p.stride;let xb=block*select(8u,1u,is32)+sb;
                let d0=bitcast<f32>(planes[base+2u*PACKED_WORDS+xb]);
                let u0=input16(base,xb,half);
                if DTYPE==8u {
                    sums[member]+=dl*d0*f32(dp4a4(lo,u0));
                } else if DTYPE==20u || DTYPE==23u {
                    let u1=input16(base,xb,1u);
                    let dot=dp4a4(lo,u0)+dp4a4(hi,u1);
                    sums[member]+=dl*d0*f32(dot);
                } else {
                    let d1=bitcast<f32>(planes[base+2u*PACKED_WORDS+xb+1u]);
                    let u1=input16(base,xb+1u,half);let ones=vec4<u32>(0x01010101u);
                    let a=dl*f32(dp4a4(lo,u0))-ml*f32(dp4a4(ones,u0));
                    let c=dh*f32(dp4a4(hi,u1))-mh*f32(dp4a4(ones,u1));
                    sums[member]+=d0*a+d1*c;
                }
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
