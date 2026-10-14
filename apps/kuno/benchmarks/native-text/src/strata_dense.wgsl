// MIT packed layouts: ggml b11232, crates/kuno-text/NOTICE; Strata fb58e0
// native_mmvq Q8_1 input arithmetic. Original GGUF weights are never converted.
fn finite(x:f32)->bool {return (bitcast<u32>(x)&0x7f800000u)!=0x7f800000u;}
fn byte_at(at:u32)->u32 {return (weight[at/4u]>>((at%4u)*8u))&255u;}
fn half_at(at:u32)->f32 {return unpack2x16float(byte_at(at)|(byte_at(at+1u)<<8u)).x;}
fn word_at(at:u32)->u32 {
    let shift=(at%4u)*8u;let lo=weight[at/4u];
    if shift==0u {return lo;} return (lo>>shift)|(weight[at/4u+1u]<<(32u-shift));
}
fn signed_byte(x:u32)->i32 {return i32(x)-select(0,256,x>=128u);}
fn table(q:u32)->u32 {
    let t=array<u32,4>(0xbfad9881u,0xf6eaddcfu,0x26190d01u,0x71594535u);
    return (t[q/4u]>>((q%4u)*8u))&255u;
}
fn nonlinear(q:u32)->u32 {
    return table(q&15u)|(table((q>>8u)&15u)<<8u)|(table((q>>16u)&15u)<<16u)|(table((q>>24u)&15u)<<24u);
}
fn scale_at(b:u32,g:u32,minimum:bool)->f32 {
    let i=g%4u;let at=b+select(4u,8u,minimum);
    var value=byte_at(at+i)&63u;
    if g>=4u {
        let c=byte_at(b+12u+i); let hi=byte_at(at+i)>>6u;
        value=select(c&15u,c>>4u,minimum)|(hi<<4u);
    }
    return f32(value);
}
fn factors(row:u32,group:u32)->vec2<f32> {
    if DTYPE==14u {
        let b=row+(group/16u)*210u;
        return vec2<f32>(half_at(b+208u)*f32(signed_byte(byte_at(b+192u+group%16u))),0.0);
    }
    if DTYPE==12u || DTYPE==13u {
        let b=row+(group/8u)*BLOCK_BYTES;
        return vec2<f32>(half_at(b)*scale_at(b,group%8u,false),half_at(b+2u)*scale_at(b,group%8u,true));
    }
    if DTYPE==23u {
        let b=row+(group/8u)*136u;let sub=group%8u;
        let lo=(byte_at(b+4u+sub/2u)>>((sub%2u)*4u))&15u;
        let hi=((byte_at(b+2u)|(byte_at(b+3u)<<8u))>>(sub*2u))&3u;
        return vec2<f32>(half_at(b)*f32(i32(lo|(hi<<4u))-32),0.0);
    }
    return vec2<f32>(half_at(row+group*BLOCK_BYTES),0.0);
}
fn codes(row:u32,group:u32,word:u32)->u32 {
    if DTYPE==14u {
        let b=row+(group/16u)*210u;let sub=group%16u;
        let half=sub/8u;let quarter=(sub%8u)/2u;let offset=(sub%2u)*16u+word*4u;
        let low=word_at(b+half*64u+(quarter%2u)*32u+offset);
        let high=word_at(b+128u+half*32u+offset);
        let nibble=select(low&0x0f0f0f0fu,(low>>4u)&0x0f0f0f0fu,quarter>=2u);
        let q=nibble|(((high>>(quarter*2u))&0x03030303u)<<4u);
        return (q+0x60606060u)^0x80808080u;
    }
    if DTYPE==12u || DTYPE==13u {
        let b=row+(group/8u)*BLOCK_BYTES;let sub=group%8u;
        let q=word_at(b+select(16u,48u,DTYPE==13u)+(sub/2u)*32u+word*4u);
        var out=select(q&0x0f0f0f0fu,(q>>4u)&0x0f0f0f0fu,sub%2u==1u);
        if DTYPE==13u {out|=((word_at(b+16u+word*4u)>>sub)&0x01010101u)<<4u;}
        return out;
    }
    if DTYPE==23u || DTYPE==20u {
        let b=row+(group/select(1u,8u,DTYPE==23u))*BLOCK_BYTES;
        let at=b+select(2u,8u+(group%8u)*16u,DTYPE==23u)+(word%4u)*4u;
        let q=word_at(at);
        return nonlinear(select(q&0x0f0f0f0fu,(q>>4u)&0x0f0f0f0fu,word>=4u));
    }
    return word_at(row+group*34u+2u+word*4u); // original Q8_0
}
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(subgroup_id) sg:u32,
        @builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_size) width:u32,
        @builtin(local_invocation_index) local:u32) {
    if arrayLength(&fault)!=1u {return;}
    if width!=32u || p.count!=4u || p.stride<PLANE_WORDS
        || arrayLength(&weight)*4u<ROWS*ROW_BYTES
        || arrayLength(&planes)<4u*p.stride || p.output_stride<ROWS
        || p.output_first>arrayLength(&output)
        || ROWS>arrayLength(&output)-p.output_first
        || 3u*p.output_stride>arrayLength(&output)-p.output_first-ROWS {
        if local==0u {atomicOr(&fault[0],8u);}return;
    }
    let row=g.x*4u+sg;var sum=vec4<f32>(0.0);
    if row<ROWS {
        for(var group=lane;group<COLUMNS/GROUP_VALUES;group+=32u) {
            let f=factors(row*ROW_BYTES,group);var dots=vec4<i32>(0);var sums=vec4<i32>(0);
            for(var word=0u;word<GROUP_VALUES/4u;word++) {
                let q=codes(row*ROW_BYTES,group,word);let at=group*(GROUP_VALUES/4u)+word;
                let x=vec4<u32>(planes[at],planes[p.stride+at],planes[2u*p.stride+at],planes[3u*p.stride+at]);
                dots+=vec4<i32>(dot4I8Packed(q,x.x),dot4I8Packed(q,x.y),dot4I8Packed(q,x.z),dot4I8Packed(q,x.w));
                if DTYPE==12u || DTYPE==13u {sums+=vec4<i32>(dot4I8Packed(0x01010101u,x.x),dot4I8Packed(0x01010101u,x.y),dot4I8Packed(0x01010101u,x.z),dot4I8Packed(0x01010101u,x.w));}
            }
            let at=2u*PACKED_WORDS+(group*GROUP_VALUES)/32u;
            let d=bitcast<vec4<f32>>(vec4<u32>(planes[at],planes[p.stride+at],planes[2u*p.stride+at],planes[3u*p.stride+at]));
            sum=fma(vec4<f32>(f.x)*d,vec4<f32>(dots),sum);
            if DTYPE==12u || DTYPE==13u {sum-=vec4<f32>(f.y)*d*vec4<f32>(sums);}
        }
    }
    let total=subgroupAdd(sum);
    if row<ROWS && lane==0u {
        if !finite(total.x) || !finite(total.y) || !finite(total.z) || !finite(total.w) {atomicOr(&fault[0],1u);}
        output[p.output_first+row]=total.x;
        output[p.output_first+p.output_stride+row]=total.y;
        output[p.output_first+2u*p.output_stride+row]=total.z;
        output[p.output_first+3u*p.output_stride+row]=total.w;
    }
}
