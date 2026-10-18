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
