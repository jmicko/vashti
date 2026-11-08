// Original GGML b11232 signed IQ4_NL codebook (MIT; crates/kuno-text/NOTICE).
// Same four packed scalar literals as quant_matvec::iq4_value, selected without
// an indexed Function array. The packed nibble caller proves index<16. Original
// F32 activations, scale products, packed weights and dot/reduction are retained.
fn dense_iq4_lane_packed(index:u32)->f32 {
    var word=0xbfad9881u;
    if index>=4u {word=0xf6eaddcfu;}
    if index>=8u {word=0x26190d01u;}
    if index>=12u {word=0x71594535u;}
    let byte=(word>>((index&3u)*8u))&255u;
    let value=i32(byte)-select(0,256,byte>=128u);
    return f32(value);
}
fn dense_iq4_packed(q:vec4<u32>)->vec4<f32> {
    return vec4<f32>(dense_iq4_lane_packed(q.x),dense_iq4_lane_packed(q.y),
                    dense_iq4_lane_packed(q.z),dense_iq4_lane_packed(q.w));
}
