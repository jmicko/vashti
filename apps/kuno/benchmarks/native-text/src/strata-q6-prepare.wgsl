// Strata Q8_1 arithmetic in the existing owning two-plane envelope. The second
// plane is explicitly zero; original requests retain their original quantizer.
// Codes use the unrounded scale, dots use the F16-representable Q8_1 scale.
fn finite(x:f32)->bool {return (bitcast<u32>(x)&0x7f800000u)!=0x7f800000u;}
fn q8_code(x:f32,d:f32)->i32 {
    if d==0.0 {return 0;}
    let value=clamp(x/d,-127.0,127.0);let low=floor(value);let fraction=value-low;
    return i32(low)+select(0,1,fraction>0.5 || (fraction==0.5 && (i32(low)&1)!=0));
}
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    // STRATA_BATCH_PREPARE_GUARD
    var maximum=0.0;var valid=true;
    for(var i=0u;i<32u;i++) {
        let x=input[input_first+block*32u+i];
        valid=valid && abs(x)<3.4028234663852886e38;
        maximum=max(maximum,abs(x));
    }
    var d=maximum/127.0;
    if !valid {atomicOr(&fault[p.c],1u);d=0.0;}
    for(var word=0u;word<8u;word++) {
        var packed=0u;
        if valid {
            for(var byte=0u;byte<4u;byte++) {
                let q=q8_code(input[input_first+block*32u+word*4u+byte],d);
                packed|=(bitcast<u32>(q)&255u)<<(byte*8u);
            }
        }
        planes[planes_first+block*8u+word]=packed;
        planes[planes_first+PACKED_WORDS+block*8u+word]=0u;
    }
    planes[planes_first+2u*PACKED_WORDS+block]=bitcast<u32>(unpack2x16float(pack2x16float(vec2<f32>(d,0.0))).x);
    planes[planes_first+2u*PACKED_WORDS+BLOCKS+block]=0u;
}
