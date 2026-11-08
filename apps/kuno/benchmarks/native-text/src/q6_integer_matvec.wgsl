// Explicit two-residual-Q8 arithmetic; original weights are never repacked.
// Rust supplies immutable geometry/bindings and runtime owning INFO offsets.
fn finite(x:f32)->bool {return x<=3.4028234663852886e38 && x>=-3.4028234663852886e38;}
fn q8_code(x:f32,d:f32)->i32 {
    if d==0.0 {return 0;}
    let value=clamp(x/d,-127.0,127.0);let low=floor(value);let fraction=value-low;
    return i32(low)+select(0,1,fraction>0.5 || (fraction==0.5 && (i32(low)&1)!=0));
}
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    let block=id.x;if block>=COLUMNS/32u {return;}
    var values:array<f32,32>;var first_codes:array<i32,32>;
    var valid=true;var maximum=0.0;
    for(var i=0u;i<32u;i++) {
        let x=input[p.a+block*32u+i];values[i]=x;
        if !(abs(x)<3.4028234663852886e38) {valid=false;}else{maximum=max(maximum,abs(x));}
    }
    var d=0.0;var rd=0.0;
    if valid {
        d=maximum/127.0;var residual_max=0.0;
        for(var i=0u;i<32u;i++) {
            let q=q8_code(values[i],d);first_codes[i]=q;
            let reconstructed=f32(q)*d;
            // Testing the product itself prevents a fused subtraction from
            // hiding finite-input reconstruction overflow (e.g. f32::MAX).
            if !finite(reconstructed) {valid=false;values[i]=0.0;}
            else {
                let residual=values[i]-reconstructed;
                if !finite(residual) {valid=false;values[i]=0.0;}
                else {values[i]=residual;residual_max=max(residual_max,abs(residual));}
            }
        }
        rd=residual_max/127.0;
    }
    if !valid {atomicOr(&fault[p.c],1u);d=0.0;rd=0.0;}
    for(var word=0u;word<8u;word++) {
        var first=0u;var second=0u;
        if valid {
            for(var byte=0u;byte<4u;byte++) {
                let at=word*4u+byte;
                first|=(bitcast<u32>(first_codes[at])&255u)<<(byte*8u);
                second|=(bitcast<u32>(q8_code(values[at],rd))&255u)<<(byte*8u);
            }
        }
        planes[p.b+block*8u+word]=first;
        planes[p.b+PACKED_WORDS+block*8u+word]=second;
    }
    planes[p.b+2u*PACKED_WORDS+block]=bitcast<u32>(d);
    planes[p.b+2u*PACKED_WORDS+BLOCKS+block]=bitcast<u32>(rd);
}
// PROJECT_STAGE
fn finite(x:f32)->bool {return x<=3.4028234663852886e38 && x>=-3.4028234663852886e38;}
// WEIGHT_HELPERS_BEGIN
fn weight_word(index:u32)->u32 {return weight[index];}
fn word_at(at:u32)->u32 {
    let shift=(at%4u)*8u;let low=weight[at/4u];
    if shift==0u {return low;}
    return (low>>shift)|(weight[at/4u+1u]<<(32u-shift));
}
fn q6_word(base:u32,sub:u32,word:u32)->u32 {
    let half=sub/8u;let quarter=(sub%8u)/2u;let side=sub%2u;
    let offset=side*16u+word*4u;
    let low=word_at(base+half*64u+(quarter%2u)*32u+offset);
    let high=word_at(base+128u+half*32u+offset);
    var nibble=low&0x0f0f0f0fu;
    if quarter>=2u {nibble=(low>>4u)&0x0f0f0f0fu;}
    let codes=nibble|(((high>>(quarter*2u))&0x03030303u)<<4u);
    // Each code is0..63; adding96 cannot carry across an8-bit lane.
    // XOR128 then yields the exact signed byte(code-32), with no borrow.
    return (codes+0x60606060u)^0x80808080u;
}
fn q6_words(base:u32,sub:u32)->vec4<u32> {
    return vec4<u32>(q6_word(base,sub,0u),q6_word(base,sub,1u),
        q6_word(base,sub,2u),q6_word(base,sub,3u));
}
// WEIGHT_HELPERS_END
fn byte_at(at:u32)->u32 {return (weight_word(at/4u)>>((at%4u)*8u))&255u;}
fn signed_byte(x:u32)->i32 {return i32(x)-select(0,256,x>=128u);}
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) local:u32,
    @builtin(subgroup_id) sg:u32,@builtin(subgroup_invocation_id) lane:u32,
    @builtin(subgroup_size) width:u32,@builtin(num_workgroups) dispatched:vec3<u32>) {
    let row=g.x*(128u/width)+sg;var sum=0.0;
    if row<ROWS && atomicLoad(&fault[p.d])==0u {
        let row_base=p.a+row*ROW_BYTES;
        for(var group=lane;group<COLUMNS/16u;group+=width) {
            let base=row_base+(group/16u)*210u;let sub=group%16u;
            let half_bits=byte_at(base+208u)|(byte_at(base+209u)<<8u);
            let factor=unpack2x16float(half_bits).x*f32(signed_byte(byte_at(base+192u+sub)));
            var first=0;var second=0;
            let codes=q6_words(base,sub);
            for(var word=0u;word<4u;word++) {
                let q=codes[word];let at=group*4u+word;
                first+=dot4I8Packed(q,planes[p.b+at]);
                second+=dot4I8Packed(q,planes[p.b+PACKED_WORDS+at]);
            }
            let d=bitcast<f32>(planes[p.b+2u*PACKED_WORDS+group/2u]);
            let rd=bitcast<f32>(planes[p.b+2u*PACKED_WORDS+BLOCKS+group/2u]);
            let pair=d*f32(first)+rd*f32(second);
            if !finite(factor) || !finite(pair) {atomicOr(&fault[p.d],1u);}
            sum+=factor*pair;
        }
    }
    let total=subgroupAdd(sum);
    if row<ROWS && lane==0u {
        if !finite(total) {atomicOr(&fault[p.d],1u);}
        if atomicLoad(&fault[p.d])==0u {output[p.c+row]=total;}
    }
    if AUDIT_TAIL {
        let index=g.x*128u+local;let end=p.e+p.f;
        for(var i=index;i<arrayLength(&output);i+=128u*dispatched.x) {
            if i<p.e || i>=end {if !finite(output[i]) {atomicOr(&fault[p.d],1u);}}
        }
    }
}
// WIDE_WEIGHT_HELPERS
// Original byte ranges, no repacked/dequantized model. Aligned-vector loading
// follows Strata fb58e0's load16_a2 mechanism; its MIT notice is retained in
// q6_integer_fixtures/NOTICE. No inferred speed or physical bandwidth claim.
fn weight_word(index:u32)->u32 {return weight[index/4u][index%4u];}
fn words_at(at:u32)->vec4<u32> {
    let low=weight[at/16u];var high=low;
    // Read the second aligned vector only when needed. For the final qh run
    // its last byte is at most block+207, inside the original210-byte block.
    if at%16u!=0u {high=weight[at/16u+1u];}
    let offset=(at%16u)/4u;let shift=(at%4u)*8u;
    var words=low;var tail=high.x;
    switch offset {
        case 1u: {words=vec4<u32>(low.y,low.z,low.w,high.x);tail=high.y;}
        case 2u: {words=vec4<u32>(low.z,low.w,high.x,high.y);tail=high.z;}
        case 3u: {words=vec4<u32>(low.w,high.x,high.y,high.z);tail=high.w;}
        default: {}
    }
    if shift==0u {return words;}
    let next=vec4<u32>(words.y,words.z,words.w,tail);
    return (words>>vec4<u32>(shift))|(next<<vec4<u32>(32u-shift));
}
fn q6_words(base:u32,sub:u32)->vec4<u32> {
    let half=sub/8u;let quarter=(sub%8u)/2u;let offset=(sub%2u)*16u;
    let low=words_at(base+half*64u+(quarter%2u)*32u+offset);
    let high=words_at(base+128u+half*32u+offset);
    var nibble=low&vec4<u32>(0x0f0f0f0fu);
    if quarter>=2u {nibble=(low>>vec4<u32>(4u))&vec4<u32>(0x0f0f0f0fu);}
    let codes=nibble|(((high>>vec4<u32>(quarter*2u))&vec4<u32>(0x03030303u))<<vec4<u32>(4u));
    return (codes+vec4<u32>(0x60606060u))^vec4<u32>(0x80808080u);
}
