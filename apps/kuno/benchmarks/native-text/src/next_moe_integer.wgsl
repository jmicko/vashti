// Two residual Q8 planes over original F32 input; original Q4_K weights.
// Workgroup owners total16000 bytes including the existing cached_input.
var<workgroup> packed_first:array<u32,640>;
var<workgroup> packed_second:array<u32,640>;
var<workgroup> scale_first:array<f32,80>;
var<workgroup> scale_second:array<f32,80>;
fn q8_code(x:f32,d:f32)->i32 {
    if d==0.0 {return 0;}
    let value=clamp(x/d,-127.0,127.0);let low=floor(value);let fraction=value-low;
    return i32(low)+select(0,1,fraction>0.5 || (fraction==0.5 && (i32(low)&1)!=0));
}
fn prepare_q8(block:u32) {
    if block>=80u {return;}
    var amax=0.0;
    for(var i=0u;i<32u;i++) {let at=block*32u+i;amax=max(amax,abs(cached_input[at/4u][at%4u]));}
    let d=amax/127.0;scale_first[block]=d;
    var rmax=0.0;
    for(var i=0u;i<32u;i++) {let at=block*32u+i;let x=cached_input[at/4u][at%4u];rmax=max(rmax,abs(x-f32(q8_code(x,d))*d));}
    let rd=rmax/127.0;scale_second[block]=rd;
    for(var word=0u;word<8u;word++) {
        var first=0u;var second=0u;
        for(var byte=0u;byte<4u;byte++) {
            let at=block*32u+word*4u+byte;let x=cached_input[at/4u][at%4u];
            let q=q8_code(x,d);let r=q8_code(x-f32(q)*d,rd);
            first|=(bitcast<u32>(q)&255u)<<(byte*8u);
            second|=(bitcast<u32>(r)&255u)<<(byte*8u);
        }
        packed_first[block*8u+word]=first;packed_second[block*8u+word]=second;
    }
}
fn packed_dot(q:u32,col:u32)->f32 {
    return scale_first[col/8u]*f32(dot4I8Packed(q,packed_first[col]))
        +scale_second[col/8u]*f32(dot4I8Packed(q,packed_second[col]));
}
fn project_fusedSUFFIX(row:u32,base:u32,lane:u32,width:u32)->vec2<f32> {
    var gate_sum=0.0;var up_sum=0.0;
    if row<640u {
        for(var block=0u;block<10u;block++) {
            let b=base/4u+block*36u;
            let gf=unpack2x16float(gateSUFFIX[b]);let uf=unpack2x16float(upSUFFIX[b]);
            let ga=gateSUFFIX[b+1u];let gz=gateSUFFIX[b+2u];let gs=gateSUFFIX[b+3u];
            let ua=upSUFFIX[b+1u];let uz=upSUFFIX[b+2u];let us=upSUFFIX[b+3u];
            for(var word=lane;word<32u;word+=width) {
                let pair=word/8u;let i=(word%8u)*4u;let col=(block*256u+pair*64u+i)/4u;
                let gq=gateSUFFIX[b+4u+word];let uq=upSUFFIX[b+4u+word];
                let low_sum=packed_dot(0x01010101u,col);let high_sum=packed_dot(0x01010101u,col+8u);
                gate_sum+=gf.x*gateSUFFIX_q4_scale(ga,gs,pair*2u)*packed_dot(gq&0x0f0f0f0fu,col)
                    -gf.y*gateSUFFIX_q4_minimum(gz,gs,pair*2u)*low_sum
                    +gf.x*gateSUFFIX_q4_scale(ga,gs,pair*2u+1u)*packed_dot((gq>>4u)&0x0f0f0f0fu,col+8u)
                    -gf.y*gateSUFFIX_q4_minimum(gz,gs,pair*2u+1u)*high_sum;
                up_sum+=uf.x*upSUFFIX_q4_scale(ua,us,pair*2u)*packed_dot(uq&0x0f0f0f0fu,col)
                    -uf.y*upSUFFIX_q4_minimum(uz,us,pair*2u)*low_sum
                    +uf.x*upSUFFIX_q4_scale(ua,us,pair*2u+1u)*packed_dot((uq>>4u)&0x0f0f0f0fu,col+8u)
                    -uf.y*upSUFFIX_q4_minimum(uz,us,pair*2u+1u)*high_sum;
            }
        }
    }
    return vec2<f32>(subgroupAdd(gate_sum),subgroupAdd(up_sum));
}
