struct Params { a:u32, b:u32, c:u32, d:u32, e:u32, f:u32, g:u32, h:u32 }
struct Control { token:u32, pos:u32, unused0:u32, unused1:u32 }
@group(0) @binding(0) var<storage,read> weight:array<u32>;
@group(0) @binding(1) var<storage,read> second:array<u32>;
@group(0) @binding(2) var<storage,read_write> arena:array<f32>;
@group(0) @binding(3) var<storage,read_write> state:array<f32>;
@group(0) @binding(4) var<uniform> p:Params;
@group(0) @binding(5) var<storage,read_write> route:array<u32>;
@group(0) @binding(6) var<uniform> ctrl:Control;
fn sigmoid(x:f32)->f32 { return 1.0/(1.0+exp(-x)); }
fn silu(x:f32)->f32 { return x*sigmoid(x); }
fn wb(offset:u32)->u32 { return (weight[offset/4u] >> (8u*(offset%4u)))&255u; }
fn wh(offset:u32)->f32 {
    return unpack2x16float(wb(offset)|(wb(offset+1u)<<8u)).x;
}
fn signed(x:u32)->f32 { return f32(i32(x)-select(0,256,x>=128u)); }
fn word_byte(x:u32,index:u32)->u32 {return (x>>(index*8u))&255u;}
fn q4_scale(a:u32,c:u32,g:u32)->f32 {
    let i=g%4u;
    return f32(select((word_byte(c,i)&15u)|((word_byte(a,i)>>6u)<<4u),word_byte(a,i)&63u,g<4u));
}
fn q4_minimum(b:u32,c:u32,g:u32)->f32 {
    let i=g%4u;
    return f32(select((word_byte(c,i)>>4u)|((word_byte(b,i)>>6u)<<4u),word_byte(b,i)&63u,g<4u));
}
fn packed(base:u32, col:u32)->f32 {
    if DTYPE==0u { return bitcast<f32>(weight[base/4u+col]); }
    if DTYPE==1u {return wh(base+col*2u);}
    if DTYPE==30u {let b=base+col*2u;return bitcast<f32>((wb(b)|(wb(b+1u)<<8u))<<16u);}
    if DTYPE==8u {let b=base+(col/32u)*34u;return wh(b)*signed(wb(b+2u+col%32u));}
    if DTYPE==6u {
        let b=base+(col/32u)*22u;let ix=col%32u;
        let qh=wb(b+2u)|(wb(b+3u)<<8u)|(wb(b+4u)<<16u)|(wb(b+5u)<<24u);
        let low=wb(b+6u+ix%16u);
        let q=((low>>((ix/16u)*4u))&15u)|(((qh>>ix)&1u)<<4u);
        return wh(b)*f32(i32(q)-16);
    }
    if DTYPE==12u {
        let b=base+(col/256u)*144u; let ix=col%256u; let g=ix/32u;
        var sc:u32; var mn:u32;
        if g<4u { sc=wb(b+4u+g)&63u; mn=wb(b+8u+g)&63u; }
        else { sc=(wb(b+8u+g)&15u)|((wb(b+g)>>6u)<<4u);
               mn=(wb(b+8u+g)>>4u)|((wb(b+4u+g)>>6u)<<4u); }
        let q=wb(b+16u+(ix/64u)*32u+ix%32u);
        let quant=select(q&15u,q>>4u,ix%64u>=32u);
        return (wh(b)*f32(sc))*f32(quant)-wh(b+2u)*f32(mn);
    }
    let b=base+(col/256u)*210u; let ix=col%256u;
    let half=ix/128u; let quarter=(ix%128u)/32u; let lane=ix%32u;
    let low=wb(b+half*64u+lane+(quarter%2u)*32u);
    let high=wb(b+128u+half*32u+lane);
    let quant=i32(((low>>((quarter/2u)*4u))&15u)|(((high>>(quarter*2u))&3u)<<4u))-32;
    let sc=signed(wb(b+192u+half*8u+quarter*2u+lane/16u));
    return wh(b+208u)*sc*f32(quant);
}
