// Original packed IQ weights times two residual Q8 activation planes.
// Explicit alternate arithmetic: NOT the default original64 F32 contract.
// Rust substitutes public immutable dtype/geometry and direct constant-index
// grid stores, once per local lane before the existing unconditional barrier.
// No request-derived code generation, host-pinned doorbells or CUDA engine.
requires packed_4x8_integer_dot_product;

const COLUMNS:u32=COLUMNS_VALUEu;
const ROWS:u32=ROWS_VALUEu;
const ROW_BYTES:u32=ROW_BYTES_VALUEu;
const DTYPE:u32=DTYPE_VALUEu;
const FIRST_SLOT:u32=FIRST_SLOT_VALUEu;
const SLOTS:u32=SLOTS_VALUEu;
struct Params {a:u32,b:u32,c:u32,d:u32,e:u32,f:u32,g:u32,h:u32}
BINDINGS_VALUE

var<workgroup> cached_input:array<f32,COLUMNS_VALUE>;
var<workgroup> packed_first:array<u32,PACKED_VALUE>;
var<workgroup> packed_second:array<u32,PACKED_VALUE>;
var<workgroup> scale_first:array<f32,BLOCKS_VALUE>;
var<workgroup> scale_second:array<f32,BLOCKS_VALUE>;
var<workgroup> quant_fault:atomic<u32>;
GRID_DECLARATION_VALUE

fn valid_group(group:u32)->bool {
    let n=arrayLength(&map);
    if p.a>n {return false;}
    if 804u>n-p.a {return false;}
    if map[p.a]!=1u || map[p.a+1u]!=0u || map[p.a+2u]!=1u {return false;}
    let groups=map[p.a+3u];
    if groups==0u || groups>10u || group>=groups {return false;}
    let slot=map[p.a+4u+group];
    if slot<FIRST_SLOT || slot-FIRST_SLOT>=SLOTS {return false;}
    return map[p.a+84u+group]==1u && map[p.a+164u+group*8u]<10u;
}
// Q8_PREPARE_BEGIN
fn q8_code(x:f32,d:f32)->i32 {
    if d==0.0 {return 0;}
    let value=clamp(x/d,-127.0,127.0);
    let low=floor(value);let fraction=value-low;
    return i32(low)+select(0,1,fraction>0.5 || (fraction==0.5 && (i32(low)&1)!=0));
}
fn clear_q8(block:u32) {
    scale_first[block]=0.0;scale_second[block]=0.0;
    for(var word=0u;word<8u;word++) {
        packed_first[block*8u+word]=0u;packed_second[block*8u+word]=0u;
    }
    atomicOr(&quant_fault,1u);
}
fn residual(x:f32,q:i32,d:f32)->f32 {
    let reconstructed=f32(q)*d;
    // Check the reconstruction itself so a contracted x-q*d cannot hide an
    // overflowing intermediate and disagree with the host plane preflight.
    if !(abs(reconstructed)<=3.4028234663852886e38) {atomicOr(&quant_fault,1u);return 0.0;}
    return x-reconstructed;
}
fn prepare_q8(block:u32) {
    if block>=BLOCKS_VALUEu {return;}
    var amax=0.0;var valid=true;
    for(var i=0u;i<32u;i++) {
        let x=cached_input[block*32u+i];
        if !(abs(x)<3.4028234663852886e38) {valid=false;}
        amax=max(amax,abs(x));
    }
    if !valid {clear_q8(block);return;}
    let d=amax/127.0;scale_first[block]=d;
    var rmax=0.0;
    for(var i=0u;i<32u;i++) {
        let x=cached_input[block*32u+i];let r=residual(x,q8_code(x,d),d);
        if !(abs(r)<=3.4028234663852886e38) {valid=false;}
        rmax=max(rmax,abs(r));
    }
    if !valid {clear_q8(block);return;}
    let rd=rmax/127.0;scale_second[block]=rd;
    for(var word=0u;word<8u;word++) {
        var first=0u;var second=0u;
        for(var byte=0u;byte<4u;byte++) {
            let x=cached_input[block*32u+word*4u+byte];
            let q=q8_code(x,d);let r=q8_code(residual(x,q,d),rd);
            first|=(bitcast<u32>(q)&255u)<<(byte*8u);
            second|=(bitcast<u32>(r)&255u)<<(byte*8u);
        }
        packed_first[block*8u+word]=first;packed_second[block*8u+word]=second;
    }
}
// Q8_PREPARE_END
fn iq4_byte(code:u32)->u32 {
    var word=0xbfad9881u;
    if code>=4u {word=0xf6eaddcfu;}
    if code>=8u {word=0x26190d01u;}
    if code>=12u {word=0x71594535u;}
    return (word>>((code&3u)*8u))&255u;
}
// Original grid magnitudes are nonzero and<=127. Bytewise complement/add-one cannot carry
// between bytes, even with every sign set; exhaustive host controls verify it.
fn iq3_signed_word(magnitude:u32,signs:u32)->u32 {
    let mask=((signs&1u)*255u)|(((signs>>1u)&1u)*255u<<8u)
        |(((signs>>2u)&1u)*255u<<16u)|(((signs>>3u)&1u)*255u<<24u);
    return (magnitude^mask)+(mask&0x01010101u);
}
fn iq4_word(codes:u32,high:bool)->u32 {
    var shift=0u;if high {shift=4u;}
    return iq4_byte((codes>>shift)&15u)
        |(iq4_byte((codes>>(shift+8u))&15u)<<8u)
        |(iq4_byte((codes>>(shift+16u))&15u)<<16u)
        |(iq4_byte((codes>>(shift+24u))&15u)<<24u);
}
fn q2_word(codes:u32)->u32 {
    let q=(codes&3u)|(((codes>>2u)&3u)<<8u)|(((codes>>4u)&3u)<<16u)|(((codes>>6u)&3u)<<24u);
    // Four original codes0..3 map to signed−1..2 independently. Adding127
    // cannot carry across any byte because every intermediate is127..130.
    return (q+0x7f7f7f7fu)^0x80808080u;
}
// Original byte-addressed block geometry, including98/18/34-byte records
// crossing u32 boundaries. Rust emits immutable bindings for gate/up or down.
PROJECTION_HELPERS_VALUE

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) local:u32,
    @builtin(subgroup_id) sg:u32,@builtin(subgroup_invocation_id) lane:u32,
    @builtin(subgroup_size) width:u32) {
    let row=g.x*(128u/width)+sg;var valid=valid_group(g.y);
    var slot=0u;var route=0u;
    if valid {slot=map[p.a+4u+g.y]-FIRST_SLOT;route=map[p.a+164u+g.y*8u];}
    INPUT_READY_VALUE
    // Every workgroup cell is initialized, including inactive groups/rows.
    // Both barriers remain unconditional across all128 invocations.
    if local==0u {atomicStore(&quant_fault,0u);}
    for(var i=local;i<COLUMNS;i+=128u) {
        var x=0.0;if valid {x=input[p.b+INPUT_OFFSET_VALUE+i];}
        cached_input[i]=x;
    }
    GRID_INITIALIZE_VALUE
    workgroupBarrier();
    prepare_q8(local);
    workgroupBarrier();
    let ready=valid && atomicLoad(&quant_fault)==0u;
    EPILOGUE_VALUE
}
