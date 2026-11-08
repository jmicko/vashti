// The prepare guard is inserted before the unchanged original32-value body.
let block=id.x;let member=id.y;
if arrayLength(&fault)==0u || p.c>=arrayLength(&fault) {return;}
var span_valid=p.g>=1u && p.g<=4u && p.f==0u && p.h==0u
    && p.d>=COLUMNS && p.e>=2u*PACKED_WORDS+2u*BLOCKS;
if p.a>arrayLength(&input) || p.b>arrayLength(&planes) {span_valid=false;}
if span_valid {
    let il=arrayLength(&input)-p.a;let pl=arrayLength(&planes)-p.b;
    let pw=2u*PACKED_WORDS+2u*BLOCKS;
    if COLUMNS>il || pw>pl {span_valid=false;}
    if span_valid {
        if p.g-1u>(il-COLUMNS)/p.d || p.g-1u>(pl-pw)/p.e {span_valid=false;}
    }
}
if !span_valid {if block==0u && member==0u {atomicOr(&fault[p.c],8u);}return;}
if block>=COLUMNS/32u || member>=p.g {return;}
let input_first=p.a+member*p.d;let planes_first=p.b+member*p.e;
// PROJECT_BATCH_STAGE
// Original Q6 code/factor decode is shared by the actual RHS rows. The scalar
// integer quartet/first/residual/scale order stays unchanged for each member.
fn member_pair(codes:vec4<u32>,group:u32,member:u32)->f32 {
    let first=p.b+member*p.h;var a=0;var b=0;
    for(var word=0u;word<4u;word++) {
        let q=codes[word];let at=group*4u+word;
        a+=dot4I8Packed(q,planes[first+at]);
        b+=dot4I8Packed(q,planes[first+PACKED_WORDS+at]);
    }
    let d=bitcast<f32>(planes[first+2u*PACKED_WORDS+group/2u]);
    let rd=bitcast<f32>(planes[first+2u*PACKED_WORDS+BLOCKS+group/2u]);
    return d*f32(a)+rd*f32(b);
}
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) local:u32,
    @builtin(subgroup_id) sg:u32,@builtin(subgroup_invocation_id) lane:u32,
    @builtin(subgroup_size) width:u32,@builtin(num_workgroups) dispatched:vec3<u32>) {
    if arrayLength(&fault)==0u || p.d>=arrayLength(&fault) {return;}
    var valid=(width==16u || width==32u) && p.g>=1u && p.g<=MEMBERS
        && (!FOUR || p.g==4u) && p.j==0u && p.k==0u && p.l==0u
        && p.h>=PLANE_WORDS && p.i>=p.f && p.f>=ROWS;
    let weight_bytes=arrayLength(&weight)*WEIGHT_WORD_BYTES;
    if p.a>weight_bytes || p.b>arrayLength(&planes)
        || p.c>arrayLength(&output) || p.e>arrayLength(&output) {valid=false;}
    if valid {
        let pl=arrayLength(&planes)-p.b;let ol=arrayLength(&output)-p.c;
        let ml=arrayLength(&output)-p.e;
        if ROWS*ROW_BYTES>weight_bytes-p.a || PLANE_WORDS>pl || ROWS>ol
            || p.f>ml || p.c<p.e {valid=false;}
        if valid {
            if p.c-p.e>p.f-ROWS || p.g-1u>(pl-PLANE_WORDS)/p.h
                || p.g-1u>(ol-ROWS)/p.i || p.g-1u>(ml-p.f)/p.i {valid=false;}
        }
    }
    if !valid {if local==0u {atomicOr(&fault[p.d],8u);}return;}
    let row=g.x*(128u/width)+sg;var sum=VECTOR<f32>(0.0);
    if row<ROWS && atomicLoad(&fault[p.d])==0u {
        let row_base=p.a+row*ROW_BYTES;
        for(var group=lane;group<COLUMNS/16u;group+=width) {
            let base=row_base+(group/16u)*210u;let sub=group%16u;
            let half_bits=byte_at(base+208u)|(byte_at(base+209u)<<8u);
            let factor=unpack2x16float(half_bits).x*f32(signed_byte(byte_at(base+192u+sub)));
            let codes=q6_words(base,sub);
            let x=member_pair(codes,group,0u);
            if !finite(factor) || !finite(x) {atomicOr(&fault[p.d],1u);}
            sum.x+=factor*x;
            if p.g>=2u {
                let y=member_pair(codes,group,1u);
                if !finite(factor) || !finite(y) {atomicOr(&fault[p.d],1u);}
                sum.y+=factor*y;
            }
            if p.g>=3u {
                let z=member_pair(codes,group,2u);
                if !finite(factor) || !finite(z) {atomicOr(&fault[p.d],1u);}
                sum.z+=factor*z;
            }
            // FOUR_SUM_BEGIN
            let w=member_pair(codes,group,3u);
            if !finite(factor) || !finite(w) {atomicOr(&fault[p.d],1u);}
            sum.w+=factor*w;
            // FOUR_SUM_END
        }
    }
    let total=subgroupAdd(sum);
    if row<ROWS && lane==0u {
        if !finite(total.x) || (p.g>=2u && !finite(total.y))
            || (p.g>=3u && !finite(total.z)) {atomicOr(&fault[p.d],1u);}
        // FOUR_FINITE_BEGIN
        if !finite(total.w) {atomicOr(&fault[p.d],1u);}
        // FOUR_FINITE_END
        if atomicLoad(&fault[p.d])==0u {
            output[p.c+row]=total.x;
            if p.g>=2u {output[p.c+p.i+row]=total.y;}
            if p.g>=3u {output[p.c+2u*p.i+row]=total.z;}
            // FOUR_OUTPUT_BEGIN
            output[p.c+3u*p.i+row]=total.w;
            // FOUR_OUTPUT_END
        }
    }
    if AUDIT_TAIL {
        let index=g.x*128u+local;
        for(var i=index;i<arrayLength(&output);i+=128u*dispatched.x) {
            var matrix_cell=false;
            for(var member=0u;member<p.g;member++) {
                let first=p.e+member*p.i;
                matrix_cell=matrix_cell || (i>=first && i<first+p.f);
            }
            if !matrix_cell && !finite(output[i]) {atomicOr(&fault[p.d],1u);}
        }
    }
}
// PROJECT_PAIRED_STAGE
// Paired original32-value mapping follows pinned MIT Strata fb58e0's wide
// Q6 kernel. q6_integer_fixtures/NOTICE retains its original MIT notice.
struct PairedCodes {lower:vec4<u32>,upper:vec4<u32>}
fn paired_codes(base:u32,position:u32)->PairedCodes {
    let shift=2u*((position&3u)>>1u);
    let high_index=2u*(position>>2u)+(position&1u);
    let low=pair_words_at(base+16u*position);
    let high=pair_words_at(base+128u+16u*high_index)>>vec4<u32>(shift);
    let lower=(low&vec4<u32>(0x0f0f0f0fu))
        |((high<<vec4<u32>(4u))&vec4<u32>(0x30303030u));
    let upper=((low>>vec4<u32>(4u))&vec4<u32>(0x0f0f0f0fu))
        |(((high>>vec4<u32>(4u))<<vec4<u32>(4u))&vec4<u32>(0x30303030u));
    return PairedCodes((lower+vec4<u32>(0x60606060u))^vec4<u32>(0x80808080u),
        (upper+vec4<u32>(0x60606060u))^vec4<u32>(0x80808080u));
}
// PAIRED_LOOP
        let position=lane&7u;let initial_block=lane>>3u;
        let scale_offset=8u*(position>>2u)+(position&3u);
        for(var block=initial_block;block<COLUMNS/256u;block+=width/8u) {
            let base=row_base+block*210u;
            let half_bits=byte_at(base+208u)|(byte_at(base+209u)<<8u);
            let d=unpack2x16float(half_bits).x;
            let factor0=d*f32(signed_byte(byte_at(base+192u+scale_offset)));
            let factor1=d*f32(signed_byte(byte_at(base+192u+scale_offset+4u)));
            let codes=paired_codes(base,position);
            let group0=block*16u+scale_offset;let group1=group0+4u;
            let x0=member_pair(codes.lower,group0,0u);
            let x1=member_pair(codes.upper,group1,0u);
            if !finite(factor0) || !finite(factor1) || !finite(x0) || !finite(x1) {atomicOr(&fault[p.d],1u);}
            sum.x+=factor0*x0;sum.x+=factor1*x1;
            if p.g>=2u {
                let y0=member_pair(codes.lower,group0,1u);
                let y1=member_pair(codes.upper,group1,1u);
                if !finite(factor0) || !finite(factor1) || !finite(y0) || !finite(y1) {atomicOr(&fault[p.d],1u);}
                sum.y+=factor0*y0;sum.y+=factor1*y1;
            }
            if p.g>=3u {
                let z0=member_pair(codes.lower,group0,2u);
                let z1=member_pair(codes.upper,group1,2u);
                if !finite(factor0) || !finite(factor1) || !finite(z0) || !finite(z1) {atomicOr(&fault[p.d],1u);}
                sum.z+=factor0*z0;sum.z+=factor1*z1;
            }
            // FOUR_SUM_BEGIN
            let w0=member_pair(codes.lower,group0,3u);
            let w1=member_pair(codes.upper,group1,3u);
            if !finite(factor0) || !finite(factor1) || !finite(w0) || !finite(w1) {atomicOr(&fault[p.d],1u);}
            sum.w+=factor0*w0;sum.w+=factor1*w1;
            // FOUR_SUM_END
        }
