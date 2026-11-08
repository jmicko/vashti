// Packed F32/subgroup Verify3 candidate; adapted from ornith/matrix_vector.wgsl.
// Existing packed decode helpers/NOTICE and all original weight formats reused.
// Fixed public three-column policy. Original F32 inputs, no quantization or
// extra private scratch. Each decoded original vec4 is reused across only the
// runtime-active columns; each component keeps the accepted count1 dot order.
fn verify3_dot(w:vec4<f32>,column4:u32)->vec3<f32> {
    var result=vec3<f32>(0.0);
    result.x=dot(w,input4[column4]);
    if p.count>=2u {result.y=dot(w,input4[column4+p.input_stride/4u]);}
    if p.count>=3u {result.z=dot(w,input4[column4+2u*(p.input_stride/4u)]);}
    return result;
}
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(subgroup_id) sg:u32,
        @builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_size) width:u32,
        @builtin(local_invocation_index) local:u32) {
    // Uniform guard before any weight/input/output access. Count never keys a
    // pipeline. The parent retains its independent complete owning finite/fault
    // proof, inactive/gap/padding audits and unchanged whole-chunk acceptance.
    if arrayLength(&fault)!=1u {return;}
    var valid=(width==16u || width==32u) && p.count>=1u && p.count<=3u
        && p.reserved==0u && p.input_first%4u==0u && p.input_stride%4u==0u
        && p.input_stride>=COLUMNS && p.output_stride>=ROWS
        && p.input_words<=arrayLength(&input4)*4u
        && p.output_words<=arrayLength(&output)
        && arrayLength(&weight)>=(ROWS*ROW_BYTES+3u)/4u;
    if p.input_first>p.input_words || p.output_first>p.output_words {valid=false;}
    if valid {
        let input_left=p.input_words-p.input_first;
        let output_left=p.output_words-p.output_first;
        if COLUMNS>input_left || ROWS>output_left {valid=false;}
        if valid {
            if p.count-1u>(input_left-COLUMNS)/p.input_stride
                || p.count-1u>(output_left-ROWS)/p.output_stride {valid=false;}
        }
    }
    if !valid {
        if local==0u {atomicOr(&fault[0],8u);}
        return;
    }
    // VERIFY3_COMPLETE_INTERVAL_GUARD
    let row=group.x*(128u/width)+sg;
    let base=row*ROW_BYTES;let input=p.input_first;
    var sum=vec3<f32>(0.0);
    if row<ROWS {
        if DTYPE==12u {
            for(var block=0u;block<COLUMNS/256u;block++) {
                let b=base/4u+block*36u;
                let factors=unpack2x16float(weight[b]);
                let a=weight[b+1u];let z=weight[b+2u];let scales=weight[b+3u];
                for(var word=lane;word<32u;word+=width) {
                    let pair=word/8u;let i=(word%8u)*4u;
                    let q=bytes4(weight[b+4u+word]);
                    let low=vec4<f32>(q&vec4<u32>(15u))*factors.x*q4_scale(a,scales,pair*2u)
                        -vec4<f32>(factors.y*q4_minimum(z,scales,pair*2u));
                    let high=vec4<f32>(q>>vec4<u32>(4u))*factors.x*q4_scale(a,scales,pair*2u+1u)
                        -vec4<f32>(factors.y*q4_minimum(z,scales,pair*2u+1u));
                    let col=(input+block*256u+pair*64u+i)/4u;
                    sum+=verify3_dot(low,col)+verify3_dot(high,col+8u);
                }
            }
        } else if DTYPE==13u || DTYPE==20u || DTYPE==23u {
            for(var c=lane*4u;c<COLUMNS;c+=width*4u) {
                sum+=verify3_dot(dense_added_packed(base,c),(input+c)/4u);
            }
        } else if DTYPE==14u {
            for(var blockgroup=0u;blockgroup<COLUMNS/256u;blockgroup+=width/16u) {
                let block=blockgroup+lane/16u;let word=lane%16u;
                if block<COLUMNS/256u {
                let half=word/8u;let i=(word%8u)*4u;
                let b=base+block*210u;let d=wh(b+208u);
                let l0=bytes4(word_at(b+half*64u+i));let l1=bytes4(word_at(b+half*64u+32u+i));
                let h=bytes4(word_at(b+128u+half*32u+i));let sc=b+192u+half*8u+i/16u;
                let q0=vec4<i32>((l0&vec4<u32>(15u))|((h&vec4<u32>(3u))<<vec4<u32>(4u)))-vec4<i32>(32);
                let q1=vec4<i32>((l1&vec4<u32>(15u))|(((h>>vec4<u32>(2u))&vec4<u32>(3u))<<vec4<u32>(4u)))-vec4<i32>(32);
                let q2=vec4<i32>((l0>>vec4<u32>(4u))|(((h>>vec4<u32>(4u))&vec4<u32>(3u))<<vec4<u32>(4u)))-vec4<i32>(32);
                let q3=vec4<i32>((l1>>vec4<u32>(4u))|(((h>>vec4<u32>(6u))&vec4<u32>(3u))<<vec4<u32>(4u)))-vec4<i32>(32);
                let col=(input+block*256u+half*128u+i)/4u;
                sum+=(d*signed(wb(sc)))*verify3_dot(vec4<f32>(q0),col);
                sum+=(d*signed(wb(sc+2u)))*verify3_dot(vec4<f32>(q1),col+8u);
                sum+=(d*signed(wb(sc+4u)))*verify3_dot(vec4<f32>(q2),col+16u);
                sum+=(d*signed(wb(sc+6u)))*verify3_dot(vec4<f32>(q3),col+24u);
                }
            }
        } else if DTYPE==6u {
            for(var group=0u;group<COLUMNS/32u;group+=width/4u) {
                let block=group+lane/4u;let i=(lane%4u)*4u;
                if block<COLUMNS/32u {
                    let b=base+block*22u;let d=wh(b);let high=word_at(b+2u);
                    let low=bytes4(word_at(b+6u+i));
                    let shifts=vec4<u32>(i,i+1u,i+2u,i+3u);
                    let q0=vec4<i32>((low&vec4<u32>(15u))|(((vec4<u32>(high)>>shifts)&vec4<u32>(1u))<<vec4<u32>(4u)))-vec4<i32>(16);
                    let q1=vec4<i32>((low>>vec4<u32>(4u))|(((vec4<u32>(high)>>(shifts+vec4<u32>(16u)))&vec4<u32>(1u))<<vec4<u32>(4u)))-vec4<i32>(16);
                    let col=(input+block*32u+i)/4u;
                    sum+=d*(verify3_dot(vec4<f32>(q0),col)+verify3_dot(vec4<f32>(q1),col+4u));
                }
            }
        } else if DTYPE==8u {
            for(var group=0u;group<COLUMNS/32u;group+=width/8u) {
                let block=group+lane/8u;let i=(lane%8u)*4u;
                if block<COLUMNS/32u {
                    let b=base+block*34u;let bytes=bytes4(word_at(b+2u+i));
                    let q=vec4<i32>(bytes)-select(vec4<i32>(0),vec4<i32>(256),bytes>=vec4<u32>(128u));
                    sum+=wh(b)*verify3_dot(vec4<f32>(q),(input+block*32u+i)/4u);
                }
            }
        } else if DTYPE==1u || DTYPE==30u {
            for(var c=lane*4u;c<COLUMNS;c+=width*4u) {
                let b=base/4u+c/2u;let a=weight[b];let z=weight[b+1u];
                var w:vec4<f32>;
                if DTYPE==1u {w=vec4<f32>(unpack2x16float(a),unpack2x16float(z));}
                else {w=bitcast<vec4<f32>>(vec4<u32>(a<<16u,a&0xffff0000u,z<<16u,z&0xffff0000u));}
                sum+=verify3_dot(w,(input+c)/4u);
            }
        } else {
            for(var c=lane*4u;c<COLUMNS;c+=width*4u) {
                let b=base/4u+c;
                let w=bitcast<vec4<f32>>(vec4<u32>(weight[b],weight[b+1u],weight[b+2u],weight[b+3u]));
                sum+=verify3_dot(w,(input+c)/4u);
            }
        }
    }
    let total=subgroupAdd(sum);
    if row<ROWS && lane==0u {
        let out=p.output_first+row;
        output[out]=total.x;
        if !(abs(total.x)<=3.4028234663852886e38) {atomicOr(&fault[0],1u);}
        if p.count>=2u {
            output[out+p.output_stride]=total.y;
            if !(abs(total.y)<=3.4028234663852886e38) {atomicOr(&fault[0],1u);}
        }
        if p.count>=3u {
            output[out+2u*p.output_stride]=total.z;
            if !(abs(total.z)<=3.4028234663852886e38) {atomicOr(&fault[0],1u);}
        }
    }
    // VERIFY3_LOGICAL_COMPLEMENT_SCAN
}
