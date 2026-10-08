// Packed word loads and vector activations; original GGUF, no requantization.
fn bytes4(x:u32)->vec4<u32> { return vec4<u32>(x,x>>8u,x>>16u,x>>24u)&vec4<u32>(255u); }
fn word_at(offset:u32)->u32 {
    let i=offset/4u;let shift=(offset%4u)*8u;
    var result=weight[i]>>shift;
    if shift!=0u {result|=weight[i+1u]<<(32u-shift);}
    return result;
}
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(subgroup_id) sg:u32,
        @builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_size) width:u32) {
    let row=group.x*(128u/width)+sg;
    let rank=group.y;
    var base=row*ROW_BYTES;var input=p.a;
    if ROUTED {base+=(route[rank]*ROWS)*ROW_BYTES;}
    if RANKED_INPUT {input+=rank*COLUMNS;}
    var sum=0.0;
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
                    sum+=dot(low,arena4[col])+dot(high,arena4[col+8u]);
                }
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
                sum+=(d*signed(wb(sc)))*dot(vec4<f32>(q0),arena4[col]);
                sum+=(d*signed(wb(sc+2u)))*dot(vec4<f32>(q1),arena4[col+8u]);
                sum+=(d*signed(wb(sc+4u)))*dot(vec4<f32>(q2),arena4[col+16u]);
                sum+=(d*signed(wb(sc+6u)))*dot(vec4<f32>(q3),arena4[col+24u]);
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
                    sum+=d*(dot(vec4<f32>(q0),arena4[col])+dot(vec4<f32>(q1),arena4[col+4u]));
                }
            }
        } else if DTYPE==8u {
            for(var group=0u;group<COLUMNS/32u;group+=width/8u) {
                let block=group+lane/8u;let i=(lane%8u)*4u;
                if block<COLUMNS/32u {
                    let b=base+block*34u;let bytes=bytes4(word_at(b+2u+i));
                    let q=vec4<i32>(bytes)-select(vec4<i32>(0),vec4<i32>(256),bytes>=vec4<u32>(128u));
                    sum+=wh(b)*dot(vec4<f32>(q),arena4[(input+block*32u+i)/4u]);
                }
            }
        } else if DTYPE==1u || DTYPE==30u {
            for(var c=lane*4u;c<COLUMNS;c+=width*4u) {
                let b=base/4u+c/2u;let a=weight[b];let z=weight[b+1u];
                var w:vec4<f32>;
                if DTYPE==1u {w=vec4<f32>(unpack2x16float(a),unpack2x16float(z));}
                else {w=bitcast<vec4<f32>>(vec4<u32>(a<<16u,a&0xffff0000u,z<<16u,z&0xffff0000u));}
                sum+=dot(w,arena4[(input+c)/4u]);
            }
        } else {
            for(var c=lane*4u;c<COLUMNS;c+=width*4u) {
                let b=base/4u+c;
                let w=bitcast<vec4<f32>>(vec4<u32>(weight[b],weight[b+1u],weight[b+2u],weight[b+3u]));
                sum+=dot(w,arena4[(input+c)/4u]);
            }
        }
    }
    let total=subgroupAdd(sum);
    if row<ROWS && lane==0u {
        let out=p.b+rank*ROWS+row;
        arena4[out/4u][out%4u]=total;
    }
}
