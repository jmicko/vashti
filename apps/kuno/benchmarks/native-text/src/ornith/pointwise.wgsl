var<workgroup> scratch:array<f32,256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group:vec3<u32>, @builtin(local_invocation_index) lane:u32) {
    if OP==0u { // token embedding
        for(var i=group.x*256u+lane;i<2048u;i+=2048u) {
            arena[i]=packed(ctrl.token*ROW_BYTES,i);
        }
    } else if OP==1u { // RMS norm, optionally add the attention residual first
        let n=p.c; let head=group.x;
        var sum=0.0;
        for(var i=lane;i<n;i+=256u) {
            let ix=p.a+head*n+i;
            var v=arena[ix];
            if p.d!=0u { v+=arena[p.e+head*n+i]; arena[ix]=v; }
            sum+=v*v;
        }
        scratch[lane]=sum; workgroupBarrier();
        for(var stride=128u;stride>0u;stride/=2u){
            if lane<stride { scratch[lane]+=scratch[lane+stride]; } workgroupBarrier();
        }
        let inv=inverseSqrt(scratch[0]/f32(n)+1e-6);
        for(var i=lane;i<n;i+=256u) {
            var v=arena[p.a+head*n+i]*inv*bitcast<f32>(weight[i]);
            if p.f!=0u { v*=silu(arena[p.g+head*n+i]); }
            arena[p.b+head*n+i]=v;
        }
    } else if OP==2u { // convolution plus SiLU, channel-major three-token history
        let c=group.x*256u+lane;
        if c<8192u {
            let old=c*3u; let x=arena[p.a+c];
            let v=state[old]*bitcast<f32>(weight[c*4u])+state[old+1u]*bitcast<f32>(weight[c*4u+1u])
                +state[old+2u]*bitcast<f32>(weight[c*4u+2u])+x*bitcast<f32>(weight[c*4u+3u]);
            state[old]=state[old+1u]; state[old+1u]=state[old+2u]; state[old+2u]=x;
            arena[p.b+c]=silu(v);
        }
    } else if OP==3u { // L2-normalize Q and K GDN heads
        let head=group.x; var sum=0.0;
        if lane<128u { let v=arena[p.a+head*128u+lane]; sum=v*v; }
        scratch[lane]=sum;workgroupBarrier();
        for(var stride=128u;stride>0u;stride/=2u){
            if lane<stride { scratch[lane]+=scratch[lane+stride]; } workgroupBarrier();
        }
        if lane<128u { arena[p.a+head*128u+lane]*=inverseSqrt(scratch[0]+1e-6); }
    } else if OP==4u { // rank-specific SiLU gate/up activations
        let i=group.x*256u+lane;
        if i<p.c { arena[p.b+i]=silu(arena[p.a+i])*arena[p.d+i]; }
    } else if OP==5u { // original rank-ordered routed merge, shared gate, FFN residual
        let i=group.x*256u+lane;
        if i<2048u {
            var sum=0.0;
            for(var rank=0u;rank<8u;rank++){ sum+=arena[p.a+rank*2048u+i]*bitcast<f32>(route[8u+rank]); }
            arena[i]+=sum+arena[p.b+i]*sigmoid(arena[p.c]);
        }
    }
}
