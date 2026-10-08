var<workgroup> scratch:array<f32,256>;
var<workgroup> normalized:array<f32,256>;
var<workgroup> probability:array<f32,4096>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group:vec3<u32>, @builtin(local_invocation_index) lane:u32) {
    let head=group.x;
    if OP==0u { // Q/K head RMS norm and text-only partial rotary embedding
        let source=p.a+head*select(512u,256u,p.d==1u);
        let x=arena[source+lane]; scratch[lane]=x*x; workgroupBarrier();
        for(var stride=128u;stride>0u;stride/=2u){
            if lane<stride { scratch[lane]+=scratch[lane+stride]; } workgroupBarrier();
        }
        normalized[lane]=x*inverseSqrt(scratch[0]/256.0+1e-6)*bitcast<f32>(weight[lane]);
        workgroupBarrier();
        var v=normalized[lane];
        if lane<64u {
            let i=lane%32u;let angle=f32(ctrl.pos)*pow(10000000.0,-f32(i)*2.0/64.0);
            if lane<32u { v=normalized[lane]*cos(angle)-normalized[lane+32u]*sin(angle); }
            else { v=normalized[lane]*cos(angle)+normalized[lane-32u]*sin(angle); }
        }
        if p.d==0u { arena[p.b+head*256u+lane]=v; }
        else {
            state[ctrl.pos*512u+head*256u+lane]=v;
            state[2097152u+ctrl.pos*512u+head*256u+lane]=arena[p.e+head*256u+lane];
        }
    } else { // softmax, weighted values, and the independent sigmoid output gate
        var maximum=-3.402823e38;
        for(var t=lane;t<=ctrl.pos;t+=256u){maximum=max(maximum,arena[p.a+head*4096u+t]);}
        scratch[lane]=maximum;workgroupBarrier();
        for(var stride=128u;stride>0u;stride/=2u){
            if lane<stride {scratch[lane]=max(scratch[lane],scratch[lane+stride]);}workgroupBarrier();
        }
        maximum=scratch[0];var total=0.0;
        for(var t=lane;t<=ctrl.pos;t+=256u){
            let v=exp(arena[p.a+head*4096u+t]-maximum);probability[t]=v;total+=v;
        }
        scratch[lane]=total;workgroupBarrier();
        for(var stride=128u;stride>0u;stride/=2u){
            if lane<stride {scratch[lane]+=scratch[lane+stride];}workgroupBarrier();
        }
        total=scratch[0];let kv=head/8u;var v=0.0;
        for(var t=0u;t<=ctrl.pos;t++){
            v+=probability[t]/total*state[2097152u+t*512u+kv*256u+lane];
        }
        arena[p.b+head*256u+lane]=v*sigmoid(arena[p.c+head*512u+256u+lane]);
    }
}
