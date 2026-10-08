var<workgroup> query:array<f32,128>;
var<workgroup> key:array<f32,128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) group:vec3<u32>, @builtin(local_invocation_index) j:u32) {
    let head=group.x; let kh=head%16u;
    query[j]=arena[p.a+kh*128u+j]*inverseSqrt(128.0);
    key[j]=arena[p.a+2048u+kh*128u+j];
    workgroupBarrier();
    let biased=arena[p.c+head]+bitcast<f32>(second[head]);
    let softplus=max(biased,0.0)+log(1.0+exp(-abs(biased)));
    let decay=exp(bitcast<f32>(weight[head])*softplus);
    let beta=sigmoid(arena[p.d+head]);
    let base=24576u+head*16384u+j;
    var predicted=0.0;
    for(var i=0u;i<128u;i++){ predicted+=state[base+i*128u]*decay*key[i]; }
    let delta=(arena[p.a+4096u+head*128u+j]-predicted)*beta;
    var output=0.0;
    for(var i=0u;i<128u;i++){
        let v=state[base+i*128u]*decay+key[i]*delta;
        state[base+i*128u]=v; output+=v*query[i];
    }
    arena[p.b+head*128u+j]=output;
}
