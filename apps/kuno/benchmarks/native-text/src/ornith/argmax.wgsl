var<workgroup> values:array<f32,256>;
var<workgroup> ids:array<u32,256>;
var<workgroup> errors:array<u32,256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group:vec3<u32>, @builtin(local_invocation_index) lane:u32) {
    var best=-3.402823e38;var id=4294967295u;var error=0u;
    if OP==0u {
        for(var i=group.x*4096u+lane;i<min((group.x+1u)*4096u,248320u);i+=256u){
            let v=arena[p.a+i];
            if !(abs(v)<=3.402823e38){error=1u;}
            if v>best || (v==best && i<id){best=v;id=i;}
        }
    } else if lane<61u {best=bitcast<f32>(route[32u+lane*3u]);id=route[33u+lane*3u];error=route[34u+lane*3u];}
    values[lane]=best;ids[lane]=id;errors[lane]=error;workgroupBarrier();
    for(var stride=128u;stride>0u;stride/=2u){
        if lane<stride {
            let other=values[lane+stride];let oid=ids[lane+stride];
            if other>values[lane] || (other==values[lane] && oid<ids[lane]){values[lane]=other;ids[lane]=oid;}
            errors[lane]|=errors[lane+stride];
        }
        workgroupBarrier();
    }
    if lane==0u {
        if OP==0u {route[32u+group.x*3u]=bitcast<u32>(values[0]);route[33u+group.x*3u]=ids[0];route[34u+group.x*3u]=errors[0];}
        else {route[256]=ids[0];route[257]=bitcast<u32>(values[0]);route[258]=errors[0];route[259]=0u;}
    }
}
