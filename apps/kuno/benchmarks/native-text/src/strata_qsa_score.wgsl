// MIT Strata fb58e0 qsa_select.dp.cpp::block_scores_multi_kernel.
// SCORE_MULTI_BEGIN
@group(0) @binding(0) var<storage,read> pooled:array<f32>;
@group(0) @binding(1) var<storage,read> query:array<f32>;
@group(0) @binding(2) var<storage,read_write> scores:array<f32>;
@group(0) @binding(3) var<storage,read_write> status:array<u32>;
@group(0) @binding(4) var<storage,read> retained:array<f32>;
@group(0) @binding(5) var<uniform> p:Params;
var<workgroup> qs:array<f32,4096>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) tid:u32,@builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_id) warp:u32,@builtin(subgroup_size) sw:u32) {
    if sw!=32u || group.y!=0u || group.z!=0u {return;}
    // Positions are the authenticated contiguous window, not inferred from data.
    let top=(p.start+p.count)/4u;
    if group.x*8u>top {return;}
    for(var i=tid;i<p.count*512u;i+=256u) {qs[i]=query[i];}
    workgroupBarrier();
    for(var block=group.x*8u+warp;block<=top&&block<p.output_stride;block+=SCORE_CTAS*8u) {
        let at=block*128u+lane*4u;let dead=(p.capacity/4u)*128u+lane*4u;
        let kp=vec4<f32>(pooled[at],pooled[at+1u],pooled[at+2u],pooled[at+3u]);
        let kd=vec4<f32>(retained[dead],retained[dead+1u],retained[dead+2u],retained[dead+3u]);
        for(var token=0u;token<p.count;token++) {
            let visible=p.start+token+1u;let complete=visible/4u;
            // The Rust score ABI omits the empty block at an exact boundary.
            if block>complete || (block==complete&&visible%4u==0u) {continue;}
            let k=select(kp,kd,block==complete);
            var total=0.0;var good=1u;
            for(var h=0u;h<4u;h++) {
                let off=token*512u+h*128u+lane*4u;
                let q=vec4<f32>(qs[off],qs[off+1u],qs[off+2u],qs[off+3u]);
                var sum=k.x*q.x+k.y*q.y+k.z*q.z+k.w*q.w;
                for(var d=16u;d>0u;d/=2u) {sum+=subgroupShuffleXor(sum,d);}
                if !finite(sum) {good=0u;}
                total+=max(sum,0.0);
            }
            if block==complete {total+=1000000000.0;}
            if lane==0u {
                let ok=good==1u&&finite(total);
                scores[token*p.output_stride+block]=select(0.0,total,ok);
                status[token*p.status_stride+block]=select(0u,1u,ok);
            }
        }
    }
}
// SCORE_MULTI_END
