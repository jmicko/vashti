// Four key partitions per adjacent vec4 column. Old state stays immutable.
// Each partition writes disjoint 32-row state intervals; whole-column validity
// is combined before publication and all invalid intervals are cleared.
struct GdnParams { input:vec4<u32>, old:vec4<u32>, next:vec4<u32>, output:vec4<u32> }
@group(0) @binding(0) var<storage,read> queries:array<f32>;
@group(0) @binding(1) var<storage,read> keys:array<f32>;
@group(0) @binding(2) var<storage,read> values:array<vec4<f32>>;
@group(0) @binding(3) var<storage,read> controls:array<f32>;
@group(0) @binding(4) var<storage,read> previous:array<vec4<f32>>;
@group(0) @binding(5) var<storage,read_write> candidate:array<vec4<f32>>;
@group(0) @binding(6) var<storage,read_write> output:array<vec4<f32>>;
@group(0) @binding(7) var<storage,read_write> status:array<u32>;
@group(0) @binding(8) var<uniform> p:GdnParams;
var<workgroup> q:array<f32,128>;
var<workgroup> k:array<f32,128>;
var<workgroup> sums:array<vec4<f32>,128>;
var<workgroup> masks:array<vec4<u32>,128>;
var<workgroup> final_masks:array<vec4<u32>,32>;
fn finite4(x:vec4<f32>)->vec4<bool> {
    return abs(x)<=vec4<f32>(bitcast<f32>(0x7f7fffffu));
}
fn both(a:vec4<bool>,b:vec4<bool>)->vec4<bool> {
    return vec4<bool>(a.x && b.x,a.y && b.y,a.z && b.z,a.w && b.w);
}
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) group:vec3<u32>, @builtin(local_invocation_index) lane:u32) {
    let h=group.x;let kh=h%KEY_HEADS;let column=lane%32u;let j=column*4u;
    let part=lane/32u;let start=part*32u;
    q[lane]=queries[p.input.x+kh*HEAD_DIM+lane];
    k[lane]=keys[p.input.y+kh*HEAD_DIM+lane];
    workgroupBarrier();
    let log_decay=controls[p.input.w+h];let beta=controls[p.input.w+VALUE_HEADS+h];
    let decay=exp(log_decay);
    let head_valid=abs(log_decay)<=bitcast<f32>(0x7f7fffffu) && log_decay<=0.0
        && abs(beta)<=bitcast<f32>(0x7f7fffffu) && beta>=0.0 && beta<=1.0;
    var valid=vec4<bool>(head_valid);
    var predicted=vec4<f32>(0.0);
    let old_base=p.old.x+h*p.old.y+j;
    let next_base=p.next.x+h*p.next.y+j;
    // Fixed public 32-vector shard; retain the exact first-pass product.
    var decayed_shard:array<vec4<f32>,32>;
    for(var i=start;i<start+32u;i++) {
        let old_value=previous[(old_base+i*p.old.z)/4u];
        valid=both(valid,finite4(old_value));
        let decayed=old_value*vec4<f32>(decay);
        decayed_shard[i-start]=decayed;
        predicted+=decayed*vec4<f32>(k[i]);
    }
    sums[lane]=predicted;masks[lane]=select(vec4<u32>(0u),vec4<u32>(1u),valid);
    workgroupBarrier();
    predicted=vec4<f32>(0.0);valid=vec4<bool>(true);
    for(var r=0u;r<4u;r++) {
        predicted+=sums[column+r*32u];
        valid=both(valid,masks[column+r*32u]==vec4<u32>(1u));
    }
    let delta=(values[(p.input.z+h*HEAD_DIM+j)/4u]-predicted)*vec4<f32>(beta);
    valid=both(both(valid,finite4(predicted)),finite4(delta));
    // All lanes follow the same barrier sequence, including invalid columns.
    var readout=vec4<f32>(0.0);
    for(var i=start;i<start+32u;i++) {
        let decayed=decayed_shard[i-start];
        let updated=decayed+vec4<f32>(k[i])*delta;
        valid=both(valid,finite4(updated));
        candidate[(next_base+i*p.next.z)/4u]=select(vec4<f32>(0.0),updated,valid);
        readout+=updated*vec4<f32>(q[i]);
    }
    // Every invocation finished consuming prediction scratch before reuse.
    workgroupBarrier();
    sums[lane]=readout;masks[lane]=select(vec4<u32>(0u),vec4<u32>(1u),valid);
    workgroupBarrier();
    if part==0u {
        readout=vec4<f32>(0.0);valid=vec4<bool>(true);
        for(var r=0u;r<4u;r++) {
            readout+=sums[column+r*32u];
            valid=both(valid,masks[column+r*32u]==vec4<u32>(1u));
        }
        valid=both(valid,finite4(readout));
        final_masks[column]=select(vec4<u32>(0u),vec4<u32>(1u),valid);
        let out=(p.output.x+h*HEAD_DIM+j)/4u;
        let flag=p.output.y+h*HEAD_DIM+j;
        output[out]=select(vec4<f32>(0.0),readout,valid);
        for(var c=0u;c<4u;c++){status[flag+c]=select(0u,1u,valid[c]);}
    }
    workgroupBarrier();
    valid=final_masks[column]==vec4<u32>(1u);
    if !all(valid) {
        for(var i=start;i<start+32u;i++) {
            let index=(next_base+i*p.next.z)/4u;
            candidate[index]=select(vec4<f32>(0.0),candidate[index],valid);
        }
    }
}
