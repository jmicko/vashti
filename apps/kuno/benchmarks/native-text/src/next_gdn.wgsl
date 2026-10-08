// Four adjacent state columns per invocation, with the original sequential
// f32 prediction/update/readout order. Old state remains immutable.
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
fn finite4(x:vec4<f32>)->vec4<bool> {
    return abs(x)<=vec4<f32>(bitcast<f32>(0x7f7fffffu));
}
fn both(a:vec4<bool>,b:vec4<bool>)->vec4<bool> {
    return vec4<bool>(a.x && b.x,a.y && b.y,a.z && b.z,a.w && b.w);
}
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) group:vec3<u32>, @builtin(local_invocation_index) lane:u32) {
    let h=group.x;let kh=h%KEY_HEADS;let j=lane*4u;
    for(var i=lane;i<HEAD_DIM;i+=32u) {
        q[i]=queries[p.input.x+kh*HEAD_DIM+i];
        k[i]=keys[p.input.y+kh*HEAD_DIM+i];
    }
    workgroupBarrier();
    if j>=HEAD_DIM {return;}
    let log_decay=controls[p.input.w+h];let beta=controls[p.input.w+VALUE_HEADS+h];
    let decay=exp(log_decay);
    let head_valid=abs(log_decay)<=bitcast<f32>(0x7f7fffffu) && log_decay<=0.0
        && abs(beta)<=bitcast<f32>(0x7f7fffffu) && beta>=0.0 && beta<=1.0;
    var valid=vec4<bool>(head_valid);
    var predicted=vec4<f32>(0.0);
    let old_base=p.old.x+h*p.old.y+j;
    let next_base=p.next.x+h*p.next.y+j;
    let out=(p.output.x+h*HEAD_DIM+j)/4u;
    let flag=p.output.y+h*HEAD_DIM+j;
    output[out]=vec4<f32>(0.0);
    for(var c=0u;c<4u;c++){status[flag+c]=0u;}
    for(var i=0u;i<HEAD_DIM;i++) {
        let old_value=previous[(old_base+i*p.old.z)/4u];
        valid=both(valid,finite4(old_value));
        candidate[(next_base+i*p.next.z)/4u]=vec4<f32>(0.0);
        let decayed=old_value*vec4<f32>(decay);
        predicted+=decayed*vec4<f32>(k[i]);
    }
    let delta=(values[(p.input.z+h*HEAD_DIM+j)/4u]-predicted)*vec4<f32>(beta);
    valid=both(both(valid,finite4(predicted)),finite4(delta));
    if any(valid) {
        var readout=vec4<f32>(0.0);
        for(var i=0u;i<HEAD_DIM;i++) {
            let decayed=previous[(old_base+i*p.old.z)/4u]*vec4<f32>(decay);
            let updated=decayed+vec4<f32>(k[i])*delta;
            valid=both(valid,finite4(updated));
            candidate[(next_base+i*p.next.z)/4u]=select(vec4<f32>(0.0),updated,valid);
            readout+=updated*vec4<f32>(q[i]);
        }
        valid=both(valid,finite4(readout));
        output[out]=select(vec4<f32>(0.0),readout,valid);
        for(var c=0u;c<4u;c++){status[flag+c]=select(0u,1u,valid[c]);}
        if !all(valid) {
            for(var i=0u;i<HEAD_DIM;i++) {
                let index=(next_base+i*p.next.z)/4u;
                candidate[index]=select(vec4<f32>(0.0),candidate[index],valid);
            }
        }
    }
}
