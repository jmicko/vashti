// COMMON_BEGIN
struct Params {proposed:u32,start:u32,accepted:u32,stride:u32,r0:u32,r1:u32,r2:u32,r3:u32}
@group(0) @binding(0) var<storage,read_write> cache:array<f32>;
@group(0) @binding(1) var<storage,read_write> positions:array<u32>;
@group(0) @binding(2) var<storage,read_write> pooled:array<f32>;
@group(0) @binding(3) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(4) var<storage,read_write> tail_state:array<f32>;
@group(0) @binding(5) var<storage,read> snapshot:array<f32>;
@group(0) @binding(6) var<storage,read> window:array<f32>;
@group(0) @binding(7) var<uniform> p:Params;
fn accepted_tail(i:u32)->f32 {
    var x=snapshot[i];let slot=i/128u;let d=i%128u;
    if p.accepted!=0u {
        let last=p.start+p.accepted-1u;
        if last>=slot {let cell=last-(last-slot)%4u;if cell>=p.start {x=unpack2x16float(pack2x16float(vec2<f32>(window[(cell-p.start)*128u+d],0.0))).x;}}
    }
    return x;
}
// COMMON_END
// CLEAR_BEGIN
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
    let i=gid.x;
    if p.accepted>p.proposed {if i==0u {atomicOr(&fault[0],1u);}return;}
    let start=p.start+p.accepted;
    let count=p.proposed-p.accepted;
    if i<count*p.stride {cache[start*p.stride+i]=0.0;}
    if i<((p.start+p.proposed)/4u-start/4u)*128u {pooled[(start/4u)*128u+i]=0.0;}
    if i<count {positions[start+i]=0u;}
    if i<384u {tail_state[i]=accepted_tail(i);}
}
// CLEAR_END
// FOLD_BEGIN
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
    let i=gid.x;
    if p.accepted>p.proposed {if i==0u {atomicOr(&fault[0],1u);}return;}
    let start=p.start+p.accepted;
    let count=p.proposed-p.accepted;
    if i<count*p.stride && bitcast<u32>(cache[start*p.stride+i])!=0u {atomicOr(&fault[0],4u);}
    if i<((p.start+p.proposed)/4u-start/4u)*128u && bitcast<u32>(pooled[(start/4u)*128u+i])!=0u {atomicOr(&fault[0],4u);}
    if i<count && positions[start+i]!=0u {atomicOr(&fault[0],4u);}
    if i<384u && bitcast<u32>(tail_state[i])!=bitcast<u32>(accepted_tail(i)){atomicOr(&fault[0],4u);}
}
// FOLD_END
