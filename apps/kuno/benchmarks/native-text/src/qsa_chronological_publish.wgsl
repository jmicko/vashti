// Each active original row retains ALL visible positions, chronologically.
// Separate identity and sticky-fault gates precede every selected-ID store.
struct Params { start:u32, count:u32, blocks:u32, stride:u32 }
@group(0) @binding(0) var<storage,read_write> selected:array<u32>;
@group(0) @binding(1) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(2) var<uniform> p:Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global:vec3<u32>) {
    let i=global.x;
    if p.start>=2048u || p.count==0u || p.count>8u ||
       p.start+p.count>2048u || p.blocks==0u || p.blocks>1024u ||
       p.blocks<(p.start+p.count)/4u || p.stride!=2051u ||
       arrayLength(&selected)!=32768u || arrayLength(&fault)!=8192u {
        if i==0u && arrayLength(&fault)>0u {atomicOr(&fault[0],4u);}
        return;
    }
    if atomicLoad(&fault[0])!=0u {return;}
    let token=i/p.stride;
    let column=i%p.stride;
    if token<p.count && column<p.start+token+1u {
        selected[i]=column;
    }
}
