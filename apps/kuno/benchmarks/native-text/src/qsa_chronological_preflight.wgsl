// Same original score/status inputs, full owners and sticky model fault.
// This first ordered dispatch writes only fault[0], never selected IDs.
struct Params { start:u32, count:u32, blocks:u32, stride:u32 }
@group(0) @binding(0) var<storage,read> scores:array<u32>;
@group(0) @binding(1) var<storage,read> status:array<u32>;
@group(0) @binding(2) var<storage,read> selected:array<u32>;
@group(0) @binding(3) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(4) var<uniform> p:Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global:vec3<u32>) {
    let i=global.x;
    if p.start>=2048u || p.count==0u || p.count>8u ||
       p.start+p.count>2048u || p.blocks==0u || p.blocks>1024u ||
       p.blocks<(p.start+p.count)/4u || p.stride!=2051u ||
       arrayLength(&scores)!=8192u || arrayLength(&status)!=8192u ||
       arrayLength(&selected)!=32768u || arrayLength(&fault)!=8192u {
        if i==0u && arrayLength(&fault)>0u {atomicOr(&fault[0],4u);}
        return;
    }
    var bad=false;
    if i<8192u {
        let token=i/p.blocks;
        let block=i%p.blocks;
        let row_live=token<p.count && block<(p.start+token+1u)/4u;
        if row_live {
            let value=bitcast<f32>(scores[i]);
            if !(value>=0.0 && value<=bitcast<f32>(0x7f7fffffu)) || status[i]!=1u {
                bad=true;
            }
        } else if scores[i]!=0u || status[i]!=0u {bad=true;}
        // Existing nonzero fault is sticky and suppresses the second phase.
        if i>0u && atomicLoad(&fault[i])!=0u {bad=true;}
    }
    if i<32768u && selected[i]!=0u {bad=true;}
    if bad {atomicOr(&fault[0],4u);}
}
