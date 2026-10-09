// Complete ideal-resident owning preflight. Each actual word is checked once
// across bounded parallel groups. Only sticky fault[0] may be written; no
// consumed header or audit word is published by this phase.
struct Params { layer:u32, start:u32, reserved0:u32, reserved1:u32 }
@group(0) @binding(0) var<storage,read> packet:array<u32>;
@group(0) @binding(1) var<storage,read> table:array<u32>;
@group(0) @binding(2) var<storage,read> map:array<u32>;
@group(0) @binding(3) var<storage,read> probability:array<u32>;
@group(0) @binding(4) var<storage,read> owner:array<u32>;
@group(0) @binding(5) var<storage,read> audit:array<u32>;
@group(0) @binding(6) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(7) var<uniform> p:Params;
fn finite(bits:u32)->bool { return abs(bitcast<f32>(bits))<=bitcast<f32>(0x7f7fffffu); }
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global:vec3<u32>) {
    let index=global.x;
    if p.layer>=48u || p.start>=4096u || p.reserved0!=0u || p.reserved1!=0u ||
       arrayLength(&packet)!=8192u || arrayLength(&table)!=32768u ||
       arrayLength(&map)!=8192u || arrayLength(&probability)!=8192u ||
       arrayLength(&owner)!=8192u || arrayLength(&audit)!=524288u ||
       arrayLength(&fault)!=8192u {
        if index==0u && arrayLength(&fault)>0u {atomicOr(&fault[0],4u);}
        return;
    }
    var bad=false;
    if index<8192u {
        let bits=packet[index];
        if index<3072u {if !finite(bits) {bad=true;}}
        else if index<3073u || index>=3093u {if bits!=0u {bad=true;}}
        if map[index]!=0u || probability[index]!=0u || owner[index]!=0u ||
           atomicLoad(&fault[index])!=0u {bad=true;}
    }
    if index<32768u {
        let slot=table[index];
        if index<24576u {if slot!=0xffffffffu && slot>=144u {bad=true;}}
        else if slot!=0u {bad=true;}
    }
    if index<9322u {
        if audit[p.layer*9322u+index]!=0u {bad=true;}
    }
    if index<76832u {
        if audit[447456u+index]!=0u {bad=true;}
    }
    if bad {atomicOr(&fault[0],4u);}
}
