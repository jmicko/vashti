// Private ideal-resident count1 mapper. Routing arithmetic remains unchanged.
// All validation precedes every consumed-header or audit store. One workgroup
// owns this entire phase; the following projection uses the same stream.
struct Params { layer:u32, start:u32, reserved0:u32, reserved1:u32 }
@group(0) @binding(0) var<storage,read> packet:array<u32>;
@group(0) @binding(1) var<storage,read> table:array<u32>;
@group(0) @binding(2) var<storage,read_write> map:array<u32>;
@group(0) @binding(3) var<storage,read_write> probability:array<u32>;
@group(0) @binding(4) var<storage,read_write> owner:array<u32>;
@group(0) @binding(5) var<storage,read_write> audit:array<u32>;
@group(0) @binding(6) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(7) var<uniform> p:Params;
var<workgroup> bad:atomic<u32>;
fn reject() { atomicOr(&bad,1u); }
fn finite(bits:u32)->bool { return abs(bitcast<f32>(bits))<=bitcast<f32>(0x7f7fffffu); }
fn id(rank:u32)->u32 { return packet[3073u+rank*2u]; }
fn weight_bits(rank:u32)->u32 { return packet[3074u+rank*2u]; }
fn map_word(index:u32)->u32 {
    if index==0u || index==2u {return 1u;}
    if index==3u {return 10u;}
    if index>=4u && index<14u {return table[p.layer*512u+id(index-4u)];}
    if index>=84u && index<94u {return 1u;}
    if index>=164u && index<244u && (index-164u)%8u==0u {return (index-164u)/8u;}
    return 0u;
}
fn owner_word(index:u32)->u32 {
    if index==0u {return 1u;}
    if index==2u {return 10u;}
    if index>=5u && index<34u && (index-5u)%3u==0u {return (index-5u)/3u;}
    return 0u;
}
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) local:u32) {
    if local==0u {
        atomicStore(&bad,0u);
        if p.layer>=48u || p.start>=4096u || p.reserved0!=0u || p.reserved1!=0u ||
           arrayLength(&packet)!=8192u || arrayLength(&table)!=32768u ||
           arrayLength(&map)!=8192u || arrayLength(&probability)!=8192u ||
           arrayLength(&owner)!=8192u || arrayLength(&audit)!=524288u ||
           arrayLength(&fault)!=8192u {reject();}
    }
    workgroupBarrier();
    if atomicLoad(&bad)!=0u {
        if local==0u && arrayLength(&fault)>0u {atomicOr(&fault[0],4u);}
        return;
    }
    let frame=p.layer*9322u;
    // Packet mixed/logit words are finite; the owning fault and every word
    // outside the exact existing top10 wire interval must remain zero.
    for(var index=local;index<8192u;index+=64u) {
        let bits=packet[index];
        if index<3072u {if !finite(bits) {reject();}}
        else if index<3073u || index>=3093u {if bits!=0u {reject();}}
        if map[index]!=0u || probability[index]!=0u || owner[index]!=0u ||
           atomicLoad(&fault[index])!=0u {reject();}
    }
    // The immutable table is independently bound/validated by the owner.
    // Selected slots are checked again below; missing entries never fallback.
    for(var index=local;index<32768u;index+=64u) {
        let slot=table[index];
        if index<24576u {if slot!=0xffffffffu && slot>=144u {reject();}}
        else if slot!=0u {reject();}
    }
    // Previous accepted layer frames are not scratch. Current frame and the
    // complete allocation tail must be clean before writing this layer once.
    for(var index=local;index<9322u;index+=64u) {
        if audit[frame+index]!=0u {reject();}
    }
    for(var index=447456u+local;index<524288u;index+=64u) {
        if audit[index]!=0u {reject();}
    }
    if local<10u {
        let chosen=id(local);
        let bits=weight_bits(local);
        let weight=bitcast<f32>(bits);
        if chosen>=512u || !finite(bits) || weight<0.0 || weight>1.0 {reject();}
        if chosen<512u {
            let slot=table[p.layer*512u+chosen];
            if slot>=144u {reject();}
            for(var previous=0u;previous<local;previous++) {
                let previous_id=id(previous);
                if previous_id==chosen {reject();}
                if previous_id<512u && table[p.layer*512u+previous_id]==slot {reject();}
            }
            if local>0u {
                let previous=id(local-1u);
                if previous<512u {
                    let old=bitcast<f32>(packet[2560u+previous]);
                    let now=bitcast<f32>(packet[2560u+chosen]);
                    if old<now || (old==now && previous>chosen) {reject();}
                }
            }
        }
    }
    if local==0u {
        var sum=0.0;
        for(var rank=0u;rank<10u;rank++) {sum+=bitcast<f32>(weight_bits(rank));}
        if !(abs(sum-1.0)<=0.00002) {reject();}
    }
    let last=id(9u);
    if last<512u {
        let cutoff=bitcast<f32>(packet[2560u+last]);
        for(var candidate=local;candidate<512u;candidate+=64u) {
            var selected=false;
            for(var rank=0u;rank<10u;rank++) {if id(rank)==candidate {selected=true;}}
            let value=bitcast<f32>(packet[2560u+candidate]);
            if !selected && (value>cutoff || (value==cutoff && candidate<last)) {reject();}
        }
    }
    workgroupBarrier();
    if atomicLoad(&bad)!=0u {
        if local==0u {atomicOr(&fault[0],4u);}
        return;
    }
    for(var index=local;index<804u;index+=64u) {map[index]=map_word(index);}
    for(var index=local;index<80u;index+=64u) {
        var bits=0u;if index<10u {bits=weight_bits(index);} probability[index]=bits;
    }
    for(var index=local;index<244u;index+=64u) {owner[index]=owner_word(index);}
    storageBarrier();
    // Copy the exact headers consumed by projections, not a second computed
    // representation. The original full packet is preserved bit for bit.
    for(var index=local;index<9322u;index+=64u) {
        var bits=0u;
        if index<8192u {bits=packet[index];}
        else if index<8996u {bits=map[index-8192u];}
        else if index<9076u {bits=probability[index-8996u];}
        else if index<9320u {bits=owner[index-9076u];}
        else if index==9320u {bits=p.layer;}
        else {bits=p.start;}
        audit[frame+index]=bits;
    }
}
