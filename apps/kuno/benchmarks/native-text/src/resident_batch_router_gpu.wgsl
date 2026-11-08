// Experimental F32 routing of actual MAX8 scores. No fixed/oracle experts.
// Complete owning preflight and selection/map publication are ordered phases.
const SOURCE_CAPACITY:u32=144u;
struct Params { layer:u32, start:u32, count:u32, margin:u32,
                version:u32, reserved0:u32, reserved1:u32, reserved2:u32 }
@group(0) @binding(0) var<storage,read> router:array<u32>;
@group(0) @binding(1) var<storage,read> table:array<u32>;
@group(0) @binding(2) var<storage,read_write> map:array<u32>;
@group(0) @binding(3) var<storage,read_write> probability:array<u32>;
@group(0) @binding(4) var<storage,read_write> audit:array<u32>;
@group(0) @binding(5) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(6) var<uniform> p:Params;
fn finite(bits:u32)->bool { return abs(bitcast<f32>(bits))<=bitcast<f32>(0x7f7fffffu); }
fn shape()->bool {
    return p.layer<48u && p.count>=1u && p.count<=3u && p.start<4096u &&
      p.start+p.count<=4096u && p.margin==0u && p.version==1u &&
      p.reserved0==0u && p.reserved1==0u && p.reserved2==0u &&
      arrayLength(&router)==8192u && arrayLength(&table)==32768u &&
      arrayLength(&map)==8192u && arrayLength(&probability)==8192u &&
      arrayLength(&audit)==524288u && arrayLength(&fault)==8192u;
}
@compute @workgroup_size(256)
fn preflight(@builtin(global_invocation_id) global:vec3<u32>) {
    let index=global.x;
    if !shape() {
        if index==0u && arrayLength(&fault)>0u {atomicOr(&fault[0],1u);}
        return;
    }
    var bad=false;
    if index<8192u {
        if index<p.count*512u {if !finite(router[index]) {bad=true;}}
        else if router[index]!=0u {bad=true;}
        if map[index]!=0u || probability[index]!=0u {bad=true;}
        if index>0u {if atomicLoad(&fault[index])!=0u {bad=true;}}
    }
    if index<32768u {
        let slot=table[index];
        if index<24576u {if slot!=0xffffffffu && slot>=SOURCE_CAPACITY {bad=true;}}
        else if slot!=0u {bad=true;}
    }
    if index<9336u {if audit[p.layer*9336u+index]!=0u {bad=true;}}
    if index<76160u {if audit[448128u+index]!=0u {bad=true;}}
    if bad {atomicOr(&fault[0],1u);}
}
var<workgroup> bad:atomic<u32>;
var<workgroup> best:array<f32,64>;
var<workgroup> best_id:array<u32,64>;
var<workgroup> picked:array<u32,33>;
var<workgroup> score:array<f32,33>;
var<workgroup> weights:array<f32,30>;
var<workgroup> wire:array<u32,804>;
var<workgroup> group_expert:array<u32,30>;
var<workgroup> minimum_gap:f32;
fn reject(flag:u32) {atomicOr(&bad,flag);}
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) local:u32) {
    if local==0u {
        atomicStore(&bad,0u);
        if !shape() {reject(1u);}
        if arrayLength(&fault)>0u {if atomicLoad(&fault[0])!=0u {reject(1u);}}
        minimum_gap=bitcast<f32>(0x7f7fffffu);
    }
    workgroupBarrier();
    if atomicLoad(&bad)!=0u {
        if local==0u && arrayLength(&fault)>0u {atomicOr(&fault[0],atomicLoad(&bad));}
        return;
    }
    // Top11 makes the inclusion boundary visible, as well as selected-order
    // gaps. Exact raw ties use ascending IDs. The parent must audit exact
    // original IDs/order from actual scores before any all-family commit.
    for(var row=0u;row<p.count;row++) {
        for(var rank=0u;rank<11u;rank++) {
            var value=bitcast<f32>(0xff7fffffu);
            var chosen=512u;
            for(var candidate=local;candidate<512u;candidate+=64u) {
                var used=false;
                for(var previous=0u;previous<rank;previous++) {
                    if picked[row*11u+previous]==candidate {used=true;}
                }
                let candidate_score=bitcast<f32>(router[row*512u+candidate]);
                if !used && (candidate_score>value ||
                   (candidate_score==value && candidate<chosen)) {
                    value=candidate_score; chosen=candidate;
                }
            }
            best[local]=value; best_id[local]=chosen;
            workgroupBarrier();
            for(var stride=32u;stride>0u;stride/=2u) {
                if local<stride {
                    let other=best[local+stride]; let other_id=best_id[local+stride];
                    if other>best[local] || (other==best[local] && other_id<best_id[local]) {
                        best[local]=other; best_id[local]=other_id;
                    }
                }
                workgroupBarrier();
            }
            if local==0u {picked[row*11u+rank]=best_id[0]; score[row*11u+rank]=best[0];}
            workgroupBarrier();
        }
        if local==0u {
            var denominator=0.0;
            for(var rank=0u;rank<10u;rank++) {
                let gap=score[row*11u+rank]-score[row*11u+rank+1u];
                if !finite(bitcast<u32>(gap)) || gap<bitcast<f32>(p.margin) {reject(2u);}
                minimum_gap=min(minimum_gap,gap);
                let difference=score[row*11u+rank]-score[row*11u];
                // Prevent nonfinite subtraction/exp or underflow conversion.
                if !finite(bitcast<u32>(difference)) || difference < -80.0 {reject(2u);}
                else {
                    let weight=exp(difference);
                    if !finite(bitcast<u32>(weight)) || weight<=0.0 {reject(2u);}
                    weights[row*10u+rank]=weight; denominator+=weight;
                }
            }
            if !finite(bitcast<u32>(denominator)) || denominator<=0.0 {reject(2u);}
            else {
                var sum=0.0;
                for(var rank=0u;rank<10u;rank++) {
                    let weight=weights[row*10u+rank]/denominator;
                    if !finite(bitcast<u32>(weight)) || weight<=0.0 || weight>1.0 {reject(2u);}
                    weights[row*10u+rank]=weight; sum+=weight;
                }
                if abs(sum-1.0)>0.000005 {reject(2u);}
            }
        }
        workgroupBarrier();
    }
    for(var index=local;index<804u;index+=64u) {wire[index]=0u;}
    workgroupBarrier();
    if local==0u {
        wire[0]=p.count; wire[2]=p.count;
        // Exact token-major/rank-major first-seen union, not ID/slot sorting.
        for(var route=0u;route<p.count*10u;route++) {
            let expert=picked[(route/10u)*11u+route%10u];
            let slot=table[p.layer*512u+expert];
            if slot>=SOURCE_CAPACITY {reject(4u);}
            var group=wire[3];
            for(var old=0u;old<wire[3];old++) {
                if group_expert[old]==expert {group=old;}
                else if wire[4u+old]==slot {reject(4u);}
            }
            if group==wire[3] {
                group_expert[group]=expert; wire[4u+group]=slot; wire[3]+=1u;
            }
            let length=wire[84u+group];
            if length>=p.count || group>=30u {reject(4u);}
            else {wire[164u+group*8u+length]=route; wire[84u+group]=length+1u;}
        }
    }
    workgroupBarrier();
    if atomicLoad(&bad)!=0u {
        if local==0u {atomicOr(&fault[0],atomicLoad(&bad));}
        return;
    }
    for(var index=local;index<804u;index+=64u) {map[index]=wire[index];}
    for(var index=local;index<80u;index+=64u) {
        var bits=0u; if index<p.count*10u {bits=bitcast<u32>(weights[index]);}
        probability[index]=bits;
    }
    storageBarrier();
    // Audit the exact consumed output buffers after publication and the full
    // actual source owner. Parent rejects unresolved/audit-failed Pending.
    let frame=p.layer*9336u;
    for(var index=local;index<9336u;index+=64u) {
        var bits=0u;
        if index<8192u {bits=router[index];}
        else if index<8436u {
            let j=index-8192u;
            if j==0u {bits=p.count;}
            else if j==2u {bits=p.count*10u;}
            else if j==3u {bits=wire[3];}
            else if j>=4u && j<4u+p.count*30u {
                let route=(j-4u)/3u; let part=(j-4u)%3u;
                if part==0u {bits=picked[(route/10u)*11u+route%10u];}
                else if part==1u {bits=probability[route];}
                else {bits=router[(route/10u)*512u+picked[(route/10u)*11u+route%10u]];}
            }
        } else if index<9240u {bits=map[index-8436u];}
        else if index<9320u {bits=probability[index-9240u];}
        else {
            let j=index-9320u;
            if j==0u {bits=0x4b425231u;}
            else if j==1u {bits=p.version;}
            else if j==2u {bits=p.layer;}
            else if j==3u {bits=p.start;}
            else if j==4u {bits=p.count;}
            else if j==5u {bits=SOURCE_CAPACITY;}
            else if j==6u {bits=wire[3];}
            else if j>=8u && j<8u+p.count*2u {
                let row=(j-8u)/2u;
                if j%2u==0u {bits=picked[row*11u+10u];}
                else {bits=router[row*512u+picked[row*11u+10u]];}
            }
            else if j==14u {bits=bitcast<u32>(minimum_gap);}
            else if j==15u {bits=p.margin;}
        }
        audit[frame+index]=bits;
    }
}
