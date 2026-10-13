    // Strata-style parallel token routing: each32-lane logical row selects
    // its own top11 while all four rows progress through shared barriers.
    let row=local/32u;let lane=local%32u;let live=row<p.count;
    for(var rank=0u;rank<11u;rank++) {
        var value=bitcast<f32>(0xff7fffffu);var chosen=512u;
        if live {
            for(var candidate=lane;candidate<512u;candidate+=32u) {
                var used=false;
                for(var previous=0u;previous<rank;previous++) {if picked[row*11u+previous]==candidate {used=true;}}
                let x=bitcast<f32>(router[row*512u+candidate]);
                if !used && (x>value || (x==value && candidate<chosen)) {value=x;chosen=candidate;}
            }
        }
        best[local]=value;best_id[local]=chosen;workgroupBarrier();
        for(var stride=16u;stride>0u;stride/=2u) {
            if lane<stride {
                let other=best[local+stride];let id=best_id[local+stride];
                if other>best[local] || (other==best[local] && id<best_id[local]) {best[local]=other;best_id[local]=id;}
            }
            workgroupBarrier();
        }
        if live && lane==0u {picked[row*11u+rank]=best_id[local];score[row*11u+rank]=best[local];}
        workgroupBarrier();
    }
    if lane==0u {
        var gap_min=bitcast<f32>(0x7f7fffffu);
        if live {
            var denominator=0.0;
            for(var rank=0u;rank<10u;rank++) {
                let gap=score[row*11u+rank]-score[row*11u+rank+1u];
                if !finite(bitcast<u32>(gap)) || gap<bitcast<f32>(p.margin) {reject(2u);}
                gap_min=min(gap_min,gap);
                let difference=score[row*11u+rank]-score[row*11u];
                if !finite(bitcast<u32>(difference)) || difference < -80.0 {reject(2u);}
                else {let weight=exp(difference);if !finite(bitcast<u32>(weight)) || weight<=0.0 {reject(2u);}weights[row*10u+rank]=weight;denominator+=weight;}
            }
            if !finite(bitcast<u32>(denominator)) || denominator<=0.0 {reject(2u);}
            else {
                var sum=0.0;
                for(var rank=0u;rank<10u;rank++) {
                    let weight=weights[row*10u+rank]/denominator;
                    if !finite(bitcast<u32>(weight)) || weight<=0.0 || weight>1.0 {reject(2u);}
                    weights[row*10u+rank]=weight;sum+=weight;
                }
                if abs(sum-1.0)>0.000005 {reject(2u);}
            }
        }
        row_gap[row]=gap_min;
    }
    workgroupBarrier();
    if local==0u {for(var r=0u;r<p.count;r++) {minimum_gap=min(minimum_gap,row_gap[r]);}}
    workgroupBarrier();
