    // Each hardware subgroup owns one real row. Cache the512 logits once,
    // then remove each chosen value from its lane-local candidate list. This
    // avoids rereading scores and checking all previous picks at every rank.
    let row=sg;let live=row<p.count;
    var candidates:array<f32,32>;
    for(var i=0u;i<32u;i++) {
        var value=bitcast<f32>(0xff7fffffu);
        let id=lane+i*width;
        if live && id<512u {value=bitcast<f32>(router[row*512u+id]);}
        candidates[i]=value;
    }
    var remaining=0xffffffffu;
    for(var rank=0u;rank<11u;rank++) {
        var value=bitcast<f32>(0xff7fffffu);var chosen=512u;
        for(var i=0u;i<512u/width;i++) {
            let x=candidates[i];let id=lane+i*width;
            if (remaining&(1u<<i))!=0u && (x>value || (x==value && id<chosen)) {value=x;chosen=id;}
        }
        let maximum=subgroupMax(value);
        let id=subgroupMin(select(512u,chosen,value==maximum));
        if live && lane==0u {picked[row*11u+rank]=id;score[row*11u+rank]=maximum;}
        if id<512u && id%width==lane {remaining&=~(1u<<(id/width));}
    }
    workgroupBarrier();
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
        if row<4u {row_gap[row]=gap_min;}
    }
    workgroupBarrier();
    if local==0u {for(var r=0u;r<p.count;r++) {minimum_gap=min(minimum_gap,row_gap[r]);}}
    workgroupBarrier();
