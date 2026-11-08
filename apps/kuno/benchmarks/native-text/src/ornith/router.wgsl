@compute @workgroup_size(1)
fn main() {
    var ids:array<u32,8>; var vals:array<f32,8>;
    for(var rank=0u;rank<8u;rank++) {
        var best=-3.402823e38;var id=0u;
        for(var e=0u;e<256u;e++) {
            var selected=false;
            for(var prior=0u;prior<rank;prior++){ selected=selected||ids[prior]==e; }
            let v=arena[p.a+e];
            if !selected && v>best {best=v;id=e;}
        }
        ids[rank]=id;vals[rank]=best;
    }
    var sum=0.0; let maximum=vals[0];
    for(var rank=0u;rank<8u;rank++){vals[rank]=exp(vals[rank]-maximum);sum+=vals[rank];}
    for(var rank=0u;rank<8u;rank++){route[rank]=ids[rank];route[8u+rank]=bitcast<u32>(vals[rank]/sum);}
}
