// MIT Strata fb58e0 Intel native_rope / qsa_decode_attn / FP16 kv append.
// COMMON_BEGIN
struct Params {count:u32,start:u32,capacity:u32,input_stride:u32,output_stride:u32,status_stride:u32,heads:u32,width:u32,block_positions:u32,epsilon:f32,theta_scale:f32,reserved:u32}
fn finite(x:f32)->bool {return abs(x)<=3.4028234663852886e38;}
fn cells(token:u32,start:u32)->u32 {let n=start+token+1u;return min(n,2051u);}
// COMMON_END
// NORM_BEGIN
@group(0) @binding(0) var<storage,read> input:array<f32>;
@group(0) @binding(1) var<storage,read> gamma:array<f32>;
@group(0) @binding(2) var<storage,read> positions:array<u32>;
@group(0) @binding(3) var<storage,read_write> output:array<f32>;
@group(0) @binding(4) var<storage,read_write> status:array<u32>;
@group(0) @binding(5) var<uniform> p:Params;
var<workgroup> sums:array<f32,8>;
var<workgroup> good:array<u32,8>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) tid:u32,@builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_id) warp:u32,@builtin(subgroup_size) sw:u32) {
    let token=group.y;let head=group.x;
    let nh=select(p.heads,(p.start+token+1u)/4u,p.block_positions!=0u);
    if sw!=32u || token>=p.count || head>=nh || group.z!=0u || (p.width!=128u && p.width!=256u) || (p.block_positions!=0u && head<p.start/4u) {return;}
    let pos_at=select(p.start+token,head*4u,p.block_positions!=0u);
    let position=positions[pos_at];
    let base=token*p.input_stride+head*p.width;
    var x=0.0;var sum=0.0;var valid=1u;
    if tid<p.width {x=input[base+tid];sum=x*x;if !finite(x) || !finite(gamma[tid]) {valid=0u;}}
    for(var d=16u;d>0u;d/=2u) {sum+=subgroupShuffleXor(sum,d);valid&=subgroupShuffleXor(valid,d);}
    if lane==0u {sums[warp]=sum;good[warp]=valid;}
    workgroupBarrier();
    sum=0.0;valid=1u;
    if lane<8u {sum=sums[lane];valid=good[lane];}
    for(var d=16u;d>0u;d/=2u) {sum+=subgroupShuffleXor(sum,d);valid&=subgroupShuffleXor(valid,d);}
    let scale=inverseSqrt(sum/f32(p.width)+p.epsilon);
    if position>4095u || !finite(scale) || scale<=0.0 {valid=0u;}
    var y=0.0;var mate=0.0;
    if tid<p.width {y=(scale*x)*gamma[tid];}
    if tid<32u {
        mate=(scale*input[base+tid+32u])*gamma[tid+32u];
        let angle=f32(position)*pow(p.theta_scale,f32(tid));let c=cos(angle);let s=sin(angle);
        let low=fma(y,c,-mate*s);let high=fma(y,s,mate*c);
        output[token*p.output_stride+head*p.width+tid]=select(0.0,low,valid==1u && finite(low));
        output[token*p.output_stride+head*p.width+tid+32u]=select(0.0,high,valid==1u && finite(high));
        if !finite(low) || !finite(high) {valid=0u;}
    } else if tid>=64u && tid<p.width {
        output[token*p.output_stride+head*p.width+tid]=select(0.0,y,valid==1u && finite(y));
        if !finite(y) {valid=0u;}
    }
    for(var d=16u;d>0u;d/=2u) {valid&=subgroupShuffleXor(valid,d);}
    if lane==0u {good[warp]=valid;}
    workgroupBarrier();
    if tid==0u {var ok=1u;for(var w=0u;w<8u;w++) {ok&=good[w];}status[token*p.status_stride+head]=ok;}
}
// NORM_END
// PACK_BEGIN
@group(0) @binding(0) var<storage,read> key:array<f32>;
@group(0) @binding(1) var<storage,read> value:array<f32>;
@group(0) @binding(2) var<storage,read> index_key:array<f32>;
@group(0) @binding(3) var<storage,read_write> candidate:array<u32>;
@group(0) @binding(4) var<storage,read_write> status:array<u32>;
@group(0) @binding(5) var<uniform> p:Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    let t=id.y;let i=id.x;if t>=p.count || i>=640u {return;}
    if i<512u {
        let off=(i%256u)*2u;var a=0.0;var b=0.0;
        if i<256u {a=key[t*512u+off];b=key[t*512u+off+1u];}else {a=value[t*512u+off];b=value[t*512u+off+1u];}
        let bits=pack2x16float(vec2<f32>(a,b));let restored=unpack2x16float(bits);
        let ok=finite(a) && finite(b) && finite(restored.x) && finite(restored.y);
        candidate[t*640u+i]=select(0u,bits,ok);
        let flag=select(0u,1u,ok);let first=(i/256u)*512u+off;
        status[t*p.status_stride+first]=flag;status[t*p.status_stride+first+1u]=flag;
    }else {
        let raw=index_key[t*128u+i-512u];let x=unpack2x16float(pack2x16float(vec2<f32>(raw,0.0))).x;candidate[t*640u+i]=select(0u,bitcast<u32>(x),finite(raw) && finite(x));
        status[t*p.status_stride+1024u+i-512u]=select(0u,1u,finite(raw) && finite(x));
    }
}
// PACK_END
// RETAINED_POOL_BEGIN
@group(0) @binding(0) var<storage,read> retained:array<f32>;
@group(0) @binding(1) var<storage,read_write> pooled:array<f32>;
@group(0) @binding(2) var<storage,read_write> mean_status:array<u32>;
@group(0) @binding(3) var<storage,read_write> norm_status:array<u32>;
@group(0) @binding(4) var<uniform> p:Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    let t=id.y;let i=id.x;if t>=p.count || i>=(p.start/4u)*128u {return;}
    let value=retained[i];pooled[t*p.output_stride+i]=value;
    if i%128u==0u {mean_status[t*p.status_stride+i/128u]=1u;norm_status[t*p.status_stride+i/128u]=1u;}
}
// RETAINED_POOL_END
// PUBLISH_POOL_BEGIN
@group(0) @binding(0) var<storage,read> pooled:array<f32>;
@group(0) @binding(1) var<storage,read_write> retained:array<f32>;
@group(0) @binding(2) var<uniform> p:Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    let i=id.x;let last=p.count-1u;
    if i<(p.start/4u)*128u || i>=((p.start+p.count)/4u)*128u {return;}
    retained[i]=pooled[last*p.input_stride+i];
}
// PUBLISH_POOL_END
// POOL_BEGIN
@group(0) @binding(0) var<storage,read> cache:array<u32>;
@group(0) @binding(1) var<storage,read> candidate:array<u32>;
@group(0) @binding(2) var<storage,read_write> means:array<f32>;
@group(0) @binding(3) var<storage,read_write> status:array<u32>;
@group(0) @binding(4) var<uniform> p:Params;
var<workgroup> flags:array<u32,64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32) {
    let t=group.y;let block=group.x;if t>=p.count || block>=(p.start+t+1u)/4u || block<p.start/4u {return;}
    var good=1u;
    for(var tile=0u;tile<2u;tile++) {
        let d=tile*64u+lane;var sum=0.0;
        for(var j=0u;j<4u;j++) {
            let token=block*4u+j;var bits=0u;
            if token<p.start {bits=cache[token*640u+512u+d];}else {bits=candidate[(token-p.start)*640u+512u+d];}
            let x=bitcast<f32>(bits);sum+=x;if !finite(x) {good=0u;}
        }
        let mean=fma(0.25,sum,0.0);means[t*p.output_stride+block*128u+d]=select(0.0,mean,finite(mean));if !finite(mean) {good=0u;}
    }
    flags[lane]=good;workgroupBarrier();
    for(var d=32u;d>0u;d/=2u) {if lane<d {flags[lane]&=flags[lane+d];}workgroupBarrier();}
    if lane==0u {status[t*p.status_stride+block]=flags[0];}
}
// POOL_END
// SCORE_BEGIN
@group(0) @binding(0) var<storage,read> pooled:array<f32>;
@group(0) @binding(1) var<storage,read> query:array<f32>;
@group(0) @binding(2) var<storage,read_write> scores:array<f32>;
@group(0) @binding(3) var<storage,read_write> status:array<u32>;
@group(0) @binding(4) var<storage,read> retained:array<f32>;
@group(0) @binding(5) var<uniform> p:Params;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_id) warp:u32,@builtin(subgroup_size) sw:u32) {
    let token=group.y;let block=group.x*8u+warp;
    let visible=p.start+token+1u;let complete=visible/4u;
    if sw!=32u || token>=p.count || block>complete || (block==complete&&visible%4u==0u) {return;}
    let off=token*p.input_stride+block*128u+lane*4u;
    var k=vec4<f32>(0.0);
    if block<complete{k=vec4<f32>(pooled[off],pooled[off+1u],pooled[off+2u],pooled[off+3u]);}
    else{let at=(p.capacity/4u)*128u+lane*4u;k=vec4<f32>(retained[at],retained[at+1u],retained[at+2u],retained[at+3u]);}
    var total=0.0;var good=1u;
    for(var h=0u;h<4u;h++) {
        let qoff=token*512u+h*128u+lane*4u;
        let q=vec4<f32>(query[qoff],query[qoff+1u],query[qoff+2u],query[qoff+3u]);
        var sum=k.x*q.x+k.y*q.y+k.z*q.z+k.w*q.w;
        for(var d=16u;d>0u;d/=2u) {sum+=subgroupShuffleXor(sum,d);}
        if !finite(sum) {good=0u;}
        total+=max(sum,0.0);
    }
    if block==complete{total+=1000000000.0;}
    if lane==0u {scores[token*p.output_stride+block]=select(0.0,total,good==1u && finite(total));status[token*p.status_stride+block]=select(0u,1u,good==1u && finite(total));}
}
// SCORE_END
// ATTENTION_BEGIN
@group(0) @binding(0) var<storage,read> query:array<f32>;
@group(0) @binding(1) var<storage,read> cache:array<u32>;
@group(0) @binding(2) var<storage,read> candidate:array<u32>;
@group(0) @binding(3) var<storage,read> selected:array<u32>;
@group(0) @binding(4) var<storage,read_write> partial:array<f32>;
@group(0) @binding(5) var<storage,read_write> stats:array<f32>;
@group(0) @binding(6) var<storage,read_write> status:array<u32>;
@group(0) @binding(7) var<uniform> p:Params;
var<workgroup> sq:array<f32,3072>;
var<workgroup> sp:array<f32,768>;
var<workgroup> row:array<u32,64>;
var<workgroup> broken:atomic<u32>;
fn kv(token:u32,offset:u32)->f32 {
    var bits=0u;if token<p.start {bits=cache[token*640u+offset/2u];}else {bits=candidate[(token-p.start)*640u+offset/2u];}
    let v=unpack2x16float(bits);return select(v.x,v.y,(offset&1u)!=0u);
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) tid:u32,@builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_id) warp:u32,@builtin(subgroup_size) sw:u32) {
    let chunk=group.x;let kvh=group.y;let t=group.z;let n=cells(t,p.start);
    if sw!=32u || t>=p.count || kvh>=2u || chunk>=(n+63u)/64u {return;}
    let here=min(64u,n-chunk*64u);let slot=kvh*33u+chunk;
    if tid==0u {atomicStore(&broken,0u);}
    for(var i=tid;i<3072u;i+=256u) {sq[i]=query[t*6144u+kvh*3072u+i];}
    if tid<64u {row[tid]=4096u;if tid<here {let tok=selected[t*2051u+chunk*64u+tid];row[tid]=tok;}}
    workgroupBarrier();
    for(var c=warp;c<64u;c+=8u) {
        if c>=here {if lane<12u {sp[lane*64u+c]=-3.4028234663852886e38;}continue;}
        let token=row[c];
        if token>=p.start+t+1u {atomicOr(&broken,1u);if lane<12u {sp[lane*64u+c]=-3.4028234663852886e38;}continue;}
        var k:array<f32,8>;for(var j=0u;j<8u;j++) {k[j]=kv(token,kvh*256u+lane*8u+j);if !finite(k[j]) {atomicOr(&broken,1u);}}
        for(var h=0u;h<12u;h++) {
            let off=h*256u+lane*8u;
            var dot=k[0]*sq[off];for(var j=1u;j<8u;j++) {dot+=k[j]*sq[off+j];}
            for(var d=16u;d>0u;d/=2u) {dot+=subgroupShuffleXor(dot,d);}
            if lane==0u {let score=dot/16.0;sp[h*64u+c]=score;status[t*p.status_stride+(kvh*12u+h)*n+chunk*64u+c]=select(0u,1u,finite(score));if !finite(score) {atomicOr(&broken,1u);}}
        }
    }
    workgroupBarrier();
    for(var h=warp;h<12u;h+=8u) {
        var maximum=max(sp[h*64u+lane],sp[h*64u+lane+32u]);
        for(var d=16u;d>0u;d/=2u) {maximum=max(maximum,subgroupShuffleXor(maximum,d));}
        var a=0.0;var b=0.0;
        if lane<here && row[lane]<p.start+t+1u {a=exp(sp[h*64u+lane]-maximum);}
        if lane+32u<here && row[lane+32u]<p.start+t+1u {b=exp(sp[h*64u+lane+32u]-maximum);}
        sp[h*64u+lane]=a;sp[h*64u+lane+32u]=b;
        var total=a+b;for(var d=16u;d>0u;d/=2u) {total+=subgroupShuffleXor(total,d);}
        if lane==0u {stats[t*1584u+(slot*12u+h)*2u]=maximum;stats[t*1584u+(slot*12u+h)*2u+1u]=total;}
    }
    workgroupBarrier();
    var accum:array<f32,12>;for(var h=0u;h<12u;h++) {accum[h]=0.0;}
    for(var c=0u;c<here;c++) {
        if row[c]>=p.start+t+1u {continue;}
        let v=kv(row[c],512u+kvh*256u+tid);
        if !finite(v) {atomicOr(&broken,1u);}
        for(var h=0u;h<12u;h++) {accum[h]=fma(sp[h*64u+c],v,accum[h]);}
    }
    for(var h=0u;h<12u;h++) {partial[t*202752u+(slot*12u+h)*256u+tid]=accum[h];if !finite(accum[h]) {atomicOr(&broken,1u);}}
    workgroupBarrier();
    if atomicLoad(&broken)!=0u {
        for(var i=tid;i<here*12u;i+=256u) {let h=i/here;let c=i%here;status[t*p.status_stride+(kvh*12u+h)*n+chunk*64u+c]=0u;}
    }
}
// ATTENTION_END
// MERGE_BEGIN
@group(0) @binding(0) var<storage,read> partial:array<f32>;
@group(0) @binding(1) var<storage,read> stats:array<f32>;
@group(0) @binding(2) var<storage,read_write> output:array<f32>;
@group(0) @binding(3) var<storage,read_write> status:array<u32>;
@group(0) @binding(4) var<uniform> p:Params;
var<workgroup> good:array<u32,8>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) d:u32,@builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_id) warp:u32,@builtin(subgroup_size) sw:u32) {
    let h=group.x;let t=group.y;if sw!=32u || h>=24u || t>=p.count {return;}
    let nc=(cells(t,p.start)+63u)/64u;var maximum=-3.4028234663852886e38;var ok=1u;
    for(var c=0u;c<nc;c++) {let slot=(h/12u)*33u+c;let m=stats[t*1584u+(slot*12u+h%12u)*2u];maximum=max(maximum,m);if !finite(m) {ok=0u;}}
    var total=0.0;var acc=0.0;
    for(var c=0u;c<nc;c++) {
        let slot=(h/12u)*33u+c;let at=t*1584u+(slot*12u+h%12u)*2u;
        let m=stats[at];if m==-3.4028234663852886e38 {continue;}
        let w=exp(m-maximum);total=fma(stats[at+1u],w,total);
        acc=fma(partial[t*202752u+(slot*12u+h%12u)*256u+d],w,acc);
    }
    var y=0.0;if total>0.0 {y=acc/total;}if !finite(y) || !finite(total) || total<=0.0 {ok=0u;}
    output[t*6144u+h*256u+d]=select(0.0,y,ok==1u);
    for(var distance=16u;distance>0u;distance/=2u) {ok&=subgroupShuffleXor(ok,distance);}
    if lane==0u {good[warp]=ok;}workgroupBarrier();
    if d==0u {var valid=1u;for(var i=0u;i<8u;i++) {valid&=good[i];}status[t*p.status_stride+h]=valid;}
}
// MERGE_END
// GATE_BEGIN
@group(0) @binding(0) var<storage,read> gate:array<f32>;
@group(0) @binding(1) var<storage,read_write> output:array<f32>;
@group(0) @binding(2) var<storage,read_write> status:array<u32>;
@group(0) @binding(3) var<uniform> p:Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    let i=id.x;let t=id.y;if t>=p.count || i>=6144u {return;}
    let x=gate[t*12288u+6144u+i];let a=output[t*6144u+i];let y=a/(1.0+exp(-x));
    let ok=finite(x) && finite(a) && finite(y);output[t*6144u+i]=select(0.0,y,ok);status[t*p.status_stride+i]=select(0u,1u,ok);
}
// GATE_END
// APPEND_BEGIN
@group(0) @binding(0) var<storage,read> candidate:array<u32>;
@group(0) @binding(1) var<storage,read_write> cache:array<u32>;
@group(0) @binding(2) var<storage,read> positions:array<u32>;
@group(0) @binding(3) var<storage,read_write> status:array<u32>;
@group(0) @binding(4) var<uniform> p:Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    let i=id.x;let t=id.y;if t>=p.count || i>=640u {return;}
    let position=p.start+t;let bits=candidate[t*640u+i];var ok=positions[position]==position;
    if i<512u {let pair=unpack2x16float(bits);ok=ok && finite(pair.x) && finite(pair.y);let first=i*2u;status[t*p.status_stride+first]=select(0u,1u,ok);status[t*p.status_stride+first+1u]=select(0u,1u,ok);}
    else {ok=ok && finite(bitcast<f32>(bits));status[t*p.status_stride+1024u+i-512u]=select(0u,1u,ok);}
    cache[position*640u+i]=select(0u,bits,ok);
}
// APPEND_END
