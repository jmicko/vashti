// MIT Strata Intel GEMM/kernels prompt path, Rust/Vulkan substitution.
// COMMON_BEGIN
struct Params {count:u32,columns:u32,rows:u32,row_bytes:u32,ib:u32,is:u32,ob:u32,os:u32,bfloat:u32,apply:u32,inject:u32,epsilon:f32,r0:u32,r1:u32,r2:u32,r3:u32}
fn finite_value(x:f32)->bool {return abs(x)<=3.4028234663852886e38;}
fn bf16(x:f32)->u32 {let bits=bitcast<u32>(x);return (bits+32767u+((bits>>16u)&1u))>>16u;}
fn image(a:f32,b:f32,kind:u32)->u32 {if kind==1u {return bf16(a)|(bf16(b)<<16u);}return pack2x16float(vec2<f32>(a,b));}
fn restored(bits:u32,odd:bool,kind:u32)->f32 {if kind==1u {return bitcast<f32>(select(bits<<16u,bits&0xffff0000u,odd));}let v=unpack2x16float(bits);return select(v.x,v.y,odd);}
// COMMON_END
// PACK_BEGIN
@group(0) @binding(0) var<storage,read> input:array<f32>;
@group(0) @binding(1) var<storage,read_write> packed:array<u32>;
@group(0) @binding(2) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(3) var<uniform> p:Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {let i=id.x;if i>=p.count*p.columns/2u {return;}let t=i/(p.columns/2u);let col=(i%(p.columns/2u))*2u;let at=p.ib+t*p.is+col;var a=input[at];var b=input[at+1u];let finite=finite_value(a)&&finite_value(b);if p.r0!=0u&&p.bfloat==0u {a=clamp(a,-65504.0,65504.0);b=clamp(b,-65504.0,65504.0);}let bits=image(a,b,p.bfloat);let good=finite&&finite_value(restored(bits,false,p.bfloat))&&finite_value(restored(bits,true,p.bfloat));packed[i]=select(0u,bits,good);if !good {atomicOr(&fault[0],1u);}}
// PACK_END
// DEQUANT_BEGIN
@group(0) @binding(0) var<storage,read> weight:array<u32>;
@group(0) @binding(1) var<storage,read_write> packed:array<u32>;
@group(0) @binding(2) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(3) var<uniform> p:Params;
fn value(row:u32,col:u32)->f32 {
    let group=col/GROUP_VALUES;let q=codes(row*p.row_bytes,group,(col%GROUP_VALUES)/4u);let byte=(q>>((col%4u)*8u))&255u;
    let f=factors(row*p.row_bytes,group);var v=f32(signed_byte(byte));if DTYPE==12u||DTYPE==13u {v=f32(byte);}return f.x*v-f.y;
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {let i=id.x;if i>=p.rows*p.columns/2u {return;}let row=i/(p.columns/2u);let col=i%(p.columns/2u)*2u;let a=value(row,col);let b=value(row,col+1u);let bits=pack2x16float(vec2<f32>(a,b));let v=unpack2x16float(bits);let good=finite_value(v.x)&&finite_value(v.y);packed[i]=select(0u,bits,good);if !good {atomicOr(&fault[0],1u);}}
// DEQUANT_END
// GEMM_BEGIN
@group(0) @binding(0) var<storage,read> input:array<u32>;
@group(0) @binding(1) var<storage,read> weight:array<u32>;
@group(0) @binding(2) var<storage,read_write> output:array<f32>;
@group(0) @binding(3) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(4) var<uniform> p:Params;
var<workgroup> a:array<f32,256>;
var<workgroup> b:array<f32,256>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) tid:u32) {
    let t=g.y*8u+tid/8u;let row=g.x*8u+tid%8u;var sum=0.0;
    for(var k=0u;k<p.columns;k+=32u) {
        for(var i=tid;i<256u;i+=64u) {let tr=g.y*8u+i/32u;let wr=g.x*8u+i/32u;let col=k+i%32u;var av=0.0;var bv=0.0;
            if tr<p.count {av=restored(input[tr*(p.columns/2u)+col/2u],(col%2u)!=0u,p.bfloat);}
            if wr<p.rows {bv=restored(weight[wr*(p.columns/2u)+col/2u],(col%2u)!=0u,p.bfloat);}
            a[i]=av;b[i]=bv;
        }workgroupBarrier();
        for(var j=0u;j<32u;j++) {sum=fma(a[(tid/8u)*32u+j],b[(tid%8u)*32u+j],sum);}workgroupBarrier();
    }
    if t<p.count&&row<p.rows {output[p.ob+t*p.os+row]=select(0.0,sum,finite_value(sum));if !finite_value(sum){atomicOr(&fault[0],1u);}}
}
// GEMM_END
// GR_NORM_BEGIN
@group(0) @binding(0) var<storage,read> previous:array<f32>;
@group(0) @binding(1) var<storage,read> block:array<f32>;
@group(0) @binding(2) var<storage,read> injection:array<f32>;
@group(0) @binding(3) var<storage,read> gamma:array<f32>;
@group(0) @binding(4) var<storage,read_write> residual:array<f32>;
@group(0) @binding(5) var<storage,read_write> normalized:array<f32>;
@group(0) @binding(6) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(7) var<uniform> p:Params;
var<workgroup> sums:array<f32,8>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) tid:u32,
 @builtin(subgroup_id) sg:u32,@builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_size) sw:u32) {
    let t=g.y;let h=g.x;if sw!=32u {if tid==0u {atomicOr(&fault[0],1u);}return;}if t>=p.count||h>=4u {return;}let base=p.ib+t*10240u+h*2560u;var sum=0.0;
    for(var col=tid;col<2560u;col+=256u) {var r=residual[base+col];if p.apply!=0u {r=fma(block[t*2560u+col],injection[t*4u+h],previous[base+col]);residual[base+col]=r;}sum+=r*r;}
    for(var d=16u;d>0u;d/=2u) {sum+=subgroupShuffleXor(sum,d);}if lane==0u {sums[sg]=sum;}workgroupBarrier();sum=0.0;if lane<8u {sum=sums[lane];}
    for(var d=16u;d>0u;d/=2u){sum+=subgroupShuffleXor(sum,d);}let rs=inverseSqrt(sum/2560.0+p.epsilon);
    for(var col=tid;col<2560u;col+=256u) {let x=(residual[base+col]*rs)*gamma[h*2560u+col];normalized[t*10240u+h*2560u+col]=select(0.0,x,finite_value(x));if !finite_value(x){atomicOr(&fault[0],1u);}}
}
// GR_NORM_END
// GR_LOW_BEGIN
@group(0) @binding(0) var<storage,read> down:array<f32>;
@group(0) @binding(1) var<storage,read_write> low:array<f32>;
@group(0) @binding(2) var<storage,read_write> injection:array<f32>;
@group(0) @binding(3) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(4) var<uniform> p:Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {let i=id.x;if i<p.count*320u {let x=down[i]/4.0;let v=x/(1.0+exp(-x));low[i]=select(0.0,v,finite_value(v));if !finite_value(v){atomicOr(&fault[0],1u);}}
    if p.inject!=0u&&i<p.count*4u {let x=injection[i]/4.0;let v=2.0/(1.0+exp(-x));injection[i]=select(0.0,v,finite_value(v));if !finite_value(v){atomicOr(&fault[0],1u);}}
}
// GR_LOW_END
// GR_MIX_BEGIN
@group(0) @binding(0) var<storage,read> normalized:array<f32>;
@group(0) @binding(1) var<storage,read> up:array<f32>;
@group(0) @binding(2) var<storage,read_write> mixed:array<f32>;
@group(0) @binding(3) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(4) var<uniform> p:Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {let i=id.x;if i>=p.count*2560u {return;}let t=i/2560u;let col=i%2560u;var sum=0.0;
    for(var h=0u;h<4u;h++) {let at=t*10240u+h*2560u+col;sum=fma(normalized[at],1.0/(1.0+exp(-up[at])),sum);}let y=sum/4.0;mixed[i]=select(0.0,y,finite_value(y));if !finite_value(y){atomicOr(&fault[0],1u);}}
// GR_MIX_END
// AB_CONTROLS_BEGIN
@group(0) @binding(0) var<storage,read> raw:array<f32>;
@group(0) @binding(1) var<storage,read> params:array<f32>;
@group(0) @binding(2) var<storage,read_write> controls:array<f32>;
@group(0) @binding(3) var<storage,read_write> status:array<u32>;
@group(0) @binding(4) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(5) var<uniform> p:Params;
fn positive_log1p(x:f32)->f32 {let y=1.0+x;if y==1.0{return x;}return log(y)*(x/(y-1.0));}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {let i=id.x;if i>=p.count*p.rows {return;}let t=i/p.rows;let h=i%p.rows;
    let v=raw[t*p.rows*2u+h]+params[h];var sp=v;if v<=20.0 {sp=positive_log1p(exp(v));}let gate=sp*params[p.rows+h];let beta=1.0/(1.0+exp(-raw[t*p.rows*2u+p.rows+h]));
    let ok=finite_value(gate)&&gate<=0.0&&finite_value(beta)&&beta>=0.0&&beta<=1.0;controls[t*p.rows*2u+h]=select(0.0,gate,ok);controls[t*p.rows*2u+p.rows+h]=select(0.0,beta,ok);status[i]=select(0u,1u,ok);if !ok{atomicOr(&fault[0],1u);}}
// AB_CONTROLS_END
