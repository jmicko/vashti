// MIT Strata fb58e0 sampler.dp.cpp; Rust-owned Vulkan device implementation.
struct Params {dims:vec4<u32>,hist:vec4<u32>,filters:vec4<f32>,penalty:vec2<f32>,seed:vec2<u32>,counter:vec2<u32>,destination:vec2<u32>,r0:vec4<u32>,r1:vec4<u32>,r2:vec4<u32>}
@group(0) @binding(0) var<storage,read> logits:array<f32>;
@group(0) @binding(1) var<storage,read> history:array<i32>;
@group(0) @binding(2) var<storage,read_write> candidates:array<vec2<u32>>;
@group(0) @binding(3) var<storage,read_write> output:array<u32>;
@group(0) @binding(4) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(5) var<storage,read> p:Params;
const NV:u32=248320u;const NB:u32=61u;const KM:u32=64u;
fn sentinel()->vec2<u32>{return vec2<u32>(NV,0xff800000u);}
fn finite(x:f32)->bool{return abs(x)<=bitcast<f32>(0x7f7fffffu)&&x==x;}
fn better(a:vec2<u32>,b:vec2<u32>)->vec2<u32>{let x=bitcast<f32>(a.y);let y=bitcast<f32>(b.y);return select(b,a,x>y||(x==y&&a.x<b.x));}
fn reduce_pair(a:vec2<u32>)->vec2<u32>{let x=subgroupMax(bitcast<f32>(a.y));let id=subgroupMin(select(0xffffffffu,a.x,bitcast<f32>(a.y)==x));return vec2<u32>(id,bitcast<u32>(x));}
fn valid()->bool{return p.dims.x>0u&&p.dims.x<=8u&&p.dims.y==NV&&p.dims.z>0u&&p.dims.z<=KM&&p.dims.w==NB&&p.hist.x<=4096u&&p.hist.y<=2147483647u&&(p.hist.y==0u||p.hist.x>0u)&&p.hist.z<=2147483647u&&p.hist.w<=1u&&p.destination.y>=2u&&arrayLength(&logits)>=p.dims.x*NV&&arrayLength(&history)>=max(1u,p.dims.x*p.hist.x)&&(p.hist.w==1u||arrayLength(&candidates)>=p.dims.x*NB*KM)&&arrayLength(&fault)>=1u&&arrayLength(&output)>p.destination.x+(p.dims.x-1u)*p.destination.y+1u&&all(p.r0==vec4<u32>(0u))&&all(p.r1==vec4<u32>(0u))&&all(p.r2==vec4<u32>(0u));}
fn penalize(x:f32,count:u32)->f32{if count==0u{return x;}var y=x;if y<=0.0{y*=p.filters.w;}else{y/=p.filters.w;}return y-(f32(count)*p.penalty.x+p.penalty.y);}
fn count_history(row:u32,id:u32)->u32{var n=0u;let size=min(p.hist.x,p.hist.y);for(var i=p.hist.x-size;i<p.hist.x;i++){if history[row*p.hist.x+i]==i32(id){n++;}}return n;}
// High32 using16-bit products: no shaderInt64 or floating-point integer math.
fn mulhi(a:u32,b:u32)->u32{let al=a&65535u;let ah=a>>16u;let bl=b&65535u;let bh=b>>16u;let w0=al*bl;let t=ah*bl+(w0>>16u);let w2=t>>16u;let w1=(t&65535u)+al*bh;return ah*bh+w2+(w1>>16u);}
fn draw(row:u32)->f32{var c0=p.counter.x+row;var c1=p.counter.y+select(0u,1u,c0<p.counter.x);var c2=p.seed.x;var c3=p.seed.y;for(var r=0u;r<10u;r++){let hi0=mulhi(0x9e3779b9u,c0);let hi1=mulhi(0xbb67ae85u,c2);let lo0=0x9e3779b9u*c0;let lo1=0xbb67ae85u*c2;let n0=hi1^c1^r;let n2=hi0^c3;c0=n0;c1=lo1;c2=n2;c3=lo0;}return f32(c0>>8u)*(1.0/16777216.0);}
