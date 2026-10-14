// Merge only completed native CPU rows. GPU rows keep their own actual stamps.
struct Params {count:u32,r0:u32,r1:u32,r2:u32}
@group(0) @binding(0) var<storage,read> cpu:array<f32>;
@group(0) @binding(1) var<storage,read> map:array<u32>;
@group(0) @binding(2) var<storage,read_write> partial:array<f32>;
@group(0) @binding(3) var<storage,read_write> status:array<u32>;
@group(0) @binding(4) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(5) var<uniform> p:Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id:vec3<u32>){
    if p.count<1u||p.count>8u||p.r0!=0u||p.r1!=0u||p.r2!=0u||arrayLength(&cpu)!=262144u
        ||arrayLength(&partial)!=262144u||arrayLength(&map)!=8192u||arrayLength(&status)!=8192u||arrayLength(&fault)!=1u{return;}
    let route=id.y;let col=id.x;if route>=p.count*10u||col>=2560u||id.z!=0u{return;}
    let flag=map[1024u+route];let value=cpu[route*2560u+col];
    if flag>1u||(bitcast<u32>(value)&0x7f800000u)==0x7f800000u {atomicOr(&fault[0],512u);return;}
    if flag==1u {
        partial[route*2560u+col]=value;
        if col==0u {status[route]=1u;status[80u+route]=1u;}
    }else if value!=0.0 {atomicOr(&fault[0],512u);}
}
