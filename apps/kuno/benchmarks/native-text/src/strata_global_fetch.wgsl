// MIT Strata verify.cpp fetch_blobs/rebase_ptrs, with split original role cells.
// BDA sources are retained by the same native graph and owning Sources object.
// STRATA_PHYSICAL_EXPERTS
struct Params { first:u32,count:u32,vectors:u32,role:u32,r0:u32,r1:u32,r2:u32,r3:u32 }
@group(0) @binding(0) var<storage,read> gate0:array<vec4<u32>>;
@group(0) @binding(1) var<storage,read> map:array<u32>;
@group(0) @binding(2) var<storage,read_write> stage:array<vec4<u32>>;
@group(0) @binding(3) var<storage,read_write> status:array<u32>;
@group(0) @binding(4) var<storage,read> addresses:array<vec4<u32>>;
@group(0) @binding(5) var<storage,read> stage_addresses:array<vec4<u32>>;
@group(0) @binding(6) var<uniform> p:Params;
var<private> strata_gate_address:vec2<u32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) lane:u32) {
    if p.count<1u || p.count>8u || p.role>2u || p.vectors==0u || p.vectors*80u>8388608u
        || arrayLength(&stage)!=8388608u || arrayLength(&addresses)!=2048u
        || arrayLength(&stage_addresses)!=2048u || arrayLength(&status)!=8192u || p.r0!=0u || p.r1!=0u || p.r2!=0u || p.r3!=0u
        || p.first>arrayLength(&map) || 804u>arrayLength(&map)-p.first {return;}
    if map[p.first]!=p.count || map[p.first+1u]!=0u || map[p.first+2u]!=p.count {return;}
    let groups=map[p.first+3u];
    if groups==0u || groups>p.count*10u || g.y>=groups || g.z!=0u {return;}
    let cell=map[p.first+4u+g.y];if cell>=512u {return;}
    let src=addresses[p.role*512u+cell];
    if src.z!=p.vectors*4u || src.w>1u {return;}
    // One real kernel stamp per selected group/role. Next-dispatch barriers
    // cover ALL vector-copy workgroups before this stamp can authorize reads.
    if g.x==0u && lane==0u {status[160u+g.y*3u+p.role]=1u;}
    if src.w==0u {return;}
    strata_gate_address=src.xy;
    let at=g.x*256u+lane;
    if at<p.vectors {stage[g.y*p.vectors+at]=gate0[at];}
}
