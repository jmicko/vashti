// Strata fb58e0 verify.cpp fetch_blobs/rebase_ptrs, translated to Rust Vulkan.
// Only original HOST weight words are copied, once for each actual union group.
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) lane:u32) {
    if p.count<1u || p.count>4u || p.role>2u || p.vectors==0u || p.vectors>419430u
        || arrayLength(&stage)!=4194304u || arrayLength(&addresses)!=2048u
        || p.first>arrayLength(&map) || 804u>arrayLength(&map)-p.first {return;}
    if map[p.first]!=p.count || map[p.first+1u]!=0u || map[p.first+2u]!=p.count {return;}
    let groups=map[p.first+3u];
    if groups>p.count*10u || g.y>=groups {return;}
    let slot=map[p.first+4u+g.y];
    if slot<240u || slot>=512u {return;}
    var shard=4u;var first=496u;var slots=16u;
    if slot<496u {shard=3u;first=368u;slots=128u;}
    if slot<368u {shard=2u;first=240u;slots=128u;}
    if arrayLength(&gate0)!=120u*p.vectors || arrayLength(&gate1)!=120u*p.vectors
        || arrayLength(&gate2)!=128u*p.vectors || arrayLength(&gate3)!=128u*p.vectors
        || arrayLength(&gate4)!=16u*p.vectors {return;}
    if slots*p.vectors!=addresses[p.role*5u+shard].z/4u {return;}
    strata_gate_address=addresses[p.role*5u+shard].xy;
    let at=g.x*256u+lane;
    if at<p.vectors {stage[g.y*p.vectors+at]=gate0[(slot-first)*p.vectors+at];}
}
