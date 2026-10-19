// PromptRoute groups own disjoint token/rank cells. This ordered dispatch
// observes their original complete map/fault owners before publishing a valid
// header for the host grouping. No workgroup can publish a partial prefix.
struct Params { count:u32,layer:u32,epoch:u32,r0:u32 }
@group(0) @binding(0) var<storage,read> logits:array<f32>;
@group(0) @binding(1) var<storage,read> remap:array<u32>;
@group(0) @binding(2) var<storage,read_write> map:array<u32>;
@group(0) @binding(3) var<storage,read_write> probability:array<f32>;
@group(0) @binding(4) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(5) var<storage,read_write> usage:array<f32>;
@group(0) @binding(6) var<uniform> p:Params;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) block:vec3<u32>,@builtin(local_invocation_index) local:u32,
        @builtin(subgroup_size) width:u32,@builtin(num_subgroups) nsg:u32) {
    if p.count<1u || p.count>PROMPT_MAX_ROWS || p.layer>=48u || p.epoch==0u || p.r0>3u
        || width!=32u || nsg!=8u || block.x!=0u || block.y!=0u || block.z!=0u
        || arrayLength(&logits)<p.count*512u || arrayLength(&remap)!=32768u
        || arrayLength(&map)!=PROMPT_RAW_WORDS || arrayLength(&probability)!=PROMPT_PROB_WORDS
        || arrayLength(&usage)!=32768u || arrayLength(&fault)<1u {return;}
    if local!=0u {return;}
    // The actual same-owner queue dependency includes every preceding group's
    // map/probability writes and fault atomics. A reset header stays invalid on
    // any error, including an earlier attention or projection fault.
    if atomicLoad(&fault[0])!=0u {return;}
    if (p.r0&1u)!=0u {
        // Keep the original route-order accumulation for optional usage. It
        // must not race between groups, nor change the adaptation's math.
        for(var route=0u;route<p.count*10u;route++) {
            let expert=map[PROMPT_LOGICAL+route];
            if expert>=512u {atomicOr(&fault[0],128u);return;}
            usage[p.layer*512u+expert]+=1.0;
        }
    }
    map[0]=p.count;map[1]=0u;map[2]=p.count;map[3]=0u;
}
