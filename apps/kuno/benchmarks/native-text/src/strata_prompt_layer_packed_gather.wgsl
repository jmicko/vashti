// Large segments only:16x16 input tiles; original source/finite/whole-owner guards.
// Original gather conversion/finite behavior, exactly one routed ordinal per row.
struct Info {count:u32,groups:u32,half_input:u32,pad:u32,spare:vec4<u32>}
@group(0) @binding(0) var<storage,read> map:array<u32>;
@group(0) @binding(1) var<storage,read> input:array<u32>;
@group(0) @binding(2) var<storage,read_write> packed:array<u32>;
@group(0) @binding(3) var<storage,read> hidden:array<u32>;
@group(0) @binding(4) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(5) var<uniform> p:Info;
fn finite(x:f32)->bool{return abs(x)<=3.4028234663852886e38;}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id:vec3<u32>){
 if p.count<1u||p.count>PROMPT_MAX_ROWS||p.groups<1u||p.groups>512u||p.half_input>1u||p.pad<1u||p.pad>PROMPT_PADDED_ROWS||p.pad%32u!=0u||p.spare.x!=1u||any(p.spare.yzw!=vec3<u32>(0u)) {atomicOr(&fault[0],128u);return;}
 if arrayLength(&map)!=65536u||arrayLength(&packed)!=PROMPT_PADDED_ROWS*1280u||arrayLength(&hidden)!=PROMPT_MAX_ROWS*320u {atomicOr(&fault[0],128u);return;}
 if map[0]!=p.count||map[1]!=0u||map[2]!=p.count||map[3]!=p.groups||map[PROMPT_LAYER_PREFIX+512u]!=p.pad {atomicOr(&fault[0],128u);return;}
 if id.x>=1280u||id.y>=p.pad {return;}
 let row=id.y;let i=row*1280u+id.x;var low=0u;var high=p.groups;
 for(var turn=0u;turn<9u;turn++){if low+1u<high {let mid=(low+high)/2u;if map[PROMPT_LAYER_PREFIX+mid]<=row {low=mid;}else{high=mid;}}}
 let group=low;let first=map[PROMPT_LAYER_PREFIX+group];let members=map[516u+group];
 if first%32u!=0u||first>row||members<1u||members>p.count||first>p.pad||((members+31u)/32u)*32u>p.pad-first {atomicOr(&fault[0],128u);return;}
 let end=first+((members+31u)/32u)*32u;
 if (group==0u&&first!=0u)||(group+1u<p.groups&&map[PROMPT_LAYER_PREFIX+group+1u]!=end)||(group+1u==p.groups&&end!=p.pad)||row>=end {atomicOr(&fault[0],128u);return;}
 let member=row-first;
 var dest=i;
 if members>=256u {dest=first*1280u+(member/16u)*(16u*1280u)+(id.x/8u)*128u+(member%16u)*8u+id.x%8u;}
 if member>=members {packed[dest]=0u;return;}
 let prefix=map[1028u+group];if prefix>p.count*10u||members>p.count*10u-prefix {atomicOr(&fault[0],128u);return;}
 let route=map[1540u+prefix+member];if route>=p.count*10u {atomicOr(&fault[0],128u);return;}
 let at=(route/10u)*2560u+(i%1280u)*2u;
 var bits=0u;var good=false;
 if p.half_input==1u {if at/2u>=arrayLength(&input){atomicOr(&fault[0],128u);return;}bits=input[at/2u];let v=unpack2x16float(bits);good=finite(v.x)&&finite(v.y);}
 else {if at+1u>=arrayLength(&input){atomicOr(&fault[0],128u);return;}let a=bitcast<f32>(input[at]);let b=bitcast<f32>(input[at+1u]);bits=pack2x16float(vec2<f32>(a,b));let v=unpack2x16float(bits);good=finite(a)&&finite(b)&&finite(v.x)&&finite(v.y);}
 packed[dest]=select(0u,bits,good);if !good {atomicOr(&fault[0],128u);}
}
