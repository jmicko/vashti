// Rust Strata port: real GPU argmax and carry only; full owners remain private.
struct Params {count:u32,start:u32,zero0:u32,zero1:u32}
@group(0) @binding(0) var<storage,read> logits:array<f32>;
@group(0) @binding(1) var<storage,read> carry:array<u32>;
@group(0) @binding(2) var<storage,read> fault:array<u32>;
@group(0) @binding(3) var<storage,read_write> output:array<u32>;
@group(0) @binding(4) var<uniform> p:Params;
var<workgroup> best:array<f32,128>;
var<workgroup> ids:array<u32,128>;
var<workgroup> bad:array<u32,128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32) {
    let row=group.x;
    if row>=4u || p.count<1u || p.count>4u || p.start+p.count>4096u ||
       arrayLength(&logits)<p.count*248320u || arrayLength(&carry)<4u*10240u ||
       arrayLength(&fault)<1u || arrayLength(&output)!=1048576u || p.zero0!=0u || p.zero1!=0u {return;}
    var value=bitcast<f32>(0xff7fffffu);var id=248320u;var invalid=0u;
    if row<p.count {
        for(var col=lane;col<248320u;col+=128u) {
            let x=logits[row*248320u+col];
            if abs(x)>bitcast<f32>(0x7f7fffffu) || x!=x {invalid=1u;}
            if x>value || (x==value && col<id) {value=x;id=col;}
        }
    }
    best[lane]=value;ids[lane]=id;bad[lane]=invalid;workgroupBarrier();
    for(var stride=64u;stride>0u;stride/=2u) {
        if lane<stride {
            let x=best[lane+stride];let next=ids[lane+stride];
            if x>best[lane] || (x==best[lane] && next<ids[lane]) {best[lane]=x;ids[lane]=next;}
            bad[lane]|=bad[lane+stride];
        }
        workgroupBarrier();
    }
    if lane==0u {
        var selected=0u;
        if row<p.count {selected=ids[0];if bad[0]!=0u || fault[0]!=0u {selected=248320u;}}
        output[8u+row]=selected;
    }
    if row==0u && lane<8u {
        var stamp=0u;
        switch lane {case 0u:{stamp=2u;}case 1u:{stamp=p.count;}case 2u:{stamp=p.start;}
            case 3u:{stamp=248320u;}case 4u:{stamp=10240u;}case 5u:{stamp=4u;}case 6u:{stamp=fault[0];}default:{}}
        output[lane]=stamp;
    }
    for(var col=lane;col<10240u;col+=128u) {
        var bits=0u;if row<p.count {bits=carry[row*10240u+col];}
        output[12u+row*10240u+col]=bits;
    }
}
