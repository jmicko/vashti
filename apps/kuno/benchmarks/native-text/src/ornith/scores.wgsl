@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group:vec3<u32>, @builtin(local_invocation_index) lane:u32) {
    let head=group.x;let token=group.y*64u+lane;
    if token<=ctrl.pos {
        let kv=head/8u;var dot=0.0;
        for(var i=0u;i<256u;i++){
            dot+=arena[p.a+head*256u+i]*state[token*512u+kv*256u+i];
        }
        arena[p.b+head*4096u+token]=dot/16.0;
    }
}
