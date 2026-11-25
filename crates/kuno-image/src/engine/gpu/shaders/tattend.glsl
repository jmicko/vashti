// The body of tattend.comp (16-lane subgroups) and tattend32.comp (wider ones: clusters of 16
// lanes); REDUCE sums over a query's 16 lanes.
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_clustered : require

layout(local_size_x = 64) in;
layout(set = 0, binding = 0) readonly buffer Queries { vec4 q[]; };
layout(set = 0, binding = 1) readonly buffer Keys { vec4 k[]; };
layout(set = 0, binding = 2) readonly buffer Values { vec4 v[]; };
layout(set = 0, binding = 3) writeonly buffer Out { vec4 o[]; };
layout(push_constant) uniform Constants { uint tokens; uint padded; uint heads; uint group; } p;

void main() {
    uint token = gl_WorkGroupID.x * 4 + (gl_LocalInvocationIndex >> 4);
    uint head = gl_WorkGroupID.y;
    uint lane = gl_LocalInvocationIndex & 15u;
    uint group = head / p.group;
    uint out_at = token * (p.heads * 32) + head * 32 + lane * 2;
    if (token >= p.tokens) {
        if (token < p.padded) {
            o[out_at] = vec4(0.0);
            o[out_at + 1] = vec4(0.0);
        }
        return;
    }
    uint at = (head * p.padded + token) * 32 + lane * 2;
    const float scale = 0.08838834764831845;
    vec4 q0 = q[at] * scale, q1 = q[at + 1] * scale;
    vec4 a0 = vec4(0.0), a1 = vec4(0.0);
    float top = -1.0 / 0.0, total = 0.0;
    uint keys = group * p.padded * 32 + lane * 2;
    for (uint s = 0; s <= token; s++) {
        uint kat = keys + s * 32;
        float score = REDUCE(dot(q0, k[kat]) + dot(q1, k[kat + 1]));
        float next = max(top, score);
        float shrink = exp(top - next);
        float weight = exp(score - next);
        total = total * shrink + weight;
        a0 = a0 * shrink + weight * v[kat];
        a1 = a1 * shrink + weight * v[kat + 1];
        top = next;
    }
    o[out_at] = a0 / total;
    o[out_at + 1] = a1 / total;
}
