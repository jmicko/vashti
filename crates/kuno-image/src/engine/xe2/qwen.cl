// Qwen Image 2.1 DiT glue as OpenCL kernels for Level Zero, so a whole step runs on one queue
// with the machine-code products and attention (no hand-offs to Vulkan between them). Each
// kernel is the same computation as the Vulkan shader named after it (gpu/shaders/q*.comp);
// see there for the layouts. Compiled to SPIR-V offline (`qwen.spv`, see qwen_glue.rs).
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL FP_CONTRACT OFF

#define DIM4 1024u
#define SG __attribute__((intel_reqd_sub_group_size(16)))
#define WG(n) __attribute__((reqd_work_group_size(n, 1, 1)))

// ---- The ConvRot quantizer over a row in local memory (rotate8.comp) ----------------------

inline __attribute__((always_inline)) void quad(float* v, uint a, uint b, uint c, uint d) {
    float va = v[a], vb = v[b], vc = v[c], vd = v[d];
    v[a] = va + vb + vc - vd;
    v[b] = va + vb - vc + vd;
    v[c] = va - vb + vc + vd;
    v[d] = -va + vb + vc + vd;
}
inline __attribute__((always_inline)) void stages(float* v) {
    for (uint m = 0; m < 4; m++) quad(v, 4 * m, 4 * m + 1, 4 * m + 2, 4 * m + 3);
    for (uint m = 0; m < 4; m++) quad(v, m, m + 4, m + 8, m + 12);
}
// Rotates `groups` groups of 256 values of `mem` in place (natural order); returns the row's
// largest magnitude. Values come from `mem` already (scaled by 1/16 here).
inline __attribute__((always_inline)) float rotate_row(local float* mem, local float* maxima, uint groups, uint rounds) {
    uint sg = get_sub_group_id(), lane = get_sub_group_local_id();
    float top = 0.0f;
    float v[16];
    for (uint r = 0; r < rounds; r++) {
        uint g = r * 16 + sg;
        bool used = g < groups;
        if (used) {
            for (uint j = 0; j < 16; j++) v[j] = mem[g * 256 + lane * 16 + j] * 0.0625f;
            stages(v);
            for (uint j = 0; j < 16; j++) mem[g * 256 + lane * 16 + j] = v[j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (used) {
            for (uint j = 0; j < 16; j++) v[j] = mem[g * 256 + j * 16 + lane];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (used) {
            stages(v);
            for (uint j = 0; j < 16; j++) {
                top = fmax(top, fabs(v[j]));
                mem[g * 256 + j * 16 + lane] = v[j];
            }
        }
    }
    top = sub_group_reduce_max(top);
    if (lane == 0) maxima[sg] = top;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint i = 0; i < 16; i++) top = fmax(top, maxima[i]);
    return top;
}
// Rounds the rotated row to INT8 with one scale (written to s[row]) in the machine-code
// product's stream layout (xe2/int8.rs).
inline __attribute__((always_inline)) void round_row(local float* mem, global uint4* q, global float* s, uint row, float top,
                      uint groups, uint rounds) {
    uint sg = get_sub_group_id(), lane = get_sub_group_local_id();
    float scale = (float)((double)top / 127.0);
    float inverse = (float)(1.0 / (double)scale);
    if (get_local_id(0) == 0) s[row] = scale;
    for (uint r = 0; r < rounds; r++) {
        uint g = r * 16 + sg;
        if (g >= groups) continue;
        uint4 word = (uint4)(0);
        for (uint b = 0; b < 16; b++) {
            float scaled = mem[g * 256 + lane * 16 + b] * inverse;
            int value = scale == 0.0f ? 0 : (int)clamp(rint(scaled), -127.0f, 127.0f);
            uint shifted = ((uint)value & 255u) << ((b & 3) * 8);
            if ((b >> 2) == 0) word.x |= shifted;
            else if ((b >> 2) == 1) word.y |= shifted;
            else if ((b >> 2) == 2) word.z |= shifted;
            else word.w |= shifted;
        }
        uint w16 = g * 16 + lane;
        uint at = (((row >> 5) * (groups * 8) + (w16 >> 1)) * 32 + (row & 31)) * 2 + (w16 & 1);
        q[at] = word;
    }
}
inline __attribute__((always_inline)) float reduce256(float value, local float* part) {
    float sum = sub_group_reduce_add(value);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_sub_group_local_id() == 0) part[get_sub_group_id()] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    sum = 0.0f;
    for (uint i = 0; i < 16; i++) sum += part[i];
    return sum;
}

// ---- LayerNorm, modulation and quantization of each row (qnormq8.comp) and, with `add`,
// the gated residual first (qresnorm8.comp) ---------------------------------------------------

inline __attribute__((always_inline)) void residual_norm(global float4* x, global const float4* m, global uint4* q,
                          global float* s, global const int4* g, global const float4* ws,
                          uint tokens, uint mod_first, uint scale_first, uint gate_first,
                          bool add, local float* mem, local float* part) {
    uint token = get_group_id(0);
    uint slot = get_local_id(0);
    bool live = token < tokens;
    float row_scale = live && add ? s[token] : 0.0f;
    float4 total = (float4)(0.0f);
    for (uint c = slot; c < DIM4; c += 256) {
        uint at = token * DIM4 + c;
        float4 d = x[at];
        if (live && add) {
            d += convert_float4(g[at]) * row_scale * ws[(scale_first >> 2) + c] * m[(gate_first >> 2) + c];
            x[at] = d;
        }
        total += d;
        mem[c * 4] = d.x;
        mem[c * 4 + 1] = d.y;
        mem[c * 4 + 2] = d.z;
        mem[c * 4 + 3] = d.w;
    }
    float mean = reduce256(total.x + total.y + total.z + total.w, part) / 4096.0f;
    float4 squares = (float4)(0.0f);
    for (uint c = slot; c < DIM4; c += 256) {
        float4 d = (float4)(mem[c * 4], mem[c * 4 + 1], mem[c * 4 + 2], mem[c * 4 + 3]) - mean;
        squares += d * d;
    }
    float norm = 1.0f / sqrt(reduce256(squares.x + squares.y + squares.z + squares.w, part) / 4096.0f + 1e-6f);
    barrier(CLK_LOCAL_MEM_FENCE);
    uint mod_at = mod_first >> 2;
    for (uint c = slot; c < DIM4; c += 256) {
        float4 t = (float4)(mem[c * 4], mem[c * 4 + 1], mem[c * 4 + 2], mem[c * 4 + 3]);
        t = live ? (t - mean) * norm * m[mod_at + c] : (float4)(0.0f);
        mem[c * 4] = t.x;
        mem[c * 4 + 1] = t.y;
        mem[c * 4 + 2] = t.z;
        mem[c * 4 + 3] = t.w;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float top = rotate_row(mem, part, 16, 1);
    round_row(mem, q, s, token, top, 16, 1);
}
SG WG(256) kernel void q_norm(global float4* x, global const float4* m, global uint4* q,
                             global float* s, uint tokens, uint mod_first) {
    local float mem[4096];
    local float part[16];
    residual_norm(x, m, q, s, 0, 0, tokens, mod_first, 0, 0, false, mem, part);
}
SG WG(256) kernel void q_resnorm(global float4* x, global const float4* m, global uint4* q,
                                global float* s, global const int4* g, global const float4* ws,
                                uint tokens, uint mod_first, uint scale_first, uint gate_first) {
    local float mem[4096];
    local float part[16];
    residual_norm(x, m, q, s, g, ws, tokens, mod_first, scale_first, gate_first, true, mem, part);
}

// ---- The gated residual alone (qresid8.comp) --------------------------------------------------
WG(64) kernel void q_resid(global float4* x, global const int4* g, global const float* as,
                           global const float4* v, global const float4* m, uint tokens,
                           uint scale_first, uint gate_first) {
    uint c = get_global_id(0);
    uint token = get_global_id(1);
    if (token >= tokens) return;
    uint at = token * DIM4 + c;
    float4 update = convert_float4(g[at]) * as[token] * v[(scale_first >> 2) + c];
    x[at] += update * m[(gate_first >> 2) + c];
}

// Four bf16 values (the products' scaled outputs) to f32.
inline __attribute__((always_inline)) float4 widen(ushort4 b) {
    return (float4)(as_float((uint)b.x << 16), as_float((uint)b.y << 16),
                    as_float((uint)b.z << 16), as_float((uint)b.w << 16));
}

// ---- Queries or keys into the attention streams (qqk.comp, without the prefix store) -------
// A group is one head's 16-token tile: each sub-group normalizes and turns one token's head
// (as before, lanes are channel pairs), the results meet in local memory in the stream's
// layout, and the group stores the tile's 4 KB block contiguously. Keys past `tokens` are not
// written (the prefix's keys go there).
SG WG(256) kernel void q_qk(global const ushort4* g, global const float4* v,
                           global const float4* turn, global uint4* o, uint tokens, uint stride,
                           uint side, uint weight_first) {
    local uint tile[1024];
    uint head = get_group_id(0);
    uint first = get_group_id(1) * 16;
    if (side == 1 && first >= tokens) return;
    uint sg = get_sub_group_id();
    uint lane = get_sub_group_local_id();
    uint token = first + sg;
    bool live = token < tokens;
    uint from = token * (3 * DIM4) + side * DIM4 + head * 32 + lane * 2;
    float4 mine[2];
    float squares = 0.0f;
    for (uint j = 0; j < 2; j++) {
        mine[j] = live ? widen(g[from + j]) : (float4)(0.0f);
        squares += dot(mine[j], mine[j]);
    }
    float norm = 1.0f / sqrt(sub_group_reduce_add(squares) / 128.0f + 1e-6f);
    uint to = side == 0 ? (lane >> 1) * 128 + (lane & 1) * 64 + sg
                        : (lane >> 1) * 128 + sg * 8 + (lane & 1) * 4;
    uint pair = side == 0 ? 16 : 1;
    for (uint j = 0; j < 2; j++) {
        float4 a = mine[j] * (norm * v[(weight_first >> 2) + lane * 2 + j]);
        float4 t = live ? turn[token * 32 + lane * 2 + j] : (float4)(1.0f, 0.0f, 1.0f, 0.0f);
        float4 r = (float4)(t.x * a.x - t.y * a.y, t.y * a.x + t.x * a.y,
                            t.z * a.z - t.w * a.w, t.w * a.z + t.z * a.w);
        tile[to + (j * 2) * pair] = as_uint((half2)((half)r.x, (half)r.y));
        tile[to + (j * 2 + 1) * pair] = as_uint((half2)((half)r.z, (half)r.w));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    uint i = get_local_id(0);
    uint4 word = (uint4)(tile[i * 4], tile[i * 4 + 1], tile[i * 4 + 2], tile[i * 4 + 3]);
    // Keys: each 4-word piece belongs to one token, (i * 4 % 128) / 8.
    if (side == 1 && first + ((i * 4) & 127) / 8 >= tokens) return;
    uint chunk = head * (stride >> 4) + (first >> 4);
    o[chunk * (side == 0 ? 256 : 512) + i] = word;
}

// ---- Queries or keys into the attention streams (qqk.comp, without the prefix store) -------
SG WG(64) kernel void q_qk_ref(global const ushort4* g, global const float4* v,
                          global const float4* turn, global uint* o, uint tokens, uint stride,
                          uint side, uint weight_first) {
    uint head = get_group_id(0) * 4 + get_sub_group_id();
    uint token = get_group_id(1);
    uint lane = get_sub_group_local_id();
    bool live = token < tokens;
    if (!live && side == 1) return;
    uint from = token * (3 * DIM4) + side * DIM4 + head * 32 + lane * 2;
    float4 mine[2];
    float squares = 0.0f;
    for (uint j = 0; j < 2; j++) {
        mine[j] = live ? widen(g[from + j]) : (float4)(0.0f);
        squares += dot(mine[j], mine[j]);
    }
    float norm = 1.0f / sqrt(sub_group_reduce_add(squares) / 128.0f + 1e-6f);
    uint tiles = stride >> 4;
    uint to = side == 0
        ? (head * tiles + (token >> 4)) * 1024 + (lane >> 1) * 128 + (lane & 1) * 64 + (token & 15)
        : (head * tiles + (token >> 4)) * 2048 + (lane >> 1) * 128 + (token & 15) * 8 + (lane & 1) * 4;
    uint pair = side == 0 ? 16 : 1;
    for (uint j = 0; j < 2; j++) {
        float4 a = mine[j] * (norm * v[(weight_first >> 2) + lane * 2 + j]);
        float4 t = live ? turn[token * 32 + lane * 2 + j] : (float4)(1.0f, 0.0f, 1.0f, 0.0f);
        float4 r = (float4)(t.x * a.x - t.y * a.y, t.y * a.x + t.x * a.y,
                            t.z * a.z - t.w * a.w, t.w * a.z + t.z * a.w);
        o[to + (j * 2) * pair] = as_uint((half2)((half)r.x, (half)r.y));
        o[to + (j * 2 + 1) * pair] = as_uint((half2)((half)r.z, (half)r.w));
    }
}

// ---- Values into the key/value stream (qvalue.comp, without the prefix store) --------------
WG(256) kernel void q_value(global const ushort4* g, global half4* vt, uint tokens, uint stride) {
    local float tile[4096];
    uint head = get_group_id(0);
    uint first = get_group_id(1) * 32;
    uint lane = get_local_id(0);
    for (uint j = 0; j < 4; j++) {
        uint index = lane + j * 256;
        uint t = index >> 5, c = index & 31;
        uint token = first + t;
        uint at = token * (3 * DIM4) + 2 * DIM4 + head * 32 + c;
        float4 value = token < tokens ? widen(g[at]) : (float4)(0.0f);
        tile[(c * 4) * 32 + t] = value.x;
        tile[(c * 4 + 1) * 32 + t] = value.y;
        tile[(c * 4 + 2) * 32 + t] = value.z;
        tile[(c * 4 + 3) * 32 + t] = value.w;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint j = 0; j < 4; j++) {
        uint index = lane + j * 256;
        uint ch = index >> 3, quad4 = index & 7;
        uint at = ch * 32 + quad4 * 4;
        uint key = first + quad4 * 4;
        vt[(head * (stride >> 4) + (key >> 4)) * 1024 + 512 + ch * 4 + ((key & 15) >> 2)] =
            (half4)((half)tile[at], (half)tile[at + 1], (half)tile[at + 2], (half)tile[at + 3]);
    }
}

// ---- The kept prefix keys and values into the stream (qprefix.comp) --------------------------
WG(64) kernel void q_prefix(global const uint4* k, global const half4* vs, global uint4* ks,
                            global half* vh, uint count, uint base, uint stride, uint capacity) {
    uint index = get_global_id(0);
    uint head = get_global_id(1);
    uint row = index >> 4, part = index & 15;
    if (row >= count) return;
    uint from = (head * capacity + row) * 16 + part;
    uint key = base + row;
    uint chunk = head * (stride >> 4) + (key >> 4);
    ks[chunk * 512 + (part >> 1) * 32 + (key & 15) * 2 + (part & 1)] = k[from];
    half4 a = vs[from * 2], b = vs[from * 2 + 1];
    uint at = chunk * 4096 + 2048 + part * 8 * 16 + (key & 15);
    vh[at] = a.x;
    vh[at + 16] = a.y;
    vh[at + 32] = a.z;
    vh[at + 48] = a.w;
    vh[at + 64] = b.x;
    vh[at + 80] = b.y;
    vh[at + 96] = b.z;
    vh[at + 112] = b.w;
}

// ---- Attention output gathered and quantized (qmergeq8.comp) ---------------------------------
SG WG(256) kernel void q_merge(global const float* o, global const float* w, global uint4* q,
                              global float* s, uint tokens, uint padded) {
    local float mem[4096];
    local float part[16];
    uint token = get_group_id(0);
    uint slot = get_local_id(0);
    for (uint i = slot; i < DIM4; i += 256) {
        uint tile = (i >> 5) * (padded >> 4) + (token >> 4);
        uint lane = token & 15;
        float scale = token < tokens ? 1.0f / w[tile * 16 + lane] : 0.0f;
        uint at = (tile * 128 + (i & 31) * 4) * 16 + lane;
        mem[i * 4] = o[at] * scale;
        mem[i * 4 + 1] = o[at + 16] * scale;
        mem[i * 4 + 2] = o[at + 32] * scale;
        mem[i * 4 + 3] = o[at + 48] * scale;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float top = rotate_row(mem, part, 16, 1);
    round_row(mem, q, s, token, top, 16, 1);
}

// The 12288-wide rotated row: 48 groups of 256, three per sub-group.
#define HIDDEN4 3072u
#define GROUPS 48u
// Element `a * 16 + b` of a group in a sub-group's transpose tile (rows padded to 17 floats so
// the 16 lanes of a column access hit different banks).
#define T(a, b) ((a) * 17u + (b))
// ---- The feed-forward activation (from the gated product, `[token][12288]` bf16) quantized ---
// Each lane keeps its 3 x 16 rotated values in registers; local memory only carries the
// transposes (one small tile per sub-group) and the sub-groups' maxima.
SG WG(256) kernel void q_hidden(global const ushort16* h, global float* s, global uint4* q,
                               uint tokens) {
    local float tiles[16 * 16 * 17];
    local float maxima[16];
    uint token = get_group_id(0);
    uint sg = get_sub_group_id(), lane = get_sub_group_local_id();
    local float* tile = tiles + sg * (16 * 17);
    bool live = token < tokens;
    float top = 0.0f;
    float w[3][16];
    for (uint r = 0; r < 3; r++) {
        uint group = r * 16 + sg;
        ushort16 raw = live ? h[token * (HIDDEN4 / 4) + group * 16 + lane] : (ushort16)(0);
        float v[16];
        ushort b[16];
        vstore16(raw, 0, b);
        for (uint j = 0; j < 16; j++) v[j] = as_float((uint)b[j] << 16) * 0.0625f;
        stages(v);
        for (uint j = 0; j < 16; j++) tile[T(lane, j)] = v[j];
        sub_group_barrier(CLK_LOCAL_MEM_FENCE);
        for (uint j = 0; j < 16; j++) v[j] = tile[T(j, lane)];
        sub_group_barrier(CLK_LOCAL_MEM_FENCE);
        stages(v);
        for (uint j = 0; j < 16; j++) {
            top = fmax(top, fabs(v[j]));
            w[r][j] = v[j];
        }
    }
    top = sub_group_reduce_max(top);
    if (lane == 0) maxima[sg] = top;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint i = 0; i < 16; i++) top = fmax(top, maxima[i]);
    float scale_out = (float)((double)top / 127.0);
    float inverse = (float)(1.0 / (double)scale_out);
    if (get_local_id(0) == 0) s[token] = scale_out;
    for (uint r = 0; r < 3; r++) {
        uint group = r * 16 + sg;
        for (uint j = 0; j < 16; j++) tile[T(j, lane)] = w[r][j];
        sub_group_barrier(CLK_LOCAL_MEM_FENCE);
        uint4 word = (uint4)(0);
        for (uint b = 0; b < 16; b++) {
            float scaled = tile[T(lane, b)] * inverse;
            int rounded = scale_out == 0.0f ? 0 : (int)clamp(rint(scaled), -127.0f, 127.0f);
            uint shifted = ((uint)rounded & 255u) << ((b & 3) * 8);
            if ((b >> 2) == 0) word.x |= shifted;
            else if ((b >> 2) == 1) word.y |= shifted;
            else if ((b >> 2) == 2) word.z |= shifted;
            else word.w |= shifted;
        }
        sub_group_barrier(CLK_LOCAL_MEM_FENCE);
        uint w16 = group * 16 + lane;
        uint at = (((token >> 5) * (GROUPS * 8) + (w16 >> 1)) * 32 + (token & 31)) * 2 + (w16 & 1);
        q[at] = word;
    }
}

#define REF_GROUPS 48u
// The 12288-wide rotated row: 28 groups in local memory, 20 through `spill`.
#define REF_KEPT 28u
#define REF_SPILLED (REF_GROUPS - REF_KEPT)
// ---- The feed-forward activation (from the gated product, `[token][12288]` bf16) quantized ---
SG WG(256) kernel void q_hidden_ref(global const ushort4* h, global float* s, global uint4* q,
                               uint tokens) {
    // The whole rotated row stays in local memory (48 KB). Each sub-group transposes its
    // group in the group's own slots before they hold the result, so no separate tile.
    local float mem[REF_GROUPS * 256];
    local float maxima[16];
    uint token = get_group_id(0);
    uint sg = get_sub_group_id(), lane = get_sub_group_local_id();
    bool live = token < tokens;
    float top = 0.0f;
    float v[16];
    for (uint r = 0; r < 3; r++) {
        uint group = r * 16 + sg;
        for (uint j = 0; j < 4; j++) {
            uint c = group * 64 + lane * 4 + j;
            float4 part = live ? widen(h[token * HIDDEN4 + c]) : (float4)(0.0f);
            v[j * 4] = part.x * 0.0625f;
            v[j * 4 + 1] = part.y * 0.0625f;
            v[j * 4 + 2] = part.z * 0.0625f;
            v[j * 4 + 3] = part.w * 0.0625f;
        }
        stages(v);
        local float* tile = mem + group * 256;
        for (uint j = 0; j < 16; j++) tile[lane * 16 + j] = v[j];
        sub_group_barrier(CLK_LOCAL_MEM_FENCE);
        for (uint j = 0; j < 16; j++) v[j] = tile[j * 16 + lane];
        sub_group_barrier(CLK_LOCAL_MEM_FENCE);
        stages(v);
        for (uint j = 0; j < 16; j++) {
            top = fmax(top, fabs(v[j]));
            tile[j * 16 + lane] = v[j];
        }
    }
    top = sub_group_reduce_max(top);
    if (lane == 0) maxima[sg] = top;
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    for (uint i = 0; i < 16; i++) top = fmax(top, maxima[i]);
    float scale_out = (float)((double)top / 127.0);
    float inverse = (float)(1.0 / (double)scale_out);
    if (get_local_id(0) == 0) s[token] = scale_out;
    for (uint r = 0; r < 3; r++) {
        uint group = r * 16 + sg;
        uint4 word = (uint4)(0);
        for (uint b = 0; b < 16; b++) {
            float value = mem[group * 256 + lane * 16 + b];
            float scaled = value * inverse;
            int rounded = scale_out == 0.0f ? 0 : (int)clamp(rint(scaled), -127.0f, 127.0f);
            uint shifted = ((uint)rounded & 255u) << ((b & 3) * 8);
            if ((b >> 2) == 0) word.x |= shifted;
            else if ((b >> 2) == 1) word.y |= shifted;
            else if ((b >> 2) == 2) word.z |= shifted;
            else word.w |= shifted;
        }
        uint w16 = group * 16 + lane;
        uint at = (((token >> 5) * (REF_GROUPS * 8) + (w16 >> 1)) * 32 + (token & 31)) * 2 + (w16 & 1);
        q[at] = word;
    }
}

// ---- Latent rows into the stream (qembed.comp) -----------------------------------------------
WG(256) kernel void q_embed(global const float* rows, global const float4* weight, global float* x,
                            uint tokens) {
    local float4 row[16];
    uint token = get_group_id(0), lane = get_local_id(0);
    if (lane < 16) {
        uint at = token * 64 + lane * 4;
        row[lane] = (float4)(rows[at], rows[at + 1], rows[at + 2], rows[at + 3]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint k = 0; k < 16; k++) {
        uint o = lane + k * 256;
        float sum = 0.0f;
        for (uint j = 0; j < 16; j++) sum += dot(row[j], weight[o * 16 + j]);
        x[token * 4096 + o] = sum;
    }
}

// ---- Output head (qfinal.comp) ----------------------------------------------------------------
SG WG(256) kernel void q_final(global const float4* x, global const float4* scale,
                              global const float4* weight, global float* out, uint tokens) {
    local float4 mem[1024];
    local float part[16];
    uint token = get_group_id(0), lane = get_local_id(0);
    float4 total = (float4)(0.0f);
    for (uint c = lane; c < DIM4; c += 256) {
        float4 d = x[token * DIM4 + c];
        mem[c] = d;
        total += d;
    }
    float mean = reduce256(total.x + total.y + total.z + total.w, part) / 4096.0f;
    float4 squares = (float4)(0.0f);
    for (uint c = lane; c < DIM4; c += 256) {
        float4 d = mem[c] - mean;
        squares += d * d;
    }
    float norm = 1.0f / sqrt(reduce256(squares.x + squares.y + squares.z + squares.w, part) / 4096.0f + 1e-6f);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint c = lane; c < DIM4; c += 256) mem[c] = (mem[c] - mean) * norm * (1.0f + scale[c]);
    barrier(CLK_LOCAL_MEM_FENCE);
    uint sg = get_sub_group_id(), sl = get_sub_group_local_id();
    for (uint k = 0; k < 4; k++) {
        uint o = sg * 4 + k;
        float sum = 0.0f;
        for (uint c = sl; c < DIM4; c += 16) sum += dot(mem[c], weight[o * DIM4 + c]);
        sum = sub_group_reduce_add(sum);
        if (sl == 0) out[token * 64 + o] = sum;
    }
}
