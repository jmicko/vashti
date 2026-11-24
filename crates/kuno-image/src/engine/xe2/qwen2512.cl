// Qwen Image 2512 DiT glue as OpenCL kernels for Level Zero (`qwen2512_gpu.rs`), compiled to
// `qwen2512.bin` by `qwen-glue.sh` like `qwen.cl`. Two streams (image, text) of 3072-wide rows
// with their own weights; one joint attention through `attend.rs`. Products are the
// machine-code INT8 kernels with ConvRot inputs (rows rotated in groups of 256, one INT8 scale
// per row). Every linear has a bias; modulation is `x * (1 + scale) + shift`, gates plain.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL FP_CONTRACT OFF

#define DIM 3072u
#define DIM4 768u
#define GROUPS 12u
#define HIDDEN 12288u
#define HIDDEN_GROUPS 48u
#define SG __attribute__((intel_reqd_sub_group_size(16)))
#define WG(n) __attribute__((reqd_work_group_size(n, 1, 1)))

// Offsets of one stream's modulation vectors (`[6][DIM]`).
#define SHIFT1 0u
#define SCALE1 DIM
#define GATE1 (2u * DIM)
#define SHIFT2 (3u * DIM)
#define SCALE2 (4u * DIM)
#define GATE2 (5u * DIM)

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
// Rotates `groups` (<= 16) groups of 256 values of `mem` in place; returns the row's largest
// magnitude.
inline __attribute__((always_inline)) float rotate_row(local float* mem, local float* maxima, uint groups) {
    uint sg = get_sub_group_id(), lane = get_sub_group_local_id();
    float top = 0.0f;
    float v[16];
    bool used = sg < groups;
    if (used) {
        for (uint j = 0; j < 16; j++) v[j] = mem[sg * 256 + lane * 16 + j] * 0.0625f;
        stages(v);
        for (uint j = 0; j < 16; j++) mem[sg * 256 + lane * 16 + j] = v[j];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (used) {
        for (uint j = 0; j < 16; j++) v[j] = mem[sg * 256 + j * 16 + lane];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (used) {
        stages(v);
        for (uint j = 0; j < 16; j++) {
            top = fmax(top, fabs(v[j]));
            mem[sg * 256 + j * 16 + lane] = v[j];
        }
    }
    top = sub_group_reduce_max(top);
    if (lane == 0) maxima[sg] = top;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint i = 0; i < 16; i++) top = fmax(top, maxima[i]);
    return top;
}
// The rotated row to INT8 with one scale (s[row]) in the product's token stream layout.
inline __attribute__((always_inline)) void round_row(local float* mem, global uint4* q, global float* s, uint row,
                                                     float top, uint groups) {
    uint sg = get_sub_group_id(), lane = get_sub_group_local_id();
    float scale = (float)((double)top / 127.0);
    float inverse = (float)(1.0 / (double)scale);
    if (get_local_id(0) == 0) s[row] = scale;
    if (sg >= groups) return;
    uint4 word = (uint4)(0);
    for (uint b = 0; b < 16; b++) {
        float scaled = mem[sg * 256 + lane * 16 + b] * inverse;
        int value = scale == 0.0f ? 0 : (int)clamp(rint(scaled), -127.0f, 127.0f);
        uint shifted = ((uint)value & 255u) << ((b & 3) * 8);
        if ((b >> 2) == 0) word.x |= shifted;
        else if ((b >> 2) == 1) word.y |= shifted;
        else if ((b >> 2) == 2) word.z |= shifted;
        else word.w |= shifted;
    }
    uint w16 = sg * 16 + lane;
    uint at = (((row >> 5) * (groups * 8) + (w16 >> 1)) * 32 + (row & 31)) * 2 + (w16 & 1);
    q[at] = word;
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
// Four bf16 values (the scaled products' outputs) to f32.
inline __attribute__((always_inline)) float4 widen(ushort4 b) {
    return (float4)(as_float((uint)b.x << 16), as_float((uint)b.y << 16),
                    as_float((uint)b.z << 16), as_float((uint)b.w << 16));
}
inline __attribute__((always_inline)) float gelu(float v) {
    return 0.5f * v * (1.0f + tanh(0.7978846f * (v + 0.044715f * v * v * v)));
}

// ---- Residual (optional), LayerNorm, modulation and quantization of each row --------------
// With `add`: x += (g * s[token] * ws + bias) * gate first (an INT32 product's sums).
inline __attribute__((always_inline)) void residual_norm(global float4* x, global const float4* m, global uint4* q,
        global float* s, global const int4* g, global const float4* vec, uint tokens, uint mod_first,
        uint scale_first, uint bias_first, uint gate_first, bool add, bool norm, local float* mem,
        local float* part) {
    uint token = get_group_id(0);
    uint slot = get_local_id(0);
    bool live = token < tokens;
    float row_scale = live && add ? s[token] : 0.0f;
    float4 total = (float4)(0.0f);
    for (uint c = slot; c < DIM4; c += 256) {
        uint at = token * DIM4 + c;
        float4 d = x[at];
        if (live && add) {
            float4 update = convert_float4(g[at]) * row_scale * vec[(scale_first >> 2) + c] +
                            vec[(bias_first >> 2) + c];
            d += update * m[(gate_first >> 2) + c];
            x[at] = d;
        }
        total += d;
        mem[c * 4] = d.x;
        mem[c * 4 + 1] = d.y;
        mem[c * 4 + 2] = d.z;
        mem[c * 4 + 3] = d.w;
    }
    if (!norm) return;
    float mean = reduce256(total.x + total.y + total.z + total.w, part) / (float)DIM;
    float4 squares = (float4)(0.0f);
    for (uint c = slot; c < DIM4; c += 256) {
        float4 d = (float4)(mem[c * 4], mem[c * 4 + 1], mem[c * 4 + 2], mem[c * 4 + 3]) - mean;
        squares += d * d;
    }
    float inv = 1.0f / sqrt(reduce256(squares.x + squares.y + squares.z + squares.w, part) / (float)DIM + 1e-6f);
    barrier(CLK_LOCAL_MEM_FENCE);
    uint shift_at = mod_first >> 2, scale_at = (mod_first + DIM) >> 2;
    for (uint c = slot; c < DIM4; c += 256) {
        float4 t = (float4)(mem[c * 4], mem[c * 4 + 1], mem[c * 4 + 2], mem[c * 4 + 3]);
        t = live ? (t - mean) * inv * (1.0f + m[scale_at + c]) + m[shift_at + c] : (float4)(0.0f);
        mem[c * 4] = t.x;
        mem[c * 4 + 1] = t.y;
        mem[c * 4 + 2] = t.z;
        mem[c * 4 + 3] = t.w;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float top = rotate_row(mem, part, GROUPS);
    round_row(mem, q, s, token, top, GROUPS);
}
SG WG(256) kernel void k_norm(global float4* x, global const float4* m, global uint4* q,
                             global float* s, uint tokens, uint mod_first) {
    local float mem[DIM];
    local float part[16];
    residual_norm(x, m, q, s, 0, 0, tokens, mod_first, 0, 0, 0, false, true, mem, part);
}
SG WG(256) kernel void k_resnorm(global float4* x, global const float4* m, global uint4* q,
                                global float* s, global const int4* g, global const float4* vec,
                                uint tokens, uint mod_first, uint scale_first, uint bias_first,
                                uint gate_first) {
    local float mem[DIM];
    local float part[16];
    residual_norm(x, m, q, s, g, vec, tokens, mod_first, scale_first, bias_first, gate_first, true,
                  true, mem, part);
}
// The gated residual alone (after the last block).
WG(64) kernel void k_resid(global float4* x, global const int4* g, global const float* as,
                           global const float4* vec, global const float4* m, uint tokens,
                           uint scale_first, uint bias_first, uint gate_first) {
    uint c = get_global_id(0);
    uint token = get_global_id(1);
    if (token >= tokens) return;
    uint at = token * DIM4 + c;
    float4 update = convert_float4(g[at]) * as[token] * vec[(scale_first >> 2) + c] +
                    vec[(bias_first >> 2) + c];
    x[at] += update * m[(gate_first >> 2) + c];
}

// ---- Queries or keys into the attention streams ------------------------------------------
// One sub-group per (token, head): bias, the head's RMS norm, rotary turns, then each token's
// row at `first + token` (a query slot or a key index) of the stream for `attend.rs`. `int8`
// rounds each row to bytes with its own scale (`Shape::int8`), else half floats. Queries live
// in `query_tiles` tiles per head, keys in `key_tiles` chunks.
SG WG(64) kernel void k_qk(global const ushort4* g, global const float4* vec, global const float4* turn,
                           global uint* o, uint tokens, uint first, uint tiles, uint side,
                           uint norm_first, uint bias_first, uint int8) {
    uint head = get_group_id(0) * 4 + get_sub_group_id();
    uint token = get_group_id(1);
    uint lane = get_sub_group_local_id();
    if (token >= tokens) return;
    uint from = token * (3 * DIM4) + side * DIM4 + head * 32 + lane * 2;
    float4 mine[2];
    float squares = 0.0f;
    for (uint j = 0; j < 2; j++) {
        mine[j] = widen(g[from + j]) + vec[(bias_first >> 2) + side * DIM4 + head * 32 + lane * 2 + j];
        squares += dot(mine[j], mine[j]);
    }
    float norm = 1.0f / sqrt(sub_group_reduce_add(squares) / 128.0f + 1e-6f);
    float4 r[2];
    float top = 0.0f;
    for (uint j = 0; j < 2; j++) {
        float4 a = mine[j] * (norm * vec[(norm_first >> 2) + lane * 2 + j]);
        float4 t = turn[token * 32 + lane * 2 + j];
        r[j] = (float4)(t.x * a.x - t.y * a.y, t.y * a.x + t.x * a.y,
                        t.z * a.z - t.w * a.w, t.w * a.z + t.z * a.w);
        float4 m = fabs(r[j]);
        top = fmax(top, fmax(fmax(m.x, m.y), fmax(m.z, m.w)));
    }
    uint at = first + token;
    uint chunk = head * tiles + (at >> 4), row = at & 15;
    if (int8 != 0) {
        float scale = fmax(sub_group_reduce_max(top), 1e-30f) / 127.0f;
        for (uint j = 0; j < 2; j++) {
            uint ch = lane * 8 + j * 4;
            uint word = as_uint(convert_char4_sat_rte(r[j] / scale));
            if (side == 0) o[chunk * 1024 + (ch >> 5) * 128 + ((ch & 31) >> 2) * 16 + row] = word;
            else o[chunk * 2048 + (ch >> 5) * 128 + row * 8 + ((ch & 31) >> 2)] = word;
        }
        if (lane == 0) ((global float*)o)[chunk * (side == 0 ? 1024 : 2048) + 512 + row] = scale;
        return;
    }
    for (uint j = 0; j < 2; j++) {
        for (uint p = 0; p < 2; p++) {
            uint ch = lane * 8 + j * 4 + p * 2;
            float2 pair = p == 0 ? r[j].xy : r[j].zw;
            uint word = as_uint((half2)((half)pair.x, (half)pair.y));
            if (side == 0) o[chunk * 1024 + (ch >> 4) * 128 + ((ch & 15) >> 1) * 16 + row] = word;
            else o[chunk * 2048 + (ch >> 4) * 128 + row * 8 + ((ch & 15) >> 1)] = word;
        }
    }
}
// Values (with bias) into the key/value stream at key index `first + token`.
WG(256) kernel void k_value(global const ushort4* g, global const float4* vec, global half* vt,
                            uint tokens, uint first, uint key_tiles, uint bias_first) {
    uint c = get_global_id(0);  // float4 of the row's 3072 values
    uint token = get_global_id(1);
    if (token >= tokens || c >= DIM4) return;
    float4 v = widen(g[token * (3 * DIM4) + 2 * DIM4 + c]) + vec[(bias_first >> 2) + 2 * DIM4 + c];
    uint head = c >> 5, ch = (c & 31) * 4;
    uint key = first + token;
    uint at = (head * key_tiles + (key >> 4)) * 4096 + 2048 + ch * 16 + (key & 15);
    vt[at] = (half)v.x;
    vt[at + 16] = (half)v.y;
    vt[at + 32] = (half)v.z;
    vt[at + 48] = (half)v.w;
}

// ---- Attention output of query slots `first..` gathered and quantized ---------------------
SG WG(256) kernel void k_merge(global const float* o, global const float* w, global uint4* q,
                              global float* s, uint tokens, uint query_tiles, uint first) {
    local float mem[DIM];
    local float part[16];
    uint token = get_group_id(0);
    uint slot = get_local_id(0);
    uint at_slot = first + token;
    for (uint i = slot; i < DIM4; i += 256) {
        uint tile = (i >> 5) * query_tiles + (at_slot >> 4);
        uint lane = at_slot & 15;
        float scale = token < tokens ? 1.0f / w[tile * 16 + lane] : 0.0f;
        uint at = (tile * 128 + (i & 31) * 4) * 16 + lane;
        mem[i * 4] = o[at] * scale;
        mem[i * 4 + 1] = o[at + 16] * scale;
        mem[i * 4 + 2] = o[at + 32] * scale;
        mem[i * 4 + 3] = o[at + 48] * scale;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float top = rotate_row(mem, part, GROUPS);
    round_row(mem, q, s, token, top, GROUPS);
}

// ---- The MLP's activation: bias, GELU, rotation and INT8 of the `[token][12288]` bf16 rows ---
#define T(a, b) ((a) * 17u + (b))
SG WG(256) kernel void k_hidden(global const ushort16* h, global const float* vec, global float* s,
                               global uint4* q, uint tokens, uint bias_first) {
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
        ushort16 raw = live ? h[token * (HIDDEN / 16) + group * 16 + lane] : (ushort16)(0);
        float v[16];
        ushort b[16];
        vstore16(raw, 0, b);
        uint c0 = group * 256 + lane * 16;
        for (uint j = 0; j < 16; j++) {
            float pre = as_float((uint)b[j] << 16) + vec[bias_first + c0 + j];
            v[j] = live ? gelu(pre) * 0.0625f : 0.0f;
        }
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
        uint at = (((token >> 5) * (HIDDEN_GROUPS * 8) + (w16 >> 1)) * 32 + (token & 31)) * 2 + (w16 & 1);
        q[at] = word;
    }
}

// ---- Latent rows (`[token][64]`) into the image stream: `img_in` with bias ----------------
WG(256) kernel void k_embed(global const float* rows, global const float4* weight,
                            global const float* bias, global float* x, uint tokens) {
    local float4 row[16];
    uint token = get_group_id(0), lane = get_local_id(0);
    if (lane < 16) {
        uint at = token * 64 + lane * 4;
        row[lane] = token < tokens ? (float4)(rows[at], rows[at + 1], rows[at + 2], rows[at + 3])
                                   : (float4)(0.0f);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint k = 0; k < DIM / 256; k++) {
        uint o = lane + k * 256;
        float sum = bias[o];
        for (uint j = 0; j < 16; j++) sum += dot(row[j], weight[o * 16 + j]);
        x[token * DIM + o] = token < tokens ? sum : 0.0f;
    }
}
// Copies `count` float4s (a prompt's text rows into its stream).
WG(256) kernel void k_copy(global const float4* from, global float4* to, uint count) {
    uint i = get_global_id(0);
    if (i < count) to[i] = from[i];
}

// ---- Output head: LayerNorm, `x * (1 + scale) + shift`, `proj_out` with bias -------------
SG WG(256) kernel void k_final(global const float4* x, global const float4* mods,
                              global const float4* weight, global const float* bias,
                              global float* out, uint tokens) {
    local float4 mem[DIM4];
    local float part[16];
    uint token = get_group_id(0), lane = get_local_id(0);
    float4 total = (float4)(0.0f);
    for (uint c = lane; c < DIM4; c += 256) {
        float4 d = x[token * DIM4 + c];
        mem[c] = d;
        total += d;
    }
    float mean = reduce256(total.x + total.y + total.z + total.w, part) / (float)DIM;
    float4 squares = (float4)(0.0f);
    for (uint c = lane; c < DIM4; c += 256) {
        float4 d = mem[c] - mean;
        squares += d * d;
    }
    float inv = 1.0f / sqrt(reduce256(squares.x + squares.y + squares.z + squares.w, part) / (float)DIM + 1e-6f);
    barrier(CLK_LOCAL_MEM_FENCE);
    // AdaLayerNormContinuous: the scale comes first, then the shift.
    for (uint c = lane; c < DIM4; c += 256)
        mem[c] = (mem[c] - mean) * inv * (1.0f + mods[c]) + mods[DIM4 + c];
    barrier(CLK_LOCAL_MEM_FENCE);
    uint sg = get_sub_group_id(), sl = get_sub_group_local_id();
    for (uint k = 0; k < 4; k++) {
        uint o = sg * 4 + k;
        float sum = 0.0f;
        for (uint c = sl; c < DIM4; c += 16) sum += dot(mem[c], weight[o * DIM4 + c]);
        sum = sub_group_reduce_add(sum);
        if (sl == 0 && token < tokens) out[token * 64 + o] = sum + bias[o];
    }
}

// ---- Modulation: `rows` outputs of an FP8 (e4m3) matrix `[rows][DIM]` times `t` plus bias ----
inline __attribute__((always_inline)) float fp8(uint b) {
    uint e = (b >> 3) & 15u, m = b & 7u;
    float v = e == 0 ? (float)m * 0.001953125f : as_float(((e + 120u) << 23) | (m << 20));
    return (b & 128u) ? -v : v;
}
SG WG(256) kernel void k_mod(global const uchar16* w, global const float* bias, global const float* t,
                            global float* out, uint rows, uint first) {
    local float tm[DIM];
    local float table[256];
    uint lane = get_local_id(0);
    for (uint i = lane; i < DIM; i += 256) tm[i] = t[i];
    table[lane] = fp8(lane);
    barrier(CLK_LOCAL_MEM_FENCE);
    uint row = get_group_id(0) * 16 + get_sub_group_id();
    uint sl = get_sub_group_local_id();
    if (row >= rows) return;
    float sum = 0.0f;
    for (uint k = sl; k < DIM / 16; k += 16) {
        uchar16 b = w[row * (DIM / 16) + k];
        uint c = k * 16;
        sum += table[b.s0] * tm[c] + table[b.s1] * tm[c + 1] + table[b.s2] * tm[c + 2] +
               table[b.s3] * tm[c + 3] + table[b.s4] * tm[c + 4] + table[b.s5] * tm[c + 5] +
               table[b.s6] * tm[c + 6] + table[b.s7] * tm[c + 7] + table[b.s8] * tm[c + 8] +
               table[b.s9] * tm[c + 9] + table[b.sa] * tm[c + 10] + table[b.sb] * tm[c + 11] +
               table[b.sc] * tm[c + 12] + table[b.sd] * tm[c + 13] + table[b.se] * tm[c + 14] +
               table[b.sf] * tm[c + 15];
    }
    sum = sub_group_reduce_add(sum);
    if (sl == 0) out[first + row] = sum + bias[row];
}
