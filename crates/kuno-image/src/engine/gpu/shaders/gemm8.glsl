// The shared body of gemm8.comp (Intel: 8x32 x 32x16 int8 matrices, 16-lane subgroups) and
// gemm8nv.comp (NVIDIA and others: 16x32 x 32x16, 32-lane subgroups). The including file
// defines SG (subgroup lanes), TM (tokens per matrix), MI (matrices down a subgroup's tokens),
// NJ (matrices across its rows) and SPLIT (subgroups across the 256 rows).
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_KHR_shader_subgroup_basic : require

layout(local_size_x = 512) in;
layout(set = 0, binding = 0) readonly buffer In { uvec4 x[]; };
layout(set = 0, binding = 1) readonly buffer Weight { uvec4 w[]; };
layout(set = 0, binding = 2) writeonly buffer Out { int y[]; };
// Operand offsets are in bytes (multiples of 16); output offsets in values.
layout(push_constant) uniform Constants {
    uint columns;
    uint in_first;
    uint weight_first;
    uint out_first;
    uint out_stride;
} p;

// Per buffer: 128 tokens x 2 words, 256 rows x 2 words.
shared uvec4 a[512];
shared uvec4 b[1024];

#define A coopmat<int8_t, gl_ScopeSubgroup, TM, 32, gl_MatrixUseA>
#define B coopmat<int8_t, gl_ScopeSubgroup, 32, 16, gl_MatrixUseB>
#define C coopmat<int32_t, gl_ScopeSubgroup, TM, 16, gl_MatrixUseAccumulator>
// A subgroup's tile: TM * MI tokens by 16 * NJ rows.
#define TOKENS_SG (TM * MI)
#define ROWS_SG (16 * NJ)

void main() {
    uint sg = gl_SubgroupID;
    uint lane = sg * SG + gl_SubgroupInvocationID;
    uint am = (sg / SPLIT) * TOKENS_SG;
    uint bn = (sg % SPLIT) * ROWS_SG;
    uint token = gl_WorkGroupID.y * 128;
    uint row = gl_WorkGroupID.x * 256;
    uint cols16 = p.columns >> 4;
    // Every lane copies one word of a weight row per step; the first 256 also one word of a
    // token row.
    bool copies = lane < 256;
    uint in16 = (p.in_first >> 4) + (token + (lane >> 1)) * cols16 + (lane & 1);
    uint weight16 = (p.weight_first >> 4) + (row + (lane >> 1)) * cols16 + (lane & 1);
    uvec4 ta, tb;
    A am_[MI];
    B bn_[NJ];
    C c[MI][NJ];
    for (uint i = 0; i < MI; i++)
        for (uint j = 0; j < NJ; j++) c[i][j] = C(0);
    if (copies) {
        a[lane] = x[in16];
    }
    b[lane] = w[weight16];
    memoryBarrierShared();
    barrier();
    uint cur = 0;
    for (uint k = 0; k < p.columns; k += 32) {
        uint next = 1 - cur;
        bool more = k + 32 < p.columns;
        uint k16 = (k + 32) >> 4;
        if (more) {
            if (copies) {
                ta = x[in16 + k16];
            }
            tb = w[weight16 + k16];
        }
        uint aa = cur * 256 + am * 2;
        uint bb = cur * 512 + bn * 2;
        for (uint i = 0; i < MI; i++)
            coopMatLoad(am_[i], a, aa + i * TM * 2, 2, gl_CooperativeMatrixLayoutRowMajor);
        for (uint j = 0; j < NJ; j++)
            coopMatLoad(bn_[j], b, bb + j * 32, 2, gl_CooperativeMatrixLayoutColumnMajor);
        for (uint j = 0; j < NJ; j++)
            for (uint i = 0; i < MI; i++) c[i][j] = coopMatMulAdd(am_[i], bn_[j], c[i][j]);
        if (more) {
            if (copies) {
                a[next * 256 + lane] = ta;
            }
            b[next * 512 + lane] = tb;
        }
        memoryBarrierShared();
        barrier();
        cur = next;
    }
    uint first = p.out_first + (token + am) * p.out_stride + row + bn;
    uint s = p.out_stride;
    for (uint i = 0; i < MI; i++)
        for (uint j = 0; j < NJ; j++)
            coopMatStore(c[i][j], y, first + i * TM * s + j * 16, s, gl_CooperativeMatrixLayoutRowMajor);
}
