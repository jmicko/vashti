// CPU-only tests of actual oneDNN eligibility. No device enumeration or content.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdio>
#include <functional>
#include <limits>
#include "fattn-onednn-shapes.hpp"

struct Shape {
    ggml_context * ctx;
    ggml_tensor * q, * k, * v, * mask, * dst;
    Shape(int rows = 33, int keys = 65, int heads = 2, bool masked = false,
          int dim = 128, bool f32 = false) {
        ctx = ggml_init({1024 * 1024, nullptr, true});
        q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim, rows, heads, 1);
        k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, dim, keys, heads, 1);
        v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, dim, keys, heads, 1);
        mask = masked ? ggml_new_tensor_2d(ctx, GGML_TYPE_F16, keys, rows) : nullptr;
        dst = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f / std::sqrt(float(dim)), 0, 0);
        ggml_flash_attn_ext_set_prec(dst, f32 ? GGML_PREC_F32 : GGML_PREC_DEFAULT);
    }
    ~Shape() { ggml_free(ctx); }
};

int main() {
    int accepted = 0, rejected = 0;
    for (int rows : {32, 33, 65, 67}) for (int keys : {65, 128, 257})
        for (int heads : {2, 32}) for (bool masked : {false, true}) for (bool f32 : {false, true}) {
            Shape s(rows, keys, heads, masked, 128, f32);
            assert(ggml_sycl_onednn_sdpa_shape_supported(s.dst)); ++accepted;
            // Staging handles token/head padding and permutation through strides.
            s.q->nb[1] += sizeof(float) * 8;
            s.k->nb[1] += sizeof(ggml_fp16_t) * 8;
            s.v->nb[1] += sizeof(ggml_fp16_t) * 8;
            assert(ggml_sycl_onednn_sdpa_shape_supported(s.dst)); ++accepted;
        }
    auto reject = [&](std::function<void(Shape &)> mutate, bool masked = false) {
        Shape s(33, 65, 2, masked); mutate(s);
        assert(!ggml_sycl_onednn_sdpa_shape_supported(s.dst)); ++rejected;
    };
    reject([](Shape &s) { s.q->type = GGML_TYPE_F16; });
    reject([](Shape &s) { s.q->nb[0] = 8; });
    reject([](Shape &s) { s.q->ne[0] = 64; });
    reject([](Shape &s) { s.q->ne[1] = 31; });
    reject([](Shape &s) { s.q->ne[2] = 3; });
    reject([](Shape &s) { s.q->ne[3] = 2; });
    reject([](Shape &s) { s.k->ne[1] = 0; });
    reject([](Shape &s) { s.k->ne[2] = 0; });
    reject([](Shape &s) { s.k->ne[3] = 2; });
    reject([](Shape &s) { s.k->nb[0] = 4; });
    reject([](Shape &s) { s.k->type = GGML_TYPE_BF16; });
    reject([](Shape &s) { s.v->ne[0] = 64; });
    reject([](Shape &s) { s.v->ne[1] = 64; });
    reject([](Shape &s) { s.v->ne[2] = 1; });
    reject([](Shape &s) { s.v->ne[3] = 2; });
    reject([](Shape &s) { s.v->nb[0] = 4; });
    reject([](Shape &s) { s.dst->src[4] = s.q; });
    reject([](Shape &s) { s.dst->type = GGML_TYPE_F16; });
    reject([](Shape &s) { s.dst->nb[1] += 4; });
    // Keep invalid outputs contiguous so each independent dimension gate is
    // exercised rather than rejected accidentally by the stride precondition.
    for (int axis = 0; axis < 4; ++axis) for (bool masked : {false, true}) {
        reject([=](Shape &s) {
            ++s.dst->ne[axis];
            s.dst->nb[0] = sizeof(float);
            for (int d = 1; d < 4; ++d) s.dst->nb[d] = s.dst->nb[d - 1] * s.dst->ne[d - 1];
            assert(ggml_is_contiguous(s.dst));
        }, masked);
    }
    reject([](Shape &s) { s.dst->op_params[3] = 17; });
    reject([](Shape &s) { s.q->ne[1] = int64_t(std::numeric_limits<int>::max()) + 1; });
    for (float scale : {0.0f, -1.0f, std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::quiet_NaN(), 1e-10f, 1e10f}) {
        reject([=](Shape &s) { std::memcpy(s.dst->op_params, &scale, sizeof(scale)); });
    }
    reject([](Shape &s) { float bias = 1; std::memcpy(&s.dst->op_params[1], &bias, sizeof(bias)); });
    reject([](Shape &s) { float softcap = 1; std::memcpy(&s.dst->op_params[2], &softcap, sizeof(softcap)); });
    reject([](Shape &s) { s.mask->type = GGML_TYPE_F32; }, true);
    reject([](Shape &s) { s.mask->ne[0] -= 1; }, true);
    reject([](Shape &s) { s.mask->ne[1] = 32; }, true);
    reject([](Shape &s) { s.mask->ne[2] = 2; }, true);
    reject([](Shape &s) { s.mask->ne[3] = 2; }, true);
    reject([](Shape &s) { s.mask->nb[0] = 4; }, true);
    reject([](Shape &s) { s.mask->nb[1] += 2; }, true);
    // Valid masked GQA and wider padded query masks retain the existing graph.
    { Shape s(33, 65, 4, true); s.k->ne[2] = s.v->ne[2] = 2;
      s.mask->ne[1] = 64; assert(ggml_sycl_onednn_sdpa_shape_supported(s.dst)); ++accepted; }
    // The FP16 V view produced by the adapter may interchange token/head
    // strides. Scalar channels stay contiguous and staging preserves the view.
    { Shape s; s.v->nb[1] = size_t(s.v->ne[0] * s.v->ne[2]) * sizeof(ggml_fp16_t);
      s.v->nb[2] = size_t(s.v->ne[0]) * sizeof(ggml_fp16_t);
      assert(ggml_sycl_onednn_sdpa_shape_supported(s.dst)); ++accepted; }
    // Existing F32 KV conversion remains eligible at prefill lengths.
    { Shape s(33, 1024); s.k->type = s.v->type = GGML_TYPE_F32;
      s.k->nb[0] = s.v->nb[0] = sizeof(float);
      assert(ggml_sycl_onednn_sdpa_shape_supported(s.dst)); ++accepted; }
    for (float scale : {0.25f, 0.125f, 0.0625f, 1.0f}) {
        Shape s; std::memcpy(s.dst->op_params, &scale, sizeof(scale));
        assert(ggml_sycl_onednn_sdpa_shape_supported(s.dst)); ++accepted;
    }
    // Unmasked GQA/other head dimensions remain native until GPU acceptance.
    reject([](Shape &s) { s.k->ne[2] = s.v->ne[2] = 1; });
    reject([](Shape &s) { s.q->ne[0] = s.k->ne[0] = s.v->ne[0] = 64; });
    assert(!ggml_sycl_onednn_sdpa_shape_supported(nullptr)); ++rejected;
    std::printf("onednn_sdpa_shape accepted=%d rejected=%d status=passed\n", accepted, rejected);
}
