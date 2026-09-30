// Synthetic, content-free regression for the pinned attention/cache adapter.
// Default: CPU oracle/fallback checks. Optional: --device CUDA <PCI address>.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <sys/resource.h>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "model/diffusion/qwen_image_2_1.hpp"

static std::vector<float> read_floats(ggml_tensor* tensor) {
    const size_t count = ggml_nelements(tensor);
    std::vector<float> result(count);
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(tensor, result.data(), 0, count * sizeof(float));
    } else {
        assert(tensor->type == GGML_TYPE_F16);
        std::vector<ggml_fp16_t> values(count);
        ggml_backend_tensor_get(tensor, values.data(), 0, count * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < count; ++i) result[i] = ggml_fp16_to_fp32(values[i]);
    }
    return result;
}

static void upload(ggml_tensor* tensor, const std::vector<float>& values) {
    assert(static_cast<size_t>(ggml_nelements(tensor)) == values.size());
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(float));
    } else {
        std::vector<ggml_fp16_t> half(values.size());
        for (size_t i = 0; i < half.size(); ++i) half[i] = ggml_fp32_to_fp16(values[i]);
        ggml_backend_tensor_set(tensor, half.data(), 0, half.size() * sizeof(ggml_fp16_t));
    }
}

static void cache_case(ggml_backend_t backend, bool sage) {
    constexpr int dim = 128, heads = 2, prefix = 65, live = 33;
    auto ctx = ggml_init({8 * 1024 * 1024, nullptr, true});
    auto k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim, live, heads, 1);
    auto v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim, heads, live, 1);
    auto pk = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, dim, prefix, heads, 1);
    auto pv = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, dim, heads, prefix, 1);
    auto inputs = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(inputs != nullptr);
    std::vector<float> kd(ggml_nelements(k)), vd(kd.size());
    std::vector<float> pkd(ggml_nelements(pk)), pvd(pkd.size());
    for (size_t i = 0; i < kd.size(); ++i) {
        kd[i] = 0.33331f + float(i % 29) * 0.00231f;
        vd[i] = -0.23331f + float(i % 31) * 0.00311f;
    }
    for (size_t i = 0; i < pkd.size(); ++i) {
        pkd[i] = 0.11113f + float(i % 37) * 0.00171f;
        pvd[i] = -0.41113f + float(i % 41) * 0.00121f;
    }
    upload(k, kd); upload(v, vd); upload(pk, pkd); upload(pv, pvd);
    pkd = read_floats(pk); pvd = read_floats(pv);
    GGMLRunnerContext context;
    context.backend = backend; context.ggml_ctx = ctx;
    context.flash_attn_enabled = true; context.sage_attn_enabled = sage;
    auto output = Qwen::QwenImage21Attention::concat_cached_kv(&context, k, v, pk, pv);
    const bool cuda = std::strncmp(ggml_backend_name(backend), "CUDA", 4) == 0;
    assert(output.first->type == (cuda && !sage ? GGML_TYPE_F16 : GGML_TYPE_F32));
    assert(output.second->type == (cuda ? GGML_TYPE_F16 : GGML_TYPE_F32));
    if (cuda && sage) assert(output.first->src[1] == k);
    auto graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output.first);
    ggml_build_forward_expand(graph, output.second);
    auto allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    assert(ggml_gallocr_alloc_graph(allocator, graph));
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);
    auto out_k = read_floats(output.first), out_v = read_floats(output.second);
    auto rounded = [](float x) { return ggml_fp16_to_fp32(ggml_fp32_to_fp16(x)); };
    for (int h = 0; h < heads; ++h) for (int t = 0; t < prefix + live; ++t) for (int d = 0; d < dim; ++d) {
        float expected_k = t < prefix ? pkd[(h * prefix + t) * dim + d] : kd[(h * live + t - prefix) * dim + d];
        float expected_v = t < prefix ? pvd[(t * heads + h) * dim + d] : vd[((t - prefix) * heads + h) * dim + d];
        if (output.first->type == GGML_TYPE_F16) expected_k = rounded(expected_k);
        if (output.second->type == GGML_TYPE_F16) expected_v = rounded(expected_v);
        assert(out_k[(h * (prefix + live) + t) * dim + d] == expected_k);
        assert(out_v[(t * heads + h) * dim + d] == expected_v);
    }
    ggml_gallocr_free(allocator); ggml_backend_buffer_free(inputs); ggml_free(ctx);
}

static void attention_case(ggml_backend_t backend, int batch, bool masked, bool sage, float kv_scale, ggml_type v_type) {
    constexpr int dim = 128, rows = 33, keys = 65, heads = 2;
    auto ctx = ggml_init({8 * 1024 * 1024, nullptr, true});
    auto q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, rows, heads * batch);
    auto k = ggml_new_tensor_3d(ctx, sage || kv_scale != 1.f ? GGML_TYPE_F32 : GGML_TYPE_F16, dim, keys, heads * batch);
    auto v = ggml_new_tensor_4d(ctx, v_type, dim, heads, keys, batch);
    auto mask = masked ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, keys, rows) : nullptr;
    auto inputs = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(inputs != nullptr);
    std::mt19937 rng(73);
    std::uniform_real_distribution<float> random(-0.7f, 0.7f);
    std::vector<float> qd(ggml_nelements(q)), kd(ggml_nelements(k)), vd(ggml_nelements(v)), md(keys * rows);
    for (float& x : qd) x = random(rng);
    for (float& x : kd) x = random(rng);
    for (float& x : vd) x = random(rng);
    upload(q, qd); upload(k, kd); upload(v, vd);
    kd = read_floats(k); vd = read_floats(v);
    if (mask) {
        for (int r = 0; r < rows; ++r) for (int t = 0; t < keys; ++t) md[r * keys + t] = t <= r + 1 ? 0.f : -INFINITY;
        upload(mask, md);
    }
    auto output = ggml_ext_attention_ext(ctx, backend, q, k, v, heads, mask, true, true, kv_scale, sage);
    auto graph = ggml_new_graph(ctx); ggml_build_forward_expand(graph, output);
    ggml_tensor* flash = nullptr; ggml_tensor* sage_op = nullptr;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto node = ggml_graph_node(graph, i);
        if (node->op == GGML_OP_FLASH_ATTN_EXT) flash = node;
        if (node->op == GGML_OP_SAGE_ATTN) sage_op = node;
        // Casting to the same type is a real copy in the pinned engine.
        if (node->op == GGML_OP_CPY && node != output) assert(node->src[0]->type != node->type);
    }
    if (masked) assert(sage_op == nullptr);
    const bool cuda = std::strncmp(ggml_backend_name(backend), "CUDA", 4) == 0;
    if (cuda && !sage && batch == 1 && kv_scale == 1.f && v_type == GGML_TYPE_F16) {
        assert(flash != nullptr && flash->src[2]->op == GGML_OP_PERMUTE);
        assert(flash->src[2]->src[0] == v && !ggml_is_contiguous(flash->src[2]));
    }
    auto allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    assert(ggml_gallocr_alloc_graph(allocator, graph));
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);
    auto actual = read_floats(output);
    double error = 0, power = 0, maximum_error = 0;
    for (int b = 0; b < batch; ++b) for (int r = 0; r < rows; ++r) for (int h = 0; h < heads; ++h) {
        std::vector<double> logits(keys);
        double maximum = -INFINITY, sum = 0;
        for (int t = 0; t < keys; ++t) {
            double dot = 0;
            for (int d = 0; d < dim; ++d) {
                float key = kd[((b * heads + h) * keys + t) * dim + d];
                if (flash) key = ggml_fp16_to_fp32(ggml_fp32_to_fp16(key * kv_scale)) / kv_scale;
                dot += qd[((b * heads + h) * rows + r) * dim + d] * double(key);
            }
            logits[t] = dot / std::sqrt(double(dim)) + (masked ? md[r * keys + t] : 0);
            maximum = std::max(maximum, logits[t]);
        }
        for (double& x : logits) { x = std::exp(x - maximum); sum += x; }
        for (int d = 0; d < dim; ++d) {
            double expected = 0;
            for (int t = 0; t < keys; ++t) {
                float value = vd[((b * keys + t) * heads + h) * dim + d];
                if (flash || sage_op) value = ggml_fp16_to_fp32(ggml_fp32_to_fp16(value * kv_scale)) / kv_scale;
                expected += logits[t] * value / sum;
            }
            const float observed = actual[((b * rows + r) * heads + h) * dim + d];
            assert(std::isfinite(observed));
            double delta = observed - expected;
            error += delta * delta; power += expected * expected;
            maximum_error = std::max(maximum_error, std::abs(delta));
        }
    }
    const double relative = std::sqrt(error / std::max(power, 1e-30));
    std::printf("attention_layout batch=%d masked=%d sage=%d kv_scale=%.1f v_f16=%d relative_l2=%.8f max_error=%.8f\n",
                batch, masked, sage_op != nullptr, kv_scale, v_type == GGML_TYPE_F16, relative, maximum_error);
    assert(relative < (sage_op ? 0.05 : 0.003));
    ggml_gallocr_free(allocator); ggml_backend_buffer_free(inputs); ggml_free(ctx);
}

int main(int argc, char** argv) {
    rlimit no_core{0, 0}; assert(setrlimit(RLIMIT_CORE, &no_core) == 0);
    ggml_backend_t backend = nullptr;
    if (argc == 1) backend = ggml_backend_cpu_init();
    else {
        assert(argc == 4 && std::strcmp(argv[1], "--device") == 0);
        ggml_backend_load_all();
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto device = ggml_backend_dev_get(i); ggml_backend_dev_props props{};
            ggml_backend_dev_get_props(device, &props);
            if (props.device_id && std::strcmp(props.device_id, argv[3]) == 0 &&
                props.name && std::strncmp(props.name, argv[2], std::strlen(argv[2])) == 0) {
                assert(backend == nullptr); backend = ggml_backend_dev_init(device, nullptr);
            }
        }
    }
    assert(backend != nullptr);
    if (std::strncmp(ggml_backend_name(backend), "CPU", 3) == 0) ggml_backend_cpu_set_n_threads(backend, 2);
    cache_case(backend, false); cache_case(backend, true);
    attention_case(backend, 1, false, false, 1, GGML_TYPE_F16);
    attention_case(backend, 1, true, false, 1, GGML_TYPE_F16);
    attention_case(backend, 2, false, false, 1, GGML_TYPE_F16);
    attention_case(backend, 1, false, false, 2, GGML_TYPE_F32);
    attention_case(backend, 1, false, false, 1, GGML_TYPE_F32);
    attention_case(backend, 1, false, true, 1, GGML_TYPE_F16);
    attention_case(backend, 1, true, true, 1, GGML_TYPE_F16);
    ggml_backend_free(backend);
    std::puts("attention_layout_passed");
}
