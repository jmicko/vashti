// Synthetic, content-free regression for the pinned attention/cache adapter.
// Default: CPU oracle/fallback checks. Optional: --device CUDA <PCI address>.
// Large graph timing only: --benchmark CUDA|Vulkan|SYCL <PCI address> [4|16|32 heads].
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <algorithm>
#include <cassert>
#include <chrono>
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
#include "stable-diffusion.h"

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
    const bool half_cache = std::strncmp(ggml_backend_name(backend), "CUDA", 4) == 0 ||
                            std::strncmp(ggml_backend_name(backend), "Vulkan", 6) == 0 ||
                            std::strncmp(ggml_backend_name(backend), "SYCL", 4) == 0;
    assert(output.first->type == (half_cache && !sage ? GGML_TYPE_F16 : GGML_TYPE_F32));
    assert(output.second->type == (half_cache ? GGML_TYPE_F16 : GGML_TYPE_F32));
    if (half_cache && sage) assert(output.first->src[1] == k);
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

static void attention_case(ggml_backend_t backend, int batch, bool masked, bool sage, float kv_scale, ggml_type v_type,
                           int dim = 128, int rows = 33, int keys = 65, int heads = 2, bool f32_precision = false) {
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
    if (flash) ggml_flash_attn_ext_set_prec(flash, f32_precision ? GGML_PREC_F32 : GGML_PREC_DEFAULT);
    const bool cuda = std::strncmp(ggml_backend_name(backend), "CUDA", 4) == 0;
    if (cuda && !sage && dim >= 64 && batch == 1 && kv_scale == 1.f && v_type == GGML_TYPE_F16) {
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
    std::printf("attention_layout dim=%d rows=%d keys=%d heads=%d batch=%d masked=%d sage=%d kv_scale=%.1f v_f16=%d f32_precision=%d relative_l2=%.8f max_error=%.8f\n",
                dim, rows, keys, heads, batch, masked, sage_op != nullptr, kv_scale, v_type == GGML_TYPE_F16, f32_precision, relative, maximum_error);
    assert(relative < (sage_op ? 0.05 : 0.003));
    ggml_gallocr_free(allocator); ggml_backend_buffer_free(inputs); ggml_free(ctx);
}

// Exercise the actual FLASH_ATTN_EXT inputs directly: the model adapter packs
// V before attention, which otherwise hides the independent strided-V branch.
// Nonzero view offsets and padding canaries catch aliasing the backing buffer
// instead of the logical input, or mistaking a padded view for dense storage.
static void sdpa_layout_case(ggml_backend_t backend, bool strided_k, bool strided_v,
                             bool masked, bool f32_precision, ggml_type kv_type = GGML_TYPE_F16,
                             int keys = 65) {
    constexpr int dim = 128, rows = 33, heads = 2, lead = 17, tail = 19;
    const int k_pitch = dim + (strided_k ? 8 : 0), v_pitch = dim + (strided_v ? 16 : 0);
    const size_t element_bytes = ggml_type_size(kv_type);
    auto ctx = ggml_init({8 * 1024 * 1024, nullptr, true});
    auto q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim, rows, heads, 1);
    auto k_storage = ggml_new_tensor_1d(ctx, kv_type, lead + heads * keys * k_pitch + tail);
    auto v_storage = ggml_new_tensor_1d(ctx, kv_type, lead + heads * keys * v_pitch + tail);
    auto view = [&](ggml_tensor* storage, int pitch) {
        return ggml_view_4d(ctx, storage, dim, keys, heads, 1,
                            pitch * element_bytes, keys * pitch * element_bytes,
                            heads * keys * pitch * element_bytes, lead * element_bytes);
    };
    auto k = view(k_storage, k_pitch), v = view(v_storage, v_pitch);
    assert(ggml_is_contiguous(k) == !strided_k);
    assert(ggml_is_contiguous(v) == !strided_v);
    auto mask = masked ? ggml_new_tensor_2d(ctx, GGML_TYPE_F16, keys, rows) : nullptr;
    auto inputs = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(inputs != nullptr);
    assert(static_cast<const char*>(k->data) == static_cast<const char*>(k_storage->data) + lead * element_bytes);
    assert(static_cast<const char*>(v->data) == static_cast<const char*>(v_storage->data) + lead * element_bytes);
    std::mt19937 rng(211);
    std::uniform_real_distribution<float> random(-0.7f, 0.7f);
    std::vector<float> qd(ggml_nelements(q)), kd(dim * keys * heads), vd(kd.size()), md(keys * rows);
    for (float& value : qd) value = random(rng);
    for (float& value : kd) value = random(rng);
    for (float& value : vd) value = random(rng);
    upload(q, qd);
    auto upload_view = [&](ggml_tensor* storage, int pitch, std::vector<float>& logical) {
        std::vector<float> data(ggml_nelements(storage), 42.f);
        for (int h = 0; h < heads; ++h) for (int t = 0; t < keys; ++t) for (int d = 0; d < dim; ++d) {
            data[lead + (h * keys + t) * pitch + d] = logical[(h * keys + t) * dim + d];
        }
        upload(storage, data);
        data = read_floats(storage);
        for (int h = 0; h < heads; ++h) for (int t = 0; t < keys; ++t) for (int d = 0; d < dim; ++d) {
            logical[(h * keys + t) * dim + d] = data[lead + (h * keys + t) * pitch + d];
        }
    };
    upload_view(k_storage, k_pitch, kd); upload_view(v_storage, v_pitch, vd);
    if (mask) {
        for (int r = 0; r < rows; ++r) for (int t = 0; t < keys; ++t) md[r * keys + t] = t <= r + 1 ? 0.f : -INFINITY;
        upload(mask, md);
    }
    auto output = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.f / std::sqrt(float(dim)), 0, 0);
    ggml_flash_attn_ext_set_prec(output, f32_precision ? GGML_PREC_F32 : GGML_PREC_DEFAULT);
    assert(ggml_backend_supports_op(backend, output));
    auto graph = ggml_new_graph(ctx); ggml_build_forward_expand(graph, output);
    auto allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    assert(ggml_gallocr_alloc_graph(allocator, graph));
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);
    auto actual = read_floats(output);
    double error = 0, power = 0, maximum_error = 0;
    for (int r = 0; r < rows; ++r) for (int h = 0; h < heads; ++h) {
        std::vector<double> logits(keys);
        double maximum = -INFINITY, sum = 0;
        for (int t = 0; t < keys; ++t) {
            double dot = 0;
            for (int d = 0; d < dim; ++d) {
                const float key = ggml_fp16_to_fp32(ggml_fp32_to_fp16(kd[(h * keys + t) * dim + d]));
                dot += qd[(h * rows + r) * dim + d] * double(key);
            }
            logits[t] = dot / std::sqrt(double(dim)) + (masked ? md[r * keys + t] : 0);
            maximum = std::max(maximum, logits[t]);
        }
        for (double& value : logits) { value = std::exp(value - maximum); sum += value; }
        for (int d = 0; d < dim; ++d) {
            double expected = 0;
            for (int t = 0; t < keys; ++t) {
                const float value = ggml_fp16_to_fp32(ggml_fp32_to_fp16(vd[(h * keys + t) * dim + d]));
                expected += logits[t] * value / sum;
            }
            const float observed = actual[(r * heads + h) * dim + d];
            assert(std::isfinite(observed));
            const double delta = observed - expected;
            error += delta * delta; power += expected * expected;
            maximum_error = std::max(maximum_error, std::abs(delta));
        }
    }
    const double relative = std::sqrt(error / std::max(power, 1e-30));
    std::printf("sdpa_layout strided_k=%d strided_v=%d masked=%d f32_precision=%d kv_f16=%d keys=%d "
                "offset_bytes=%zu relative_l2=%.8f max_error=%.8f\n",
                strided_k, strided_v, masked, f32_precision, kv_type == GGML_TYPE_F16,
                keys, lead * element_bytes, relative, maximum_error);
    assert(relative < 0.003);
    ggml_gallocr_free(allocator); ggml_backend_buffer_free(inputs); ggml_free(ctx);
}

static int attention_benchmark(ggml_backend_t backend, int heads) {
    constexpr int dim = 128, rows = 4056, keys = 8192;
    const size_t query_bytes = size_t(dim) * rows * heads * sizeof(float);
    const size_t cache_bytes = size_t(dim) * keys * heads * sizeof(ggml_fp16_t);
    const size_t input_bytes = query_bytes + 2 * cache_bytes;
    // Leave room for attention scratch, transposed V in the baseline layout,
    // and the output. This is an admission guard, not a workspace estimate.
    constexpr size_t minimum_free_bytes = 1024ull * 1024 * 1024;
    ggml_backend_dev_props props{};
    ggml_backend_dev_get_props(ggml_backend_get_device(backend), &props);
    if (props.type != GGML_BACKEND_DEVICE_TYPE_GPU || props.memory_free < minimum_free_bytes) {
        std::printf("attention_benchmark_error phase=0 available_bytes=%zu required_bytes=%zu\n",
                    props.memory_free, minimum_free_bytes);
        return 1;
    }
    struct Resources {
        ggml_context* context = nullptr;
        ggml_backend_buffer_t inputs = nullptr;
        ggml_gallocr_t allocator = nullptr;
        ~Resources() {
            ggml_gallocr_free(allocator);
            ggml_backend_buffer_free(inputs);
            ggml_free(context);
        }
    } resources;
    resources.context = ggml_init({8 * 1024 * 1024, nullptr, true});
    if (!resources.context) {
        std::puts("attention_benchmark_error phase=1 status=1");
        return 1;
    }
    auto ctx = resources.context;
    auto q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, rows, heads);
    auto k = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, dim, keys, heads);
    auto v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, dim, heads, keys, 1);
    resources.inputs = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!resources.inputs) {
        std::puts("attention_benchmark_error phase=2 status=1");
        return 1;
    }
    std::mt19937 rng(73);
    std::uniform_real_distribution<float> random(-0.7f, 0.7f);
    for (auto tensor : {q, k, v}) {
        // One temporary at a time bounds host scratch even with 16 heads.
        std::vector<float> values(ggml_nelements(tensor));
        for (auto& value : values) value = random(rng);
        upload(tensor, values);
    }
    auto output = ggml_ext_attention_ext(ctx, backend, q, k, v, heads,
                                         nullptr, true, true, 1.f, false);
    auto graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    ggml_tensor* flash = nullptr;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto node = ggml_graph_node(graph, i);
        if (node->op == GGML_OP_FLASH_ATTN_EXT) flash = node;
        if (!ggml_backend_supports_op(backend, node)) {
            std::puts("attention_benchmark_error phase=3 status=1");
            return 1;
        }
    }
    if (!flash || flash->src[1]->type != GGML_TYPE_F16 || flash->src[2]->type != GGML_TYPE_F16) {
        std::puts("attention_benchmark_error phase=3 status=2");
        return 1;
    }
    resources.allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!resources.allocator || !ggml_gallocr_alloc_graph(resources.allocator, graph)) {
        std::puts("attention_benchmark_error phase=4 status=1");
        return 1;
    }
    const size_t workspace_bytes = ggml_gallocr_get_buffer_size(resources.allocator, 0);
    const int backend_id = std::strncmp(ggml_backend_name(backend), "CUDA", 4) == 0 ? 1 :
                           std::strncmp(ggml_backend_name(backend), "SYCL", 4) == 0 ? 3 : 2;
    std::printf("attention_benchmark backend=%d queries=%d keys=%d dim=%d heads=%d batch=1 kv_f16=1 "
                "input_bytes=%zu output_bytes=%zu workspace_bytes=%zu graph_nodes=%d\n",
                backend_id, rows, keys, dim, heads, input_bytes, ggml_nbytes(output), workspace_bytes,
                ggml_graph_n_nodes(graph));
    std::fflush(stdout);
    // One warm evaluation followed by exactly three measured evaluations.
    // Synchronization is part of each complete graph evaluation; no operation
    // callback or extra per-node synchronization is installed.
    for (int iteration = 0; iteration < 4; ++iteration) {
        const auto started = std::chrono::steady_clock::now();
        auto status = ggml_backend_graph_compute(backend, graph);
        ggml_backend_synchronize(backend);
        if (status != GGML_STATUS_SUCCESS) {
            std::printf("attention_benchmark_error phase=5 status=%d\n", int(status));
            return 1;
        }
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::printf("attention_benchmark iteration=%d measured=%d seconds=%.8f\n",
                    iteration, iteration != 0, seconds);
        std::fflush(stdout);
    }
    auto actual = read_floats(output);
    for (float value : actual) {
        if (!std::isfinite(value)) {
            std::puts("attention_benchmark_error phase=6 status=1");
            return 1;
        }
    }
    std::puts("attention_benchmark_passed finite=1");
    return 0;
}

int main(int argc, char** argv) {
    rlimit no_core{0, 0}; assert(setrlimit(RLIMIT_CORE, &no_core) == 0);
    const bool benchmark = argc > 1 && std::strcmp(argv[1], "--benchmark") == 0;
    int benchmark_heads = 16;
    if (benchmark) {
        // Reject large CPU runs and unbounded sizes before initializing backends.
        if ((argc != 4 && argc != 5) ||
            (std::strcmp(argv[2], "CUDA") != 0 && std::strcmp(argv[2], "Vulkan") != 0 &&
             std::strcmp(argv[2], "SYCL") != 0) ||
            (argc == 5 && std::strcmp(argv[4], "4") != 0 && std::strcmp(argv[4], "16") != 0 &&
             std::strcmp(argv[4], "32") != 0)) {
            std::fputs("attention_benchmark_error phase=7 status=1\n", stderr);
            return 2;
        }
        if (argc == 5) benchmark_heads = std::strcmp(argv[4], "4") == 0 ? 4 :
                                        std::strcmp(argv[4], "32") == 0 ? 32 : 16;
        ggml_log_set([](enum ggml_log_level, const char*, void*) {}, nullptr);
        sd_set_log_callback([](enum sd_log_level_t, const char*, void*) {}, nullptr);
    }
    ggml_backend_t backend = nullptr;
    if (argc == 1) backend = ggml_backend_cpu_init();
    else {
        assert(benchmark || (argc == 4 && std::strcmp(argv[1], "--device") == 0));
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
    if (benchmark) {
        if (!backend) {
            std::puts("attention_benchmark_error phase=8 status=1");
            return 1;
        }
        const int status = attention_benchmark(backend, benchmark_heads);
        ggml_backend_free(backend);
        return status;
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
    // Every side of query, key, and head fragment boundaries. The 65x1025
    // case also activates mask optimization. Each has an independent dense
    // oracle, with F16 and F32 QK accumulators and causal/unmasked variants.
    struct shape { int dim, rows, keys; };
    const shape shapes[] = {
        {8, 2, 1}, {24, 15, 7}, {64, 16, 8}, {72, 17, 15},
        {128, 31, 16}, {128, 32, 17}, {128, 33, 31}, {128, 47, 32},
        {128, 65, 33}, {128, 15, 63}, {128, 16, 64}, {128, 17, 65},
        {72, 33, 127}, {256, 17, 33}, {128, 65, 1025},
    };
    for (const auto& test : shapes) for (bool masked : {false, true}) for (bool f32 : {false, true}) {
        attention_case(backend, 1, masked, false, 1, GGML_TYPE_F16,
                       test.dim, test.rows, test.keys, 2, f32);
    }
    for (bool strided_k : {false, true}) for (bool strided_v : {false, true})
        for (bool masked : {false, true}) for (bool f32 : {false, true}) {
            sdpa_layout_case(backend, strided_k, strided_v, masked, f32);
        }
    // Preserve the existing F32-to-F16 staging path at supported prefill sizes.
    for (bool masked : {false, true}) {
        sdpa_layout_case(backend, true, true, masked, true, GGML_TYPE_F32, 1025);
    }
    ggml_backend_free(backend);
    std::puts("attention_layout_passed");
}
