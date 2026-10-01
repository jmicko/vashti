// Numeric-only copy oracle: CPU by default; explicit physical SYCL device modes.
// Wide modes allocate a device buffer but touch only a tiny strided view. They
// isolate byte addressing without billions of work-items or large host staging.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <sys/resource.h>
#include <vector>

static_assert(GGML_MAX_NAME == 160);
static size_t scalar_size(ggml_type type) { assert(type == GGML_TYPE_F32 || type == GGML_TYPE_F16); return type == GGML_TYPE_F32 ? 4 : 2; }
static float value(int row, int col) { return float(row * 8 + col - 4) / 4.f; }
static void set_scalar(ggml_tensor * tensor, size_t offset, float v) {
    if (tensor->type == GGML_TYPE_F32) ggml_backend_tensor_set(tensor, &v, offset, sizeof(v));
    else { auto h = ggml_fp32_to_fp16(v); ggml_backend_tensor_set(tensor, &h, offset, sizeof(h)); }
}
static float get_scalar(const ggml_tensor * tensor, size_t offset) {
    if (tensor->type == GGML_TYPE_F32) { float v; ggml_backend_tensor_get(tensor, &v, offset, sizeof(v)); return v; }
    ggml_fp16_t h; ggml_backend_tensor_get(tensor, &h, offset, sizeof(h)); return ggml_fp16_to_fp32(h);
}
static void compute(ggml_backend_t backend, ggml_context * ctx, ggml_tensor * result) {
    assert(ggml_backend_supports_op(backend, result));
    auto graph = ggml_new_graph(ctx); ggml_build_forward_expand(graph, result);
    auto alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    assert(ggml_gallocr_alloc_graph(alloc, graph));
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);
    ggml_gallocr_free(alloc); // Results owned by the caller must be read first.
}

static void strided(ggml_backend_t backend, size_t stride, ggml_type source_type,
                    ggml_type dest_type, bool wide_dest, int id) {
    constexpr int cols = 3, rows = 2;
    constexpr size_t prefix = 16;
    const auto wide_type = wide_dest ? dest_type : source_type;
    const size_t element = scalar_size(wide_type), bytes = prefix + stride + cols * element + 16;
    auto ctx = ggml_init({1024 * 1024, nullptr, true}); assert(ctx);
    auto storage = ggml_new_tensor_1d(ctx, wide_type, (bytes + element - 1) / element);
    auto view = ggml_view_2d(ctx, storage, cols, rows, stride, prefix);
    auto small = ggml_new_tensor_2d(ctx, wide_dest ? source_type : dest_type, cols, rows);
    auto inputs = ggml_backend_alloc_ctx_tensors(ctx, backend); assert(inputs);
    auto source = wide_dest ? small : view;
    auto dest = wide_dest ? view : small;
    // Every accessed position is initialized, plus surrounding write canaries.
    for (int row = 0; row < rows; ++row) {
        set_scalar(storage, prefix + row * stride - element, -123.f);
        set_scalar(storage, prefix + row * stride + cols * element, 123.f);
        for (int col = 0; col < cols; ++col) {
            set_scalar(dest, row * dest->nb[1] + col * dest->nb[0], -99.f);
            set_scalar(source, row * source->nb[1] + col * source->nb[0], value(row, col));
        }
    }
    auto result = ggml_cpy(ctx, source, dest);
    std::printf("copy_start id=%d stride=%llu source_type=%d destination_type=%d wide_destination=%d storage_bytes=%zu elements=6\n",
                id, (unsigned long long)stride, int(source_type), int(dest_type), int(wide_dest), ggml_nbytes(storage)); std::fflush(stdout);
    const auto start = std::chrono::steady_clock::now();
    compute(backend, ctx, result);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) assert(get_scalar(dest, row * dest->nb[1] + col * dest->nb[0]) == value(row, col));
        assert(get_scalar(storage, prefix + row * stride - element) == -123.f);
        assert(get_scalar(storage, prefix + row * stride + cols * element) == 123.f);
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    std::printf("copy_pass id=%d checks=10 ms=%.6f\n", id, ms); std::fflush(stdout);
    ggml_backend_buffer_free(inputs); ggml_free(ctx);
}

static void contiguous_permute(ggml_backend_t backend, ggml_type type, int id) {
    auto ctx = ggml_init({1024 * 1024, nullptr, true}); assert(ctx);
    auto original = ggml_new_tensor_4d(ctx, type, 7, 5, 3, 2);
    auto input_buffer = ggml_backend_alloc_ctx_tensors(ctx, backend); assert(input_buffer);
    for (int i = 0; i < 210; ++i) set_scalar(original, i * scalar_size(type), float(i - 100) / 4.f);
    auto source = ggml_permute(ctx, original, 1, 0, 2, 3);
    auto result = ggml_cont(ctx, source);
    auto graph = ggml_new_graph(ctx); ggml_build_forward_expand(graph, result);
    auto alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)); assert(ggml_gallocr_alloc_graph(alloc, graph));
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS); ggml_backend_synchronize(backend);
    for (int n = 0; n < 2; ++n) for (int z = 0; z < 3; ++z) for (int y = 0; y < 7; ++y) for (int x = 0; x < 5; ++x) {
        const int source_index = ((n * 3 + z) * 5 + x) * 7 + y;
        const int dest_index = ((n * 3 + z) * 7 + y) * 5 + x;
        assert(get_scalar(result, dest_index * scalar_size(type)) == float(source_index - 100) / 4.f);
    }
    std::printf("copy_permute_pass id=%d checks=210 type=%d\n", id, int(type)); std::fflush(stdout);
    ggml_gallocr_free(alloc); ggml_backend_buffer_free(input_buffer); ggml_free(ctx);
}

static void quantized_rows(ggml_backend_t backend, int id) {
    // Exact synthetic Q8_0 blocks: half scale0.25 and signed integer lanes.
    // Test padded source dequantization and padded same-type block copying.
    constexpr size_t block_bytes = 34, stride = 102;
    auto ctx = ggml_init({1024 * 1024, nullptr, true}); assert(ctx);
    auto storage = ggml_new_tensor_1d(ctx, GGML_TYPE_Q8_0, 128);
    auto source = ggml_view_2d(ctx, storage, 32, 2, stride, 0);
    auto quant_dest = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 32, 2);
    auto float_dest = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 2);
    auto inputs = ggml_backend_alloc_ctx_tensors(ctx, backend); assert(inputs);
    unsigned char data[block_bytes]{};
    auto scale = ggml_fp32_to_fp16(0.25f); std::memcpy(data, &scale, 2);
    for (int row = 0; row < 2; ++row) {
        for (int col = 0; col < 32; ++col) data[2 + col] = uint8_t(int8_t(col - 16 + row));
        ggml_backend_tensor_set(source, data, row * stride, sizeof(data));
    }
    compute(backend, ctx, ggml_cpy(ctx, source, quant_dest));
    compute(backend, ctx, ggml_cpy(ctx, source, float_dest));
    for (int row = 0; row < 2; ++row) {
        unsigned char actual[block_bytes]; ggml_backend_tensor_get(quant_dest, actual, row * block_bytes, block_bytes);
        assert(std::memcmp(actual, &scale, 2) == 0);
        for (int col = 0; col < 32; ++col) {
            assert(int8_t(actual[2 + col]) == col - 16 + row);
            assert(get_scalar(float_dest, (row * 32 + col) * 4) == float(col - 16 + row) / 4.f);
        }
    }
    std::printf("copy_quantized_pass id=%d checks=130\n", id); std::fflush(stdout);
    ggml_backend_buffer_free(inputs); ggml_free(ctx);
}

int main(int argc, char ** argv) {
    rlimit limit{0, 0}; assert(setrlimit(RLIMIT_CORE, &limit) == 0);
    assert(argc == 1 || argc == 5);
    ggml_backend_t backend = nullptr;
    if (argc == 1) { backend = ggml_backend_cpu_init(); ggml_backend_cpu_set_n_threads(backend, 2); }
    else {
        assert(!std::strcmp(argv[1], "--device") && !std::strcmp(argv[2], "SYCL"));
        ggml_backend_load_all();
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto d = ggml_backend_dev_get(i); ggml_backend_dev_props props{}; ggml_backend_dev_get_props(d, &props);
            if (props.name && !std::strncmp(props.name, "SYCL", 4) && props.device_id && !std::strcmp(props.device_id, argv[3])) {
                assert(!backend); backend = ggml_backend_dev_init(d, nullptr);
            }
        }
    }
    assert(backend);
    const int mode = argc == 1 ? 1 : (argv[4][1] == 0 ? argv[4][0] - '0' : -1);
    assert(mode >= 1 && mode <= 3);
    if (mode == 1) {
        contiguous_permute(backend, GGML_TYPE_F32, 1);
        contiguous_permute(backend, GGML_TYPE_F16, 2);
        quantized_rows(backend, 3);
    }
    const size_t stride = mode == 1 ? 192 : (size_t(1) << (mode == 2 ? 31 : 32)) + 64;
    int id = 4;
    for (auto source : {GGML_TYPE_F32, GGML_TYPE_F16}) for (auto dest : {GGML_TYPE_F32, GGML_TYPE_F16})
        for (bool wide_dest : {false, true}) strided(backend, stride, source, dest, wide_dest, id++);
    ggml_backend_free(backend);
    std::puts("copy_acceptance_passed");
}
