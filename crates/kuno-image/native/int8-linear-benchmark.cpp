// Synthetic, content-free ConvRot quantizer/GEMM phase benchmark.
// Default: tiny CPU numeric checks. Explicit --device <backend prefix> <PCI>
// runs the same checks on that device. Add --benchmark <shape ID> <repeats>:
// 1 = 4056 rows, K4096/N4096; 2 = K4096/N24576; 3 = K12288/N4096.
// These are Qwen Image 2.1 projection, fused gate/up, and MLP down shapes.
// No files, model weights, user text, or image data are loaded or written.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <sys/resource.h>
#include <vector>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

static constexpr int group_size = 256;
static constexpr int h4[4][4] = {{1,1,1,-1}, {1,1,-1,1}, {1,-1,1,1}, {-1,1,1,1}};

// Independent dense tensor-product oracle, deliberately not the backend's FHT.
static std::vector<double> dense_rotation(const float * x) {
    std::vector<double> result(group_size);
    for (int i = 0; i < group_size; ++i) for (int j = 0; j < group_size; ++j) {
        int a = i, b = j, sign = 1;
        for (int digit = 0; digit < 4; ++digit) {
            sign *= h4[a % 4][b % 4]; a /= 4; b /= 4;
        }
        result[i] += double(x[j]) * sign / 16.;
    }
    return result;
}

// Fast inverse for benchmark preparation. H4 is symmetric and orthogonal
// after normalization, so the forward and inverse transforms are identical.
static void fast_rotation(float * x) {
    for (int i = 0; i < group_size; ++i) x[i] *= 1.f / 16.f;
    for (int stride = 1; stride < group_size; stride *= 4) {
        for (int block = 0; block < group_size; block += 4 * stride) for (int i = 0; i < stride; ++i) {
            const int at = block + i;
            const float a = x[at], b = x[at+stride], c = x[at+2*stride], d = x[at+3*stride];
            x[at] = a+b+c-d; x[at+stride] = a+b-c+d;
            x[at+2*stride] = a-b+c+d; x[at+3*stride] = -a+b+c+d;
        }
    }
}

static std::vector<float> read_f32(ggml_tensor * tensor) {
    assert(tensor->type == GGML_TYPE_F32);
    std::vector<float> values(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
    return values;
}

static void compute(ggml_backend_t backend, ggml_cgraph * graph) {
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);
}

static void measure(ggml_backend_t backend, ggml_cgraph * graph, int phase,
                    int rows, int k, int n, uint64_t bytes, int repetitions) {
    for (int pass = 0; pass <= repetitions; ++pass) {
        const auto start = std::chrono::steady_clock::now();
        compute(backend, graph);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::printf("int8_phase phase=%d rows=%d k=%d n=%d bytes=%llu cold=%d pass=%d seconds=%.9f\n",
                    phase, rows, k, n, (unsigned long long)bytes, pass == 0, pass, seconds);
        std::fflush(stdout);
    }
}

static void run_case(ggml_backend_t backend, int rows, int k, int n, int repetitions, bool numeric, bool has_bias) {
    assert(rows > 0 && k % group_size == 0 && n > 0);
    const uint64_t resident_bytes = uint64_t(rows)*k*9 + uint64_t(k)*n*3 + uint64_t(rows)*n*8 + uint64_t(rows+n)*8;
    assert(resident_bytes <= 2ull * 1024 * 1024 * 1024);
    auto ctx = ggml_init({8 * 1024 * 1024, nullptr, true});
    auto x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, rows);
    auto w = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, k, n);
    auto ws = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    auto bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    auto input_buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(input_buffer != nullptr);
    std::mt19937 random(47);
    std::vector<float> xd(size_t(rows)*k), scales(n), bd(n);
    std::vector<int8_t> wd(size_t(n)*k);
    for (int row = 0; row < rows; ++row) for (int i = 0; i < k; ++i)
        xd[size_t(row)*k+i] = row == 0 ? 0.f : (int(random()%127)-63) / 64.f;
    for (auto & value : wd) value = int8_t(int(random()%31)-15);
    for (int out = 0; out < n; ++out) {
        scales[out] = (out % 5 + 1) / 128.f;
        bd[out] = (out % 17 - 8) / 32.f;
    }
    ggml_backend_tensor_set(x, xd.data(), 0, xd.size()*sizeof(float));
    ggml_backend_tensor_set(w, wd.data(), 0, wd.size());
    ggml_backend_tensor_set(ws, scales.data(), 0, scales.size()*sizeof(float));
    ggml_backend_tensor_set(bias, bd.data(), 0, bd.size()*sizeof(float));
    auto packed = ggml_quantize_i8_convrot(ctx, x, group_size);
    auto output = ggml_mul_mat_i8_tensorwise(ctx, w, packed, ws, has_bias ? bias : nullptr, group_size);
    assert(ggml_backend_supports_op(backend, packed));
    assert(ggml_backend_supports_op(backend, output));
    auto combined = ggml_new_graph(ctx); ggml_build_forward_expand(combined, output);
    assert(ggml_graph_n_nodes(combined) == 2);
    auto allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    assert(ggml_gallocr_alloc_graph(allocator, combined));
    // Reuse the complete graph's allocations. Keep packed->op/src intact:
    // backend GEMM support checks its logical quantizer input and packed layout.
    auto quant_graph = ggml_new_graph(ctx); ggml_graph_add_node(quant_graph, packed);
    auto gemm_graph = ggml_new_graph(ctx); ggml_graph_add_node(gemm_graph, output);
    if (!numeric) {
        // Phase 1: quantizer; 2: already packed GEMM; 3: combined.
        measure(backend, quant_graph, 1, rows, k, n, ggml_nbytes(x)+ggml_nbytes(packed), repetitions);
        measure(backend, gemm_graph, 2, rows, k, n, ggml_nbytes(w)+ggml_nbytes(packed)+ggml_nbytes(output), repetitions);
        measure(backend, combined, 3, rows, k, n, ggml_nbytes(x)+ggml_nbytes(w)+ggml_nbytes(packed)+ggml_nbytes(output), repetitions);
    } else {
        compute(backend, quant_graph);
        compute(backend, gemm_graph);
    }
    std::vector<int8_t> quantized(size_t(rows)*k);
    std::vector<float> activation_scales(rows);
    ggml_backend_tensor_get(packed, quantized.data(), 0, quantized.size());
    const size_t scale_offset = size_t(k) * ((rows+3)/4*4);
    ggml_backend_tensor_get(packed, activation_scales.data(), scale_offset, size_t(rows)*sizeof(float));
    auto actual = read_f32(output);
    for (float value : actual) assert(std::isfinite(value));

    double maximum_scale_error = 0., maximum_gemm_error = 0.; int maximum_quant_error = 0;
    if (numeric) {
        for (int row = 0; row < rows; ++row) {
            std::vector<double> rotated(k); double maximum = 0.;
            for (int base = 0; base < k; base += group_size) {
                auto group = dense_rotation(xd.data()+size_t(row)*k+base);
                for (int i = 0; i < group_size; ++i) {
                    rotated[base+i] = group[i]; maximum = std::max(maximum, std::abs(group[i]));
                }
            }
            const double expected_scale = maximum / 127.;
            maximum_scale_error = std::max(maximum_scale_error, std::abs(activation_scales[row]-expected_scale));
            assert(std::abs(activation_scales[row]-expected_scale) <= 1e-7 + expected_scale*1e-6);
            for (int i = 0; i < k; ++i) {
                const int expected = expected_scale == 0. ? 0 : std::clamp(int(std::nearbyint(rotated[i]/expected_scale)), -127, 127);
                const int difference = std::abs(int(quantized[size_t(row)*k+i])-expected);
                maximum_quant_error = std::max(maximum_quant_error, difference);
                assert(difference <= 1); // Tie/FP32 rounding may differ by one quantizer unit.
            }
            for (int out = 0; out < n; ++out) {
                int64_t sum = 0;
                for (int i = 0; i < k; ++i) sum += int(quantized[size_t(row)*k+i]) * int(wd[size_t(out)*k+i]);
                const double expected = double(sum)*activation_scales[row]*scales[out] + (has_bias ? bd[out] : 0.f);
                const double delta = std::abs(actual[size_t(row)*n+out]-expected);
                maximum_gemm_error = std::max(maximum_gemm_error, delta);
                assert(delta <= 2e-6 + std::abs(expected)*2e-6);
            }
        }
    }

    // Compare against a dense matrix for the SAME dequantized source weight.
    // Stored I8 weights are already rotated; invert H256, then convert to F16.
    // Input remains the original FP32 x, so comparison includes activation
    // quantization and F16 weight/compute rounding, not only GEMM error.
    auto dense_ctx = ggml_init({8 * 1024 * 1024, nullptr, true});
    auto dense_w = ggml_new_tensor_2d(dense_ctx, GGML_TYPE_F16, k, n);
    auto dense_x = ggml_new_tensor_2d(dense_ctx, GGML_TYPE_F32, k, rows);
    auto dense_bias = ggml_new_tensor_1d(dense_ctx, GGML_TYPE_F32, n);
    auto dense_inputs = ggml_backend_alloc_ctx_tensors(dense_ctx, backend);
    assert(dense_inputs != nullptr);
    std::vector<ggml_fp16_t> half_weight(size_t(n)*k);
    float group[group_size];
    for (int out = 0; out < n; ++out) for (int base = 0; base < k; base += group_size) {
        for (int i = 0; i < group_size; ++i) group[i] = wd[size_t(out)*k+base+i] * scales[out];
        std::vector<double> oracle;
        if (numeric) oracle = dense_rotation(group);
        fast_rotation(group);
        for (int i = 0; i < group_size; ++i) {
            if (numeric) assert(std::abs(group[i]-oracle[i]) < 1e-6);
            half_weight[size_t(out)*k+base+i] = ggml_fp32_to_fp16(group[i]);
        }
    }
    ggml_backend_tensor_set(dense_w, half_weight.data(), 0, half_weight.size()*sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(dense_x, xd.data(), 0, xd.size()*sizeof(float));
    ggml_backend_tensor_set(dense_bias, bd.data(), 0, bd.size()*sizeof(float));
    auto dense_output = ggml_mul_mat(dense_ctx, dense_w, dense_x);
    ggml_mul_mat_set_prec(dense_output, GGML_PREC_F32);
    assert(ggml_backend_supports_op(backend, dense_output));
    if (has_bias) dense_output = ggml_add(dense_ctx, dense_output, dense_bias);
    auto dense_graph = ggml_new_graph(dense_ctx); ggml_build_forward_expand(dense_graph, dense_output);
    auto dense_allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    assert(ggml_gallocr_alloc_graph(dense_allocator, dense_graph));
    if (numeric) compute(backend, dense_graph);
    else measure(backend, dense_graph, 4, rows, k, n, ggml_nbytes(dense_w)+ggml_nbytes(dense_x)+ggml_nbytes(dense_output), repetitions);
    auto dense_actual = read_f32(dense_output);
    double delta_power = 0., dense_power = 0.;
    for (size_t i = 0; i < actual.size(); ++i) {
        assert(std::isfinite(dense_actual[i]));
        const double delta = actual[i]-dense_actual[i];
        delta_power += delta*delta; dense_power += double(dense_actual[i])*dense_actual[i];
    }
    const double relative = std::sqrt(delta_power/std::max(dense_power, 1e-30));
    assert(relative < 0.03); // Different activation precision; not bit-equivalent.
    std::printf("int8_validation rows=%d k=%d n=%d numeric=%d bias=%d quant_max_units=%d scale_max_error=%.9g gemm_max_error=%.9g dense_relative_l2=%.9g resident_bound_bytes=%llu\n",
                rows,k,n,numeric,has_bias,maximum_quant_error,maximum_scale_error,maximum_gemm_error,relative,(unsigned long long)resident_bytes);
    std::fflush(stdout);
    ggml_gallocr_free(dense_allocator); ggml_backend_buffer_free(dense_inputs); ggml_free(dense_ctx);
    ggml_gallocr_free(allocator); ggml_backend_buffer_free(input_buffer); ggml_free(ctx);
}

static int bounded_number(const char * text, int maximum) {
    char * end = nullptr; const long value = std::strtol(text, &end, 10);
    assert(end != text && *end == '\0' && value > 0 && value <= maximum);
    return int(value);
}

int main(int argc, char ** argv) {
    rlimit no_core{0,0}; assert(setrlimit(RLIMIT_CORE, &no_core) == 0);
    assert(argc == 1 || argc == 4 || argc == 7);
    ggml_backend_t backend = nullptr;
    if (argc == 1) backend = ggml_backend_cpu_init();
    else {
        assert(std::strcmp(argv[1], "--device") == 0);
        ggml_backend_load_all();
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto device = ggml_backend_dev_get(i); ggml_backend_dev_props props{};
            ggml_backend_dev_get_props(device, &props);
            if (props.device_id && std::strcmp(props.device_id, argv[3]) == 0 && props.name &&
                std::strncmp(props.name, argv[2], std::strlen(argv[2])) == 0) {
                assert(backend == nullptr); backend = ggml_backend_dev_init(device, nullptr);
            }
        }
    }
    assert(backend != nullptr);
    if (argc == 1) ggml_backend_cpu_set_n_threads(backend, 2);
    // The 67x128 case executes a complete 8x8 cooperative tile plus a scalar
    // row tail, so every bounded asymmetric candidate has an independent oracle.
    for (const auto shape : {std::vector<int>{3,256,64}, {17,256,32}, {31,512,64},
                            {31,512,65}, {33,256,64}, {35,512,128}, {67,512,128}})
        run_case(backend, shape[0], shape[1], shape[2], 0, true, true);
    if (argc == 7) {
        assert(std::strcmp(argv[4], "--benchmark") == 0);
        // Tiny acceptance warmed generic pipelines; pass zero is the first
        // invocation of each large timed graph, not a process-cold claim.
        std::puts("int8_phase_status validation_warmup=1 process_cold=0");
        const int shape = bounded_number(argv[5],3), repetitions = bounded_number(argv[6],8);
        const int k = shape == 3 ? 12288 : 4096;
        const int n = shape == 2 ? 24576 : 4096;
        run_case(backend, 4056, k, n, repetitions, false, false);
    }
    ggml_backend_free(backend);
    std::puts("int8_benchmark_passed");
}
