// Synthetic materialized-F32 arithmetic oracle. Default initializes CPU only.
// A SYCL build with KUNO_SYCL_FUSION_DIRECT verifies the actual fast kernel,
// then the eligible backend graph; GPU invocation is explicit and PCI-bound.
#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <limits>
#include <sys/resource.h>
#include <vector>
#ifdef __SSE__
#include <xmmintrin.h>
#endif
#include "ggml.h"
static_assert(GGML_MAX_NAME == 160, "Use the image engine GGML_MAX_NAME=160 ABI");
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "mul-add-policy.hpp"
#include "pci-address.hpp"
#include "binary32-reference.hpp"
#ifdef KUNO_SYCL_FUSION_DIRECT
#include "ggml-backend-impl.h"
#include "binbcast.hpp"
#include "fusion.hpp"
#endif

struct BackendError { int phase; };
static bool backend_safe = true;
static bool device_backend_mode = false;
[[noreturn]] static void uncertain_backend(int phase) {
    backend_safe = false;
    if (device_backend_mode) {
        // Device/backend ownership is uncertain. Do not unwind host buffers
        // that a submitted upload/readback may still borrow, or retry/free any
        // device resource. A top-level catch would run those destructors first.
        std::fprintf(stderr, "sycl_mul_add_error phase=%d\n", phase);
        std::fflush(stderr);
        std::_Exit(2);
    }
    throw BackendError{phase}; // CPU-only injected fixtures retain unwind checks.
}
static void require(bool value, int phase) { if (!value) throw BackendError{phase}; }
static void quiet_log(ggml_log_level, const char *, void *) {}

// Test-local link wrappers observe real backend dispatch without adding state
// or counters to the production archive. Scopes never outlive their stack data.
struct RouteCounts { size_t attempts = 0, fused = 0, mul = 0, add = 0; };
static thread_local RouteCounts * route_sink = nullptr;
struct RouteScope {
    RouteCounts counts;
    RouteCounts * previous = route_sink;
    RouteScope() noexcept { route_sink = &counts; }
    ~RouteScope() noexcept { route_sink = previous; }
    RouteScope(const RouteScope &) = delete;
    RouteScope & operator=(const RouteScope &) = delete;
};
static void require_routes(const RouteCounts & c, bool fused) {
    require(c.attempts == size_t(fused) && c.fused == size_t(fused) &&
            c.mul == size_t(!fused) && c.add == size_t(!fused), 18);
}
#ifdef KUNO_SYCL_FUSION_DIRECT
extern "C" bool kuno_real_mul_add(ggml_backend_sycl_context &, ggml_tensor *, ggml_tensor *)
    asm("__real__Z24ggml_sycl_op_mul_add_f32R25ggml_backend_sycl_contextP11ggml_tensorS2_");
extern "C" bool kuno_wrap_mul_add(ggml_backend_sycl_context &, ggml_tensor *, ggml_tensor *)
    asm("__wrap__Z24ggml_sycl_op_mul_add_f32R25ggml_backend_sycl_contextP11ggml_tensorS2_");
extern "C" bool kuno_wrap_mul_add(ggml_backend_sycl_context & ctx, ggml_tensor * mul, ggml_tensor * add) {
    if (route_sink) ++route_sink->attempts;
    const bool submitted = kuno_real_mul_add(ctx, mul, add);
    if (submitted && route_sink) ++route_sink->fused;
    return submitted;
}
extern "C" void kuno_real_mul(ggml_backend_sycl_context &, ggml_tensor *)
    asm("__real__Z13ggml_sycl_mulR25ggml_backend_sycl_contextP11ggml_tensor");
extern "C" void kuno_wrap_mul(ggml_backend_sycl_context &, ggml_tensor *)
    asm("__wrap__Z13ggml_sycl_mulR25ggml_backend_sycl_contextP11ggml_tensor");
extern "C" void kuno_wrap_mul(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    kuno_real_mul(ctx, dst);
    if (route_sink) ++route_sink->mul;
}
extern "C" void kuno_real_add(ggml_backend_sycl_context &, ggml_tensor *)
    asm("__real__Z13ggml_sycl_addR25ggml_backend_sycl_contextP11ggml_tensor");
extern "C" void kuno_wrap_add(ggml_backend_sycl_context &, ggml_tensor *)
    asm("__wrap__Z13ggml_sycl_addR25ggml_backend_sycl_contextP11ggml_tensor");
extern "C" void kuno_wrap_add(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    kuno_real_add(ctx, dst);
    if (route_sink) ++route_sink->add;
}
#endif
#ifdef KUNO_SYCL_FUSION_FAILURE_TEST
static int injected_backend_phase = 0;
static int synchronization_attempts = 0;
static int initialization_attempts = 0;
static std::vector<int> cleanup_order;
static void record_cleanup(int phase) { cleanup_order.push_back(phase); }
#else
static void record_cleanup(int) {}
#endif

// Preserve the pinned backend's wait() semantics; successful return is not a
// claim that deferred device errors are detected. Any thrown backend call or
// failed submission makes ownership uncertain for every live resource.
template <typename Call>
static decltype(auto) backend_call(int phase, Call && call) {
    if (!backend_safe) uncertain_backend(14);
    try {
#ifdef KUNO_SYCL_FUSION_FAILURE_TEST
        if (phase == 14) ++synchronization_attempts;
        if (phase == 2) ++initialization_attempts;
        if (injected_backend_phase == phase) throw BackendError{phase};
#endif
        return call();
    } catch (...) { uncertain_backend(phase); }
}
static void backend_result(bool successful, int phase) {
    if (!successful) uncertain_backend(phase);
}
static bool synchronize_noexcept(ggml_backend_t backend) noexcept {
    if (!backend_safe) {
        if (device_backend_mode) uncertain_backend(14);
        return false;
    }
    try {
        if (backend) backend_call(14, [&] { ggml_backend_synchronize(backend); });
        return true;
    } catch (...) {
        if (device_backend_mode) uncertain_backend(14);
        return false;
    }
}
struct Backend {
    ggml_backend_t value = nullptr;
    bool release() noexcept {
        if (!value) {
            if (!backend_safe && device_backend_mode) uncertain_backend(16);
            return backend_safe;
        }
        if (!synchronize_noexcept(value)) return false;
        try {
            backend_call(16, [&] { ggml_backend_free(value); }); record_cleanup(4);
            value = nullptr;
            return true;
        } catch (...) {
            if (device_backend_mode) uncertain_backend(16);
            backend_safe = false; return false;
        }
    }
    ~Backend() noexcept { (void)release(); }
};
struct Graph {
    ggml_backend_t backend;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t inputs = nullptr;
    ggml_gallocr_t allocator = nullptr;
    explicit Graph(ggml_backend_t b) : backend(b) {
        require(backend_safe, 14);
        ctx = ggml_init({2 * 1024 * 1024, nullptr, true}); require(ctx, 4);
    }
    ~Graph() noexcept {
        if (!synchronize_noexcept(backend)) return;
        try {
            if (allocator) { backend_call(16, [&] { ggml_gallocr_free(allocator); }); record_cleanup(1); }
            if (inputs) { backend_call(16, [&] { ggml_backend_buffer_free(inputs); }); record_cleanup(2); }
            if (ctx) { ggml_free(ctx); record_cleanup(3); }
        } catch (...) {
            if (device_backend_mode) uncertain_backend(16);
            backend_safe = false;
        }
    }
};

enum class FiniteFamily { bounded = 2, wide = 3, rounding = 4 };
static uint32_t f32_bits(float value) {
    uint32_t bits; std::memcpy(&bits, &value, sizeof(bits)); return bits;
}
static float f32_value(uint32_t bits) {
    float value; std::memcpy(&value, &bits, sizeof(value)); return value;
}
static uint32_t public_mix(uint32_t index) {
    index ^= index >> 16; index *= 0x7feb352dU;
    index ^= index >> 15; index *= 0x846ca68bU;
    return index ^ (index >> 16);
}
static uint32_t finite_bits(uint32_t mixed, unsigned exponent) {
    return (mixed & 0x807fffffU) | (exponent << 23);
}
struct Values {
    static constexpr size_t control_count = 21;
    int columns, rows;
    std::vector<float> activation, gate, residual;
    Values(int c, int r) : columns(c), rows(r), activation(size_t(c) * r), gate(c), residual(size_t(c) * r) {}
    Values(int c, int r, bool controls) : Values(c, r) {
        const float e = std::ldexp(1.f, -23), minimum = std::numeric_limits<float>::min();
        const float tiny = std::numeric_limits<float>::denorm_min(), maximum = std::numeric_limits<float>::max();
        const float inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
        auto from_bits = [](uint32_t bits) { float value; std::memcpy(&value, &bits, sizeof(value)); return value; };
        const float nan_a = from_bits(0x7fc00011), nan_b = from_bits(0xffc00022), nan_c = from_bits(0x7fc00033);
        struct Triple { float a, g, r; };
        const Triple cases[] = {
            {1.f + e, 1.f - e, -1.f}, {1.f + e, 1.f + e, -1.f},
            {maximum, 2.f, -maximum}, {maximum, 2.f, -inf},
            {minimum, .5f, 0.f}, {tiny, 2.f, -tiny}, {tiny, .5f, 0.f},
            {-0.f, 1.f, -0.f}, {0.f, -1.f, 0.f}, {-0.f, -1.f, -0.f},
            {inf, 0.f, 3.f}, {nan, 2.f, 1.f}, {2.f, nan, 1.f}, {2.f, 3.f, nan},
            {minimum, 1.f - e, -minimum}, {1.f, 1.f, e / 2.f}, {-1.f, 1.f, -e / 2.f},
            {nan_a, 1.f, nan_b}, {nan_b, 1.f, nan_a}, {nan_b, nan_a, 1.f}, {nan_a, nan_b, nan_c}
        };
        constexpr size_t count = sizeof(cases) / sizeof(cases[0]);
        static_assert(count == control_count, "Public control-index diagnostic bound");
        for (int col = 0; col < c; ++col) {
            gate[col] = controls ? cases[col % count].g : float(col % 19 - 9) / 32.f;
            for (int row = 0; row < r; ++row) {
                const size_t i = size_t(row) * c + col;
                activation[i] = controls ? cases[col % count].a : float((i * 13 + 7) % 101) / 16.f - 3.f;
                residual[i] = controls ? cases[col % count].r : float((i * 17 + 5) % 97) / 32.f - 1.f;
            }
        }
    }
    Values(int c, int r, FiniteFamily family) : Values(c, r) {
        // Public full-mantissa bit patterns, never a runtime/private seed. The
        // bounded family models TANH gates; wide also exercises the generic
        // eligible operation outside that model's gate range.
        struct Triple { uint32_t a, g, r; };
        static constexpr Triple ties[] = {
            {0x3f800001U, 0x3fc00000U, 0xbfc00000U}, // MUL halfway, odd lower.
            {0x3f800003U, 0x3fc00000U, 0xbfc00000U}, // MUL halfway, even lower.
            {0xbf800001U, 0x3fc00000U, 0x3fc00000U},
            {0xbf800003U, 0x3fc00000U, 0x3fc00000U},
            {0x3f800000U, 0x3f800000U, 0x33800000U}, // ADD halfway, even lower.
            {0x3f800001U, 0x3f800000U, 0x33800000U}, // ADD halfway, odd lower.
            {0xbf800000U, 0x3f800000U, 0xb3800000U},
            {0xbf800001U, 0x3f800000U, 0xb3800000U},
            {0x3dcccccdU, 0x3e4ccccdU, 0xbccccccdU}, // Decimal-derived F32.
            {0x3e99999aU, 0x3f333333U, 0xbe570a3dU},
            {0x3f800001U, 0x3f7fffffU, 0xbf800000U}, // FMA-sensitive.
            {0x3f800001U, 0x3f800001U, 0xbf800000U},
            {0x00800001U, 0x3f000000U, 0x80400000U}, // Subnormal halfway.
            {0x00000003U, 0x3f000000U, 0x80000001U},
            {0x80800001U, 0x3f000000U, 0x00400000U},
            {0x80000003U, 0x3f000000U, 0x00000001U}
        };
        for (int col = 0; col < c; ++col) {
            const auto mixed = public_mix(uint32_t(col) + 0x243f6a88U);
            const auto & tie = ties[size_t(col) % (sizeof(ties) / sizeof(ties[0]))];
            const auto gb = family == FiniteFamily::rounding ? tie.g :
                finite_bits(mixed, family == FiniteFamily::bounded ? 104U + unsigned(col) % 23U : unsigned(col) % 255U);
            gate[col] = f32_value(gb);
            for (int row = 0; row < r; ++row) {
                const size_t i = size_t(row) * c + col;
                const auto am = public_mix(uint32_t(i) + 0x85a308d3U);
                const auto rm = public_mix(uint32_t(i) + 0x13198a2eU);
                uint32_t ab, rb;
                if (family == FiniteFamily::rounding) {
                    if (row % 4 == 0) { ab = tie.a; rb = tie.r; }
                    else {
                        ab = finite_bits(am, 120U + uint32_t(i) % 16U);
                        const auto product = kuno_binary32::mul(ab, gb);
                        auto magnitude = product & 0x7fffffffU;
                        if (row % 4 == 2) ++magnitude;
                        if (row % 4 == 3 && magnitude) --magnitude;
                        rb = ((product ^ 0x80000000U) & 0x80000000U) | magnitude;
                    }
                } else {
                    ab = finite_bits(am, family == FiniteFamily::bounded ? 112U + uint32_t(i) % 32U : (uint32_t(i) * 37U + unsigned(col)) % 255U);
                    rb = finite_bits(rm, family == FiniteFamily::bounded ? 112U + uint32_t(i * 7) % 32U : (uint32_t(i) * 61U + unsigned(col) * 17U) % 255U);
                }
                require(kuno_binary32::finite(ab) && kuno_binary32::finite(gb) && kuno_binary32::finite(rb), 17);
                activation[i] = f32_value(ab); residual[i] = f32_value(rb);
            }
        }
    }
};

static std::vector<float> reference(const Values & v, bool product_first) {
    std::vector<float> output(v.activation.size());
    for (size_t i = 0; i < output.size(); ++i) {
        // A product of two finite F32 operands is exact in double. Round at
        // each former storage boundary; never compute a fused expression.
        // Expanded finite inputs cross-check this strict-host path against
        // the independent integer reference instead of assuming its accuracy.
        volatile float product = float(double(v.activation[i]) * double(v.gate[i % v.columns]));
        const long double left = product_first ? product : v.residual[i];
        const long double right = product_first ? v.residual[i] : product;
        volatile float sum = float(left + right);
        output[i] = sum;
    }
    return output;
}
static std::vector<float> integer_reference(const Values & v, bool product_first) {
    std::vector<float> output(v.activation.size());
    for (size_t i = 0; i < output.size(); ++i) {
        const auto product = kuno_binary32::mul(f32_bits(v.activation[i]), f32_bits(v.gate[i % v.columns]));
        const auto residual = f32_bits(v.residual[i]);
        output[i] = f32_value(product_first ? kuno_binary32::add(product, residual) : kuno_binary32::add(residual, product));
    }
    return output;
}

struct HostControlChecks {
    bool zero, positive, fma_distinct, nearest, mxcsr_supported, ftz, daz;
};
static HostControlChecks host_control_checks(const Values & values, float expected) {
    HostControlChecks result{expected == 0.f, !std::signbit(expected),
        std::fma(values.activation[0], values.gate[0], values.residual[0]) != expected,
        std::fegetround() == FE_TONEAREST, false, false, false};
#ifdef __SSE__
    const auto mxcsr = _mm_getcsr();
    result.mxcsr_supported = true;
    result.ftz = (mxcsr & (1U << 15)) != 0;
    result.daz = (mxcsr & (1U << 6)) != 0;
#endif
    return result;
}
static void report_host_controls(const HostControlChecks & c, int stage) {
    std::fprintf(stderr, "sycl_mul_add_host_control stage=%d zero=%d positive=%d fma_distinct=%d nearest=%d mxcsr_supported=%d ftz=%d daz=%d\n",
        stage, c.zero, c.positive, c.fma_distinct, c.nearest, c.mxcsr_supported, c.ftz, c.daz);
}

struct MismatchCounts {
    size_t total = 0, finite = 0, subnormal = 0, signed_zero = 0, zero_class = 0;
    size_t nan_both = 0, nan_payload = 0, nan_sign = 0, nan_class = 0, infinity = 0;
};
static MismatchCounts mismatch_counts(const std::vector<float> & a, const std::vector<float> & b) {
    MismatchCounts counts;
    for (size_t i = 0; i < a.size(); ++i) {
        const auto left = f32_bits(a[i]), right = f32_bits(b[i]);
        if (left == right) continue;
        ++counts.total;
        auto magnitude = [](uint32_t bits) { return bits & 0x7fffffffU; };
        auto is_nan = [&](uint32_t bits) { return magnitude(bits) > 0x7f800000U; };
        auto is_subnormal = [&](uint32_t bits) { return magnitude(bits) != 0 && magnitude(bits) < 0x00800000U; };
        const bool left_nan = is_nan(left), right_nan = is_nan(right);
        if (left_nan || right_nan) {
            if (left_nan && right_nan) {
                ++counts.nan_both;
                counts.nan_payload += (left & 0x007fffffU) != (right & 0x007fffffU);
                counts.nan_sign += (left >> 31) != (right >> 31);
            } else ++counts.nan_class;
        } else if (magnitude(left) == 0x7f800000U || magnitude(right) == 0x7f800000U) ++counts.infinity;
        else {
            counts.zero_class += (magnitude(left) == 0) != (magnitude(right) == 0);
            if (magnitude(left) == 0 && magnitude(right) == 0) ++counts.signed_zero;
            else if (is_subnormal(left) || is_subnormal(right)) ++counts.subnormal;
            else ++counts.finite;
        }
    }
    return counts;
}
static std::array<size_t, Values::control_count> control_mismatch_counts(
        const std::vector<float> & a, const std::vector<float> & b, int columns) {
    require(a.size() == b.size() && columns > 0, 17);
    std::array<size_t, Values::control_count> counts{};
    for (size_t i = 0; i < a.size(); ++i) if (f32_bits(a[i]) != f32_bits(b[i])) {
        ++counts[(i % size_t(columns)) % Values::control_count];
    }
    return counts;
}
static void exact(const std::vector<float> & a, const std::vector<float> & b, int phase, int control_columns = 0) {
    require(a.size() == b.size(), phase);
    if (std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0) return;
    const auto c = mismatch_counts(a, b);
    // Fixed categories/counts only. Payload/sign and zero-class counts are
    // subsets; the six primary categories partition every bit mismatch.
    std::fprintf(stderr, "sycl_mul_add_mismatch phase=%d total=%zu finite=%zu subnormal=%zu signed_zero=%zu zero_class=%zu nan_both=%zu nan_payload=%zu nan_sign=%zu nan_class=%zu infinity=%zu\n",
        phase, c.total, c.finite, c.subnormal, c.signed_zero, c.zero_class,
        c.nan_both, c.nan_payload, c.nan_sign, c.nan_class, c.infinity);
    if (control_columns > 0) {
        const auto counts = control_mismatch_counts(a, b, control_columns);
        for (size_t index = 0; index < counts.size(); ++index) if (counts[index]) {
            std::fprintf(stderr, "sycl_mul_add_control_mismatch index=%zu count=%zu\n", index, counts[index]);
        }
    }
    throw BackendError{phase}; // Diagnostics never weaken exact equality.
}
static void unchanged(ggml_tensor * tensor, const std::vector<float> & expected, int phase) {
    std::vector<float> actual(expected.size());
    backend_call(phase, [&] { ggml_backend_tensor_get(tensor, actual.data(), 0, actual.size() * sizeof(float)); });
    exact(actual, expected, phase);
}

static std::vector<float> run(ggml_backend_t backend, const Values & v, bool product_first,
                              bool candidate_graph, bool device, int alias) {
    const bool eligible = candidate_graph && !product_first;
    const size_t bytes = v.activation.size() * sizeof(float);
    if (device) {
        size_t free = 0, total = 0;
        backend_call(3, [&] { ggml_backend_dev_memory(ggml_backend_get_device(backend), &free, &total); });
        require(bytes <= (size_t(1024) * 1024 * 1024) / 4, 3);
        const size_t needed = bytes * 4 + v.gate.size() * sizeof(float) + 64 * 1024 * 1024;
        require(free > needed, 3);
    }
    Graph resources(backend);
    auto ctx = resources.ctx;
    auto a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, v.columns, v.rows);
    auto g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, v.columns);
    auto r = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, v.columns, v.rows);
    for (auto t : {a, g, r}) ggml_set_input(t);
    resources.inputs = backend_call(4, [&] { return ggml_backend_alloc_ctx_tensors(ctx, backend); });
    backend_result(resources.inputs, 4);
    auto upload = [&]() {
        backend_call(4, [&] { ggml_backend_tensor_set(a, v.activation.data(), 0, bytes); });
        backend_call(4, [&] { ggml_backend_tensor_set(g, v.gate.data(), 0, v.gate.size() * sizeof(float)); });
        backend_call(4, [&] { ggml_backend_tensor_set(r, v.residual.data(), 0, bytes); });
    };
    upload();
    auto mul = ggml_mul(ctx, a, g);
    auto add = product_first ? ggml_add(ctx, mul, r) : ggml_add(ctx, r, mul);
    if (!candidate_graph) ggml_set_output(mul); // forces the materialized baseline
    ggml_set_output(add);
    auto graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, add);
    int index = -1;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto t = ggml_graph_node(graph, i);
        require(backend_call(5, [&] { return ggml_backend_supports_op(backend, t); }), 5);
        if (t == mul) index = i;
    }
    require(index >= 0 && ggml_can_fuse(graph, index, {GGML_OP_MUL, GGML_OP_ADD}) == candidate_graph, 5);
    require(ggml_sycl_mul_add_graph_safe(graph, index) == eligible, 5);
    resources.allocator = backend_call(4, [&] { return ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)); });
    backend_result(resources.allocator, 4);
    size_t workspace_size = 0;
    backend_call(4, [&] { ggml_gallocr_reserve_n_size(resources.allocator, graph, nullptr, nullptr, &workspace_size); });
    require(workspace_size <= size_t(1024) * 1024 * 1024, 3);
    backend_result(backend_call(4, [&] { return ggml_gallocr_reserve(resources.allocator, graph); }), 4);
    backend_result(backend_call(4, [&] { return ggml_gallocr_alloc_graph(resources.allocator, graph); }), 4);
    std::vector<float> result(v.activation.size());
    auto compute_graph = [&] {
        RouteScope routes;
        backend_result(backend_call(9, [&] { return ggml_backend_graph_compute(backend, graph); }) == GGML_STATUS_SUCCESS, 9);
#ifdef KUNO_SYCL_FUSION_DIRECT
        if (device) require_routes(routes.counts, eligible);
#endif
    };
#ifdef KUNO_SYCL_FUSION_DIRECT
    if (device) require(ggml_sycl_can_fuse(graph, index, {GGML_OP_MUL, GGML_OP_ADD}) == eligible, 6);
    if (device && eligible) {
        backend_result(backend_call(6, [&] { return ggml_backend_is_sycl(backend); }) && backend->context, 6);
        require(ggml_sycl_can_fuse(graph, index, {GGML_OP_MUL, GGML_OP_ADD}), 6);
        auto original = add->data;
        if (alias == 1) add->data = a->data;
        if (alias == 2) add->data = r->data;
        require(ggml_sycl_mul_add_memory_safe(mul, add), 6);
        auto context = static_cast<ggml_backend_sycl_context *>(backend->context);
        {
            RouteScope routes;
            backend_result(backend_call(7, [&] { return ggml_sycl_op_mul_add_f32(*context, mul, add); }), 7);
            require_routes(routes.counts, true);
        }
        backend_call(14, [&] { ggml_backend_synchronize(backend); });
        backend_call(8, [&] { ggml_backend_tensor_get(add, result.data(), 0, bytes); });
        unchanged(g, v.gate, 8);
        if (alias != 1) unchanged(a, v.activation, 8);
        if (alias != 2) unchanged(r, v.residual, 8);
        add->data = original;
        require(ggml_sycl_mul_add_memory_safe(mul, add) &&
                ggml_sycl_can_fuse(graph, index, {GGML_OP_MUL, GGML_OP_ADD}), 6);
        upload();
        compute_graph();
        backend_call(14, [&] { ggml_backend_synchronize(backend); });
        std::vector<float> graph_result(result.size());
        backend_call(8, [&] { ggml_backend_tensor_get(add, graph_result.data(), 0, bytes); });
        exact(graph_result, result, 10);
    } else
#else
    (void)alias;
#endif
    {
        // Rejected reverse-order candidates remain unmarked/generic-fusable.
        // They execute both legacy operations; do not force the fused helper.
        auto original = add->data;
        if (alias == 1) add->data = a->data;
        if (alias == 2) add->data = r->data;
        compute_graph();
        backend_call(14, [&] { ggml_backend_synchronize(backend); });
        backend_call(8, [&] { ggml_backend_tensor_get(add, result.data(), 0, bytes); });
        unchanged(g, v.gate, 8);
        if (alias != 1) unchanged(a, v.activation, 8);
        if (alias != 2) unchanged(r, v.residual, 8);
        add->data = original;
        if (alias != 0) upload();
    }
    unchanged(a, v.activation, 8);
    unchanged(g, v.gate, 8);
    unchanged(r, v.residual, 8);
    return result;
}

int main(int argc, char ** argv) {
    try {
        rlimit limit{0, 0};
        require(setrlimit(RLIMIT_CORE, &limit) == 0 && std::fesetround(FE_TONEAREST) == 0, 1);
        const bool device = argc != 1;
        std::string pci;
        if (device) {
#ifdef KUNO_SYCL_FUSION_DIRECT
            require(argc == 4 && std::strcmp(argv[1], "--device") == 0 && std::strcmp(argv[2], "SYCL") == 0, 1);
            pci = ggml_sycl_canonical_pci_address(argv[3]);
            require(!pci.empty(), 1);
#else
            throw BackendError{1};
#endif
        }
        // A public control measured before backend initialization distinguishes
        // the host compiler/library assumptions from subsequent device work.
        const Values preinit_values(23, 1, true);
        const auto preinit_expected = reference(preinit_values, false);
        report_host_controls(host_control_checks(preinit_values, preinit_expected[0]), 0);
        ggml_log_set(quiet_log, nullptr);
        Backend backend;
        if (!device) {
            backend.value = backend_call(2, [&] { return ggml_backend_cpu_init(); });
            backend_result(backend.value, 2);
            backend_call(2, [&] { ggml_backend_cpu_set_n_threads(backend.value, 1); });
        } else {
#ifdef KUNO_SYCL_FUSION_DIRECT
            device_backend_mode = true; // Before any backend discovery/init call.
            backend_call(2, [&] { ggml_backend_load_all(); });
            ggml_backend_dev_t selected = nullptr;
            for (size_t i = 0; i < backend_call(2, [&] { return ggml_backend_dev_count(); }); ++i) {
                auto d = backend_call(2, [&] { return ggml_backend_dev_get(i); });
                ggml_backend_dev_props p{};
                backend_call(2, [&] { ggml_backend_dev_get_props(d, &p); });
                if (p.type == GGML_BACKEND_DEVICE_TYPE_GPU && p.name && std::strncmp(p.name, "SYCL", 4) == 0 &&
                    p.device_id && ggml_sycl_canonical_pci_address(p.device_id) == pci) {
                    require(!selected, 2);
                    selected = d;
                }
            }
            require(selected, 2);
            backend.value = backend_call(2, [&] { return ggml_backend_dev_init(selected, nullptr); });
            backend_result(backend.value, 2);
#else
            throw BackendError{1};
#endif
        }
        int cases = 0;
        auto test = [&](int columns, int rows, bool controls, bool first, int alias) {
            Values values(columns, rows, controls);
            auto expected = reference(values, first);
            auto legacy = run(backend.value, values, first, false, device, 0);
            auto candidate = run(backend.value, values, first, true, device, alias);
            exact(legacy, candidate, 11, controls ? columns : 0);
            if (!device) {
                for (size_t i = 0; i < expected.size(); ++i) {
                    if (std::isnan(expected[i])) require(std::isnan(legacy[i]), 12);
                    else require(std::memcmp(&expected[i], &legacy[i], sizeof(float)) == 0, 12);
                }
            }
            if (controls) {
                const auto checks = host_control_checks(values, expected[0]);
                if (!checks.zero || !checks.positive || !checks.fma_distinct) report_host_controls(checks, 1);
                require(checks.zero && checks.positive && checks.fma_distinct, 12);
            }
            std::printf("sycl_mul_add_arithmetic case=%d columns=%d rows=%d controls=%d product_first=%d alias=%d pass=1\n",
                        ++cases, columns, rows, controls, first, alias);
            std::fflush(stdout);
        };
        for (bool first : {false, true}) {
            for (bool controls : {false, true}) {
                for (int alias = 0; alias <= (device ? 2 : 0); ++alias) test(23, 65, controls, first, alias);
                test(4096, 9, controls, first, 0);
            }
            if (device) for (int rows : {4056, 8214}) test(4096, rows, false, first, 0);
        }
        auto finite_test = [&](int columns, int rows, FiniteFamily family, bool first, int alias) {
            Values values(columns, rows, family);
            std::vector<float> expected;
            if (!device) {
                expected = integer_reference(values, first);
                exact(expected, reference(values, first), 19);
            }
            auto legacy = run(backend.value, values, first, false, device, 0);
            auto candidate = run(backend.value, values, first, true, device, alias);
            exact(legacy, candidate, 11);
            if (!device) exact(expected, legacy, 12);
            std::printf("sycl_mul_add_arithmetic case=%d columns=%d rows=%d controls=0 product_first=%d alias=%d family=%d pass=1\n",
                        ++cases, columns, rows, first, alias, int(family));
            std::fflush(stdout);
        };
        for (auto family : {FiniteFamily::bounded, FiniteFamily::wide, FiniteFamily::rounding}) {
            for (bool first : {false, true}) {
                for (int alias = 0; alias <= (device ? 2 : 0); ++alias) finite_test(23, 65, family, first, alias);
                finite_test(4096, 9, family, first, 0);
            }
        }
        if (device) for (bool first : {false, true}) for (int rows : {4056, 8214}) {
            finite_test(4096, rows, FiniteFamily::bounded, first, 0);
        }
        backend_result(backend.release(), 16);
        std::printf("sycl_mul_add_arithmetic passed=%d device=%d\n", cases, device);
        return 0;
    } catch (const BackendError & error) {
        std::fprintf(stderr, "sycl_mul_add_error phase=%d\n", error.phase);
    } catch (...) {
        std::fprintf(stderr, "sycl_mul_add_error phase=13\n");
    }
    return 2;
}
