// Content-free CPU regression for graph-local Qwen21 positional-matrix sharing.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <sys/resource.h>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "core/segment_graph_bindings.h"
#include "model/diffusion/qwen_image_2_1.hpp"

struct BackendError { int phase; };
static bool backend_uncertain = false;

// Preserve the existing wait() contract. A thrown backend call makes resource
// ownership uncertain; do not retry a wait or release it during unwinding.
template <typename Call>
static decltype(auto) backend_call(Call&& call) {
    if (backend_uncertain) throw BackendError{21};
    try { return call(); }
    catch (...) { backend_uncertain = true; throw; }
}

#ifdef KUNO_ROPE_FAILURE_TEST
static std::vector<int> cleanup_order;
static int injected_output = 0;
static float injected_value = 0;
static bool injected_entry_exception = false;
static void record_cleanup(int event) { cleanup_order.push_back(event); }
#else
static void record_cleanup(int) {}
#endif

static size_t pe_copies(ggml_cgraph* graph, ggml_tensor* raw) {
    size_t count = 0;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto node = ggml_graph_node(graph, i);
        if (node->op == GGML_OP_CONT && node->src[0]->op == GGML_OP_PERMUTE && node->src[0]->src[0] == raw) ++count;
    }
    return count;
}

static std::vector<float> values(ggml_tensor* tensor) {
    std::vector<float> result(ggml_nelements(tensor));
    backend_call([&] { ggml_backend_tensor_get(tensor, result.data(), 0, ggml_nbytes(tensor)); });
    return result;
}

static void same(const std::vector<float>& a, const std::vector<float>& b) {
    assert(a.size() == b.size());
    assert(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

static float rounded_product(float a, float b) {
    volatile float result = a * b;
    return result;
}

static std::vector<float> oracle(const std::vector<float>& x, const std::vector<float>& pe,
                                 int dim, int heads, int rows, int batch, bool interleaved) {
    std::vector<float> result(x.size());
    for (int n = 0; n < batch; ++n) for (int h = 0; h < heads; ++h) for (int t = 0; t < rows; ++t) for (int j = 0; j < dim / 2; ++j) {
        const int d0 = interleaved ? 2 * j : j;
        const int d1 = interleaved ? 2 * j + 1 : j + dim / 2;
        const size_t in = size_t(dim) * (h + heads * (t + rows * n));
        const size_t out = size_t(dim) * (t + rows * (h + heads * n));
        const size_t p = size_t(t * (dim / 2) + j) * 4;
        result[out + d0] = rounded_product(x[in + d0], pe[p]) + rounded_product(x[in + d1], pe[p + 1]);
        result[out + d1] = rounded_product(x[in + d0], pe[p + 2]) + rounded_product(x[in + d1], pe[p + 3]);
    }
    return result;
}

static void arithmetic_case(int dim, int rows, int heads, int batch, bool interleaved,
                            const std::vector<float>& pd) {
    auto ctx = ggml_init({16 * 1024 * 1024, nullptr, false});
    assert(ctx);
    auto x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim, heads, rows, batch);
    auto pe = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 2, dim / 2, rows);
    std::vector<float> xd(ggml_nelements(x));
    for (size_t i = 0; i < xd.size(); ++i) xd[i] = float(int(i * 17 % 127) - 63) / 32.f;
    std::memcpy(x->data, xd.data(), ggml_nbytes(x));
    std::memcpy(pe->data, pd.data(), ggml_nbytes(pe));
    auto original = Rope::apply_rope(ctx, x, pe, interleaved);
    auto prepared = Rope::prepare_rope_pe(ctx, pe);
    auto shared_q = Rope::apply_rope_with_prepared_pe(ctx, x, prepared, interleaved);
    auto shared_k = Rope::apply_rope_with_prepared_pe(ctx, x, prepared, interleaved);
    auto graph = ggml_new_graph(ctx);
    for (auto output : {original, shared_q, shared_k}) ggml_build_forward_expand(graph, output);
    assert(pe_copies(graph, pe) == 2); // One original and one shared preparation.
    assert(ggml_graph_compute_with_ctx(ctx, graph, 1) == GGML_STATUS_SUCCESS);
    auto expected = oracle(xd, pd, dim, heads, rows, batch, interleaved);
    same(expected, std::vector<float>(static_cast<float*>(original->data), static_cast<float*>(original->data) + expected.size()));
    assert(std::memcmp(original->data, shared_q->data, ggml_nbytes(original)) == 0);
    assert(std::memcmp(original->data, shared_k->data, ggml_nbytes(original)) == 0);
    assert(std::memcmp(pe->data, pd.data(), ggml_nbytes(pe)) == 0);
    ggml_free(ctx);
}

static Rope::Embedding embedding(const Qwen::QwenImage21Layout& layout, bool suffix,
                                 bool circular_x, bool circular_y) {
    Rope::Embedding result;
    const size_t offset = suffix ? layout.prefix_length : 0;
    result.ids.assign(layout.positions.begin() + offset, layout.positions.end());
    result.positions.token_count = result.ids.size();
    for (auto region : layout.rope_layout.images) if (region.begin >= offset) {
        region.begin -= offset;
        result.positions.images.push_back(region);
    }
    result.values = Rope::embed_nd(result.ids, 1, 10000.f, {16, 56, 56}, result.layout, &result.frequencies);
    Rope::apply_circular(result, circular_x, circular_y);
    return result;
}

static void sharing_topology() {
    auto ctx = ggml_init({16 * 1024 * 1024, nullptr, false});
    auto x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 4, 2, 2, 1);
    auto pe = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 2, 2, 2);
    auto old_graph = ggml_new_graph_custom(ctx, 4096, false);
    auto new_graph = ggml_new_graph_custom(ctx, 4096, false);
    auto prepared = Rope::prepare_rope_pe(ctx, pe);
    for (int i = 0; i < 64; ++i) {
        ggml_build_forward_expand(old_graph, Rope::apply_rope(ctx, x, pe));
        ggml_build_forward_expand(new_graph, Rope::apply_rope_with_prepared_pe(ctx, x, prepared));
    }
    assert(pe_copies(old_graph, pe) == 64);
    assert(pe_copies(new_graph, pe) == 1);
    size_t coefficient_views = 0;
    for (int i = 0; i < ggml_graph_n_nodes(new_graph); ++i) {
        auto node = ggml_graph_node(new_graph, i);
        if (node->op == GGML_OP_VIEW && node->src[0] == prepared) ++coefficient_views;
    }
    assert(coefficient_views == 128);
    ggml_free(ctx);
}

static void check_plan(const sd::ggml_graph_cut::Plan& plan, ggml_cgraph* graph,
                       ggml_tensor* prepared, int layers) {
    assert(plan.valid && plan.has_cuts);
    int index = -1;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) if (ggml_graph_node(graph, i) == prepared) index = i;
    assert(index >= 0);
    int internal = 0, consumers = 0;
    for (const auto& segment : plan.segments) {
        if (std::find(segment.internal_node_indices.begin(), segment.internal_node_indices.end(), index) != segment.internal_node_indices.end()) {
            ++internal;
            assert(segment.group_name == "qwen_image_2_1.prelude");
        }
        for (const auto& input : segment.input_refs) if (input.node_index == index) {
            assert(input.type == sd::ggml_graph_cut::Segment::INPUT_PREVIOUS_CUT);
            ++consumers;
        }
    }
    assert(internal == 1 && consumers == layers);
}

static void model_topology(ggml_backend_t cpu, Qwen::QwenImage21PrefixCache::Mode mode) {
    // Small real Qwen21 graph covers the model's shared operand wiring, references,
    // negative coordinates and separate prefix STORE/REUSE tensor shapes.
    auto ctx = ggml_init({16 * 1024 * 1024, nullptr, true});
    auto one = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_name(one, "ggml_runner_build_in_tensor:one");
    Qwen::QwenImage21Config config;
    config.in_channels = config.out_channels = 4;
    config.hidden_size = config.context_dim = 16;
    config.head_dim = 8;
    config.intermediate_size = 24;
    config.num_layers = 3;
    Qwen::QwenImage21Model model(config);
    model.init(ctx);
    sd::Tensor<int32_t> slots({5}, {0, 1, 1, 0, 0});
    auto layout = Qwen::QwenImage21Layout::build(5, slots, {{2, 4}, {3, 3}});
    const bool reuse = mode == Qwen::QwenImage21PrefixCache::Mode::REUSE;
    auto pe = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 2, 4, reuse ? 9 : layout.positions.size());
    auto x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 3, 4, 1);
    auto ref = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 4, 2, 4, 1);
    auto context = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 5, 1);
    auto time = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    auto pk = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 8, layout.prefix_length, 2, 1);
    auto pv = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 8, 2, layout.prefix_length, 1);
    GGMLRunnerContext runner;
    runner.backend = cpu;
    runner.ggml_ctx = ctx;
    runner.get_cache_tensor = [&](const std::string& name) { return name.back() == 'k' ? pk : pv; };
    std::vector<ggml_tensor*> persisted;
    runner.cache_tensor = [&](const std::string&, ggml_tensor* tensor) { persisted.push_back(tensor); };
    Qwen::QwenImage21PrefixCache cache;
    cache.mode = mode;
    cache.name = "synthetic";
    cache.prefix_length = layout.prefix_length;
    std::vector<ggml_tensor*> masks(layout.segments.size(), nullptr);
    auto output = model.forward(&runner, x, time, reuse ? nullptr : context, {ref}, pe, layout, masks, cache);
    auto graph = ggml_new_graph_custom(ctx, 4096, false);
    for (auto tensor : persisted) ggml_build_forward_expand(graph, tensor);
    ggml_build_forward_expand(graph, output);
    assert(pe_copies(graph, pe) == 1);
    ggml_tensor* prepared = nullptr;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto node = ggml_graph_node(graph, i);
        if (node->op == GGML_OP_CONT && node->src[0]->op == GGML_OP_PERMUTE && node->src[0]->src[0] == pe) prepared = node;
    }
    auto plan = sd::ggml_graph_cut::build_plan(cpu, graph, {}, nullptr);
    check_plan(plan, graph, prepared, config.num_layers);
    ggml_free(ctx);
}

static void segmented_case(ggml_backend_t cpu, int recipe) {
    auto ctx = ggml_init({16 * 1024 * 1024, nullptr, true});
    auto x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 8, 2, 3, 1);
    auto pe = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 2, 4, 3);
    auto inputs = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    assert(inputs);
    std::vector<float> xd(ggml_nelements(x)), pd(ggml_nelements(pe));
    for (size_t i = 0; i < xd.size(); ++i) xd[i] = float(int(i % 23) - 11) / 8.f;
    for (size_t i = 0; i < pd.size(); ++i) pd[i] = float(int((i + recipe) % 17) - 8) / 16.f;
    ggml_backend_tensor_set(x, xd.data(), 0, ggml_nbytes(x));
    ggml_backend_tensor_set(pe, pd.data(), 0, ggml_nbytes(pe));
    auto prepared = Rope::prepare_rope_pe(ctx, pe);
    sd::ggml_graph_cut::mark_graph_cut(prepared, "qwen_image_2_1.prelude", "rope_pe");
    auto joint = ggml_cont(ctx, x);
    sd::ggml_graph_cut::mark_graph_cut(joint, "qwen_image_2_1.prelude", "joint");
    for (int layer = 0; layer < 2; ++layer) {
        auto q = Rope::apply_rope_with_prepared_pe(ctx, joint, prepared);
        auto k = Rope::apply_rope_with_prepared_pe(ctx, joint, prepared);
        auto sum = ggml_add(ctx, q, k);
        joint = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, sum, 8, 3, 2, 1), 0, 2, 1, 3));
        sd::ggml_graph_cut::mark_graph_cut(joint, "layer." + std::to_string(layer), "joint");
    }
    auto graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, joint);
    auto plan = sd::ggml_graph_cut::build_plan(cpu, graph, {}, nullptr);
    check_plan(plan, graph, prepared, 2);
    auto allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu));
    assert(ggml_gallocr_alloc_graph(allocr, graph));
    assert(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS);
    const auto expected = values(joint);
    const auto prepared_values = values(prepared);
    sd::GraphCutTensorCache cache(cpu);
    {
        sd::SegmentGraphBindings bindings(cache, plan, graph);
        for (const auto& segment : plan.segments) {
            cache.prune(segment.live_cut_names);
            bindings.reset(segment);
            assert(bindings.bind_cached_inputs(segment, "synthetic_rope"));
            ggml_context* segment_ctx = nullptr;
            auto segment_graph = sd::ggml_graph_cut::build_segment_graph(graph, segment, &segment_ctx);
            auto segment_allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu));
            assert(ggml_gallocr_alloc_graph(segment_allocr, segment_graph));
            assert(ggml_backend_graph_compute(cpu, segment_graph) == GGML_STATUS_SUCCESS);
            assert(cache.capture(graph, segment, "synthetic_rope") == GGML_STATUS_SUCCESS);
            if (segment.group_name == "qwen_image_2_1.prelude") same(values(cache.get(prepared->name)), prepared_values);
            if (&segment == &plan.segments.back()) same(values(joint), expected);
            cache.prune(segment.future_cut_names);
            bindings.restore();
            ggml_gallocr_free(segment_allocr);
            ggml_free(segment_ctx);
        }
    }
    cache.clear();
    assert(cache.get(prepared->name) == nullptr);
    assert(cache.resident_bytes(ggml_backend_get_device(cpu)) == 0);
    same(values(pe), pd);
    ggml_gallocr_free(allocr);
    ggml_backend_buffer_free(inputs);
    ggml_free(ctx);
}

struct BackendOutputs {
    std::vector<float> q, k;
};

static void backend_same(const std::vector<float>& a, const std::vector<float>& b, int phase) {
    if (a.size() != b.size() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) != 0) {
        throw BackendError{phase};
    }
}

static void backend_finite(const std::vector<float>& values, int phase) {
    for (float value : values) if (!std::isfinite(value)) throw BackendError{phase};
}

static bool backend_sync_noexcept(ggml_backend_t backend) noexcept {
    if (backend_uncertain) return false;
    try {
        backend_call([&] { ggml_backend_synchronize(backend); });
        return true;
    } catch (...) {
        return false;
    }
}

static void backend_free_noexcept(ggml_backend_t backend) noexcept {
    if (!backend) return;
    if (!backend_sync_noexcept(backend)) return;
    try {
        ggml_backend_free(backend);
        record_cleanup(4);
    } catch (...) {
        backend_uncertain = true;
    }
}

struct BackendResources {
    ggml_backend_t backend;
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t inputs = nullptr;
    ggml_gallocr_t allocator = nullptr;
    explicit BackendResources(ggml_backend_t value) : backend(value) {
        if (backend_uncertain) throw BackendError{21};
    }
    BackendResources(const BackendResources&) = delete;
    BackendResources& operator=(const BackendResources&) = delete;
    ~BackendResources() noexcept {
        if (!backend_sync_noexcept(backend)) return;
        try {
            if (allocator) { ggml_gallocr_free(allocator); record_cleanup(1); }
            if (inputs) { ggml_backend_buffer_free(inputs); record_cleanup(2); }
            if (ctx) { ggml_free(ctx); record_cleanup(3); }
        } catch (...) { backend_uncertain = true; }
    }
};

static BackendOutputs backend_variant(ggml_backend_t backend, int dim, int rows,
                                      int heads, int batch, bool interleaved,
                                      const std::vector<float>& qd, const std::vector<float>& kd,
                                      const std::vector<float>& pd, bool shared) {
    // Each variant owns a fresh context: freed gallocr bindings must never be
    // reused as if their dangling data pointers were external allocations.
    BackendResources resources(backend);
    resources.ctx = ggml_init({16 * 1024 * 1024, nullptr, true});
    if (!resources.ctx) throw BackendError{1};
    auto ctx = resources.ctx;
    auto q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim, heads, rows, batch);
    auto k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim, heads, rows, batch);
    auto pe = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 2, dim / 2, rows);
    const size_t input_bytes = ggml_nbytes(q) + ggml_nbytes(k) + ggml_nbytes(pe);
    constexpr size_t slack = 64 * 1024 * 1024;
    constexpr size_t workspace_cap = 2ull * 1024 * 1024 * 1024;
    ggml_backend_dev_props props{};
    ggml_backend_dev_get_props(ggml_backend_get_device(backend), &props);
    const bool gpu = props.type == GGML_BACKEND_DEVICE_TYPE_GPU;
    if (gpu && (props.memory_free < slack || input_bytes > props.memory_free - slack)) {
        throw BackendError{2};
    }
    // Allocate only inputs before constructing the compute intermediates.
    resources.inputs = backend_call([&] { return ggml_backend_alloc_ctx_tensors(ctx, backend); });
    if (!resources.inputs) throw BackendError{3};
    backend_call([&] { ggml_backend_tensor_set(q, qd.data(), 0, ggml_nbytes(q)); });
    backend_call([&] { ggml_backend_tensor_set(k, kd.data(), 0, ggml_nbytes(k)); });
    backend_call([&] { ggml_backend_tensor_set(pe, pd.data(), 0, ggml_nbytes(pe)); });
    auto prepared = shared ? Rope::prepare_rope_pe(ctx, pe) : nullptr;
    auto qo = shared ? Rope::apply_rope_with_prepared_pe(ctx, q, prepared, interleaved)
                     : Rope::apply_rope(ctx, q, pe, interleaved);
    auto ko = shared ? Rope::apply_rope_with_prepared_pe(ctx, k, prepared, interleaved)
                     : Rope::apply_rope(ctx, k, pe, interleaved);
    // Gallocr may recycle earlier outputs otherwise; CPU allocated contexts
    // would mask that bug. Retain prepared PE too for the immutability check.
    ggml_set_output(qo);
    ggml_set_output(ko);
    if (prepared) ggml_set_output(prepared);
    auto graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, qo);
    ggml_build_forward_expand(graph, ko);
    if (pe_copies(graph, pe) != (shared ? 1 : 2)) throw BackendError{9};
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        if (!ggml_backend_supports_op(backend, ggml_graph_node(graph, i))) {
            throw BackendError{4};
        }
    }
    resources.allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    size_t workspace_bytes = 0;
    if (!resources.allocator) throw BackendError{5};
    backend_call([&] { ggml_gallocr_reserve_n_size(resources.allocator, graph, nullptr, nullptr, &workspace_bytes); });
    ggml_backend_dev_get_props(ggml_backend_get_device(backend), &props);
    if (workspace_bytes > workspace_cap ||
        (gpu && (props.memory_free < slack || workspace_bytes > props.memory_free - slack))) {
        throw BackendError{6};
    }
    // The pinned size-only reserve populates the plan but leaves its backing
    // buffers null; explicitly allocate them before alloc_graph reuses the plan.
    if (!backend_call([&] { return ggml_gallocr_reserve(resources.allocator, graph); }) ||
        !backend_call([&] { return ggml_gallocr_alloc_graph(resources.allocator, graph); })) {
        throw BackendError{7};
    }
    // graph_compute synchronizes internally, so wrap that call as well.
    auto status = backend_call([&] { return ggml_backend_graph_compute(backend, graph); });
    backend_call([&] { ggml_backend_synchronize(backend); });
    if (status != GGML_STATUS_SUCCESS) throw BackendError{8};
    backend_same(values(pe), pd, 10);
    backend_same(values(q), qd, 11);
    backend_same(values(k), kd, 12);
    if (prepared) {
        // Independent physical permutation: raw [2,2,D/2,L] -> [2,D/2,L,2].
        std::vector<float> expected(pd.size());
        for (int row = 0; row < rows; ++row) for (int pair = 0; pair < dim / 2; ++pair) {
            for (int component = 0; component < 2; ++component) for (int result = 0; result < 2; ++result) {
                expected[result + 2 * (pair + (dim / 2) * (row + rows * component))] =
                    pd[component + 2 * (result + 2 * (pair + (dim / 2) * row))];
            }
        }
        backend_same(values(prepared), expected, 13);
    }
    BackendOutputs result{values(qo), values(ko)};
#ifdef KUNO_ROPE_FAILURE_TEST
    if (injected_output == 1) result.q.back() = injected_value;
    if (injected_output == 2) result.k.back() = injected_value;
#endif
    backend_finite(result.q, 19);
    backend_finite(result.k, 20);
    return result;
}

static void backend_case(ggml_backend_t backend, int dim, int rows, int heads,
                         int batch, bool interleaved, const std::vector<float>& pd,
                         bool scalar_check) {
    const size_t count = size_t(dim) * rows * heads * batch;
    std::vector<float> qd(count), kd(count);
    for (size_t i = 0; i < count; ++i) {
        qd[i] = float(int(i * 17 % 127) - 63) / 32.f;
        kd[i] = float(int(i * 23 % 113) - 56) / 64.f;
    }
    // Only one variant's device workspace is live; comparisons stay in RAM.
    auto legacy = backend_variant(backend, dim, rows, heads, batch, interleaved, qd, kd, pd, false);
    auto shared = backend_variant(backend, dim, rows, heads, batch, interleaved, qd, kd, pd, true);
    backend_same(legacy.q, shared.q, 14);
    backend_same(legacy.k, shared.k, 15);
    if (scalar_check) {
        backend_same(legacy.q, oracle(qd, pd, dim, heads, rows, batch, interleaved), 16);
        backend_same(legacy.k, oracle(kd, pd, dim, heads, rows, batch, interleaved), 17);
    }
}

static void backend_cases(ggml_backend_t backend, bool public_sizes) {
    size_t index = 0;
    auto run_case = [&](int dim, int rows, int heads, int batch, bool interleaved,
                         const std::vector<float>& pe, bool scalar_check) {
        if (public_sizes) {
            std::printf("rope_device_case index=%zu dim=%d rows=%d heads=%d batch=%d interleaved=%d phase=0\n",
                        ++index, dim, rows, heads, batch, interleaved);
            std::fflush(stdout);
        }
        backend_case(backend, dim, rows, heads, batch, interleaved, pe, scalar_check);
        if (public_sizes) {
            std::printf("rope_device_case index=%zu exact=1 input_unchanged=1 phase=1\n", index);
            std::fflush(stdout);
        }
    };
    for (bool interleaved : {false, true}) {
        const std::vector<float> pe{0.5f, -0.25f, 0.25f, 0.5f, -0.75f, -0.5f, 0.5f, -0.75f,
                                    0.25f, 0.75f, -0.75f, 0.25f, 0.5f, 0.25f, -0.25f, 0.5f};
        run_case(4, 2, 2, 2, interleaved, pe, true);
    }
    for (int text : {3, 7}) {
        sd::Tensor<int32_t> slots({text + 4});
        slots.data()[1] = slots.data()[2] = 1;
        slots.data()[text] = slots.data()[text + 1] = 2;
        auto layout = Qwen::QwenImage21Layout::build(text + 4, slots, {{2, 4}, {4, 2}, {3, 5}});
        for (bool suffix : {false, true}) for (bool cx : {false, true}) for (bool cy : {false, true}) {
            auto pe = embedding(layout, suffix, cx, cy);
            for (bool interleaved : {false, true}) run_case(128, pe.ids.size(), 2, 1, interleaved, pe.values, false);
        }
    }
    auto odd = Qwen::QwenImage21Layout::build(0, {}, {{5, 13}});
    auto odd_pe = embedding(odd, false, false, false);
    for (bool interleaved : {false, true}) run_case(128, 65, 2, 2, interleaved, odd_pe.values, false);
    if (public_sizes) for (auto shape : {std::pair<int64_t, int64_t>{111, 74}, {52, 78}}) {
        auto layout = Qwen::QwenImage21Layout::build(3, {}, {shape});
        for (bool circular : {false, true}) {
            auto pe = embedding(layout, true, circular, circular);
            run_case(128, shape.first * shape.second, 32, 1, true, pe.values, false);
        }
    }
}

int main(int argc, char** argv) {
    rlimit no_core{0, 0};
    if (setrlimit(RLIMIT_CORE, &no_core) != 0) {
        std::fputs("rope_test_error phase=1\n", stderr);
        return 2;
    }
    if (argc != 1) {
        if (argc != 4 || std::strcmp(argv[1], "--device") != 0 ||
            (std::strcmp(argv[2], "CUDA") != 0 && std::strcmp(argv[2], "Vulkan") != 0 && std::strcmp(argv[2], "SYCL") != 0)) {
            std::fputs("rope_device_error phase=1\n", stderr);
            return 2;
        }
        try {
            ggml_log_set([](enum ggml_log_level, const char*, void*) {}, nullptr);
            sd_set_log_callback([](enum sd_log_level_t, const char*, void*) {}, nullptr);
#ifdef KUNO_ROPE_FAILURE_TEST
            if (injected_entry_exception) throw 1;
#endif
            backend_call([] { ggml_backend_load_all(); });
            ggml_backend_dev_t selected = nullptr;
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                auto device = ggml_backend_dev_get(i);
                ggml_backend_dev_props props{};
                ggml_backend_dev_get_props(device, &props);
                if (props.type == GGML_BACKEND_DEVICE_TYPE_GPU && props.device_id &&
                    std::strcmp(props.device_id, argv[3]) == 0 && props.name &&
                    std::strncmp(props.name, argv[2], std::strlen(argv[2])) == 0) {
                    if (selected) {
                        std::fputs("rope_device_error phase=2\n", stderr);
                        return 2;
                    }
                    selected = device;
                }
            }
            if (!selected) {
                std::fputs("rope_device_error phase=3\n", stderr);
                return 2;
            }
            std::unique_ptr<ggml_backend, decltype(&backend_free_noexcept)> backend(
                backend_call([&] { return ggml_backend_dev_init(selected, nullptr); }), backend_free_noexcept);
            if (!backend) {
                std::fputs("rope_device_error phase=4\n", stderr);
                return 2;
            }
            backend_cases(backend.get(), true);
            backend.reset();
            if (backend_uncertain) throw BackendError{21};
            std::puts("RoPE device legacy/prepared bitwise equality and input immutability passed");
            return 0;
        } catch (const BackendError& error) {
            std::fprintf(stderr, "rope_backend_error phase=%d\n", error.phase);
            return 1;
        } catch (...) {
            std::fputs("rope_device_error phase=5\n", stderr);
            return 1;
        }
    }
    std::unique_ptr<ggml_backend, decltype(&backend_free_noexcept)> cpu(ggml_backend_cpu_init(), backend_free_noexcept);
    assert(cpu);
    ggml_backend_cpu_set_n_threads(cpu.get(), 1);
    for (bool interleaved : {false, true}) for (int dim : {4, 8, 128}) for (int rows : {2, 3, 65}) {
        std::vector<float> pe(size_t(dim) * rows * 2);
        for (size_t i = 0; i < pe.size(); i += 4) {
            const float c = float(int(i % 11) - 5) / 8.f;
            const float s = float(int(i % 7) - 3) / 8.f;
            pe[i] = c; pe[i + 1] = -s; pe[i + 2] = s; pe[i + 3] = c;
        }
        arithmetic_case(dim, rows, 2, 2, interleaved, pe);
    }
    for (int text : {3, 7}) for (bool editing : {false, true}) {
        sd::Tensor<int32_t> slots;
        std::vector<std::pair<int64_t, int64_t>> shapes{{3, 5}};
        if (editing) {
            slots = sd::Tensor<int32_t>({text + 4});
            slots.data()[1] = slots.data()[2] = 1;
            slots.data()[text] = slots.data()[text + 1] = 2;
            shapes = {{2, 4}, {4, 2}, {3, 5}};
        }
        auto layout = Qwen::QwenImage21Layout::build(editing ? text + 4 : text, slots, shapes);
        for (bool suffix : {false, true}) for (bool cx : {false, true}) for (bool cy : {false, true}) {
            auto pe = embedding(layout, suffix, cx, cy);
            for (bool interleaved : {false, true}) arithmetic_case(128, pe.ids.size(), 2, 1, interleaved, pe.values);
        }
    }
    sharing_topology();
    for (auto mode : {Qwen::QwenImage21PrefixCache::Mode::NONE, Qwen::QwenImage21PrefixCache::Mode::STORE, Qwen::QwenImage21PrefixCache::Mode::REUSE}) model_topology(cpu.get(), mode);
    segmented_case(cpu.get(), 0);
    segmented_case(cpu.get(), 1); // Fresh graph with different coefficients after cleanup.
    try {
        backend_cases(cpu.get(), false); // Exercises the device-buffer API without loading GPUs.
    } catch (const BackendError& error) {
        std::fprintf(stderr, "rope_backend_error phase=%d\n", error.phase);
        return 1;
    } catch (...) {
        std::fputs("rope_backend_error phase=18\n", stderr);
        return 1;
    }
    cpu.reset();
    if (backend_uncertain) {
        std::fputs("rope_backend_error phase=21\n", stderr);
        return 1;
    }
    std::puts("RoPE scalar oracle, exact sharing, Qwen prefix layouts and segmented lifetime passed");
}
