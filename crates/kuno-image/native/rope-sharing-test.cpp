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
#include <vector>

#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "core/segment_graph_bindings.h"
#include "model/diffusion/qwen_image_2_1.hpp"

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
    ggml_backend_tensor_get(tensor, result.data(), 0, ggml_nbytes(tensor));
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

int main() {
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu(ggml_backend_cpu_init(), ggml_backend_free);
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
    std::puts("RoPE scalar oracle, exact sharing, Qwen prefix layouts and segmented lifetime passed");
}
