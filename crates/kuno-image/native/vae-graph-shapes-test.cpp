// Build the actual pinned Qwen 2.1 2D VAE decode graph with metadata only.
// No model files, backend registration, device enumeration, tensor data allocation,
// data population, scheduling, or graph execution.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "model/vae/wan_vae.hpp"
#include "presets.hpp"
#include <cassert>
#include <cstdint>
#include <cstdio>

// Isolate graph construction from every backend. Default VAE attention below
// does not request capabilities; an accidental backend call fails this test.
bool sd_backend_is(ggml_backend_t, const std::string &) { std::abort(); }
bool sd_backend_supports_cuda_mma(ggml_backend_t) { std::abort(); }
void log_printf(sd_log_level_t, const char *, int, const char *, ...) {}
void ggml_ext_im_set_f32_1d(const ggml_tensor *, int, float) { std::abort(); }

// Use the engine's parameterized attention graph builder with the same defaults
// as the normal GGMLRunnerContext wrapper. No runner or backend is constructed.
ggml_tensor * ggml_ext_attention_ext(GGMLRunnerContext * ctx,
                                    ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                                    int64_t n_head, ggml_tensor * mask,
                                    bool skip_reshape, bool flash_attn, float kv_scale) {
    assert(ctx->backend == nullptr && !ctx->flash_attn_enabled);
    assert(ctx->attn_scale == 0.f && !ctx->sage_attn_enabled);
    return ggml_ext_attention_ext(ctx->ggml_ctx, nullptr, q, k, v, n_head,
                                  mask, skip_reshape, flash_attn, kv_scale, false);
}

static void check_graph(int width, int height, uint64_t expected_largest) {
    ggml_context * params = ggml_init({16 * 1024 * 1024, nullptr, true});
    ggml_context * ctx = ggml_init({16 * 1024 * 1024, nullptr, true});
    assert(params != nullptr && ctx != nullptr);
    {
        WAN::WanVAE vae(true, VERSION_QWEN_IMAGE_2_1, true);
        vae.init(params);
        GGMLRunnerContext runner{};
        runner.ggml_ctx = ctx;
        ggml_tensor * latent = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                                                  width / 16, height / 16, 1, 64);
        ggml_tensor * output = vae.decode(&runner, latent);
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16384, false);
        ggml_build_forward_expand(graph, output);
        assert(output->ne[0] == width && output->ne[1] == height);
        assert(output->ne[2] == 1 && output->ne[3] == 4);
        uint64_t largest = 0, work_items = 0, largest_kernel_entries = 0;
        int im2col_count = 0, over_4_gib = 0;
        int wide_cont_count = 0;
        uint64_t largest_cont_offset = 0;
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            assert(node->data == nullptr && node->buffer == nullptr);
            for (const ggml_tensor * source : node->src) {
                if (source != nullptr) assert(source->data == nullptr && source->buffer == nullptr);
            }
            if (node->op == GGML_OP_CONT && node->type == GGML_TYPE_F32 &&
                node->src[0]->type == GGML_TYPE_F32 &&
                !ggml_is_contiguous(node->src[0])) {
                uint64_t source_offset = 0, destination_offset = 0;
                for (int axis = 0; axis < 4; ++axis) {
                    source_offset += uint64_t(node->src[0]->ne[axis] - 1) * node->src[0]->nb[axis];
                    destination_offset += uint64_t(node->ne[axis] - 1) * node->nb[axis];
                }
                const uint64_t max_offset = std::max(source_offset, destination_offset);
                largest_cont_offset = std::max(largest_cont_offset, max_offset);
                if (max_offset > INT32_MAX) {
                    ++wide_cont_count;
                    // The old signed element index fits, but byte offsets do not.
                    assert(ggml_nelements(node) <= INT32_MAX);
                    std::printf("{\"width\":%d,\"height\":%d,\"wide_cont_node\":%d,"
                                "\"elements\":%lld,\"source_last_byte_offset\":%llu,"
                                "\"destination_last_byte_offset\":%llu,\"device_execution\":false}\n",
                                width, height, i, (long long)ggml_nelements(node),
                                (unsigned long long)source_offset,
                                (unsigned long long)destination_offset);
                }
            }
            if (node->op != GGML_OP_IM2COL) continue;
            ++im2col_count;
            assert(node->type == GGML_TYPE_F16);
            assert(node->src[1]->type == GGML_TYPE_F32);
            assert(reinterpret_cast<const int32_t *>(node->op_params)[6] == 1);
            const uint64_t bytes = ggml_nbytes(node);
            if (bytes > (UINT64_C(1) << 32)) ++over_4_gib;
            if (bytes > largest) {
                largest = bytes;
                const uint64_t kernel_entries = node->ne[0];
                const uint64_t block = SYCL_IM2COL_BLOCK_SIZE;
                const uint64_t padded = ((kernel_entries + block - 1) / block) * block;
                work_items = padded * node->ne[1] * node->ne[2] * node->ne[3];
                largest_kernel_entries = kernel_entries;
            }
        }
        assert(ggml_graph_n_nodes(graph) == 996 && im2col_count == 46);
        assert(largest == expected_largest);
        assert(largest_kernel_entries == 288 * 3 * 3);
        if (width == 512) assert(over_4_gib == 0 && work_items == 738197504);
        else if (width == 1248) assert(over_4_gib > 0 && work_items == 2923954176);
        else assert(width == 1184 && height == 1776 && over_4_gib > 0 && work_items == 5921439744);
        if (width == 1184) assert(wide_cont_count > 0 && largest_cont_offset == 2422407164);
        else assert(wide_cont_count == 0 && largest_cont_offset <= INT32_MAX);
        std::printf("{\"width\":%d,\"height\":%d,\"graph_nodes\":%d,"
                    "\"im2col_nodes\":%d,\"over_4_gib\":%d,"
                    "\"largest_im2col_bytes\":%llu,\"legacy_work_items\":%llu,"
                    "\"wide_cont_nodes\":%d,\"largest_cont_byte_offset\":%llu,"
                    "\"device_execution\":false}\n", width, height,
                    ggml_graph_n_nodes(graph), im2col_count, over_4_gib,
                    static_cast<unsigned long long>(largest),
                    static_cast<unsigned long long>(work_items), wide_cont_count,
                    static_cast<unsigned long long>(largest_cont_offset));
    }
    ggml_free(ctx);
    ggml_free(params);
}

int main() {
    static_assert(sizeof(size_t) >= 8, "large graph metadata requires a 64-bit host");
    check_graph(512, 512, UINT64_C(1358954496));
    check_graph(1248, 832, UINT64_C(5382733824));
    check_graph(1184, 1776, UINT64_C(10900832256));
}
