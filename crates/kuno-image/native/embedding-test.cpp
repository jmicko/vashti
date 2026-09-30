// Synthetic regression for the pinned engine's ComfyUI INT8 embedding adapter.
// Independent dense H4 tensor-product oracle checks inverse rotation and row scales.
#include <cmath>
#include <cassert>
#include <cstring>
#include "ggml-cpu.h"
#include "cache-budget.h"
#include "backend-choice.h"
#include "model/common/ggml_block.hpp"
#include "pipeline/diffusion_engine.h"
#include "pipeline/request.h"
#include "model/vae/vae.hpp"
int main() {
    {
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu(ggml_backend_cpu_init(), ggml_backend_free);
        StableDiffusionGGML engine;
        engine.version = VERSION_QWEN_IMAGE_2_1;
        engine.first_stage_model = std::make_shared<FakeVAE>(engine.version, cpu.get());
        sd_img_gen_params_t params;
        sd_img_gen_params_init(&params);
        params.width = 1184;
        params.height = 1776;
        sd::pipeline::GenerationRequest request(&engine, &params);
        request.align_generation_request_size();
        assert(request.width == 1184 && request.height == 1776);
    }
    const char* intel = "0000:2d:00.0";
    const char* nvidia = "0000:04:00.0";
    std::vector<KunoImageBackend> devices{{intel,"Vulkan1",true}, {nvidia,"CUDA0",true}, {nvidia,"Vulkan0",true}};
    assert(kuno_choose_image_backend(devices,intel)==0);
    assert(kuno_choose_image_backend(devices,nvidia)==1);
    std::reverse(devices.begin(),devices.end());
    assert(kuno_choose_image_backend(devices,nvidia)==1);
    assert(kuno_choose_image_backend(devices,"0000:00:00.0")==-1);
    devices.push_back({nvidia,"CUDA1",true});
    assert(kuno_choose_image_backend(devices,nvidia)==-2);
    devices={{nvidia,"CPU",false}, {intel,"CUDA0",true}};
    assert(kuno_choose_image_backend(devices,nvidia)==-1);
    constexpr uint64_t gib=1024ull*1024*1024;
    assert(kuno_qwen_prefix_cache_fits(11*gib,0,100,true));
    assert(!kuno_qwen_prefix_cache_fits(11*gib,1024*1024,100,true));
    assert(!kuno_qwen_prefix_cache_fits(11*gib,1024*1024,100,false));
    assert(kuno_qwen_prefix_cache_fits(29*gib,1024*1024,100,true));
    assert(!kuno_qwen_prefix_cache_fits(29*gib,4*1024*1024,100,true));
    assert(!kuno_qwen_prefix_cache_fits(11*gib,0,32768,true));
    assert(kuno_qwen_prefix_cache_fits(10*gib,1024*1024,100,true,true,3*gib));
    assert(!kuno_qwen_prefix_cache_fits(10*gib,2*1024*1024,100,true,true,3*gib));
    assert(!kuno_qwen_prefix_cache_fits(2*gib,0,0,false,true,3*gib));
    // Vulkan keeps its full-weight reserve even with half-precision cache.
    // A two-reference guided edit fits FP16 on a free B70 but not FP32.
    assert(kuno_qwen_prefix_cache_fits(20*gib,2*1024*1024,100,true,true));
    assert(!kuno_qwen_prefix_cache_fits(20*gib,2*1024*1024,100,true,false));
    assert(!kuno_qwen_prefix_cache_fits(10*gib,1024*1024,100,true,true));
    assert(!kuno_qwen_prefix_cache_fits(8*gib,0,0,false,true));
    const auto cache_bytes = kuno_qwen_prefix_cache_bytes(1024*1024,100,true,true);
    assert(kuno_qwen_prefix_cache_fits(9*gib+cache_bytes,1024*1024,100,true,true));
    assert(!kuno_qwen_prefix_cache_fits(9*gib+cache_bytes-1,1024*1024,100,true,true));
    assert(kuno_qwen_prefix_cache_bytes(1024*1024,100,true,false) == 2*cache_bytes);
    assert(kuno_qwen_host_prefix_cache_fits(10*gib,32*gib,1024*1024,100,true));
    assert(!kuno_qwen_host_prefix_cache_fits(10*gib,28*gib,1024*1024,100,true));
    assert(!kuno_qwen_host_prefix_cache_fits(10*gib,0,1024*1024,100,true));
    assert(!kuno_qwen_host_prefix_cache_fits(24*gib,48*gib,1024*1024,100,true));
    assert(!kuno_qwen_host_prefix_cache_fits(10*gib,48*gib,0,100,true));
    assert(!kuno_qwen_host_prefix_cache_fits(2*gib,48*gib,1024*1024,100,true));
    assert(!kuno_qwen_host_prefix_cache_fits(10*gib,48*gib,10*1024*1024,100,true));
    assert(!kuno_qwen_host_prefix_cache_fits(UINT64_MAX,UINT64_MAX,UINT64_MAX,UINT64_MAX,true));
    assert(!kuno_qwen_prefix_cache_fits(UINT64_MAX,0,UINT64_MAX,true,true));
    for (int group : {0, 4, 16, 256}) for (bool scalar : {false, true}) {
        const int dim = group ? group * 2 : 8, rows = 3;
        auto* ctx = ggml_init({8*1024*1024, nullptr, false});
        int64_t shape[] = {dim, rows}, scale_shape[] = {scalar ? 1 : rows};
        TensorStorage weights("e.weight", GGML_TYPE_I8, shape, 2, 0);
        weights.is_int8_tensorwise = true; weights.int8_convrot = group != 0;
        weights.int8_convrot_group_size = group;
        TensorStorage scales("e.weight_scale", GGML_TYPE_F32, scale_shape, 1, 0);
        String2TensorStorage storage{{"e.weight", weights}, {"e.weight_scale", scales}};
        Embedding embedding(rows, dim); embedding.init(ctx, storage, "e");
        std::map<std::string, ggml_tensor*> params; embedding.get_param_tensors(params, "e");
        auto* q = static_cast<int8_t*>(params.at("e.weight")->data);
        for (int i=0; i<rows*dim; ++i) q[i] = int8_t((i*17)%251 - 125);
        auto* scale = static_cast<float*>(params.at("e.weight_scale")->data);
        for (int i=0; i<scale_shape[0]; ++i) scale[i] = 0.125f * (i+1);
        auto* ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 2);
        int32_t selected[] = {2, 0, 2, 1}; std::memcpy(ids->data, selected, sizeof(selected));
        GGMLRunnerContext runner; runner.ggml_ctx = ctx;
        auto* out = embedding.forward(&runner, ids);
        auto* graph = ggml_new_graph(ctx); ggml_build_forward_expand(graph, out);
        assert(ggml_graph_compute_with_ctx(ctx, graph, 3) == GGML_STATUS_SUCCESS);
        assert(out->ne[0] == dim && out->ne[1] == 2 && out->ne[2] == 2);
        const int H[4][4]={{1,1,1,-1},{1,1,-1,1},{1,-1,1,1},{-1,1,1,1}};
        for (int i=0; i<4; ++i) for (int k=0; k<dim; ++k) {
            double expected = 0;
            if (!group) expected = q[selected[i]*dim+k];
            else for (int j=0; j<group; ++j) {
                int sign=1, a=j, b=k%group;
                for (int p=group; p>1; p/=4) { sign*=H[a%4][b%4]; a/=4; b/=4; }
                expected += q[selected[i]*dim+(k/group)*group+j]*sign/std::sqrt(group);
            }
            expected *= scale[scalar ? 0 : selected[i]];
            assert(std::abs(static_cast<float*>(out->data)[i*dim+k]-expected)<0.0001);
        }
        ggml_free(ctx);
    }
    for (float factor : {1.f, 1.f/32.f}) {
        auto* ctx = ggml_init({8*1024*1024, nullptr, false});
        auto* w = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, 256, 64);
        auto* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 4);
        auto* scales = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
        auto* bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
        for (int n=0; n<64; ++n) {
            for (int k=0; k<256; ++k) static_cast<int8_t*>(w->data)[n*256+k]=n+1;
            static_cast<float*>(scales->data)[n]=0.125f;
            static_cast<float*>(bias->data)[n]=n-20;
        }
        for (int row=0; row<4; ++row) for (int k=0; k<256; ++k)
            static_cast<float*>(x->data)[row*256+k]=row-2;
        auto* out = ggml_ext_linear_i8_tensorwise(ctx, x, w, scales, bias, 256, factor);
        auto* graph=ggml_new_graph(ctx); ggml_build_forward_expand(graph,out);
        bool packed=false;
        for (int i=0; i<ggml_graph_n_nodes(graph); ++i) {
            auto* node=ggml_graph_node(graph,i);
            if (node->op==GGML_OP_MUL_MAT) {
                assert(node->src[1]->type==GGML_TYPE_I8);
                assert(node->src[1]->op==GGML_OP_QUANTIZE_I8_CONVROT);
                packed=true;
            }
        }
        assert(packed && ggml_graph_compute_with_ctx(ctx,graph,3)==GGML_STATUS_SUCCESS);
        // A normalized regular Hadamard matrix preserves a constant row.
        for (int row=0; row<4; ++row) for (int n=0; n<64; ++n) {
            float expected=256.f*(n+1)*0.125f*(row-2)+(n-20);
            assert(std::abs(static_cast<float*>(out->data)[row*64+n]-expected)<0.001f);
        }
        ggml_free(ctx);
    }
    std::puts("INT8 embedding rotation and scaled projection regressions passed");
}
