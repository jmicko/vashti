#include "stable-diffusion.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "cache-budget.h"
#include "experimental-precision.h"
#include "backend-choice.h"
#include <cstring>
#include <cstdio>
#include <string>
#include <memory>
#include <algorithm>
#include <csignal>
#include <atomic>
static std::atomic<bool> device_lost{false};

static bool supports_qwen_int8(ggml_backend_dev_t device) {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({64 * 1024, nullptr, true}), ggml_free);
    if (!ctx) return false;
    auto* weights = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I8, 256, 64);
    auto* input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 256, 4);
    auto* scales = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 64);
    auto* packed = ggml_quantize_i8_convrot(ctx.get(), input, 256);
    auto* output = ggml_mul_mat_i8_tensorwise(ctx.get(), weights, packed, scales, nullptr, 256);
    return ggml_backend_dev_supports_op(device, packed) && ggml_backend_dev_supports_op(device, output);
}

static std::atomic<bool> cancelled{false};
static_assert(std::atomic<bool>::is_always_lock_free);
static void request_cancel(int) { cancelled.store(true,std::memory_order_relaxed); }
struct SignalScope {
    decltype(std::signal(SIGTERM,request_cancel)) term;
    decltype(std::signal(SIGINT,request_cancel)) interrupt;
    SignalScope() : term(std::signal(SIGTERM,request_cancel)), interrupt(std::signal(SIGINT,request_cancel)) {}
    ~SignalScope() { std::signal(SIGTERM,term);std::signal(SIGINT,interrupt); }
};
using PreviewReceiver = void (*)(int, const uint8_t*, uint32_t, uint32_t, uint32_t, void*);
struct Progress { sd_ctx_t* context = nullptr; PreviewReceiver preview = nullptr; void* userdata = nullptr; };
static void preview(int step, int count, sd_image_t* frames, bool noisy, void* data) {
    auto* state = static_cast<Progress*>(data);
    if (cancelled || noisy || !state->preview || count != 1 || !frames || !frames[0].data || (frames[0].channel != 3 && frames[0].channel != 4) || frames[0].width > 512 || frames[0].height > 512) return;
    state->preview(step, frames[0].data, frames[0].width, frames[0].height, frames[0].channel, state->userdata);
}

static void discard_sd(enum sd_log_level_t level, const char* text, void*) {
    if (text && std::strstr(text,"host prefix cache pipeline active")) std::fprintf(stderr,"kuno-image: cache transfer pipeline active\n");
    if (level != SD_LOG_ERROR || !text) return;
    if (std::strstr(text,"ErrorDeviceLost")) device_lost.store(true);
    // Library-owned source locations are useful diagnostics without emitting
    // any formatted message (which can contain prompts, paths, or tensor names).
    char file[64] = {}; int line = 0;
    if (std::sscanf(text, "%63[a-zA-Z0-9_.]:%d", file, &line) != 2 || line < 1 || line > 50000) return;
    const char* sources[] = {"image.cpp", "conditioner.hpp", "model_loader.cpp", "model_manager.cpp", "diffusion_engine.cpp", "ggml_extend_backend.cpp", "compute_workspace.cpp", "model_builders.cpp", "runner.cpp", "ggml_extend.hpp", "graph_executor.cpp", "graph_partition.cpp"};
    for (const auto* allowed : sources) if (std::strcmp(file,allowed)==0) {
        std::fprintf(stderr,"kuno-image: backend-error-source %s:%d\n",allowed,line); return;
    }
}
static void discard_ggml(enum ggml_log_level, const char* text, void*) {
    if (text && std::strstr(text,"ErrorDeviceLost")) device_lost.store(true);
}
static void progress(int step, int steps, float, void* data) {
    auto* state=static_cast<Progress*>(data);
    if (cancelled && state->context) sd_cancel_generation(state->context,SD_CANCEL_ALL);
    std::fprintf(stderr,"kuno-image: progress %d/%d\n",step,steps);
}
extern "C" int kuno_qwen_generate(const char* directory,const char* pci,const char* prompt,const char* negative,
    int width,int height,int steps,float cfg,float shift,int64_t seed,int version,sd_image_t* references,int reference_count,uint64_t host_available,int fast_attention,
    void (*receive)(const uint8_t*,size_t,uint32_t,void*),PreviewReceiver preview_receiver,void* userdata) {
    try {
        cancelled=0; device_lost.store(false);
        SignalScope signals; Progress state; state.preview=preview_receiver; state.userdata=userdata;
        sd_set_preview_callback(preview_receiver ? preview : nullptr, PREVIEW_PROJ, 1, true, false, &state);
        sd_set_log_callback(discard_sd,nullptr);ggml_log_set(discard_ggml,nullptr);sd_set_progress_callback(progress,&state);
        ggml_backend_load_all();
        std::string backend;
        std::string budget;
        uint64_t budget_bytes = 0;
        std::vector<KunoImageBackend> candidates;
        for (size_t i=0;i<ggml_backend_dev_count();++i) {
            ggml_backend_dev_props props{};ggml_backend_dev_get_props(ggml_backend_dev_get(i),&props);
            candidates.push_back({props.device_id, props.name, props.type==GGML_BACKEND_DEVICE_TYPE_GPU});
        }
        const int selected = kuno_choose_image_backend(candidates, pci);
        if (selected < 0) return 2;
        {
            const auto device = ggml_backend_dev_get(selected);
            ggml_backend_dev_props props{};ggml_backend_dev_get_props(device,&props);
            if (version && !supports_qwen_int8(device)) return 9;
            const size_t reserve=std::max<size_t>(1024ull*1024*1024,props.memory_total/10);
            if (props.memory_free<=reserve) return 6;
            budget_bytes=props.memory_free-reserve;
            budget=std::to_string(double(budget_bytes)/(1024.0*1024*1024));
            backend=props.name;
        }
        if (backend.empty()) return 2;
        if (fast_attention && (!version || backend.compare(0,4,"CUDA") != 0)) return 10;
        std::fprintf(stderr,"kuno-image: physical GPU verified\n");
        std::fprintf(stderr,"kuno-image: backend %s\n", backend.compare(0,4,"CUDA")==0 ? "CUDA" : "Vulkan");
        const std::string base(directory);
        const auto diffusion=base+(version ? "/qwen_image_2.1_int8_convrot.safetensors" : "/qwen_image_2512_fp8_e4m3fn.safetensors");
        const auto encoder=base+(version ? "/qwen3vl_8b_int8_convrot.safetensors" : "/qwen_2.5_vl_7b_fp8_scaled.safetensors");
        const auto vae=base+(version ? "/qwen_image_2.1_vae_bf16.safetensors" : "/qwen_image_vae.safetensors");
        sd_ctx_params_t context;sd_ctx_params_init(&context);
        context.diffusion_model_path=diffusion.c_str();context.llm_path=encoder.c_str();context.vae_path=vae.c_str();
        context.backend=backend.c_str();context.params_backend="cpu";context.max_vram=budget.c_str();
        uint64_t reference_pixels = 0;
        for (int i=0; i<reference_count; ++i) reference_pixels += uint64_t(references[i].width) * references[i].height;
        const bool cuda_backend = backend.compare(0, 4, "CUDA") == 0;
        const bool vulkan_backend = backend.compare(0, 6, "Vulkan") == 0;
        const bool mlp_f16 = kuno_qwen_mlp_f16_experiment(version != 0, vulkan_backend);
        const bool fp16_cache = version && (cuda_backend || vulkan_backend);
        const uint64_t working_reserve = kuno_qwen_working_reserve(cuda_backend, mlp_f16);
        if (mlp_f16) {
            if (budget_bytes < working_reserve || !kuno_qwen_mlp_f16_host_fits(host_available)) return 6;
            context.tensor_type_rules = "^model\\.diffusion_model\\.transformer_blocks\\.[0-9]+\\.img_mlp\\.(gate_up|out)\\.weight$=f16";
            std::fprintf(stderr,"kuno-image: experimental dense MLP enabled\n");
        }
        // Host-cache overlap is validated only on CUDA. Vulkan uses a resident
        // FP16 cache when the full-weight budget fits, otherwise disables it.
        if (version && cuda_backend && kuno_qwen_host_prefix_cache_fits(budget_bytes, host_available,
                reference_pixels, std::strlen(prompt) + std::strlen(negative), cfg != 1.f)) {
            context.model_args="qwen_image_2_1_prefix_cache_f16=true,qwen_image_2_1_prefix_cache_cpu=true";
            std::fprintf(stderr,"kuno-image: prefix cache in bounded system RAM\n");
        } else if (version && !kuno_qwen_prefix_cache_fits(budget_bytes, reference_pixels,
                std::strlen(prompt) + std::strlen(negative), cfg != 1.f, fp16_cache, working_reserve)) {
            context.model_args="qwen_image_2_1_prefix_cache=false";
            std::fprintf(stderr,"kuno-image: prefix cache disabled for memory budget\n");
        } else if (fp16_cache) {
            context.model_args="qwen_image_2_1_prefix_cache_f16=true";
        }
        context.sage_attn=fast_attention != 0;
        if (fast_attention) std::fprintf(stderr,"kuno-image: fast attention enabled\n");
        context.n_threads=8;context.diffusion_flash_attn=true;context.enable_mmap=true;
        context.rng_type=CPU_RNG;context.sampler_rng_type=CPU_RNG;
        std::fprintf(stderr,"kuno-image: loading models\n");
        std::unique_ptr<sd_ctx_t,decltype(&free_sd_ctx)> ctx(new_sd_ctx(&context),free_sd_ctx);
        if (!ctx) return device_lost.load() ? 8 : 3;
        if (cancelled) return device_lost.load() ? 8 : 7;
        state.context=ctx.get();
        sd_img_gen_params_t request;sd_img_gen_params_init(&request);
        request.prompt=prompt;request.negative_prompt=negative;request.width=width;request.height=height;request.batch_count=1;
        request.seed=seed;request.strength=1.0f;request.sample_params.sample_steps=steps;
        request.sample_params.sample_method=EULER_SAMPLE_METHOD;request.sample_params.scheduler=version ? FLUX_SCHEDULER : SIMPLE_SCHEDULER;
        request.sample_params.guidance.txt_cfg=cfg;if (!version) request.sample_params.flow_shift=shift;
        request.ref_images=references;request.ref_images_count=reference_count;
        if (version) request.ref_image_args="resize_before_vae=false";
        sd_image_t* images=nullptr;int count=0;
        std::fprintf(stderr,"kuno-image: generating\n");
        bool ok=generate_image(ctx.get(),&request,&images,&count);
        if (cancelled) { if (images) free_sd_images(images,count); return device_lost.load() ? 8 : 7; }
        if (!ok || !images || count!=1 || images[0].width!=(uint32_t)width || images[0].height!=(uint32_t)height || (images[0].channel!=3 && images[0].channel!=4) || !images[0].data) {
            if (images) free_sd_images(images,count);
            return device_lost.load() ? 8 : 4;
        }
        receive(images[0].data,(size_t)width*height*images[0].channel,images[0].channel,userdata);
        free_sd_images(images,count);
        return 0;
    } catch (const std::exception& error) {
        return device_lost.load() || std::strstr(error.what(),"ErrorDeviceLost") ? 8 : 5;
    } catch (...) { return device_lost.load() ? 8 : 5; }
}
