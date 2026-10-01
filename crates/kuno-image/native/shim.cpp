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
struct KunoQwenSession {
    // Declared first so handlers outlive both output delivery and model/context
    // destruction. SIGTERM while idle requests cooperative worker shutdown;
    // the supervisor closes stdin rather than forcing default-signal exit.
    SignalScope signals;
    std::string directory, pci, backend;
    std::unique_ptr<sd_ctx_t, decltype(&free_sd_ctx)> context{nullptr, free_sd_ctx};
    int version = -1;
    bool fast = false, mlp_f16 = false;
    uint64_t budget_bytes = 0;
    KunoQwenSession(const char* directory, const char* pci) : directory(directory), pci(pci) {}
};

struct CallbackScope {
    CallbackScope(Progress& state, bool previews) {
        sd_set_preview_callback(previews ? preview : nullptr, PREVIEW_PROJ, 1, true, false, &state);
        sd_set_log_callback(discard_sd, nullptr);
        ggml_log_set(discard_ggml, nullptr);
        sd_set_progress_callback(progress, &state);
        sd_set_backend_eval_callback(nullptr, nullptr);
    }
    ~CallbackScope() {
        sd_set_preview_callback(nullptr, PREVIEW_PROJ, 1, true, false, nullptr);
        sd_set_progress_callback(nullptr, nullptr);
        sd_set_backend_eval_callback(nullptr, nullptr);
        // Content-discarding log callbacks have no request-owned userdata.
    }
};

extern "C" int kuno_qwen_session_new(const char* directory, const char* pci, void** out) {
    if (!directory || !pci || !out) return 5;
    *out = nullptr;
    try {
        *out = new KunoQwenSession(directory, pci);
        return 0;
    } catch (...) { return 5; }
}

extern "C" void kuno_qwen_session_free(void* session) {
    delete static_cast<KunoQwenSession*>(session);
}

extern "C" int kuno_qwen_session_generate(void* opaque, const char* prompt, const char* negative,
    int width, int height, int steps, float cfg, float shift, int64_t seed, int version,
    sd_image_t* references, int reference_count, uint64_t host_available, int fast_attention,
    void (*receive)(const uint8_t*,size_t,uint32_t,void*), PreviewReceiver preview_receiver, void* userdata) {
    auto* session = static_cast<KunoQwenSession*>(opaque);
    if (!session || !prompt || !negative || !receive) return 5;
    cancelled = false;
    device_lost = false;
    Progress state;
    state.preview = preview_receiver;
    state.userdata = userdata;
    CallbackScope callbacks(state, preview_receiver != nullptr);
    int result = 5;
    bool request_prepared = false;
    try {
        ggml_backend_load_all();
        std::vector<KunoImageBackend> candidates;
        for (size_t i=0; i<ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_props props{};
            ggml_backend_dev_get_props(ggml_backend_dev_get(i), &props);
            candidates.push_back({props.device_id, props.name, props.type==GGML_BACKEND_DEVICE_TYPE_GPU});
        }
        const int selected = kuno_choose_image_backend(candidates, session->pci.c_str());
        if (selected < 0) return 2;
        const auto device = ggml_backend_dev_get(selected);
        ggml_backend_dev_props props{};
        ggml_backend_dev_get_props(device, &props);
        const std::string backend(props.name ? props.name : "");
        const bool cuda_backend = backend.compare(0, 4, "CUDA") == 0;
        const bool vulkan_backend = backend.compare(0, 6, "Vulkan") == 0;
        const bool sycl_backend = backend.compare(0, 4, "SYCL") == 0;
        if (backend.empty() || (sycl_backend && !kuno_image_sycl_experiment_enabled())) return 2;
        if (version && !supports_qwen_int8(device)) return 9;
        if (sycl_backend && !version) return 9;
        if (fast_attention && (!version || !cuda_backend)) return 10;
        const bool mlp_f16 = kuno_qwen_mlp_f16_experiment(version != 0, vulkan_backend);
        const bool fp16_cache = version && (cuda_backend || vulkan_backend || sycl_backend);
        const uint64_t working_reserve = kuno_qwen_working_reserve(cuda_backend, mlp_f16);
        if (mlp_f16 && !kuno_qwen_mlp_f16_host_fits(host_available)) return 6;

        // Context identity contains only public model/backend configuration.
        // Dynamic free VRAM excludes our retained weights; keep the original
        // explicit budget and let the residency manager reclaim its own params.
        if (!session->context || session->version != version || session->fast != (fast_attention != 0) ||
            session->mlp_f16 != mlp_f16 || session->backend != backend) {
            session->context.reset();
            ggml_backend_dev_get_props(device, &props);
            const size_t reserve = std::max<size_t>(1024ull*1024*1024, props.memory_total/10);
            if (props.memory_free <= reserve) return 6;
            session->budget_bytes = props.memory_free - reserve;
            if (mlp_f16 && session->budget_bytes < working_reserve) return 6;
            const std::string budget = std::to_string(double(session->budget_bytes)/(1024.0*1024*1024));
            const auto diffusion = session->directory + (version ? "/qwen_image_2.1_int8_convrot.safetensors" : "/qwen_image_2512_fp8_e4m3fn.safetensors");
            const auto encoder = session->directory + (version ? "/qwen3vl_8b_int8_convrot.safetensors" : "/qwen_2.5_vl_7b_fp8_scaled.safetensors");
            const auto vae = session->directory + (version ? "/qwen_image_2.1_vae_bf16.safetensors" : "/qwen_image_vae.safetensors");
            sd_ctx_params_t context;
            sd_ctx_params_init(&context);
            context.diffusion_model_path = diffusion.c_str();
            context.llm_path = encoder.c_str();
            context.vae_path = vae.c_str();
            context.backend = backend.c_str();
            context.params_backend = "cpu";
            context.max_vram = budget.c_str();
            context.conditioning_cache_size = 0;
            if (mlp_f16) context.tensor_type_rules = "^model\\.diffusion_model\\.transformer_blocks\\.[0-9]+\\.img_mlp\\.(gate_up|out)\\.weight$=f16";
            context.sage_attn = fast_attention != 0;
            context.n_threads = 8;
            context.diffusion_flash_attn = true;
            context.enable_mmap = true;
            context.rng_type = CPU_RNG;
            context.sampler_rng_type = CPU_RNG;
            std::fprintf(stderr, "kuno-image: physical GPU verified\n");
            std::fprintf(stderr, "kuno-image: backend %s\n", cuda_backend ? "CUDA" : sycl_backend ? "SYCL" : "Vulkan");
            std::fprintf(stderr, "kuno-image: loading models\n");
            session->context.reset(new_sd_ctx(&context));
            if (!session->context) return device_lost.load() ? 8 : 3;
            session->version = version;
            session->fast = fast_attention != 0;
            session->mlp_f16 = mlp_f16;
            session->backend = backend;
        } else {
            std::fprintf(stderr, "kuno-image: reusing resident models\n");
        }
        state.context = session->context.get();
        uint64_t reference_pixels = 0;
        for (int i=0; i<reference_count; ++i) reference_pixels += uint64_t(references[i].width) * references[i].height;
        // Prompt, reference shape, CFG and host memory are evaluated anew. No
        // request content enters the context key or survives the cleanup API.
        const bool host_cache = version && cuda_backend && kuno_qwen_host_prefix_cache_fits(
            session->budget_bytes, host_available, reference_pixels,
            std::strlen(prompt) + std::strlen(negative), cfg != 1.f);
        const bool prefix_cache = version && (host_cache || kuno_qwen_prefix_cache_fits(
            session->budget_bytes, reference_pixels, std::strlen(prompt) + std::strlen(negative),
            cfg != 1.f, fp16_cache, working_reserve));
        if (!sd_kuno_prepare_image_request(state.context, prefix_cache, fp16_cache, host_cache)) {
            session->context.reset();
            return 11;
        }
        request_prepared = true;
        if (host_cache) std::fprintf(stderr, "kuno-image: prefix cache in bounded system RAM\n");
        else if (version && !prefix_cache) std::fprintf(stderr, "kuno-image: prefix cache disabled for memory budget\n");
        if (cancelled) result = device_lost.load() ? 8 : 7;
        else {
            sd_img_gen_params_t request;
            sd_img_gen_params_init(&request);
            request.prompt = prompt;
            request.negative_prompt = negative;
            request.width = width;
            request.height = height;
            request.batch_count = 1;
            request.seed = seed;
            request.strength = 1.0f;
            request.sample_params.sample_steps = steps;
            request.sample_params.sample_method = EULER_SAMPLE_METHOD;
            request.sample_params.scheduler = version ? FLUX_SCHEDULER : SIMPLE_SCHEDULER;
            request.sample_params.guidance.txt_cfg = cfg;
            if (!version) request.sample_params.flow_shift = shift;
            request.ref_images = references;
            request.ref_images_count = reference_count;
            if (version) request.ref_image_args = "resize_before_vae=false";
            struct Images {
                sd_image_t* data = nullptr;
                int count = 0;
                ~Images() { if (data) free_sd_images(data, count); }
            } images;
            std::fprintf(stderr, "kuno-image: generating\n");
            const bool ok = generate_image(state.context, &request, &images.data, &images.count);
            if (cancelled) result = device_lost.load() ? 8 : 7;
            else if (!ok || !images.data || images.count != 1 || images.data[0].width != (uint32_t)width ||
                images.data[0].height != (uint32_t)height || (images.data[0].channel != 3 && images.data[0].channel != 4) || !images.data[0].data) {
                result = device_lost.load() ? 8 : 4;
            } else {
                receive(images.data[0].data, (size_t)width*height*images.data[0].channel, images.data[0].channel, userdata);
                result = 0;
            }
        }
    } catch (const std::exception& error) {
        result = device_lost.load() || std::strstr(error.what(), "ErrorDeviceLost") ? 8 : 5;
    } catch (...) { result = device_lost.load() ? 8 : 5; }
    // A successful result is not acknowledged until cleanup is synchronized.
    // Any uncertainty destroys the complete context before returning to Rust.
    bool cleared = !request_prepared;
    try {
        if (request_prepared && session->context) cleared = sd_kuno_clear_image_request(session->context.get());
    } catch (...) { cleared = false; }
    state.context = nullptr;
    if (!cleared) result = device_lost.load() ? 8 : 11;
    if (result != 0) session->context.reset();
    if (cleared && request_prepared) std::fprintf(stderr, "kuno-image: request content cleared\n");
    return result;
}
