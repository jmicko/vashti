// CPU-only regression for the pinned engine's request cleanup and weight policy.
// Build with the patched ggml_runner.cpp before the existing engine archive.
#include "core/ggml_runner.h"
#include "ggml-cpu.h"
#include <cassert>
#include <memory>

struct ResidencyProbe : DeviceResidencyManager {
    int evictions = 0;
    bool segmented_compute_enabled() const override { return false; }
    bool prefetch_enabled() const override { return false; }
    void set_workspace_reclaimer(uintptr_t, std::function<bool()>) override {}
    void remove_runtime_owner(uintptr_t) override {}
    bool fits_compute_backend_capacity(const DeviceMemoryRequest&, const std::vector<ggml_tensor*>&) const override { return true; }
    bool assign_compute_backend(const std::vector<ggml_tensor*>&, ggml_backend_t) override { return true; }
    ggml_tensor* resolve_param_tensor(ggml_tensor* tensor) const override { return tensor; }
    bool prepare_params(const std::vector<ggml_tensor*>&) override { return true; }
    void release_compute_backend_params(const std::vector<ggml_tensor*>&) override {}
    void evict_compute_backend_params(const std::vector<ggml_tensor*>&) override { ++evictions; }
    WeightResidencyInfo inspect_compute_backend_params(const std::vector<ggml_tensor*>&) const override { return {}; }
    void update_runtime_residency(uintptr_t, ggml_backend_t, size_t) override {}
    bool ensure_compute_backend_capacity(const DeviceMemoryRequest&, const std::vector<ggml_tensor*>&,
        const std::vector<std::vector<ggml_tensor*>>&, const std::vector<ggml_tensor*>&) override { return true; }
    WeightPrefetchResult prefetch_params(uintptr_t, const std::vector<ggml_tensor*>&) override { return WeightPrefetchResult::Unsupported; }
    bool activate_prefetched_params(uintptr_t, const std::vector<ggml_tensor*>&) override { return false; }
    void clear_prefetched_params(uintptr_t) override {}
};

struct RequestRunner : GGMLRunner {
    using GGMLRunner::GGMLRunner;
    std::string get_desc() override { return "request-cleanup-test"; }
    void stage_request(ggml_tensor* input) {
        reset_compute_ctx();
        set_backend_tensor_data(ggml_new_tensor_1d(compute_ctx, GGML_TYPE_F32, 4), input->data);
        auto* graph = ggml_new_graph(compute_ctx);
        ggml_build_forward_expand(graph, input);
        cache_.stage("request-only", input);
        assert(cache_.capture(graph) == GGML_STATUS_SUCCESS);
        cache_.graph_end(true);
        assert(get_cache_tensor_by_name("request-only") != nullptr);
    }
    bool request_empty() const {
        return compute_ctx == nullptr && cache_.empty() && backend_tensor_data_map.empty() &&
            graph_cut_plan_cache_.graph_cut_plans.empty() && cache_staging_bytes_ == 0;
    }
    void set_active(bool value) { graph_active_ = value; }
};

int main() {
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu(ggml_backend_cpu_init(), ggml_backend_free);
    assert(cpu);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> source(ggml_init({65536, nullptr, true}), ggml_free);
    auto* tensor = ggml_new_tensor_1d(source.get(), GGML_TYPE_F32, 4);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(source.get(), cpu.get()), ggml_backend_buffer_free);
    assert(buffer);
    const float values[] = {1.f, 2.f, 3.f, 4.f};
    ggml_backend_tensor_set(tensor, values, 0, sizeof(values));
    auto probe = std::make_shared<ResidencyProbe>();
    {
        RequestRunner runner(cpu.get(), probe);
        probe->retain_weights_between_requests = true;
        for (int job = 0; job < 2; ++job) {
            assert(runner.runner_start());
            runner.stage_request(tensor);
            assert(runner.clear_request_state());
            assert(runner.request_empty());
            assert(probe->evictions == 0);
        }
        assert(runner.runner_start());
        runner.stage_request(tensor);
        runner.set_active(true);
        assert(!runner.clear_request_state());
        assert(!runner.request_empty());
        runner.set_active(false);
        assert(runner.clear_request_state());
        assert(runner.request_empty());
        probe->retain_weights_between_requests = false;
        assert(runner.runner_start());
        runner.stage_request(tensor);
        assert(runner.clear_request_state());
        assert(runner.request_empty());
        assert(probe->evictions == 1);
    }
}
