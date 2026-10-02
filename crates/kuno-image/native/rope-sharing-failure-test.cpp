// CPU-only regressions for finite output, fixed entry errors, and failed-wait
// ownership. No device enumeration or GPU backend is invoked.
#define KUNO_ROPE_FAILURE_TEST
#define main rope_sharing_test_main
#include "rope-sharing-test.cpp"
#undef main
#include "ggml-backend-impl.h"
#include <limits>

static void require(bool condition) { if (!condition) throw 1; }

int main() {
    try {
        ggml_log_set([](enum ggml_log_level, const char*, void*) {}, nullptr);
        sd_set_log_callback([](enum sd_log_level_t, const char*, void*) {}, nullptr);
        cleanup_order.reserve(64);
        std::unique_ptr<ggml_backend, decltype(&backend_free_noexcept)> cpu(ggml_backend_cpu_init(), backend_free_noexcept);
        require(cpu != nullptr);
        ggml_backend_cpu_set_n_threads(cpu.get(), 1);
        const std::vector<float> pe{0.5f, -0.25f, 0.25f, 0.5f, -0.75f, -0.5f, 0.5f, -0.75f,
                                    0.25f, 0.75f, -0.75f, 0.25f, 0.5f, 0.25f, -0.25f, 0.5f};
        const std::vector<float> q(16, 0.25f), k(16, -0.125f);
        int nonfinite_cases = 0, reuse_cases = 0;
        for (int output : {1, 2}) for (float bad : {
                 std::numeric_limits<float>::quiet_NaN(), INFINITY, -INFINITY}) {
            injected_output = output;
            injected_value = bad;
            cleanup_order.clear();
            int phase = 0;
            try { backend_variant(cpu.get(), 4, 2, 2, 1, true, q, k, pe, true); }
            catch (const BackendError& error) { phase = error.phase; }
            require(phase == (output == 1 ? 19 : 20));
            require(!backend_uncertain && cleanup_order == std::vector<int>({1, 2, 3}));
            ++nonfinite_cases;
            injected_output = 0;
            cleanup_order.clear();
            backend_case(cpu.get(), 4, 2, 2, 1, true, pe, true);
            require(!backend_uncertain && cleanup_order == std::vector<int>({1, 2, 3, 1, 2, 3}));
            ++reuse_cases;
        }

        // A fake backend throws from graph_compute's internal wait. Only CPU
        // handles are owned; confirm no wait retry/free and no next-case launch.
        int waits = 0, frees = 0;
        struct FakeState { int* waits; int* frees; } state{&waits, &frees};
        ggml_backend fake{};
        fake.context = &state;
        fake.iface.graph_compute = [](ggml_backend_t, ggml_cgraph*) { return GGML_STATUS_SUCCESS; };
        fake.iface.synchronize = [](ggml_backend_t backend) {
            ++*static_cast<FakeState*>(backend->context)->waits;
            throw 1;
        };
        fake.iface.free = [](ggml_backend_t backend) { ++*static_cast<FakeState*>(backend->context)->frees; };
        ggml_context* saved_ctx = nullptr;
        ggml_backend_buffer_t saved_inputs = nullptr;
        ggml_gallocr_t saved_allocator = nullptr;
        cleanup_order.clear();
        {
            BackendResources resources(&fake);
            saved_ctx = resources.ctx = ggml_init({1024 * 1024, nullptr, true});
            require(saved_ctx != nullptr);
            ggml_new_tensor_1d(saved_ctx, GGML_TYPE_F32, 4);
            saved_inputs = resources.inputs = ggml_backend_alloc_ctx_tensors(saved_ctx, cpu.get());
            saved_allocator = resources.allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu.get()));
            require(saved_inputs && saved_allocator);
            bool caught = false;
            try { backend_call([&] { return ggml_backend_graph_compute(&fake, nullptr); }); }
            catch (...) { caught = true; }
            require(caught && backend_uncertain);
        }
        backend_free_noexcept(&fake);
        require(waits == 1 && frees == 0 && cleanup_order.empty());
        int rejected_phase = 0;
        try { BackendResources rejected(cpu.get()); }
        catch (const BackendError& error) { rejected_phase = error.phase; }
        require(rejected_phase == 21 && waits == 1);
        // No real asynchronous work occurred in this fake scenario; reclaim
        // the deliberately retained CPU handles after verifying ownership.
        backend_uncertain = false;
        ggml_gallocr_free(saved_allocator);
        ggml_backend_buffer_free(saved_inputs);
        ggml_free(saved_ctx);
        cpu.reset();
        require(!backend_uncertain);

        // Throw before loader execution inside the actual device-entry try.
        // Valid syntax exercises the fixed exception result without enumeration.
        injected_entry_exception = true;
        char name[] = "cpu_rope_failure_fixture", option[] = "--device", backend[] = "SYCL", device[] = "fixture";
        char* argv[] = {name, option, backend, device, nullptr};
        require(rope_sharing_test_main(4, argv) == 1);
        std::printf("rope_cleanup_passed nonfinite_cases=%d reuse_cases=%d failed_wait_cases=1 entry_cases=1\n",
                    nonfinite_cases, reuse_cases);
        return 0;
    } catch (...) {
        std::puts("rope_cleanup_error phase=1");
        return 1;
    }
}
