// CPU-only injected backend uncertainty. One fixture per process; no GPU init.
#define KUNO_SYCL_FUSION_FAILURE_TEST 1
#define main kuno_mul_add_oracle_main
#include "sycl-mul-add-arithmetic-test.cpp"
#undef main
#include <sys/wait.h>
#include <unistd.h>

static std::string pipe_text(int fd) {
    std::string result;
    char buffer[128];
    for (;;) {
        const auto n = read(fd, buffer, sizeof(buffer));
        require(n >= 0 && result.size() + size_t(n) <= 256, 20);
        if (!n) break;
        result.append(buffer, size_t(n));
    }
    close(fd);
    return result;
}
static void device_fatal_fixture(const std::string & name) {
    const bool entry = name == "device_fatal_entry" || name == "device_fatal_entry_false";
    const int phase = entry ? 2 : name == "device_fatal_query" ? 3 :
        name == "device_fatal_submit_throw" || name == "device_fatal_submit_false" ? 7 :
        name == "device_fatal_compute_false" ? 9 : name == "device_fatal_sync" || name == "device_fatal_unsafe" ? 14 :
        name == "device_fatal_free" ? 16 : 0;
    require(phase != 0 && !device_backend_mode && initialization_attempts == 0, 20);
    int witness_pipe[2], diagnostic_pipe[2];
    require(pipe(witness_pipe) == 0 && pipe(diagnostic_pipe) == 0, 20);
    const auto child = fork(); require(child >= 0, 20);
    if (child == 0) {
        close(witness_pipe[0]); close(diagnostic_pipe[0]);
        if (dup2(diagnostic_pipe[1], STDERR_FILENO) < 0) std::_Exit(92);
        close(diagnostic_pipe[1]);
        try {
            Backend backend;
            if (!entry) {
                backend.value = backend_call(2, [&] { return ggml_backend_cpu_init(); });
                backend_result(backend.value, 2);
            }
            Graph graph(backend.value);
            if (backend.value) {
                auto input = ggml_new_tensor_1d(graph.ctx, GGML_TYPE_F32, 7);
                ggml_set_input(input);
                graph.inputs = backend_call(4, [&] { return ggml_backend_alloc_ctx_tensors(graph.ctx, backend.value); });
                backend_result(graph.inputs, 4);
                graph.allocator = backend_call(4, [&] { return ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend.value)); });
                backend_result(graph.allocator, 4);
            }
            // This borrowed-host witness would unwind BEFORE Graph. Device
            // uncertainty must terminate without its destructor or any retry.
            struct HostWitness {
                int fd;
                std::vector<float> values = std::vector<float>(257, .5f);
                ~HostWitness() noexcept { (void)write(fd, "U", 1); }
            } host{witness_pipe[1]};
            require(write(witness_pipe[1], "S", 1) == 1, 20);
            device_backend_mode = true; // Simulated boundary, CPU backend only.
            if (name == "device_fatal_entry_false" || name == "device_fatal_submit_false" || name == "device_fatal_compute_false") backend_result(false, phase);
            else if (name == "device_fatal_unsafe") {
                backend_safe = false; backend_call(3, [] {});
            } else {
                injected_backend_phase = phase;
                if (name == "device_fatal_sync") (void)synchronize_noexcept(backend.value);
                else if (name == "device_fatal_free") (void)backend.release();
                else backend_call(phase, [] {});
            }
            std::_Exit(90); // Returning/retrying is never acceptable.
        } catch (...) { std::_Exit(91); }
    }
    close(witness_pipe[1]); close(diagnostic_pipe[1]);
    int status = 0; require(waitpid(child, &status, 0) == child, 20);
    const auto witness = pipe_text(witness_pipe[0]), diagnostic = pipe_text(diagnostic_pipe[0]);
    require(WIFEXITED(status) && WEXITSTATUS(status) == 2 && witness == "S", 20);
    require(diagnostic == "sycl_mul_add_error phase=" + std::to_string(phase) + "\n", 20);
    require(!device_backend_mode && backend_safe && initialization_attempts == 0 && cleanup_order.empty(), 20);
}

int main(int argc, char ** argv) {
    try {
        require(argc == 2, 20);
        rlimit limit{0, 0};
        require(setrlimit(RLIMIT_CORE, &limit) == 0, 20);
        ggml_log_set(quiet_log, nullptr);
        const std::string name = argv[1];
        if (name.rfind("device_fatal_", 0) == 0) {
            device_fatal_fixture(name);
        } else if (name == "finite_families") {
            require(std::fesetround(FE_TONEAREST) == 0, 20);
            for (auto family : {FiniteFamily::bounded, FiniteFamily::wide, FiniteFamily::rounding}) {
                const Values values(4096, 9, family), repeated(4096, 9, family);
                require(values.activation == repeated.activation && values.gate == repeated.gate && values.residual == repeated.residual, 20);
                const auto expected = integer_reference(values, false);
                exact(expected, reference(values, false), 20);
                exact(expected, integer_reference(values, true), 20);
                size_t up = 0, down = 0, fma_distinct = 0, zero_results = 0;
                std::array<bool, 255> a_exponents{}, g_exponents{}, r_exponents{};
                for (size_t i = 0; i < values.activation.size(); ++i) {
                    const float a = values.activation[i], g = values.gate[i % values.columns], r = values.residual[i];
                    const auto ab = f32_bits(a), gb = f32_bits(g), rb = f32_bits(r);
                    require(kuno_binary32::finite(ab) && kuno_binary32::finite(gb) && kuno_binary32::finite(rb), 20);
                    a_exponents[(ab >> 23) & 255] = true; g_exponents[(gb >> 23) & 255] = true; r_exponents[(rb >> 23) & 255] = true;
                    if (family == FiniteFamily::bounded) require((gb & 0x7fffffffU) < 0x3f800000U, 20);
                    const float product = f32_value(kuno_binary32::mul(ab, gb));
                    if (std::isfinite(product)) {
                        const double unrounded = double(a) * double(g);
                        up += double(product) > unrounded; down += double(product) < unrounded;
                    }
                    fma_distinct += f32_bits(std::fma(a, g, r)) != f32_bits(expected[i]);
                    zero_results += (f32_bits(expected[i]) & 0x7fffffffU) == 0;
                }
                require(up && down && fma_distinct, 20);
                if (family == FiniteFamily::wide) {
                    require(std::all_of(a_exponents.begin(), a_exponents.end(), [](bool b) { return b; }) &&
                            std::all_of(g_exponents.begin(), g_exponents.end(), [](bool b) { return b; }) &&
                            std::all_of(r_exponents.begin(), r_exponents.end(), [](bool b) { return b; }), 20);
                }
                if (family == FiniteFamily::rounding) {
                    require(zero_results >= size_t(values.columns) * 2, 20);
                    require(kuno_binary32::mul(f32_bits(values.activation[0]), f32_bits(values.gate[0])) == 0x3fc00002U &&
                            kuno_binary32::mul(f32_bits(values.activation[1]), f32_bits(values.gate[1])) == 0x3fc00004U, 20);
                }
                std::printf("sycl_mul_add_finite_family family=%d rounded_up=%zu rounded_down=%zu fma_distinct=%zu zero_results=%zu pass=1\n",
                            int(family), up, down, fma_distinct, zero_results);
            }
            require(initialization_attempts == 0, 20);
        } else if (name == "route_scope") {
            require(route_sink == nullptr, 20);
            {
                RouteScope outer;
                ++route_sink->add;
                {
                    RouteScope inner;
                    ++route_sink->attempts; ++route_sink->fused;
                    require_routes(inner.counts, true);
                }
                require(route_sink == &outer.counts && outer.counts.add == 1 && outer.counts.attempts == 0, 20);
                try { RouteScope unwind; ++route_sink->fused; throw BackendError{18}; }
                catch (const BackendError & error) { require(error.phase == 18, 20); }
                require(route_sink == &outer.counts, 20);
                ++route_sink->mul;
                require_routes(outer.counts, false);
            }
            require(route_sink == nullptr && initialization_attempts == 0, 20);
        } else if (name == "host_control") {
            require(std::fesetround(FE_TONEAREST) == 0, 20);
            const Values values(23, 1, true);
            const auto expected = reference(values, false);
            const auto c = host_control_checks(values, expected[0]);
            require(c.zero && c.positive && c.fma_distinct && c.nearest, 20);
            const auto negative_zero = host_control_checks(values, -0.f);
            require(negative_zero.zero && !negative_zero.positive && negative_zero.fma_distinct, 20);
            require(std::fesetround(FE_DOWNWARD) == 0, 20);
            const auto downward = host_control_checks(values, expected[0]);
            require(!downward.nearest, 20);
            require(std::fesetround(FE_TONEAREST) == 0 && initialization_attempts == 0, 20);
        } else if (name == "categorical") {
            auto from_bits = [](uint32_t bits) { float value; std::memcpy(&value, &bits, sizeof(value)); return value; };
            const float tiny = std::numeric_limits<float>::denorm_min(), inf = std::numeric_limits<float>::infinity();
            const float na = from_bits(0x7fc00011), nb = from_bits(0xffc00022), nc = from_bits(0x7fc00033);
            const std::vector<float> a = {1.f, tiny, 0.f, na, na, na, inf, 0.f, -0.f, 1.f};
            const std::vector<float> b = {std::nextafter(1.f, 2.f), 0.f, -0.f, nc, nb, 1.f, -inf, 1.f, tiny, 1.f};
            const auto c = mismatch_counts(a, b);
            require(c.total == 9 && c.finite == 2 && c.subnormal == 2 && c.signed_zero == 1 && c.zero_class == 3 &&
                c.nan_both == 2 && c.nan_payload == 2 && c.nan_sign == 1 && c.nan_class == 1 && c.infinity == 1, 20);
            require(c.finite + c.subnormal + c.signed_zero + c.nan_both + c.nan_class + c.infinity == c.total, 20);
            require(mismatch_counts(a, a).total == 0 && initialization_attempts == 0, 20);
            std::vector<float> rows_a(46, 0.f), rows_b(46, 0.f);
            for (int index : {0, 21, 23, 44, 3}) rows_b[index] = 1.f;
            const auto indices = control_mismatch_counts(rows_a, rows_b, 23);
            require(indices[0] == 4 && indices[3] == 1, 20);
            size_t total = 0; for (auto count : indices) total += count;
            require(total == 5, 20);
            bool rejected = false;
            try { exact(a, b, 11); } catch (const BackendError & error) { rejected = error.phase == 11; }
            require(rejected && backend_safe && cleanup_order.empty() && synchronization_attempts == 0, 20);
        } else if (name == "entry" || name == "entry_exception") {
            char a0[] = "oracle", a1[] = "--device", a2[] = "SYCL", a3[] = "invalid";
            char * arguments[] = {a0, a1, a2, a3};
            if (name == "entry_exception") injected_backend_phase = 2;
            require(kuno_mul_add_oracle_main(name == "entry" ? 4 : 1, arguments) == 2, 20);
            require(backend_safe == (name == "entry") &&
                initialization_attempts == (name == "entry" ? 0 : 1) &&
                synchronization_attempts == 0 && cleanup_order.empty(), 20);
        } else {
            const bool numeric = name == "numeric";
            const int expected_phase = numeric ? 11 : name == "query" ? 3 :
                name == "submit_throw" || name == "submit_false" ? 7 :
                name == "compute_false" ? 9 : name == "sync" ? 14 : name == "free" ? 16 : 0;
            require(expected_phase != 0, 20);
            int observed_phase = 0;
            try {
                Backend backend;
                backend.value = backend_call(2, [&] { return ggml_backend_cpu_init(); });
                backend_result(backend.value, 2);
                Graph graph(backend.value);
                auto input = ggml_new_tensor_1d(graph.ctx, GGML_TYPE_F32, 7);
                ggml_set_input(input);
                graph.inputs = backend_call(4, [&] { return ggml_backend_alloc_ctx_tensors(graph.ctx, backend.value); });
                backend_result(graph.inputs, 4);
                graph.allocator = backend_call(4, [&] { return ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend.value)); });
                backend_result(graph.allocator, 4);
                if (numeric) require(false, 11);
                else if (name == "query" || name == "submit_throw") {
                    injected_backend_phase = expected_phase;
                    backend_call(expected_phase, [] { return true; });
                } else if (name == "submit_false" || name == "compute_false") {
                    backend_result(false, expected_phase);
                } else injected_backend_phase = expected_phase; // destructor sync/free failure
            } catch (const BackendError & error) { observed_phase = error.phase; }
            if (numeric) {
                require(observed_phase == 11 && backend_safe && synchronization_attempts == 2, 20);
                require(cleanup_order == std::vector<int>({1, 2, 3, 4}), 20);
            } else {
                require(!backend_safe && cleanup_order.empty(), 20);
                require(synchronization_attempts == (name == "sync" || name == "free" ? 1 : 0), 20);
                require(observed_phase == (name == "sync" || name == "free" ? 0 : expected_phase), 20);
                // An uncertain backend must never be called again, even if a
                // later call would succeed. Its live resources leave via exit.
                bool invoked = false;
                try { backend_call(3, [&] { invoked = true; }); }
                catch (const BackendError & error) { require(error.phase == 14, 20); }
                require(!invoked, 20);
            }
        }
        std::printf("sycl_mul_add_failure fixture=%s pass=1 gpu_initialized=0\n", argv[1]);
        return 0;
    } catch (...) {
        std::fputs("sycl_mul_add_failure_error phase=20\n", stderr);
        return 2;
    }
}
