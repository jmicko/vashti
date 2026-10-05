// CPU-only negative regression: corrupt synthetic host results after blocking
// readback, then check fixed failure exits, cleanup order, and backend reuse.
// Compile/link with the same CPU libraries as attention-layout-test.cpp.
#define KUNO_ATTENTION_FAILURE_TEST
#define main attention_layout_test_main
#include "attention-layout-test.cpp"
#undef main

static void selected_case(ggml_backend_t backend, int which) {
    if (which == 1) cache_case(backend, false);
    if (which == 2) attention_case(backend, 1, false, false, 1, GGML_TYPE_F16);
    if (which == 3) sdpa_layout_case(backend, false, true, false, false);
}

static Failure expected_phase(int which, InjectedFault fault) {
    if (fault == InjectedFault::exception) return Failure::exception;
    if (which == 1) return Failure::cache_mismatch;
    if (which == 2) return fault == InjectedFault::mismatch ? Failure::attention_mismatch : Failure::attention_nonfinite;
    return fault == InjectedFault::mismatch ? Failure::sdpa_mismatch : Failure::sdpa_nonfinite;
}

int main() {
    try {
        cleanup_order.reserve(64);
        int negative_cases = 0, reuse_cases = 0;
        for (int which : {1, 2, 3}) for (auto fault : {
                 InjectedFault::mismatch, InjectedFault::nan,
                 InjectedFault::positive_infinity, InjectedFault::negative_infinity,
                 InjectedFault::exception}) {
            injected_case = which;
            injected_fault = fault;
            cleanup_failed = false;
            case_number = 0;
            cleanup_order.clear();
            char name[] = "cpu_failure_fixture";
            char* argv[] = {name, nullptr};
            // main catches after all case/backend guards have unwound. Status
            // 1, rather than a signal or success, is the externally used result.
            check(attention_layout_test_main(1, argv) == 1);
            check(!cleanup_failed && cleanup_order == std::vector<int>({1, 2, 3, 4}));
            ++negative_cases;

            cleanup_order.clear();
            {
                BackendGuard owner;
                owner.value = ggml_backend_cpu_init();
                check(owner.value != nullptr);
                ggml_backend_cpu_set_n_threads(owner.value, 2);
                Failure observed = Failure::invariant;
                try { selected_case(owner.value, which); }
                catch (const TestFailure& failure) { observed = failure.phase; }
                catch (...) { observed = Failure::exception; }
                check(observed == expected_phase(which, fault));
                check(!cleanup_failed && cleanup_order == std::vector<int>({1, 2, 3}));
                cleanup_order.clear();
                injected_fault = InjectedFault::none;
                // A successful graph on the same backend verifies that case
                // cleanup did not destroy ownership needed by the next case.
                selected_case(owner.value, which);
                check(!cleanup_failed && cleanup_order == std::vector<int>({1, 2, 3}));
            }
            check(!cleanup_failed && cleanup_order == std::vector<int>({1, 2, 3, 4}));
            ++reuse_cases;
        }
        std::printf("attention_cleanup_passed negative_cases=%d reuse_cases=%d\n", negative_cases, reuse_cases);
        return 0;
    } catch (...) {
        std::puts("attention_cleanup_error status=1");
        return 1;
    }
}
