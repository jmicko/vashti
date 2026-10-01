// CPU-only exercise of the exact compiler's host pre-submission range check.
// There are no queue/device/context APIs, kernel submissions or user inputs.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <sycl/sycl.hpp>
#include <sycl/detail/id_queries_fit_in_int.hpp>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>

int main() {
#ifndef __SYCL_DEVICE_ONLY__
    const std::array<sycl::nd_range<3>, 4> cases{{
        {sycl::range<3>(512, 512, 2816), sycl::range<3>(1, 1, 256)},
        {sycl::range<3>(832, 1248, 2816), sycl::range<3>(1, 1, 256)},
        {sycl::range<3>(1, 1, INT32_MAX), sycl::range<3>(1, 1, 1)},
        {sycl::range<3>(1, 1, UINT64_C(2147483648)), sycl::range<3>(1, 1, 1)},
    }};
    int rejected_count = 0;
    for (const auto & range : cases) {
        bool rejected = false;
        try {
            sycl::detail::checkValueRange(range);
        } catch (const sycl::exception & e) {
            assert(e.code() == sycl::make_error_code(sycl::errc::nd_range));
            rejected = true;
        }
#if __SYCL_ID_QUERIES_FIT_IN_INT__
        assert(rejected == (range.get_global_range().size() > INT32_MAX));
#elif __SYCL_ID_QUERIES_FIT_IN_UINT__
        assert(rejected == (range.get_global_range().size() > UINT32_MAX));
#else
        assert(!rejected);
#endif
        if (rejected) ++rejected_count;
    }
#ifdef KUNO_EXPECT_SIZE_T_RANGES
    static_assert(!__SYCL_ID_QUERIES_FIT_IN_INT__ && !__SYCL_ID_QUERIES_FIT_IN_UINT__);
    assert(rejected_count == 0);
#else
    static_assert(__SYCL_ID_QUERIES_FIT_IN_INT__);
    assert(rejected_count == 2);
#endif
    std::printf("{\"cases\":%zu,\"fit_int\":%d,\"fit_uint\":%d,"
                "\"rejected_count\":%d,\"device_execution\":false}\n", cases.size(),
                __SYCL_ID_QUERIES_FIT_IN_INT__, __SYCL_ID_QUERIES_FIT_IN_UINT__, rejected_count);
#endif
}
