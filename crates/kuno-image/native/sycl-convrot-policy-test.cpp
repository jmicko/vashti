// CPU-only guards for the optional SYCL ConvRot prototype. No device discovery.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdio>
#include "ggml.h"
#include "convrot-policy.hpp"
#include "pci-address.hpp"

int main() {
    auto ctx = ggml_init({8*1024*1024, nullptr, true});
    int checks = 0;
    auto check = [&](bool actual, bool expected) { assert(actual == expected); ++checks; };
    auto w = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, 256, 7);
    auto x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 256, 3, 2, 2);
    auto scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 7);
    auto bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 7);
    auto q = ggml_quantize_i8_convrot(ctx, x, 256);
    auto y = ggml_mul_mat_i8_tensorwise(ctx, w, q, scale, bias, 256);
    check(ggml_sycl_convrot_geometry(q, false), true);
    check(ggml_sycl_convrot_geometry(y, true), true);
    auto mutate = [&](ggml_tensor * t, auto change, auto op, bool matmul) {
        const ggml_tensor original = *t; change(*t);
        check(ggml_sycl_convrot_geometry(op, matmul), false); *t = original;
    };
    mutate(q, [](auto & t){t.ne[1]++;}, y, true);
    mutate(q, [](auto & t){t.op = GGML_OP_NONE;}, y, true);
    mutate(q, [](auto & t){t.src[0] = nullptr;}, y, true);
    mutate(q, [](auto & t){ggml_set_op_params_i32(&t,0,64);}, y, true);
    mutate(q, [](auto & t){t.ne[2] = 2;}, q, false);
    mutate(x, [](auto & t){t.type = GGML_TYPE_F16;}, q, false);
    mutate(x, [](auto & t){t.nb[1] += 4;}, q, false);
    mutate(w, [](auto & t){t.ne[2] = 2;}, y, true);
    mutate(w, [](auto & t){t.type = GGML_TYPE_F16;}, y, true);
    mutate(w, [](auto & t){t.nb[0] = 2;}, y, true);
    mutate(scale, [](auto & t){t.ne[0]--;}, y, true);
    mutate(bias, [](auto & t){t.type = GGML_TYPE_I32;}, y, true);
    mutate(y, [](auto & t){t.ne[2]++;}, y, true);
    mutate(y, [](auto & t){t.src[2] = nullptr;}, y, true);
    mutate(y, [](auto & t){ggml_set_op_params_i32(&t,2,0);}, y, true);
    auto direct = ggml_mul_mat_i8_tensorwise(ctx, w, x, scale, nullptr, 256);
    check(ggml_sycl_convrot_geometry(direct, true), true);
    auto plain = ggml_mul_mat_i8_tensorwise(ctx, w, x, scale, nullptr, 0);
    check(ggml_sycl_convrot_geometry(plain, true), true);
    for (int k : {4,12,132100,132104,132108}) {
        auto input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, 5);
        auto weight = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, k, 7);
        auto out = ggml_mul_mat_i8_tensorwise(ctx, weight, input, scale, nullptr, 0);
        check(ggml_sycl_convrot_geometry(out, true), k <= INT32_MAX/(128*127));
    }
    for (int k : {132096,132352}) {
        auto input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, 5);
        auto weight = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, k, 7);
        auto out = ggml_mul_mat_i8_tensorwise(ctx, weight, input, scale, nullptr, 256);
        check(ggml_sycl_convrot_geometry(out, true), k <= INT32_MAX/(128*127));
    }
    check(ggml_sycl_convrot_geometry(nullptr, false), false);
    for (auto pci : {"0000:2d:00.0","abcd:FF:1F.7"}) {
        check(!ggml_sycl_canonical_pci_address(pci).empty(), true);
    }
    for (auto pci : {"", "2d:00.0", "0000:2d:00.00", "0000:2d:20.0", "0000:2d:00.8",
                     "0000:2d:00.g", "0000:-d:00.0", "0000:2d:00.0x", "0000:2d:00:0", "0000:2d:+0.0"}) {
        check(ggml_sycl_canonical_pci_address(pci).empty(), true);
    }
    assert(ggml_sycl_canonical_pci_address("ABCD:EF:1F.7") == "abcd:ef:1f.7"); ++checks;
    ggml_free(ctx);
    std::printf("sycl_convrot_policy passed=%d\n", checks);
}
