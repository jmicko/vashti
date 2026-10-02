// CPU-only topology/geometry/alias guards. No backend/device initialization.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdio>
#include <cstdint>
#include <utility>
#include <sys/resource.h>
#include "ggml.h"
static_assert(GGML_MAX_NAME == 160, "Use the image engine GGML_MAX_NAME=160 ABI");
#include "ggml-impl.h"
#include "mul-add-policy.hpp"

struct Fixture {
    ggml_context * ctx = ggml_init({1024 * 1024, nullptr, true});
    ggml_tensor * a, * gate, * residual, * mul, * add;
    ggml_cgraph * graph;
    int mul_index = -1;
    Fixture(int64_t columns = 7, int64_t rows = 5, bool product_first = false) {
        assert(ctx);
        a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, columns, rows);
        gate = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, columns);
        residual = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, columns, rows);
        mul = ggml_mul(ctx, a, gate);
        add = product_first ? ggml_add(ctx, mul, residual) : ggml_add(ctx, residual, mul);
        graph = ggml_new_graph_custom(ctx, 64, false);
        ggml_build_forward_expand(graph, add);
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) if (ggml_graph_node(graph, i) == mul) mul_index = i;
        assert(mul_index >= 0);
        a->data = reinterpret_cast<void *>(uintptr_t(0x10000000));
        residual->data = reinterpret_cast<void *>(uintptr_t(0x20000000));
        gate->data = reinterpret_cast<void *>(uintptr_t(0x30000000));
        add->data = reinterpret_cast<void *>(uintptr_t(0x40000000));
    }
    ~Fixture() { ggml_free(ctx); }
    bool topology() { return ggml_sycl_mul_add_graph_safe(graph, mul_index); }
};

int main() {
    rlimit limit{0, 0};
    if (setrlimit(RLIMIT_CORE, &limit) != 0) return 2;
    int checks = 0;
    auto check = [&](bool value, bool expected) {
        ++checks;
        if (value != expected) std::fprintf(stderr, "sycl_mul_add_policy_error check=%d\n", checks);
        assert(value == expected);
    };
    for (bool first : {false, true}) {
        Fixture f(7, 5, first);
        check(ggml_can_fuse(f.graph, f.mul_index, {GGML_OP_MUL, GGML_OP_ADD}), true);
        check(f.topology(), !first);
        check(ggml_sycl_mul_add_geometry(f.mul, f.add), !first);
        check(ggml_sycl_mul_add_memory_safe(f.mul, f.add), !first);
        auto mutate = [&](ggml_tensor * tensor, auto change) {
            const auto original = *tensor;
            change(*tensor);
            check(ggml_sycl_mul_add_geometry(f.mul, f.add), false);
            *tensor = original;
        };
        mutate(f.mul, [](auto & t) { t.op = GGML_OP_ADD; });
        mutate(f.add, [](auto & t) { t.op = GGML_OP_MUL; });
        for (auto t : {f.a, f.gate, f.residual, f.mul, f.add}) {
            mutate(t, [](auto & v) { v.type = GGML_TYPE_F16; });
            mutate(t, [](auto & v) { v.nb[0] += 4; });
        }
        mutate(f.a, [](auto & t) { t.ne[1]++; });
        mutate(f.residual, [](auto & t) { t.ne[0]++; });
        mutate(f.gate, [](auto & t) { t.ne[1] = 2; });
        mutate(f.gate, [](auto & t) { t.ne[0]--; });
        mutate(f.mul, [](auto & t) { t.ne[2] = 2; });
        mutate(f.add, [](auto & t) { t.ne[3] = 2; });
        mutate(f.mul, [](auto & t) { t.src[0] = nullptr; });
        mutate(f.mul, [](auto & t) { t.src[1] = nullptr; });
        mutate(f.add, [&](auto & t) { t.src[0] = t.src[1] = f.mul; });
        mutate(f.add, [&](auto & t) { t.src[0] = f.a; t.src[1] = f.residual; });
        {
            auto original0 = f.mul->src[0], original1 = f.mul->src[1];
            std::swap(f.mul->src[0], f.mul->src[1]);
            check(ggml_sycl_mul_add_geometry(f.mul, f.add), false);
            f.mul->src[0] = original0; f.mul->src[1] = original1;
        }
        for (auto t : {f.a, f.residual, f.mul, f.add}) mutate(t, [&](auto & v) { v.view_src = f.a; });
        f.gate->view_src = f.a;
        check(ggml_sycl_mul_add_geometry(f.mul, f.add), !first);
        f.gate->view_src = nullptr;
        for (auto tensor : {f.a, f.residual, f.gate, f.add}) {
            const auto original = tensor->data;
            tensor->data = nullptr;
            check(ggml_sycl_mul_add_memory_safe(f.mul, f.add), false);
            tensor->data = reinterpret_cast<void *>(uintptr_t(original) + 1);
            check(ggml_sycl_mul_add_memory_safe(f.mul, f.add), false);
            tensor->data = reinterpret_cast<void *>(UINTPTR_MAX - 3);
            check(ggml_sycl_mul_add_memory_safe(f.mul, f.add), false);
            tensor->data = original;
        }
        const auto destination = f.add->data;
        for (auto tensor : {f.a, f.residual}) {
            f.add->data = tensor->data;
            check(ggml_sycl_mul_add_memory_safe(f.mul, f.add), !first);
            f.add->data = reinterpret_cast<void *>(uintptr_t(tensor->data) + 4);
            check(ggml_sycl_mul_add_memory_safe(f.mul, f.add), false);
            f.add->data = reinterpret_cast<void *>(uintptr_t(tensor->data) - 4);
            check(ggml_sycl_mul_add_memory_safe(f.mul, f.add), false);
            const auto bytes = size_t(f.mul->ne[0] * f.mul->ne[1]) * sizeof(float);
            f.add->data = reinterpret_cast<void *>(uintptr_t(tensor->data) + bytes);
            check(ggml_sycl_mul_add_memory_safe(f.mul, f.add), !first);
            f.add->data = reinterpret_cast<void *>(uintptr_t(tensor->data) - bytes);
            check(ggml_sycl_mul_add_memory_safe(f.mul, f.add), !first);
        }
        f.add->data = f.gate->data;
        check(ggml_sycl_mul_add_memory_safe(f.mul, f.add), false);
        f.add->data = destination;
        const auto gate_address = f.gate->data;
        const auto bytes = size_t(f.mul->ne[0] * f.mul->ne[1]) * sizeof(float);
        f.gate->data = reinterpret_cast<void *>(uintptr_t(destination) + bytes);
        check(ggml_sycl_mul_add_memory_safe(f.mul, f.add), !first);
        f.gate->data = reinterpret_cast<void *>(uintptr_t(destination) + bytes - 4);
        check(ggml_sycl_mul_add_memory_safe(f.mul, f.add), false);
        f.gate->data = reinterpret_cast<void *>(uintptr_t(destination) - size_t(f.mul->ne[0]) * sizeof(float));
        check(ggml_sycl_mul_add_memory_safe(f.mul, f.add), !first);
        f.gate->data = gate_address;
        const auto residual_address = f.residual->data;
        f.residual->data = reinterpret_cast<void *>(uintptr_t(f.a->data) + 4);
        check(ggml_sycl_mul_add_memory_safe(f.mul, f.add), !first); // read-only overlap
        f.residual->data = residual_address;
        ggml_set_output(f.mul);
        check(f.topology(), false);
    }
    for (bool first : {false, true}) for (auto rows : {1, 65, 4056, 8214}) {
        Fixture f(4096, rows, first);
        check(f.topology(), !first);
        check(ggml_sycl_mul_add_geometry(f.mul, f.add), !first);
    }
    for (auto dims : {std::pair<int64_t, int64_t>{0, 5}, {-1, 5}, {7, 0}, {7, -1}, {int64_t(1) << 61, 3}}) {
        Fixture f;
        for (auto tensor : {f.a, f.residual, f.mul, f.add}) {
            tensor->ne[0] = dims.first; tensor->ne[1] = dims.second;
        }
        f.gate->ne[0] = dims.first;
        check(ggml_sycl_mul_add_geometry(f.mul, f.add), false);
    }
    {
        Fixture f;
        auto extra = ggml_scale(f.ctx, f.mul, 2.f);
        ggml_build_forward_expand(f.graph, extra);
        check(f.topology(), false);
    }
    {
        Fixture f;
        auto view = ggml_view_tensor(f.ctx, f.mul);
        ggml_build_forward_expand(f.graph, view);
        check(ggml_can_fuse(f.graph, f.mul_index, {GGML_OP_MUL, GGML_OP_ADD}), true);
        check(f.topology(), false);
        auto nested = ggml_view_tensor(f.ctx, view);
        ggml_build_forward_expand(f.graph, nested);
        check(f.topology(), false);
    }
    {
        Fixture f;
        f.mul->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
        check(f.topology(), false);
    }
    check(ggml_sycl_mul_add_geometry(nullptr, nullptr), false);
    check(ggml_sycl_mul_add_memory_safe(nullptr, nullptr), false);
    check(ggml_sycl_mul_add_graph_safe(nullptr, -1), false);
    { Fixture f; check(ggml_sycl_mul_add_graph_safe(f.graph, -1), false);
      check(ggml_sycl_mul_add_graph_safe(f.graph, ggml_graph_n_nodes(f.graph)), false); }
    std::printf("sycl_mul_add_policy passed=%d\n", checks);
}
