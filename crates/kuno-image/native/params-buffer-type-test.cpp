// CPU-only policy regression. Fake GGML interfaces make driver initialization
// and host-provider calls observable without linking or loading GPU libraries.
#include <cassert>
#include <cstdio>
#include <cstring>

#include "ggml-backend-impl.h"
#include "core/params_buffer_type.h"

static ggml_backend_buffer_type plain_cpu{};
static unsigned driver_initializations = 0;
static unsigned buffer_allocations = 0;

struct FakeBackend {
    const char* registry_name;
    enum ggml_backend_dev_type type;
    bool provide_host = true;
    unsigned host_requests = 0;
    unsigned default_requests = 0;
    ggml_backend_reg registry{};
    ggml_backend_device device{};
    ggml_backend backend{};
    ggml_backend_buffer_type preferred{}, host{};

    FakeBackend(const char* name, enum ggml_backend_dev_type kind) : registry_name(name), type(kind) {
        registry.context = this;
        registry.iface.get_name = [](ggml_backend_reg_t reg) {
            return static_cast<FakeBackend*>(reg->context)->registry_name;
        };
        device.context = this;
        device.reg = &registry;
        device.iface.get_type = [](ggml_backend_dev_t dev) {
            return static_cast<FakeBackend*>(dev->context)->type;
        };
        device.iface.get_buffer_type = [](ggml_backend_dev_t dev) {
            auto* self = static_cast<FakeBackend*>(dev->context);
            ++self->default_requests;
            return &self->preferred;
        };
        device.iface.get_host_buffer_type = [](ggml_backend_dev_t dev) -> ggml_backend_buffer_type_t {
            auto* self = static_cast<FakeBackend*>(dev->context);
            ++self->host_requests;
            return self->provide_host ? &self->host : nullptr;
        };
        device.iface.init_backend = [](ggml_backend_dev_t, const char*) -> ggml_backend_t {
            ++driver_initializations;
            return nullptr;
        };
        preferred.device = &device;
        host.device = &device;
        preferred.iface.alloc_buffer = host.iface.alloc_buffer = [](ggml_backend_buffer_type_t, size_t) -> ggml_backend_buffer_t {
            ++buffer_allocations;
            return nullptr;
        };
        backend.device = &device;
    }
};

// Same interface dispatch as pinned ggml-backend.cpp; deliberately no registries
// are registered globally and no backend libraries are linked into this test.
extern "C" ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend) {
    return backend->device;
}
extern "C" enum ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t dev) {
    return dev->iface.get_type(dev);
}
extern "C" ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t dev) {
    return dev->reg;
}
extern "C" const char* ggml_backend_reg_name(ggml_backend_reg_t reg) {
    return reg->iface.get_name(reg);
}
extern "C" ggml_backend_buffer_type_t ggml_backend_dev_host_buffer_type(ggml_backend_dev_t dev) {
    return dev->iface.get_host_buffer_type != nullptr ? dev->iface.get_host_buffer_type(dev) : nullptr;
}
extern "C" ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend) {
    return backend->device->iface.get_buffer_type(backend->device);
}
extern "C" ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type() {
    return &plain_cpu;
}

int main() {
    FakeBackend cpu("CPU", GGML_BACKEND_DEVICE_TYPE_CPU);
    FakeBackend cuda("CUDA", GGML_BACKEND_DEVICE_TYPE_GPU);
    FakeBackend vulkan0("Vulkan", GGML_BACKEND_DEVICE_TYPE_GPU);
    FakeBackend vulkan1("Vulkan", GGML_BACKEND_DEVICE_TYPE_GPU);
    FakeBackend integrated("Vulkan", GGML_BACKEND_DEVICE_TYPE_IGPU);
    FakeBackend other("Other", GGML_BACKEND_DEVICE_TYPE_GPU);
    ggml_backend_buffer_type split{};
    unsigned cases = 0;
    auto check = [&](ggml_backend_t compute, ggml_backend_t params, ggml_backend_buffer_type_t split_type,
                     ggml_backend_buffer_type_t expected) {
        assert(sd_params_buffer_type(compute, params, split_type) == expected);
        ++cases;
    };

    // The dangerous provider models upstream Vulkan's global device-zero host
    // buffer. CPU storage must never ask either Vulkan device for that provider.
    vulkan1.host.device = &vulkan0.device;
    check(&vulkan0.backend, &cpu.backend, nullptr, &plain_cpu);
    check(&vulkan1.backend, &cpu.backend, nullptr, &plain_cpu);
    check(&integrated.backend, &cpu.backend, nullptr, &plain_cpu);
    assert(vulkan0.host_requests == 0 && vulkan1.host_requests == 0 && integrated.host_requests == 0);

    // CUDA keeps pinned storage. CPU fallbacks and same-backend split storage
    // retain their original semantics; non-CPU params do not take the new gate.
    check(&cuda.backend, &cpu.backend, nullptr, &cuda.host);
    assert(cuda.host_requests == 1);
    check(&other.backend, &cpu.backend, nullptr, &other.host);
    other.provide_host = false;
    check(&other.backend, &cpu.backend, nullptr, &cpu.preferred);
    other.device.iface.get_host_buffer_type = nullptr;
    check(&other.backend, &cpu.backend, nullptr, &cpu.preferred);
    check(&vulkan1.backend, &vulkan1.backend, &split, &split);
    check(&vulkan1.backend, &vulkan1.backend, nullptr, &vulkan1.preferred);
    check(&cpu.backend, &cpu.backend, nullptr, &cpu.preferred);
    check(&vulkan1.backend, &cuda.backend, nullptr, &vulkan1.host);
    assert(vulkan1.host_requests == 1);
    check(nullptr, &cpu.backend, nullptr, &cpu.preferred);
    check(&vulkan1.backend, nullptr, nullptr, nullptr);

    // Missing metadata must take the established default path rather than
    // dereference a null device/registry or invent a Vulkan classification.
    vulkan1.device.reg = nullptr;
    check(&vulkan1.backend, &cpu.backend, nullptr, &vulkan1.host);
    vulkan1.device.reg = &vulkan1.registry;
    vulkan1.registry_name = nullptr;
    check(&vulkan1.backend, &cpu.backend, nullptr, &vulkan1.host);
    vulkan1.backend.device = nullptr;
    check(&vulkan1.backend, &cpu.backend, nullptr, &cpu.preferred);

    assert(driver_initializations == 0 && buffer_allocations == 0);
    std::printf("params_buffer_type_cases=%u driver_initializations=%u buffer_allocations=%u status=passed\n",
                cases, driver_initializations, buffer_allocations);
}
