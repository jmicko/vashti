#pragma once
#include <cstring>
#include <cstdlib>
#include <vector>

struct KunoImageBackend {
    const char* pci;
    const char* name;
    bool gpu;
};

inline bool kuno_image_sycl_experiment_enabled() {
#ifdef KUNO_NATIVE_SYCL_DEV
    const char* value = std::getenv("KUNO_EXPERIMENTAL_QWEN_SYCL");
    return value && std::strcmp(value, "1") == 0;
#else
    return false;
#endif
}

// CUDA and Vulkan may expose the same card. Select the implementation only
// after matching its physical identity; never fall back to another card/CPU.
inline int kuno_choose_image_backend(const std::vector<KunoImageBackend>& devices,
                                     const char* pci) {
    int selected = -1, priority = 0;
    bool ambiguous = false;
    const bool sycl = kuno_image_sycl_experiment_enabled();
    for (size_t i = 0; i < devices.size(); ++i) {
        const auto& d = devices[i];
        if (!d.gpu || !d.pci || !d.name || std::strcmp(d.pci, pci) != 0) continue;
        // A requested development SYCL run must match SYCL on this card;
        // a missing/ambiguous backend must not quietly benchmark Vulkan.
        const int rank = sycl ? (std::strncmp(d.name, "SYCL", 4) == 0 ? 3 : 0)
                       : std::strncmp(d.name, "CUDA", 4) == 0 ? 2
                       : std::strncmp(d.name, "Vulkan", 6) == 0 ? 1 : 0;
        if (rank > priority) {
            selected = static_cast<int>(i);
            priority = rank;
            ambiguous = false;
        } else if (rank && rank == priority) {
            ambiguous = true;
        }
    }
    return ambiguous ? -2 : selected;
}
