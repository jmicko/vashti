#pragma once
#include <cstring>
#include <vector>

struct KunoImageBackend {
    const char* pci;
    const char* name;
    bool gpu;
};

// CUDA and Vulkan may expose the same card. Select the implementation only
// after matching its physical identity; never fall back to another card/CPU.
inline int kuno_choose_image_backend(const std::vector<KunoImageBackend>& devices,
                                     const char* pci) {
    int selected = -1, priority = 0;
    bool ambiguous = false;
    for (size_t i = 0; i < devices.size(); ++i) {
        const auto& d = devices[i];
        if (!d.gpu || !d.pci || !d.name || std::strcmp(d.pci, pci) != 0) continue;
        const int rank = std::strncmp(d.name, "CUDA", 4) == 0 ? 2
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
