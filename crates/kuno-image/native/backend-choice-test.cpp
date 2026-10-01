// CPU-only physical routing and two-stage development opt-in regression.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <algorithm>
#include <cassert>
#include <cstdio>
#include "backend-choice.h"

int main() {
    unsetenv("KUNO_EXPERIMENTAL_QWEN_SYCL");
    const char* b70 = "0000:2d:00.0";
    const char* cuda = "0000:04:00.0";
    std::vector<KunoImageBackend> cards{{b70,"SYCL0",true}, {b70,"Vulkan1",true},
        {cuda,"CUDA0",true}, {cuda,"Vulkan0",true}, {b70,"CPU",false}};
    assert(!kuno_image_sycl_experiment_enabled());
    assert(kuno_choose_image_backend(cards, b70) == 1);
    assert(kuno_choose_image_backend(cards, cuda) == 2);
    assert(kuno_choose_image_backend(cards, "0000:00:00.0") == -1);
    for (const char* value : {"", "0", "true", "01", "1 ", "2"}) {
        setenv("KUNO_EXPERIMENTAL_QWEN_SYCL", value, 1);
        assert(!kuno_image_sycl_experiment_enabled());
        assert(kuno_choose_image_backend(cards, b70) == 1);
    }
    setenv("KUNO_EXPERIMENTAL_QWEN_SYCL", "1", 1);
#ifdef KUNO_NATIVE_SYCL_DEV
    assert(kuno_image_sycl_experiment_enabled());
    assert(kuno_choose_image_backend(cards, b70) == 0);
    assert(kuno_choose_image_backend(cards, cuda) == -1);
    std::reverse(cards.begin(), cards.end());
    assert(kuno_choose_image_backend(cards, b70) == 4);
    cards.push_back({b70,"SYCL1",true});
    assert(kuno_choose_image_backend(cards, b70) == -2);
    cards = {{b70,"SYCL0",false}, {cuda,"SYCL1",true}, {nullptr,"SYCL2",true},
             {b70,nullptr,true}, {b70,"Vulkan0",true}};
    assert(kuno_choose_image_backend(cards, b70) == -1);
#else
    assert(!kuno_image_sycl_experiment_enabled());
    assert(kuno_choose_image_backend(cards, b70) == 1);
#endif
    unsetenv("KUNO_EXPERIMENTAL_QWEN_SYCL");
    std::puts("backend_choice cpu_only status=passed");
}
