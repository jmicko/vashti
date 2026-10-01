// CPU-only regression. Unused generation code is discarded at link time; the
// context destructor stub checks handler ordering without loading a backend.
#include "shim.cpp"
#include <cassert>

static volatile std::sig_atomic_t term_calls = 0;
static volatile std::sig_atomic_t int_calls = 0;
static bool context_freed_with_handler = false;
static void previous_handler(int signal) {
    if (signal == SIGTERM) ++term_calls;
    if (signal == SIGINT) ++int_calls;
}

extern "C" void free_sd_ctx(sd_ctx_t*) {
    cancelled = false;
    std::raise(SIGTERM);
    context_freed_with_handler = cancelled.load() && term_calls == 0;
}

int main() {
    std::signal(SIGTERM, previous_handler);
    std::signal(SIGINT, previous_handler);
    void* session = nullptr;
    assert(kuno_qwen_session_new("public-models", "0000:01:00.0", &session) == 0);
    assert(session);
    cancelled = false;
    std::raise(SIGTERM); // Idle between requests must not terminate the process.
    assert(cancelled.load() && term_calls == 0);
    cancelled = false;
    std::raise(SIGINT); // Handler also covers Rust encoding/output after generate.
    assert(cancelled.load() && int_calls == 0);
    static_cast<KunoQwenSession*>(session)->context.reset(reinterpret_cast<sd_ctx_t*>(1));
    kuno_qwen_session_free(session);
    assert(context_freed_with_handler);
    std::raise(SIGTERM);
    std::raise(SIGINT);
    assert(term_calls == 1 && int_calls == 1); // Original handlers restored last.
}
