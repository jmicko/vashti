#pragma once

// Synthetic benchmark diagnostics only, pinned to oneDNN 3.11.4 verbose v1.
// Consume records in RAM and retain only counters. Never use verbose with
// content-bearing workers. See onednn-verbose-classifier.md for evidence limits.
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <string_view>

namespace kuno_onednn_verbose {
enum class record {
    ignored, version_3_11_4, version_conflict, filter_enabled, malformed,
    graph_sdp_path_internal_unknown, graph_decomposed, graph_sdp_unknown,
    nested_micro_creation, nested_reference_creation, primitive_sdp_unknown,
    micro_execution, reference_execution, gpu_matmul_execution, gpu_softmax_execution,
};

inline bool finite_nonnegative(std::string_view text) {
    if (text.empty()) return false;
    double value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    return parsed.ec == std::errc() &&
           parsed.ptr == text.data() + text.size() && std::isfinite(value) && value >= 0;
}

inline bool nested_creation(std::string_view op) {
    return op == "create_nested:cache_miss" || op == "create_nested:cache_hit" ||
           op == "create_nested:kernel_cache_hit" || op == "create_nested:persistent_cache_hit" ||
           op == "create_nested:nested_primitive_cache_hit" ||
           op == "create_nested:compiled_partition_cache_hit";
}

inline record classify(std::string_view line) {
    if (line.substr(0, 15) != "onednn_verbose,") return record::ignored;
    // Fixed bounded views, with empty fields preserved. No raw record storage.
    if (line.size() > 16384 || line.find('\0') != std::string_view::npos) return record::malformed;
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.remove_suffix(1);
    std::array<std::string_view, 16> fields{};
    size_t count = 0, first = 0;
    for (;;) {
        if (count == fields.size()) return record::malformed;
        const auto comma = line.find(',', first);
        fields[count++] = line.substr(first, comma == std::string_view::npos ? comma : comma - first);
        if (comma == std::string_view::npos) break;
        first = comma + 1;
    }
    if (count < 3) return record::malformed;
    if (fields[1] != "v1") return record::version_conflict;
    size_t shift = 0;
    if (finite_nonnegative(fields[2])) shift = 1;
    const auto api = fields[2 + shift];
    if (api == "info" && count > 3 + shift && fields[3 + shift].substr(0, 8) == "oneDNN v") {
        const auto version = fields[3 + shift];
        return version.substr(0, 15) == "oneDNN v3.11.4 " ? record::version_3_11_4 : record::version_conflict;
    }
    if (api == "common" && count > 4 + shift && fields[3 + shift] == "info" &&
        fields[4 + shift] == "filter format is enabled") return record::filter_enabled;
    if (api != "graph" && api != "primitive") return record::ignored;
    if (count < 4 + shift) return record::malformed;
    const auto op = fields[3 + shift];
    const bool nested = nested_creation(op);
    if (!nested && op.substr(0, 14) == "create_nested:") return record::malformed;
    if (op != "exec" && !nested) return record::ignored;
    if (count != (api == "graph" ? 14 : 13) + shift ||
        !finite_nonnegative(fields[count - 1])) return record::malformed;
    if (fields[4 + shift] != "gpu") return record::ignored;
    if (api == "graph") {
        if (op != "exec" || fields[6 + shift] != "sdp") return record::ignored;
        if (fields[10 + shift] != "fpm:strict" || fields[12 + shift] != "dnnl_backend")
            return record::graph_sdp_unknown;
        const auto implementation = fields[11 + shift];
        if (implementation == "sdp_primitive_kernel_t") return record::graph_sdp_path_internal_unknown;
        if (implementation == "larger_partition_kernel_t" || implementation == "sdp_decomp_kernel_t")
            return record::graph_decomposed;
        return record::graph_sdp_unknown;
    }
    const auto kind = fields[5 + shift], implementation = fields[6 + shift];
    if (kind == "sdpa") {
        if (implementation == "ocl:micro:reusable")
            return nested ? record::nested_micro_creation : record::micro_execution;
        if (implementation == "ocl:ref:any")
            return nested ? record::nested_reference_creation : record::reference_execution;
        return record::primitive_sdp_unknown;
    }
    if (op == "exec" && kind == "matmul") return record::gpu_matmul_execution;
    if (op == "exec" && kind == "softmax") return record::gpu_softmax_execution;
    return record::ignored;
}

struct counts {
    bool version_3_11_4 = false, version_conflict = false, filtered = false;
    uint64_t malformed = 0, graph_sdp_paths = 0, graph_decomposed = 0, graph_unknown = 0;
    uint64_t nested_micro = 0, nested_reference = 0, primitive_unknown = 0;
    uint64_t micro_exec = 0, reference_exec = 0, gpu_matmul = 0, gpu_softmax = 0;
    void consume(std::string_view line) {
        switch (classify(line)) {
            case record::version_3_11_4: version_3_11_4 = true; break;
            case record::version_conflict: version_conflict = true; break;
            case record::filter_enabled: filtered = true; break;
            case record::malformed: ++malformed; break;
            case record::graph_sdp_path_internal_unknown: ++graph_sdp_paths; break;
            case record::graph_decomposed: ++graph_decomposed; break;
            case record::graph_sdp_unknown: ++graph_unknown; break;
            case record::nested_micro_creation: ++nested_micro; break;
            case record::nested_reference_creation: ++nested_reference; break;
            case record::primitive_sdp_unknown: ++primitive_unknown; break;
            case record::micro_execution: ++micro_exec; break;
            case record::reference_execution: ++reference_exec; break;
            case record::gpu_matmul_execution: ++gpu_matmul; break;
            case record::gpu_softmax_execution: ++gpu_softmax; break;
            case record::ignored: break;
        }
    }
    // Requires a fresh isolated process executing only the known linear SDP
    // graph, ONEDNN_VERBOSE=all with no filter, and complete untruncated capture.
    // Successful device waits, numerical checks, and a peak-memory bound remain
    // independent mandatory tests; this predicate does not replace them.
    bool optimized_graph_evidence(bool complete_unfiltered_capture, bool isolated_known_sdp_chain) const {
        return complete_unfiltered_capture && isolated_known_sdp_chain && version_3_11_4 &&
               !version_conflict && !filtered && !malformed && graph_sdp_paths && nested_micro &&
               !graph_decomposed && !graph_unknown && !nested_reference && !primitive_unknown &&
               !reference_exec && !gpu_matmul && !gpu_softmax;
    }
};
} // namespace kuno_onednn_verbose
