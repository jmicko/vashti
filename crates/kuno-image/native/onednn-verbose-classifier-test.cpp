// CPU-only synthetic verbose parser/regression tool. No record persistence.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "onednn-verbose-classifier.hpp"
#include <cassert>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
using namespace kuno_onednn_verbose;
static const char *graph = "onednn_verbose,v1,graph,exec,gpu,0,sdp,qk;scale;softmax;pv,,synthetic_metadata,fpm:strict,sdp_primitive_kernel_t,dnnl_backend,0.125";
static const char *nested = "onednn_verbose,v1,primitive,create_nested:cache_miss,gpu,sdpa,ocl:micro:reusable,undef,synthetic_metadata,attr-scratchpad:user,alg:softmax_accurate,synthetic_problem,0.125";
static const char *version = "onednn_verbose,v1,info,oneDNN v3.11.4 (commit synthetic)";
static int stdin_counters(bool complete, bool isolated) {
    counts value; std::array<char, 16386> buffer{};
    for (;;) {
        std::cin.getline(buffer.data(), buffer.size());
        const auto size = std::cin.gcount();
        if (std::cin.bad()) return 2;
        if (std::cin.fail() && !std::cin.eof()) {
            ++value.malformed; std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        } else if (size) value.consume(std::string_view(buffer.data(), std::strlen(buffer.data())));
        if (std::cin.eof()) break;
    }
    std::cout << "{\"version_3_11_4\":" << value.version_3_11_4
        << ",\"version_conflict\":" << value.version_conflict
        << ",\"filtered\":" << value.filtered << ",\"malformed\":" << value.malformed
        << ",\"graph_sdp_paths_internal_unknown\":" << value.graph_sdp_paths
        << ",\"graph_decomposed\":" << value.graph_decomposed
        << ",\"graph_unknown\":" << value.graph_unknown
        << ",\"nested_micro_creations\":" << value.nested_micro
        << ",\"nested_reference_creations\":" << value.nested_reference
        << ",\"primitive_unknown\":" << value.primitive_unknown
        << ",\"micro_executions\":" << value.micro_exec
        << ",\"reference_executions\":" << value.reference_exec
        << ",\"gpu_matmul_executions\":" << value.gpu_matmul
        << ",\"gpu_softmax_executions\":" << value.gpu_softmax
        << ",\"optimized_graph_evidence\":" << value.optimized_graph_evidence(complete, isolated) << "}\n";
    return 0;
}
int main(int argc, char **argv) {
    if (argc > 1) {
        bool complete = false, isolated = false;
        if (std::strcmp(argv[1], "--stdin")) return 2;
        for (int i = 2; i < argc; ++i) {
            if (!std::strcmp(argv[i], "--capture-complete")) complete = true;
            else if (!std::strcmp(argv[i], "--isolated-chain")) isolated = true;
            else return 2;
        }
        return stdin_counters(complete, isolated);
    }
    assert(classify("onednn_verbose,v1,graph,exec,gpu,0,sdp,qk;scale;softmax;pv,,in0_f16:0:strided:variable:1x4x32x64:8192s2048s64s1 in1_f16:1:strided:variable:1x4x64x32:8192s2048s1s64 in2_f16:2:strided:variable:1x4x32x64:8192s2048s64s1 in3_f32:3:strided:variable:1:1 out0_f16:7:strided:variable:1x4x32x64:8192s2048s64s1,fpm:strict,sdp_primitive_kernel_t,dnnl_backend,0.125") == record::graph_sdp_path_internal_unknown);
    assert(classify("onednn_verbose,v1,graph,exec,gpu,0,sdp,qk;scale;softmax;pv,,in0_f16:0:strided:variable:1x4x32x64:8192s2048s64s1 in1_f16:1:strided:variable:1x4x64x32:8192s2048s1s64 in2_f16:2:strided:variable:1x4x32x64:8192s2048s64s1 in3_f32:3:strided:variable:1:1 out0_f16:7:strided:variable:1x4x32x64:8192s2048s64s1,fpm:strict,larger_partition_kernel_t,dnnl_backend,0.125") == record::graph_decomposed);
    assert(classify("onednn_verbose,v1,1906515000000.000000,graph,exec,gpu,0,sdp,qk;scale;softmax;pv,,in0_f16:0:strided:variable:1x4x32x64:8192s2048s64s1 in1_f16:1:strided:variable:1x4x64x32:8192s2048s1s64 in2_f16:2:strided:variable:1x4x32x64:8192s2048s64s1 in3_f32:3:strided:variable:1:1 out0_f16:7:strided:variable:1x4x32x64:8192s2048s64s1,fpm:strict,sdp_primitive_kernel_t,dnnl_backend,0.125") == record::graph_sdp_path_internal_unknown);
    assert(classify("onednn_verbose,v1,graph,exec,cpu,0,sdp,qk;scale;softmax;pv,,in0_f16:0:strided:variable:1x4x32x64:8192s2048s64s1 in1_f16:1:strided:variable:1x4x64x32:8192s2048s1s64 in2_f16:2:strided:variable:1x4x32x64:8192s2048s64s1 in3_f32:3:strided:variable:1:1 out0_f16:7:strided:variable:1x4x32x64:8192s2048s64s1,fpm:strict,sdp_decomp_kernel_t,dnnl_backend,0.125") == record::ignored);
    assert(classify("onednn_verbose,v1,primitive,exec,gpu,sdpa,ocl:micro:reusable,undef,query_f16::blocked:abcd::f0 key_f16::blocked:abdc::f0 val_f16::blocked:abcd::f0 dst_f16::blocked:abcd::f0,attr-scratchpad:user,alg:softmax_accurate scl:mul:f32:device,1x4x32x64:1x4x64x32:1x4x32x64,0.125") == record::micro_execution);
    assert(classify("onednn_verbose,v1,primitive,exec,gpu,sdpa,ocl:ref:any,undef,query_f16::blocked:abcd::f0 key_f16::blocked:abdc::f0 val_f16::blocked:abcd::f0 dst_f16::blocked:abcd::f0,attr-scratchpad:user,alg:softmax_accurate scl:mul:f32:device,1x4x32x64:1x4x64x32:1x4x32x64,0.125") == record::reference_execution);
    assert(classify("onednn_verbose,v1,graph,compile:cache_miss,gpu,0,sdp,qk;scale;softmax;pv,,in0_f16:0:strided:variable:1x4x32x64:8192s2048s64s1 in1_f16:1:strided:variable:1x4x64x32:8192s2048s1s64 in2_f16:2:strided:variable:1x4x32x64:8192s2048s64s1 in3_f32:3:strided:variable:1:1 out0_f16:7:strided:variable:1x4x32x64:8192s2048s64s1,fpm:strict,sdp_primitive_kernel_t,dnnl_backend,0.125") == record::ignored);
    assert(classify("onednn_verbose,v1,graph,create:check,sdp_primitive_kernel_t,only supports f32 intermediates.,sdp_primitive_config.cpp:154") == record::ignored);
    assert(classify("onednn_verbose,v1,graph,create:dispatch,compile,sdpa is dispatched to (sdp_primitive_kernel_t),sdp.hpp:84") == record::ignored);
    assert(classify("onednn_verbose,v1,graph,create:dispatch,compile,sdpa is failed to dispatch,sdp.hpp:86") == record::ignored);
    assert(classify("onednn_verbose,v1,graph,exec,gpu,0,sdp,qk;scale;softmax;pv,,in0_f16:0:strided:variable:1x4x32x64:8192s2048s64s1 in1_f16:1:strided:variable:1x4x64x32:8192s2048s1s64 in2_f16:2:strided:variable:1x4x32x64:8192s2048s64s1 in3_f32:3:strided:variable:1:1 out0_f16:7:strided:variable:1x4x32x64:8192s2048s64s1,fpm:strict,future_sdp_kernel_t,dnnl_backend,0.125") == record::graph_sdp_unknown);
    assert(classify("onednn_verbose,v1,graph,exec,gpu,0,sdp,qk;scale;softmax;pv,,in0_f16:0:strided:variable:1x4x32x64:8192s2048s64s1 in1_f16:1:strided:variable:1x4x64x32:8192s2048s1s64 in2_f16:2:strided:variable:1x4x32x64:8192s2048s64s1 in3_f32:3:strided:variable:1:1 out0_f16:7:strided:variable:1x4x32x64:8192s2048s64s1,fpm:strict,sdp_primitive_kernel_t,dnnl_backend") == record::malformed);
    assert(classify("onednn_verbose,v1,primitive,create_nested:cache_miss,gpu,sdpa,ocl:micro:reusable,undef,query_f16::blocked:abcd::f0 key_f16::blocked:abdc::f0 val_f16::blocked:abcd::f0 dst_f16::blocked:abcd::f0,attr-scratchpad:user,alg:softmax_accurate scl:mul:f32:device,1x4x32x64:1x4x64x32:1x4x32x64,0.125") == record::nested_micro_creation);
    assert(classify("onednn_verbose,v1,primitive,create_nested:cache_miss,gpu,sdpa,ocl:ref:any,undef,query_f16::blocked:abcd::f0 key_f16::blocked:abdc::f0 val_f16::blocked:abcd::f0 dst_f16::blocked:abcd::f0,attr-scratchpad:user,alg:softmax_accurate scl:mul:f32:device,1x4x32x64:1x4x64x32:1x4x32x64,0.125") == record::nested_reference_creation);
    assert(classify(nested) == record::nested_micro_creation);
    assert(classify(version) == record::version_3_11_4);
    assert(classify("onednn_verbose,v1,info,oneDNN v3.11.40 (commit synthetic)") == record::version_conflict);
    assert(classify("onednn_verbose,v2,graph,exec,gpu,0,sdp") == record::version_conflict);
    assert(classify("unrelated sdp_primitive_kernel_t") == record::ignored);
    assert(!finite_nonnegative({}));
    for (const auto *s : {"nan", "inf", "-1", "1ms", "1e999"}) assert(!finite_nonnegative(s));
    counts value; value.consume(version); value.consume(graph);
    assert(!value.optimized_graph_evidence(true, true));
    value.consume(nested); assert(value.optimized_graph_evidence(true, true));
    assert(!value.optimized_graph_evidence(false, true));
    assert(!value.optimized_graph_evidence(true, false));
    for (const auto *conflict : {
        "onednn_verbose,v1,graph,exec,gpu,0,sdp,ops,,metadata,fpm:strict,larger_partition_kernel_t,dnnl_backend,1",
        "onednn_verbose,v1,graph,exec,gpu,0,sdp,ops,,metadata,fpm:strict,future_kernel,dnnl_backend,1",
        "onednn_verbose,v1,primitive,exec,gpu,matmul,ocl:gemm,forward,metadata,attrs,aux,problem,1",
        "onednn_verbose,v1,primitive,exec,gpu,softmax,ocl:ref:any,forward,metadata,attrs,aux,problem,1",
        "onednn_verbose,v1,primitive,create_nested:cache_miss,gpu,sdpa,ocl:ref:any,undef,metadata,attrs,aux,problem,1",
        "onednn_verbose,v1,primitive,create_nested:cache_miss,gpu,sdpa,future_kernel,undef,metadata,attrs,aux,problem,1",
        "onednn_verbose,v1,common,info,filter format is enabled, hit components: graph",
        "onednn_verbose,v1,graph,exec,gpu,0,sdp,ops,,metadata,fpm:strict,sdp_primitive_kernel_t,dnnl_backend,nan",
        "onednn_verbose,v1,graph,exec,gpu,0,sdp,ops,,metadata,fpm:strict,sdp_primitive_kernel_t,dnnl_backend",
        "onednn_verbose,v1,info,oneDNN v3.12.0 (commit synthetic)",
    }) {
        auto copy = value; copy.consume(conflict); assert(!copy.optimized_graph_evidence(true, true));
    }
    counts no_graph; no_graph.consume(version); no_graph.consume(nested);
    assert(!no_graph.optimized_graph_evidence(true, true));
    assert(classify(std::string("onednn_verbose,") + std::string(17000, 'x')) == record::malformed);
    std::cout << "onednn_verbose_classifier status=passed\n";
}
