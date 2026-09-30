// CPU-only real ModelLoader/Linear regression. All files contain synthetic weights.
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <unistd.h>
#include <vector>
#include "ggml-cpu.h"
#include "json.hpp"
#include "model_loader.h"
#include "model/common/ggml_block.hpp"
#include "stable-diffusion.h"
#include "experimental-precision.h"

using Json = nlohmann::json;
static const char* rules = "^model\\.diffusion_model\\.transformer_blocks\\.[0-9]+\\.img_mlp\\.(gate_up|out)\\.weight$=f16";
static const std::string prefix = "model.diffusion_model.";
struct Directory {
    char path[64] = "/tmp/kuno-mlp-loader-XXXXXX";
    Directory() { assert(mkdtemp(path)); }
    ~Directory() { std::filesystem::remove_all(path); }
};
struct Fixture {
    std::string module = "transformer_blocks.3.img_mlp.gate_up";
    int group = 256, columns = 512, outputs = 3;
    bool convrot = true, include_scale = true, scale_matrix = true;
    bool scale_rank3 = false, weight_rank3 = false, include_weight = true;
    std::string scale_dtype = "F32";
    std::vector<float> scales = {.015625f, 0.f, .046875f};
    std::vector<int8_t> weights;
    std::vector<float> bias;
    void write(const std::string& path) {
        weights.resize(columns * outputs);
        for (size_t i = 0; i < weights.size(); ++i) weights[i] = static_cast<int8_t>((i * 17) % 256 - 128);
        bias.resize(outputs);
        for (int i = 0; i < outputs; ++i) bias[i] = .125f * (i - 1);
        Json header = Json::object();
        std::vector<uint8_t> data;
        auto add = [&](const std::string& name, const std::string& dtype, Json shape, const void* bytes, size_t n) {
            const size_t begin = data.size();
            const auto* source = static_cast<const uint8_t*>(bytes);
            if (n) data.insert(data.end(), source, source + n);
            header[name] = {{"dtype", dtype}, {"shape", shape}, {"data_offsets", {begin, data.size()}}};
        };
        if (include_weight) {
            const Json shape = weight_rank3 ? Json{outputs, columns / 2, 2} : Json{outputs, columns};
            add(module + ".weight", "I8", shape, weights.data(), weights.size());
            add(module + ".bias", "F32", {outputs}, bias.data(), bias.size() * sizeof(float));
        }
        if (include_scale) {
            const Json shape = scale_rank3 ? Json{scales.size(), 1, 1} : (scale_matrix ? Json{scales.size(), 1} : Json{scales.size()});
            if (scale_dtype == "F16" || scale_dtype == "BF16") {
                std::vector<uint16_t> encoded(scales.size());
                for (size_t i = 0; i < scales.size(); ++i) {
                    if (scale_dtype == "F16") encoded[i] = ggml_fp32_to_fp16(scales[i]);
                    else { auto value = ggml_fp32_to_bf16(scales[i]); std::memcpy(&encoded[i], &value, 2); }
                }
                add(module + ".weight_scale", scale_dtype, shape, encoded.data(), encoded.size() * 2);
            } else add(module + ".weight_scale", scale_dtype, shape, scales.data(), scales.size() * 4);
        }
        const std::string quant = Json{{"format", "int8_tensorwise"}, {"convrot", convrot}, {"convrot_groupsize", group}}.dump();
        add(module + ".comfy_quant", "U8", {quant.size()}, quant.data(), quant.size());
        const std::string encoded = header.dump();
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        uint64_t length = encoded.size();
        for (int i = 0; i < 8; ++i) file.put(static_cast<char>(length >> (8 * i)));
        file.write(encoded.data(), encoded.size());
        file.write(reinterpret_cast<const char*>(data.data()), data.size());
        assert(file.good());
    }
    std::vector<ggml_fp16_t> oracle() const {
        const int H[4][4] = {{1,1,1,-1},{1,1,-1,1},{1,-1,1,1},{-1,1,1,1}};
        std::vector<ggml_fp16_t> result(weights.size());
        for (int r = 0; r < outputs; ++r) for (int k = 0; k < columns; ++k) {
            double value = 0;
            for (int j = 0; j < group; ++j) {
                int sign = 1, a = j, b = k % group;
                for (int p = group; p > 1; p /= 4) { sign *= H[a % 4][b % 4]; a /= 4; b /= 4; }
                value += weights[r * columns + (k / group) * group + j] * sign;
            }
            value *= scales[scales.size() == 1 ? 0 : r] / std::sqrt(group);
            result[r * columns + k] = ggml_fp32_to_fp16(static_cast<float>(value));
        }
        return result;
    }
};
static bool load_params(ModelLoader& loader, std::map<std::string, ggml_tensor*>& params, bool mmap) {
    std::set<std::string> names;
    for (const auto& item : params) names.insert(item.first);
    return loader.load_tensors([&](const TensorStorage& storage, ggml_tensor** dst) {
        const auto found = params.find(storage.name);
        *dst = found == params.end() ? nullptr : found->second;
        return true;
    }, mmap, &names, false);
}
static void verify(ModelLoader& loader, const Fixture& fixture, bool mmap, ModelLoader::FileId scoped_id = 0) {
    const std::string module = prefix + fixture.module, name = module + ".weight";
    const auto& source = loader.get_tensor_storage_map().at(name);
    assert(source.type == GGML_TYPE_I8 && source.is_int8_tensorwise && source.int8_convrot);
    assert(source.expected_type == GGML_TYPE_F16);
    auto* ctx = ggml_init({16 * 1024 * 1024, nullptr, false});
    Linear linear(fixture.columns, fixture.outputs, true);
    linear.init(ctx, loader.get_tensor_storage_map(), module);
    std::map<std::string, ggml_tensor*> params;
    linear.get_param_tensors(params, module);
    assert(params.at(name)->type == GGML_TYPE_F16 && params.count(module + ".weight_scale") == 0);
    if (mmap) {
        std::map<std::string, ggml_tensor*> weight_only{{name, params.at(name)}};
        auto no_mmap = loader.mmap_tensors(weight_only, {}, false);
        assert(no_mmap.empty() && params.at(name)->buffer == nullptr);
    }
    if (scoped_id == 0) assert(load_params(loader, params, mmap));
    else {
        std::set<std::string> names;
        for (const auto& item : params) names.insert(item.first);
        assert(loader.load_file_tensors(scoped_id, VERSION_QWEN_IMAGE_2_1,
            [&](const TensorStorage& source, ggml_tensor** dst) {
                const auto found = params.find(source.name);
                *dst = found == params.end() ? nullptr : found->second; return true;
            }, names, mmap));
    }
    const auto expected = fixture.oracle();
    const auto* loaded = static_cast<const ggml_fp16_t*>(params.at(name)->data);
    for (size_t i = 0; i < expected.size(); ++i) {
        // Both signs of zero represent the same reconstructed weight.
        if (loaded[i] != expected[i] && !(ggml_fp16_to_fp32(loaded[i]) == 0.f && ggml_fp16_to_fp32(expected[i]) == 0.f)) {
            std::fprintf(stderr, "synthetic_half_mismatch group=%d index=%zu loaded=%u expected=%u\n", fixture.group, i, unsigned(loaded[i]), unsigned(expected[i]));
            assert(false);
        }
    }
    auto* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, fixture.columns, 5);
    auto* values = static_cast<float*>(x->data);
    for (int i = 0; i < fixture.columns * 5; ++i) values[i] = ((i * 11) % 37 - 18) / 32.f;
    GGMLRunnerContext runner; runner.ggml_ctx = ctx;
    auto* result = linear.forward(&runner, x);
    auto* graph = ggml_new_graph(ctx); ggml_build_forward_expand(graph, result);
    bool dense = false;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto* node = ggml_graph_node(graph, i);
        assert(node->op != GGML_OP_QUANTIZE_I8_CONVROT);
        if (node->op == GGML_OP_MUL_MAT) {
            dense = true;
            assert(node->src[0]->type == GGML_TYPE_F16 && node->src[1]->type == GGML_TYPE_F32);
            int32_t precision; std::memcpy(&precision, node->op_params, sizeof(precision));
            assert(precision == GGML_PREC_F32);
        }
    }
    assert(dense && ggml_graph_compute_with_ctx(ctx, graph, 3) == GGML_STATUS_SUCCESS);
    for (int r = 0; r < 5; ++r) for (int n = 0; n < fixture.outputs; ++n) {
        double sum = fixture.bias[n];
        for (int k = 0; k < fixture.columns; ++k) sum += values[r * fixture.columns + k] * ggml_fp16_to_fp32(expected[n * fixture.columns + k]);
        assert(std::fabs(static_cast<float*>(result->data)[r * fixture.outputs + n] - sum) <= 5e-5 * (1 + std::fabs(sum)));
    }
    ggml_free(ctx);
}
static bool failure(ModelLoader& loader, const Fixture& fixture) {
    auto* ctx = ggml_init({16 * 1024 * 1024, nullptr, false});
    std::map<std::string, ggml_tensor*> params{{prefix + fixture.module + ".weight", ggml_new_tensor_2d(ctx, GGML_TYPE_F16, fixture.columns, fixture.outputs)}};
    const bool failed = !load_params(loader, params, true);
    ggml_free(ctx); return failed;
}
int main() {
    sd_set_log_callback([](enum sd_log_level_t, const char*, void*) {}, nullptr);
    sd_set_progress_callback([](int, int, float, void*) {}, nullptr);
    Directory directory;
    const std::string path = std::string(directory.path) + "/synthetic.safetensors";
    constexpr uint64_t gib = 1024ull * 1024 * 1024;
    assert(kuno_qwen_working_reserve(false, true) == 14 * gib);
    assert(kuno_qwen_working_reserve(false, false) == 9 * gib);
    assert(kuno_qwen_working_reserve(true, false) == 3 * gib);
    assert(!kuno_qwen_mlp_f16_host_fits(32 * gib - 1));
    assert(kuno_qwen_mlp_f16_host_fits(32 * gib));
    int64_t oversized_shape[] = {INT64_MAX - 3, 2};
    TensorStorage oversized("", GGML_TYPE_I8, oversized_shape, 2, 0);
    oversized.is_int8_tensorwise = oversized.int8_convrot = true;
    oversized.int8_convrot_group_size = 4;
    assert(!int8_dense_shape_valid(oversized));
    for (const char* value : {"", "0", "true", "01", "1 "}) {
        setenv("KUNO_EXPERIMENTAL_QWEN_MLP_F16", value, 1);
        assert(!kuno_qwen_mlp_f16_experiment(true, true));
    }
    setenv("KUNO_EXPERIMENTAL_QWEN_MLP_F16", "1", 1);
    assert(kuno_qwen_mlp_f16_experiment(true, true));
    assert(!kuno_qwen_mlp_f16_experiment(false, true));
    assert(!kuno_qwen_mlp_f16_experiment(true, false));
    int cases = 0;
    for (int group : {4, 16, 256}) for (const char* dtype : {"F32", "F16", "BF16"}) for (bool scalar : {false, true}) for (bool mmap : {false, true}) {
        Fixture fixture; fixture.group = group; fixture.columns = group * 2; fixture.scale_dtype = dtype;
        if (scalar) fixture.scales = {.015625f};
        fixture.module = (cases % 2) ? "transformer_blocks.3.img_mlp.out" : "transformer_blocks.3.img_mlp.gate_up";
        fixture.write(path);
        ModelLoader loader; loader.set_n_threads(3); ModelLoader::FileId id;
        assert(loader.add_file(path, prefix, &id));
        loader.set_wtype_override(GGML_TYPE_COUNT, rules);
        const auto raw_bytes = fixture.weights.size();
        assert(loader.get_params_mem_size(nullptr) == static_cast<int64_t>(raw_bytes * 2 + fixture.scales.size() * (fixture.scale_dtype == "F32" ? 4 : 2) + fixture.bias.size() * 4 + 3 * 128));
        assert(loader.file_tensors(id, VERSION_QWEN_IMAGE_2_1).at(prefix + fixture.module + ".weight").expected_type == GGML_TYPE_F16);
        verify(loader, fixture, mmap);
        if (cases == 0) verify(loader, fixture, mmap, id);
        ++cases;
    }
    // Default/global override, unrelated modules, and malformed opt-ins stay I8.
    for (const char* env : {"", "0", "01", "1"}) {
        setenv("KUNO_EXPERIMENTAL_QWEN_MLP_F16", env, 1);
        Fixture fixture; fixture.write(path);
        ModelLoader loader; assert(loader.add_file(path, prefix));
        loader.set_wtype_override(GGML_TYPE_F16);
        assert(loader.get_tensor_storage_map().at(prefix + fixture.module + ".weight").expected_type == GGML_TYPE_COUNT);
        loader.set_wtype_override(GGML_TYPE_COUNT, rules);
        assert((loader.get_tensor_storage_map().at(prefix + fixture.module + ".weight").expected_type == GGML_TYPE_F16) == (std::string(env) == "1"));
    }
    setenv("KUNO_EXPERIMENTAL_QWEN_MLP_F16", "1", 1);
    for (const char* module : {"transformer_blocks.3.attn.to_q", "transformer_blocks.3.img_mlp.proj", "transformer_blocks.x.img_mlp.out", "transformer_blocks.3.img_mlp.out.extra"}) {
        Fixture fixture; fixture.module = module; fixture.write(path);
        ModelLoader loader; assert(loader.add_file(path, prefix)); loader.set_wtype_override(GGML_TYPE_COUNT, ".*weight$=f16");
        assert(loader.get_tensor_storage_map().at(prefix + fixture.module + ".weight").expected_type == GGML_TYPE_COUNT);
    }
    for (int invalid = 0; invalid < 12; ++invalid) {
        Fixture fixture;
        if (invalid == 0) fixture.include_scale = false;
        if (invalid == 1) fixture.scales = {1, 1};
        if (invalid == 2) fixture.scales[0] = std::numeric_limits<float>::quiet_NaN();
        if (invalid == 3) fixture.scales[0] = std::numeric_limits<float>::infinity();
        if (invalid == 4) fixture.scales[0] = -1;
        if (invalid == 5) fixture.scale_dtype = "I32";
        if (invalid == 6) fixture.scales[0] = 100000.f;
        if (invalid == 7) fixture.convrot = false;
        if (invalid == 8) fixture.group = 64 * 2;
        if (invalid == 9) fixture.scale_rank3 = true;
        if (invalid == 10) fixture.weight_rank3 = true;
        if (invalid == 11) fixture.scales.clear();
        fixture.write(path); ModelLoader loader;
        if (loader.add_file(path, prefix)) { loader.set_wtype_override(GGML_TYPE_COUNT, rules); assert(failure(loader, fixture)); }
    }
    // Refresh preserves rules and notices scale-only changes to the logical file.
    Fixture fixture; fixture.write(path); ModelLoader loader;
    assert(loader.add_file(path, prefix)); loader.set_wtype_override(GGML_TYPE_COUNT, rules); verify(loader, fixture, true);
    fixture.scales[0] *= 2; fixture.write(path);
    std::filesystem::last_write_time(path, std::filesystem::file_time_type::clock::now() + std::chrono::seconds(1));
    assert(failure(loader, fixture)); assert(loader.refresh_files()); verify(loader, fixture, true);
    // Invalid destinations must fail without writing past their allocation.
    auto* ctx = ggml_init({1024 * 1024, nullptr, false});
    std::map<std::string, ggml_tensor*> wrong{{prefix + fixture.module + ".weight", ggml_new_tensor_2d(ctx, GGML_TYPE_F16, fixture.columns, fixture.outputs - 1)}};
    assert(!load_params(loader, wrong, true));
    auto* backing = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, fixture.columns + 2, fixture.outputs);
    wrong.begin()->second = ggml_view_2d(ctx, backing, fixture.columns, fixture.outputs, (fixture.columns + 2) * 2, 0);
    assert(!load_params(loader, wrong, true));
    wrong.begin()->second = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, fixture.columns, fixture.outputs);
    assert(!load_params(loader, wrong, true)); ggml_free(ctx);
    // Separate logical scale sources are unsupported until dependency tracking
    // exists; indexed physical shards within one logical FileId remain valid.
    fixture.include_scale = false; fixture.write(path);
    ModelLoader cross_file; assert(cross_file.add_file(path, prefix));
    Fixture scale_file; scale_file.include_weight = false;
    const std::string scale_path = std::string(directory.path) + "/synthetic-scales.safetensors";
    scale_file.write(scale_path); assert(cross_file.add_file(scale_path, prefix));
    cross_file.set_wtype_override(GGML_TYPE_COUNT, rules); assert(failure(cross_file, fixture));
    // The same physical shards are valid when one safetensors index owns both.
    const std::string index_path = std::string(directory.path) + "/synthetic.safetensors.index.json";
    Json index = {{"weight_map", {
        {fixture.module + ".weight", "synthetic.safetensors"},
        {fixture.module + ".bias", "synthetic.safetensors"},
        {fixture.module + ".comfy_quant", "synthetic.safetensors"},
        {fixture.module + ".weight_scale", "synthetic-scales.safetensors"}
    }}};
    { std::ofstream file(index_path); file << index.dump(); assert(file.good()); }
    ModelLoader sharded; ModelLoader::FileId sharded_id;
    assert(sharded.add_file(index_path, prefix, &sharded_id));
    sharded.set_wtype_override(GGML_TYPE_COUNT, rules);
    fixture.scales = scale_file.scales;
    verify(sharded, fixture, true, sharded_id);
    scale_file.scales[0] *= 2; scale_file.write(scale_path); fixture.scales = scale_file.scales;
    std::filesystem::last_write_time(scale_path, std::filesystem::file_time_type::clock::now() + std::chrono::seconds(2));
    assert(failure(sharded, fixture)); assert(sharded.refresh_files());
    verify(sharded, fixture, true, sharded_id);
    unsetenv("KUNO_EXPERIMENTAL_QWEN_MLP_F16");
    std::printf("mlp_f16_loader_cases=%d invalid_cases=12 refresh=1 policy=1 destinations=3 cross_file=1 scoped_load=3 indexed_shards=1 status=passed\n", cases);
}
