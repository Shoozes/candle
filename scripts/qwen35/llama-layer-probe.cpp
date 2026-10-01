// Artifact-specific diagnostic for the pinned llama.cpp b11026 CPU graph.
#include <windows.h>

#include "llama.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template <typename T>
T symbol(HMODULE module, const char * name) {
    auto address = GetProcAddress(module, name);
    if (!address) {
        throw std::runtime_error(std::string("missing DLL symbol: ") + name);
    }
    return reinterpret_cast<T>(address);
}

using TensorGet = decltype(&ggml_backend_tensor_get);

struct Capture {
    TensorGet get = nullptr;
    std::filesystem::path output;
    int pass = 0;
    int count = 0;
    int token = -1;
    bool timeline = false;
};

bool capture_tensor(ggml_tensor * tensor, bool ask, void * user_data) {
    auto & capture = *static_cast<Capture *>(user_data);
    const std::string name(tensor->name);
    if (name == "model.input_embed" && !ask && capture.timeline) {
        capture.token++;
    }
    const bool timeline = capture.timeline &&
                          (name == "model.input_embed" ||
                           name.rfind("l_out-", 0) == 0 || name == "Kcur-19");
    if (capture.pass != 2 && !timeline) {
        return false;
    }
    const bool layer = name.rfind("l_out-", 0) == 0;
    const bool first = name == "model.input_embed" || name == "attn_norm-0" ||
                       name == "attn_post_norm-0" || name == "ffn_gate-0" ||
                       name == "ffn_up-0" || name == "ffn_swiglu-0" ||
                       name == "ffn_out-0" || name == "post_ffn-0" ||
                       name == "linear_attn_qkv_mixed-0" || name == "conv_output_silu-0" ||
                       name == "final_output-0" || name == "linear_attn_out-0" ||
                       name == "attn_residual-0" || name == "attn_norm-2" ||
                       name == "linear_attn_qkv_mixed-2" || name == "conv_output_silu-2" ||
                       name == "final_output-2" || name == "linear_attn_out-2" ||
                       name == "attn_residual-2" || name == "attn_norm-1" ||
                       name == "linear_attn_qkv_mixed-1" || name == "conv_output_silu-1" ||
                       name == "final_output-1" || name == "linear_attn_out-1" ||
                       name == "attn_residual-1" || name == "attn_norm-18" ||
                       name == "linear_attn_qkv_mixed-18" ||
                       name == "conv_output_raw-18" ||
                       name == "conv_output_silu-18" ||
                       name == "q_conv_predelta-18" ||
                       name == "k_conv_predelta-18" ||
                       name == "v_conv_predelta-18" ||
                       name == "a_softplus-18" || name == "gate-18" ||
                       name == "beta_sigmoid-18" ||
                       name == "q_in-18" || name == "k_in-18" ||
                       name == "attn_output-18" ||
                       name == "final_output-18" ||
                       name == "linear_attn_out-18" ||
                       name == "attn_residual-18" ||
                       name == "attn_post_norm-18" ||
                       name == "ffn_gate-18" || name == "ffn_up-18" ||
                       name == "ffn_swiglu-18" || name == "ffn_out-18" ||
                       name == "attn_norm-19" ||
                       name == "Qcur_full-19" || name == "Qcur_normed-19" ||
                       name == "Kcur_normed-19" || name == "gate_reshaped-19" ||
                       name == "Qcur-19" || name == "Kcur-19" || name == "Vcur-19" ||
                       name == "kq-19" || name == "kq_soft_max-19" ||
                       name == "kqv-19" || name == "attn_pregate-19" ||
                       name == "gate_sigmoid-19" ||
                       name == "attn_gated-19" || name == "attn_output-19" ||
                       name == "attn_residual-19" || name == "attn_post_norm-19" ||
                       name == "ffn_gate-19" || name == "ffn_up-19" ||
                       name == "ffn_swiglu-19" || name == "ffn_out-19";
    const bool target_27 = name == "attn_norm-27" ||
                           name == "Qcur_full-27" || name == "Qcur_normed-27" ||
                           name == "Kcur_normed-27" || name == "Qcur-27" ||
                           name == "Kcur-27" || name == "Vcur-27" ||
                           name == "kq-27" || name == "kq_soft_max-27" ||
                           name == "attn_pregate-27" || name == "gate_sigmoid-27" ||
                           name == "attn_gated-27" || name == "attn_output-27" ||
                           name == "attn_residual-27" || name == "ffn_out-27";
    const bool final = name == "result_norm";
    if (!layer && !first && !target_27 && !final) {
        return false;
    }
    if (tensor->type != GGML_TYPE_F32 || tensor->ne[0] <= 0 || tensor->ne[0] > 10000 ||
        tensor->ne[1] <= 0 || tensor->nb[0] != sizeof(float)) {
        return false;
    }
    size_t length = 1;
    for (int dimension = 0; dimension < 4; ++dimension) {
        if (tensor->ne[dimension] <= 0 || length > 10000 / static_cast<size_t>(tensor->ne[dimension])) {
            return false;
        }
        length *= static_cast<size_t>(tensor->ne[dimension]);
    }
    if (ask) {
        return true;
    }
    std::vector<float> values(length);
    size_t index = 0;
    for (int64_t i3 = 0; i3 < tensor->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < tensor->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < tensor->ne[1]; ++i1) {
                const size_t offset = static_cast<size_t>(i1) * tensor->nb[1] +
                                      static_cast<size_t>(i2) * tensor->nb[2] +
                                      static_cast<size_t>(i3) * tensor->nb[3];
                capture.get(tensor, values.data() + index, offset,
                            static_cast<size_t>(tensor->ne[0]) * sizeof(float));
                index += static_cast<size_t>(tensor->ne[0]);
            }
        }
    }
    const std::string filename = timeline
        ? "timeline-" + name + "-" + std::to_string(capture.token) + ".bin"
        : name + ".bin";
    std::ofstream output(capture.output / filename, std::ios::binary);
    if (!output) {
        throw std::runtime_error("cannot write intermediate tensor");
    }
    output.write(reinterpret_cast<const char *>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(float)));
    double sum = 0.0;
    double squares = 0.0;
    for (float value : values) {
        sum += value;
        squares += static_cast<double>(value) * value;
    }
    std::cout << name << " n=" << length << " sum=" << sum << " l2=" << std::sqrt(squares)
              << " first=" << values[0] << "," << values[1] << "," << values[2]
              << "," << values[3] << '\n';
    capture.count++;
    return true;
}

std::vector<llama_token> prompt_tokens(const std::filesystem::path & fixture) {
    std::ifstream input(fixture);
    if (!input) {
        throw std::runtime_error("cannot open fixture");
    }
    const std::string json((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    const auto key = json.find("\"prompt_token_ids\"");
    const auto left = json.find('[', key);
    const auto right = json.find(']', left);
    if (key == std::string::npos || left == std::string::npos || right == std::string::npos) {
        throw std::runtime_error("prompt_token_ids absent");
    }
    std::vector<llama_token> tokens;
    size_t cursor = left + 1;
    while (cursor < right) {
        if (json[cursor] >= '0' && json[cursor] <= '9') {
            size_t consumed = 0;
            const auto value = std::stoll(json.substr(cursor, right - cursor), &consumed);
            if (value < 0 || value > INT32_MAX) {
                throw std::runtime_error("invalid prompt token ID");
            }
            tokens.push_back(static_cast<llama_token>(value));
            cursor += consumed;
        } else {
            cursor++;
        }
    }
    if (tokens.size() != 49) {
        throw std::runtime_error("expected 49 fixture prompt tokens");
    }
    return tokens;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc < 4 || argc > 6) {
            throw std::runtime_error("usage: llama-layer-probe MODEL FIXTURE OUTPUT_DIR [UBATCH] [TOKENS]");
        }
        const int ubatch = argc >= 5 ? std::stoi(argv[4]) : 32;
        const int token_count = argc == 6 ? std::stoi(argv[5]) : 49;
        if (ubatch != 1 && ubatch != 32) {
            throw std::runtime_error("UBATCH must be 1 or 32");
        }
        if (token_count < 1 || token_count > 49) {
            throw std::runtime_error("TOKENS must be between 1 and 49");
        }
        const std::filesystem::path output_dir(argv[3]);
        std::filesystem::create_directories(output_dir);
        SetDllDirectoryW(L"C:\\llamacpp\\tools-b11026");
        HMODULE llama = LoadLibraryW(L"C:\\llamacpp\\tools-b11026\\llama.dll");
        HMODULE ggml = LoadLibraryW(L"C:\\llamacpp\\tools-b11026\\ggml-base.dll");
        HMODULE ggml_loader = LoadLibraryW(L"C:\\llamacpp\\tools-b11026\\ggml.dll");
        if (!llama || !ggml || !ggml_loader) {
            throw std::runtime_error("pinned llama/ggml DLL load failed");
        }
        auto backend_init = symbol<decltype(&llama_backend_init)>(llama, "llama_backend_init");
        auto backend_free = symbol<decltype(&llama_backend_free)>(llama, "llama_backend_free");
        auto model_defaults = symbol<decltype(&llama_model_default_params)>(llama, "llama_model_default_params");
        auto context_defaults = symbol<decltype(&llama_context_default_params)>(llama, "llama_context_default_params");
        auto model_load = symbol<decltype(&llama_model_load_from_file)>(llama, "llama_model_load_from_file");
        auto model_free = symbol<decltype(&llama_model_free)>(llama, "llama_model_free");
        auto context_init = symbol<decltype(&llama_init_from_model)>(llama, "llama_init_from_model");
        auto context_free = symbol<decltype(&llama_free)>(llama, "llama_free");
        auto batch_one = symbol<decltype(&llama_batch_get_one)>(llama, "llama_batch_get_one");
        auto decode = symbol<decltype(&llama_decode)>(llama, "llama_decode");
        auto tensor_get = symbol<TensorGet>(ggml, "ggml_backend_tensor_get");
        auto backend_load_all = symbol<decltype(&ggml_backend_load_all_from_path)>(ggml_loader, "ggml_backend_load_all_from_path");

        const auto tokens = prompt_tokens(argv[2]);
        backend_load_all("C:\\llamacpp\\tools-b11026");
        backend_init();
        auto model_params = model_defaults();
        model_params.n_gpu_layers = 0;
        auto * model = model_load(argv[1], model_params);
        if (!model) {
            throw std::runtime_error("llama model load failed");
        }
        Capture capture{tensor_get, output_dir, 0, 0, -1, ubatch == 1};
        auto context_params = context_defaults();
        context_params.n_ctx = 256;
        context_params.n_batch = 32;
        context_params.n_ubatch = ubatch;
        context_params.n_seq_max = 1;
        context_params.n_threads = 4;
        context_params.n_threads_batch = 4;
        context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        context_params.type_k = GGML_TYPE_F32;
        context_params.type_v = GGML_TYPE_F32;
        context_params.offload_kqv = false;
        context_params.op_offload = false;
        context_params.cb_eval = capture_tensor;
        context_params.cb_eval_user_data = &capture;
        auto * context = context_init(model, context_params);
        if (!context) {
            throw std::runtime_error("llama context init failed");
        }
        const int first_count = token_count < 32 ? token_count : 32;
        auto first = batch_one(const_cast<llama_token *>(tokens.data()), first_count);
        capture.pass = token_count <= 32 ? 2 : 1;
        if (decode(context, first) != 0) {
            throw std::runtime_error("llama first prompt chunk failed");
        }
        if (token_count > 32) {
            auto second = batch_one(const_cast<llama_token *>(tokens.data() + 32), token_count - 32);
            capture.pass = 2;
            if (decode(context, second) != 0) {
                throw std::runtime_error("llama second prompt chunk failed");
            }
        }
        std::cout << "captured=" << capture.count << '\n';
        context_free(context);
        model_free(model);
        backend_free();
        return capture.count >= 32 ? 0 : 1;
    } catch (const std::exception & error) {
        std::cerr << "llama-layer-probe: " << error.what() << '\n';
        return 1;
    }
}
