// Finite CPU teacher-forced reference probe; built against the locked llama.cpp.
#include "llama.h"
#include "json.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;
struct BackendGuard {
    BackendGuard() { llama_backend_init(); }
    ~BackendGuard() { llama_backend_free(); }
};
static std::string text_file(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open input file");
    std::ostringstream bytes; bytes << in.rdbuf(); return bytes.str();
}
int main(int argc, char **argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--version") {
            std::cout << "candle-lfm2-moe-reference 582eab718608b99792286729890aac1675547c29\n";
            return 0;
        }
        if (argc != 5) throw std::runtime_error("usage: probe MODEL PROMPT_UTF8 FORCED_JSON PROBE_IDS_JSON");
        const std::string prompt = text_file(argv[2]);
        const auto forced = json::parse(text_file(argv[3])).get<std::vector<llama_token>>();
        const auto probes = json::parse(text_file(argv[4])).get<std::vector<llama_token>>();
        if (prompt.empty() || prompt.size() > 49152 || forced.empty() || forced.size() > 32 || probes.empty() || probes.size() > 64)
            throw std::runtime_error("reference input exceeds frozen bounds");
        BackendGuard backend;
        auto mp = llama_model_default_params(); mp.n_gpu_layers = 0;
        mp.load_mode = LLAMA_LOAD_MODE_MMAP;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
        if (!model) throw std::runtime_error("model load failed");
        const auto *vocab = llama_model_get_vocab(model.get());
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        for (const auto *ids : {&forced, &probes}) for (auto id : *ids)
            if (id < 0 || id >= n_vocab) throw std::runtime_error("token ID outside vocabulary");
        std::vector<llama_token> tokens(prompt.size() + 16);
        auto n = llama_tokenize(vocab, prompt.data(), static_cast<int32_t>(prompt.size()), tokens.data(), static_cast<int32_t>(tokens.size()), false, true);
        if (n <= 0 || n + forced.size() > 12288) throw std::runtime_error("tokenization/context bound failed");
        tokens.resize(n);
        auto cp = llama_context_default_params(); cp.n_ctx = 12288; cp.n_batch = 128; cp.n_ubatch = 128;
        cp.n_threads = 2; cp.n_threads_batch = 2; cp.type_k = GGML_TYPE_F32; cp.type_v = GGML_TYPE_F32;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED; cp.offload_kqv = false;
        std::unique_ptr<llama_context, decltype(&llama_free)> context(llama_init_from_model(model.get(), cp), llama_free);
        if (!context) throw std::runtime_error("context creation failed");
        for (size_t i = 0; i < tokens.size(); i += 128) {
            auto batch = llama_batch_get_one(tokens.data() + i, static_cast<int32_t>(std::min(size_t(128), tokens.size() - i)));
            if (llama_decode(context.get(), batch) != 0) throw std::runtime_error("prefill failed");
        }
        json top1 = json::array(), values = json::array();
        for (size_t i = 0; i < forced.size(); ++i) {
            const auto *logits = llama_get_logits_ith(context.get(), -1);
            if (!logits) throw std::runtime_error("missing logits");
            for (int32_t id = 0; id < n_vocab; ++id) if (!std::isfinite(logits[id])) throw std::runtime_error("nonfinite logits");
            top1.push_back(std::max_element(logits, logits + n_vocab) - logits);
            json row = json::array(); for (auto id : probes) row.push_back(logits[id]); values.push_back(row);
            if (i + 1 < forced.size()) {
                auto token = forced[i]; auto batch = llama_batch_get_one(&token, 1);
                if (llama_decode(context.get(), batch) != 0) throw std::runtime_error("decode failed");
            }
        }
        const json result = {
            {"reference_revision", "582eab718608b99792286729890aac1675547c29"},
            {"teacher_forced", true},
            {"tokens", forced},
            {"input_token_ids", tokens},
            {"probe_top1", top1},
            {"probe_token_ids", probes},
            {"probe_logits", values},
        };
        std::cout << result.dump() << '\n';
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
