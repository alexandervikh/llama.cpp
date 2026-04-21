#include "llama.h"
#include "llama-spec-prefill.h"
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <algorithm>

struct Args {
    std::string model_path;
    std::string spec_model_path;
    float keep_ratio = 0.25f;
    int lookahead = 8;
    int pool_kernel = 13;
    int chunk_size = 32;
    std::string prompt_file;
    std::string out_file;
};

static void print_usage() {
    std::cout << "Usage: llama-spec-prefill-run "
              << "--model <path> --spec-model <path> --prompt-file <path> --out <path> "
              << "[--keep-ratio <ratio>] [--lookahead <n>] [--pool <n>] [--chunk-size <n>]\n";
}

static std::string greedy_decode(llama_context * ctx, const llama_vocab * vocab, int n_tokens_max) {
    std::string output;
    llama_token last_token = llama_vocab_bos(vocab);

    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    for (int i = 0; i < n_tokens_max; i++) {
        llama_batch batch = llama_batch_get_one(&last_token, 1);
        if (llama_decode(ctx, batch) != 0) break;

        llama_token new_token_id = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, new_token_id)) break;

        char buf[128];
        int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n > 0) output.append(std::string(buf, n));
        last_token = new_token_id;
    }

    llama_sampler_free(smpl);
    return output;
}

int main(int argc, char ** argv) {
    Args args;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) args.model_path = argv[++i];
        else if (arg == "--spec-model" && i + 1 < argc) args.spec_model_path = argv[++i];
        else if (arg == "--keep-ratio" && i + 1 < argc) args.keep_ratio = std::stof(argv[++i]);
        else if (arg == "--lookahead" && i + 1 < argc) args.lookahead = std::stoi(argv[++i]);
        else if (arg == "--pool" && i + 1 < argc) args.pool_kernel = std::stoi(argv[++i]);
        else if (arg == "--chunk-size" && i + 1 < argc) args.chunk_size = std::stoi(argv[++i]);
        else if (arg == "--prompt-file" && i + 1 < argc) args.prompt_file = argv[++i];
        else if (arg == "--out" && i + 1 < argc) args.out_file = argv[++i];
        else {
            print_usage();
            return 1;
        }
    }

    if (args.model_path.empty() || args.spec_model_path.empty() || args.prompt_file.empty() || args.out_file.empty()) {
        print_usage();
        return 1;
    }

    llama_model_params model_params = llama_model_default_params();
    llama_model * model_base = llama_model_load_from_file(args.model_path.c_str(), model_params);
    llama_model * model_spec = llama_model_load_from_file(args.spec_model_path.c_str(), model_params);

    if (!model_base || !model_spec) {
        std::cerr << "Failed to load models\n";
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 8192;
    ctx_params.n_batch = 2048;
    llama_context * ctx_base = llama_init_from_model(model_base, ctx_params);
    llama_context * ctx_spec = llama_init_from_model(model_spec, ctx_params);

    if (!ctx_base || !ctx_spec) {
        std::cerr << "Failed to create contexts\n";
        return 1;
    }

    llama_spec_prefill_params sp_params;
    sp_params.keep_ratio = args.keep_ratio;
    sp_params.n_lookahead = args.lookahead;
    sp_params.pool_kernel_size = args.pool_kernel;
    sp_params.use_chunking = true;
    sp_params.chunk_size = args.chunk_size;

    llama_spec_prefill_context * sp_ctx = llama_spec_prefill_init_with_params(ctx_base, ctx_spec, sp_params);
    const llama_vocab * vocab = llama_model_get_vocab(model_base);

    std::ifstream infile(args.prompt_file);
    std::ofstream outfile(args.out_file);
    std::string line;
    int prompt_id = 0;

    while (std::getline(infile, line)) {
        if (line.empty()) continue;

        // Simple JSONL prompt extraction: {"id": N, "prompt": "...", ...}
        // Handle both "prompt":"..." and "prompt": "..." (with space)
        size_t key_pos = line.find("\"prompt\":");
        if (key_pos == std::string::npos) continue;
        size_t start = line.find('"', key_pos + 9);  // first " after "prompt":
        if (start == std::string::npos) continue;
        start++;  // skip opening quote

        // Find closing quote, accounting for escaped quotes
        size_t end = start;
        while (end < line.size()) {
            if (line[end] == '\\') { end += 2; continue; }
            if (line[end] == '"') break;
            end++;
        }
        if (end >= line.size()) continue;
        std::string prompt_str = line.substr(start, end - start);
        // Unescape \n
        for (size_t p = 0; (p = prompt_str.find("\\n", p)) != std::string::npos; p++)
            prompt_str.replace(p, 2, "\n");

        // Tokenize
        int n_prompt = -llama_tokenize(vocab, prompt_str.c_str(), prompt_str.size(), nullptr, 0, true, true);
        std::vector<llama_token> tokens(n_prompt);
        llama_tokenize(vocab, prompt_str.c_str(), prompt_str.size(), tokens.data(), tokens.size(), true, true);

        // Reset KV cache fully between prompts
        llama_memory_clear(llama_get_memory(ctx_base), false);
        llama_memory_clear(llama_get_memory(ctx_spec), false);

        auto t_start = std::chrono::high_resolution_clock::now();
        int n_kept = llama_spec_prefill(sp_ctx, tokens.data(), n_prompt, args.lookahead, args.keep_ratio);
        auto t_end = std::chrono::high_resolution_clock::now();

        double ttft_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        std::string output = greedy_decode(ctx_base, vocab, 64);

        // JSON-escape output string
        std::string esc;
        for (char c : output) {
            switch (c) {
                case '"':  esc += "\\\""; break;
                case '\\': esc += "\\\\"; break;
                case '\n': esc += "\\n";  break;
                case '\r': esc += "\\r";  break;
                case '\t': esc += "\\t";  break;
                default:   esc += c;
            }
        }
        outfile << "{\"id\":" << prompt_id++
                << ",\"output\":\"" << esc
                << "\",\"n_kept\":" << n_kept
                << ",\"n_total\":" << n_prompt
                << ",\"ttft_ms\":" << std::fixed << std::setprecision(2) << ttft_ms << "}\n";
    }

    llama_spec_prefill_free(sp_ctx);
    llama_free(ctx_base);
    llama_free(ctx_spec);
    llama_model_free(model_base);
    llama_model_free(model_spec);

    return 0;
}
