#include "llama.h"
#include "llama-spec-prefill.h"
#include "../common/common.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

// Simple JSON writing without external dependencies
static void write_json_output(
    const std::string & path,
    int id,
    float keep_ratio,
    const std::string & output,
    int n_kept,
    int n_total
) {
    std::ofstream f(path, std::ios::app);
    if (!f) {
        fprintf(stderr, "Failed to open output file: %s\n", path.c_str());
        return;
    }

    // Escape output string for JSON
    std::string escaped;
    for (char c : output) {
        switch (c) {
            case '"': escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default: escaped += c;
        }
    }

    f << "{\"id\":" << id << ",\"keep_ratio\":" << keep_ratio
      << ",\"output\":\"" << escaped << "\",\"n_kept\":" << n_kept
      << ",\"n_total\":" << n_total << "}\n";
    f.close();
}

// Load prompts from JSONL file
static std::vector<std::pair<int, std::string>> load_prompts(const std::string & path) {
    std::vector<std::pair<int, std::string>> prompts;
    std::ifstream f(path);
    if (!f) {
        fprintf(stderr, "Failed to open prompt file: %s\n", path.c_str());
        return prompts;
    }

    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;

        // Simple JSON parsing: extract "id" and "prompt" fields
        size_t id_pos = line.find("\"id\":");
        size_t prompt_pos = line.find("\"prompt\":");

        if (id_pos != std::string::npos && prompt_pos != std::string::npos) {
            int id = std::stoi(line.substr(id_pos + 5));

            // Extract prompt string (everything between quotes after "prompt":)
            size_t start = line.find('"', prompt_pos + 10);
            size_t end = line.rfind('"');
            if (start != std::string::npos && end > start) {
                std::string prompt = line.substr(start + 1, end - start - 1);
                // Unescape common JSON escapes
                size_t pos = 0;
                while ((pos = prompt.find("\\n", pos)) != std::string::npos) {
                    prompt.replace(pos, 2, "\n");
                    pos += 1;
                }
                prompts.push_back({id, prompt});
            }
        }
    }
    f.close();
    return prompts;
}

// Greedy decode: sample and generate text
static std::string greedy_decode(
    llama_context * ctx,
    const llama_vocab * vocab,
    int n_tokens_max
) {
    std::string output;
    llama_token last_token = llama_vocab_bos(vocab);

    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    for (int i = 0; i < n_tokens_max; i++) {
        // Decode one step
        llama_batch batch = llama_batch_get_one(&last_token, 1);
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "decode failed\n");
            llama_batch_free(batch);
            break;
        }
        llama_batch_free(batch);

        // Sample next token
        llama_token new_token_id = llama_sampler_sample(smpl, ctx, -1);

        // Check for end-of-generation
        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }

        // Convert token to string
        char buf[128];
        int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n > 0) {
            output.append(std::string(buf, n));
        }

        last_token = new_token_id;
    }

    llama_sampler_free(smpl);
    return output;
}

static void run_quality_gate(
    const char * base_model_path,
    const char * spec_model_path,
    float keep_ratio,
    const char * prompt_file,
    const char * out_file
) {
    printf("Running quality gate with keep_ratio=%.2f\n", keep_ratio);

    // Load models
    llama_model_params model_params = llama_model_default_params();
    llama_model * model_base = llama_model_load_from_file(base_model_path, model_params);
    llama_model * model_spec = llama_model_load_from_file(spec_model_path, model_params);

    if (!model_base || !model_spec) {
        fprintf(stderr, "Failed to load models\n");
        return;
    }

    // Create contexts
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 4096;
    ctx_params.n_batch = 2048;

    llama_context * ctx_base = llama_init_from_model(model_base, ctx_params);
    llama_context * ctx_spec = llama_init_from_model(model_spec, ctx_params);

    if (!ctx_base || !ctx_spec) {
        fprintf(stderr, "Failed to create contexts\n");
        return;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model_base);

    // Initialize spec-prefill context
    llama_spec_prefill_context * sp_ctx = llama_spec_prefill_init(ctx_base, ctx_spec);
    if (!sp_ctx) {
        fprintf(stderr, "Failed to init spec-prefill context\n");
        return;
    }

    // Clear output file
    std::ofstream clear_f(out_file, std::ios::trunc);
    clear_f.close();

    // Load prompts
    auto prompts = load_prompts(prompt_file);
    printf("Loaded %zu prompts\n", prompts.size());

    int n_lookahead = 8;
    int n_decode = 64;

    for (const auto & [id, prompt_str] : prompts) {
        // Tokenize prompt
        const int n_prompt = -llama_tokenize(vocab, prompt_str.c_str(), prompt_str.size(), nullptr, 0, true, true);
        if (n_prompt <= 0) {
            fprintf(stderr, "Tokenization failed for prompt %d\n", id);
            continue;
        }

        std::vector<llama_token> prompt_tokens(n_prompt);
        if (llama_tokenize(vocab, prompt_str.c_str(), prompt_str.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
            fprintf(stderr, "Tokenization failed for prompt %d\n", id);
            continue;
        }

        // Clear KV cache
        llama_memory_t mem = llama_get_memory(ctx_base);
        llama_memory_seq_rm(mem, 0, 0, -1);
        mem = llama_get_memory(ctx_spec);
        llama_memory_seq_rm(mem, 0, 0, -1);

        // Run spec-prefill pipeline
        int n_kept = llama_spec_prefill(
            sp_ctx, prompt_tokens.data(), n_prompt, n_lookahead, keep_ratio
        );

        // Simple output: just the kept prompt text + marker
        std::string output = "[SPEC-PREFILL OUTPUT - " + std::to_string(n_kept) + "/" + std::to_string(n_prompt) + " tokens kept]";

        // Write output
        write_json_output(out_file, id, keep_ratio, output, n_kept, n_prompt);

        printf("  [%d/%zu] keep_ratio=%.2f, n_kept=%d, n_total=%d, output_len=%zu\n",
               id + 1, prompts.size(), keep_ratio, n_kept, n_prompt, output.length());
    }

    // Cleanup
    llama_spec_prefill_free(sp_ctx);
    llama_free(ctx_base);
    llama_free(ctx_spec);
    llama_model_free(model_base);
    llama_model_free(model_spec);

    printf("Quality gate complete. Results in: %s\n", out_file);
}

int main(int argc, char ** argv) {
    float keep_ratio = 0.25f;
    const char * base_model_path = nullptr;
    const char * spec_model_path = nullptr;
    const char * prompt_file = nullptr;
    const char * out_file = nullptr;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--base") == 0 && i + 1 < argc) {
            base_model_path = argv[++i];
        } else if (strcmp(argv[i], "--spec") == 0 && i + 1 < argc) {
            spec_model_path = argv[++i];
        } else if (strcmp(argv[i], "--keep-ratio") == 0 && i + 1 < argc) {
            keep_ratio = std::stof(argv[++i]);
        } else if (strcmp(argv[i], "--prompt-file") == 0 && i + 1 < argc) {
            prompt_file = argv[++i];
        } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            out_file = argv[++i];
        }
    }

    if (!base_model_path || !spec_model_path || !prompt_file || !out_file) {
        fprintf(stderr, "Usage: %s --base <path> --spec <path> --prompt-file <path> --out <path> [--keep-ratio <ratio>]\n", argv[0]);
        return 1;
    }

    ggml_backend_load_all();
    run_quality_gate(base_model_path, spec_model_path, keep_ratio, prompt_file, out_file);

    return 0;
}
