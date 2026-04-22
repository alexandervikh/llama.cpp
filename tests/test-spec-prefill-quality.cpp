#include "llama.h"
#include "llama-spec-prefill.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

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

        size_t id_pos = line.find("\"id\":");
        size_t prompt_pos = line.find("\"prompt\":");

        if (id_pos != std::string::npos && prompt_pos != std::string::npos) {
            int id = std::stoi(line.substr(id_pos + 5));

            size_t start = line.find('"', prompt_pos + 9);
            size_t end = line.rfind('"');
            if (start != std::string::npos && end > start) {
                std::string prompt = line.substr(start + 1, end - start - 1);
                
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

static std::string greedy_decode(
    llama_context * ctx,
    const llama_vocab * vocab,
    int n_tokens_max,
    llama_token first_token,
    int pos_start
) {
    std::string output;

    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    llama_batch batch = llama_batch_init(1, 0, 1);

    llama_token new_token_id = first_token;

    for (int i = 0; i < n_tokens_max; i++) {
        batch.n_tokens      = 1;
        batch.token[0]      = new_token_id;
        batch.pos[0]        = (llama_pos)(pos_start + i);
        batch.n_seq_id[0]   = 1;
        batch.seq_id[0][0]  = 0;
        batch.logits[0]     = 1;

        int ret = llama_decode(ctx, batch);
        if (ret != 0) {
            fprintf(stderr, "decode failed at step %d, ret=%d, pos=%d\n", i, ret, pos_start + i);
            break;
        }

        llama_token sampled = llama_sampler_sample(smpl, ctx, -1);
        fprintf(stderr, "  [decode step %d] pos=%d sampled_token=%d\n", i, pos_start + i, sampled);

        if (llama_vocab_is_eog(vocab, sampled)) {
            fprintf(stderr, "  [decode step %d] EOS reached, stopping\n", i);
            break;
        }

        char buf[128];
        int n = llama_token_to_piece(vocab, sampled, buf, sizeof(buf), 0, true);
        if (n > 0) {
            output.append(std::string(buf, n));
        }

        new_token_id = sampled;
    }

    llama_batch_free(batch);
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
    (void)spec_model_path;
    printf("Running quality gate with keep_ratio=%.2f\n", keep_ratio);

    llama_model_params model_params = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(base_model_path, model_params);
    if (!model) {
        fprintf(stderr, "Failed to load model from %s\n", base_model_path);
        return;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 4096;
    ctx_params.n_batch = 2048;

    llama_context * ctx_base = llama_init_from_model(model, ctx_params);
    if (!ctx_base) {
        fprintf(stderr, "Failed to create base context\n");
        llama_model_free(model);
        return;
    }

    llama_context * ctx_spec = llama_init_from_model(model, ctx_params);
    if (!ctx_spec) {
        fprintf(stderr, "Failed to create spec context\n");
        llama_free(ctx_base);
        llama_model_free(model);
        return;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_spec_prefill_context * sp_ctx = llama_spec_prefill_init(ctx_base, ctx_spec);
    if (!sp_ctx) {
        fprintf(stderr, "Failed to init spec-prefill context\n");
        llama_free(ctx_spec);
        llama_free(ctx_base);
        llama_model_free(model);
        return;
    }

    std::ofstream clear_f(out_file, std::ios::trunc);
    clear_f.close();

    auto prompts = load_prompts(prompt_file);
    printf("Loaded %zu prompts\n", prompts.size());

    int n_lookahead = 8;
    int n_decode = 64;

    for (const auto & [id, prompt_str] : prompts) {
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

            llama_memory_seq_rm(llama_get_memory(ctx_base), 0, 0, -1);
        llama_memory_seq_rm(llama_get_memory(ctx_spec), 0, 0, -1);

        int n_kept = llama_spec_prefill(
            sp_ctx, prompt_tokens.data(), n_prompt, n_lookahead, keep_ratio
        );

        if (n_kept <= 0) {
            fprintf(stderr, "spec-prefill returned n_kept=%d for prompt %d, skipping decode\n", n_kept, id);
            std::string output = "[SPEC-PREFILL FAILED - n_kept=0]";
            write_json_output(out_file, id, keep_ratio, output, 0, n_prompt);
            continue;
        }

        printf("  [%d/%zu] keep_ratio=%.2f, n_kept=%d, n_total=%d\n",
               id + 1, prompts.size(), keep_ratio, n_kept, n_prompt);

        // Use the last kept token as the first decode input
        llama_token first_decode_token = prompt_tokens[n_kept - 1];
        fprintf(stderr, "  [DEBUG] first_decode_token=%d, n_kept=%d\n", first_decode_token, n_kept);
        std::string output = greedy_decode(ctx_base, vocab, n_decode, first_decode_token, n_kept);

        write_json_output(out_file, id, keep_ratio, output, n_kept, n_prompt);
    }

    llama_spec_prefill_free(sp_ctx);
    llama_free(ctx_spec);
    llama_free(ctx_base);
    llama_model_free(model);

    printf("Quality gate complete. Results in: %s\n", out_file);
}

int main(int argc, char ** argv) {
    float keep_ratio = 0.25f;
    const char * base_model_path = nullptr;
    const char * spec_model_path = nullptr;
    const char * prompt_file = nullptr;
    const char * out_file = nullptr;

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
