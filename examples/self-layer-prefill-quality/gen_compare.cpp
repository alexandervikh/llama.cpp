#include "llama.h"
#include "llama-self-layer-prefill.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static std::string generate(llama_context * ctx, const llama_vocab * vocab,
                            const std::vector<llama_token> & prompt, int n_gen,
                            bool skip_prompt_decode = false) {
    std::string result;
    
    // Decode prompt (skip if already decoded by prefill)
    if (!skip_prompt_decode) {
        llama_batch batch = llama_batch_get_one((llama_token *)prompt.data(), (int)prompt.size());
        if (llama_decode(ctx, batch) != 0) {
            return "[decode failed]";
        }
    }
    
    // Generate tokens
    for (int i = 0; i < n_gen; i++) {
        float * logits = llama_get_logits_ith(ctx, -1);
        if (!logits) break;
        
        // Greedy sampling
        int n_vocab = llama_vocab_n_tokens(vocab);
        llama_token best = 0;
        float best_logit = logits[0];
        for (int v = 1; v < n_vocab; v++) {
            if (logits[v] > best_logit) {
                best_logit = logits[v];
                best = v;
            }
        }
        
        // Check for EOS
        if (llama_vocab_is_eog(vocab, best)) break;
        
        // Decode token to text
        char buf[256];
        int len = llama_token_to_piece(vocab, best, buf, sizeof(buf), 0, true);
        if (len > 0) result.append(buf, len);
        
        // Decode next token
        llama_batch single = llama_batch_get_one(&best, 1);
        if (llama_decode(ctx, single) != 0) break;
    }
    
    return result;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [keep_ratio] [n_lookahead]\n", argv[0]);
        return 1;
    }
    
    const char * model_path = argv[1];
    float keep_ratio = (argc > 2) ? atof(argv[2]) : 0.5f;
    int n_lookahead = (argc > 3) ? atoi(argv[3]) : 0;
    
    // Load model
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;
    
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    
    const llama_vocab * vocab = llama_model_get_vocab(model);
    
    // Context params
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 2048;
    cparams.n_batch = 2048;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    
    // Test prompts
    std::vector<std::string> prompts = {
        "Write a haiku about the ocean:\n",
        "The capital of France is",
        "def fibonacci(n):\n    ",
        "Explain quantum computing in one sentence:",
    };
    
    printf("=== Generation Comparison ===\n");
    printf("Model: %s\n", model_path);
    printf("Keep ratio: %.2f\n", keep_ratio);
    printf("Look-ahead tokens: %d\n\n", n_lookahead);
    
    for (const auto & prompt_str : prompts) {
        printf("----------------------------------------\n");
        printf("PROMPT: %s\n", prompt_str.c_str());
        
        // Tokenize
        std::vector<llama_token> tokens(prompt_str.size() + 32);
        int n_tokens = llama_tokenize(vocab, prompt_str.c_str(), prompt_str.size(),
                                      tokens.data(), tokens.size(), true, true);
        if (n_tokens < 0) {
            printf("[tokenization failed]\n\n");
            continue;
        }
        tokens.resize(n_tokens);
        
        // Create context for baseline
        llama_context * ctx = llama_init_from_model(model, cparams);
        if (!ctx) {
            printf("[context creation failed]\n\n");
            continue;
        }
        
        // Baseline generation
        llama_memory_clear(llama_get_memory(ctx), false);
        std::string baseline = generate(ctx, vocab, tokens, 50, false);
        printf("\nBASELINE:\n%s%s\n", prompt_str.c_str(), baseline.c_str());
        
        // Self-layer prefill generation
        llama_memory_clear(llama_get_memory(ctx), false);
        
        llama_self_layer_prefill_params slp_params;
        slp_params.n_early_layers = 8;
        slp_params.keep_ratio = keep_ratio;
        slp_params.pool_kernel_size = 1;
        slp_params.use_chunking = false;
        slp_params.n_lookahead_tokens = n_lookahead;
        slp_params.lookahead_temp = 0.f;
        
        int n_kept = llama_self_layer_prefill_partial(ctx, tokens.data(), n_tokens, &slp_params);
        
        if (n_kept > 0) {
            // Skip prompt decode - prefill already decoded and filled KV cache
            std::string filtered = generate(ctx, vocab, tokens, 50, true);
            printf("\nFILTERED (kept %d/%d = %.0f%%):\n%s%s\n", 
                   n_kept, n_tokens, 100.0f * n_kept / n_tokens,
                   prompt_str.c_str(), filtered.c_str());
        } else {
            printf("\nFILTERED: [prefill failed: %d]\n", n_kept);
        }
        
        llama_free(ctx);
        printf("\n");
    }
    
    llama_model_free(model);
    return 0;
}
