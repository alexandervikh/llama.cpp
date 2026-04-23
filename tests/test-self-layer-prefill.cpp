// Self-layer prefill smoke test (requires --model <gguf>).
#include "llama.h"
#include "llama-self-layer-prefill.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], "--model") == 0) {
            model_path = argv[++i];
        }
    }
    if (!model_path) {
        printf("%s: skip (pass --model <gguf> to run)\n", argv[0]);
        return 0;
    }

    llama_model_params mp = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) {
        fprintf(stderr, "load model failed\n");
        return 1;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 1024;
    cp.n_batch = 512;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "init context failed\n");
        llama_model_free(model);
        return 1;
    }

    const int n_prompt = 128;
    std::vector<llama_token> toks((size_t) n_prompt);
    for (int i = 0; i < n_prompt; i++) {
        toks[(size_t) i] = 1000 + (i % 100);
    }

    llama_self_layer_prefill_params P;
    P.n_early_layers = std::max(1, (int) llama_model_n_layer(model) / 4);
    P.keep_ratio = 0.25f;
    P.pool_kernel_size = 13;

    std::vector<float> shallow, deep, full;
    float p_sf = 0, p_sd = 0;
    int r = llama_self_layer_prefill_attention_profiles(
        ctx, toks.data(), n_prompt, &P, shallow, deep, full, &p_sf, &p_sd);
    if (r != 0) {
        fprintf(stderr, "attention_profiles returned %d\n", r);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    if (shallow.size() != (size_t) n_prompt || full.size() != (size_t) n_prompt) {
        fprintf(stderr, "bad score lengths\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    const int n_kept = llama_self_layer_prefill(ctx, toks.data(), n_prompt, &P);
    if (n_kept < 1 || n_kept > n_prompt) {
        fprintf(stderr, "prefill returned %d\n", n_kept);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    printf("OK pearson(shallow,full)=%.4f pearson(shallow,deep)=%.4f n_kept=%d\n", p_sf, p_sd, n_kept);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
