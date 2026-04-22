#include "llama.h"
#include "llama-spec-prefill.h"
#include <cstdio>
#include <vector>
#include <cstring>

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <model>\n", argv[0]); return 1; }
    const char * model_path = argv[1];

    ggml_backend_load_all();

    printf("Loading model (first instance)...\n");
    llama_model_params mp = llama_model_default_params();
    llama_model * model1 = llama_model_load_from_file(model_path, mp);
    if (!model1) { fprintf(stderr, "Failed to load model\n"); return 1; }

    printf("Loading model (second instance)...\n");
    llama_model * model2 = llama_model_load_from_file(model_path, mp);
    if (!model2) { fprintf(stderr, "Failed to load model 2\n"); llama_model_free(model1); return 1; }

    printf("Creating context 1...\n");
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = 2048;
    llama_context * ctx1 = llama_init_from_model(model1, cp);
    if (!ctx1) { fprintf(stderr, "Failed ctx1\n"); llama_model_free(model1); llama_model_free(model2); return 1; }

    printf("Creating context 2...\n");
    llama_context * ctx2 = llama_init_from_model(model2, cp);
    if (!ctx2) { fprintf(stderr, "Failed ctx2\n"); llama_free(ctx1); llama_model_free(model1); llama_model_free(model2); return 1; }

    printf("Init spec-prefill...\n");
    auto sp = llama_spec_prefill_init(ctx1, ctx2);
    if (!sp) { fprintf(stderr, "Failed sp_init\n"); }

    const char * prompt = "Explain machine learning.";
    int n_prompt = -llama_tokenize(llama_model_get_vocab(model1), prompt, strlen(prompt), nullptr, 0, true, true);
    printf("Prompt tokens: %d\n", n_prompt);

    if (n_prompt > 0) {
        std::vector<llama_token> tokens(n_prompt);
        llama_tokenize(llama_model_get_vocab(model1), prompt, strlen(prompt), tokens.data(), n_prompt, true, true);

        llama_memory_seq_rm(llama_get_memory(ctx1), 0, 0, -1);
        llama_memory_seq_rm(llama_get_memory(ctx2), 0, 0, -1);

        printf("Running spec-prefill...\n");
        int n_kept = llama_spec_prefill(sp, tokens.data(), n_prompt, 8, 0.5f);
        printf("n_kept: %d\n", n_kept);
    }

    printf("Cleaning up...\n");
    if (sp) llama_spec_prefill_free(sp);
    llama_free(ctx1);
    llama_free(ctx2);
    llama_model_free(model1);
    llama_model_free(model2);
    printf("Done!\n");
    return 0;
}
