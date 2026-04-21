#include "llama.h"
#include "llama-spec-prefill.h"
#include "../common/common.h"
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <base-model> <spec-model> <dump-file>\n", argv[0]);
        return 1;
    }

    const char * base_model_path = argv[1];
    const char * spec_model_path = argv[2];
    const char * dump_file = argv[3];

    llama_model_params model_params = llama_model_default_params();
    llama_model * model_base = llama_model_load_from_file(base_model_path, model_params);
    llama_model * model_spec = llama_model_load_from_file(spec_model_path, model_params);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 4096;
    ctx_params.n_batch = 2048;
    llama_context * ctx_base = llama_init_from_model(model_base, ctx_params);
    llama_context * ctx_spec = llama_init_from_model(model_spec, ctx_params);

    llama_spec_prefill_context * sp_ctx = llama_spec_prefill_init(ctx_base, ctx_spec);
    llama_spec_prefill_set_dump_path(sp_ctx, dump_file);

    // Sample prompts for parity check
    std::vector<std::string> prompts = {
        "The capital of France is Paris.",
        "Quantum computing uses qubits to perform calculations.",
        "Llama.cpp is a C++ port of Meta's Llama model.",
        "Speculative prefill optimizes the prompt processing phase."
    };

    for (size_t i = 0; i < prompts.size(); i++) {
        const llama_vocab * vocab = llama_model_get_vocab(model_base);
        int n_prompt = -llama_tokenize(vocab, prompts[i].c_str(), prompts[i].size(), nullptr, 0, true, true);
        std::vector<llama_token> tokens(n_prompt);
        llama_tokenize(vocab, prompts[i].c_str(), prompts[i].size(), tokens.data(), tokens.size(), true, true);

        printf("Processing prompt %zu/%zu...\n", i + 1, prompts.size());
        llama_spec_prefill(sp_ctx, tokens.data(), n_prompt, 8, 0.25f);
    }

    printf("Traces dumped to %s\n", dump_file);

    llama_spec_prefill_free(sp_ctx);
    llama_free(ctx_base);
    llama_free(ctx_spec);
    llama_model_free(model_base);
    llama_model_free(model_spec);

    return 0;
}
