#include "llama.h"
#include "llama-spec-prefill.h"
#include <cstdio>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model-path>\n", argv[0]);
        return 1;
    }

    const char * model_path = argv[1];
    
    printf("========================================\n");
    printf("Importance Scoring Debug Test\n");
    printf("========================================\n\n");
    
    // Load model
    llama_model_params model_params = llama_model_default_params();
    llama_model * model = llama_load_model_from_file(model_path, model_params);
    
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    
    // Create contexts
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 512;
    ctx_params.n_batch = 512;
    
    llama_context * ctx_base = llama_new_context_with_model(model, ctx_params);
    llama_context * ctx_spec = llama_new_context_with_model(model, ctx_params);
    
    llama_spec_prefill_context * sp_ctx = llama_spec_prefill_init(ctx_base, ctx_spec);
    
    // Test prompt
    std::vector<llama_token> prompt = {1, 450, 2043, 338, 263, 1925, 1426, 393, 508, 1925, 2058, 278, 17366, 11203};
    // "The world is a wonderful place where great adventures await those brave enough to seek them"
    
    printf("Test prompt: %zu tokens\n\n", prompt.size());
    
    // Generate lookahead to populate statistics
    std::vector<llama_token> lookahead(8);
    int n_lookahead = llama_spec_prefill_generate_lookahead(
        sp_ctx, prompt.data(), prompt.size(), lookahead.data(), lookahead.size()
    );
    
    printf("Generated %d lookahead tokens\n", n_lookahead);
    printf("Lookahead stats populated: %zu entries\n\n", sp_ctx->lookahead_stats.size());
    
    // Print lookahead statistics
    if (!sp_ctx->lookahead_stats.empty()) {
        printf("Lookahead Statistics:\n");
        printf("%-8s | %-12s | %-12s\n", "Pos", "Max Logit", "Entropy");
        printf("---------+--------------+--------------\n");
        
        for (const auto & stat : sp_ctx->lookahead_stats) {
            printf("%-8d | %12.4f | %12.4f\n", 
                   stat.position, stat.max_logit, stat.entropy);
        }
        printf("\n");
    }
    
    // Compute importance scores
    std::vector<float> importance;
    llama_spec_prefill_compute_importance(sp_ctx, prompt.data(), prompt.size(), importance);
    
    printf("Token Importance Scores:\n");
    printf("%-8s | %-12s\n", "Token #", "Importance");
    printf("---------+--------------\n");
    
    for (size_t i = 0; i < importance.size(); i++) {
        printf("%-8zu | %12.6f\n", i, importance[i]);
    }
    printf("\n");
    
    // Show distribution
    float min_imp = importance[0];
    float max_imp = importance[0];
    float sum_imp = 0.0f;
    
    for (float imp : importance) {
        if (imp < min_imp) min_imp = imp;
        if (imp > max_imp) max_imp = imp;
        sum_imp += imp;
    }
    
    float avg_imp = sum_imp / importance.size();
    
    printf("Statistics:\n");
    printf("  Min importance: %.6f\n", min_imp);
    printf("  Max importance: %.6f\n", max_imp);
    printf("  Avg importance: %.6f\n", avg_imp);
    printf("  Range: %.6f\n", max_imp - min_imp);
    printf("\n");
    
    // Cleanup
    llama_spec_prefill_free(sp_ctx);
    llama_free(ctx_base);
    llama_free(ctx_spec);
    llama_free_model(model);
    
    printf("✓ Test completed successfully\n");
    
    return 0;
}
