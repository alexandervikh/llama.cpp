#include "llama.h"
#include "llama-spec-prefill.h"
#include "../common/common.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ========================================
// Test Functions
// ========================================

static bool test_dual_model_loading(const char * model_path) {
    printf("[1] Running test_dual_model_loading...\n");
    printf("  Testing dual model initialization...\n");
    
    // Load model twice (simulating base and spec models)
    llama_model_params model_params = llama_model_default_params();
    llama_model * model_base = llama_load_model_from_file(model_path, model_params);
    llama_model * model_spec = llama_load_model_from_file(model_path, model_params);
    
    if (!model_base || !model_spec) {
        printf("  ✗ FAILED: Could not load models\n");
        return false;
    }
    
    // Create contexts
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 512;
    ctx_params.n_batch = 512;
    
    llama_context * ctx_base = llama_new_context_with_model(model_base, ctx_params);
    llama_context * ctx_spec = llama_new_context_with_model(model_spec, ctx_params);
    
    if (!ctx_base || !ctx_spec) {
        printf("  ✗ FAILED: Could not create contexts\n");
        if (model_base) llama_free_model(model_base);
        if (model_spec) llama_free_model(model_spec);
        return false;
    }
    
    // Initialize spec prefill context
    llama_spec_prefill_context * sp_ctx = llama_spec_prefill_init(ctx_base, ctx_spec);
    
    if (!sp_ctx) {
        printf("  ✗ FAILED: Could not initialize spec prefill context\n");
        llama_free(ctx_base);
        llama_free(ctx_spec);
        llama_free_model(model_base);
        llama_free_model(model_spec);
        return false;
    }
    
    printf("  ✓ Dual model loading successful\n");
    
    // Cleanup
    llama_spec_prefill_free(sp_ctx);
    llama_free(ctx_base);
    llama_free(ctx_spec);
    llama_free_model(model_base);
    llama_free_model(model_spec);
    
    printf("  ✓ PASSED\n\n");
    return true;
}

static bool test_spec_model_lookahead_generation(const char * model_path) {
    printf("[2] Running test_spec_model_lookahead_generation...\n");
    printf("  Testing lookahead generation...\n");
    
    llama_model_params model_params = llama_model_default_params();
    llama_model * model_base = llama_load_model_from_file(model_path, model_params);
    llama_model * model_spec = llama_load_model_from_file(model_path, model_params);
    
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 512;
    ctx_params.n_batch = 512;
    
    llama_context * ctx_base = llama_new_context_with_model(model_base, ctx_params);
    llama_context * ctx_spec = llama_new_context_with_model(model_spec, ctx_params);
    
    llama_spec_prefill_context * sp_ctx = llama_spec_prefill_init(ctx_base, ctx_spec);
    
    // Test prompt
    llama_token prompt[] = {1, 450, 2043, 338};  // "The world is"
    int n_prompt = 4;
    int n_lookahead = 3;
    
    std::vector<llama_token> lookahead_tokens(n_lookahead);
    int n_generated = llama_spec_prefill_generate_lookahead(
        sp_ctx, prompt, n_prompt, lookahead_tokens.data(), n_lookahead
    );
    
    bool success = (n_generated == n_lookahead);
    
    if (success) {
        printf("  ✓ Generated %d lookahead tokens\n", n_generated);
        printf("  ✓ PASSED\n\n");
    } else {
        printf("  ✗ FAILED: Expected %d tokens, got %d\n", n_lookahead, n_generated);
    }
    
    llama_spec_prefill_free(sp_ctx);
    llama_free(ctx_base);
    llama_free(ctx_spec);
    llama_free_model(model_base);
    llama_free_model(model_spec);
    
    return success;
}

static bool test_attention_score_extraction(const char * model_path) {
    printf("[3] Running test_attention_score_extraction...\n");
    printf("  Testing Q/K tensor extraction...\n");
    
    llama_model_params model_params = llama_model_default_params();
    llama_model * model_base = llama_load_model_from_file(model_path, model_params);
    llama_model * model_spec = llama_load_model_from_file(model_path, model_params);
    
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 512;
    ctx_params.n_batch = 512;
    
    llama_context * ctx_base = llama_new_context_with_model(model_base, ctx_params);
    llama_context * ctx_spec = llama_new_context_with_model(model_spec, ctx_params);
    
    llama_spec_prefill_context * sp_ctx = llama_spec_prefill_init(ctx_base, ctx_spec);
    
    llama_token lookahead[] = {100, 200, 300};
    int n_lookahead = 3;
    
    int result = llama_spec_prefill_extract_qk(sp_ctx, lookahead, n_lookahead);
    
    bool success = (result == 0);
    
    if (success) {
        printf("  ✓ Q/K extraction successful (simplified for POC)\n");
        printf("  ✓ PASSED\n\n");
    } else {
        printf("  ✗ FAILED\n\n");
    }
    
    llama_spec_prefill_free(sp_ctx);
    llama_free(ctx_base);
    llama_free(ctx_spec);
    llama_free_model(model_base);
    llama_free_model(model_spec);
    
    return success;
}

static bool test_attention_computation(const char * model_path) {
    printf("[4] Running test_attention_computation...\n");
    printf("  Testing attention computation...\n");
    
    llama_model_params model_params = llama_model_default_params();
    llama_model * model_base = llama_load_model_from_file(model_path, model_params);
    llama_model * model_spec = llama_load_model_from_file(model_path, model_params);
    
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 512;
    ctx_params.n_batch = 512;
    
    llama_context * ctx_base = llama_new_context_with_model(model_base, ctx_params);
    llama_context * ctx_spec = llama_new_context_with_model(model_spec, ctx_params);
    
    llama_spec_prefill_context * sp_ctx = llama_spec_prefill_init(ctx_base, ctx_spec);
    
    llama_token prompt[] = {1, 450, 2043, 338};
    int n_prompt = 4;
    llama_token lookahead[] = {100, 200, 300};
    int n_lookahead = 3;
    
    int result = llama_spec_prefill_compute_attention(
        sp_ctx, prompt, n_prompt, lookahead, n_lookahead
    );
    
    bool success = (result == 0);
    
    if (success) {
        printf("  ✓ Attention computation successful (simplified for POC)\n");
        printf("  ✓ PASSED\n\n");
    } else {
        printf("  ✗ FAILED\n\n");
    }
    
    llama_spec_prefill_free(sp_ctx);
    llama_free(ctx_base);
    llama_free(ctx_spec);
    llama_free_model(model_base);
    llama_free_model(model_spec);
    
    return success;
}

static bool test_token_importance_aggregation(const char * model_path) {
    printf("[5] Running test_token_importance_aggregation...\n");
    printf("  Testing token importance aggregation...\n");
    
    llama_model_params model_params = llama_model_default_params();
    llama_model * model_base = llama_load_model_from_file(model_path, model_params);
    llama_model * model_spec = llama_load_model_from_file(model_path, model_params);
    
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 512;
    ctx_params.n_batch = 512;
    
    llama_context * ctx_base = llama_new_context_with_model(model_base, ctx_params);
    llama_context * ctx_spec = llama_new_context_with_model(model_spec, ctx_params);
    
    llama_spec_prefill_context * sp_ctx = llama_spec_prefill_init(ctx_base, ctx_spec);
    
    llama_token prompt[] = {1, 450, 2043, 338, 263, 1407, 2058, 716, 286, 278};
    int n_prompt = 10;
    
    std::vector<float> token_importance;
    int result = llama_spec_prefill_compute_importance(sp_ctx, prompt, n_prompt, token_importance);
    
    bool success = (result == 0 && token_importance.size() == (size_t)n_prompt);
    
    if (success) {
        printf("  ✓ Importance scores created: %zu values\n", token_importance.size());
        printf("  ✓ PASSED\n\n");
    } else {
        printf("  ✗ FAILED\n\n");
    }
    
    llama_spec_prefill_free(sp_ctx);
    llama_free(ctx_base);
    llama_free(ctx_spec);
    llama_free_model(model_base);
    llama_free_model(model_spec);
    
    return success;
}

static bool test_token_filtering(const char * model_path) {
    printf("[6] Running test_token_filtering...\n");
    printf("  Testing token filtering...\n");
    
    llama_model_params model_params = llama_model_default_params();
    llama_model * model_base = llama_load_model_from_file(model_path, model_params);
    llama_model * model_spec = llama_load_model_from_file(model_path, model_params);
    
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 512;
    ctx_params.n_batch = 512;
    
    llama_context * ctx_base = llama_new_context_with_model(model_base, ctx_params);
    llama_context * ctx_spec = llama_new_context_with_model(model_spec, ctx_params);
    
    llama_spec_prefill_context * sp_ctx = llama_spec_prefill_init(ctx_base, ctx_spec);
    
    llama_token prompt[] = {1, 450, 2043, 338, 263};
    int prompt_positions[] = {0, 1, 2, 3, 4};
    int n_prompt = 5;
    float keep_ratio = 0.6f;
    
    std::vector<float> token_importance;
    llama_spec_prefill_compute_importance(sp_ctx, prompt, n_prompt, token_importance);
    
    std::vector<llama_token> filtered_tokens;
    std::vector<int> filtered_positions;
    
    int result = llama_spec_prefill_filter_tokens(
        sp_ctx, prompt, prompt_positions, n_prompt,
        token_importance, keep_ratio,
        filtered_tokens, filtered_positions
    );
    
    int expected_filtered = (int)(n_prompt * keep_ratio);
    bool success = (result == 0 && (int)filtered_tokens.size() == expected_filtered);
    
    if (success) {
        printf("  ✓ Filtered to %zu tokens (%d%% kept)\n", 
               filtered_tokens.size(), (int)(keep_ratio * 100));
        printf("  ✓ PASSED\n\n");
    } else {
        printf("  ✗ FAILED\n\n");
    }
    
    llama_spec_prefill_free(sp_ctx);
    llama_free(ctx_base);
    llama_free(ctx_spec);
    llama_free_model(model_base);
    llama_free_model(model_spec);
    
    return success;
}

static bool test_base_model_execution_with_filtered_prompt(const char * model_path) {
    printf("[7] Running test_base_model_execution_with_filtered_prompt...\n");
    printf("  Testing base model with filtered tokens...\n");
    
    llama_model_params model_params = llama_model_default_params();
    llama_model * model_base = llama_load_model_from_file(model_path, model_params);
    llama_model * model_spec = llama_load_model_from_file(model_path, model_params);
    
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 512;
    ctx_params.n_batch = 512;
    
    llama_context * ctx_base = llama_new_context_with_model(model_base, ctx_params);
    llama_context * ctx_spec = llama_new_context_with_model(model_spec, ctx_params);
    
    llama_spec_prefill_context * sp_ctx = llama_spec_prefill_init(ctx_base, ctx_spec);
    
    // Use a realistic prompt and filter it
    llama_token prompt[] = {1, 450, 2043, 338, 263};  // "The world is"
    int prompt_positions[] = {0, 1, 2, 3, 4};
    int n_prompt = 5;
    float keep_ratio = 0.6f;
    
    // Compute importance and filter
    std::vector<float> token_importance;
    llama_spec_prefill_compute_importance(sp_ctx, prompt, n_prompt, token_importance);
    
    std::vector<llama_token> filtered_tokens;
    std::vector<int> filtered_positions;
    llama_spec_prefill_filter_tokens(
        sp_ctx, prompt, prompt_positions, n_prompt,
        token_importance, keep_ratio,
        filtered_tokens, filtered_positions
    );
    
    // Now test processing with base model
    int result = llama_spec_prefill_process_base(
        sp_ctx, filtered_tokens.data(), filtered_positions.data(), (int)filtered_tokens.size()
    );
    
    bool success = (result == 0);
    
    if (success) {
        printf("  ✓ Base model processed %zu filtered tokens\n", filtered_tokens.size());
        printf("  ✓ PASSED\n\n");
    } else {
        printf("  ✗ FAILED: llama_decode returned %d\n\n", result);
    }
    
    llama_spec_prefill_free(sp_ctx);
    llama_free(ctx_base);
    llama_free(ctx_spec);
    llama_free_model(model_base);
    llama_free_model(model_spec);
    
    return success;
}

static bool test_end_to_end_spec_prefill(const char * model_path) {
    printf("[8] Running test_end_to_end_spec_prefill...\n");
    printf("  Testing end-to-end pipeline...\n");
    
    llama_model_params model_params = llama_model_default_params();
    llama_model * model_base = llama_load_model_from_file(model_path, model_params);
    llama_model * model_spec = llama_load_model_from_file(model_path, model_params);
    
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 512;
    ctx_params.n_batch = 512;
    
    llama_context * ctx_base = llama_new_context_with_model(model_base, ctx_params);
    llama_context * ctx_spec = llama_new_context_with_model(model_spec, ctx_params);
    
    llama_spec_prefill_context * sp_ctx = llama_spec_prefill_init(ctx_base, ctx_spec);
    
    llama_token prompt[] = {1, 450, 2043, 338, 263};
    int n_prompt = 5;
    int n_lookahead = 3;
    float keep_ratio = 0.6f;
    
    int n_filtered = llama_spec_prefill(
        sp_ctx, prompt, n_prompt, n_lookahead, keep_ratio
    );
    
    bool success = (n_filtered > 0);
    
    if (success) {
        printf("  ✓ End-to-end pipeline successful\n");
        printf("  ✓ PASSED\n\n");
    } else {
        printf("  ✗ FAILED\n\n");
    }
    
    llama_spec_prefill_free(sp_ctx);
    llama_free(ctx_base);
    llama_free(ctx_spec);
    llama_free_model(model_base);
    llama_free_model(model_spec);
    
    return success;
}

// ========================================
// Main Test Runner
// ========================================

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model-path>\n", argv[0]);
        return 1;
    }
    
    const char * model_path = argv[1];
    
    printf("========================================\n");
    printf("Speculative Prefill POC Tests\n");
    printf("========================================\n\n");
    
    int total = 0;
    int passed = 0;
    
    // Run all tests
    if (test_dual_model_loading(model_path)) passed++;
    total++;
    
    if (test_spec_model_lookahead_generation(model_path)) passed++;
    total++;
    
    if (test_attention_score_extraction(model_path)) passed++;
    total++;
    
    if (test_attention_computation(model_path)) passed++;
    total++;
    
    if (test_token_importance_aggregation(model_path)) passed++;
    total++;
    
    if (test_token_filtering(model_path)) passed++;
    total++;
    
    if (test_base_model_execution_with_filtered_prompt(model_path)) passed++;
    total++;
    
    if (test_end_to_end_spec_prefill(model_path)) passed++;
    total++;
    
    // Print summary
    printf("========================================\n");
    printf("Test Summary:\n");
    printf("  Total: %d\n", total);
    printf("  Passed: %d (%.1f%%)\n", passed, 100.0f * passed / total);
    printf("  Failed: %d (%.1f%%)\n", total - passed, 100.0f * (total - passed) / total);
    printf("========================================\n");
    
    return (passed == total) ? 0 : 1;
}
