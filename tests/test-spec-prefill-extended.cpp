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

// Helper: create base test fixtures
static bool create_spec_prefill_fixtures(
    const char * model_path,
    llama_model *& model_base,
    llama_model *& model_spec,
    llama_context *& ctx_base,
    llama_context *& ctx_spec,
    llama_spec_prefill_context *& sp_ctx,
    int n_ctx = 512,
    int n_batch = 512
) {
    llama_model_params model_params = llama_model_default_params();
    model_base = llama_load_model_from_file(model_path, model_params);
    model_spec = llama_load_model_from_file(model_path, model_params);
    
    if (!model_base || !model_spec) {
        if (model_base) llama_free_model(model_base);
        if (model_spec) llama_free_model(model_spec);
        return false;
    }
    
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = n_batch;
    
    ctx_base = llama_new_context_with_model(model_base, ctx_params);
    ctx_spec = llama_new_context_with_model(model_spec, ctx_params);
    
    if (!ctx_base || !ctx_spec) {
        if (ctx_base) llama_free(ctx_base);
        if (ctx_spec) llama_free(ctx_spec);
        llama_free_model(model_base);
        llama_free_model(model_spec);
        return false;
    }
    
    sp_ctx = llama_spec_prefill_init(ctx_base, ctx_spec);
    if (!sp_ctx) {
        llama_free(ctx_base);
        llama_free(ctx_spec);
        llama_free_model(model_base);
        llama_free_model(model_spec);
        return false;
    }
    
    return true;
}

static void destroy_spec_prefill_fixtures(
    llama_spec_prefill_context * sp_ctx,
    llama_context * ctx_base,
    llama_context * ctx_spec,
    llama_model * model_base,
    llama_model * model_spec
) {
    if (sp_ctx) llama_spec_prefill_free(sp_ctx);
    if (ctx_base) llama_free(ctx_base);
    if (ctx_spec) llama_free(ctx_spec);
    if (model_base) llama_free_model(model_base);
    if (model_spec) llama_free_model(model_spec);
}

static bool test_dual_model_loading(const char * model_path) {
    printf("[1] Running test_dual_model_loading...\n");
    printf("  Testing dual model initialization...\n");
    
    llama_model * model_base = nullptr, * model_spec = nullptr;
    llama_context * ctx_base = nullptr, * ctx_spec = nullptr;
    llama_spec_prefill_context * sp_ctx = nullptr;
    
    bool success = create_spec_prefill_fixtures(model_path, model_base, model_spec, ctx_base, ctx_spec, sp_ctx);
    
    if (success) {
        printf("  ✓ Dual model loading successful\n");
        printf("  ✓ PASSED\n\n");
    } else {
        printf("  ✗ FAILED: Could not create fixtures\n\n");
    }
    
    destroy_spec_prefill_fixtures(sp_ctx, ctx_base, ctx_spec, model_base, model_spec);
    return success;
}

static bool test_spec_model_lookahead_generation(const char * model_path) {
    printf("[2] Running test_spec_model_lookahead_generation...\n");
    printf("  Testing lookahead generation...\n");
    
    llama_model * model_base = nullptr, * model_spec = nullptr;
    llama_context * ctx_base = nullptr, * ctx_spec = nullptr;
    llama_spec_prefill_context * sp_ctx = nullptr;
    
    if (!create_spec_prefill_fixtures(model_path, model_base, model_spec, ctx_base, ctx_spec, sp_ctx)) {
        printf("  ✗ FAILED: Could not create fixtures\n\n");
        return false;
    }
    
    llama_token prompt[] = {1, 450, 2043, 338};
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
    
    destroy_spec_prefill_fixtures(sp_ctx, ctx_base, ctx_spec, model_base, model_spec);
    return success;
}

static bool test_attention_score_extraction(const char * model_path) {
    printf("[3] Running test_attention_score_extraction...\n");
    printf("  Testing Q/K tensor extraction...\n");
    
    llama_model * model_base = nullptr, * model_spec = nullptr;
    llama_context * ctx_base = nullptr, * ctx_spec = nullptr;
    llama_spec_prefill_context * sp_ctx = nullptr;
    
    if (!create_spec_prefill_fixtures(model_path, model_base, model_spec, ctx_base, ctx_spec, sp_ctx)) {
        printf("  ✗ FAILED: Could not create fixtures\n\n");
        return false;
    }
    
    int n_lookahead = 3;
    llama_token prompt[] = {1, 450, 2043, 338};
    int n_prompt = 4;

    // First generate lookahead tokens so the spec model has a computation graph
    // with Q/K tensors to extract
    std::vector<llama_token> gen_lookahead(n_lookahead);
    int n_generated = llama_spec_prefill_generate_lookahead(
        sp_ctx, prompt, n_prompt, gen_lookahead.data(), n_lookahead
    );
    if (n_generated <= 0) {
        printf("  ✗ FAILED: generate_lookahead returned %d\n\n", n_generated);
        destroy_spec_prefill_fixtures(sp_ctx, ctx_base, ctx_spec, model_base, model_spec);
        return false;
    }

    int result = llama_spec_prefill_extract_qk(sp_ctx, gen_lookahead.data(), n_generated);
    
    bool success = (result == 0);
    
    if (success) {
        printf("  ✓ Q/K extraction successful (simplified for POC)\n");
        printf("  ✓ PASSED\n\n");
    } else {
        printf("  ✗ FAILED\n\n");
    }
    
    destroy_spec_prefill_fixtures(sp_ctx, ctx_base, ctx_spec, model_base, model_spec);
    return success;
}

static bool test_attention_computation(const char * model_path) {
    printf("[4] Running test_attention_computation...\n");
    printf("  Testing attention computation...\n");
    
    llama_model * model_base = nullptr, * model_spec = nullptr;
    llama_context * ctx_base = nullptr, * ctx_spec = nullptr;
    llama_spec_prefill_context * sp_ctx = nullptr;
    
    if (!create_spec_prefill_fixtures(model_path, model_base, model_spec, ctx_base, ctx_spec, sp_ctx)) {
        printf("  ✗ FAILED: Could not create fixtures\n\n");
        return false;
    }
    
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
    
    destroy_spec_prefill_fixtures(sp_ctx, ctx_base, ctx_spec, model_base, model_spec);
    return success;
}

static bool test_token_importance_aggregation(const char * model_path) {
    printf("[5] Running test_token_importance_aggregation...\n");
    printf("  Testing token importance aggregation...\n");
    
    llama_model * model_base = nullptr, * model_spec = nullptr;
    llama_context * ctx_base = nullptr, * ctx_spec = nullptr;
    llama_spec_prefill_context * sp_ctx = nullptr;
    
    if (!create_spec_prefill_fixtures(model_path, model_base, model_spec, ctx_base, ctx_spec, sp_ctx)) {
        printf("  ✗ FAILED: Could not create fixtures\n\n");
        return false;
    }
    
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
    
    destroy_spec_prefill_fixtures(sp_ctx, ctx_base, ctx_spec, model_base, model_spec);
    return success;
}

static bool test_token_filtering(const char * model_path) {
    printf("[6] Running test_token_filtering...\n");
    printf("  Testing token filtering...\n");
    
    llama_model * model_base = nullptr, * model_spec = nullptr;
    llama_context * ctx_base = nullptr, * ctx_spec = nullptr;
    llama_spec_prefill_context * sp_ctx = nullptr;
    
    if (!create_spec_prefill_fixtures(model_path, model_base, model_spec, ctx_base, ctx_spec, sp_ctx)) {
        printf("  ✗ FAILED: Could not create fixtures\n\n");
        return false;
    }
    
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
    
    destroy_spec_prefill_fixtures(sp_ctx, ctx_base, ctx_spec, model_base, model_spec);
    return success;
}

static bool test_base_model_execution_with_filtered_prompt(const char * model_path) {
    printf("[7] Running test_base_model_execution_with_filtered_prompt...\n");
    printf("  Testing base model with filtered tokens...\n");
    
    llama_model * model_base = nullptr, * model_spec = nullptr;
    llama_context * ctx_base = nullptr, * ctx_spec = nullptr;
    llama_spec_prefill_context * sp_ctx = nullptr;
    
    if (!create_spec_prefill_fixtures(model_path, model_base, model_spec, ctx_base, ctx_spec, sp_ctx)) {
        printf("  ✗ FAILED: Could not create fixtures\n\n");
        return false;
    }
    
    llama_token prompt[] = {1, 450, 2043, 338, 263};
    int prompt_positions[] = {0, 1, 2, 3, 4};
    int n_prompt = 5;
    float keep_ratio = 0.6f;
    
    std::vector<float> token_importance;
    llama_spec_prefill_compute_importance(sp_ctx, prompt, n_prompt, token_importance);
    
    std::vector<llama_token> filtered_tokens;
    std::vector<int> filtered_positions;
    llama_spec_prefill_filter_tokens(
        sp_ctx, prompt, prompt_positions, n_prompt,
        token_importance, keep_ratio,
        filtered_tokens, filtered_positions
    );
    
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
    
    destroy_spec_prefill_fixtures(sp_ctx, ctx_base, ctx_spec, model_base, model_spec);
    return success;
}

static bool test_end_to_end_spec_prefill(const char * model_path) {
    printf("[8] Running test_end_to_end_spec_prefill...\n");
    printf("  Testing end-to-end pipeline...\n");
    
    llama_model * model_base = nullptr, * model_spec = nullptr;
    llama_context * ctx_base = nullptr, * ctx_spec = nullptr;
    llama_spec_prefill_context * sp_ctx = nullptr;
    
    if (!create_spec_prefill_fixtures(model_path, model_base, model_spec, ctx_base, ctx_spec, sp_ctx)) {
        printf("  ✗ FAILED: Could not create fixtures\n\n");
        return false;
    }
    
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
    
    destroy_spec_prefill_fixtures(sp_ctx, ctx_base, ctx_spec, model_base, model_spec);
    return success;
}

// ========================================
// NEW EDGE CASE TESTS (9-15)
// ========================================

static bool test_short_prompt_single_token(const char * model_path) {
    printf("[9] Running test_short_prompt_single_token...\n");
    printf("  Testing with single-token prompt...\n");
    
    llama_model * model_base = nullptr, * model_spec = nullptr;
    llama_context * ctx_base = nullptr, * ctx_spec = nullptr;
    llama_spec_prefill_context * sp_ctx = nullptr;
    
    if (!create_spec_prefill_fixtures(model_path, model_base, model_spec, ctx_base, ctx_spec, sp_ctx)) {
        printf("  ✗ FAILED: Could not create fixtures\n\n");
        return false;
    }
    
    llama_token prompt[] = {1};  // Single token (BOS)
    int n_prompt = 1;
    int n_lookahead = 2;
    
    std::vector<llama_token> lookahead_tokens(n_lookahead);
    int n_generated = llama_spec_prefill_generate_lookahead(
        sp_ctx, prompt, n_prompt, lookahead_tokens.data(), n_lookahead
    );
    
    bool success = (n_generated > 0);
    
    if (success) {
        printf("  ✓ Generated %d tokens from single-token prompt\n", n_generated);
        printf("  ✓ PASSED\n\n");
    } else {
        printf("  ✗ FAILED: No lookahead tokens generated from single-token prompt\n\n");
    }
    
    destroy_spec_prefill_fixtures(sp_ctx, ctx_base, ctx_spec, model_base, model_spec);
    return success;
}

static bool test_single_token_lookahead(const char * model_path) {
    printf("[10] Running test_single_token_lookahead...\n");
    printf("  Testing with 1-token lookahead...\n");
    
    llama_model * model_base = nullptr, * model_spec = nullptr;
    llama_context * ctx_base = nullptr, * ctx_spec = nullptr;
    llama_spec_prefill_context * sp_ctx = nullptr;
    
    if (!create_spec_prefill_fixtures(model_path, model_base, model_spec, ctx_base, ctx_spec, sp_ctx)) {
        printf("  ✗ FAILED: Could not create fixtures\n\n");
        return false;
    }
    
    llama_token prompt[] = {1, 450, 2043};
    int n_prompt = 3;
    int n_lookahead = 1;  // Minimal lookahead
    
    std::vector<llama_token> lookahead_tokens(n_lookahead);
    int n_generated = llama_spec_prefill_generate_lookahead(
        sp_ctx, prompt, n_prompt, lookahead_tokens.data(), n_lookahead
    );
    
    bool success = (n_generated == 1);
    
    if (success) {
        printf("  ✓ Generated exactly 1 lookahead token\n");
        printf("  ✓ PASSED\n\n");
    } else {
        printf("  ✗ FAILED: Expected 1, got %d\n", n_generated);
    }
    
    destroy_spec_prefill_fixtures(sp_ctx, ctx_base, ctx_spec, model_base, model_spec);
    return success;
}

static bool test_keep_ratio_one(const char * model_path) {
    printf("[11] Running test_keep_ratio_one...\n");
    printf("  Testing with keep_ratio=1.0 (keep all tokens)...\n");
    
    llama_model * model_base = nullptr, * model_spec = nullptr;
    llama_context * ctx_base = nullptr, * ctx_spec = nullptr;
    llama_spec_prefill_context * sp_ctx = nullptr;
    
    if (!create_spec_prefill_fixtures(model_path, model_base, model_spec, ctx_base, ctx_spec, sp_ctx)) {
        printf("  ✗ FAILED: Could not create fixtures\n\n");
        return false;
    }
    
    llama_token prompt[] = {1, 450, 2043, 338, 263, 1407, 2058, 716};
    int prompt_positions[] = {0, 1, 2, 3, 4, 5, 6, 7};
    int n_prompt = 8;
    float keep_ratio = 1.0f;  // Keep all
    
    std::vector<float> token_importance;
    llama_spec_prefill_compute_importance(sp_ctx, prompt, n_prompt, token_importance);
    
    std::vector<llama_token> filtered_tokens;
    std::vector<int> filtered_positions;
    
    int result = llama_spec_prefill_filter_tokens(
        sp_ctx, prompt, prompt_positions, n_prompt,
        token_importance, keep_ratio,
        filtered_tokens, filtered_positions
    );
    
    bool success = (result == 0 && (int)filtered_tokens.size() == n_prompt);
    
    if (success) {
        printf("  ✓ All %d tokens retained with keep_ratio=1.0\n", (int)filtered_tokens.size());
        printf("  ✓ PASSED\n\n");
    } else {
        printf("  ✗ FAILED: Expected %d tokens, got %zu\n", n_prompt, filtered_tokens.size());
    }
    
    destroy_spec_prefill_fixtures(sp_ctx, ctx_base, ctx_spec, model_base, model_spec);
    return success;
}

static bool test_keep_ratio_near_zero(const char * model_path) {
    printf("[12] Running test_keep_ratio_near_zero...\n");
    printf("  Testing with keep_ratio=0.05 (keep almost none)...\n");
    
    llama_model * model_base = nullptr, * model_spec = nullptr;
    llama_context * ctx_base = nullptr, * ctx_spec = nullptr;
    llama_spec_prefill_context * sp_ctx = nullptr;
    
    if (!create_spec_prefill_fixtures(model_path, model_base, model_spec, ctx_base, ctx_spec, sp_ctx)) {
        printf("  ✗ FAILED: Could not create fixtures\n\n");
        return false;
    }
    
    llama_token prompt[] = {1, 450, 2043, 338, 263, 1407, 2058, 716, 286, 278};
    int prompt_positions[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int n_prompt = 10;
    float keep_ratio = 0.05f;  // Keep only 0-1 tokens
    
    std::vector<float> token_importance;
    llama_spec_prefill_compute_importance(sp_ctx, prompt, n_prompt, token_importance);
    
    std::vector<llama_token> filtered_tokens;
    std::vector<int> filtered_positions;
    
    int result = llama_spec_prefill_filter_tokens(
        sp_ctx, prompt, prompt_positions, n_prompt,
        token_importance, keep_ratio,
        filtered_tokens, filtered_positions
    );
    
    // With very low keep_ratio, should keep 0 or very few tokens
    bool success = (result == 0 && (int)filtered_tokens.size() <= 2);
    
    if (success) {
        printf("  ✓ Minimal tokens kept: %zu (keep_ratio=0.05)\n", filtered_tokens.size());
        printf("  ✓ PASSED\n\n");
    } else {
        printf("  ✗ FAILED: Expected ≤2 tokens, got %zu\n", filtered_tokens.size());
    }
    
    destroy_spec_prefill_fixtures(sp_ctx, ctx_base, ctx_spec, model_base, model_spec);
    return success;
}

static bool test_deterministic_output(const char * model_path) {
    printf("[13] Running test_deterministic_output...\n");
    printf("  Testing deterministic behavior...\n");
    
    llama_model * model_base = nullptr, * model_spec = nullptr;
    llama_context * ctx_base = nullptr, * ctx_spec = nullptr;
    llama_spec_prefill_context * sp_ctx1 = nullptr, * sp_ctx2 = nullptr;
    
    if (!create_spec_prefill_fixtures(model_path, model_base, model_spec, ctx_base, ctx_spec, sp_ctx1)) {
        printf("  ✗ FAILED: Could not create fixtures\n\n");
        return false;
    }
    sp_ctx2 = llama_spec_prefill_init(ctx_base, ctx_spec);
    if (!sp_ctx2) {
        destroy_spec_prefill_fixtures(sp_ctx1, ctx_base, ctx_spec, model_base, model_spec);
        printf("  ✗ FAILED: Could not create second context\n\n");
        return false;
    }
    
    llama_token prompt[] = {1, 450, 2043, 338, 263};
    int n_prompt = 5;
    int n_lookahead = 3;
    float keep_ratio = 0.5f;
    
    // Run same pipeline twice
    int n_filtered_1 = llama_spec_prefill(sp_ctx1, prompt, n_prompt, n_lookahead, keep_ratio);
    int n_filtered_2 = llama_spec_prefill(sp_ctx2, prompt, n_prompt, n_lookahead, keep_ratio);
    
    bool success = (n_filtered_1 == n_filtered_2);
    
    if (success) {
        printf("  ✓ Deterministic: run1=%d, run2=%d (match)\n", n_filtered_1, n_filtered_2);
        printf("  ✓ PASSED\n\n");
    } else {
        printf("  ✗ FAILED: run1=%d, run2=%d (mismatch)\n", n_filtered_1, n_filtered_2);
    }
    
    llama_spec_prefill_free(sp_ctx2);
    destroy_spec_prefill_fixtures(sp_ctx1, ctx_base, ctx_spec, model_base, model_spec);
    return success;
}

static bool test_large_context_window(const char * model_path) {
    printf("[14] Running test_large_context_window...\n");
    printf("  Testing with larger context window (4096)...\n");
    
    llama_model * model_base = nullptr, * model_spec = nullptr;
    llama_context * ctx_base = nullptr, * ctx_spec = nullptr;
    llama_spec_prefill_context * sp_ctx = nullptr;
    
    // Use larger context window
    if (!create_spec_prefill_fixtures(model_path, model_base, model_spec, ctx_base, ctx_spec, sp_ctx, 4096, 512)) {
        printf("  ✗ FAILED: Could not create fixtures with large context\n\n");
        return false;
    }
    
    // Fill with repeating pattern tokens
    llama_token prompt[32];
    for (int i = 0; i < 32; i++) {
        prompt[i] = 1 + (i % 100);  // Tokens 1-100 cycling
    }
    int n_prompt = 32;
    int n_lookahead = 5;
    
    std::vector<llama_token> lookahead_tokens(n_lookahead);
    int n_generated = llama_spec_prefill_generate_lookahead(
        sp_ctx, prompt, n_prompt, lookahead_tokens.data(), n_lookahead
    );
    
    bool success = (n_generated > 0);
    
    if (success) {
        printf("  ✓ Handled 32-token prompt with 4096 context\n");
        printf("  ✓ Generated %d lookahead tokens\n", n_generated);
        printf("  ✓ PASSED\n\n");
    } else {
        printf("  ✗ FAILED: No tokens generated with large context\n\n");
    }
    
    destroy_spec_prefill_fixtures(sp_ctx, ctx_base, ctx_spec, model_base, model_spec);
    return success;
}

static bool test_importance_score_range(const char * model_path) {
    printf("[15] Running test_importance_score_range...\n");
    printf("  Testing importance scores are in valid range [0, 1]...\n");
    
    llama_model * model_base = nullptr, * model_spec = nullptr;
    llama_context * ctx_base = nullptr, * ctx_spec = nullptr;
    llama_spec_prefill_context * sp_ctx = nullptr;
    
    if (!create_spec_prefill_fixtures(model_path, model_base, model_spec, ctx_base, ctx_spec, sp_ctx)) {
        printf("  ✗ FAILED: Could not create fixtures\n\n");
        return false;
    }
    
    llama_token prompt[] = {1, 450, 2043, 338, 263, 1407, 2058, 716, 286, 278, 100, 200, 300, 400, 500};
    int n_prompt = 15;
    
    std::vector<float> token_importance;
    int result = llama_spec_prefill_compute_importance(sp_ctx, prompt, n_prompt, token_importance);
    
    bool success = (result == 0 && token_importance.size() == (size_t)n_prompt);
    
    // Check all scores are in valid range
    if (success) {
        float min_val = token_importance[0], max_val = token_importance[0];
        for (size_t i = 1; i < token_importance.size(); i++) {
            min_val = std::min(min_val, token_importance[i]);
            max_val = std::max(max_val, token_importance[i]);
        }
        
        bool in_range = (min_val >= 0.0f && max_val <= 1.0f);
        if (in_range) {
            printf("  ✓ All %d scores in [%.4f, %.4f] (valid range [0,1])\n", n_prompt, min_val, max_val);
            printf("  ✓ PASSED\n\n");
        } else {
            printf("  ✗ FAILED: Scores out of range: [%.4f, %.4f]\n", min_val, max_val);
            success = false;
        }
    } else {
        printf("  ✗ FAILED: Importance computation failed\n\n");
    }
    
    destroy_spec_prefill_fixtures(sp_ctx, ctx_base, ctx_spec, model_base, model_spec);
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
    printf("Speculative Prefill POC Tests (Extended)\n");
    printf("========================================\n\n");
    
    int total = 0;
    int passed = 0;
    
    // Original tests (1-8)
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
    
    // New edge case tests (9-15)
    if (test_short_prompt_single_token(model_path)) passed++;
    total++;
    
    if (test_single_token_lookahead(model_path)) passed++;
    total++;
    
    if (test_keep_ratio_one(model_path)) passed++;
    total++;
    
    if (test_keep_ratio_near_zero(model_path)) passed++;
    total++;
    
    if (test_deterministic_output(model_path)) passed++;
    total++;
    
    if (test_large_context_window(model_path)) passed++;
    total++;
    
    if (test_importance_score_range(model_path)) passed++;
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
