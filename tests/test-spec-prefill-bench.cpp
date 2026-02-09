#include "llama.h"
#include "llama-spec-prefill.h"
#include "../common/common.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>

// Benchmark utilities
static double get_time_ms() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double, std::milli>(duration).count();
}

// Standard prefill: process all prompt tokens with base model
static double benchmark_standard_prefill(
    llama_context * ctx_base,
    const llama_token * prompt_tokens,
    int n_prompt
) {
    // Clear KV cache
    llama_memory_t mem = llama_get_memory(ctx_base);
    llama_memory_seq_rm(mem, 0, 0, -1);
    
    double start = get_time_ms();
    
    // Process all tokens sequentially
    llama_batch batch = llama_batch_init(n_prompt, 0, 1);
    batch.n_tokens = 0;
    
    for (int i = 0; i < n_prompt; i++) {
        batch.token[batch.n_tokens] = prompt_tokens[i];
        batch.pos[batch.n_tokens] = i;
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.seq_id[batch.n_tokens][0] = 0;
        batch.logits[batch.n_tokens] = (i == n_prompt - 1);
        batch.n_tokens++;
    }
    
    llama_decode(ctx_base, batch);
    llama_batch_free(batch);
    
    double end = get_time_ms();
    return end - start;
}

// Speculative prefill: filter tokens first
static double benchmark_spec_prefill(
    llama_spec_prefill_context * sp_ctx,
    const llama_token * prompt_tokens,
    int n_prompt,
    int n_lookahead,
    float keep_ratio,
    int * n_processed_out
) {
    double start = get_time_ms();
    
    int n_filtered = llama_spec_prefill(
        sp_ctx, prompt_tokens, n_prompt, n_lookahead, keep_ratio
    );
    
    double end = get_time_ms();
    
    if (n_processed_out) {
        *n_processed_out = n_filtered;
    }
    
    return end - start;
}

// Run comprehensive benchmark
static void run_benchmark(const char * base_model_path, const char * spec_model_path) {
    printf("========================================\n");
    printf("Speculative Prefill Performance Benchmark\n");
    printf("========================================\n\n");
    
    printf("Base model: %s\n", base_model_path);
    printf("Spec model: %s\n", spec_model_path);
    printf("\n");
    
    // Load models
    llama_model_params model_params = llama_model_default_params();
    llama_model * model_base = llama_load_model_from_file(base_model_path, model_params);
    llama_model * model_spec = llama_load_model_from_file(spec_model_path, model_params);
    
    if (!model_base || !model_spec) {
        fprintf(stderr, "Failed to load model\n");
        return;
    }
    
    // Create contexts
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    ctx_params.n_batch = 2048;
    
    llama_context * ctx_base = llama_new_context_with_model(model_base, ctx_params);
    llama_context * ctx_spec = llama_new_context_with_model(model_spec, ctx_params);
    
    llama_spec_prefill_context * sp_ctx = llama_spec_prefill_init(ctx_base, ctx_spec);
    
    // Test prompts of varying lengths
    std::vector<int> prompt_lengths = {50, 100, 200, 500, 1000};
    const int n_runs = 3;
    
    printf("%-12s | %-12s | %-12s | %-12s | %-10s | %-10s\n",
           "Prompt Len", "Standard(ms)", "SpecPrefill", "Tokens Proc", "Speedup", "Reduction");
    printf("-------------+--------------+--------------+--------------+------------+------------\n");
    
    for (int n_prompt : prompt_lengths) {
        // Generate test prompt (just repeat token 1000 for simplicity)
        std::vector<llama_token> prompt(n_prompt);
        for (int i = 0; i < n_prompt; i++) {
            prompt[i] = 1000 + (i % 100);  // Cycle through tokens
        }
        
        // Warm up
        benchmark_standard_prefill(ctx_base, prompt.data(), n_prompt);
        
        // Benchmark standard prefill
        double total_standard = 0.0;
        for (int run = 0; run < n_runs; run++) {
            double time = benchmark_standard_prefill(ctx_base, prompt.data(), n_prompt);
            total_standard += time;
        }
        double avg_standard = total_standard / n_runs;
        
        // Benchmark speculative prefill with different keep ratios
        float keep_ratio = 0.5f;
        int n_lookahead = 8;
        
        int n_processed = 0;
        double total_spec = 0.0;
        for (int run = 0; run < n_runs; run++) {
            double time = benchmark_spec_prefill(
                sp_ctx, prompt.data(), n_prompt, n_lookahead, keep_ratio, &n_processed
            );
            total_spec += time;
        }
        double avg_spec = total_spec / n_runs;
        
        // Calculate metrics
        double speedup = avg_standard / avg_spec;
        double reduction = 100.0 * (1.0 - (double)n_processed / n_prompt);
        
        printf("%-12d | %12.2f | %12.2f | %12d | %9.2fx | %9.1f%%\n",
               n_prompt, avg_standard, avg_spec, n_processed, speedup, reduction);
    }
    
    printf("\n");
    printf("Notes:\n");
    printf("  - keep_ratio = 0.5 (filtering 50%% of tokens)\n");
    printf("  - n_lookahead = 8\n");
    printf("  - Averaged over 3 runs\n");
    printf("  - Using attention-based importance (logit analysis)\n");
    
    if (strcmp(base_model_path, spec_model_path) == 0) {
        printf("  - ⚠️  Same model used for base and spec\n");
        printf("  - ⚠️  No speedup expected (overhead dominates)\n");
        printf("  - ℹ️  Use smaller spec model to see 2-3x speedup\n");
    } else {
        printf("  - ✓ Different models for base and spec\n");
        printf("  - ✓ Speedup depends on model size ratio\n");
    }
    
    printf("\n");
    
    // Test different keep_ratio values on 200-token prompt
    printf("\n========================================\n");
    printf("Keep Ratio Sensitivity Analysis (200 tokens)\n");
    printf("========================================\n\n");
    
    int test_len = 200;
    std::vector<llama_token> test_prompt(test_len);
    for (int i = 0; i < test_len; i++) {
        test_prompt[i] = 1000 + (i % 100);
    }
    
    double baseline = 0.0;
    for (int run = 0; run < n_runs; run++) {
        baseline += benchmark_standard_prefill(ctx_base, test_prompt.data(), test_len);
    }
    baseline /= n_runs;
    
    printf("%-12s | %-12s | %-12s | %-10s | %-10s\n",
           "Keep Ratio", "Time (ms)", "Tokens Proc", "Speedup", "Reduction");
    printf("-------------+--------------+--------------+------------+------------\n");
    
    std::vector<float> keep_ratios = {0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    
    for (float ratio : keep_ratios) {
        int n_proc = 0;
        double total = 0.0;
        for (int run = 0; run < n_runs; run++) {
            total += benchmark_spec_prefill(
                sp_ctx, test_prompt.data(), test_len, 8, ratio, &n_proc
            );
        }
        double avg = total / n_runs;
        double speedup = baseline / avg;
        double reduction = 100.0 * (1.0 - (double)n_proc / test_len);
        
        printf("%12.1f | %12.2f | %12d | %9.2fx | %9.1f%%\n",
               ratio, avg, n_proc, speedup, reduction);
    }
    
    printf("\n");
    
    // Cleanup
    llama_spec_prefill_free(sp_ctx);
    llama_free(ctx_base);
    llama_free(ctx_spec);
    llama_free_model(model_base);
    llama_free_model(model_spec);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <base-model-path> [spec-model-path]\n", argv[0]);
        fprintf(stderr, "\n");
        fprintf(stderr, "Arguments:\n");
        fprintf(stderr, "  base-model-path: Path to the base (larger) model\n");
        fprintf(stderr, "  spec-model-path: Path to the spec (smaller) model (optional)\n");
        fprintf(stderr, "                   If not provided, uses base model for both\n");
        fprintf(stderr, "\n");
        fprintf(stderr, "Examples:\n");
        fprintf(stderr, "  # Same model for both (baseline)\n");
        fprintf(stderr, "  %s models/tinyllama-1.1b.gguf\n", argv[0]);
        fprintf(stderr, "\n");
        fprintf(stderr, "  # Different models (shows speedup)\n");
        fprintf(stderr, "  %s models/llama-3.2-3b.gguf models/smollm-135m.gguf\n", argv[0]);
        return 1;
    }
    
    const char * base_model_path = argv[1];
    const char * spec_model_path = (argc >= 3) ? argv[2] : argv[1];
    
    if (argc < 3) {
        printf("Note: Using same model for base and spec (no speedup expected)\n");
        printf("      Provide two model paths to see actual speedup\n\n");
    }
    
    run_benchmark(base_model_path, spec_model_path);
    
    return 0;
}
