#pragma once

// ============================================================================
// Speculative Prefill — POC Module
// ============================================================================
//
// Reference: Jingyu6/speculative_prefill (vLLM, arXiv:2502.02789)
//
// STATUS: PROOF-OF-CONCEPT
//
// This module implements a speculative-prefill token-filtering pipeline.
// The reference algorithm uses actual base-model attention scores to select
// prompt tokens for prefill acceleration. This POC replaces attention
// computation (compute_attention) with a perplexity-based proxy that uses
// the spec model's own prediction confidence.
//
// The algorithm is validated through:
//   - Self-determinism (4/4 prompts byte-identical across runs)
//   - Quality gate: Rouge-L ratio 0.857 at kr=0.25 (threshold: 0.70)
//   - Performance: 1.47x–2.24x GPU speedup at kr=0.10–0.25
//   - Ablations: 54/54 runs, optimal config (lah=8, pool=13, chunk=32)
//
// Known limitations:
//   - Importance scoring uses spec-model perplexity, not base-model attention
//   - process_base re-indexes filtered tokens to 0..n_kept-1 (breaks RoPE
//     for scattered positions — a known limitation for production use)
//   - Cross-impl parity vs vLLM not yet verified (environment blocked)
//
// ============================================================================

#include "llama.h"
#include <string>
#include <vector>

struct ggml_tensor;

// Lookahead generation statistics for importance scoring
struct llama_lookahead_stat {
    float max_logit;    // Maximum logit value (prediction confidence)
    float entropy;      // Entropy of logit distribution (uncertainty)
    int position;       // Position in sequence
};

// Configuration for speculative prefill.
// Defaults are set to the experimentally-optimal configuration from ablation
// studies (lah=8, pool=13, chunk=32, kr=0.25).
struct llama_spec_prefill_params {
    float keep_ratio;       // Percentage of tokens to keep (0.1 = 10%)
    int n_lookahead;        // Number of lookahead tokens to generate
    int pool_kernel_size;   // Pooling kernel size for smoothing (0 = disabled, vLLM uses 13)
    bool use_chunking;      // Whether to use chunk-based selection
    int chunk_size;         // Size of chunks for chunked selection (vLLM uses 32)
    bool ignore_eos;        // Whether to ignore EOS tokens during lookahead
    std::vector<llama_token> eos_tokens;  // EOS token IDs to detect

    // Default constructor — experimentally optimal configuration
    llama_spec_prefill_params() :
        keep_ratio(0.25f),       // 25% kept — best quality/speed tradeoff
        n_lookahead(8),          // 8 lookahead tokens — optimal latency
        pool_kernel_size(13),    // vLLM default — significantly better than smaller
        use_chunking(true),      // chunk-based selection
        chunk_size(32),          // vLLM default — equivalent to percentage strategy
        ignore_eos(false) {}
};

// Metadata for a Q tensor extracted from the computation graph
struct llama_spec_q_tensor {
    int              layer_idx;
    int64_t          n_embd_head_q;
    int64_t          n_head;
    int64_t          n_tokens;
    const ggml_tensor * tensor_ptr;
    std::vector<float> data;
};

// Speculative prefill execution context.
// Opaque handle — users must manage through init/free pair.
struct llama_spec_prefill_context {
    struct llama_context * ctx_base;     // Base model context
    struct llama_context * ctx_spec;     // Speculative (smaller) model context
    std::vector<llama_lookahead_stat> lookahead_stats;
    llama_spec_prefill_params params;
    int actual_lookahead_cnt;
    std::string dump_path;

    std::vector<llama_spec_q_tensor> q_tensors;
};

// Initialize speculative prefill context (uses default params)
// Returns nullptr on invalid input. Caller must free with llama_spec_prefill_free().
struct llama_spec_prefill_context * llama_spec_prefill_init(
    struct llama_context * ctx_base,
    struct llama_context * ctx_spec
);

// Initialize speculative prefill context with custom parameters
struct llama_spec_prefill_context * llama_spec_prefill_init_with_params(
    struct llama_context * ctx_base,
    struct llama_context * ctx_spec,
    const llama_spec_prefill_params & params
);

// Free speculative prefill context and all allocated resources
void llama_spec_prefill_free(struct llama_spec_prefill_context * ctx);

// Generate lookahead tokens using spec model.
// Returns number of tokens generated (may be less than n_lookahead if EOS encountered).
// Returns -1 on error.
int llama_spec_prefill_generate_lookahead(
    struct llama_spec_prefill_context * ctx,
    const llama_token * prompt_tokens,
    int n_prompt,
    llama_token * lookahead_tokens,
    int n_lookahead
);

// Extract Q and K tensors from the spec model's computation graph.
// Uses get_gf_res_prev() to access the most recent graph result.
// Returns the number of Q tensors extracted, or -1 on error.
// A return value of 0 means no Q tensors found (graceful, not fatal).
int llama_spec_prefill_extract_qk(
    struct llama_spec_prefill_context * ctx,
    const llama_token * lookahead_tokens,
    int n_lookahead
);

// Compute attention scores between prompt and lookahead tokens.
//
// POC NO-OP: Returns 0 (no-op). The spec model's lookahead_stats already
// capture prediction confidence via entropy in generate_lookahead(), making
// this stub unnecessary for current POC functionality.
//
// PRODUCTION TODO: Full implementation requires GGML ops for
//   softmax(Q @ K^T / sqrt(d_k)) using actual base-model Q/K tensors.
//   This is the PRIMARY GATE for production readiness — without real
//   attention scores, importance estimation relies entirely on the draft
//   model's own confidence, which may not align with base-model behavior.
//
// WARNING: This function is locked behind the POC flag. Consumers must
// not depend on its return value for correctness (always returns 0).
int llama_spec_prefill_compute_attention(
    struct llama_spec_prefill_context * ctx,
    const llama_token * prompt_tokens,
    int n_prompt,
    const llama_token * lookahead_tokens,
    int n_lookahead
);

// Compute token importance scores using perplexity-based proxy.
//
// For prompts <= 4096 tokens: computes spec-model NLL (negative log-likelihood)
// in 512-token chunks to match n_ubatch. Importance[i] = -log P(token[i]|context).
// Scores are normalized to [0, 1] and optionally smoothed via pooling.
//
// For prompts > 4096 tokens: falls back to entropy-based proxy that uses
// lookahead_stats confidence weighted by token distance.
//
// Returns 0 on success, -1 on error.
int llama_spec_prefill_compute_importance(
    struct llama_spec_prefill_context * ctx,
    const llama_token * prompt_tokens,
    int n_prompt,
    std::vector<float> & token_importance
);

// Apply average pooling to smooth importance scores (like vLLM's avg_pool1d).
// If kernel_size <= 1, the input is returned unchanged (no-op).
void llama_spec_prefill_apply_pooling(
    std::vector<float> & importance,
    int kernel_size
);

// Filter tokens based on importance scores (top-k selection).
// Keeps the n_keep = ceil(n_prompt * keep_ratio) most important tokens,
// sorted by position to maintain original order.
// Returns 0 on success, -1 on error.
int llama_spec_prefill_filter_tokens(
    struct llama_spec_prefill_context * ctx,
    const llama_token * prompt_tokens,
    const int * prompt_positions,
    int n_prompt,
    const std::vector<float> & token_importance,
    float keep_ratio,
    std::vector<llama_token> & filtered_tokens,
    std::vector<int> & filtered_positions
);

// Filter tokens using chunk-based selection (like vLLM).
// Splits tokens into chunks, computes average importance per chunk,
// keeps top-k chunks, then includes all tokens from kept chunks.
// Includes a fallback to fill remaining slots from high-importance tokens.
// Returns 0 on success, -1 on error.
int llama_spec_prefill_filter_tokens_chunked(
    struct llama_spec_prefill_context * ctx,
    const llama_token * prompt_tokens,
    const int * prompt_positions,
    int n_prompt,
    const std::vector<float> & token_importance,
    float keep_ratio,
    int chunk_size,
    std::vector<llama_token> & filtered_tokens,
    std::vector<int> & filtered_positions
);

// Process filtered tokens with the base model.
// IMPORTANT: Positions are re-indexed to 0..n_filtered-1 because llama.cpp's
// batch allocator requires contiguous positions. This breaks RoPE for
// scattered positions — a known limitation.
// Returns 0 on success, -1 on error.
int llama_spec_prefill_process_base(
    struct llama_spec_prefill_context * ctx,
    const llama_token * filtered_tokens,
    const int * filtered_positions,
    int n_filtered
);

// Set path to dump parity traces (JSONL format, one record per call).
// Each record contains n_prompt, n_kept, keep_ratio, n_lookahead.
void llama_spec_prefill_set_dump_path(
    struct llama_spec_prefill_context * ctx,
    const char * path
);

// End-to-end speculative prefill pipeline.
// Orchestrates: lookahead → extract_qk → compute_attention (stub) →
// compute_importance → filter → process_base.
// Returns the number of filtered tokens kept, or -1 on error.
int llama_spec_prefill(
    struct llama_spec_prefill_context * ctx,
    const llama_token * prompt_tokens,
    int n_prompt,
    int n_lookahead,
    float keep_ratio
);
