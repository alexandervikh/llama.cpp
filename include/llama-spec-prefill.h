#pragma once

#include "llama.h"
#include <string>
#include <vector>

// Lookahead generation statistics for importance scoring
struct llama_lookahead_stat {
    float max_logit;    // Maximum logit value (prediction confidence)
    float entropy;      // Entropy of logit distribution (uncertainty)
    int position;       // Position in sequence
};

// Configuration for speculative prefill (matches vLLM config structure)
struct llama_spec_prefill_params {
    float keep_ratio;       // Percentage of tokens to keep (0.1 = 10%)
    int n_lookahead;        // Number of lookahead tokens to generate
    int pool_kernel_size;   // Pooling kernel size for smoothing (0 = disabled, vLLM uses 13)
    bool use_chunking;      // Whether to use chunk-based selection
    int chunk_size;         // Size of chunks for chunked selection (vLLM uses 32)
    bool ignore_eos;        // Whether to ignore EOS tokens during lookahead
    std::vector<llama_token> eos_tokens;  // EOS token IDs to detect
    
    // Default constructor with vLLM-like defaults
    llama_spec_prefill_params() :
        keep_ratio(0.5f),
        n_lookahead(8),
        pool_kernel_size(13),
        use_chunking(true),
        chunk_size(32),
        ignore_eos(false) {}
};

// Speculative Prefill Context
struct llama_spec_prefill_context {
    llama_context * ctx_base;  // Base model context
    llama_context * ctx_spec;  // Speculative (smaller) model context
    std::vector<llama_lookahead_stat> lookahead_stats;  // Statistics from lookahead generation
    llama_spec_prefill_params params;  // Configuration parameters
    int actual_lookahead_cnt;  // Actual number of lookahead tokens (may be less if EOS hit)
    std::string dump_path;     // Optional path to dump parity traces (empty = disabled)
};

// Initialize speculative prefill context
llama_spec_prefill_context * llama_spec_prefill_init(
    llama_context * ctx_base,
    llama_context * ctx_spec
);

// Initialize speculative prefill context with custom parameters
llama_spec_prefill_context * llama_spec_prefill_init_with_params(
    llama_context * ctx_base,
    llama_context * ctx_spec,
    const llama_spec_prefill_params & params
);

// Free speculative prefill context
void llama_spec_prefill_free(llama_spec_prefill_context * ctx);

// Generate lookahead tokens using spec model
int llama_spec_prefill_generate_lookahead(
    llama_spec_prefill_context * ctx,
    const llama_token * prompt_tokens,
    int n_prompt,
    llama_token * lookahead_tokens,
    int n_lookahead
);

// Extract Q and K tensors from KV cache (simplified stub for POC)
int llama_spec_prefill_extract_qk(
    llama_spec_prefill_context * ctx,
    const llama_token * lookahead_tokens,
    int n_lookahead
);

// Compute attention scores (simplified stub for POC)
int llama_spec_prefill_compute_attention(
    llama_spec_prefill_context * ctx,
    const llama_token * prompt_tokens,
    int n_prompt,
    const llama_token * lookahead_tokens,
    int n_lookahead
);

// Compute token importance scores with pooling/smoothing
int llama_spec_prefill_compute_importance(
    llama_spec_prefill_context * ctx,
    const llama_token * prompt_tokens,
    int n_prompt,
    std::vector<float> & token_importance
);

// Apply average pooling to smooth importance scores (like vLLM's avg_pool1d)
void llama_spec_prefill_apply_pooling(
    std::vector<float> & importance,
    int kernel_size
);

// Filter tokens based on importance scores (supports chunk mode)
int llama_spec_prefill_filter_tokens(
    llama_spec_prefill_context * ctx,
    const llama_token * prompt_tokens,
    const int * prompt_positions,
    int n_prompt,
    const std::vector<float> & token_importance,
    float keep_ratio,
    std::vector<llama_token> & filtered_tokens,
    std::vector<int> & filtered_positions
);

// Filter tokens using chunk-based selection (like vLLM)
int llama_spec_prefill_filter_tokens_chunked(
    llama_spec_prefill_context * ctx,
    const llama_token * prompt_tokens,
    const int * prompt_positions,
    int n_prompt,
    const std::vector<float> & token_importance,
    float keep_ratio,
    int chunk_size,
    std::vector<llama_token> & filtered_tokens,
    std::vector<int> & filtered_positions
);

// Process filtered tokens with base model
int llama_spec_prefill_process_base(
    llama_spec_prefill_context * ctx,
    const llama_token * filtered_tokens,
    const int * filtered_positions,
    int n_filtered
);

// Set path to dump parity traces (JSONL format, one record per call)
void llama_spec_prefill_set_dump_path(
    llama_spec_prefill_context * ctx,
    const char * path
);

// End-to-end speculative prefill pipeline
int llama_spec_prefill(
    llama_spec_prefill_context * ctx,
    const llama_token * prompt_tokens,
    int n_prompt,
    int n_lookahead,
    float keep_ratio
);
