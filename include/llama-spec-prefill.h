#pragma once

#include "llama.h"
#include <vector>

// Speculative Prefill Context
struct llama_spec_prefill_context {
    llama_context * ctx_base;  // Base model context
    llama_context * ctx_spec;  // Speculative (smaller) model context
};

// Initialize speculative prefill context
llama_spec_prefill_context * llama_spec_prefill_init(
    llama_context * ctx_base,
    llama_context * ctx_spec
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

// Compute token importance scores
int llama_spec_prefill_compute_importance(
    llama_spec_prefill_context * ctx,
    const llama_token * prompt_tokens,
    int n_prompt,
    std::vector<float> & token_importance
);

// Filter tokens based on importance scores
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

// Process filtered tokens with base model
int llama_spec_prefill_process_base(
    llama_spec_prefill_context * ctx,
    const llama_token * filtered_tokens,
    const int * filtered_positions,
    int n_filtered
);

// End-to-end speculative prefill pipeline
int llama_spec_prefill(
    llama_spec_prefill_context * ctx,
    const llama_token * prompt_tokens,
    int n_prompt,
    int n_lookahead,
    float keep_ratio
);
