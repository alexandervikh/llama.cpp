#pragma once

// Self-layer prefill — draft-free token importance from base-model Q/K tensors
// extracted from the post-decode computation graph (see SELF_LAYER_PREFILL_PLAN.md).
//
// Limitations:
// - Build llama_context with flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED so
//   the graph exposes per-layer Qcur/Kcur tensors (FlashAttention fuses them away).
// - Requires n_prompt <= llama_n_batch(ctx) so the last graph covers all tokens.
// - After scoring, filtered prefill re-runs all layers on kept tokens (re-indexed
//   positions), matching spec-prefill's contiguous-batch constraint.

#include "llama.h"

#include <cstdint>
#include <vector>

struct llama_self_layer_prefill_params {
    int   n_early_layers   = 8;    // "shallow": layers [0, n_early_layers)
    float keep_ratio       = 0.25f;
    int   pool_kernel_size = 13;   // 1 = disabled
    bool  use_chunking     = false;
    int   chunk_size       = 32;

    llama_self_layer_prefill_params() = default;
};

// Run one full-graph prefill, extract per-layer Q/K, compute mean (over heads)
// last-query -> key softmax attention mass per token, aggregated over layer ranges:
//   out_shallow: layers [0, n_early_layers)
//   out_deep:    layers [n_early_layers, n_layer)
//   out_full:    all layers
// Also fills pearson_shallow_vs_full and pearson_shallow_vs_deep when non-null.
// Returns 0 on success, -1 on argument / decode / extract failure, -2 if
// n_prompt > llama_n_batch(ctx).
int llama_self_layer_prefill_attention_profiles(
    struct llama_context * ctx,
    const llama_token * prompt_tokens,
    int n_prompt,
    const llama_self_layer_prefill_params * params,
    std::vector<float> & out_shallow,
    std::vector<float> & out_deep,
    std::vector<float> & out_full,
    float * pearson_shallow_vs_full,
    float * pearson_shallow_vs_deep);

// End-to-end: profiles + optional pooling + filter + full re-decode on kept
// tokens (contiguous positions 0..n_kept-1). Returns n_kept, or -1 / -2 on error.
int llama_self_layer_prefill(
    struct llama_context * ctx,
    const llama_token * prompt_tokens,
    int n_prompt,
    const llama_self_layer_prefill_params * params);

// Phase 1 (Option C): one full prefill, then prune low-importance positions from the
// KV cache via llama_memory_seq_rm. TTFT ~= baseline (single decode); decode after this
// is faster and KV memory smaller. The last prompt position is always kept; original
// positions are preserved (no remap, RoPE phases stay valid).
// Returns n_kept (>=1), -1 on argument/decode/extract failure, -2 if n_prompt > n_batch.
int llama_self_layer_prefill_with_kv_prune(
    struct llama_context * ctx,
    const llama_token * prompt_tokens,
    int n_prompt,
    const llama_self_layer_prefill_params * params);

// Initialize self-layer prefill caching for a fixed n_early value.
// Call this once at startup to pre-cache the partial graph for layers [0, n_early).
// This eliminates graph-rebuild overhead on every llama_self_layer_prefill_partial call.
// Returns 0 on success, -1 on invalid n_early.
int llama_self_layer_prefill_init(
    struct llama_context * ctx,
    int n_early);

// Check if self-layer prefill caching is initialized.
bool llama_self_layer_prefill_is_init(struct llama_context * ctx);

// Get the cached n_early value (0 if not initialized).
int llama_self_layer_prefill_get_n_early(struct llama_context * ctx);

// Phase 2 (Option A, simplified): partial-layer score followed by full re-decode on the
// kept tokens. Two passes:
//
//   1. score:  partial decode of layers [0, n_early_layers) on all n_prompt tokens,
//              extract Q/K and compute per-token importance via the cheap last-row proxy.
//   2. resume: memory_clear + standard llama_decode on the K kept tokens at contiguous
//              positions 0..K-1.
//
// Speedup math (vs baseline L*F): total = N*F + L*K, win when K/F < 1 - N/L.
//
// Note: this is NOT the plan's literal "KV-shared partial resume" (the unified KV cache
// allocates slots per (seq, position) across all layers, so reusing layers [0,N) KV in
// the resume pass while writing fresh layers [N,L) at the same positions is not
// supported by the current memory module). The simplified architecture preserves the
// same TTFT-win condition with a small constant-factor cost (resume re-projects Q/K
// for layers [0,N) on K tokens).
//
// Identity gate: with N=0 and kr=1.0 this becomes baseline llama_decode (zero-layer
// score is short-circuited), which makes byte-equality vs baseline hold.
//
// Requirements: model arch must be LLM_ARCH_LLAMA or LLM_ARCH_LLAMA4 (non-iSWA), and
// llama_context must be built with flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED so
// per-layer Qcur/Kcur are visible in the partial graph.
//
// Returns n_kept on success, -1 on argument/decode/extract failure, -2 if
// n_prompt > llama_n_batch(ctx), -3 if arch not supported.
int llama_self_layer_prefill_partial(
    struct llama_context * ctx,
    const llama_token * prompt_tokens,
    int n_prompt,
    const llama_self_layer_prefill_params * params);
