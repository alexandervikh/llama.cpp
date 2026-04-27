#pragma once

// LazyLLM — draft-free progressive token pruning for long-context prefill.
// Reference: arXiv:2407.14057 (Fu et al., Apple, 2024).
//
// STATUS: PROOF-OF-CONCEPT
//
// Phase 1: attention extraction with flash_attn disabled (FA=off for LazyLLM
//   scoring context only; the user's main inference context is unaffected).
// Phase 2+: in-graph side-channel softmax for FA=on (see llama-graph.cpp).
//
// Known limitations:
// - FA must be disabled for score extraction in Phase 1.
// - Llama-2 / Qwen architecture only; MoE and recurrent not tested.
// - Position handling: kept tokens use ORIGINAL positions (no re-indexing),
//   which is the key difference from spec-prefill's contiguous re-index.

#include "llama.h"
#include <vector>
#include <unordered_map>
#include <cstdint>

struct ggml_tensor;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

struct llama_lazyllm_params {
    // Pruning schedule: one entry per pruning point.
    // pruning_layers[i] = layer index AFTER which pruning occurs.
    // keep_ratios[i]    = fraction of tokens to keep at that point.
    // Must have pruning_layers.size() == keep_ratios.size().
    std::vector<int>   pruning_layers;   // e.g. {8, 16, 24}
    std::vector<float> keep_ratios;      // e.g. {0.7f, 0.5f, 0.3f}

    int   pool_kernel_size = 13;   // avg-pool window for score smoothing (1 = off)
    bool  verbose          = false;

    // Paper-recommended default schedule for Llama-2-7B (32 layers):
    llama_lazyllm_params() {
        pruning_layers = {8, 16, 24};
        keep_ratios    = {0.7f, 0.5f, 0.3f};
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Aux Cache — stores hidden states of dropped tokens for decode-stage revival
// ─────────────────────────────────────────────────────────────────────────────

struct llama_lazyllm_aux_cache {
    // Per-pruning-point slabs of hidden states for dropped tokens.
    // slab[i] = flat float array, shape [n_dropped_at_point_i, n_embd].
    std::vector<std::vector<float>> slabs;

    // Map: original token position → (pruning_point_index, slab_row_offset).
    std::unordered_map<int32_t, std::pair<int32_t, int32_t>> index;

    int32_t n_embd = 0;  // model hidden dimension

    void reset() {
        for (auto & s : slabs) s.clear();
        index.clear();
    }

    // Store hidden state for a dropped token at a specific pruning point.
    // hidden: pointer to n_embd floats.
    void store(int32_t token_pos, int32_t pruning_point, int32_t embd, const float * hidden);

    // Retrieve stored hidden state. Returns nullptr if not found.
    const float * get(int32_t token_pos) const;

    // Which pruning point this token was dropped at. Returns -1 if not found.
    int32_t get_pruning_point(int32_t token_pos) const;

    // Total bytes stored.
    size_t total_bytes() const;
};

// ─────────────────────────────────────────────────────────────────────────────
// Context (opaque handle managed by init/free)
// ─────────────────────────────────────────────────────────────────────────────

struct llama_lazyllm_context {
    struct llama_context * ctx_base;   // the main inference context
    llama_lazyllm_params   params;

    // Current-run state (reset per llama_lazyllm_prefill call):
    std::vector<int32_t>  kept_indices;   // token positions kept after last pruning
    std::vector<float>    last_scores;    // importance scores from last extraction

    llama_lazyllm_aux_cache aux_cache;

    bool aux_cache_enabled = false;

    // Phase 5: decode-stage KV pruning state
    std::vector<llama_pos> alive_positions;    // positions currently in KV cache for decode_seq_id
    llama_seq_id           decode_seq_id           = 0;
    int32_t                decode_step             = 0;   // how many decode steps done
    bool                   decode_pruning_enabled  = false;
    float                  decode_keep_ratio       = 0.7f; // fraction of old positions to keep per step
};

// ─────────────────────────────────────────────────────────────────────────────
// Lifecycle
// ─────────────────────────────────────────────────────────────────────────────

// Create a LazyLLM context. Uses default params.
// Returns nullptr on error. Caller must call llama_lazyllm_free().
struct llama_lazyllm_context * llama_lazyllm_init(
    struct llama_context * ctx_base);

// Create with custom params.
struct llama_lazyllm_context * llama_lazyllm_init_with_params(
    struct llama_context * ctx_base,
    const llama_lazyllm_params & params);

// Free all resources.
void llama_lazyllm_free(struct llama_lazyllm_context * ctx);

// ─────────────────────────────────────────────────────────────────────────────
// Phase 1 — Attention score extraction (graph-based, FA=off)
// ─────────────────────────────────────────────────────────────────────────────

// Extract per-token importance scores from a layer's kq_soft_max tensor.
//
// Precondition: ctx_base has just run decode_partial() with flash_attn=false
//   so that kq_soft_max tensors exist in the computation graph.
//
// layer:      which transformer layer's attention to extract
// last_pos:   index of the query position to read from (use n_tokens-1 for last)
// out_scores: output vector, size = n_tokens; scores[i] = mean-over-heads attention
//             weight from token `last_pos` to token `i`
//
// Returns 0 on success, -1 on error (tensor not found, FA might be on, etc.)
int llama_lazyllm_extract_attention(
    struct llama_lazyllm_context * ctx,
    int layer,
    int last_pos,
    std::vector<float> & out_scores);

// ─────────────────────────────────────────────────────────────────────────────
// Score processing utilities
// ─────────────────────────────────────────────────────────────────────────────

// Apply 1D average pooling to smooth importance scores (paper §3.2).
// If pool_kernel_size <= 1, input is returned unchanged.
void llama_lazyllm_apply_pooling(
    std::vector<float> & scores,
    int pool_kernel_size);

// Select top-keep_ratio tokens by score.
// Always keeps the last token (index n_tokens-1).
// Returns kept indices in original ascending order (not sorted by score).
// n_tokens: total tokens; keep_ratio: fraction to keep.
std::vector<int32_t> llama_lazyllm_top_k_indices(
    const std::vector<float> & scores,
    float keep_ratio,
    int n_tokens);

// ─────────────────────────────────────────────────────────────────────────────
// Phase 2 — Single-point prefill orchestration
// ─────────────────────────────────────────────────────────────────────────────

// Single-pruning-point prefill.
//
// Flow (Phase 1 POC — simplified clear-and-re-decode):
//   1. decode_partial([0, l_prune)) with FA=off → fills KV for early layers
//   2. Extract attention scores at layer l_prune-1
//   3. Apply pooling; compute top-k kept indices
//   4. Clear KV; decode_partial([0, n_layer)) on kept tokens at ORIGINAL positions
//
// True two-stage KV sharing (passing hidden states as embeddings for il_start=l_prune)
// is Phase 2 and is left as a TODO.
//
// batch:            the full prompt batch (tokens, positions)
// l_prune:          pruning layer index (prune after this layer)
// keep_ratio:       fraction of tokens to retain
// out_kept_indices: output: which token positions were kept (original positions)
//
// Returns n_kept on success, -1 on error.
int llama_lazyllm_prefill_single(
    struct llama_lazyllm_context * ctx,
    const struct llama_batch & batch,
    int l_prune,
    float keep_ratio,
    std::vector<int32_t> & out_kept_indices);

// ─────────────────────────────────────────────────────────────────────────────
// Phase 3 — Multi-point progressive prefill
// ─────────────────────────────────────────────────────────────────────────────

// Full progressive prefill using ctx->params schedule.
//
// Flow (Phase 3 POC — simplified clear-and-re-decode per pruning point):
//   for each (l_prune, keep_ratio) in params.pruning_layers × params.keep_ratios:
//     decode_partial([0, l_prune)) on currently-surviving tokens
//     extract scores at l_prune-1
//     prune to top-keep_ratio tokens with original positions
//   clear KV; decode_partial([0, n_layer)) on surviving tokens
//
// Optionally writes dropped-token hidden states to ctx->aux_cache.
//
// Returns number of tokens surviving all pruning stages, -1 on error.
int llama_lazyllm_prefill(
    struct llama_lazyllm_context * ctx,
    const struct llama_batch & batch);

// Warmup: pre-compile all partial CUDA graph configurations used by this ctx.
// Call once before the first timed llama_lazyllm_prefill.  After warmup, all
// partial CUDA kernels are compiled and cached, eliminating JIT overhead.
//
// batch: a representative batch (same n_tokens as the real batch; content
//        does not matter — the warmup is discarded).
// Returns 0 on success.
int llama_lazyllm_warmup(
    struct llama_lazyllm_context * ctx,
    const struct llama_batch & batch);

// ─────────────────────────────────────────────────────────────────────────────
// Aux Cache API
// ─────────────────────────────────────────────────────────────────────────────

// Enable aux cache writes during next llama_lazyllm_prefill call.
void llama_lazyllm_aux_cache_enable(struct llama_lazyllm_context * ctx);

// Disable aux cache writes.
void llama_lazyllm_aux_cache_disable(struct llama_lazyllm_context * ctx);

// Reset aux cache (call between requests).
void llama_lazyllm_aux_cache_reset(struct llama_lazyllm_context * ctx);

// Retrieve stored hidden state for a dropped token. Returns nullptr if not cached.
const float * llama_lazyllm_aux_cache_get(
    struct llama_lazyllm_context * ctx,
    int32_t token_pos);

// Total bytes stored in aux cache.
size_t llama_lazyllm_aux_cache_bytes(struct llama_lazyllm_context * ctx);

// ─────────────────────────────────────────────────────────────────────────────
// Phase 5a — Decode-stage dynamic KV pruning
// ─────────────────────────────────────────────────────────────────────────────

// Enable decode-stage KV pruning.
// Must be called before the first llama_lazyllm_decode_step().
//   keep_ratio: fraction of existing KV positions to retain each step (0 < keep_ratio <= 1)
//   seq_id:     sequence id whose KV entries are managed
void llama_lazyllm_decode_pruning_enable(
        struct llama_lazyllm_context * lz_ctx,
        float keep_ratio,
        llama_seq_id seq_id);

// Single decode step with KV pruning.
// Call this INSTEAD of llama_decode() during token generation.
// batch must contain exactly one token.
//
// Algorithm:
//   1. llama_decode(ctx_base, batch) — normal forward pass
//   2. Extract attention scores from each configured pruning layer
//   3. Aggregate scores across layers (mean)
//   4. Remove KV entries for old positions that score below the keep threshold
//   5. Record the new token's position as alive
//
// Returns the total number of alive KV positions, or -1 on error.
int llama_lazyllm_decode_step(
        struct llama_lazyllm_context * lz_ctx,
        const struct llama_batch     & batch);

// Return the current number of alive KV positions.
int32_t llama_lazyllm_decode_n_alive(
        const struct llama_lazyllm_context * lz_ctx);

// ─────────────────────────────────────────────────────────────────────────────
// Phase 5b — Revival via Aux Cache
// ─────────────────────────────────────────────────────────────────────────────
//
// When a token that was dropped at pruning layer L is needed again during
// decode (it re-enters the top-k attention set), this function revives it:
//   1. Retrieve hidden state from aux cache (stored at layer L)
//   2. Build a single-token embedding batch at that hidden state
//   3. Run decode_partial([L, n_layers)) for just that one token
//   4. After this call, the token has valid KV entries for all layers L..n_layers
//   5. Add token_pos to alive_positions
//
// Call before llama_lazyllm_decode_step when you want to pre-revive tokens.
// aux_cache_enable() must have been called before llama_lazyllm_prefill().
//
// token_pos: the original token position (as used during prefill)
// Returns 0 on success, -1 if token not in aux cache or revival failed.
int llama_lazyllm_revive_token(
        struct llama_lazyllm_context * lz_ctx,
        int32_t token_pos);
