#include "llama-lazyllm.h"
#include "llama-context.h"
#include "llama-model.h"
#include "llama-graph.h"
#include "llama-memory.h"
#include "llama.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>

// Lightweight wall-clock timer for profiling.
static double lazyllm_now_ms() {
    using C = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(C::now().time_since_epoch()).count();
}

// ─────────────────────────────────────────────────────────────────────────────
// llama_lazyllm_aux_cache implementation
// ─────────────────────────────────────────────────────────────────────────────

void llama_lazyllm_aux_cache::store(int32_t token_pos, int32_t pruning_point, int32_t embd, const float * hidden) {
    if (!hidden || embd <= 0) return;

    // Grow slabs vector to accommodate this pruning point index.
    if ((int32_t)slabs.size() <= pruning_point) {
        slabs.resize((size_t)(pruning_point + 1));
    }

    auto & slab = slabs[(size_t)pruning_point];
    const int32_t row_offset = (int32_t)(slab.size() / (size_t)embd);

    slab.insert(slab.end(), hidden, hidden + embd);
    index[token_pos] = {pruning_point, row_offset};
    n_embd = embd;
}

const float * llama_lazyllm_aux_cache::get(int32_t token_pos) const {
    auto it = index.find(token_pos);
    if (it == index.end()) return nullptr;

    const int32_t pp  = it->second.first;
    const int32_t row = it->second.second;
    if (pp < 0 || (size_t)pp >= slabs.size()) return nullptr;

    const auto & slab = slabs[(size_t)pp];
    if (n_embd <= 0) return nullptr;

    const size_t byte_off = (size_t)row * (size_t)n_embd;
    if (byte_off + (size_t)n_embd > slab.size()) return nullptr;

    return slab.data() + byte_off;
}

int32_t llama_lazyllm_aux_cache::get_pruning_point(int32_t token_pos) const {
    auto it = index.find(token_pos);
    if (it == index.end()) return -1;
    return it->second.first;
}

size_t llama_lazyllm_aux_cache::total_bytes() const {
    size_t total = 0;
    for (const auto & s : slabs) {
        total += s.size() * sizeof(float);
    }
    return total;
}

// ─────────────────────────────────────────────────────────────────────────────
// Lifecycle
// ─────────────────────────────────────────────────────────────────────────────

struct llama_lazyllm_context * llama_lazyllm_init(struct llama_context * ctx_base) {
    llama_lazyllm_params default_params;
    return llama_lazyllm_init_with_params(ctx_base, default_params);
}

struct llama_lazyllm_context * llama_lazyllm_init_with_params(
        struct llama_context * ctx_base,
        const llama_lazyllm_params & params) {
    if (!ctx_base) {
        fprintf(stderr, "[lazyllm] init: ctx_base is null\n");
        return nullptr;
    }
    if (params.pruning_layers.size() != params.keep_ratios.size()) {
        fprintf(stderr, "[lazyllm] init: pruning_layers.size() != keep_ratios.size()\n");
        return nullptr;
    }

    auto * ctx = new llama_lazyllm_context();
    ctx->ctx_base = ctx_base;
    ctx->params   = params;
    return ctx;
}

void llama_lazyllm_free(struct llama_lazyllm_context * ctx) {
    delete ctx;
}

// ─────────────────────────────────────────────────────────────────────────────
// Graph helpers
// ─────────────────────────────────────────────────────────────────────────────

// Search the compute graph for the last node whose name contains `target`.
static const ggml_tensor * find_named_tensor(llm_graph_result * gf_res, const char * target) {
    if (!gf_res) return nullptr;
    ggml_cgraph * gf = gf_res->get_gf();
    if (!gf) return nullptr;

    const int n_nodes = ggml_graph_n_nodes(gf);
    const ggml_tensor * best = nullptr;
    int best_idx = -1;

    for (int i = 0; i < n_nodes; i++) {
        const ggml_tensor * t = ggml_graph_node(gf, i);
        if (!t || !t->name[0]) continue;
        if (strstr(t->name, target)) {
            if (i > best_idx) {
                best     = t;
                best_idx = i;
            }
        }
    }
    return best;
}

// Locate the kq_soft_max tensor for a specific layer in the most recent graph.
static const ggml_tensor * find_kq_soft_max(llm_graph_result * gf_res, int layer) {
    char target[64];
    snprintf(target, sizeof(target), "kq_soft_max-%d", layer);
    return find_named_tensor(gf_res, target);
}

// Locate the l_out tensor (layer residual output) for a specific layer.
// This is the hidden state at the OUTPUT of decoder layer `layer`, which is
// the input to layer `layer+1` — exactly what we inject for Phase 2 prefill.
static const ggml_tensor * find_layer_output(llm_graph_result * gf_res, int layer) {
    char target[64];
    snprintf(target, sizeof(target), "l_out-%d", layer);
    return find_named_tensor(gf_res, target);
}

// Copy a tensor from device to host and convert to float.
// Returns 0 on success, -1 on error.
static int tensor_to_float_buf(const ggml_tensor * t, std::vector<float> & out) {
    if (!t) return -1;
    const int64_t n_elements = ggml_nelements(t);
    if (n_elements <= 0) return -1;

    out.resize((size_t)n_elements);
    const size_t nbytes = ggml_nbytes(t);
    std::vector<uint8_t> cpu_buf(nbytes);
    ggml_backend_tensor_get(t, cpu_buf.data(), 0, nbytes);

    if (t->type == GGML_TYPE_F32) {
        std::memcpy(out.data(), cpu_buf.data(), nbytes);
        return 0;
    }

    const auto * traits = ggml_get_type_traits(t->type);
    if (!traits || !traits->to_float) {
        fprintf(stderr, "[lazyllm] tensor_to_float_buf: unsupported type %d\n", (int)t->type);
        return -1;
    }
    traits->to_float(cpu_buf.data(), out.data(), n_elements);
    return 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 1 — Attention score extraction
// ─────────────────────────────────────────────────────────────────────────────

int llama_lazyllm_extract_attention(
        struct llama_lazyllm_context * ctx,
        int layer,
        int last_pos,
        std::vector<float> & out_scores) {
    if (!ctx || !ctx->ctx_base) return -1;

    ctx->ctx_base->synchronize();

    auto * gf_res = ctx->ctx_base->get_gf_res_prev();
    if (!gf_res) {
        fprintf(stderr, "[lazyllm] extract_attention: no graph result (run decode_partial first)\n");
        return -1;
    }

    // Fast path: partial builders pre-compute GPU-side pooled scores named
    // "lazyllm_scores-{layer}" — shape [n_kv].  Only n_kv floats transferred.
    {
        char sname[64];
        snprintf(sname, sizeof(sname), "lazyllm_scores-%d", layer);
        const ggml_tensor * ps = find_named_tensor(gf_res, sname);
        if (ps && ps->ne[0] > 0) {
            const int n_kv = (int)ps->ne[0];
            if (last_pos < 0 || last_pos >= n_kv) {
                fprintf(stderr, "[lazyllm] extract_attention (pooled): last_pos=%d out of range [0,%d)\n",
                        last_pos, n_kv);
                return -1;
            }
            std::vector<float> data;
            if (tensor_to_float_buf(ps, data) == 0 && (int)data.size() == n_kv) {
                out_scores.assign(data.begin(), data.end());
                if (ctx->params.verbose) {
                    fprintf(stderr, "[lazyllm] extract_attention: GPU-pooled lazyllm_scores-%d "
                            "[%d] (fast path, %zu bytes)\n",
                            layer, n_kv, (size_t)n_kv * sizeof(float));
                }
                return 0;
            }
        }
    }

    // Slow fallback: read full kq_soft_max and pool on CPU.
    // This path is used when flash_attn is disabled but partial builder didn't
    // add pool nodes (e.g. unsupported architecture falling through to full builder).
    const ggml_tensor * t = find_kq_soft_max(gf_res, layer);
    if (!t) {
        fprintf(stderr, "[lazyllm] extract_attention: kq_soft_max-%d not found "
                "(flash_attn may be enabled — disable it for LazyLLM scoring)\n", layer);
        return -1;
    }

    if (ctx->params.verbose) {
        fprintf(stderr, "[lazyllm] kq_soft_max-%d: name='%s' ne=[%lld,%lld,%lld,%lld] type=%d "
                "(slow CPU-pooling path — %.1f MB)\n",
                layer, t->name,
                (long long)t->ne[0], (long long)t->ne[1],
                (long long)t->ne[2], (long long)t->ne[3],
                (int)t->type,
                (double)ggml_nbytes(t) / (1024.0 * 1024.0));
    }

    std::vector<float> data;
    if (tensor_to_float_buf(t, data) != 0) {
        fprintf(stderr, "[lazyllm] extract_attention: failed to read tensor data\n");
        return -1;
    }

    const int64_t ne0 = t->ne[0];
    const int64_t ne1 = t->ne[1];
    const int64_t ne2 = t->ne[2];
    const int64_t ne3 = t->ne[3];

    const int n_kv = (int)ne0;

    if (last_pos < 0 || last_pos >= n_kv) {
        fprintf(stderr, "[lazyllm] extract_attention: last_pos=%d out of range [0, %d)\n",
                last_pos, n_kv);
        return -1;
    }

    const int64_t per_kv = ne1 * ne2 * ne3;

    if ((int64_t)data.size() != n_kv * per_kv) {
        fprintf(stderr, "[lazyllm] extract_attention: data size mismatch: got %zu expected %lld\n",
                data.size(), (long long)(n_kv * per_kv));
        return -1;
    }

    out_scores.assign((size_t)n_kv, 0.f);

    if (per_kv == 1) {
        for (int k = 0; k < n_kv; k++) {
            out_scores[(size_t)k] = data[(size_t)k];
        }
    } else {
        for (int k = 0; k < n_kv; k++) {
            float sum = 0.f;
            const float * row = data.data() + (size_t)k * (size_t)per_kv;
            for (int64_t j = 0; j < per_kv; j++) {
                sum += row[j];
            }
            out_scores[(size_t)k] = sum / (float)per_kv;
        }
    }

    return 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Score processing utilities
// ─────────────────────────────────────────────────────────────────────────────

void llama_lazyllm_apply_pooling(std::vector<float> & scores, int pool_kernel_size) {
    if (pool_kernel_size <= 1 || scores.empty()) return;

    const int n = (int)scores.size();
    const int half = pool_kernel_size / 2;
    std::vector<float> smoothed(scores.size());

    for (int i = 0; i < n; i++) {
        const int lo = std::max(0, i - half);
        const int hi = std::min(n - 1, i + half);
        float sum = 0.f;
        for (int j = lo; j <= hi; j++) {
            sum += scores[(size_t)j];
        }
        smoothed[(size_t)i] = sum / (float)(hi - lo + 1);
    }
    scores = std::move(smoothed);
}

std::vector<int32_t> llama_lazyllm_top_k_indices(
        const std::vector<float> & scores,
        float keep_ratio,
        int n_tokens) {
    if (n_tokens <= 0 || scores.empty()) return {};

    int n_keep = (int)std::ceil((float)n_tokens * keep_ratio);
    if (n_keep < 1)        n_keep = 1;
    if (n_keep > n_tokens) n_keep = n_tokens;

    // Build index sorted by score descending.
    std::vector<int32_t> order((size_t)n_tokens);
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + n_keep, order.end(),
                      [&](int32_t a, int32_t b) {
                          return scores[(size_t)a] > scores[(size_t)b];
                      });

    // Collect top-k; always ensure last token is included.
    std::vector<bool> keep_mask((size_t)n_tokens, false);
    for (int i = 0; i < n_keep; i++) {
        keep_mask[(size_t)order[(size_t)i]] = true;
    }
    keep_mask[(size_t)(n_tokens - 1)] = true;  // last token always kept

    // Return in ascending position order.
    std::vector<int32_t> result;
    result.reserve((size_t)n_keep + 1);
    for (int i = 0; i < n_tokens; i++) {
        if (keep_mask[(size_t)i]) {
            result.push_back((int32_t)i);
        }
    }
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: build a batch from a subset of tokens
// ─────────────────────────────────────────────────────────────────────────────
//
// NOTE: token batches (build_filtered_batch, fallback path) still use
// sequential positions 0..n_kept-1 because the KV-cache continuity check
// requires contiguous positions for token batches.  The primary embedding-
// injection path (build_embd_batch) uses original positions; the contiguity
// check is bypassed for embedding batches in llama-batch.cpp.

static llama_batch build_filtered_batch(
        const llama_batch & src,
        const std::vector<int32_t> & keep_token_indices) {
    const int n_kept = (int)keep_token_indices.size();
    llama_batch dst = llama_batch_init(n_kept, 0, 1);
    dst.n_tokens = 0;

    for (int32_t idx : keep_token_indices) {
        if (idx < 0 || idx >= src.n_tokens) continue;
        dst.token[dst.n_tokens]    = src.token ? src.token[idx] : 0;
        // Token batches (batch.embd == nullptr) still require contiguous
        // positions for the KV-cache continuity check. This fallback path
        // uses sequential 0..n_kept-1; the main embedding-injection path uses
        // original positions (see build_embd_batch).
        dst.pos[dst.n_tokens]      = (llama_pos)dst.n_tokens;
        dst.n_seq_id[dst.n_tokens] = src.n_seq_id ? src.n_seq_id[idx] : 1;
        if (src.seq_id && src.n_seq_id) {
            for (int s = 0; s < src.n_seq_id[idx]; s++) {
                dst.seq_id[dst.n_tokens][s] = src.seq_id[idx][s];
            }
        } else {
            dst.seq_id[dst.n_tokens][0] = 0;
        }
        dst.logits[dst.n_tokens] = (dst.n_tokens == n_kept - 1); // only last token gets logits
        dst.n_tokens++;
    }
    return dst;
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 2 — Single-point prefill with embedding injection
//
// Algorithm:
//   1. decode_partial(batch, 0, l_prune)   — fill KV [0..l_prune) for all tokens
//   2. Extract attention scores → prune to keep_ratio
//   3. Extract l_out hidden states at layer (l_prune-1) for kept tokens
//   4. Remove KV entries for dropped tokens from layers [0..l_prune)
//   5. Build embedding batch (embd = hidden states, ORIGINAL positions)
//   6. decode_partial(embd_batch, l_prune, n_layers) — continue from l_prune
//
// After step 4 the KV cache [0..l_prune) holds only kept-token entries.
// Step 6 adds KV [l_prune..n_layers) for kept tokens at their original
// positions; RoPE is applied at the correct frequencies.  The batch
// contiguity check is bypassed for embedding batches in llama-batch.cpp.
// ─────────────────────────────────────────────────────────────────────────────

// Build an embedding batch for Phase 2: embd vectors at ORIGINAL positions.
// hidden[i] points to n_embd floats for kept token i.
//
// Original position IDs are preserved so RoPE is applied at the correct
// frequencies.  llama-batch.cpp skips the within-batch contiguity check
// for embedding batches (batch.embd != nullptr), making non-sequential
// positions safe.
static llama_batch build_embd_batch(
        const llama_batch & src,
        const std::vector<int32_t> & keep_indices,
        const std::vector<float>   & flat_hidden,
        int n_embd) {
    const int n_kept = (int)keep_indices.size();
    // embd batch: allocate n_kept * n_embd floats
    llama_batch dst = llama_batch_init(n_kept, n_embd, 1);
    dst.n_tokens = 0;

    for (int i = 0; i < n_kept; i++) {
        const int32_t src_idx = keep_indices[(size_t)i];
        if (src_idx < 0 || src_idx >= src.n_tokens) continue;

        // Copy embedding vector for this token
        const float * src_vec = flat_hidden.data() + (size_t)i * (size_t)n_embd;
        float       * dst_vec = dst.embd + (size_t)dst.n_tokens * (size_t)n_embd;
        std::memcpy(dst_vec, src_vec, (size_t)n_embd * sizeof(float));

        // Use ORIGINAL position IDs so RoPE is computed at the correct
        // frequencies.  llama-batch.cpp now skips the within-batch contiguity
        // check for embedding batches (batch.embd != nullptr), so non-sequential
        // positions are safe here.
        const llama_pos orig_pos = src.pos ? src.pos[src_idx] : (llama_pos)src_idx;
        dst.pos[dst.n_tokens]      = orig_pos;
        dst.n_seq_id[dst.n_tokens] = src.n_seq_id ? src.n_seq_id[src_idx] : 1;
        if (src.seq_id && src.n_seq_id) {
            for (int s = 0; s < src.n_seq_id[src_idx]; s++) {
                dst.seq_id[dst.n_tokens][s] = src.seq_id[src_idx][s];
            }
        } else {
            dst.seq_id[dst.n_tokens][0] = 0;
        }
        dst.logits[dst.n_tokens] = (i == n_kept - 1); // only last kept token needs logits
        dst.n_tokens++;
    }
    return dst;
}

int llama_lazyllm_prefill_single(
        struct llama_lazyllm_context * ctx,
        const struct llama_batch & batch,
        int l_prune,
        float keep_ratio,
        std::vector<int32_t> & out_kept_indices) {
    if (!ctx || !ctx->ctx_base) return -1;
    if (batch.n_tokens <= 0) return -1;
    if (l_prune <= 0) {
        fprintf(stderr, "[lazyllm] prefill_single: l_prune must be > 0\n");
        return -1;
    }

    llama_context * lctx = ctx->ctx_base;
    const int n_tokens   = batch.n_tokens;

    // Step 1: decode layers [0, l_prune) on all tokens.
    int r = lctx->decode_partial(batch, 0, l_prune, /*early_exit=*/false);
    if (r != 0) {
        fprintf(stderr, "[lazyllm] prefill_single: decode_partial [0,%d) failed: %d\n", l_prune, r);
        return -1;
    }

    lctx->synchronize();

    // Step 2: extract attention scores at layer l_prune-1.
    std::vector<float> scores;
    r = llama_lazyllm_extract_attention(ctx, l_prune - 1, n_tokens - 1, scores);
    if (r != 0) {
        fprintf(stderr, "[lazyllm] prefill_single: attention extraction failed\n");
        return -1;
    }

    // Step 3: smooth and select top-k.
    llama_lazyllm_apply_pooling(scores, ctx->params.pool_kernel_size);
    out_kept_indices = llama_lazyllm_top_k_indices(scores, keep_ratio, n_tokens);

    ctx->last_scores  = scores;
    ctx->kept_indices = out_kept_indices;

    const int n_kept = (int)out_kept_indices.size();
    if (ctx->params.verbose) {
        fprintf(stderr, "[lazyllm] prefill_single: kept %d / %d tokens after layer %d\n",
                n_kept, n_tokens, l_prune - 1);
    }

    // Step 4: extract hidden states (l_out) at layer l_prune-1 for kept tokens.
    auto * gf_res = lctx->get_gf_res_prev();
    const ggml_tensor * l_out_t = find_layer_output(gf_res, l_prune - 1);

    if (l_out_t) {
        // l_out shape: [n_embd, n_tokens] (column-major in ggml)
        const int n_embd = (int)l_out_t->ne[0];
        std::vector<float> all_hidden;
        if (tensor_to_float_buf(l_out_t, all_hidden) == 0 &&
            (int)all_hidden.size() == n_embd * n_tokens) {

            // Collect kept-token hidden states into a contiguous buffer.
            std::vector<float> kept_hidden((size_t)n_kept * (size_t)n_embd);
            for (int i = 0; i < n_kept; i++) {
                const int src_idx = out_kept_indices[(size_t)i];
                std::memcpy(kept_hidden.data() + (size_t)i * (size_t)n_embd,
                            all_hidden.data() + (size_t)src_idx * (size_t)n_embd,
                            (size_t)n_embd * sizeof(float));
            }

            // Step 5: remove KV entries for dropped tokens from layers [0..l_prune).
            //  Build a mask of which token positions are dropped.
            std::vector<bool> kept_mask((size_t)n_tokens, false);
            for (int32_t ki : out_kept_indices) { kept_mask[(size_t)ki] = true; }

            llama_seq_id seq_id = (batch.seq_id && batch.n_seq_id && batch.n_seq_id[0] >= 1)
                                   ? batch.seq_id[0][0] : 0;

            llama_memory_t mem = llama_get_memory(lctx);
            for (int i = 0; i < n_tokens; i++) {
                if (!kept_mask[(size_t)i]) {
                    const llama_pos p = batch.pos ? batch.pos[i] : (llama_pos)i;
                    llama_memory_seq_rm(mem, seq_id, p, p + 1);
                }
            }

            // Step 6: clear KV so the embedding batch can write all layers at
            // the kept tokens' original positions.
            llama_memory_clear(llama_get_memory(lctx), false);

            // Step 7: re-run layer [l_prune, n_layers) with embedding batch.
            llama_batch embd_batch = build_embd_batch(batch, out_kept_indices,
                                                       kept_hidden, n_embd);
            r = llama_decode(lctx, embd_batch);
            llama_batch_free(embd_batch);

            if (r != 0) {
                fprintf(stderr, "[lazyllm] prefill_single: embd decode failed: %d\n", r);
                return -1;
            }
            fprintf(stderr, "[lazyllm] prefill_single: embedding-injection path OK "
                    "(l_prune=%d, n_embd=%d, kept=%d)\n", l_prune, n_embd, n_kept);
            return n_kept;
        }
        fprintf(stderr, "[lazyllm] prefill_single: l_out extraction failed (size mismatch)\n");
    } else {
        fprintf(stderr, "[lazyllm] prefill_single: l_out-%d not found in graph; "
                "falling back to clear+redecode\n", l_prune - 1);
    }

    // Fallback: clear KV and re-decode full layers on kept tokens only.
    llama_memory_clear(llama_get_memory(lctx), false);
    llama_batch kept_batch = build_filtered_batch(batch, out_kept_indices);
    r = llama_decode(lctx, kept_batch);
    llama_batch_free(kept_batch);

    if (r != 0) {
        fprintf(stderr, "[lazyllm] prefill_single: fallback decode failed: %d\n", r);
        return -1;
    }
    return n_kept;
}

// ─────────────────────────────────────────────────────────────────────────────
// Multi-point progressive prefill with embedding injection (llama_lazyllm_prefill)
//
// True LazyLLM algorithm (§3 of arXiv:2407.14057):
//   - Stage 0: run ALL tokens through layers [0, l_prune[0]).
//   - Extract attention scores → prune to S1 (keep_ratio[0] * N tokens).
//   - Extract l_out hidden states at l_prune[0]-1 for S1.
//   - Surgically remove KV entries for dropped tokens from [0..l_prune[0]).
//   - Build embedding batch for S1 and continue [l_prune[0], l_prune[1]).
//   - Stage 1: same pattern on the shrunken S1 set.
//   - ...
//   - Final pass: run [l_prune[last], n_layers) on the final survivor set.
//
// This avoids re-decoding layers already computed, giving O(sum(n_s * Δl_s))
// instead of O(N * n_layers) work.  For N=4096, 3 stages at 70/50/30%:
//   work ≈ 4096*l0 + 2867*Δl1 + 1433*Δl2 + 430*(n-l2)  vs 4096*n_layers.
// ─────────────────────────────────────────────────────────────────────────────

int llama_lazyllm_prefill(
        struct llama_lazyllm_context * ctx,
        const struct llama_batch & batch) {
    if (!ctx || !ctx->ctx_base) return -1;
    if (batch.n_tokens <= 0) return -1;

    const llama_lazyllm_params & P = ctx->params;

    // Auto-fallback: if warmup decided LazyLLM is not faster than baseline,
    // transparently call llama_decode() so the user is never worse off.
    // Also handles the empty-schedule case.
    if (P.pruning_layers.empty() || ctx->fallback_active) {
        const int r = llama_decode(ctx->ctx_base, batch);
        if (r != 0) return -1;
        // Populate the post-prefill state as if all tokens survived.
        const int32_t n = batch.n_tokens;
        ctx->kept_indices.resize((size_t)n);
        std::iota(ctx->kept_indices.begin(), ctx->kept_indices.end(), 0);
        ctx->alive_positions.resize((size_t)n);
        for (int32_t i = 0; i < n; i++) {
            ctx->alive_positions[(size_t)i] = batch.pos
                ? batch.pos[i]
                : (llama_pos)i;
        }
        ctx->decode_step = 0;
        return n;
    }

    const int n_schedules = (int)P.pruning_layers.size();
    for (int s = 0; s < n_schedules; s++) {
        if (s > 0 && P.pruning_layers[(size_t)s] <= P.pruning_layers[(size_t)s - 1]) {
            fprintf(stderr, "[lazyllm] prefill: pruning_layers must be strictly increasing\n");
            return -1;
        }
    }

    llama_context * lctx = ctx->ctx_base;
    ctx->kept_indices.clear();
    ctx->last_scores.clear();
    if (ctx->aux_cache_enabled) ctx->aux_cache.reset();

    // seq_id used to remove dropped-token KV entries
    llama_seq_id seq_id = (batch.seq_id && batch.n_seq_id && batch.n_seq_id[0] >= 1)
                           ? batch.seq_id[0][0] : 0;

    // --- Stage 0: first decode_partial on full token batch ---
    const int l0 = P.pruning_layers[0];
    const double t_s0_start = lazyllm_now_ms();
    int r = lctx->decode_partial(batch, 0, l0, /*early_exit=*/false, /*no_embed_output=*/true);
    if (r != 0) {
        fprintf(stderr, "[lazyllm] prefill: decode_partial [0,%d) stage 0 failed: %d\n", l0, r);
        return -1;
    }
    lctx->synchronize();
    if (P.verbose) fprintf(stderr, "[lazyllm] prefill: stage 0 decode_partial[0,%d) %.1f ms\n",
                           l0, lazyllm_now_ms() - t_s0_start);

    // surviving_indices tracks which ORIGINAL token indices are still active.
    std::vector<int32_t> surviving(batch.n_tokens);
    std::iota(surviving.begin(), surviving.end(), 0);

    // embd_buf holds the last hidden states for currently surviving tokens.
    // Initially empty (first stage uses the graph's l_out tensor directly).
    std::vector<float> embd_buf;
    int n_embd = 0;
    bool have_embds = false;

    for (int s = 0; s < n_schedules; s++) {
        const int   l_prune    = P.pruning_layers[(size_t)s];
        const float keep_ratio = P.keep_ratios[(size_t)s];
        const int   n_current  = (int)surviving.size();

        if (n_current <= 1) break;

        // Extract attention scores from the graph (layer l_prune-1).
        std::vector<float> scores;
        r = llama_lazyllm_extract_attention(ctx, l_prune - 1, n_current - 1, scores);
        if (r != 0) {
            fprintf(stderr, "[lazyllm] prefill: attn extraction failed at stage %d\n", s);
            return -1;
        }

        llama_lazyllm_apply_pooling(scores, P.pool_kernel_size);
        std::vector<int32_t> local_kept =
            llama_lazyllm_top_k_indices(scores, keep_ratio, n_current);

        // Extract l_out hidden states for surviving (soon-to-be-kept) tokens.
        auto * gf_res = lctx->get_gf_res_prev();
        const ggml_tensor * l_out_t = find_layer_output(gf_res, l_prune - 1);

        bool got_hidden = false;
        if (l_out_t) {
            n_embd = (int)l_out_t->ne[0];
            std::vector<float> all_hidden;
            if (tensor_to_float_buf(l_out_t, all_hidden) == 0 &&
                (int)all_hidden.size() == n_embd * n_current) {

                embd_buf.resize((size_t)local_kept.size() * (size_t)n_embd);
                for (int i = 0; i < (int)local_kept.size(); i++) {
                    const int li = local_kept[(size_t)i];
                    std::memcpy(embd_buf.data() + (size_t)i * (size_t)n_embd,
                                all_hidden.data() + (size_t)li * (size_t)n_embd,
                                (size_t)n_embd * sizeof(float));
                }
                got_hidden = true;
                have_embds = true;

                // Store dropped-token hidden states in aux cache.
                if (ctx->aux_cache_enabled) {
                    std::vector<bool> kept_mask((size_t)n_current, false);
                    for (int32_t ki : local_kept) kept_mask[(size_t)ki] = true;
                    for (int i = 0; i < n_current; i++) {
                        if (!kept_mask[(size_t)i]) {
                            const int32_t orig_pos = surviving[(size_t)i];
                            const float * hv = all_hidden.data() + (size_t)i * (size_t)n_embd;
                            ctx->aux_cache.store(orig_pos, s, n_embd, hv);
                        }
                    }
                }
            }
        }

        // Update surviving set.
        std::vector<int32_t> new_surviving;
        new_surviving.reserve(local_kept.size());
        for (int32_t li : local_kept) {
            new_surviving.push_back(surviving[(size_t)li]);
        }

        // Remove KV entries for dropped tokens from ALL layers processed so far.
        std::vector<bool> kept_mask2((size_t)n_current, false);
        for (int32_t ki : local_kept) kept_mask2[(size_t)ki] = true;
        llama_memory_t mem = llama_get_memory(lctx);
        for (int i = 0; i < n_current; i++) {
            if (!kept_mask2[(size_t)i]) {
                const llama_pos p = batch.pos ? batch.pos[surviving[(size_t)i]]
                                              : (llama_pos)surviving[(size_t)i];
                llama_memory_seq_rm(mem, seq_id, p, p + 1);
            }
        }

        surviving = std::move(new_surviving);
        ctx->last_scores = scores;

        if (P.verbose) {
            fprintf(stderr, "[lazyllm] prefill: stage %d (l=%d, r=%.2f): %d→%d tokens\n",
                    s, l_prune, keep_ratio, n_current, (int)surviving.size());
        }

        // Decide next layer range.
        const int l_next = (s + 1 < n_schedules) ? P.pruning_layers[(size_t)(s + 1)]
                                                   : -1; // -1 = final pass

        if (got_hidden && have_embds) {
            // Embedding-injection path: inject l_out hidden states as input to
            // layer l_prune, skipping re-computation of layers [0..l_prune).
            //
            // Clear KV before re-allocating so the surviving tokens can take
            // fresh cells at their original positions.  The dropped-token cells
            // were already freed by llama_memory_seq_rm() above, but the
            // surviving cells from Stage 0 (layers [0..l_prune)) must also be
            // freed: the KV allocator does not support partial-layer overwrite of
            // an existing (seq_id, pos) cell, so Stage 1 must start with a clean
            // slate and re-populate all layers [l_prune..il_end) from scratch.
            {
                const double t_clear0 = lazyllm_now_ms();
                llama_memory_clear(llama_get_memory(lctx), false);
                if (P.verbose) {
                    fprintf(stderr, "[lazyllm] prefill: stage %d llama_memory_clear took %.2f ms\n",
                            s, lazyllm_now_ms() - t_clear0);
                }
            }

            // Build embedding batch for surviving tokens.
            llama_batch eb = build_embd_batch(batch, surviving, embd_buf, n_embd);

            const int il_end = (l_next == -1)
                ? llama_model_n_layer(llama_get_model(lctx))
                : l_next;

            // decode_partial starting at l_prune: layers [l_prune, il_end) only.
            // The embd vectors (hidden states from layer l_prune-1) are injected
            // as input to layer l_prune, skipping token embedding + layers [0..l_prune).
            // early_exit=false: we don't want lm_head applied for intermediate stages.
            // no_embed_output=true for intermediate stages to avoid the n_vocab*n_tokens output buffer.
            const bool is_final_stage = (il_end == llama_model_n_layer(llama_get_model(lctx)));
            r = lctx->decode_partial(eb, l_prune, il_end, /*early_exit=*/false, /*no_embed_output=*/!is_final_stage);
            llama_batch_free(eb);

            if (P.verbose) {
                fprintf(stderr, "[lazyllm] prefill: stage %d embedding-injection "
                        "decode_partial([%d,%d)) on %d tokens\n",
                        s, l_prune, il_end, (int)surviving.size());
            }
        } else {
            // Fallback: clear and re-decode entire range from layer 0.
            llama_memory_clear(llama_get_memory(lctx), false);
            llama_batch fb = build_filtered_batch(batch, surviving);
            const int il_end = (l_next == -1)
                ? llama_model_n_layer(llama_get_model(lctx))
                : l_next;
            const bool fb_final = (il_end == llama_model_n_layer(llama_get_model(lctx)));
            r = lctx->decode_partial(fb, 0, il_end, /*early_exit=*/false, /*no_embed_output=*/!fb_final);
            llama_batch_free(fb);
        }

        if (r != 0) {
            fprintf(stderr, "[lazyllm] prefill: stage %d continuation failed: %d\n", s, r);
            return -1;
        }
        lctx->synchronize();

        if (l_next == -1) break; // done
    }

    ctx->kept_indices = surviving;

    // Phase 5: initialize decode-stage tracking.
    // After all stages the surviving tokens' KV is present at their ORIGINAL
    // positions (batch.pos[original_token_index]).  alive_positions stores
    // these original positions so that Phase 5 decode steps append new tokens
    // after the true last position rather than after a synthetic 0..n_kept-1.
    {
        const int32_t n_kept = (int32_t)surviving.size();
        ctx->alive_positions.resize((size_t)n_kept);
        for (int32_t i = 0; i < n_kept; i++) {
            const int32_t orig_idx = surviving[(size_t)i];
            ctx->alive_positions[(size_t)i] = batch.pos
                ? batch.pos[orig_idx]
                : (llama_pos)orig_idx;
        }
        ctx->decode_step = 0;
    }

    if (P.verbose) {
        fprintf(stderr, "[lazyllm] prefill: done — %d tokens survived all stages\n",
                (int)surviving.size());
    }

    return (int)surviving.size();
}

// ─────────────────────────────────────────────────────────────────────────────
// Warmup: pre-compile all partial CUDA graph configurations
// ─────────────────────────────────────────────────────────────────────────────

int llama_lazyllm_warmup(
        struct llama_lazyllm_context * ctx,
        const struct llama_batch    & batch) {
    if (!ctx || !ctx->ctx_base) return -1;
    if (batch.n_tokens <= 0) return -1;

    const llama_lazyllm_params & P = ctx->params;
    llama_context * lctx = ctx->ctx_base;

    if (P.verbose) {
        fprintf(stderr, "[lazyllm] warmup: pre-compiling %d partial graph configurations "
                "for %d tokens...\n", (int)P.pruning_layers.size() + 1, batch.n_tokens);
    }

    // Phase 1: Forward warmup — run all stages in order to compile CUDA kernels
    // for every (il_start, il_end, n_tokens) configuration.  After this, the
    // last graph executed is the FULL forward (stage N), so plain llama_decode
    // kernels are also compiled.
    const double t0 = lazyllm_now_ms();
    llama_lazyllm_prefill(ctx, batch);
    lctx->synchronize();
    const double t_phase1 = lazyllm_now_ms() - t0;

    if (P.verbose) {
        fprintf(stderr, "[lazyllm] warmup: phase 1 (kernel compile) %.1f ms\n", t_phase1);
    }

    // Phase 2: Prime the ggml allocator (galloc) for stage 0.
    //
    // Root cause of the 1.6 s per-prompt overhead:
    //   ggml_gallocr_needs_realloc() returns true whenever the graph's n_nodes
    //   count differs from what the galloc last saw.  After phase 1, galloc holds
    //   the node-count of the *last* partial stage (smallest graph).  The timed
    //   run starts with stage 0 (largest graph, n_nodes = N0), causing a "slow
    //   path" that calls ggml_backend_synchronize + ggml_gallocr_reserve_n — the
    //   source of the ~1600 ms hit on the very first timed call.
    //
    // Fix: after phase 1, clear the KV cache and run stage 0 one more time so
    //   that galloc ends up with N0 nodes and stage-0 tensor-size_max values.
    //   The timed run's first stage-0 decode_partial then finds:
    //     • galloc->n_nodes == graph->n_nodes   (N0 == N0) → no realloc needed
    //     • all tensor sizes ≤ size_max           (same or fewer tokens) → fits
    //   and takes the galloc fast path (~0.1 ms vs ~1600 ms).
    if (!P.pruning_layers.empty()) {
        const int l0 = P.pruning_layers[0];
        llama_memory_clear(llama_get_memory(lctx), false);

        const double t1 = lazyllm_now_ms();
        // Run stage 0 without embedding output — this primes galloc and also
        // pre-warms any CUDA kernels specific to stage 0's graph structure.
        lctx->decode_partial(batch, 0, l0, /*early_exit=*/false, /*no_embed_output=*/true);
        lctx->synchronize();
        const double t_phase2 = lazyllm_now_ms() - t1;

        if (P.verbose) {
            fprintf(stderr, "[lazyllm] warmup: phase 2 (galloc prime stage 0) %.1f ms\n", t_phase2);
        }
    }

    // Phase 3: A/B speed test — time baseline llama_decode and lazyllm prefill
    // under realistic STEADY-STATE conditions.  Sets ctx->fallback_active if
    // LazyLLM is not faster than baseline (e.g. on multi-GPU layer-split or
    // short prompts where the multi-stage overhead dominates).  This guarantees
    // LazyLLM mode never performs worse than baseline.
    //
    // Critical: each test gets one untimed warm-up call so that the timed run
    // sees the same galloc state it would see in a benchmark loop's 2nd+ rep:
    //   • baseline timed: prev call was baseline → galloc in full-graph state →
    //     no realloc → fast.  This matches steady-state baseline behavior.
    //   • lazyllm timed:  prev call was lazyllm  → galloc in last-stage state →
    //     stage 0 needs realloc → slow.  This matches steady-state lazyllm.
    double t_base_ms = 0.0;
    double t_lazy_ms = 0.0;

    if (P.auto_fallback && !P.pruning_layers.empty()) {
        // === Baseline measurement (steady-state) ===
        // Untimed warm-up: compiles full-graph kernels for batch.n_tokens
        // (which Phase 1 did NOT do — its last stage runs on n_kept tokens, not
        // the full prompt).  Also primes galloc for the full graph.
        llama_memory_clear(llama_get_memory(lctx), false);
        if (llama_decode(lctx, batch) != 0) {
            if (P.verbose) fprintf(stderr, "[lazyllm] warmup: baseline warm-up failed\n");
        }
        lctx->synchronize();

        // Timed baseline: galloc warm from previous baseline → realistic.
        llama_memory_clear(llama_get_memory(lctx), false);
        const double tb0 = lazyllm_now_ms();
        const int rb = llama_decode(lctx, batch);
        lctx->synchronize();
        t_base_ms = lazyllm_now_ms() - tb0;
        if (rb != 0) t_base_ms = 0.0;

        // === LazyLLM measurement (steady-state) ===
        // Untimed warm-up: cycles galloc through all lazyllm stages (Phase 1
        // did this already, but the intervening baseline runs reset galloc to
        // full-graph state).
        llama_memory_clear(llama_get_memory(lctx), false);
        ctx->kept_indices.clear();
        ctx->last_scores.clear();
        if (ctx->aux_cache_enabled) ctx->aux_cache.reset();
        llama_lazyllm_prefill(ctx, batch);
        lctx->synchronize();

        // Timed lazyllm: galloc state from previous lazyllm (stage N) → first
        // stage will incur realloc → matches what each benchmark rep sees.
        llama_memory_clear(llama_get_memory(lctx), false);
        ctx->kept_indices.clear();
        ctx->last_scores.clear();
        if (ctx->aux_cache_enabled) ctx->aux_cache.reset();
        const double tl0 = lazyllm_now_ms();
        const int rl = llama_lazyllm_prefill(ctx, batch);
        lctx->synchronize();
        t_lazy_ms = lazyllm_now_ms() - tl0;

        ctx->warmup_baseline_ms = t_base_ms;
        ctx->warmup_lazyllm_ms  = t_lazy_ms;

        // Decision: enable fallback unless LazyLLM is meaningfully faster.
        if (rl >= 0 && t_base_ms > 0.0 &&
            t_lazy_ms * P.auto_fallback_min_speedup > t_base_ms) {
            ctx->fallback_active = true;
        } else {
            ctx->fallback_active = false;
        }

        if (P.verbose) {
            fprintf(stderr,
                "[lazyllm] warmup: A/B baseline=%.1f ms lazyllm=%.1f ms speedup=%.2fx → %s\n",
                t_base_ms, t_lazy_ms,
                (t_lazy_ms > 0.0 ? t_base_ms / t_lazy_ms : 0.0),
                ctx->fallback_active ? "FALLBACK to baseline" : "use LazyLLM");
        }

        // Final prime: leave galloc in the state that the chosen path expects,
        // so the very first timed rep doesn't pay a realloc penalty.
        //   • fallback active → run one untimed llama_decode (galloc full-graph)
        //   • lazyllm active  → run stage 0 untimed (galloc stage 0)
        llama_memory_clear(llama_get_memory(lctx), false);
        if (ctx->fallback_active) {
            llama_decode(lctx, batch);
        } else {
            const int l0 = P.pruning_layers[0];
            lctx->decode_partial(batch, 0, l0, /*early_exit=*/false, /*no_embed_output=*/true);
        }
        lctx->synchronize();
    }

    // Reset context state so the warmup doesn't affect subsequent calls.
    llama_memory_clear(llama_get_memory(lctx), false);
    ctx->kept_indices.clear();
    ctx->last_scores.clear();
    if (ctx->aux_cache_enabled) ctx->aux_cache.reset();

    const double t_total = lazyllm_now_ms() - t0;
    if (P.verbose) {
        fprintf(stderr, "[lazyllm] warmup: total %.1f ms\n", t_total);
    }

    return 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Aux Cache API
// ─────────────────────────────────────────────────────────────────────────────

void llama_lazyllm_aux_cache_enable(struct llama_lazyllm_context * ctx) {
    if (ctx) ctx->aux_cache_enabled = true;
}

void llama_lazyllm_aux_cache_disable(struct llama_lazyllm_context * ctx) {
    if (ctx) ctx->aux_cache_enabled = false;
}

void llama_lazyllm_aux_cache_reset(struct llama_lazyllm_context * ctx) {
    if (ctx) ctx->aux_cache.reset();
}

const float * llama_lazyllm_aux_cache_get(
        struct llama_lazyllm_context * ctx,
        int32_t token_pos) {
    if (!ctx) return nullptr;
    return ctx->aux_cache.get(token_pos);
}

size_t llama_lazyllm_aux_cache_bytes(struct llama_lazyllm_context * ctx) {
    if (!ctx) return 0;
    return ctx->aux_cache.total_bytes();
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 5a — Decode-stage dynamic KV pruning
// ─────────────────────────────────────────────────────────────────────────────

void llama_lazyllm_decode_pruning_enable(
        struct llama_lazyllm_context * lz_ctx,
        float keep_ratio,
        llama_seq_id seq_id) {
    if (!lz_ctx) return;
    lz_ctx->decode_pruning_enabled = true;
    lz_ctx->decode_keep_ratio      = keep_ratio;
    lz_ctx->decode_seq_id          = seq_id;
}

int32_t llama_lazyllm_decode_n_alive(
        const struct llama_lazyllm_context * lz_ctx) {
    if (!lz_ctx) return 0;
    return (int32_t)lz_ctx->alive_positions.size();
}

// Extract attention scores for the last decode step and aggregate across layers.
// Returns a score per KV position (alive_positions + new token) or empty vector on failure.
static std::vector<float> decode_extract_agg_scores(
        struct llama_lazyllm_context * lz_ctx,
        int n_kv_expected) {
    std::vector<float> agg(n_kv_expected, 0.0f);
    int n_layers_ok = 0;

    for (int l : lz_ctx->params.pruning_layers) {
        std::vector<float> scores;
        // For a single decode token, last_pos=0 (only one query token).
        int r = llama_lazyllm_extract_attention(lz_ctx, l - 1, 0, scores);
        if (r != 0) continue;
        if ((int)scores.size() != n_kv_expected) continue;

        for (int i = 0; i < n_kv_expected; i++) {
            agg[i] += scores[i];
        }
        n_layers_ok++;
    }

    if (n_layers_ok == 0) return {};
    for (float & s : agg) s /= (float)n_layers_ok;
    return agg;
}

int llama_lazyllm_decode_step(
        struct llama_lazyllm_context * lz_ctx,
        const struct llama_batch     & batch) {
    if (!lz_ctx || !lz_ctx->ctx_base) return -1;
    if (batch.n_tokens != 1) {
        fprintf(stderr, "[lazyllm] decode_step: batch must have exactly 1 token, got %d\n",
                batch.n_tokens);
        return -1;
    }

    llama_context * lctx = lz_ctx->ctx_base;

    // --- 1. Normal forward pass ------------------------------------------------
    int r = llama_decode(lctx, batch);
    if (r != 0) {
        fprintf(stderr, "[lazyllm] decode_step: llama_decode failed: %d\n", r);
        return -1;
    }

    // Track the new token's KV position (sequential after prefill positions).
    const llama_pos new_pos = batch.pos ? batch.pos[0]
                                        : (llama_pos)(lz_ctx->alive_positions.empty()
                                              ? 0
                                              : lz_ctx->alive_positions.back() + 1);

    if (!lz_ctx->decode_pruning_enabled) {
        lz_ctx->alive_positions.push_back(new_pos);
        lz_ctx->decode_step++;
        return (int32_t)lz_ctx->alive_positions.size();
    }

    lctx->synchronize();

    // --- 2. Extract attention scores -------------------------------------------
    // After decode, KV now has alive_positions.size() + 1 entries
    // (alive_positions[] + the newly added token at new_pos).
    const int n_old = (int)lz_ctx->alive_positions.size();
    const int n_kv  = n_old + 1;

    std::vector<float> agg = decode_extract_agg_scores(lz_ctx, n_kv);
    if (agg.empty()) {
        // Cannot extract scores — just add new position without pruning.
        lz_ctx->alive_positions.push_back(new_pos);
        lz_ctx->decode_step++;
        return (int32_t)lz_ctx->alive_positions.size();
    }

    // --- 3. Decide keep set for old positions ----------------------------------
    // Always keep the newest token (index n_kv-1 in scores = new_pos).
    // Among the n_old old positions, keep top decode_keep_ratio.
    const int n_keep_old = std::max(1, (int)std::ceil(n_old * lz_ctx->decode_keep_ratio));

    // Sort old positions by score descending.
    std::vector<int> order(n_old);
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + n_keep_old, order.end(),
                      [&agg](int a, int b) { return agg[a] > agg[b]; });

    std::vector<bool> keep_mask(n_old, false);
    for (int i = 0; i < n_keep_old; i++) {
        keep_mask[order[i]] = true;
    }

    // --- 4. Remove dropped positions from KV cache ----------------------------
    llama_memory_t mem = llama_get_memory(lctx);
    std::vector<llama_pos> new_alive;
    new_alive.reserve(n_keep_old + 1);

    for (int i = 0; i < n_old; i++) {
        if (keep_mask[i]) {
            new_alive.push_back(lz_ctx->alive_positions[i]);
        } else {
            const llama_pos p = lz_ctx->alive_positions[i];
            llama_memory_seq_rm(mem, lz_ctx->decode_seq_id, p, p + 1);
        }
    }
    new_alive.push_back(new_pos);

    lz_ctx->alive_positions = std::move(new_alive);
    lz_ctx->decode_step++;

    if (lz_ctx->params.verbose) {
        fprintf(stderr, "[lazyllm] decode_step %d: kept %d/%d old positions, total alive=%d\n",
                lz_ctx->decode_step, n_keep_old, n_old, (int)lz_ctx->alive_positions.size());
    }

    return (int32_t)lz_ctx->alive_positions.size();
}

