#pragma once
// ─── GPU-side attention score pooling for LazyLLM ────────────────────────────
//
// lazyllm_add_score_pool() appends GPU-side reduction ops for the LAST layer
// in [il_start, il_end), producing a "lazyllm_scores-{il_end-1}" tensor of
// shape [n_kv] — the average attention weight received by each key position
// from the LAST QUERY TOKEN, averaged over all heads.
//
// Works with BOTH non-FA and FA modes:
//
//  Non-FA (flash_attn disabled):
//    Searches for "kq_soft_max-{il}" — the full softmax matrix present in the
//    graph.  Slices the last-query row and reduces over heads.
//    Transfer: n_kv * sizeof(float) ≈ 16 KB (vs ~2 GB without pooling).
//
//  FA (flash_attn_ext enabled):
//    "kq_soft_max-{il}" does not exist.  Falls back to computing a lightweight
//    QK scoring branch from the named "Qcur-{il}" / "Kcur-{il}" (or their
//    QK-norm variants "Qcur_normed-{il}" / "Kcur_normed-{il}") tensors that
//    every partial graph builder places in the graph before calling build_attn.
//
//    Shapes at that point (after RoPE, before build_attn):
//      Qcur: [d_head, n_head,    n_tokens]
//      Kcur: [d_head, n_head_kv, n_tokens]
//    Since the KV cache is cleared before every stage, Kcur IS the full cache
//    and contains exactly n_tokens entries — the same set used for scoring.
//
//    QK scoring branch (tiny extra cost, O(n_kv * n_head_kv)):
//      1. Extract last-query vector from Qcur [d_head, n_head, 1]
//         (for GQA: average n_head/n_head_kv queries per kv-head → [d_head, n_head_kv, 1])
//      2. scores = Kcur^T @ q_per_kv  → [n_tokens, n_head_kv]
//      3. Scale by kq_scale, softmax per head
//      4. Mean over kv-heads → [n_tokens]
//
// Only the last layer is pooled to avoid keeping all kq_soft_max tensors live
// simultaneously (46 GB for 28 layers at 4k context → OOM).

#include <cmath>
#include <cstdio>
#include <cstring>
#include "ggml.h"

// ─── helper: pool from an existing kq_soft_max tensor ────────────────────────
// kq shape: [n_kv, n_tokens, n_head, n_stream]
// Produces: lazyllm_scores-{il} [n_kv]
static void lazyllm_pool_from_kq_soft_max(ggml_context * ctx0, ggml_cgraph * gf,
                                          ggml_tensor * kq, int il) {
    const int64_t n_kv     = kq->ne[0];
    const int64_t n_tokens = kq->ne[1];
    const int64_t n_head   = kq->ne[2];
    const int64_t n_stream = kq->ne[3];
    if (n_kv <= 0 || n_tokens <= 0 || n_head <= 0) return;

    const size_t last_q_offset = (size_t)(n_tokens - 1) * (size_t)kq->nb[1];
    ggml_tensor * last_q = ggml_view_4d(ctx0, kq,
            n_kv, 1, n_head, n_stream,
            kq->nb[1], kq->nb[2], kq->nb[3],
            last_q_offset);
    ggml_tensor * cont = ggml_cont(ctx0, last_q);
    ggml_tensor * flat = ggml_reshape_2d(ctx0, cont, n_kv, n_head * n_stream);
    ggml_tensor * T    = ggml_cont(ctx0, ggml_permute(ctx0, flat, 1, 0, 2, 3));
    ggml_tensor * avg  = ggml_mean(ctx0, T);
    ggml_tensor * out  = ggml_reshape_1d(ctx0, avg, n_kv);

    char sname[64];
    snprintf(sname, sizeof(sname), "lazyllm_scores-%d", il);
    ggml_set_name(out, sname);
    ggml_build_forward_expand(gf, out);
}

// ─── helper: pool from Qcur / Kcur tensors (FA-compat path) ──────────────────
// Qcur shape: [d_head, n_head,    n_tokens]  (after RoPE, before build_attn)
// Kcur shape: [d_head, n_head_kv, n_tokens]  (same; Kcur == full K cache here
//             because llama_memory_clear is called before every LazyLLM stage)
// Produces: lazyllm_scores-{il} [n_tokens]
//
// Strategy: flatten the head dimension into K so we can use a simple 2D matmul.
//   For GQA (n_head > n_head_kv), average query-head groups first so both Q and K
//   have the same n_head_kv dimension, then flatten:
//     q_flat:  [d_head * n_head_kv, 1]
//     K_flat:  [d_head * n_head_kv, n_tokens]
//     scores = K_flat^T @ q_flat  →  [n_tokens, 1]
//   The dot product sums per-head contributions; relative ordering matches the
//   per-head-average attention weight, which is all that matters for pruning.
static void lazyllm_pool_from_qcur_kcur(ggml_context * ctx0, ggml_cgraph * gf,
                                        ggml_tensor * Qcur, ggml_tensor * Kcur,
                                        int il, float kq_scale) {
    // Qcur: [d_head, n_head,    n_tokens]
    const int64_t d_head    = Qcur->ne[0];
    const int64_t n_head    = Qcur->ne[1];
    const int64_t n_tokens  = Qcur->ne[2];
    // Kcur: [d_head, n_head_kv, n_tokens]
    const int64_t n_head_kv = Kcur->ne[1];
    if (d_head <= 0 || n_head <= 0 || n_tokens <= 0 || n_head_kv <= 0) return;

    if (kq_scale == 0.0f) kq_scale = 1.0f / sqrtf((float)d_head);

    const int64_t n_head_ratio = (n_head > n_head_kv) ? (n_head / n_head_kv) : 1;
    const int64_t K_dim = d_head * n_head_kv;  // flattened K dimension


    // ── Step 1: extract last-query vector from Qcur ───────────────────────────
    // Qcur: [d_head, n_head, n_tokens]; last token at dim-2 index n_tokens-1.
    const size_t last_q_off = (size_t)(n_tokens - 1) * (size_t)Qcur->nb[2];
    ggml_tensor * q_last = ggml_view_3d(ctx0, Qcur,
            d_head, n_head, 1,
            Qcur->nb[1], Qcur->nb[2],
            last_q_off);
    q_last = ggml_cont(ctx0, q_last); // [d_head, n_head, 1] contiguous

    // ── Step 2: GQA grouping — average query-head groups to n_head_kv ─────────
    // Result: q_avg [d_head, n_head_kv] then flattened to q_flat [K_dim, 1].
    ggml_tensor * q_avg;
    if (n_head_ratio <= 1) {
        // MHA: n_head == n_head_kv — cont ensures reshape is valid
        q_avg = ggml_reshape_2d(ctx0, ggml_cont(ctx0, q_last), d_head, n_head_kv);
    } else {
        // GQA: [d_head, n_head, 1] → group n_head_ratio query heads → [d_head, n_head_kv]
        // Reshape to [d_head, n_head_ratio, n_head_kv, 1]
        ggml_tensor * q_r = ggml_reshape_4d(ctx0, q_last, d_head, n_head_ratio, n_head_kv, 1);
        // Permute → [n_head_ratio, d_head, n_head_kv, 1]; mean over dim-0 → [1, d_head, n_head_kv, 1]
        ggml_tensor * q_p = ggml_cont(ctx0, ggml_permute(ctx0, q_r, 1, 0, 2, 3));
        ggml_tensor * q_m = ggml_mean(ctx0, q_p);     // [1, d_head, n_head_kv, 1]
        // q_m has strides [4, type_sz, d_head*type_sz, ...]; use ggml_cont before reshape
        q_avg = ggml_reshape_2d(ctx0, ggml_cont(ctx0, q_m), d_head, n_head_kv);
    }
    // q_avg: [d_head, n_head_kv]
    // Flatten → q_flat [K_dim, 1]  (2D for simple matmul)
    ggml_tensor * q_flat = ggml_reshape_2d(ctx0, ggml_cont(ctx0, q_avg), K_dim, 1);

    // ── Step 3: flatten Kcur → K_flat [K_dim, n_tokens] ──────────────────────
    // Kcur: [d_head, n_head_kv, n_tokens]; contiguous in memory.
    // Flattening d_head × n_head_kv gives the same head-major ordering as q_flat.
    // Use Kcur->ne[2] (not n_tokens from Qcur) for safety if shapes diverge.
    const int64_t n_kv_tokens = Kcur->ne[2];
    ggml_tensor * K_flat = ggml_reshape_2d(ctx0, ggml_cont(ctx0, Kcur), K_dim, n_kv_tokens);

    // ── Step 4: scores = K_flat^T @ q_flat → [n_kv_tokens, 1] ──────────────
    // ggml_mul_mat(a=[K,M], b=[K,N]) → [M,N]
    // a=K_flat: [K_dim, n_kv_tokens], b=q_flat: [K_dim, 1] → [n_kv_tokens, 1]
    ggml_tensor * scores_raw = ggml_mul_mat(ctx0, K_flat, q_flat);

    // ── Step 5: scale + softmax → [n_kv_tokens, 1] ───────────────────────────
    scores_raw = ggml_scale(ctx0, scores_raw, kq_scale);
    ggml_tensor * scores_sm = ggml_soft_max(ctx0, scores_raw);

    // ── Step 6: squeeze to [n_kv_tokens] ─────────────────────────────────────
    ggml_tensor * out = ggml_reshape_1d(ctx0, scores_sm, n_kv_tokens);

    char sname[64];
    snprintf(sname, sizeof(sname), "lazyllm_scores-%d", il);
    ggml_set_name(out, sname);
    ggml_build_forward_expand(gf, out);
}

// ─── tensor lookup helpers ────────────────────────────────────────────────────
static ggml_tensor * lazyllm_find_tensor(ggml_context * ctx0, const char * name) {
    for (ggml_tensor * t = ggml_get_first_tensor(ctx0); t;
         t = ggml_get_next_tensor(ctx0, t)) {
        if (strcmp(t->name, name) == 0) return t;
    }
    return nullptr;
}

// Find the LAST tensor in the context with the given name.
// Partial builders call cb(Qcur, "Qcur", il) TWICE — once on the 2D projection
// output (before reshape) and once on the 3D post-RoPE tensor.  We want the
// latest allocation, which is the post-RoPE version with shape [d_head, n_head, n_tokens].
static ggml_tensor * lazyllm_find_tensor_last(ggml_context * ctx0, const char * name) {
    ggml_tensor * found = nullptr;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx0); t;
         t = ggml_get_next_tensor(ctx0, t)) {
        if (strcmp(t->name, name) == 0) found = t;
    }
    return found;
}

// ─── main entry point ─────────────────────────────────────────────────────────
// Call ONCE per partial graph, AFTER the layer loop, BEFORE the final
// ggml_build_forward_expand(gf, cur).
//
// kq_scale: 1/sqrt(d_head).  Pass 0.0f to auto-detect from Qcur dimensions.
static void lazyllm_add_score_pool(ggml_context * ctx0, ggml_cgraph * gf,
                                   int il_start, int il_end,
                                   float kq_scale = 0.0f) {
    if (il_end <= il_start) return;
    const int il = il_end - 1;

    // ── Path 1: non-FA — use kq_soft_max directly ────────────────────────────
    {
        char name[64];
        snprintf(name, sizeof(name), "kq_soft_max-%d", il);
        ggml_tensor * kq = lazyllm_find_tensor(ctx0, name);
        if (kq) {
            lazyllm_pool_from_kq_soft_max(ctx0, gf, kq, il);
            return;
        }
    }

    // ── Path 2: FA — build QK scoring branch from Qcur / Kcur ───────────────
    // Try QK-normed variants first (models with use_kq_norm=true), then plain.
    {
        char qnorm[64], knorm[64], qraw[64], kraw[64];
        snprintf(qnorm, sizeof(qnorm), "Qcur_normed-%d", il);
        snprintf(knorm, sizeof(knorm), "Kcur_normed-%d", il);
        snprintf(qraw,  sizeof(qraw),  "Qcur-%d",        il);
        snprintf(kraw,  sizeof(kraw),  "Kcur-%d",        il);

        // Use LAST match — cb() is called twice (pre-reshape and post-RoPE);
        // the post-RoPE tensor is the later allocation with shape [d_head,n_head,n_tokens].
        ggml_tensor * Qcur = lazyllm_find_tensor_last(ctx0, qnorm);
        if (!Qcur) Qcur = lazyllm_find_tensor_last(ctx0, qraw);

        ggml_tensor * Kcur = lazyllm_find_tensor_last(ctx0, knorm);
        if (!Kcur) Kcur = lazyllm_find_tensor_last(ctx0, kraw);

        if (Qcur && Kcur) {
            lazyllm_pool_from_qcur_kcur(ctx0, gf, Qcur, Kcur, il, kq_scale);
        }
        // If neither tensor is found, scoring simply doesn't happen this stage.
    }
}
