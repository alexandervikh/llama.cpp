#pragma once
// ─── GPU-side attention score pooling for LazyLLM ────────────────────────────
//
// lazyllm_add_score_pool() appends GPU-side reduction ops for the LAST layer
// in [il_start, il_end), producing a "lazyllm_scores-{il_end-1}" tensor of
// shape [n_kv] — the average attention weight received by each key position
// from the LAST QUERY TOKEN, averaged over all heads.
//
// Only the last layer is pooled because llama_lazyllm_extract_attention()
// always queries "lazyllm_scores-{il_end-1}" (the final executed attention
// layer whose scores drive token pruning).  Adding pooling ops for every layer
// would keep all kq_soft_max tensors alive simultaneously: for 28 layers at
// 4k context that is ~46 GB, causing OOM on a single GPU.
//
// Strategy (memory-efficient):
//   1. Select last-query slice via ggml_view_4d  → [n_kv, 1, n_head, 1]
//      (zero-copy strided view, no allocation)
//   2. ggml_cont                                 → [n_kv, 1, n_head, 1] contiguous
//      (n_kv * n_head * sizeof(float) ≈ 512 KB for n_kv=4096, n_head=32)
//   3. Reshape+permute+cont+mean                 → [n_kv]
//      (1 MB total intermediate memory, vs ~2 GB for the naive approach)
//
// Result: only n_kv * sizeof(float) ≈ 16 KB is transferred to the CPU by
// llama_lazyllm_extract_attention(), replacing a potentially 2+ GB transfer.

#include <cstdio>
#include <cstring>
#include "ggml.h"

static void lazyllm_add_score_pool(ggml_context * ctx0, ggml_cgraph * gf,
                                   int il_start, int il_end) {
    // Only pool the LAST layer — it's the one extract_attention queries.
    // This avoids keeping all kq_soft_max tensors live simultaneously.
    if (il_end <= il_start) return;
    const int il = il_end - 1;
    {
        char name[64];
        snprintf(name, sizeof(name), "kq_soft_max-%d", il);

        // Find the kq_soft_max tensor for this layer.
        ggml_tensor * kq = nullptr;
        for (ggml_tensor * t = ggml_get_first_tensor(ctx0); t;
             t = ggml_get_next_tensor(ctx0, t)) {
            if (strcmp(t->name, name) == 0) { kq = t; break; }
        }
        if (!kq) return;

        // kq_soft_max shape: [n_kv, n_tokens, n_head, n_stream]
        const int64_t n_kv     = kq->ne[0];
        const int64_t n_tokens = kq->ne[1];
        const int64_t n_head   = kq->ne[2];
        const int64_t n_stream = kq->ne[3];
        if (n_kv <= 0 || n_tokens <= 0 || n_head <= 0) return;

        // Step 1: strided view selecting the LAST query token's attention
        // across all heads and streams → [n_kv, 1, n_head, n_stream].
        // Memory: 0 extra bytes (pure view).
        const size_t last_q_offset = (size_t)(n_tokens - 1) * (size_t)kq->nb[1];
        ggml_tensor * last_q = ggml_view_4d(ctx0, kq,
                n_kv, 1, n_head, n_stream,
                kq->nb[1],   // nb[1]: stride for ne[1]=1 (value doesn't matter)
                kq->nb[2],   // nb[2]: stride per head = n_kv * n_tokens * sizeof(f32)
                kq->nb[3],   // nb[3]: stride per stream
                last_q_offset);

        // Step 2: make contiguous so reshape is valid.
        // Memory: n_kv * n_head * n_stream * sizeof(float) ≈ 512 KB.
        ggml_tensor * cont = ggml_cont(ctx0, last_q);

        // Step 3: reshape to [n_kv, n_head * n_stream] (view, no copy).
        ggml_tensor * flat = ggml_reshape_2d(ctx0, cont, n_kv, n_head * n_stream);

        // Step 4: transpose to [n_head * n_stream, n_kv] and make contiguous.
        // Memory: another ≈ 512 KB.
        ggml_tensor * T = ggml_cont(ctx0, ggml_permute(ctx0, flat, 1, 0, 2, 3));

        // Step 5: mean over ne[0] (n_head * n_stream) → [1, n_kv, 1, 1].
        ggml_tensor * avg = ggml_mean(ctx0, T);

        // Step 6: reshape to [n_kv] for easy retrieval.
        ggml_tensor * scores = ggml_reshape_1d(ctx0, avg, n_kv);

        char sname[64];
        snprintf(sname, sizeof(sname), "lazyllm_scores-%d", il);
        ggml_set_name(scores, sname);
        ggml_build_forward_expand(gf, scores);
    }
}
