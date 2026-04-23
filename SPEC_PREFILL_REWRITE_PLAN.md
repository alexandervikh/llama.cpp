# Speculative Prefill — Rewrite Plan

**Date:** 2026-04-23
**Branch:** `spec-prefill`
**Reference:** Liu et al., "Speculative Prefill: Turbocharging TTFT with Lightweight and Training-Free Token Importance Estimation", ICML 2025, arXiv:2502.02789
**Upstream repo:** https://github.com/Jingyu6/speculative_prefill

---

## 1. Diagnosis — Implementation Diverges From Paper

### 1.1 Paper Algorithm

1. Draft model runs forward on the full prompt once.
2. Draft performs **N small lookahead steps** (N ~ 8–16), storing per-step Q vectors for every layer/head.
3. For each lookahead step `j` and every prompt position `i`, compute attention:
   `a_{i,j,l,h} = softmax(Q_{M+j,l,h} · K_{i,l,h}^T / sqrt(d_head))`
   where `M` is the prompt length.
4. Aggregate to a scalar per prompt position:
   `score_i = mean_{j in N}( max_{l, h}( a_{i,j,l,h} ) )`
5. Chunk the prompt into contiguous blocks (≈ 16 tokens), average scores per block, select **top-K blocks** by ratio `kr`.
6. Feed kept tokens to the base model in a **single forward pass**, preserving original positions (RoPE correctness).

### 1.2 Current Implementation Problems

File: `src/llama-spec-prefill.cpp`.

| Issue | Location | Impact |
|-------|----------|--------|
| Uses logit-perplexity as importance signal, not attention weights | `llama_spec_prefill_compute_importance` (line 433) | Wrong signal; runs full draft forward just to compute perplexity. |
| Fallback path is a fabricated distance-weighted entropy heuristic | same function, lines ~530–560 | Not the paper's method; produces meaningless scores. |
| Lookahead phase generates `n_ctx` tokens sequentially | commit `ae83b1def`, later tuned in `8644e5e99` | Dominates latency at long contexts. Paper uses N=8–16. |
| No chunk/block-level top-K selector; token-level threshold only | filter step in `llama_spec_prefill` | Misaligns with paper's block selection, hurts quality at low `kr`. |
| Q-tensor extraction started but unused by scorer | commit `8644e5e99` | Infrastructure half-built. |
| Re-indexed positions after filter may break RoPE | flagged in `SPEC_PREFILL_LLAMA_ANALYSIS.md` §5.3 | Quality risk on kept tokens. |

### 1.3 Benchmark Consequence

- pp512–pp4096: **0.22×–0.35× (3.3× slower than baseline)** across all keep ratios.
- pp8192: 1.37×–2.20× only because entropy fallback path is cheaper — accidental, not by design.
- Root cause: perplexity scoring costs as much as a full base-model prefill, negating filter savings.

---

## 2. Rewrite Plan

### 2.1 Remove

- Delete perplexity proxy branch in `llama_spec_prefill_compute_importance` (lines ~452–515).
- Delete distance/entropy fallback branch (lines ~517–560).
- Keep old impl only if gated behind `-DSPEC_PREFILL_LEGACY` for A/B testing (optional).

### 2.2 Add — Attention Score Extraction

New function: `extract_attention_scores(ctx_spec, n_prompt, n_lookahead) -> std::vector<float>`.

Steps:
1. Run draft forward on full prompt (already done in generate_lookahead).
2. Run N small lookahead decode steps, each producing one token.
3. For each layer `l` and head `h`, extract Q tensor from each lookahead step and K cache from prompt positions.
4. Compute `Q · K^T / sqrt(d_head)`, apply softmax over prompt dimension.
5. Reduce: `max` over `(l, h)`, then `mean` over N lookahead steps.
6. Return `std::vector<float>` length `n_prompt`.

Implementation notes:
- Llama.cpp fused flash-attention skips materialized attention weights. **Must disable flash-attn on draft context** (`--no-flash-attn` or `cparams.flash_attn = false`) or install a ggml graph callback that captures pre-softmax scores.
- Alternative: compute attention manually post-forward using `llama_get_layer_q()` + `llama_kv_cache_view_*` accessors. Slower per step but avoids graph surgery.
- Reuse existing Q-tensor extraction scaffolding from commit `8644e5e99`.

### 2.3 Cap Lookahead N

- Replace `n_lookahead = n_ctx` with `n_lookahead = params.spec_n_lookahead` defaulted to 8.
- Expose `--spec-n-lookahead` flag in `llama-bench` and `spec-prefill-run`.
- Paper ablates N; 8–16 is sufficient.

### 2.4 Add — Block Top-K Selector

New function: `select_top_k_blocks(scores, chunk_size, kr) -> std::vector<int>`.

Steps:
1. Partition `[0, n_prompt)` into contiguous blocks of `chunk_size` (default 16).
2. `block_score[b] = mean(scores[b * chunk_size : (b+1) * chunk_size])`.
3. Sort block indices by `block_score` descending.
4. Keep top `ceil(kr * n_blocks)` blocks.
5. Expand kept blocks back to token indices.
6. Always include: first block (BOS / system tokens), last block (most recent context), any mandatory positions from `params.spec_pool`.

### 2.5 Preserve Original Positions

- When building the filtered batch for the base model, set `batch.pos[k] = original_prompt_index[k]`, **not** `k`.
- Verify RoPE consumes `batch.pos`, not sequence index, in the filtered decode path.
- Add unit test: two prompts where filtered output with original positions matches a reference prompt with those tokens at their original positions.

### 2.6 Async Overlap (Optional, Phase 2)

- Draft forward + score compute on draft CUDA stream.
- Base model can start the filtered decode as soon as the top-K kept-index list is ready.
- Use separate `llama_context` streams if available; otherwise serialize but minimize idle time.

### 2.7 Benchmark Re-run

Matrix:

| Parameter | Values |
|-----------|--------|
| `pp` | 512, 1024, 2048, 4096, 8192, 16384 |
| `kr` | 0.10, 0.25, 0.50 |
| `N` (lookahead) | 4, 8, 16 |
| `chunk_size` | 8, 16, 32 |

Expected inflection where spec-prefill beats baseline: **pp ≥ 2048** (vs. current pp ≥ 8192).

---

## 3. Expected Outcome

- **pp512–pp2048:** break-even to 1.5× speedup. Draft prompt-forward still dominant at small context. Paper does not win short contexts either — designed for long prompts.
- **pp4096+:** 2×–4× realistic on Llama-3.1-8B + Llama-3.2-1B pair.
- **Paper reports:** 7.66× TTFT on Llama-3.1-405B-FP8 + Llama-3.2-1B. Model-size ratio matters; 8B/1B pair has ~8× compute ratio vs. paper's ~400× ratio, so expect proportionally lower speedup.
- Short-context regime may never win. That is acceptable — out of scope for this technique.

---

## 4. Minimum Viable Rewrite

Gut `llama_spec_prefill_compute_importance` to ~80 lines:

```
int llama_spec_prefill_compute_importance(
    llama_spec_prefill_context * ctx,
    const llama_token * prompt_tokens,
    int n_prompt,
    std::vector<float> & token_importance
) {
    if (!ctx || !prompt_tokens || n_prompt <= 0) return -1;
    token_importance.resize(n_prompt, 0.0f);

    // 1. Draft prompt forward (already done by generate_lookahead caller).
    // 2. Extract attention scores from N lookahead steps.
    if (extract_attention_scores(ctx, n_prompt, ctx->params.spec_n_lookahead,
                                 token_importance) != 0) {
        return -1;
    }
    // Normalize, optional pooling.
    normalize_scores(token_importance);
    if (ctx->params.pool_kernel_size > 1)
        llama_spec_prefill_apply_pooling(token_importance, ctx->params.pool_kernel_size);
    return 0;
}
```

Caller switches from token-level threshold to block selector:

```
std::vector<int> kept = select_top_k_blocks(token_importance,
                                            ctx->params.chunk_size,
                                            ctx->params.keep_ratio);
```

Filtered batch build preserves original positions.

---

## 5. Risks

| Risk | Mitigation |
|------|------------|
| ggml graph does not expose softmaxed attention when flash-attn is fused | Disable flash-attn on draft context, or add graph callback. Document perf cost. |
| Q/K extraction slow per layer | Batch across layers; accept N=8 minimum. Profile before optimizing. |
| RoPE with non-contiguous positions may hit untested paths | Parity test: filtered decode vs. reference decode on same positions. |
| Paper uses 70B/405B base; 8B base may not preserve quality at kr=0.1 | Run LongBench or in-house quality eval before claiming parity. |
| Draft model may disagree with base on salient tokens | Accept; paper shows this works empirically for Llama family. Validate with ablation. |

---

## 6. Phasing

| Phase | Scope | Deliverable |
|-------|-------|-------------|
| 1 | Delete wrong scoring, stub attention extractor returning uniform | Builds and runs; baseline parity |
| 2 | Real attention extraction via graph callback or manual Q·K^T | Non-trivial scores, unit-tested |
| 3 | Block top-K selector, original-position preservation | Paper-faithful pipeline |
| 4 | Benchmark re-run + quality eval (LongBench subset) | Numbers for report |
| 5 | Async overlap + flash-attn compatibility | Optional perf polish |

---

## 7. Files to Touch

| File | Change |
|------|--------|
| `src/llama-spec-prefill.cpp` | Rewrite `compute_importance`, add `extract_attention_scores`, add `select_top_k_blocks` |
| `include/llama-spec-prefill.h` | New public params `spec_n_lookahead`, `chunk_size` |
| `tools/llama-bench/llama-bench.cpp` | Expose new flags |
| `examples/spec-prefill-run/*` | Same flag additions |
| `tests/test-spec-prefill-parity.cpp` | Add original-position RoPE parity test |
| `tests/test-spec-prefill-unit.cpp` | Unit tests for block selector and score aggregation |
| `SPEC_PREFILL_README.md` | Update algorithm description to match paper |

---

*Plan derived from SPEC_PREFILL_LLAMA_ANALYSIS.md (benchmark report) + arXiv:2502.02789 (paper) + inspection of `src/llama-spec-prefill.cpp` on branch `spec-prefill`.*
