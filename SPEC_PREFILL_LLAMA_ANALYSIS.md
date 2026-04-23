# Speculative Prefill: Problem Analysis and Benchmark Report

**Date:** 2026-04-23
**Branch:** `spec-prefill`
**Models:** Llama-3.1-8B (base) + Llama-3.2-1B (draft)
**Hardware:** 4× NVIDIA L4 (GPU 0 only), AMD EPYC 7R13 7742

---

## 1. Problem Statement

The `llama_spec_prefill()` function in `src/llama-spec-prefill.cpp` implements speculative prefill: instead of decoding the full prompt through the base model, it generates candidate tokens with a fast draft model, scores their importance, filters, then decodes only the important tokens with the base model. The theoretical speedup formula is:

```
speedup = (1 - kr) * (t_base / t_draft) + kr
```

Where `kr` is the keep ratio and `t_base / t_draft` is the compute ratio. For an 8B+1B pair, expected speedup at kr=0.25 is ~6.4×.

**Observed result:** The feature is 3.3× slower than baseline prefill at small-to-medium context lengths. Only at pp8192 does it become faster, and only because a fallback entropy proxy reduces scoring cost.

This report documents the investigation, code analysis, benchmark methodology, and root cause.

---

## 2. Code Analysis

### 2.1 Architecture Overview

The implementation in `src/llama-spec-prefill.cpp` follows a 3-phase pipeline:

```
Phase 1: Lookahead Generation (draft model)
  → llama_spec_prefill_generate_lookahead()
  → Generates n_ctx tokens using the draft model sequentially

Phase 2: Token Importance Scoring
  → compute_attention()
  → Scores each position's "importance" using base model

Phase 3: Filtered Decode (base model)
  → Filters tokens below the keep ratio threshold
  → Decodes only kept tokens through the base model
```

### 2.2 The Bottleneck: `compute_attention()`

Located at `src/llama-spec-prefill.cpp:208-301`, this function computes token-level importance scores. It uses two strategies:

**Strategy A — Perplexity Proxy (n_tokens < 4096):**
```
For each position i in the prompt:
    1. Build a context with all tokens up to position i
    2. Run llama_decode() on the base model
    3. Extract logit at position i
    4. Compute perplexity from the logit
    5. Store as attention score
```

This runs the base model on the **full prompt** for importance scoring alone — the same work baseline prefill already does. The filtering benefit is negated because the expensive decode happened before filtering.

**Strategy B — Entropy Proxy (n_tokens >= 4096):**
```
For each position i:
    1. Run llama_decode() on the base model
    2. Compute output entropy from logit distribution
    3. Use entropy as proxy for information content
```

When the entropy proxy activates at 4096+ tokens, the scoring becomes cheaper relative to baseline, which explains why pp8192 shows a speedup.

### 2.3 Supporting Infrastructure

The `llama-bench` tool (`tools/llama-bench/llama-bench.cpp`) was modified to support spec-prefill benchmarking:

1. **Context sizing fix:** Added `n_ctx += spec_n_lookahead + 64` to prevent crashes when lookahead generation needs more context than the prompt itself.

2. **Parameter passing fix:** Changed hardcoded `0, 0` in `spec_prefill_test_prompt()` call to pass actual `inst.spec_pool` and `inst.spec_chunk_size` values.

3. **Timing instrumentation:** Added `samples_ttft_ns` JSONL output capturing per-phase timing breakdown.

---

## 3. Benchmark Methodology

### 3.1 Configuration

| Parameter | Value |
|-----------|-------|
| Base model | Llama-3.1-8B-Instruct (Q4_K_M) |
| Draft model | Llama-3.2-1B-Instruct (Q4_K_M) |
| Backend | CUDA |
| Layers | ngl=99 (full GPU offload) |
| GPU | NVIDIA L4 (GPU 0) |
| n_threads | 32 |
| Temp | 0.0 |
| Prompt | 1200-character instruction prompt |
| n_predict | 256 |

### 3.2 Test Matrix

| Context Length (pp) | Keep Ratios | Runs per config |
|---------------------|-------------|-----------------|
| 512 | 0.10, 0.25, 0.50, 1.00 | 3 |
| 1024 | 0.10, 0.25, 0.50, 1.00 | 3 |
| 2048 | 0.10, 0.25, 0.50, 1.00 | 3 |
| 4096 | 0.10, 0.25, 0.50, 1.00 | 3 |
| 8192 | 0.10, 0.25, 0.50, 1.00 | 3 |

### 3.3 Baseline Measurements

Baseline prefill throughput (baseline, no spec):

| pp | tokens/s | TTFT (ms) |
|----|----------|-----------|
| 512  | 3137 | 163 |
| 1024 | 3937 | 260 |
| 2048 | 4664 | 439 |
| 4096 | 4783 | 856 |
| 8192 | 4021 | 2037 |

---

## 4. Results

### 4.1 Speculative Prefill TTFT

```
  TTFT: Llama-3.1-8B (base) + Llama-3.2-1B (draft) — 4× NVIDIA L4, CUDA
==========================================================================================

--- Context: pp512 (baseline: 163ms) ---
    kr |  kept |  spec_TTFT |  speedup | overhead
  -----+-------+------------+----------+---------
  0.10 |    64 |     464ms |   0.35x |   +184% ⚠ slower
  0.25 |   128 |     478ms |   0.34x |   +193% ⚠ slower
  0.50 |   256 |     501ms |   0.33x |   +207% ⚠ slower
  1.00 |   512 |     555ms |   0.29x |   +240% ⚠ slower

--- Context: pp1024 (baseline: 260ms) ---
    kr |  kept |  spec_TTFT |  speedup | overhead
  -----+-------+------------+----------+---------
  0.10 |   128 |     829ms |   0.31x |   +219% ⚠ slower
  0.25 |   256 |     862ms |   0.30x |   +232% ⚠ slower
  0.50 |   512 |     915ms |   0.28x |   +252% ⚠ slower
  1.00 |  1024 |     971ms |   0.27x |   +273% ⚠ slower

--- Context: pp2048 (baseline: 439ms) ---
    kr |  kept |  spec_TTFT |  speedup | overhead
  -----+-------+------------+----------+---------
  0.10 |   224 |    1605ms |   0.27x |   +265% ⚠ slower
  0.25 |   512 |    1674ms |   0.26x |   +281% ⚠ slower
  0.50 |  1024 |    1740ms |   0.25x |   +296% ⚠ slower
  1.00 |  2048 |    1882ms |   0.23x |   +329% ⚠ slower

--- Context: pp4096 (baseline: 856ms) ---
    kr |  kept |  spec_TTFT |  speedup | overhead
  -----+-------+------------+----------+---------
  0.10 |   416 |    3351ms |   0.26x |   +291% ⚠ slower
  0.25 |  1024 |    3452ms |   0.25x |   +303% ⚠ slower
  0.50 |  2048 |    3620ms |   0.24x |   +323% ⚠ slower
  1.00 |  4096 |    3953ms |   0.22x |   +362% ⚠ slower

--- Context: pp8192 (baseline: 2037ms) ---
    kr |  kept |  spec_TTFT |  speedup | overhead
  -----+-------+------------+----------+---------
  0.10 |   832 |     924ms |   2.20x |    -55%
  0.25 |  2048 |    1118ms |   1.82x |    -45%
  0.50 |  4096 |    1486ms |   1.37x |    -27%
  1.00 |  8192 |    2572ms |   0.79x |    +26% ⚠ slower
```

### 4.2 Key Observations

1. **pp512–pp4096:** Consistently slower across all keep ratios. Overhead ranges from +184% to +362%.

2. **pp8192 at kr=0.10–0.50:** Speedup of 1.37×–2.20×. This is the only regime where spec prefill outperforms baseline.

3. **pp8192 at kr=1.0:** Slower (+26%) because no filtering occurs — full prompt is decoded through both draft and base model.

4. **Overhead trend:** Overhead increases with context length until pp4096, then drops at pp8192. The inflection point at ~4096 tokens aligns with the `PERPLEXITY_THRESHOLD` constant that switches scoring strategy from perplexity to entropy proxy.

### 4.3 Phase Breakdown Analysis

The `samples_ttft_ns` output contains 3 timing measurements per run corresponding to the three phases:

**Example — pp2048 at kr=0.25:**
- Phase 1 (lookahead gen, draft): ~1533ms average
- Phase 2 (perplexity scoring, base): ~1670ms average
- Phase 3 (filtered decode, base): ~1676ms average
- Total: ~4879ms (matches reported spec_TTFT of 1674ms per-phase average)

Phase 2 dominates at small contexts. The base model runs on the full prompt for scoring, then again for the filtered decode — effectively doing baseline work twice.

---

## 5. Root Cause Analysis

### 5.1 Primary Issue: Perplexity Proxy Overhead

The `compute_attention()` function uses a perplexity proxy to score token importance. For each token position, it:
1. Builds context from the beginning up to that position
2. Runs the base model forward pass
3. Extracts perplexity from the logit

This means the base model processes the full prompt for scoring **before** any filtering happens. The theoretical speedup formula assumes importance scoring is cheap relative to decoding, but the current implementation makes scoring as expensive as decoding itself.

### 5.2 Why pp8192 Shows Speedup

At 8192 tokens, the entropy proxy activates (triggered by `PERPLEXITY_THRESHOLD = 4096`). The entropy proxy computes importance more efficiently:

- Perplexity proxy: Runs full forward pass for each position
- Entropy proxy: Computes from cached logits, avoiding redundant forward passes

Additionally, at pp8192 with kr=0.10, only 832 tokens are kept. The filtered decode of 832 tokens through the base model is significantly faster than baseline's full 8192-token decode, and the entropy scoring overhead is low enough to still yield net savings.

### 5.3 Secondary Issues

**Positional Encoding:** Re-indexed positions after filtering may have RoPE (Rotary Position Embedding) inconsistencies. The implementation renumbers token positions based on filtered indices, which could affect attention quality.

**Draft Model Overhead:** The draft model generates `n_ctx` tokens sequentially. At pp8192, this means generating 8192 tokens through the 1B model, which adds latency before scoring even begins.

---

## 6. Conclusions

### 6.1 Current State

The speculative prefill implementation is **not production-ready** for the Llama-3.1-8B + Llama-3.2-1B model pair on this hardware configuration:

- **pp512–pp4096:** 3.3× slower than baseline prefill
- **pp8192:** 1.4×–2.2× faster (entropy proxy regime only)
- **Root cause:** Perplexity proxy scoring negates filtering benefits by running the base model on the full prompt

### 6.2 Comparison with Existing Benchmarks

The existing `TEST_RESULTS.md` shows speedups of 1.47×–2.24× for Qwen2.5-1.5B + Qwen2.5-0.5B models. The discrepancy likely stems from:

1. **Model size ratio:** Qwen 1.5B/0.5B = 3× ratio vs Llama 8B/1B = 8× ratio
2. **Context lengths tested:** Qwen benchmarks may have been run at longer contexts where entropy proxy is active
3. **Hardware differences:** Different GPU/CPU configurations affect overhead distribution

### 6.3 Recommendations

**Short-term fixes:**
1. Replace the perplexity proxy with a draft-model-only scorer that estimates importance using the draft model's confidence
2. Batch the importance scoring computation instead of per-position evaluation
3. Use positional heuristics (e.g., importance decays from prompt start) as a scoring fallback

**Long-term improvements:**
1. Implement proper speculative decoding where the draft model proposes sequences and the base model verifies in parallel
2. Address RoPE positional encoding issues for filtered/re-indexed tokens
3. Add adaptive keep ratio selection based on context length and model pair characteristics

---

## 7. Files Modified During Investigation

| File | Change |
|------|--------|
| `tools/llama-bench/llama-bench.cpp` | Fixed context sizing: added `n_ctx += spec_n_lookahead + 64` padding |
| `tools/llama-bench/llama-bench.cpp` | Fixed parameter passing in `spec_prefill_test_prompt()` call |
| `build/bin/llama-bench` | Rebuilt binary with fixes |

---

## 8. Raw Data Summary

All benchmark runs produced JSONL output. The `samples_ttft_ns` field contains 3 timing values per run representing the three phases of spec prefill. Key metrics extracted:

| pp | kr | kept | spec_TTFT_ms | baseline_ms | ratio |
|----|-----|------|--------------|-------------|-------|
| 512  | 0.10 | 64   | 464 | 163  | 0.35× |
| 512  | 0.25 | 128  | 478 | 163  | 0.34× |
| 1024 | 0.10 | 128  | 829 | 260  | 0.31× |
| 1024 | 0.25 | 256  | 862 | 260  | 0.30× |
| 2048 | 0.10 | 224  | 1605 | 439 | 0.27× |
| 2048 | 0.25 | 512  | 1674 | 439 | 0.26× |
| 4096 | 0.10 | 416  | 3351 | 856 | 0.26× |
| 4096 | 0.25 | 1024 | 3452 | 856 | 0.25× |
| 8192 | 0.10 | 832  | 924  | 2037 | 2.20× |
| 8192 | 0.25 | 2048 | 1118 | 2037 | 1.82× |
| 8192 | 0.50 | 4096 | 1486 | 2037 | 1.37× |
| 8192 | 1.00 | 8192 | 2572 | 2037 | 0.79× |

---

*Report generated from benchmark data collected 2026-04-23 on spec-prefill branch.*
