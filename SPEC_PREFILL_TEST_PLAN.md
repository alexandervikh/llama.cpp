# Spec-Prefill Test Plan (C++ / llama.cpp)

Reference: Jingyu6/speculative_prefill (vLLM, paper arXiv:2502.02789).
Goal: validate that the C++ port preserves the reference algorithm's **behavior, quality, and speed** claims.

## 1. Algorithmic Parity (correctness)
Mirror the reference pipeline under a fixed seed and small models (Llama-3.2-1B draft, 7B base), confirming each stage produces equivalent outputs:
- Lookahead generation from draft model (`look_ahead_cnt=8`).
- Attention-score extraction between lookahead Q and prompt K.
- Per-token importance aggregation, average-pool smoothing (`pool_kernel_size=13`).
- Token selection: both `percentage` strategy (e.g. keep 10%) and `chunk`-based (`chunk_size=32`).
- Position-id preservation for kept tokens in base-model forward pass.

Acceptance: same kept-token indices (or ≥95% IoU) as reference for a fixed prompt set.

## 2. Unit / Component Tests (extend existing `test-spec-prefill`)
Expand current 8-case suite with:
- Edge cases: very short prompts (<lookahead), single-token prompts, EOS in lookahead, keep_ratio=1.0 (identity), keep_ratio≈0.
- Determinism: same inputs → same filtered prompt across runs.
- Tokenizer/model mismatch detection (base vs. draft vocab).
- Memory/leak checks via existing CI sanitizer build.

## 2.5 Quality Sanity Gate (pre-parity)
Cheap, fast go/no-go before committing to CUDA-side parity work. Runs on any box with just the tiny smoke model already in use.
- 20 prompts from LongBench `narrativeqa`.
- Same tiny model (Qwen 0.5B) in both base and spec slots — we are testing the *filter*, not the speedup.
- Sweep `keep_ratio ∈ {1.0, 0.5, 0.25, 0.1}`.
- Metric: Rouge-L / F1 vs reference answers.
- Acceptance: score at `kr=0.25` within 30% of `kr=1.0` baseline; output is coherent English (no garbage) at `kr=0.1`.
- If this fails, the filter or position handling has a bug — fix that before spending time on Sections 1 / 3 / 4.

## 3. Quality Evaluation (downstream)
Reproduce the two benchmarks the paper leans on:
- **LongBench** subset (QA, summarization, few-shot, synthetic).
- **RULER / Needle-in-Haystack** at 4k–32k context.

Run three conditions per task: baseline (no prefill), spec-prefill @10%, spec-prefill @25%. Report task metrics (F1/EM/Rouge). Acceptance: within ~2 pts of baseline at 25%, within ~5 pts at 10% on "compressible" tasks — matching the paper's trend.

## 4. Performance Evaluation (the whole point)
- **TTFT** at context lengths 2k / 8k / 32k, batch=1 — must improve monotonically with context length.
- **Throughput / QPS** under concurrent requests (paper's headline metric).
- **Overhead breakdown**: draft-model forward, attention scoring, filtering — confirm attention compute dominates (so GPU savings > CPU overhead, as noted in commit `c4413e93c`).

Run on both CPU and a CUDA build; flag regressions.

## 5. Ablations
- Lookahead count: 1, 4, 8, 16.
- Pool kernel size: 1 (off), 7, 13.
- Chunk vs. percentage strategies.
- Draft-model size (0.5B vs 1B vs 3B) — quality/speed tradeoff curve.

## 6. Integration & Regression
- Ensure non-spec-prefill paths are byte-identical (no perf regression when feature disabled).
- Add spec-prefill to existing CI matrix with a tiny model and one LongBench-lite task as a smoke gate.
- Compare against a simple baseline (random token drop, last-N window) to prove the draft-model signal is actually doing work.
