# Spec-Prefill Test Plan (C++ / llama.cpp)

Reference: Jingyu6/speculative_prefill (vLLM, paper arXiv:2502.02789).
Goal: validate that the C++ port preserves the reference algorithm's **behavior, quality, and speed** claims.

## Canonical model pair (matches paper)

- **Base**: `meta-llama/Meta-Llama-3.1-8B-Instruct` (GGUF Q4_K_M, F16 for parity work)
- **Draft**: `meta-llama/Llama-3.2-1B-Instruct` (GGUF Q4_K_M, F16 for parity work)

Same tokenizer family (128k Llama-3 BPE), dense architecture, full attention + GQA — matches the reference impl's hard `LlamaForCausalLM` assertion. Both fit on a single L4 (8B Q4 ≈ 5 GB + 1B Q4 ≈ 0.7 GB).

Historical runs used Qwen3-0.6B self-spec for quick turnaround; those results are retained but are **not** a paper repro. See §8 for extension targets (Qwen, gpt-oss) and their separate acceptance criteria.

## 1. Algorithmic Parity (correctness)

Split into two sub-goals because cross-impl parity depends on vLLM, which does not install cleanly in our env.

### 1a. Self-determinism (required, cheap)
Same C++ build, same inputs, fixed seed → byte-identical kept-token indices across runs.
- **Acceptance:** 100% match on ≥ 20 prompts at `keep_ratio ∈ {0.10, 0.25, 1.0}`.

### 1b. Cross-impl parity vs vLLM reference (STRETCH — DEFERRED)
Mirror the reference pipeline stage-by-stage:
- Lookahead generation (`look_ahead_cnt=8`, greedy).
- Attention-score extraction (lookahead Q × prompt K).
- Per-token importance aggregation + average-pool (`pool_kernel_size=13`).
- Selection: `percentage` (10%) and `chunk` (`chunk_size=32`).
- Position-id preservation.

**Status: DEFERRED.** vLLM 0.8.5 environment cannot be stood up (torch/xformers dependency conflicts). 

**Current validation:** `tools/spec-prefill-parity-mock.sh` compares C++ output against Python ref-impl (`tools/spec-prefill-ref-impl.py`). This validates algorithmic self-consistency but is NOT cross-impl parity — it compares C++ against our own Python re-implementation.

**Acceptance (when vLLM env becomes available):** mean kept-index IoU ≥ 0.95 at `kr=0.25`, ≥ 0.90 at `kr=0.10`; pre-pool / post-pool importance Pearson correlation ≥ 0.98 on 50-prompt suite.

## 2. Unit / Component Tests

Expand current suite in `tests/test-spec-prefill.cpp` and `tests/test-spec-prefill-extended.cpp`.

- Edge cases: prompt < `n_lookahead`, single-token, EOS mid-lookahead, `keep_ratio=1.0` (identity), `keep_ratio≈0`.
- Determinism: same inputs → same filtered prompt across runs.
- Tokenizer/vocab mismatch detection on init (base vs draft).
- **Q-tensor extraction** (added 2026-04-22, commit `68c8becf2`): `is_q_tensor_name()` matches per-layer Q tensors, `llama_spec_q_tensor` carries layer metadata, `get_gf_res_prev()` returns the graph result on which it was computed.
- **Lookahead init**: token=0 (not `prompt_tokens[n_prompt-1]`) — regression guard for the fix in `68c8becf2`.
- Memory/leak checks via a dedicated ASan/UBSan build (see guide §2.2); **currently not in CI** — either wire it in or mark as deferred in the status table.

## 2.5 Quality Sanity Gate (pre-parity)

Cheap go/no-go before spending time on §1b or §3.
- 20 prompts from `THUDM/LongBench` `narrativeqa` split, truncated to 2k tokens.
- **Same-model self-spec** on the canonical draft (`Llama-3.2-1B-Instruct` in both slots) — we are testing the filter, not speedup.
- Sweep `keep_ratio ∈ {1.0, 0.5, 0.25, 0.1}`; temperature=0.
- Metric: Rouge-L vs reference answer.
- **Acceptance:** `Rouge-L(kr=0.25) / Rouge-L(kr=1.0) ≥ 0.70`; output at `kr=0.1` is coherent English on manual spot-check (5 samples).
- If this fails: filter or position handling is broken; do not proceed to §1b / §3 / §4.

## 3. Quality Evaluation (downstream)

Split into smoke proxy (cheap, already shipped) vs the real paper benchmarks (required for a defensible claim).

### 3a. Synthetic proxy (done)
`eval/quality_prompts.jsonl` — 20 hand-crafted long-context prompts. Scored with `eval/score_rouge.py`. This is a **smoke test**, not LongBench.
- **Acceptance:** matches the §2.5 gate; same threshold (`ratio ≥ 0.70`).
- **Do not report this as LongBench results.**

### 3b. Real LongBench + RULER (required for release claim)
Load the actual HuggingFace datasets:
- `THUDM/LongBench` — pick 4–6 tasks: `narrativeqa`, `qasper`, `multifieldqa_en`, `hotpotqa`, `gov_report`, `passage_retrieval_en`.
- `RULER` / Needle-in-Haystack at ctx ∈ {4k, 8k, 16k, 32k}.

Conditions per task: baseline (no prefill), kr=0.25, kr=0.10.

**Acceptance:**
- LongBench `kr=0.25`: within **2 pts** of baseline on average.
- LongBench `kr=0.10`: within **5 pts** on compressible tasks (QA, summarization); larger gap on retrieval is tolerated and documented.
- RULER pass-rate within **5 pts** at 8k; degradation at 32k documented (not a hard fail).

## 4. Performance Evaluation

- **TTFT** at ctx ∈ {2k, 8k, 32k}, batch=1 — must improve monotonically with context length.
- **QPS** under concurrent requests (the paper's headline metric).
- **Overhead breakdown**: draft forward, QK extract, attention compute, filter.

### Builds
- **GPU (CUDA)** — reference build for all long-context numbers.
- **CPU** — known to hang on prompts > ~4k tokens (bug tracked separately). Report ≤ 4k only; mark pp8k/16k/32k as `N/A (CPU build bug)` until fixed.
- **Metal** — if an M-series host is available; otherwise mark deferred.

**Acceptance:** TTFT speedup ≥ 1.5× at 8k ctx on GPU with `kr=0.25` vs baseline prefill on the same build.

## 5. Ablations

All ablations run the §3b LongBench subset on the canonical pair unless stated otherwise.

- `n_lookahead ∈ {1, 4, 8, 16}`
- `pool_kernel_size ∈ {1, 7, 13}` (1 = effectively off)
- `use_chunking ∈ {false, true}` × `chunk_size ∈ {16, 32, 64}`
- **Draft-model size (same tokenizer family only)**: `Llama-3.2-1B-Instruct` vs `Llama-3.2-3B-Instruct`. Do **not** mix tokenizer families (Qwen draft with Llama base fails the vocab check by design).

**Output:** quality-vs-speedup scatter with Pareto front. Current defaults (`lah=8, pool=13, chunk=32`) must sit on or within 2% of the front; if not, update `llama_spec_prefill_params` defaults.

## 6. Integration & Regression

- **Feature-off identity:** `llama-perplexity` / `llama-bench` on `wikitext-2` with spec-prefill compiled but not invoked must be bit-identical to the pre-feature build.
- **CI smoke gate:** `ctest -R spec-prefill` on a tiny model pair (Llama-3.2-1B self-spec) + 10-prompt quality check, runtime budget < 5 min.
- **Sanity baselines:** random-drop and last-N-window filters at the same keep_ratio. Spec-prefill must beat both by ≥ 5 pts Rouge-L average — proves the draft-model signal is doing real work (historical gap measured: 0.77 against random-like).

## 7. Release Hygiene Gates

Pre-merge checklist:
- No core dumps in the tree (`git ls-files 'core.*'` empty).
- All test artifacts either committed or in `.gitignore` (`git status --porcelain` clean except for intended edits).
- `ctest -R spec-prefill` passes on a fresh clone with no local build cache.
- `SPEC_PREFILL_{TEST_PLAN,TEST_GUIDE,RUNBOOK,README,FINAL_REPORT}.md` and `VALIDATION_SUMMARY.md` are consistent (no section claimed PASS that is not actually pass in the numeric results).
- `tools/spec-prefill-*.py` scripts are committed (not untracked) if they are referenced from the docs.

## 8. Extension Targets (not paper repro — separate criteria)

These are research extensions beyond the paper's evaluated scope. Treat results as new data points, not reproductions.

### 8a. Qwen family (same-tokenizer, dense or MoE)
- `Qwen2.5-7B-Instruct` base + `Qwen2.5-0.5B-Instruct` draft — dense, same tokenizer, analogous to paper setup. Safe extension.
- `Qwen3-8B` + `Qwen3-0.6B` — same, paper-style.
- Acceptance: same thresholds as §3b for the corresponding ctx lengths.

### 8b. gpt-oss (MoE + sliding-window attention)
- `gpt-oss-120B` base + `gpt-oss-20B` draft — the only matched-tokenizer option (`o200k_harmony`).
- **Architectural hazards to instrument before trusting any numbers:**
  1. **Sliding-window layers** interleave with full-attn layers. Log per-layer `attn_type` and either (a) restrict Q extraction to full-attn layers, or (b) normalize per-layer before aggregation so SWA layers do not systematically under-weight early prompt tokens.
  2. **MoE routing** — attention itself is dense, so Q/K extraction is unaffected; but the "importance transferability" claim from the paper was established only across dense Llama family members, so filter quality is unvalidated for MoE targets.
  3. **Harmony chat template** (`<|start|>`, `<|message|>`, `<|channel|>`, `<|return|>`) — prompts must be wrapped, otherwise scores are garbage for chat-style tasks.
- Acceptance: TTFT speedup measured but quality gap thresholds **loosened** (target: kr=0.25 within 5 pts instead of 2; retrieval tasks documented but not gated). Self-spec (20B/20B) runs first as a plumbing test before any 120B/20B run.

---

## Execution Order (suggested)

1. §2 (unit + Q-tensor + lookahead-init regression) — get CI green.
2. §2.5 (quality sanity gate on canonical pair) — cheap go/no-go.
3. §4 TTFT sweep on canonical pair, GPU — confirm speedup claim on our hardware.
4. §1a (self-determinism) — trivially falls out of §2 runs.
5. §3b (real LongBench + RULER) — this is the paper repro; results go in the report.
6. §5 (ablations) — tune defaults on the canonical pair.
7. §1b (cross-impl parity vs vLLM) — only if the vLLM env can be stood up; otherwise mark deferred.
8. §6 (integration + sanity baselines) — ship gate.
9. §7 (release hygiene) — pre-merge checklist.
10. §8 (extensions: Qwen, gpt-oss) — research follow-ups, separate report.
