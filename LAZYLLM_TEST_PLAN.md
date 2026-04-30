# LazyLLM — Accuracy Verification Test Plan

Goal: increase confidence that our LazyLLM implementation in `lazy_llm` branch
preserves model quality at the pruning ratios where we report TTFT speedups.
All public benchmarks below are sized so a single configuration (one model ×
one keep-ratio × one task) completes in **30–60 minutes** on a single L4.

The plan layers four kinds of evidence: **identity** (LazyLLM at kr=1.0 must
match baseline exactly), **paper-faithful quality** (LongBench, the paper's
own benchmark), **information-preservation** (NIAH/RULER stress tests), and
**reasoning** (GSM8K, MMLU subsets to detect silent capability loss).

---

## Index

1. [Identity & cross-implementation tests](#1-identity--cross-implementation-tests)
2. [LongBench (paper-faithful)](#2-longbench-paper-faithful)
3. [Needle-in-a-Haystack](#3-needle-in-a-haystack-niah)
4. [RULER](#4-ruler)
5. [Reasoning sanity (GSM8K, MMLU)](#5-reasoning-sanity-gsm8k-mmlu)
6. [Code completion (HumanEval, RepoBench)](#6-code-completion)
7. [Determinism & stress tests](#7-determinism--stress-tests)
8. [CI gates](#8-ci-gates)

---

## 1. Identity & cross-implementation tests

These are **fast** correctness checks (< 5 min each), not benchmarks. They run
in `ctest` and on every CI build.

| Test | What it proves | Time | Status |
|------|----------------|------|--------|
| `test-lazyllm-extract` | QK extraction on a tiny graph matches a hand-computed reference | < 30 s | exists |
| `test-lazyllm-pool` | Score smoothing (max-pool over window) | < 5 s | exists |
| `test-lazyllm-topk` | Top-K selection always keeps last token, breaks ties stably | < 5 s | exists |
| **NEW** `test-lazyllm-identity-kr1` | LazyLLM at kr=1.0 produces **bit-identical logits** to baseline | ~10 s | TODO |
| **NEW** `test-lazyllm-rope-positions` | After pruning, kept tokens carry their **original positions** (regression for the RoPE fix) | ~5 s | TODO |
| **NEW** `test-lazyllm-fallback` | Auto-fallback path activates when LazyLLM is slower; output identical to baseline | ~30 s | TODO |
| **NEW** `tools/lazyllm-ref.py` parity | Python reference oracle vs. C++ implementation: **cosine ≥ 0.999** of pruned-token-set indicators on 50 prompts | ~5 min | partial — needs CI hook |

**Acceptance**: all six pass under `ctest -L lazyllm` on every build.

---

## 2. LongBench (paper-faithful)

LongBench is the **paper's own benchmark**. It has 16 subsets organized into
six task categories. Each subset has 200 examples. Our `score-longbench.py`
already routes per-subset to the correct metric.

### 2a. Single-task quality runs (one model, one config, ~45 min each)

Single L4, Llama-3.1-8B-Instruct Q8_0, n_ctx=4096, middle-truncation,
keep_ratio=0.5/0.5/0.5, all 200 examples, baseline + LazyLLM.

| Subset | Type | Metric | Paper baseline (LLaMA-2-7B) | Time/run |
|--------|------|--------|----------------------------|----------|
| **hotpotqa** | multi-doc QA | F1 | 25.4 | ~40 min |
| **2wikimqa** | multi-doc QA | F1 | 32.8 | ~35 min |
| **musique** | multi-doc QA | F1 | 9.4 | ~40 min |
| **narrativeqa** | single-doc QA | F1 | 18.7 | ~50 min |
| **qasper** | single-doc QA | F1 | 19.2 | ~30 min |
| **multifieldqa_en** | single-doc QA | F1 | 36.8 | ~30 min |
| **gov_report** | summarization | Rouge-L | 27.3 | ~60 min (long generation) |
| **qmsum** | summarization | Rouge-L | 20.8 | ~50 min |
| **multi_news** | summarization | Rouge-L | 26.4 | ~45 min |
| **trec** | classification | Accuracy | 61.5 | ~25 min |
| **triviaqa** | few-shot | F1 | 77.8 | ~30 min |
| **samsum** | summarization | Rouge-L | 40.7 | ~30 min |
| **passage_retrieval_en** | synthetic | Accuracy | 9.2 | ~30 min |
| **passage_count** | synthetic | Accuracy | 4.5 | ~25 min |
| **lcc** | code | EditSim | 52.4 | ~30 min |
| **repobench-p** | code | EditSim | 43.7 | ~35 min |

**Acceptance gates** (per subset):
- F1 / Rouge-L / Acc / EditSim **95% CI lo ≥ baseline − 2 percentage points**
- TTFT **median speedup ≥ 1.5×** at kr=0.5, **≥ 2.0×** at kr=0.3

**Command**:

```bash
# Single subset, ~45 min on L4
python3 scripts/get-longbench.py --subset hotpotqa --out hotpotqa.jsonl
./build/bin/llama-lazyllm-run \
    --model models/llama-3.1-8b-instruct-q8_0.gguf \
    --prompts-file hotpotqa.jsonl --n-prompts 200 \
    --n-gpu-layers 999 --n-ctx 4096 \
    --pruning-layers 8 16 24 --keep-ratios 0.5 0.5 0.5 \
    --truncation middle \
    --max-new-tokens "$(python3 scripts/longbench_max_new.py hotpotqa)" \
    --out-csv hotpotqa.csv
python3 scripts/score-longbench.py --subset hotpotqa hotpotqa.csv \
    --out-md hotpotqa-report.md
```

### 2b. Smoke run — all 16 subsets, 50 examples each (~6 hours)

Useful as a nightly job. Drives the `LAZYLLM_REPORT.md` rendered by
`scripts/lazyllm-paper-reproduction.sh`.

---

## 3. Needle-in-a-Haystack (NIAH)

**Why**: LazyLLM prunes by attention score. NIAH directly stresses whether the
"needle" (an injected fact) survives pruning. If LazyLLM drops the needle, F1
on hotpotqa hides the failure mode but NIAH exposes it.

**Setup**: synthetic — no dataset download needed. Insert
`"The magic number is 7421."` at depth `d ∈ {0, 25, 50, 75, 100}` into Paul
Graham essays padded to length `L ∈ {2k, 4k, 8k}`. Ask
`"What is the magic number?"`. Score: exact-match on the digit string.

**Time**: 5 depths × 3 lengths × 5 trials × 2 paths (BL, LZ) = 150 prompts ≈
**~30 min** on L4 for an 8B model.

**Acceptance gates**:
- Baseline pass rate ≥ 95% at all (depth, length) cells (sanity)
- LazyLLM pass rate **≥ 90%** at kr=0.5; **≥ 80%** at kr=0.3
- Heatmap shape (depth × length) qualitatively matches baseline; no
  catastrophic stripe at any single depth

**TODO scripts to add**:
- `scripts/niah-build.py` — generate prompts as JSONL with metadata
- `scripts/niah-score.py` — render pass/fail heatmap as PNG + markdown table

---

## 4. RULER

**Why**: NIAH only tests single-fact retrieval. RULER (NVIDIA, arXiv:2404.06654)
extends it with multi-key, multi-value, multi-query, variable tracking, common
words, frequent words, and QA tasks — 13 tasks total. It is the de-facto
modern standard for long-context evaluation.

**Setup**: synthetic generator (NVIDIA repo). Each task has ~100 examples per
length. Run a **single length** at **subset of tasks** to fit the time budget.

| Configuration | Tasks | Length | Examples | Time on L4 |
|---------------|-------|--------|----------|------------|
| **Quick** (recommended) | niah_single_1, niah_multikey_1, vt, fwe | 4k | 100 each = 400 | **~45 min** |
| Full single-length | all 13 tasks | 4k | 1300 | ~3 h |
| Long-context stress | 4 tasks | 16k | 400 | ~90 min (multi-GPU) |

**Acceptance gate**: per-task accuracy delta within ±5 pp of baseline at kr=0.5.

**TODO**: `scripts/ruler-build.py` wrapping the NVIDIA generator
(`pip install ruler-eval` or vendor the synthetic_data scripts).

---

## 5. Reasoning sanity (GSM8K, MMLU)

**Why**: LongBench mostly tests retrieval/comprehension. Reasoning
(multi-step math, knowledge recall) is a different failure mode — pruning
might keep the "right" passages but break the chain-of-thought.

### 5a. GSM8K (grade-school math)

- 1319 examples; subsample 250 to fit budget
- Format: 8-shot CoT prompt
- Metric: exact-match on final numeric answer
- Context: short (~1.5k tokens) — tests prefill correctness, not long-context
- **Time**: ~45 min for 250 prompts × 256 generated tokens × 2 paths on L4

**Acceptance**: accuracy delta ≤ 3 pp absolute. (LazyLLM should have **near
zero** effect here because contexts are short and pruning keeps most tokens
under stage scheduling.)

### 5b. MMLU (knowledge)

- 14k MCQ questions across 57 subjects; subsample 500 stratified
- Format: 5-shot, MCQ; score the highest-logit option
- Context: short (~2k)
- **Time**: ~30 min for 500 prompts × forward-pass-only on L4

**Acceptance**: accuracy delta ≤ 2 pp absolute.

**TODO**: `tools/lazyllm-eval.py` already supports F1/EditSim/Acc; add
`--mode mcq-logits` to score multiple-choice via logit comparison rather
than generated text.

---

## 6. Code completion (HumanEval, RepoBench)

**Why**: code completion is sensitive to pruning the function signature or
prior context. Already covered partially by LongBench `lcc` and `repobench-p`,
but a dedicated HumanEval run gives a standard external number.

### 6a. HumanEval

- 164 examples; full set fits the budget
- Metric: pass@1 (run generated code in sandbox)
- **Time**: ~30 min for 164 problems × 256 generated tokens × 2 paths

**Acceptance**: pass@1 delta ≤ 5 pp at kr=0.5.

**TODO**: `scripts/humaneval-run.sh` + sandbox runner (use the official
`human-eval` Python package).

---

## 7. Determinism & stress tests

| Test | Description | Time | Why |
|------|-------------|------|-----|
| **Seed determinism** | Run same prompt 3× with `--seed 42`, assert identical output bytes | < 1 min | catches non-deterministic CUDA paths |
| **Multi-GPU parity** | Same model on 1×L4 vs 4×L4: F1 deltas ≤ 1 pp on hotpotqa-50 | ~15 min | covers the recent multi-GPU regression |
| **OOM graceful** | Push n_ctx until `cudaMalloc` fails; assert clean error not crash | ~5 min | regression test for `llama-graph` reserve path |
| **Long-prompt edge cases** | n_prompt = 1, 2, 3, n_ctx-1, n_ctx | ~5 min | catches off-by-one in pruning batch construction |
| **Empty pruning schedule** | `--keep-ratios 1.0 1.0 1.0`: must equal baseline | ~2 min | identity gate (already covered by §1) |
| **Single-stage pruning** | `--pruning-layers 16 --keep-ratios 0.5`: simpler than 3-stage, sanity check | ~5 min | reduces complexity surface |

**TODO**: add as `pytest` cases under `tests/python/test_lazyllm_e2e.py` so
they run alongside C++ ctests.

---

## 8. CI gates

The following matrix should run automatically on each PR to the `lazy_llm`
branch:

### PR gate (~10 min total)

| Test | Time | Required to merge |
|------|------|-------------------|
| `ctest -L lazyllm` (4 unit tests + 3 new identity/RoPE/fallback tests) | < 2 min | yes |
| Smoke: hotpotqa, 20 prompts, kr=0.5, llama-3.2-3b | ~5 min | yes — F1 within ±3 pp |
| Smoke: NIAH 4k, 1 depth, 5 trials, llama-3.2-3b | ~3 min | yes — pass rate ≥ 80% |

### Nightly gate (~2 h total)

| Test | Time | Threshold |
|------|------|-----------|
| LongBench: hotpotqa + qasper + gov_report (200 each), llama-3.1-8b | ~2 h | F1/Rouge-L CI_lo ≥ baseline − 2 pp |
| RULER quick (4 tasks @ 4k), llama-3.1-8b | ~45 min | per-task delta ≤ 5 pp |
| GSM8K 250-sample, llama-3.1-8b | ~45 min | accuracy delta ≤ 3 pp |
| TTFT regression: hotpotqa, kr=0.3, llama-3.1-8b 4K | ~10 min | speedup ≥ 2.0× |

### Weekly gate (~12 h)

Full LongBench (all 16 subsets, 200 examples each) on the three flagship
models: Llama-3.2-3B, Llama-2-7B, Llama-3.1-8B. Renders the official
`LAZYLLM_REPORT.md`.

---

## Time budget summary (single L4, 8B model)

| Test | Examples | Time |
|------|----------|------|
| Unit tests (`ctest -L lazyllm`) | — | < 2 min |
| LongBench single subset (200 ex) | 200 × 2 paths | 30–60 min |
| NIAH (5 depths × 3 lengths × 5 trials) | 75 × 2 paths | ~30 min |
| RULER quick (4 tasks @ 4k) | 400 × 2 paths | ~45 min |
| GSM8K 250-sample | 250 × 2 paths | ~45 min |
| MMLU 500-sample (logit-only) | 500 × 1 fwd | ~30 min |
| HumanEval (164 problems) | 164 × 2 paths | ~30 min |
| Multi-GPU parity (hotpotqa-50) | 50 × 2 paths × 2 configs | ~15 min |

**Recommended first round** (fits in ~2 h, gives full picture):

1. Unit tests — 2 min
2. NIAH 4k — 30 min  *(catches needle-loss bugs)*
3. RULER quick — 45 min  *(modern standard, multi-task)*
4. LongBench hotpotqa (paper-comparable) — 40 min  *(direct paper number)*

These four together verify identity, retrieval correctness, multi-skill
robustness, and paper reproduction.

---

## Open work to enable this plan

| Item | Type | Effort |
|------|------|--------|
| `test-lazyllm-identity-kr1` | C++ ctest | 1 day |
| `test-lazyllm-rope-positions` | C++ ctest | 1 day |
| `test-lazyllm-fallback` | C++ ctest | 0.5 day |
| `scripts/niah-build.py` + `niah-score.py` | Python | 1 day |
| `scripts/ruler-build.py` (vendor NVIDIA generator) | Python | 1 day |
| `scripts/gsm8k-eval.py` | Python | 0.5 day |
| `scripts/mmlu-eval.py` (logit-only) | Python | 0.5 day |
| `scripts/humaneval-run.sh` (sandbox) | Shell + Python | 1 day |
| `tools/lazyllm-eval.py --mode mcq-logits` | Python | 0.5 day |
| GitHub Actions workflow for PR/nightly/weekly gates | YAML | 1 day |

Total: ~7 person-days for the full pipeline; ~2 days to land the "first
round" set (NIAH + RULER quick + identity tests).
