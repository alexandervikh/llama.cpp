# Self-Layer Prefill — Execution Plan

**Status:** proposal / research track.
**Scope:** variant of spec-prefill that eliminates the external draft model entirely, using self-attention scores from the base model's own early layers as the importance signal.
**Companion docs:** `SPEC_PREFILL_TEST_PLAN.md` (current draft-based impl), `SPEC_PREFILL_TEST_GUIDE.md`.

---

## 1. Motivation

Current spec-prefill needs a second model:
- Matched tokenizer (hard constraint — blocks gpt-oss without gpt-oss draft).
- Extra GPU memory for draft weights + KV.
- Extra lookahead forward pass.
- Cross-impl transferability assumption (small → big, same family).

Observation: **importance scoring only needs attention. Attention only needs Q·K. Q·K is produced by the base model's own early layers for free** during prefill. Drop low-score tokens before the expensive deep layers run.

## 2. Mechanism

### 2.1 Pipeline
1. Run base layers `0..N` on full prompt → produces hidden states + attention matrices.
2. Aggregate per-token importance from layers `0..N` (mean / max / learned weights).
3. Optional average-pool smoothing (`pool_kernel_size=13`, same as paper).
4. Select top-`keep_ratio` tokens, preserving original position IDs.
5. Run base layers `N+1..L` **only on kept tokens** — this is where the speedup lives.
6. Continue generation normally from the last kept position.

### 2.2 Which Q to score with?
Three options, listed by expected quality:

| Option | Mechanism | Cost | Quality risk |
|---|---|---|---|
| **A. Last-position Q** | `score[i] = Σ_layers softmax(Q[last] · K[i])` | free | may miss info not attended by last token |
| **B. Mean-over-prompt Q** | average Q across all prompt positions | free | smears signal |
| **C. Lookahead Q (spec-prefill hybrid)** | generate 1 dummy token via early-exit head, use its Q | +tiny head | best signal, closest to paper |

**Default:** A. Try C if A fails §6.3 sanity gate.

### 2.3 Position-ID handling
Rotary embeddings key off absolute position. After filtering, layers `N+1..L` see fewer tokens but must use their **original** position IDs. Same mechanism already solved in `src/llama-spec-prefill.cpp` (`process_base` sequential re-index path).

### 2.4 Layer-N choice
- Too shallow (`N<3`): attention patterns still local/syntactic, weak signal.
- Too deep (`N>L/2`): prefill cost of layers `0..N` eats the savings.
- Paper intuition + LazyLLM evidence: `N ≈ L/4` to `L/3` (e.g., layers 0..8 of a 32-layer model).
- Ablate in §7.

## 3. Relationship to prior work

| Technique | What it does | How this differs |
|---|---|---|
| **SpecPrefill** (arXiv:2502.02789) | Separate draft model generates lookahead, computes Q·K for importance | Drops draft, uses base's own early layers |
| **LazyLLM** (Apple, arXiv:2407.14057) | Progressive token pruning during prefill using self-attention | Near-identical to this plan; LazyLLM prunes at multiple layers with adaptive thresholds |
| **H2O** (arXiv:2306.14048) | Heavy-Hitter oracle for KV-cache eviction during decode | Different target (decode, not prefill) |
| **FastGen** (arXiv:2310.01801) | Adaptive KV compression using attention structure | Decode-time |
| **Scissorhands** (arXiv:2305.17118) | KV-cache compression via token-importance persistence | Decode-time |
| **Dynamic Context Pruning** (arXiv:2305.15805) | Learnable token dropping | Requires training |
| **LayerSkip** (Meta, arXiv:2404.16710) | Early-exit self-speculative *decode* | Decode acceleration, not prefill filtering |

**Closest analog:** LazyLLM. Read it before implementing. This plan is a simplified single-threshold variant of LazyLLM's progressive pruning.

## 4. Comparison vs current spec-prefill

| Axis | Spec-prefill (current) | Self-layer prefill |
|---|---|---|
| Draft model | Required, same tokenizer | **None** |
| Works on gpt-oss-20B solo | No (no smaller draft) | **Yes** |
| Works on any model | Only if matched draft exists | **Yes, universally** |
| Extra weights | Draft GGUF (0.5–1B) | None |
| Extra KV cache | Draft KV | None |
| Lookahead forward | Separate draft pass | Free (layers 0..N of base) |
| Tokenizer mismatch risk | Real | **None by construction** |
| Transferability assumption | Small → big, same family | **Shallow → deep, same model** |
| Paper reproducibility | Yes (SpecPrefill) | No (LazyLLM adjacent, new for llama.cpp) |
| Implementation complexity | Higher (two contexts, sync) | **Lower (single forward, one slice)** |

## 5. Implementation plan (llama.cpp)

### Phase -1 — Branch setup

Current branch: `spec-prefill` (8 commits ahead of master: `ae83b1def` … `c2c857d6b`).
History pattern: all spec-prefill work on feature branch, `spec-prefill:` commit prefix, `backup-before-cleanup` kept as safety net.

**Base from `spec-prefill`, not master** — reuse existing infra:
- Test harness (`test-spec-prefill*.cpp`, CTest wiring)
- Eval scripts (`eval/score_rouge.py`, `eval/run_longbench.py`, `eval/quality_prompts.jsonl`)
- Position-ID / KV-reset machinery (`process_base`, `llama_memory_clear`)
- Driver pattern (`examples/spec-prefill-run/`)

```bash
# From repo root, current branch = spec-prefill
git fetch origin
git checkout spec-prefill
git pull --ff-only origin spec-prefill
git checkout -b self-layer-prefill
# optional safety net matching existing convention
git branch backup-before-self-layer
```

**Commit prefix:** `self-layer-prefill:` (mirrors `spec-prefill:` style).

**Divergence rules:**
- Do NOT modify `src/llama-spec-prefill.cpp` or `include/llama-spec-prefill.h` from this branch — new files only (`llama-self-layer-prefill.*`).
- Share eval scripts by adding flags, not forking.
- If a fix benefits both features (e.g., position-ID edge case), cherry-pick or branch off a shared base commit.

**Merge target:** back to `spec-prefill` when §6.1 gates pass. From there, either to `master` alongside spec-prefill or as replacement — decided by §6.3 head-to-head result.

### Phase 0 — Instrumentation (1–2 days)
- Add `--dump-attn-at-layer N` flag to `llama-cli` for experimentation.
- Dump per-layer attention scores on a fixed prompt set.
- Verify shape + values match expectation.

### Phase 1 — Minimal prototype (3–5 days)
- New source: `src/llama-self-layer-prefill.cpp`, header `include/llama-self-layer-prefill.h`.
- Public API (mirror `llama-spec-prefill.h`):
  ```c
  struct llama_self_layer_prefill_params {
      int   n_early_layers;      // N
      float keep_ratio;          // e.g. 0.25
      int   pool_kernel_size;    // 13
      bool  use_chunking;
      int   chunk_size;
      enum score_strategy { LAST_Q, MEAN_Q, LOOKAHEAD_Q };
  };
  ```
- Hook into `llama_decode` path: intercept after layer `N`, compute scores, build kept-index set, re-run layers `N+1..L` on filtered tokens.
- Reuse `process_base` position-ID handling.
- No new model weights; no prediction head yet.

### Phase 2 — CLI + example (1–2 days)
- `examples/self-layer-prefill-run/main.cpp` — mirror `spec-prefill-run` flags.
- Accept `--model`, `--early-layers`, `--keep-ratio`, `--pool`, `--score-strategy`, `--prompt-file`, `--out`.

### Phase 3 — Tests (3–5 days)
- `tests/test-self-layer-prefill.cpp` — edge cases, determinism, `kr=1.0` identity, position-ID preservation.
- `tests/test-self-layer-prefill-quality.cpp` — §2.5-style sanity gate.
- `tests/test-self-layer-prefill-integration.cpp` — sanity baselines (random, last-N).
- ASan build, CTest labels.

### Phase 4 — Benchmarks (2–3 days)
- Extend `test-spec-prefill-bench.cpp` or new `test-self-layer-prefill-bench.cpp`.
- Head-to-head vs spec-prefill on canonical pair (Llama-3.1-8B).
- TTFT sweep ctx ∈ {2k, 8k, 32k}, `kr ∈ {0.1, 0.25, 0.5}`, `N ∈ {L/8, L/4, L/3, L/2}`.

### Phase 5 — gpt-oss application (2–3 days)
- Run self-layer prefill on gpt-oss-20B solo.
- Log per-layer `attn_type` (full / SWA) — pick `N` so layers `0..N` are full-attn only.
- Report TTFT + quality against no-filter baseline.

## 6. Test plan

### 6.1 Required gates (must pass before merge)

| Gate | Mechanism | Acceptance |
|---|---|---|
| Unit / determinism | Same input + fixed seed → identical kept set | 100% on 20 prompts |
| `kr=1.0` identity | No filtering → byte-identical to plain prefill | exact match |
| Sanity baselines | Beat random-drop and last-N-window | ≥ 5 pts Rouge-L |
| Quality proxy | §2.5-style ratio on Llama-3.2-1B self-test | `ratio ≥ 0.70` at `kr=0.25` |
| Feature-off identity | Compiled-in but not invoked → bit-identical baseline | pass |

### 6.2 Real benchmarks (release claim)
Same datasets as `SPEC_PREFILL_TEST_PLAN.md` §3b:
- LongBench (6 tasks) at `kr ∈ {0.10, 0.25, 1.0}`.
- RULER / NiH at ctx ∈ {4k, 8k, 16k, 32k}.

**Acceptance:**
- LongBench `kr=0.25`: within 2 pts of baseline on average.
- LongBench `kr=0.10`: within 5 pts on compressible tasks.
- RULER pass-rate within 5 pts at 8k.
- TTFT speedup ≥ 1.5× at 8k ctx on GPU with `kr=0.25`.

### 6.3 Head-to-head vs spec-prefill

On the canonical Llama-3.1-8B + Llama-3.2-1B pair, run both:
- Spec-prefill (current, with draft)
- Self-layer prefill (this proposal, no draft)

Report matched table: quality (Rouge-L per task), TTFT, peak memory. No formal acceptance — this is for deciding which method ships as default.

## 7. Ablations

| Axis | Values | Goal |
|---|---|---|
| `N` (early layers) | {L/8, L/4, L/3, L/2} | find cost/quality sweet spot |
| Score strategy | {LAST_Q, MEAN_Q, LOOKAHEAD_Q} | validate signal choice |
| `pool_kernel_size` | {1, 7, 13} | carry-over from paper |
| `use_chunking` × `chunk_size` | {false} ∪ {true × 16, 32, 64} | carry-over |
| `keep_ratio` | {0.1, 0.25, 0.5, 1.0} | quality curve |

## 8. Risks & open questions

1. **Signal quality** — early-layer self-attention may be weaker than lookahead-driven attention. If §6.1 sanity gate fails, fall back to LOOKAHEAD_Q (option C) which reintroduces a tiny logit-lens head.
2. **Model-specific `N`** — optimal layer cutoff may differ per architecture. Start with `N=L/4`, ablate per model.
3. **SWA interference (gpt-oss)** — if any layer in `0..N` is sliding-window, importance signal under-weights early tokens. Mitigation: restrict N to full-attn prefix, or score only full-attn layers.
4. **MoE (gpt-oss)** — attention itself is dense, so pre-routing. No new issue beyond matched-tokenizer spec-prefill.
5. **Position-ID correctness after drop** — covered by existing `process_base` machinery, but needs a dedicated regression test.
6. **Memory win is smaller than expected** — KV cache for layers `0..N` is still allocated for the full prompt. Only layers `N+1..L` see the reduction. Measure and report honestly.
7. **Reproducibility** — not a paper repro. LazyLLM numbers are the closest prior art; cite and compare.

## 9. Execution order

1. Phase 0 instrumentation.
2. Phase 1 prototype on Llama-3.1-8B.
3. §6.1 gates on Llama-3.2-1B self-test.
4. Phase 2 CLI + Phase 3 tests.
5. Phase 4 TTFT benchmarks + §6.3 head-to-head vs spec-prefill.
6. Phase 5 gpt-oss-20B application — the killer use case.
7. Ablations §7.
8. §6.2 real LongBench + RULER.
9. Write-up. Decide: ship as alternative mode, or replace draft-based spec-prefill?

## 10. Deliverables

- `include/llama-self-layer-prefill.h`
- `src/llama-self-layer-prefill.cpp`
- `examples/self-layer-prefill-run/`
- `tests/test-self-layer-prefill*.cpp`
- `eval/run_self_layer_benchmarks.sh`
- `SELF_LAYER_PREFILL_REPORT.md` (results)
- CI job wired to `ctest -L self-layer-prefill`

## 11. Out of scope

- Training any new head (Medusa / EAGLE / learned router) — kept as follow-up if §6.1 fails with option A and C.
- Progressive multi-layer pruning (full LazyLLM) — start single-threshold; add later if quality-speed curve demands it.
- Decode-time KV compression (H2O / Scissorhands territory).
- Cross-model distillation.
