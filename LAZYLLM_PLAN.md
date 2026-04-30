# LazyLLM — Implementation, Testing & Benchmarking Plan

**Branch:** `lazy_llm`
**Reference paper:** *LazyLLM: Dynamic Token Pruning for Efficient Long Context LLM Inference* — Fu et al., Apple, 2024. [arXiv:2407.14057](https://arxiv.org/abs/2407.14057)
**Companion docs:** `RESULTS_REPORT.md`, `quality-results/LAZYLLM_COMPREHENSIVE_REPORT.md`, `SELF_LAYER_PREFILL_PLAN.md`

> **Goal of this document.** Bring the existing POC to a state where we can claim *paper-faithful reproduction* of LazyLLM in `llama.cpp`. That means: same models, same datasets, same hyperparameters, same metric definitions, with TTFT speedup and quality numbers within publishable tolerance of the paper.

---

## 0. TL;DR

- The C++ POC already implements multi-stage progressive pruning, on-GPU score pooling, embedding injection, and an auto-fallback safety net for multi-GPU.
- TTFT speedup on **single-GPU instruct Llama** matches or beats the paper (2.40–2.41× at `kr=0.3/0.3/0.3`).
- **Quality has not been validated against the paper's actual benchmark** — only on a 20-sample HotpotQA subset with non-paper prompt formatting.
- **Multi-GPU layer-split provides parity at best**, not a speedup.
- This plan closes those two gaps and adds the missing architecture coverage.

---

## 1. Paper specification — what we must match

### 1.1 Models (exactly as in the paper)

| Role | Model | HF path | Source-of-truth weights |
|------|-------|---------|-------------------------|
| Primary 7B | **LLaMA-2-7B (base)** | `meta-llama/Llama-2-7b-hf` | Llama-2 license; gated |
| Primary 7B chat | **LLaMA-2-7B-Chat** | `meta-llama/Llama-2-7b-chat-hf` | Llama-2 license; gated |
| Secondary 7B (heterogeneous arch) | **XGen-7B-8K-base** | `Salesforce/xgen-7b-8k-base` | Apache-2.0 |
| Larger model (paper Table 2) | **LLaMA-2-13B** | `meta-llama/Llama-2-13b-hf` | Llama-2 license; gated |

> **Why these specifically:** the paper's headline TTFT numbers (2.34× / 2.65× max for 7B, 2.16× for 13B) and its quality tables are computed on these checkpoints. Substituting Llama-3 or instruction-tuned variants is fine for *additional* numbers (we already have those) but **does not constitute paper reproduction** — Llama-3's tokenizer, RoPE base, and FFN sizing differ.

GGUF conversion (one-time, per checkpoint):

```bash
# After accepting the meta-llama license + huggingface-cli login
python3 convert_hf_to_gguf.py \
    --outfile models/llama-2-7b-f16.gguf \
    --outtype f16 \
    /path/to/hf-cache/Llama-2-7b-hf

# Quantize to match practical hardware (paper used FP16 on A100; we mirror with Q8_0 on L4).
./build/bin/llama-quantize models/llama-2-7b-f16.gguf models/llama-2-7b-q8_0.gguf Q8_0
./build/bin/llama-quantize models/llama-2-7b-f16.gguf models/llama-2-7b-q4_k_m.gguf Q4_K_M
```

Same procedure for `Llama-2-7b-chat-hf`, `xgen-7b-8k-base`, `Llama-2-13b-hf`.

> **Quantization note.** The paper uses FP16 weights on A100. Our reference GPU is 4× L4 24 GiB (96 GiB total). Llama-2-7B-FP16 fits on a single L4 (~14 GiB). Llama-2-13B-FP16 needs a 2-GPU split or Q8_0 single-GPU. **All headline numbers must be reported at FP16 first; quantized rows are supplementary.**

### 1.2 Hardware (and adaptation)

| | Paper | This plan (reference run) | Notes |
|--|------|---------------------------|-------|
| GPU | 4× NVIDIA A100 80 GB | 4× NVIDIA L4 24 GB | ~½ memory bandwidth, ~⅓ FP16 TFLOPS |
| Driver | not stated | 550.x | |
| CUDA | not stated | 12.x | |

We cannot reproduce A100 wall-clocks; we **can** reproduce *speedup ratios* (LazyLLM TTFT ÷ baseline TTFT), which is what the paper actually claims.

### 1.3 Datasets (exactly as in the paper)

LongBench (Bai et al., 2023), all 16 English subsets, 6 categories:

| Category | Sub-dataset | Avg length | Metric |
|----------|-------------|-----------:|--------|
| Single-Doc QA | NarrativeQA | 18,409 | F1 |
| | Qasper | 3,619 | F1 |
| | MultiFieldQA-en | 4,559 | F1 |
| Multi-Doc QA | **HotpotQA** | 9,151 | F1 |
| | **2WikiMQA** | 4,887 | F1 |
| | **MuSiQue** | 11,214 | F1 |
| Summarization | GovReport | 8,734 | Rouge-L |
| | QMSum | 10,614 | Rouge-L |
| | MultiNews | 2,113 | Rouge-L |
| Few-shot | TREC | 5,177 | Cls-Acc |
| | TriviaQA | 8,209 | F1 |
| | SAMSum | 6,258 | Rouge-L |
| Synthetic | PassageCount | 11,141 | Acc |
| | PassageRetrieval-en | 9,289 | Acc |
| Code | LCC | 1,235 | Edit-sim |
| | RepoBench-P | 4,206 | Edit-sim |

The paper's headline 2.34× speedup is reported on **Multi-Doc QA**. We must report all 16; speedup on the long subsets (Single-Doc QA, Summarization, Multi-Doc QA) is the primary claim.

### 1.4 Hyperparameters (paper §4.1, Table 1)

| Param | Paper value | Our default (matches) |
|-------|-------------|------------------------|
| Pruning points | 3 progressive | `pruning_layers = {8, 16, 24}` for 32-layer (= L/4, L/2, 3L/4) |
| Keep ratios | progressive 0.7 / 0.5 / 0.3 | `keep_ratios = {0.7f, 0.5f, 0.3f}` |
| Aggressive schedule (paper Table 4 ablation) | 0.5 / 0.4 / 0.3 | secondary report |
| Smoothing pool | size-13 average pool | `pool_kernel_size = 13` |
| Scoring | softmax(QKᵀ/√d) of last query position | implemented in `lazyllm-pool.h` |
| Token-position handling | original positions preserved (RoPE) | **partially**: pruning preserves positions, but **deep re-decode currently uses sequential positions** — see §3 gap #1 |
| Aux Cache (decode revival) | yes | implemented (`llama_lazyllm_aux_cache_*`) |
| Flash Attention | not used in scoring path | mirrored (FA off for score extraction) |

### 1.5 Generation settings per LongBench subset

These come from `THUDM/LongBench/config/dataset2maxlen.json` and `dataset2prompt.json`. The paper inherits them. We must use the same:

| Subset | `max_new_tokens` | `n_ctx` (paper used 4096) |
|--------|-----------------:|---------------------------|
| narrativeqa | 128 | 4096 |
| qasper | 128 | 4096 |
| multifieldqa_en | 64 | 4096 |
| hotpotqa | 32 | 4096 |
| 2wikimqa | 32 | 4096 |
| musique | 32 | 4096 |
| gov_report | 512 | 4096 |
| qmsum | 512 | 4096 |
| multi_news | 512 | 4096 |
| trec | 64 | 4096 |
| triviaqa | 32 | 4096 |
| samsum | 128 | 4096 |
| passage_count | 32 | 4096 |
| passage_retrieval_en | 32 | 4096 |
| lcc | 64 | 4096 |
| repobench-p | 64 | 4096 |

Decoding: **greedy** (`temp=0`, `top_k=1`) — paper uses greedy; we must too.
Truncation: when prompt > `n_ctx − max_new_tokens`, the paper drops the **middle** of the context (LongBench convention), keeping the head and tail. We currently do tail-only truncation — see §3 gap #2.

---

## 2. Current implementation status

### 2.1 What's working

| Component | Status | Source |
|-----------|--------|--------|
| Multi-stage progressive pruning prefill | ✅ | `src/llama-lazyllm.cpp` |
| Per-layer attention extraction (FA-off) | ✅ | `llama_lazyllm_extract_attention` |
| Flash-attention compatible scoring | ✅ | `lazyllm-pool.h::lazyllm_find_tensor_last` |
| On-GPU score pooling | ✅ (≈16 KB transfer per stage vs ≈2 GB naive) | `lazyllm-pool.h` |
| Average-pool smoothing (k=13) | ✅ | `llama_lazyllm_apply_pooling` |
| Top-K with last-token always kept | ✅ | `llama_lazyllm_top_k_indices` |
| Embedding injection (`embd != null` batches) | ✅ | `src/llama-batch.cpp` continuity bypass |
| `decode_partial(il_start, il_end)` | ✅ | `src/llama-context.cpp` |
| `LLM_GRAPH_TYPE_PARTIAL` graphs | ✅ | LLAMA, LLAMA4, QWEN2, QWEN3, OPENAI_MOE |
| Aux cache (dropped-token hidden states) | ✅ API + tests | `llama_lazyllm_aux_cache_*` |
| Decode-stage KV pruning | ✅ API | `llama_lazyllm_decode_step` |
| Aux-cache token revival | ✅ API | `llama_lazyllm_revive_token` |
| `llama-bench` integration (`--lazyllm`) | ✅ | `tools/llama-bench/llama-bench.cpp` |
| Auto-fallback A/B test (3-rep, drop cold-start) | ✅ | `llama_lazyllm_warmup` |
| Python reference oracle | ✅ | `tools/lazyllm-ref.py` |
| LongBench dataset downloader | ✅ | `scripts/get-longbench.py` (multi_doc_qa only) |
| F1 scorer | ✅ | `scripts/score-f1.py` |

### 2.2 Numbers we have

From `RESULTS_REPORT.md` (single L4, FA on, no CUDA graphs):

| Model | Ctx | Schedule | Baseline TTFT | LazyLLM TTFT | Speedup |
|-------|----:|----------|--------------:|-------------:|--------:|
| Llama-3.1-8B Q8 | 4K | 0.7/0.5/0.3 | 1747 ms | 954 ms | 1.83× |
| Llama-3.1-8B Q8 | 4K | 0.5/0.4/0.3 | 1749 ms | 725 ms | **2.41×** |
| Llama-2-7B Q8 | 4K | 0.7/0.5/0.3 | 1632 ms | 914 ms | 1.79× |
| Llama-2-7B Q8 | 4K | 0.5/0.4/0.3 | 1638 ms | 683 ms | **2.40×** |

> These already cross the paper's 2.34× target — but on **Llama-3.1-8B Q8**, not on **Llama-2-7B FP16** as in the paper, and TTFT is timed on a *synthetic* prompt with a non-paper schedule. The plan below redoes these on the right model + dataset.

---

## 3. Open gaps (must close before "paper reproduction" can be claimed)

Numbered for cross-reference from §4–§7.

| # | Gap | Severity | Owner area |
|---|-----|----------|------------|
| **1** | Deep re-decode uses sequential positions `0..n_kept−1`, not original positions. Breaks RoPE phase → degrades instruction-following. | **Blocker for quality** | `src/llama-lazyllm.cpp` `prefill_remainder()` |
| **2** | Truncation is tail-only; LongBench convention is mid-truncation (head+tail). | **Blocker for quality numbers** | `examples/lazyllm-run/main.cpp` |
| **3** | Multi-GPU layer-split delivers parity, not speedup. Per-stage sync overhead dominates. | **Blocker for multi-GPU claims** | `src/llama-lazyllm.cpp`, ggml-cuda sched |
| **4** | Cross-pass KV reuse not implemented (paper "ideal" path). Adds ~10–15% overhead vs theoretical bound. | High | `src/llama-memory*` |
| **5** | LongBench downloader covers only `multi_doc_qa` (3 / 16 subsets). | High (blocks coverage) | `scripts/get-longbench.py` |
| **6** | Quality reports built from 20-prompt samples, no bootstrap CI on the full 200-per-subset paper subset. | High | `tools/lazyllm-eval.py` |
| **7** | `qwen35` / `qwen35moe` partial graph builders missing. | Medium (out of paper scope, in our scope) | `src/models/` |
| **8** | gpt-oss (OPENAI_MOE) quality drops sharply — likely SWA-layer / pruning-on-SWA interaction. | Medium | `src/models/openai-moe-partial.cpp` |
| **9** | Aux Cache revival path (`llama_lazyllm_revive_token`) has no end-to-end test against a generation that benefits from it. | Medium | tests + `examples/lazyllm-run/` |
| **10** | LongBench official scoring uses dataset-specific metrics (Rouge-L for summarization, EditSim for code). We only have F1. | Medium | `scripts/score-*.py` |
| **11** | XGen-7B-8K not yet loaded (different arch than Llama-2). Needs GGUF conversion test + verification. | Medium | conversion scripts |
| **12** | Llama-2-13B (paper's larger model) not yet benchmarked. | Medium | bench infra |
| **13** | No CTest label `ctest -L lazyllm` — tests exist but aren't grouped. | Low | `tests/CMakeLists.txt` |
| **14** | No CI job exercising `--lazyllm` in `llama-bench`. | Low | `.github/workflows/` |
| **15** | `LAZYLLM_REPORT.md` (single canonical results doc) doesn't exist; status is split across 5 .md files. | Low | docs |

---

## 4. Implementation plan

Phases are sized to be independently shippable. Each phase ends with a concrete artifact that can be PR-reviewed in isolation. Estimated effort assumes one engineer.

### Phase A — Quality fundamentals (gaps #1, #2)  [3–4 days]

**A1. Original-position re-decode.** When the deep re-decode pass runs on the kept tokens, build a `llama_batch` whose `pos[i]` = original position of token `kept_indices[i]`, **not** sequential. The unified KV cache already supports non-contiguous positions for embedding batches (commit `67eb126d2` opened that path); extend it to plain token batches when the LazyLLM context flag is set.

   - Touches: `src/llama-lazyllm.cpp::prefill_remainder()`, `src/llama-batch.cpp` (extend the embd-batch continuity bypass to `lazyllm_active` token batches).
   - Acceptance: an instrument that dumps RoPE positions from the deep pass shows the original `pos` values, and a smoke run on Llama-2-7B-Chat HotpotQA produces coherent answers (manual eyeball check of 5 prompts).

**A2. Mid-truncation.** Implement `truncate_middle(prompt, n_ctx, max_new)` that keeps the first `(n_ctx - max_new)/2` tokens and the last `(n_ctx - max_new)/2` tokens, dropping the middle. Use it in `examples/lazyllm-run/main.cpp` when `--longbench` is set.

   - Touches: `examples/lazyllm-run/main.cpp`, `tools/lazyllm-eval.py`.
   - Acceptance: regression test loads a 12 KB-token prompt and asserts that both head and tail of the original survive in the truncated prompt.

### Phase B — Coverage (gaps #5, #11)  [2 days]

**B1. Full LongBench loader.** Extend `scripts/get-longbench.py` to all 16 subsets, with the dataset-specific prompt template and `max_new_tokens` taken from `THUDM/LongBench` config. Add `--all-tasks` flag.

**B2. XGen-7B-8K conversion path.** Verify `convert_hf_to_gguf.py` produces a working GGUF; run a 1-prompt smoke decode. Add to `models/README.md` (new file) the exact `huggingface-cli` + `convert_hf_to_gguf.py` invocation.

### Phase C — Eval harness upgrade (gaps #6, #10)  [3–4 days]

**C1. Per-task metric routing.** `scripts/score-longbench.py` (new) dispatches to the correct metric per subset:
  - F1 → `compute_f1` (existing in `lazyllm-eval.py`)
  - Rouge-L → `rouge_score.rouge_scorer.RougeScorer(['rougeL']).score(...)`
  - Cls-Acc / Acc → exact match against canonical class labels
  - Edit-sim → `fuzz.ratio` from `rapidfuzz`

  Output: one CSV per subset, plus an aggregate `longbench_summary.md`.

**C2. Bootstrap CI.** Reuse `bootstrap_ci()` from `tools/lazyllm-eval.py`. Report mean ± 95 % CI per subset.

**C3. Sample size = 200 per subset (paper default).** LongBench v1 ships 200 examples per English subset. Use *all* of them; not the current 20-50.

### Phase D — Multi-GPU performance (gap #3)  [5–7 days, research]

This is the only true research item in this plan. Multi-GPU layer-split blocks the per-stage `decode_partial` calls because every stage barrier flushes the inter-device pipeline. Two approaches to investigate (in order):

**D1. Stage fusion.** Combine consecutive `decode_partial([0,8))`, `decode_partial([8,16))`, `decode_partial([16,24))` into a single CUDA graph with side-output tensors for the score-pool tensors at layers 8, 16, 24. Attention extraction happens once per stage; CPU-side top-k selection still happens inline, but the per-stage `cudaDeviceSynchronize` is amortized.

**D2. Score-on-GPU full ablation.** Even with `lazyllm-pool.h`'s GPU pool, the *selection* (top-k, pool, mask) happens on host. Move it on-device with a single fused kernel; the next-stage graph reads the surviving-token mask directly. This eliminates the host round-trip per stage.

   - Acceptance: 4× L4 layer-split Llama-3.1-8B at 8K tokens shows ≥ 1.4× LazyLLM TTFT vs baseline (currently 1.0× = parity).

### Phase E — Architecture coverage (gaps #7, #8)  [2 days]

**E1. SWA interaction.** For `OPENAI_MOE` (gpt-oss), some layers are sliding-window. The pruning layer choice must be restricted to **full-attention layers**. Add a `validate_pruning_layers_against_arch(arch, layers)` function that returns an error if any pruning layer falls on an SWA layer; document in `include/llama-lazyllm.h`.

**E2. qwen35 / qwen35moe partial builders.** Mirror `src/models/qwen3-partial.cpp`. Skip if the team decides this is out of scope; update `LAZYLLM_REPORT.md` to mark these architectures unsupported.

### Phase F — Determinism, hygiene, CI (gaps #13, #14, #15)  [1–2 days]

**F1. CTest label.** Add `set_tests_properties(... PROPERTIES LABELS "lazyllm")` to all `test-lazyllm-*` and `test-self-layer-prefill*` so `ctest -L lazyllm` runs the suite.

**F2. CI job.** GitHub-actions workflow `.github/workflows/lazyllm-smoke.yml` that builds with `GGML_CUDA=OFF` and runs `ctest -L lazyllm` on a TinyLlama checkpoint (cached as workflow artifact). Wall-clock budget < 10 min.

**F3. Canonical report.** Replace `RESULTS_REPORT.md`, `quality-results/LAZYLLM_COMPREHENSIVE_REPORT.md`, `quality-results-fix2/*.md` with one `LAZYLLM_REPORT.md` that cites raw CSVs by sha + path. Old reports get archived to `bench-results/archive/`.

### Phase G — Stretch (paper-ideal, gap #4)  [research, 5+ days]

**G1. Per-layer position drop in `llama-memory`.** Lets the deep pass *re-use* the KV written by the early pass for layers `[0, l_prune)`, and only writes new KV for layers `[l_prune, n_layer)` on kept tokens. This is the paper's "ideal" path; it eliminates the constant-factor cost of re-decoding the early layers on kept tokens.

   - Treat as optional. Phases A–F deliver the paper's reported numbers without it.

---

## 5. Test plan

All paths use **deterministic settings**: `seed=42`, greedy decode, fixed `n_threads`, `GGML_CUDA_GRAPHS=OFF` for reproducibility.

### 5.1 Unit tests (must all pass on every PR)

Place under `tests/`. CTest label: `lazyllm`.

| Test | File | What it asserts |
|------|------|------------------|
| Identity | `tests/test-lazyllm-identity.cpp` (new) | `keep_ratios = {1.0, 1.0, 1.0}` ⇒ logits **bit-identical** to `llama_decode` (same RNG, FA off). |
| Top-K correctness | extend `tests/test-lazyllm-extract.cpp` | Synthetic scores → indices match a NumPy `np.argsort` reference; last index always present. |
| Pool kernel k=13 | `tests/test-lazyllm-pool.cpp` (new) | 1D average pool against a NumPy reference on a 4 K-element vector; k=1 short-circuits unchanged. |
| Original-position decode | `tests/test-lazyllm-positions.cpp` (new) | After `prefill`, the next decode step reads RoPE phases consistent with the *original* token positions. Inspect `Q · K` of one head against an FP64 reference. |
| Aux cache round-trip | `tests/test-lazyllm-aux-cache.cpp` (new) | `store(pos, h); get(pos) == h` byte-for-byte. Total bytes accounting matches sum of slabs. |
| Embedding-batch continuity bypass | `tests/test-lazyllm-batch.cpp` (new) | A non-contiguous `pos` array with `embd != null` is accepted by `llama_batch`. |
| Auto-fallback decision | `tests/test-lazyllm-fallback.cpp` (new) | Rig the warmup A/B by injecting fake timings; verify `fallback_active` flips at the 1.05× threshold and `prefill()` short-circuits. |
| Determinism | extend `tests/test-lazyllm-extract.cpp` | Same prompt + same params + same seed ⇒ identical kept indices over 5 runs. |
| Decode-stage KV prune | `tests/test-lazyllm-decode.cpp` (new) | `keep_ratio = 1.0` decode ⇒ identical to baseline; `keep_ratio < 1.0` reduces alive count by the expected amount. |
| Revival | `tests/test-lazyllm-revive.cpp` (new) | A revived token's KV at layers `[0, l_prune)` matches what the original prefill wrote (FP64 reference within `1e-3`). |

### 5.2 Cross-implementation parity (gap #1 verification)

Compare C++ output against the Python reference (`tools/lazyllm-ref.py`):

| Test | Reference oracle | Acceptance |
|------|------------------|------------|
| Score parity (Phase 1) | HF `transformers` with `attn_implementation="eager"` and the same scoring math | Pearson(C++ scores, Python scores) ≥ 0.95 over 5 prompts × 3 layers |
| Kept-set parity | Above scores → top-k via the Python `top_k_indices` re-implementation | IoU(C++ kept, Python kept) ≥ 0.95 |
| Logit parity (kr=1.0) | HF `transformers` greedy generation | First 32 generated tokens identical |

Driver: `tests/test-lazyllm-parity.cpp` (new), gated behind `LAZYLLM_TEST_HF=1` env var because it requires HF + torch.

### 5.3 Integration tests

| Test | What it covers |
|------|----------------|
| `tests/test-lazyllm-integration.cpp` (new) | end-to-end `llama_lazyllm_prefill → generate_tokens → output text` on a TinyLlama checkpoint, asserts non-empty output and no NaN logits. |
| `tests/test-lazyllm-archs.cpp` (new) | Runs prefill on one model per supported arch (`LLAMA`, `QWEN2`, `QWEN3`, `OPENAI_MOE`), asserts no error returned. |
| `tests/test-lazyllm-multi-gpu.cpp` (skipped if `<2` GPUs) | Layer-split + row-split mode each return the auto-fallback decision and produce non-NaN logits. |

### 5.4 Stress / regression

| Test | What it covers |
|------|----------------|
| ASan smoke | `tests/test-lazyllm-asan.cpp` — 1000 iterations of prefill+decode under `-fsanitize=address`. |
| Long-context boundary | `n_ctx ∈ {2 K, 4 K, 8 K, 16 K, 32 K}` decode + first-token sample. Asserts no buffer overflow. |
| Schedule edge cases | `keep_ratios = {0.0, 0.5, 1.0}`; `pruning_layers = {0}`, `{n_layer-1}`, single-element. |

---

## 6. Benchmarking plan

### 6.1 Environment lock-in

Every benchmark run must record (and the report must cite):

```
date, git sha, build flags, GPU model/driver, CUDA version,
n_gpu_layers, split-mode (none|layer|row), n_threads, n_batch, n_ubatch,
n_ctx, FA on/off, GGML_CUDA_GRAPHS on/off
```

Build profile (single canonical build):

```bash
cmake -B build -G Ninja \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_GRAPHS=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_NATIVE=OFF
cmake --build build -j --target llama-lazyllm-run llama-bench llama-cli llama-quantize
```

`GGML_CUDA_GRAPHS=OFF` matches the existing benchmarks' fairness baseline (CUDA graph replay artificially accelerates baseline TTFT for repeated identical prompts).

### 6.2 TTFT speedup (the headline claim)

**Models (paper-faithful):**
- `models/llama-2-7b-f16.gguf` (primary)
- `models/llama-2-7b-chat-f16.gguf`
- `models/xgen-7b-8k-base-f16.gguf`
- `models/llama-2-13b-f16.gguf` (Q8_0 fallback if VRAM-bound)

**Configurations (paper Table 1):**
- `pruning_layers = {8, 16, 24}` (= L/4, L/2, 3L/4 for 32-layer)
- `keep_ratios = {0.7, 0.5, 0.3}`  (paper "Standard")
- Ablation: `keep_ratios = {0.5, 0.4, 0.3}` (paper "Aggressive")
- `pool_kernel_size = 13`

**Context-length sweep:** 2 K, 4 K, 8 K, 16 K, 32 K (32 K only on XGen-7B-8K and where memory allows).

**Driver:** `llama-bench --lazyllm --lazyllm-layers 8 16 24 --lazyllm-ratios 0.7 0.5 0.3 -p {ctx} -n 0 -ngl 99 -r 5`. Five reps, drop the first (cold), report median ± stdev of remaining four.

**Output table format (per model):**

```
| n_prompt | baseline t/s | LazyLLM t/s | baseline TTFT ms | LazyLLM TTFT ms | speedup | notes |
```

### 6.3 Quality (LongBench, paper §4.2)

**Driver:** `tools/run_longbench.sh` (new), wraps:

1. `python3 scripts/get-longbench.py --all-tasks --n 200 --out datasets/longbench/`
2. For each model + each subset: run `llama-lazyllm-run` twice (baseline + LazyLLM) on the same 200 prompts with subset-specific `max_new_tokens`, write CSV.
3. `python3 scripts/score-longbench.py --csv-dir results/<run-id>/ --out results/<run-id>/longbench_summary.md`

**Reporting:** for each (model × subset × schedule) — `score_baseline`, `score_lazyllm`, `delta`, `delta_95ci_lo`, `delta_95ci_hi`, `ttft_speedup_median`.

**Aggregate cell (the one number that maps to the paper's quality claim):**

```
Average score across 16 subsets, baseline vs LazyLLM, with 95% bootstrap CI
on the per-prompt delta.
```

The paper reports this aggregate as ≤ 1 % drop on standard schedule. Our equivalent must come within the CI of zero (i.e., not statistically significantly worse).

### 6.4 Scaling sweeps (paper §4.3, ablations)

| Sweep | Values | Held fixed | Goal |
|-------|--------|-----------|------|
| Pruning depth | layers `{8,16,24}` vs `{6,12,18}` vs `{4,8,12}` | model = Llama-2-7B, ctx = 4 K, schedule = 0.7/0.5/0.3 | reproduce paper Fig. 5 |
| Keep ratios | `{0.7,0.5,0.3}`, `{0.5,0.4,0.3}`, `{0.3,0.3,0.3}` | model = Llama-2-7B, ctx = 4 K, layers = {8,16,24} | reproduce paper Table 4 |
| Pool kernel | `k ∈ {1, 7, 13, 21}` | model = Llama-2-7B, ctx = 4 K, schedule = 0.7/0.5/0.3 | reproduce paper Table 5 |
| Aux cache on/off | bool | model = Llama-2-7B, schedule = 0.7/0.5/0.3 | reproduce paper Table 6 |

### 6.5 Multi-GPU scaling

| GPUs | Split mode | Models | Context |
|------|------------|--------|---------|
| 1 | n/a | 7B, 13B-Q8 | 4 K, 8 K |
| 2 | layer | 7B, 13B-FP16 | 4 K, 8 K |
| 2 | row | 7B, 13B-FP16 | 4 K, 8 K |
| 4 | layer | 7B, 13B-FP16 | 4 K, 8 K, 16 K |
| 4 | row | 7B, 13B-FP16 | 4 K, 8 K, 16 K |

For each cell, report TTFT speedup **after** Phase D's fixes. Currently expect parity (1.0×) on layer-split; that's the baseline-to-beat.

---

## 7. Acceptance gates

A run "reproduces the paper" iff **all** of the following pass:

| Gate | Threshold | Source |
|------|-----------|--------|
| **G1 — Identity** | `kr=1.0` ⇒ bit-identical logits to `llama_decode` over 100 prompts | §5.1 |
| **G2 — Score parity** | Pearson(C++ scores, Python scores) ≥ 0.95 over 5 prompts × 3 layers | §5.2 |
| **G3 — Kept-set parity** | IoU(C++ kept, Python kept) ≥ 0.95 | §5.2 |
| **G4 — TTFT speedup, paper config** | Llama-2-7B FP16, ctx = 4 K, schedule = 0.7/0.5/0.3, multi-doc-QA prompt: median speedup ≥ **2.0×** (target 2.34×) on single-GPU | §6.2 |
| **G5 — TTFT speedup, paper config, 13B** | Llama-2-13B Q8_0, ctx = 4 K: median speedup ≥ **1.9×** (target 2.16×) | §6.2 |
| **G6 — Quality, aggregate** | Mean LongBench score (16 subsets, 200 prompts each) drop ≤ 2 percentage points; 95 % bootstrap CI on delta crosses or contains zero | §6.3 |
| **G7 — Quality, multi-doc-QA** | Average F1 (HotpotQA + 2WikiMQA + MuSiQue) drop ≤ 3 pts | §6.3 |
| **G8 — Multi-GPU non-regression** | Auto-fallback engaged ⇒ LazyLLM TTFT within 5 % of baseline on every supported split mode | §6.5 |
| **G9 — Multi-GPU win (Phase D)** | At least one multi-GPU configuration shows ≥ 1.4× speedup | §6.5, §4 D |
| **G10 — CI stability** | `ctest -L lazyllm` 100 % pass over 10 consecutive runs | §5 |

If G4 / G6 / G7 fail: ship as "LazyLLM POC, paper-adjacent" and document the gap. **Do not** claim paper reproduction.

---

## 8. Reproducibility — exact commands

Save as `scripts/lazyllm-paper-reproduction.sh` (new). Skeleton:

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
RUN_ID=${1:-$(date +%Y%m%d-%H%M%S)}
OUT="$REPO/results/$RUN_ID"
mkdir -p "$OUT"

# 1. Build (canonical, FA off, no CUDA graphs)
cmake -B "$REPO/build" -G Ninja \
    -DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=OFF \
    -DCMAKE_BUILD_TYPE=Release -DLLAMA_NATIVE=OFF
cmake --build "$REPO/build" -j --target llama-lazyllm-run llama-bench llama-cli

# 2. Datasets — full LongBench
python3 "$REPO/scripts/get-longbench.py" --all-tasks --n 200 \
    --out "$REPO/datasets/longbench/"

# 3. Models — must already be GGUF-converted (see §1.1)
declare -a MODELS=(
    "models/llama-2-7b-f16.gguf"
    "models/llama-2-7b-chat-f16.gguf"
    "models/xgen-7b-8k-base-f16.gguf"
    "models/llama-2-13b-q8_0.gguf"
)

# 4. TTFT sweep
for M in "${MODELS[@]}"; do
    NAME=$(basename "$M" .gguf)
    "$REPO/build/bin/llama-bench" \
        -m "$REPO/$M" -ngl 99 \
        --lazyllm --lazyllm-layers 8 16 24 --lazyllm-ratios 0.7 0.5 0.3 \
        -p 2048,4096,8192,16384 -n 0 -r 5 \
        -o md > "$OUT/ttft-$NAME.md"
done

# 5. LongBench quality
for M in "${MODELS[@]}"; do
    NAME=$(basename "$M" .gguf)
    for SUBSET in narrativeqa qasper multifieldqa_en \
                  hotpotqa 2wikimqa musique \
                  gov_report qmsum multi_news \
                  trec triviaqa samsum \
                  passage_count passage_retrieval_en \
                  lcc repobench-p; do
        # max_new_tokens table from §1.5 — pre-baked into the script
        MNT=$(python3 "$REPO/scripts/longbench_max_new.py" "$SUBSET")
        "$REPO/build/bin/llama-lazyllm-run" \
            --model "$REPO/$M" \
            --prompts-file "$REPO/datasets/longbench/$SUBSET.jsonl" \
            --pruning-layers 8 16 24 --keep-ratios 0.7 0.5 0.3 \
            --pool-size 13 --n-ctx 4096 --max-tokens "$MNT" \
            --truncation middle --temp 0.0 --seed 42 \
            --n-gpu-layers 99 --repeat 1 \
            --out-csv "$OUT/quality-$NAME-$SUBSET.csv"
    done
done

# 6. Score
python3 "$REPO/scripts/score-longbench.py" \
    --csv-dir "$OUT/" --out "$OUT/longbench_summary.md"

# 7. Render the canonical report
python3 "$REPO/scripts/render-report.py" --run-dir "$OUT" \
    --out "$OUT/LAZYLLM_REPORT.md"
```

Files marked "new" in §4 (notably `scripts/score-longbench.py`, `scripts/longbench_max_new.py`, `scripts/render-report.py`, `--truncation middle` in the binary) must land before this script can run end-to-end. Phase A–C in §4 covers them.

---

## 9. Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Llama-2 license re-distribution blocks CI | Med | High | Use TinyLlama / Qwen 0.6B in CI; reserve Llama-2 for off-CI nightly. |
| FP16 13B doesn't fit on 4× L4 even split | High | Med | Quantize to Q8_0; document the deviation in the report. |
| Mid-truncation changes scoring distribution vs tail-only | Med | Low (paper aligns) | Track both; report both in §6.3. |
| Multi-GPU win (Phase D) requires deep ggml-cuda changes | High | Med | Phase D is its own milestone; ship Phases A–C first. |
| Aux Cache memory grows unbounded for repeated long prompts | Low | Med | Add `aux_cache_max_bytes` param; LRU eviction. |
| HF parquet schema for LongBench changes | Low | Low | Pin to a known commit of `THUDM/LongBench`. |
| Bootstrap CI is over-tight on small subsets (e.g. PassageCount has 200 examples but very high variance) | Med | Low | Report n, mean, CI explicitly per subset; do not aggregate in a misleading way. |

---

## 10. Timeline

Sequential, single-engineer:

| Week | Phases | Exit criteria |
|------|--------|---------------|
| 1 | A1, A2, B1, B2 | Llama-2-7B-Chat HotpotQA 200-prompt run produces coherent outputs; mid-truncation regression test green |
| 2 | C1, C2, C3 | Full LongBench eval runs end-to-end on Llama-2-7B; bootstrap CI rendered |
| 3 | TTFT sweep (§6.2), Quality run (§6.3) on Llama-2-7B FP16, Llama-2-7B-Chat | G4 + G6 + G7 evaluated; first draft of `LAZYLLM_REPORT.md` |
| 4 | XGen-7B-8K + Llama-2-13B sweeps; ablations §6.4 | G5 evaluated; ablation tables filled |
| 5 | F1–F3 (CI, ctest label, canonical report) | `ctest -L lazyllm` green; CI workflow merged |
| 6 | D1 + D2 (multi-GPU work, research-y) | G8 confirmed; G9 attempted |
| 7 (stretch) | G — paper-ideal cross-pass KV reuse | If touched, gated by a separate design review |

---

## 11. Out of scope

- Training any new prediction head (Medusa / EAGLE / learned router).
- Decode-time KV compression beyond the existing `llama_lazyllm_decode_step`.
- Cross-model distillation.
- Models the paper does not evaluate (Llama-3, Llama-3.1, Llama-3.2, gpt-oss-20B, Qwen2/3) — these stay in the supplementary report, not the paper-reproduction claim.
- Energy / power measurements.

---

## 12. Deliverables

When this plan is executed, the branch should produce:

1. `LAZYLLM_REPORT.md` — single canonical results doc, supersedes the existing five.
2. `scripts/lazyllm-paper-reproduction.sh` — one-command repro.
3. `scripts/score-longbench.py`, `scripts/longbench_max_new.py`, `scripts/render-report.py` — eval pipeline.
4. `scripts/get-longbench.py` — extended to all 16 subsets.
5. New tests in §5 — all under CTest label `lazyllm`.
6. `.github/workflows/lazyllm-smoke.yml` — CI job.
7. Updated `include/llama-lazyllm.h` doc comments reflecting Phase A1 (original-position re-decode) once landed.
8. Archived bench artifacts under `bench-results/<run-id>/`.

Then: a clean PR sequence to upstream — small, reviewable, each with a single feature gate.
