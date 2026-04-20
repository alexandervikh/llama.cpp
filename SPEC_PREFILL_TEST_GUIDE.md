# Spec-Prefill Test Guide — How To Execute

Companion to `SPEC_PREFILL_TEST_PLAN.md`. For each section of the plan, this doc spells out **what to build, what to run, what to compare, and what to assert**. All paths are relative to repo root.

Relevant existing artifacts:
- Public API: `include/llama-spec-prefill.h`
- Implementation: `src/llama-spec-prefill.cpp`
- Tests: `tests/test-spec-prefill.cpp`, `tests/test-spec-prefill-bench.cpp`, `tests/test-importance-debug.cpp`
- Reference: `configs/config_p1_full_lah8.yaml` from Jingyu6/speculative_prefill (`look_ahead_cnt=8`, `pool_kernel_size=13`, `chunk=True`, `chunk_size=32`, `percentage=0.1`).

---

## Section 1 — Algorithmic Parity with Reference

**Objective:** prove our C++ pipeline produces the same filtered token set as the Python/vLLM reference for identical inputs.

### 1.1 Pick matched model pair
- Draft: `Llama-3.2-1B-Instruct` (GGUF + HF).
- Base: `Llama-3.1-8B-Instruct` (GGUF + HF) — paper uses 70B; 8B is fine for parity, it's the *algorithm* under test, not the model.
- Quantization: F16 on both sides to eliminate quant-induced divergence.

### 1.2 Build a "trace dump" harness (Python side)
Fork the reference repo locally. Inside `speculative_prefill/`, instrument the pipeline to dump, per request, to JSON:
- Input token IDs + positions.
- Lookahead token IDs.
- Per-token importance scores (pre-pool and post-pool).
- Final kept indices (both `percentage` and `chunk` strategies).

Run the reference on ~50 prompts (mix of LongBench samples, 1k–16k tokens).

### 1.3 Build a matching dump in C++
Add a debug mode to `llama_spec_prefill_context` (gated by env var `LLAMA_SPEC_PREFILL_DUMP=path/`) that writes the same five fields at the same pipeline points in `src/llama-spec-prefill.cpp`.

### 1.4 Compare
Write `scripts/compare_spec_prefill_traces.py` that loads both JSONs and reports, per prompt:
- Lookahead token match rate (sampling determinism permitting — force greedy).
- Importance-score Pearson correlation (pre- and post-pool).
- **IoU of kept indices** — this is the primary parity metric.

**Acceptance:** mean kept-index IoU ≥ 0.95 at `keep_ratio=0.25`, ≥ 0.90 at `keep_ratio=0.10`. Pool correlation ≥ 0.98.

---

## Section 2 — Unit / Component Tests

**Objective:** harden the existing C++ suite against edge cases and CI regressions.

### 2.1 Extend `tests/test-spec-prefill.cpp`
Add these cases (each as a `test_*` function registered in `main()`):
- `test_short_prompt`: prompt length < `n_lookahead` — expect graceful fallback (no filtering, full prompt passes through).
- `test_single_token`: 1-token prompt.
- `test_keep_ratio_one`: `keep_ratio=1.0` — output must equal input byte-for-byte (positions preserved).
- `test_keep_ratio_tiny`: `keep_ratio=0.01` — must keep ≥1 token, no crash.
- `test_eos_in_lookahead`: craft prompt that triggers EOS mid-lookahead — verify `actual_lookahead_cnt` < `n_lookahead` and downstream stages handle it.
- `test_determinism`: run same input twice, `memcmp` on filtered tokens + positions.
- `test_vocab_mismatch`: init with base and draft that have different vocab sizes — expect clean error from `llama_spec_prefill_init`.

### 2.2 Sanitizer build
```
cmake -B build-asan -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"
cmake --build build-asan --target test-spec-prefill -j
./build-asan/bin/test-spec-prefill models/qwen2.5-0.5b-instruct-q4_k_m.gguf
```
**Acceptance:** zero ASan/UBSan reports across the full suite.

### 2.3 Wire into CTest
Add an entry in `tests/CMakeLists.txt` so `ctest -R spec-prefill` runs the suite; gate it on a tiny model that CI can download (~0.5B Qwen).

---

## Section 2.5 — Quality Sanity Gate

**Objective:** cheap go/no-go on "does the filter produce sensible outputs at all?" — runs before Section 1 parity to avoid wasting CUDA time validating a possibly-buggy port.

### 2.5.1 Dataset
- 20 prompts from `THUDM/LongBench` `narrativeqa` split (`test[:20]`).
- Truncate contexts to 2048 tokens (so it fits in 0.5B ctx window).

### 2.5.2 Driver
Use the existing `test-spec-prefill-bench` binary or a tiny shim that calls `llama_spec_prefill()` + greedy decode for 64 tokens. Same-model pair: Qwen 0.5B as both base and spec slots.

### 2.5.3 Conditions
Run each prompt at `keep_ratio ∈ {1.0, 0.5, 0.25, 0.1}`, temperature=0. Save generated text + n_kept to JSONL.

### 2.5.4 Scoring
Compute Rouge-L and token-level F1 against the reference answers using simple Python (rouge-score package).

### 2.5.5 Acceptance
- At `kr=0.25`: average Rouge-L within **30%** of `kr=1.0` baseline.
- At `kr=0.10`: output is still coherent English (manual spot check on 5 samples).
- If either fails → file a bug and pause progression. The filter or position remapping is broken; parity work would only confirm our output matches a C++ bug.

## Section 3 — Quality Evaluation (Downstream Benchmarks)

**Objective:** confirm that aggressive token filtering does *not* tank task metrics on the benchmarks the paper calls out.

### 3.1 Add a `llama-cli`-style driver
Create `examples/spec-prefill-run/` with a binary `llama-spec-prefill-run` that:
- Accepts `--model`, `--spec-model`, `--keep-ratio`, `--lookahead`, `--pool`, `--chunk-size`, `--prompt-file`, `--out`.
- Internally calls `llama_spec_prefill_init_with_params(...)`, runs filtering, feeds filtered tokens into base model via `llama_spec_prefill_process_base`, then decodes a response with the existing sampler.
- Emits JSONL: `{prompt_id, output, n_kept, n_total, ttft_ms, tok_per_sec}`.

### 3.2 LongBench
Use HuggingFace dataset `THUDM/LongBench`. Pick 4–6 representative tasks (e.g. `narrativeqa`, `qasper`, `multifieldqa_en`, `hotpotqa`, `gov_report`, `passage_retrieval_en`).

```bash
python eval/run_longbench.py \
    --driver ./build/bin/llama-spec-prefill-run \
    --model  $BASE_GGUF --spec-model $DRAFT_GGUF \
    --keep-ratio 0.10 0.25 1.00 \
    --out results/longbench/
```
Score with LongBench's official `metrics.py` (F1 / Rouge-L / EM depending on task).

### 3.3 RULER / Needle-in-Haystack
Same driver, plug into the RULER harness at context lengths 4k, 8k, 16k, 32k. Pass-rate is the metric.

### 3.4 Acceptance
- LongBench at `keep_ratio=0.25`: within **2 points** of baseline on average.
- LongBench at `keep_ratio=0.10`: within **5 points** on "compressible" tasks (QA, summarization); larger gap tolerated on retrieval tasks.
- RULER pass-rate: within **5 points** at 8k, degradation at 32k documented (not a hard fail).

---

## Section 4 — Performance Evaluation

**Objective:** verify the whole point — lower TTFT, higher QPS.

### 4.1 Extend `tests/test-spec-prefill-bench.cpp`
It already benchmarks `standard_prefill` vs `spec_prefill`. Add:
- Context-length sweep: 2k, 4k, 8k, 16k, 32k (synthesize with repeated wiki paragraphs).
- Warmup: 3 runs discarded, 10 measured, median reported.
- Output CSV: `ctx_len, keep_ratio, ttft_baseline_ms, ttft_spec_ms, speedup, kept_tokens`.

Run:
```bash
cmake --build build --target test-spec-prefill-bench -j
./build/bin/test-spec-prefill-bench \
    --base  $BASE_GGUF --spec $DRAFT_GGUF \
    --ctx-lens 2048,4096,8192,16384,32768 \
    --keep-ratios 0.1,0.25,0.5 \
    --csv results/perf/ttft.csv
```

### 4.2 QPS / Throughput
Use `tools/server/` with concurrent clients (existing `server-bench.py`). Two server configs: with and without spec-prefill enabled. Push requests at increasing rates; report max sustained QPS at p99 TTFT ≤ SLA.

### 4.3 Overhead breakdown
Add lightweight timers around the four pipeline stages in `src/llama-spec-prefill.cpp` (lookahead, QK extract, attention compute, filter). Emit via a `LLAMA_SPEC_PREFILL_PROFILE=1` env flag. Expected shape: attention-compute dominates on GPU, lookahead dominates on CPU — this explains the commit-noted "2.16× vs 2.42×" regression after adding pooling/chunking.

### 4.4 Builds
- CPU: default cmake.
- CUDA: `-DGGML_CUDA=ON`.
- Metal: `-DGGML_METAL=ON`.

Report a table per backend. **Acceptance:** TTFT speedup ≥ 1.5× at 8k context on GPU with `keep_ratio=0.25`.

---

## Section 5 — Ablations

**Objective:** characterize the quality/speed surface so defaults can be chosen with data.

### 5.1 Grid
Run the LongBench subset from 3.2 across:
- `n_lookahead ∈ {1, 4, 8, 16}`
- `pool_kernel_size ∈ {1, 7, 13}` (1 = effectively off)
- `use_chunking ∈ {false, true}` × `chunk_size ∈ {16, 32, 64}`
- Draft model size ∈ {0.5B Qwen, 1B Llama, 3B Llama}

Drive with a bash matrix calling `llama-spec-prefill-run` from 3.1; collect all results into one dataframe.

### 5.2 Plots
Generate quality-vs-speedup scatter plots (one point per config) and a Pareto front. Tooling: existing `visualization/` scripts in the reference repo, or a small matplotlib notebook under `experiments/`.

**Acceptance:** current defaults (`lah=8, pool=13, chunk=32`) sit on or within 2% of the Pareto front — if not, update defaults in `llama_spec_prefill_params` ctor.

---

## Section 6 — Integration & Regression

**Objective:** make sure shipping this feature doesn't break existing users.

### 6.1 Feature-off identity
Run `llama-perplexity` and `llama-bench` on a fixed dataset (`wikitext-2`) in two trees: pre-`ae83b1def` (before spec-prefill landed) and current HEAD with spec-prefill compiled in but **not invoked**. Numbers must match bit-for-bit.

### 6.2 CI smoke gate
Add a job to `.github/workflows/build.yml`:
- Downloads tiny Qwen 0.5B (base) + a smaller draft.
- Runs `ctest -R spec-prefill`.
- Runs one LongBench-lite task (10 prompts) and checks metric is within threshold.

Total runtime budget: < 5 min.

### 6.3 Sanity baselines
Implement two trivial filters in `tests/test-spec-prefill.cpp`:
- Random token drop at same `keep_ratio`.
- Keep last-N tokens at same `keep_ratio`.

Run them through the same LongBench subset. **Acceptance:** spec-prefill beats both baselines by ≥5 points average — this proves the draft-model signal is doing real work, not just "any compression helps."

---

## Execution Order (suggested)

1. Section 2 (unit tests) — quickest, gets CI green.
2. Section 2.5 (quality sanity gate) — cheap go/no-go before CUDA work.
3. Section 4.1 (perf sweep) — confirms speedup claim on our hardware before deeper work.
4. Section 1 (parity) — hardest; unblocks trust in the port.
5. Section 3 (quality) — requires parity confidence to be interpretable.
6. Section 5 (ablations) — tune defaults.
7. Section 6 (integration) — ship gate.
