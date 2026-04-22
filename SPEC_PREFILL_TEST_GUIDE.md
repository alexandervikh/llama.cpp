# Spec-Prefill Test Guide — How To Execute

Companion to `SPEC_PREFILL_TEST_PLAN.md`. For each section of the plan, this doc spells out **what to build, what to run, what to compare, and what to assert**. All paths are relative to repo root.

Relevant existing artifacts:
- Public API: `include/llama-spec-prefill.h`
- Implementation: `src/llama-spec-prefill.cpp`
- Tests: `tests/test-spec-prefill.cpp`, `tests/test-spec-prefill-extended.cpp`, `tests/test-spec-prefill-integration.cpp`, `tests/test-spec-prefill-quality.cpp`, `tests/test-spec-prefill-parity.cpp`
- Driver: `examples/spec-prefill-run/main.cpp`
- Reference hyperparams: `configs/config_p1_full_lah8.yaml` from Jingyu6/speculative_prefill (`look_ahead_cnt=8`, `pool_kernel_size=13`, `chunk=True`, `chunk_size=32`, `percentage=0.1`).

## Canonical model pair

Paper repro:
- Base: `meta-llama/Meta-Llama-3.1-8B-Instruct`
- Draft: `meta-llama/Llama-3.2-1B-Instruct`

Fetch once:
```bash
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct --local-dir models/llama-3.1-8b
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download meta-llama/Llama-3.2-1B-Instruct   --local-dir models/llama-3.2-1b
python convert_hf_to_gguf.py models/llama-3.1-8b --outtype q4_k_m  --outfile models/llama-3.1-8b-instruct-q4_k_m.gguf
python convert_hf_to_gguf.py models/llama-3.2-1b --outtype q4_k_m  --outfile models/llama-3.2-1b-instruct-q4_k_m.gguf
# F16 builds for parity / ablation work:
python convert_hf_to_gguf.py models/llama-3.1-8b --outtype f16     --outfile models/llama-3.1-8b-instruct-f16.gguf
python convert_hf_to_gguf.py models/llama-3.2-1b --outtype f16     --outfile models/llama-3.2-1b-instruct-f16.gguf
```

Export shell vars used throughout this guide:
```bash
export BASE_GGUF=models/llama-3.1-8b-instruct-q4_k_m.gguf
export DRAFT_GGUF=models/llama-3.2-1b-instruct-q4_k_m.gguf
export BASE_F16=models/llama-3.1-8b-instruct-f16.gguf
export DRAFT_F16=models/llama-3.2-1b-instruct-f16.gguf
```

---

## Section 1 — Algorithmic Parity with Reference

**Objective:** prove our C++ pipeline is (1a) internally deterministic and (1b) produces the same filtered token set as the vLLM reference for identical inputs.

### 1.1 Canonical pair + quantization choice
- Use the canonical pair above.
- Parity work: F16 on both sides to eliminate quant-induced divergence.
- Smoke / throughput work: Q4_K_M.

### 1.2 Self-determinism (§1a — required)
Run the existing parity test:
```bash
./build/bin/test-spec-prefill-parity $BASE_F16 $DRAFT_F16
```
**Acceptance:** 100% kept-index match across two independent runs per prompt on a 20-prompt suite at `kr ∈ {0.10, 0.25, 1.0}`.

### 1.3 Cross-impl parity vs vLLM (§1b — stretch)

**Known blocker (as of 2026-04-22):** vLLM 0.8.5 pins `torch==2.6.0`, `torchvision==0.21.0`, `xformers==0.0.29.post2`. The ambient env has incompatible versions; attempts to install into the main venv fail. Workaround: clean conda env (see `speculative_prefill/README.md`). If the env cannot be stood up, §1b is **deferred** — do not mark the parity section complete.

Fallback note: `tools/spec-prefill-ref-impl.py` mirrors our own C++ algorithm in Python. It is useful for self-consistency debugging, but **it is not a vLLM reference** and does not validate cross-impl parity.

#### 1.3.1 Trace dump (vLLM side)
Fork Jingyu6/speculative_prefill. In `speculative_prefill/vllm_patch/worker/look_ahead_spec_worker.py`, instrument `_forward_with_query_dump` to write per-request JSON containing:
- Input token IDs + positions.
- Lookahead token IDs (force greedy, temp=0).
- Pre-pool and post-pool importance scores per layer.
- Final kept indices (both `percentage` and `chunk`).

Run on ~50 prompts (1k–16k tokens, LongBench mix).

#### 1.3.2 Trace dump (C++ side)
`include/llama-spec-prefill.h` already exposes `set_dump_path(const char *)`. Invoke:
```bash
LLAMA_SPEC_PREFILL_DUMP=/tmp/cpp_traces \
    ./build/bin/llama-spec-prefill-run \
        --model $BASE_F16 --spec-model $DRAFT_F16 \
        --prompt-file eval/parity_prompts.jsonl \
        --keep-ratio 0.25 0.10
```

#### 1.3.3 Compare
```bash
python scripts/compare_spec_prefill_traces.py \
    --vllm /tmp/vllm_traces --cpp /tmp/cpp_traces \
    --out  results/parity/
```
Report: lookahead match rate, importance-score Pearson (pre and post pool), **IoU of kept indices**.

**Acceptance:** mean IoU ≥ 0.95 at `kr=0.25`, ≥ 0.90 at `kr=0.10`. Pool correlation ≥ 0.98.

---

## Section 2 — Unit / Component Tests

**Objective:** harden the C++ suite against edge cases, regressions, and memory errors.

### 2.1 Suite coverage

`tests/test-spec-prefill.cpp` (core, 8 tests) and `tests/test-spec-prefill-extended.cpp` (15 tests) already cover most edges. Regressions to keep enforced:

- `test_short_prompt` / `test_single_token` / `test_eos_in_lookahead` — graceful fallback.
- `test_keep_ratio_identity` (`kr=1.0`) — byte-identical to input.
- `test_keep_ratio_zero` / `test_keep_ratio_tiny` — ≥ 1 token kept, no crash.
- `test_determinism` — `memcmp` filtered tokens + positions across two runs.
- `test_tokenizer_mismatch` — init with mismatched base/draft vocab fails cleanly.
- `test_q_tensor_extraction` (add if missing) — exercises `is_q_tensor_name()`, the `llama_spec_q_tensor` struct, and `get_gf_res_prev()` path added in commit `68c8becf2`. Assert that per-layer Q comes from the expected graph result and has shape `[n_embd_head, n_head, n_tokens]`.
- `test_lookahead_init_token_zero` — regression guard: lookahead must start from `token=0`, not `prompt_tokens[n_prompt-1]`. The latter caused the original quality failure that `68c8becf2` fixed.

### 2.2 Sanitizer build

**STATUS (as of 2026-04-22): not yet executed in CI.** Build exists locally on demand; add a `ctest -L asan` target so it runs on PRs.

```bash
cmake -B build-asan -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" \
      -DCMAKE_C_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"
cmake --build build-asan --target test-spec-prefill test-spec-prefill-extended test-spec-prefill-integration -j
./build-asan/bin/test-spec-prefill           $DRAFT_GGUF
./build-asan/bin/test-spec-prefill-extended  $DRAFT_GGUF
./build-asan/bin/test-spec-prefill-integration $BASE_GGUF $DRAFT_GGUF
```
**Acceptance:** zero ASan/UBSan reports across all three binaries.

### 2.3 CTest wiring

`tests/CMakeLists.txt` already registers the suites (commit `68c8becf2`). For CI, add a labelled target:
```cmake
set_tests_properties(test-spec-prefill               PROPERTIES LABELS "spec-prefill")
set_tests_properties(test-spec-prefill-extended      PROPERTIES LABELS "spec-prefill")
set_tests_properties(test-spec-prefill-integration   PROPERTIES LABELS "spec-prefill")
set_tests_properties(test-spec-prefill-parity        PROPERTIES LABELS "spec-prefill")
```
Run with `ctest -L spec-prefill --output-on-failure`.

---

## Section 2.5 — Quality Sanity Gate

**Objective:** cheap go/no-go on "does the filter produce sensible outputs at all?" — runs before §1b and §3b.

### 2.5.1 Dataset
- 20 prompts from `THUDM/LongBench` `narrativeqa` split, truncated to 2048 tokens (fits draft context cleanly).
- File: `tests/prompts_quality_gate.jsonl` (already in-tree).

### 2.5.2 Driver
Use the shipped `test-spec-prefill-quality` binary (single model load, batch memory fix landed in `68c8becf2`):
```bash
./build/bin/test-spec-prefill-quality \
    --model $DRAFT_GGUF --spec $DRAFT_GGUF \
    --prompts tests/prompts_quality_gate.jsonl \
    --keep-ratios 1.0,0.5,0.25,0.1 \
    --out results/smoke/
```
**Same-model self-spec on `Llama-3.2-1B-Instruct`** — we are testing the *filter*, not speedup. This removes tokenizer-mismatch risk and runs on any GPU.

### 2.5.3 Scoring
```bash
python eval/score_rouge.py --in results/smoke/ --out results/smoke/score.json
```

### 2.5.4 Acceptance
- `Rouge-L(kr=0.25) / Rouge-L(kr=1.0) ≥ 0.70` (the threshold actually asserted by `test-spec-prefill-quality`). Historical measurement: 0.857.
- Output at `kr=0.10` is coherent English on 5 manual spot-checks.
- If either fails → pause; do not proceed to §1b / §3b.

---

## Section 3 — Quality Evaluation (Downstream Benchmarks)

**Objective:** confirm aggressive token filtering preserves task metrics on the benchmarks the paper calls out.

### 3.1 Driver

`examples/spec-prefill-run/main.cpp` → `llama-spec-prefill-run`:
```bash
cmake --build build --target llama-spec-prefill-run -j
./build/bin/llama-spec-prefill-run \
    --model $BASE_GGUF --spec-model $DRAFT_GGUF \
    --keep-ratio 0.25 --lookahead 8 --pool 13 --chunk-size 32 \
    --prompt-file some.jsonl --out out.jsonl
```
Emits JSONL: `{prompt_id, output, n_kept, n_total, ttft_ms, tok_per_sec}`.

### 3.2 Smoke proxy (§3a — done)

Shipped in `eval/quality_prompts.jsonl` (20 hand-crafted prompts). Score with `eval/score_rouge.py`. This is a **smoke test**, not LongBench. Do not report it as §3b results.

### 3.3 Real LongBench (§3b — required for release claim)

`eval/run_longbench.py` currently loads a local JSONL. For the real repro it must load HF:

```python
from datasets import load_dataset
tasks = ["narrativeqa", "qasper", "multifieldqa_en",
         "hotpotqa", "gov_report", "passage_retrieval_en"]
for t in tasks:
    ds = load_dataset("THUDM/LongBench", t, split="test")
    # feed ds into the driver, one JSONL per task
```

Run:
```bash
python eval/run_longbench.py \
    --driver ./build/bin/llama-spec-prefill-run \
    --model $BASE_GGUF --spec-model $DRAFT_GGUF \
    --tasks narrativeqa qasper multifieldqa_en hotpotqa gov_report passage_retrieval_en \
    --keep-ratio 0.10 0.25 1.00 \
    --out results/longbench/
```
Score with LongBench's official `metrics.py` (F1 / Rouge-L / EM per task).

### 3.4 RULER / Needle-in-Haystack

Plug the same driver into the RULER harness at ctx ∈ {4k, 8k, 16k, 32k}. Pass-rate is the metric.

### 3.5 Acceptance
- LongBench `kr=0.25`: within **2 pts** of baseline on average.
- LongBench `kr=0.10`: within **5 pts** on compressible tasks; larger gap on retrieval documented.
- RULER pass-rate: within **5 pts** at 8k; degradation at 32k documented (not hard fail).

---

## Section 4 — Performance Evaluation

**Objective:** verify the whole point — lower TTFT, higher QPS.

### 4.1 TTFT sweep
```bash
./build/bin/test-spec-prefill-bench \
    --base $BASE_GGUF --spec $DRAFT_GGUF \
    --ctx-lens 2048,4096,8192,16384,32768 \
    --keep-ratios 0.1,0.25,0.5 \
    --csv results/perf/ttft.csv
```
Warmup 3 runs discarded, 10 measured, median reported. CSV columns: `ctx_len, keep_ratio, ttft_baseline_ms, ttft_spec_ms, speedup, kept_tokens`.

### 4.2 QPS / Throughput
`tools/server/` + `tools/spec-prefill-qps-bench.py` with concurrent clients (1, 2, 4, 8). Historical numbers (Qwen3-0.6B on 4×L4): peak 0.88 QPS at concurrency 4. Report against the canonical pair.

### 4.3 Overhead breakdown
`LLAMA_SPEC_PREFILL_PROFILE=1` emits timers around lookahead / QK extract / attention compute / filter. Expected shape: attention-compute dominates on GPU, lookahead on CPU.

### 4.4 Builds

| Backend | Status | Notes |
|---|---|---|
| CUDA (`-DGGML_CUDA=ON`) | **Reference build** | All long-context numbers come from here. |
| CPU (default cmake) | **Limited** | Hangs on prompts > ~4k tokens (separate bug, not spec-prefill). Report pp512 / pp2k only; mark pp8k / pp16k / pp32k as `N/A (CPU build bug)`. |
| Metal (`-DGGML_METAL=ON`) | **Deferred** | Run if an M-series host becomes available. |

**Acceptance:** TTFT speedup ≥ 1.5× at 8k ctx on GPU with `kr=0.25` vs baseline prefill.

### 4.5 Known bug patterns (regression anchors)

Document these so the next person knows what to look for — all discovered during the 2026-04 test campaign:

1. **Flash Attention tensor naming** — `extract_qk()` must match both `q` and `k` naming variants used by FA. Add a unit assertion.
2. **GPU tensor data access** — always use `ggml_backend_tensor_get()` for CUDA reads; direct pointer access silently returns junk.
3. **F32 type handling** — direct `memcpy` for F32 tensors (no dtype conversion).
4. **JSONL prompt parser** — handle trailing newlines + empty lines; silent truncation caused a quality regression.
5. **`keep_ratio` chunk rounding** — chunk-based filtering with small `n_chunks` previously kept 1 token instead of `ceil(n_prompt * keep_ratio)`. Fallback now engages; keep the fallback covered by a test.
6. **Lookahead ordering** — `generate_lookahead()` must be called **before** filtering; reversed order produced coherent-looking but wrong kept sets.

---

## Section 5 — Ablations

**Objective:** characterize the quality/speed surface so defaults can be chosen with data.

### 5.1 Grid
Run the §3b LongBench subset across:
- `n_lookahead ∈ {1, 4, 8, 16}`
- `pool_kernel_size ∈ {1, 7, 13}` (1 = effectively off)
- `use_chunking ∈ {false, true}` × `chunk_size ∈ {16, 32, 64}`
- **Draft-model size (matched tokenizer only)**: `Llama-3.2-1B-Instruct` vs `Llama-3.2-3B-Instruct`.

**Do not** mix tokenizer families — a Qwen draft on a Llama base fails the vocab check by design (see §8).

Drive with a bash matrix calling `llama-spec-prefill-run`; collect into one dataframe:
```bash
bash eval/run_ablations.sh --base $BASE_GGUF --draft $DRAFT_GGUF --out results/ablations/
```

### 5.2 Plots
Quality-vs-speedup scatter + Pareto front via `tools/spec-prefill-ablations.py` (or a small matplotlib notebook under `experiments/`).

**Acceptance:** current defaults (`lah=8, pool=13, chunk=32`) sit on or within 2% of the Pareto front; otherwise update `llama_spec_prefill_params` defaults.

---

## Section 6 — Integration & Regression

**Objective:** ensure shipping does not break existing users.

### 6.1 Feature-off identity
```bash
./build/bin/llama-perplexity -m $BASE_GGUF -f wikitext-2-raw/wiki.test.raw > pre.txt
./build/bin/llama-bench      -m $BASE_GGUF > bench-pre.txt
# compare against a pre-spec-prefill HEAD
```
Numbers must match bit-for-bit when spec-prefill is compiled in but not invoked.

### 6.2 CI smoke gate
Add to `.github/workflows/build.yml`:
- Download `Llama-3.2-1B-Instruct` (Q4 GGUF, ~0.7 GB).
- `ctest -L spec-prefill --output-on-failure`.
- Run one LongBench-lite task (10 prompts) via `llama-spec-prefill-run` and assert Rouge-L within threshold.
- Budget: < 5 min total.

### 6.3 Sanity baselines
`tests/test-spec-prefill-integration.cpp` already implements random-drop and last-N-window filters. Acceptance: spec-prefill beats both by ≥ 5 pts Rouge-L average. Historical gap vs random-like: **0.77** — well above threshold.

---

## Section 7 — Release Hygiene

Pre-merge checklist (run manually before opening the PR):

```bash
# 1. No core dumps tracked or ignored-but-huge
ls core.* 2>/dev/null && echo "FAIL: core dumps present" || echo "ok"

# 2. Clean tree except intended edits
git status --porcelain

# 3. All referenced scripts committed
git ls-files tools/spec-prefill-*.py

# 4. Fresh-clone ctest passes
rm -rf /tmp/fresh-clone && git clone . /tmp/fresh-clone && cd /tmp/fresh-clone
cmake -B build -DGGML_CUDA=ON && cmake --build build --target ctest-deps-spec-prefill -j
ctest --test-dir build -L spec-prefill --output-on-failure

# 5. Docs consistency: no section claimed PASS that is not numerically pass
grep -E "PASS|FAIL|PARTIAL" SPEC_PREFILL_FINAL_REPORT.md VALIDATION_SUMMARY.md TEST_RESULTS.md
```

All five must succeed.

---

## Section 8 — Extension Targets (not paper repro)

Use only after §3b is green on the canonical pair. These are research extensions with their own acceptance.

### 8.1 Qwen family (safe extension)
- `Qwen2.5-7B-Instruct` + `Qwen2.5-0.5B-Instruct`
- `Qwen3-8B` + `Qwen3-0.6B`

Both dense, same tokenizer within family, analogous to the paper. Same §3b thresholds apply.

### 8.2 gpt-oss (MoE + SWA — hazardous)

Only matched-tokenizer pair available (`o200k_harmony`): `gpt-oss-120B` base + `gpt-oss-20B` draft. No smaller draft exists — cross-family drafts (Qwen, Llama) fail the vocab check and must not be attempted.

**Plumbing test first (self-spec):**
```bash
./build/bin/llama-spec-prefill-run \
    --model gpt-oss-20b.gguf --spec-model gpt-oss-20b.gguf \
    --prompt-file tests/prompts_harmony.jsonl \
    --keep-ratio 0.25
```

**Architectural instrumentation required before trusting any numbers:**

1. **Sliding-window attention** — gpt-oss alternates full and SWA layers. Extend the per-layer dump to record `attn_type` and inspect which layer's Q is being used for importance. Two mitigations:
   - Restrict Q extraction to full-attn layers only.
   - Normalize per-layer scores before aggregation so SWA layers don't systematically under-weight early tokens.
2. **MoE** — attention runs per-token before expert routing, so Q/K extraction is unaffected. However, the paper's transferability claim is unvalidated on MoE — expect wider quality variance.
3. **Harmony template** — prompts must be wrapped with `<|start|>…<|message|>…<|return|>` etc. See `models/templates/openai-gpt-oss-120b.jinja`.

**Acceptance (loosened):** TTFT measured and reported; quality gap tolerance at `kr=0.25` loosened to **5 pts** (vs 2 pts for Llama). Retrieval-task degradation documented but not gated.

---

## Execution Order (suggested)

1. §2 (unit + Q-tensor + lookahead-init regression) — get CI green.
2. §2.5 (quality sanity gate, canonical pair, self-spec) — cheap go/no-go.
3. §4.1 (TTFT sweep on canonical pair, GPU) — confirm speedup on our hardware.
4. §1.2 (self-determinism) — falls out of §2 runs.
5. §3b (real LongBench + RULER on canonical pair) — paper repro, goes in the report.
6. §5 (ablations) — tune defaults.
7. §1.3 (cross-impl vs vLLM) — only if the vLLM env can be stood up; otherwise deferred.
8. §6 (integration + sanity baselines) — ship gate.
9. §7 (release hygiene) — pre-merge.
10. §8 (Qwen / gpt-oss extensions) — research follow-ups, separate report.

If §1.3's vLLM env cannot be prepared, do **not** treat §3b numbers as "parity-validated" — they are behavioral, not cross-impl parity.
