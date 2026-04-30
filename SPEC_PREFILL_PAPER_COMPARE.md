# Spec-Prefill vs Paper — How to Test and Interpret Gaps

Reference paper: [Speculative Prefill: Turbocharging TTFT…](https://arxiv.org/abs/2502.02789) (arXiv:2502.02789).

This doc ties together [`SPEC_PREFILL_TEST_PLAN.md`](SPEC_PREFILL_TEST_PLAN.md) and the automated runner **`tools/paper_compare_spec_prefill.py`**.

## What the paper reports (headline)

| Claim | Typical setup in paper |
|--------|-------------------------|
| Large TTFT reduction | vLLM, Llama-3.1 **70B / 405B**, low keep %, multi-GPU / TP |
| High end-to-end QPS | Server benchmark, fixed timeout, real query mix |
| Quality | LongBench, RULER, short tasks; often “within a few %” of dense prefill at moderate `kr` |

### Paper headline numbers (TTFT / throughput)

From the paper text and figures (vLLM, **Llama-3.1-70B / 405B**, not our 8B+1B GGUF stack):

| Paper metric | Reported value |
|--------------|----------------|
| TTFT vs dense prefill | **Up to ~7.66×** faster (caption: 405B-Instruct-FP8, **~10% tokens** kept; Fig. 3 / §4.7.2) |
| End-to-end max QPS | **Up to ~7×** (abstract; server setting with real queries) |
| vs MInference (70B) | **~2.54×–6.54×** relative TTFT in one comparison (Fig. 5 text) |

These are **not** row-aligned with llama.cpp timings below (different models, engine, and prompts).

### Example run: our TTFT vs paper reference (same hardware session)

Setup: **Llama-3.1-8B-Instruct** + **Llama-3.2-1B-Instruct** (Q4_K_M), `-ngl 99`, `lah=8`, `pool=13`, `chunk=32`; **`tools/spec-prefill-ttft-bench.py`**, **`--skip-32k --paper`** (includes **kr=0.10** per Fig. 3 caption). Baseline = dense prefill via **`llama-cli`** (`Prompt: … t/s` inference). Importance defaults to **lookahead entropy** (cheap); set **`LLAMA_SPEC_PREFILL_PERPLEXITY=1`** to restore the legacy perplexity path on the spec model for short prompts.

**GPU sample** (`/tmp/ttft_paper_after_entropy_first/ttft_results.json`, CUDA L4-class):

| Context | kr | Baseline (ms) | Spec TTFT (ms) | Our speedup | Paper TTFT speedup (reference)* |
|--------|-----|---------------|----------------|-------------|-----------------------------------|
| 2k | 1.00 | ~451 | ~288 | **~1.57×** | Identity baseline ≈**1×**; paper short-ctx gains smaller (**Fig. 6**). |
| 2k | 0.10 | ~451 | ~219 | **~2.06×** | Fig. 3 cap **~7.66×** @ ~10% tokens (**405B**); same `kr` axis, different stack. |
| 2k | 0.25 | ~451 | ~177 | **~2.54×** | Same family of curves (**Fig. 3**). |
| 8k | 1.00 | ~1694 | ~1229 | **~1.38×** | Dense baseline ≈**1×**. |
| 8k | 0.10 | ~1694 | ~508 | **~3.34×** | Paper peak regime (low keep **%**). |
| 8k | 0.25 | ~1694 | ~486 | **~3.49×** | §4 gate **PASS** (≥**1.5×**). |

**CPU smoke** (Llama-3.2-**1B** self-spec, **`-ngl 0`**, **`--only-2k`**, kr 1.0 / 0.25): spec-prefill TTFT ~**180 ms** vs dense baseline ~**107 ms** (**~0.59×**) — overhead-dominated on CPU; use GPU for paper-like TTFT.

Run **`tools/run_paper_aligned_benchmark.sh`** (GPU paper grid + optional CPU quick; set **`SKIP_CPU_BENCH=1`** to omit CPU).

\*Liu, Chen & Zhang, *Speculative Prefill: Turbocharging TTFT…*, [arXiv:2502.02789](https://arxiv.org/abs/2502.02789). Values cite **abstract**, **Fig. 3–5**, **§4.7.2** — **vLLM**, **70B/405B**, not llama.cpp.

Your **llama.cpp** port is a different engine, often smaller models, and may use a different **importance-scoring cost model** (see `SPEC_PREFILL_LLAMA_ANALYSIS.md`). **Numeric parity with the paper tables is not expected** unless you reproduce **stack, model sizes, and benchmarks**.

## One-command validation (local)

Prerequisites:

- Build `llama-spec-prefill-run` (e.g. `cmake --build build --target llama-spec-prefill-run`).
- GGUF models for the **canonical pair** (plan §“Canonical model pair”):  
  `Llama-3.1-8B-Instruct` + `Llama-3.2-1B-Instruct` (same tokenizer family).

```bash
cd /path/to/llama.cpp
python3 tools/paper_compare_spec_prefill.py --all \
  --build-dir build \
  --base  models/Llama-3.1-8B-Instruct.Q4_K_M.gguf \
  --draft models/Llama-3.2-1B-Instruct.Q4_K_M.gguf \
  --spec-run build/bin/llama-spec-prefill-run \
  -o build/paper_compare
```

Artifacts:

- `build/paper_compare/paper_compare_results.json` — machine-readable summary  
- Per-phase outputs under `build/paper_compare/` (e.g. `longbench/`, `ttft/`)

### Phases (mapped to the test plan)

| Phase | Plan § | What it checks |
|-------|--------|----------------|
| `vllm` | §1b | `import vllm` (optional; needed for true cross-impl IoU vs official stack) |
| `ctest` | §2 | `ctest -R spec-prefill` |
| `determinism` | §1a | Two runs, same `n_kept` / `n_total` |
| `parity` | §1 (mock) | `tools/spec-prefill-parity-mock.sh` — C++ smoke + JSONL prompt |
| `longbench` | §3b | `eval/run_longbench.py` → driver invokes `llama-spec-prefill-run` |
| `score` | §3b | `eval/score_longbench.py` — Rouge-L vs baseline `kr=1.0` |
| `ttft` | §4 | `tools/spec-prefill-ttft-bench.py` — writes `ttft/ttft_results.json` |

Run a subset:

```bash
python3 tools/paper_compare_spec_prefill.py --phase ctest determinism parity --build-dir build --base /path/model.gguf -o out/pc
```

Quick LongBench smoke (few prompts per task):

```bash
python3 tools/paper_compare_spec_prefill.py --phase longbench score \
  --base … --draft … --n-per-task 3 --longbench-synthetic -o out/pc
```

## §4 TTFT gate (project bar, not paper headline)

`spec-prefill-ttft-bench.py` records speedup = baseline `prompt_eval` ms / spec `ttft_ms` and sets `acceptance_gate` using **8k context, kr=0.25**, default **≥ 1.5×** (`--min-speedup`). Use `--strict` on that script only if you want a non-zero exit when the gate fails.

This is **not** the paper’s “7.66×” figure; it is the internal threshold from `SPEC_PREFILL_TEST_PLAN.md` §4.

## If results don’t match the paper

1. **Document deltas**: model IDs, quantization, GPU count, `kr`, lookahead/pool/chunk, llama.cpp commit.
2. **Classify**:
   - **Parity / determinism fails** → implementation bug; fix before trusting benchmarks.
   - **Quality fails** (LongBench / Rouge thresholds) → conservative defaults, task-specific notes.
   - **Speed only** → often **expected** if scoring path differs from paper’s cheap speculator attention; optimize scoring or narrow claims to regimes where you measure wins (e.g. long context).

3. **Shrink the claim** (still publishable): “Filtered prefill with bounded Rouge drop on LongBench subset at kr=0.25 on Llama 8B/1B in llama.cpp,” not “reproduces Liu et al. QPS on 405B.”

## Cross-impl parity (vLLM / IoU)

Full §1b (IoU vs vLLM kept indices) needs a working **vLLM + PyTorch** environment and the official reference repo — see `SPEC_PREFILL_TEST_GUIDE.md`. Until then, treat **`tools/spec-prefill-ref-impl.py`** as **internal** consistency with C++, not proof of match to the shipping vLLM path.

`spec-prefill-parity-mock.sh` validates that **`llama-spec-prefill-run` accepts JSONL prompts** and runs end-to-end. Full C++/Python **n_kept** equality requires extending `spec-prefill-ref-impl.py` to emit the same JSONL line format as the C++ driver (or comparing IoU on indices).

## Related files

- [`SPEC_PREFILL_TEST_PLAN.md`](SPEC_PREFILL_TEST_PLAN.md) — acceptance thresholds  
- [`SPEC_PREFILL_RUNBOOK.md`](SPEC_PREFILL_RUNBOOK.md) — manual step-by-step  
- [`tools/paper_compare_spec_prefill.py`](tools/paper_compare_spec_prefill.py) — orchestrator  
