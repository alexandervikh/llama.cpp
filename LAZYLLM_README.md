# LazyLLM Branch README

This branch adds a proof-of-concept implementation of LazyLLM-style dynamic
token pruning for long-context prefill in `llama.cpp`.

LazyLLM is based on [*LazyLLM: Dynamic Token Pruning for Efficient Long Context
LLM Inference*](https://arxiv.org/abs/2407.14057) (Fu et al., Apple, 2024). The
core idea is to run early transformer layers on the full prompt, score tokens by
attention from the final prompt token, progressively drop low-importance tokens,
and finish the remaining layers on a much smaller token set. The target win is
lower TTFT (time to first token) for long prompts without requiring a draft
model.

## What This Branch Adds

- Core LazyLLM runtime in `include/llama-lazyllm.h` and `src/llama-lazyllm.cpp`.
- Partial layer-range execution via `llama_context::decode_partial()`.
- Partial graph builders for Llama, Qwen2, Qwen3, and OpenAI MoE-style models.
- GPU-side attention score pooling that works with both standard attention and
  flash attention graph shapes.
- Progressive multi-stage pruning with original token positions preserved for
  RoPE correctness.
- Optional auto-fallback to baseline prefill when LazyLLM is not faster.
- Optional decode-stage KV pruning and an auxiliary cache API for dropped-token
  hidden states.
- `llama-bench` support for TTFT comparison through `--lazyllm`.
- `llama-lazyllm-run`, a standalone quality and TTFT evaluation harness.
- Python scripts for LongBench, NIAH, RULER, GSM8K, and MMLU-style evaluation.
- LazyLLM-focused C++ and Python tests under `tests/`.

## Implementation Overview

The public API is centered on `llama_lazyllm_context`:

1. Create a context with `llama_lazyllm_init_with_params()`.
2. Configure pruning layers and keep ratios, for example `{8, 16, 24}` and
   `{0.7, 0.5, 0.3}` for a 32-layer model.
3. Optionally call `llama_lazyllm_warmup()` to compile partial graph variants and
   make the auto-fallback decision.
4. Call `llama_lazyllm_prefill()` instead of `llama_decode()` for prompt prefill.
5. Continue normal generation from the logits and KV state left by LazyLLM.

The prefill path runs a sequence of partial graph passes:

- Stage 0 runs layers `[0, L0)` on the full prompt.
- Attention scores are extracted from layer `L0 - 1`.
- Scores are smoothed with a pool kernel, then top-k token positions are kept.
- Hidden states for kept tokens are injected into the next partial pass.
- The process repeats until the final stage, which produces logits for the last
  surviving token.

Kept tokens retain their original positions rather than being re-indexed to
`0..n_kept-1`. This is important for RoPE models because position re-indexing
changes rotary phases and can silently degrade output quality.

## Build

CUDA builds are the main tested path:

```bash
cmake -B build -G Ninja \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_GRAPHS=OFF \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --target llama-bench llama-lazyllm-run
```

`GGML_CUDA_GRAPHS=OFF` is used in the branch reports so baseline and LazyLLM
timings are comparable and not dominated by graph replay effects.

## Benchmark With llama-bench

Use `--lazyllm` to emit an additional LazyLLM row next to the baseline prefill
row:

```bash
./build/bin/llama-bench \
  -m models/llama-3.1-8b-q8_0.gguf \
  -ngl 99 \
  -p 4096 \
  -n 0 \
  -r 5 \
  --lazyllm \
  --lazyllm-layers 8,16,24 \
  --lazyllm-ratios 0.5,0.5,0.5 \
  -o md
```

If `--lazyllm-layers` is omitted, `llama-bench` derives the pruning points from
model depth at roughly one quarter, one half, and three quarters of the layers.

## Run Quality + TTFT Evaluation

`llama-lazyllm-run` runs baseline and LazyLLM on the same prompts and writes a
CSV containing prompt length, kept token count, TTFT, TPOT, speedup, generated
outputs, and reference answers.

```bash
./build/bin/llama-lazyllm-run \
  --model models/llama-3.1-8b-instruct-q8_0.gguf \
  --prompts-file datasets/longbench/hotpotqa.jsonl \
  --n-gpu-layers 99 \
  --n-ctx 4096 \
  --max-tokens 32 \
  --n-prompts 200 \
  --pruning-layers 8 16 24 \
  --keep-ratios 0.5 0.5 0.5 \
  --pool-size 13 \
  --truncation middle \
  --out-csv results/hotpotqa.csv
```

Score LongBench output with:

```bash
python3 scripts/score-longbench.py \
  --csv-dir results/ \
  --model llama-3.1-8b-instruct \
  --out results/longbench_summary.md
```

For a fuller paper-reproduction run:

```bash
./scripts/lazyllm-paper-reproduction.sh \
  --models-dir models \
  --out-dir results/lazyllm-paper
```

The reproduction script expects GGUF versions of Llama-2-7B, Llama-2-7B-Chat,
XGen-7B-8K, and Llama-2-13B as described in `LAZYLLM_PLAN.md`.

## Tests

Build and run the LazyLLM test label:

```bash
cmake --build build --target \
  test-lazyllm-extract \
  test-lazyllm-pool \
  test-lazyllm-topk \
  test-lazyllm-identity-kr1 \
  test-lazyllm-rope-positions \
  test-lazyllm-fallback
ctest --test-dir build -L lazyllm --output-on-failure
```

The branch includes tests for:

- Attention extraction.
- Score pooling.
- Stable top-k selection with last-token preservation.
- Identity behavior when keep ratio is `1.0`.
- RoPE original-position preservation.
- Auto-fallback behavior.
- Python end-to-end smoke coverage.

## Current Results Snapshot

The branch reports the strongest TTFT wins on single-GPU long-prefill runs:

- Llama-3.1-8B Q8_0, 4K context, keep ratio `0.3/0.3/0.3`: about `2.41x` TTFT
  speedup.
- Llama-2-7B Q8_0, 4K context, keep ratio `0.3/0.3/0.3`: about `2.40x` TTFT
  speedup.
- Llama-2-7B Q8_0, 4K context, keep ratio `0.5/0.5/0.5`: about `1.79x` TTFT
  speedup.

Quality results are more nuanced. Instruction-tuned models maintain or improve
HotpotQA F1 in the small reported samples, while base models are more sensitive
to aggressive pruning. Treat the numbers in `LAZYLLM_RESULTS.md` and
`quality-results*/` as branch evidence, not a complete paper reproduction.

## Known Limitations

- The implementation is still marked as proof-of-concept.
- Multi-GPU layer-split can be slower than baseline; the auto-fallback path is
  intended to avoid shipping a worse runtime in those cases.
- Cross-pass KV reuse is not fully implemented, so there is still overhead versus
  an ideal LazyLLM implementation.
- Paper-faithful validation depends on model availability, full LongBench runs,
  and exact model/dataset/hyperparameter matching.
- Some architecture support is still newer and should be validated per model
  before relying on quality results.

## Key Files

- `include/llama-lazyllm.h` - public LazyLLM API and configuration.
- `src/llama-lazyllm.cpp` - pruning, scoring, warmup, fallback, aux cache, and
  decode-pruning implementation.
- `src/models/lazyllm-pool.h` - GPU-side score pooling for non-FA and FA paths.
- `src/models/*-partial.cpp` - partial graph builders for supported families.
- `examples/lazyllm-run/main.cpp` - benchmark and quality harness.
- `tools/llama-bench/llama-bench.cpp` - `--lazyllm` benchmark integration.
- `scripts/lazyllm-paper-reproduction.sh` - end-to-end reproduction driver.
- `LAZYLLM_PLAN.md`, `LAZYLLM_RESULTS.md`, and `LAZYLLM_TEST_PLAN.md` - deeper
  planning, results, and validation notes.
