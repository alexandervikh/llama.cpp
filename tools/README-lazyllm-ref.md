# LazyLLM Python reference oracle

This directory holds the Phase 0 reference implementation that the C++ POC
(in `src/llama-lazyllm.cpp`) is validated against. See the top-level
[`RESULTS_REPORT.md`](../RESULTS_REPORT.md) for the benchmark results.

## Install

```bash
pip install transformers torch accelerate scipy numpy
```

(For Llama-2 you must accept the license on HuggingFace and `huggingface-cli login`.)

## Quick smoke (no large download)

```bash
python tools/lazyllm-ref.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompt "What is 2+2?" \
  --max-new-tokens 8
```

## Dump per-layer attention scores

```bash
python tools/lazyllm-ref.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompt-file tests/data/lazyllm-test-prompt.txt \
  --pruning-layers 4 8 12 \
  --keep-ratios 1.0 1.0 1.0 \
  --dump-scores tests/data/lazyllm-oracle-scores.npz
```

Generates an `.npz` consumable by the C++ Phase 1 cross-validation step
(see `tests/test-lazyllm-extract.cpp`).

## Paper-reproduction gate (LongBench multi-doc QA)

LongBench data is not bundled. Download the JSONL files into
`datasets/longbench/` from the official repo:
<https://github.com/THUDM/LongBench>

Then:

```bash
python tools/lazyllm-eval.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --longbench-dir datasets/longbench/ \
  --n-examples 50 \
  --with-baseline
```

The pass-gate per the plan is **F1 ≥ 22.0 AND TTFT speedup ≥ 2.0×**, written
to the `quality-results/` directory.

## Notes

- HF transformers is run with `attn_implementation="eager"` so attention
  weights are exposed (a hard requirement for LazyLLM scoring).
- Pruning preserves **original token positions** for `position_ids` — this is
  critical for RoPE correctness across layers. Re-indexing positions to
  `[0, n_kept)` would silently break the model's positional bias.
- `past_key_values` is **not** modified after pruning; KV cells written by
  earlier (pre-prune) layers persist with their original positions.
- Greedy decoding only (`do_sample=False`).
