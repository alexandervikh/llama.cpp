# LazyLLM Paper-Reproduction Report

Date: 2026-04-26T02:21:23.796921
Input: quality-results/llama2-7b-4k-paper.csv
Prompts: 20

## TTFT Speedup

| Metric | Value |
|---|---|
| Baseline median TTFT | 3095 ms |
| LazyLLM median TTFT | 5564 ms |
| Speedup (median) | 0.558× |
| Gate ≥2.0× | FAIL ❌ |

## Quality (F1)

| Metric | Baseline | LazyLLM | Delta |
|---|---|---|---|
| F1 mean | 0.0000 | 0.0051 | +0.0051 |
| F1 95% CI lo | 0.0000 | 0.0000 | — |
| F1 95% CI hi | 0.0000 | 0.0129 | — |
| Gate CI_lo ≥ baseline-2% | — | PASS ✅ | — |

## Paper Target (LLaMA-2-7B, multi_doc_qa)

| Metric | Paper | This run | Status |
|---|---|---|---|
| TTFT speedup | 2.34× | 0.56× | ❌ |
| F1 score | ≥22.0 | 0.5 | TBD |
