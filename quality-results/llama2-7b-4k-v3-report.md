# LazyLLM Paper-Reproduction Report

Date: 2026-04-26T02:54:47.304008
Input: quality-results/llama2-7b-4k-v3.csv
Prompts: 20

## TTFT Speedup

| Metric | Value |
|---|---|
| Baseline median TTFT | 3109 ms |
| LazyLLM median TTFT | 5518 ms |
| Speedup (median) | 0.563× |
| Gate ≥2.0× | FAIL ❌ |

## Quality (F1)

| Metric | Baseline | LazyLLM | Delta |
|---|---|---|---|
| F1 mean | 0.0263 | 0.0114 | -0.0149 |
| F1 95% CI lo | 0.0084 | 0.0000 | — |
| F1 95% CI hi | 0.0475 | 0.0272 | — |
| Gate CI_lo ≥ baseline-2% | — | FAIL ❌ | — |

## Paper Target (LLaMA-2-7B, multi_doc_qa)

| Metric | Paper | This run | Status |
|---|---|---|---|
| TTFT speedup | 2.34× | 0.56× | ❌ |
| F1 score | ≥22.0 | 1.1 | TBD |
