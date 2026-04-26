# LazyLLM Paper-Reproduction Report

Date: 2026-04-26T03:32:21.581468
Input: quality-results/llama3.1-70b-2k.csv
Prompts: 10

## TTFT Speedup

| Metric | Value |
|---|---|
| Baseline median TTFT | 6297 ms |
| LazyLLM median TTFT | 6957 ms |
| Speedup (median) | 0.902× |
| Gate ≥2.0× | FAIL ❌ |

## Quality (F1)

| Metric | Baseline | LazyLLM | Delta |
|---|---|---|---|
| F1 mean | 0.0532 | 0.0160 | -0.0372 |
| F1 95% CI lo | 0.0000 | 0.0000 | — |
| F1 95% CI hi | 0.1132 | 0.0480 | — |
| Gate CI_lo ≥ baseline-2% | — | FAIL ❌ | — |

## Paper Target (LLaMA-2-7B, multi_doc_qa)

| Metric | Paper | This run | Status |
|---|---|---|---|
| TTFT speedup | 2.34× | 0.90× | ❌ |
| F1 score | ≥22.0 | 1.6 | TBD |
