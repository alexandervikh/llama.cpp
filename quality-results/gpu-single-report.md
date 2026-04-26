# LazyLLM Paper-Reproduction Report

Date: 2026-04-26T01:50:10.667887
Input: quality-results/gpu-single.csv
Prompts: 5

## TTFT Speedup

| Metric | Value |
|---|---|
| Baseline median TTFT | 13 ms |
| LazyLLM median TTFT | 198 ms |
| Speedup (median) | 0.045× |
| Gate ≥2.0× | FAIL ❌ |

## Quality (F1)

| Metric | Baseline | LazyLLM | Delta |
|---|---|---|---|
| F1 mean | 0.2215 | 0.0364 | -0.1851 |
| F1 95% CI lo | 0.1292 | 0.0000 | — |
| F1 95% CI hi | 0.3152 | 0.1091 | — |
| Gate CI_lo ≥ baseline-2% | — | FAIL ❌ | — |

## Paper Target (LLaMA-2-7B, multi_doc_qa)

| Metric | Paper | This run | Status |
|---|---|---|---|
| TTFT speedup | 2.34× | 0.04× | ❌ |
| F1 score | ≥22.0 | 3.6 | TBD |
