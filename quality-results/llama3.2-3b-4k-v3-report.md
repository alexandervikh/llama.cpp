# LazyLLM Paper-Reproduction Report

Date: 2026-04-26T02:48:17.355238
Input: quality-results/llama3.2-3b-4k-v3.csv
Prompts: 20

## TTFT Speedup

| Metric | Value |
|---|---|
| Baseline median TTFT | 1741 ms |
| LazyLLM median TTFT | 3956 ms |
| Speedup (median) | 0.439× |
| Gate ≥2.0× | FAIL ❌ |

## Quality (F1)

| Metric | Baseline | LazyLLM | Delta |
|---|---|---|---|
| F1 mean | 0.0259 | 0.0104 | -0.0156 |
| F1 95% CI lo | 0.0114 | 0.0000 | — |
| F1 95% CI hi | 0.0435 | 0.0260 | — |
| Gate CI_lo ≥ baseline-2% | — | FAIL ❌ | — |

## Paper Target (LLaMA-2-7B, multi_doc_qa)

| Metric | Paper | This run | Status |
|---|---|---|---|
| TTFT speedup | 2.34× | 0.44× | ❌ |
| F1 score | ≥22.0 | 1.0 | TBD |
