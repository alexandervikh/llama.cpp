# LazyLLM Paper-Reproduction Report

Date: 2026-04-26T03:08:14.913153
Input: quality-results/llama3.1-8b-4k.csv
Prompts: 14

## TTFT Speedup

| Metric | Value |
|---|---|
| Baseline median TTFT | 3224 ms |
| LazyLLM median TTFT | 5771 ms |
| Speedup (median) | 0.559× |
| Gate ≥2.0× | FAIL ❌ |

## Quality (F1)

| Metric | Baseline | LazyLLM | Delta |
|---|---|---|---|
| F1 mean | 0.0999 | 0.0077 | -0.0922 |
| F1 95% CI lo | 0.0106 | 0.0000 | — |
| F1 95% CI hi | 0.2580 | 0.0192 | — |
| Gate CI_lo ≥ baseline-2% | — | FAIL ❌ | — |

## Paper Target (LLaMA-2-7B, multi_doc_qa)

| Metric | Paper | This run | Status |
|---|---|---|---|
| TTFT speedup | 2.34× | 0.56× | ❌ |
| F1 score | ≥22.0 | 0.8 | TBD |
