# LazyLLM Paper-Reproduction Report

Date: 2026-04-26T03:21:59.260581
Input: quality-results/gpt-oss-20b-4k.csv
Prompts: 10

## TTFT Speedup

| Metric | Value |
|---|---|
| Baseline median TTFT | 2257 ms |
| LazyLLM median TTFT | 14278 ms |
| Speedup (median) | 0.158× |
| Gate ≥2.0× | FAIL ❌ |

## Quality (F1)

| Metric | Baseline | LazyLLM | Delta |
|---|---|---|---|
| F1 mean | 0.0335 | 0.0049 | -0.0286 |
| F1 95% CI lo | 0.0071 | 0.0000 | — |
| F1 95% CI hi | 0.0647 | 0.0146 | — |
| Gate CI_lo ≥ baseline-2% | — | FAIL ❌ | — |

## Paper Target (LLaMA-2-7B, multi_doc_qa)

| Metric | Paper | This run | Status |
|---|---|---|---|
| TTFT speedup | 2.34× | 0.16× | ❌ |
| F1 score | ≥22.0 | 0.5 | TBD |
