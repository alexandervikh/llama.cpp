# LazyLLM Paper-Reproduction Report

Date: 2026-04-26T02:21:23.859175
Input: quality-results/llama3.2-3b-4k.csv
Prompts: 20

## TTFT Speedup

| Metric | Value |
|---|---|
| Baseline median TTFT | 1740 ms |
| LazyLLM median TTFT | 3970 ms |
| Speedup (median) | 0.439× |
| Gate ≥2.0× | FAIL ❌ |

## Quality (F1)

| Metric | Baseline | LazyLLM | Delta |
|---|---|---|---|
| F1 mean | 0.0028 | 0.0119 | +0.0091 |
| F1 95% CI lo | 0.0000 | 0.0000 | — |
| F1 95% CI hi | 0.0083 | 0.0262 | — |
| Gate CI_lo ≥ baseline-2% | — | PASS ✅ | — |

## Paper Target (LLaMA-2-7B, multi_doc_qa)

| Metric | Paper | This run | Status |
|---|---|---|---|
| TTFT speedup | 2.34× | 0.44× | ❌ |
| F1 score | ≥22.0 | 1.2 | TBD |
