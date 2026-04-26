# LazyLLM Paper-Reproduction Report

Date: 2026-04-26T21:11:11.835609
Input: /home/coder/.cursor/worktrees/llama.cpp__SSH__coder-vscode.coder-sp.compactif.ai--alexandervikhorev--bench.main_/meo/quality-results-fix2/llama3.1-8b-4k.csv
Prompts: 20

## TTFT Speedup

| Metric | Value |
|---|---|
| Baseline median TTFT | 1639 ms |
| LazyLLM median TTFT | 1806 ms |
| Speedup (median) | 0.906× |
| Gate ≥2.0× | FAIL ❌ |

## Quality (F1)

| Metric | Baseline | LazyLLM | Delta |
|---|---|---|---|
| F1 mean | 0.0955 | 0.1366 | +0.0410 |
| F1 95% CI lo | 0.0287 | 0.0290 | — |
| F1 95% CI hi | 0.2014 | 0.2827 | — |
| Gate CI_lo ≥ baseline-2% | — | FAIL ❌ | — |

## Paper Target (LLaMA-2-7B, multi_doc_qa)

| Metric | Paper | This run | Status |
|---|---|---|---|
| TTFT speedup | 2.34× | 0.91× | ❌ |
| F1 score | ≥22.0 | 13.7 | TBD |
