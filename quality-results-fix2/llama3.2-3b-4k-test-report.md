# LazyLLM Paper-Reproduction Report

Date: 2026-04-26T20:56:18.732772
Input: /home/coder/.cursor/worktrees/llama.cpp__SSH__coder-vscode.coder-sp.compactif.ai--alexandervikhorev--bench.main_/meo/quality-results-fix2/llama3.2-3b-4k-test.csv
Prompts: 3

## TTFT Speedup

| Metric | Value |
|---|---|
| Baseline median TTFT | 914 ms |
| LazyLLM median TTFT | 1113 ms |
| Speedup (median) | 0.821× |
| Gate ≥2.0× | FAIL ❌ |

## Quality (F1)

| Metric | Baseline | LazyLLM | Delta |
|---|---|---|---|
| F1 mean | 0.0142 | 0.0290 | +0.0148 |
| F1 95% CI lo | 0.0000 | 0.0000 | — |
| F1 95% CI hi | 0.0426 | 0.0870 | — |
| Gate CI_lo ≥ baseline-2% | — | PASS ✅ | — |

## Paper Target (LLaMA-2-7B, multi_doc_qa)

| Metric | Paper | This run | Status |
|---|---|---|---|
| TTFT speedup | 2.34× | 0.82× | ❌ |
| F1 score | ≥22.0 | 2.9 | TBD |
