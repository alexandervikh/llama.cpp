# LazyLLM Paper-Reproduction Report

Date: 2026-04-26T21:17:46.300854
Input: /home/coder/.cursor/worktrees/llama.cpp__SSH__coder-vscode.coder-sp.compactif.ai--alexandervikhorev--bench.main_/meo/quality-results-fix2/llama2-7b-2k.csv
Prompts: 20

## TTFT Speedup

| Metric | Value |
|---|---|
| Baseline median TTFT | 1022 ms |
| LazyLLM median TTFT | 615 ms |
| Speedup (median) | 1.667× |
| Gate ≥2.0× | FAIL ❌ |

## Quality (F1)

| Metric | Baseline | LazyLLM | Delta |
|---|---|---|---|
| F1 mean | 0.0324 | 0.0422 | +0.0098 |
| F1 95% CI lo | 0.0131 | 0.0200 | — |
| F1 95% CI hi | 0.0560 | 0.0671 | — |
| Gate CI_lo ≥ baseline-2% | — | PASS ✅ | — |

## Paper Target (LLaMA-2-7B, multi_doc_qa)

| Metric | Paper | This run | Status |
|---|---|---|---|
| TTFT speedup | 2.34× | 1.67× | ❌ |
| F1 score | ≥22.0 | 4.2 | TBD |
