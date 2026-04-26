# LazyLLM Paper-Reproduction Report

Date: 2026-04-26T21:11:12.017698
Input: /home/coder/.cursor/worktrees/llama.cpp__SSH__coder-vscode.coder-sp.compactif.ai--alexandervikhorev--bench.main_/meo/quality-results-fix2/llama2-7b-4k.csv
Prompts: 20

## TTFT Speedup

| Metric | Value |
|---|---|
| Baseline median TTFT | 1773 ms |
| LazyLLM median TTFT | 1759 ms |
| Speedup (median) | 1.010× |
| Gate ≥2.0× | FAIL ❌ |

## Quality (F1)

| Metric | Baseline | LazyLLM | Delta |
|---|---|---|---|
| F1 mean | 0.0263 | 0.0362 | +0.0099 |
| F1 95% CI lo | 0.0079 | 0.0178 | — |
| F1 95% CI hi | 0.0479 | 0.0568 | — |
| Gate CI_lo ≥ baseline-2% | — | PASS ✅ | — |

## Paper Target (LLaMA-2-7B, multi_doc_qa)

| Metric | Paper | This run | Status |
|---|---|---|---|
| TTFT speedup | 2.34× | 1.01× | ❌ |
| F1 score | ≥22.0 | 3.6 | TBD |
