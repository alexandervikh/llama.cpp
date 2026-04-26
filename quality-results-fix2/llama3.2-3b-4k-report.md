# LazyLLM Paper-Reproduction Report

Date: 2026-04-26T20:56:00.535297
Input: /home/coder/.cursor/worktrees/llama.cpp__SSH__coder-vscode.coder-sp.compactif.ai--alexandervikhorev--bench.main_/meo/quality-results-fix2/llama3.2-3b-4k.csv
Prompts: 20

## TTFT Speedup

| Metric | Value |
|---|---|
| Baseline median TTFT | 1748 ms |
| LazyLLM median TTFT | 1087 ms |
| Speedup (median) | 1.608× |
| Gate ≥2.0× | FAIL ❌ |

## Quality (F1)

| Metric | Baseline | LazyLLM | Delta |
|---|---|---|---|
| F1 mean | 0.0259 | 0.0405 | +0.0145 |
| F1 95% CI lo | 0.0116 | 0.0193 | — |
| F1 95% CI hi | 0.0440 | 0.0632 | — |
| Gate CI_lo ≥ baseline-2% | — | PASS ✅ | — |

## Paper Target (LLaMA-2-7B, multi_doc_qa)

| Metric | Paper | This run | Status |
|---|---|---|---|
| TTFT speedup | 2.34× | 1.61× | ❌ |
| F1 score | ≥22.0 | 4.0 | TBD |
