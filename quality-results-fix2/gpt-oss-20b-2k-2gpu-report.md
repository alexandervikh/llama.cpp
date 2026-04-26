# LazyLLM Paper-Reproduction Report

Date: 2026-04-26T21:31:45.615092
Input: /home/coder/.cursor/worktrees/llama.cpp__SSH__coder-vscode.coder-sp.compactif.ai--alexandervikhorev--bench.main_/meo/quality-results-fix2/gpt-oss-20b-2k-2gpu.csv
Prompts: 20

## TTFT Speedup

| Metric | Value |
|---|---|
| Baseline median TTFT | 623 ms |
| LazyLLM median TTFT | 842 ms |
| Speedup (median) | 0.740× |
| Gate ≥2.0× | FAIL ❌ |

## Quality (F1)

| Metric | Baseline | LazyLLM | Delta |
|---|---|---|---|
| F1 mean | 0.0263 | 0.0196 | -0.0067 |
| F1 95% CI lo | 0.0085 | 0.0000 | — |
| F1 95% CI hi | 0.0496 | 0.0462 | — |
| Gate CI_lo ≥ baseline-2% | — | FAIL ❌ | — |

## Paper Target (LLaMA-2-7B, multi_doc_qa)

| Metric | Paper | This run | Status |
|---|---|---|---|
| TTFT speedup | 2.34× | 0.74× | ❌ |
| F1 score | ≥22.0 | 2.0 | TBD |
