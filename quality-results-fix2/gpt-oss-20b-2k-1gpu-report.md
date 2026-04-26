# LazyLLM Paper-Reproduction Report

Date: 2026-04-26T21:41:11.874041
Input: /home/coder/.cursor/worktrees/llama.cpp__SSH__coder-vscode.coder-sp.compactif.ai--alexandervikhorev--bench.main_/meo/quality-results-fix2/gpt-oss-20b-2k-1gpu.csv
Prompts: 20

## TTFT Speedup

| Metric | Value |
|---|---|
| Baseline median TTFT | 1326 ms |
| LazyLLM median TTFT | 1021 ms |
| Speedup (median) | 1.297× |
| Gate ≥2.0× | FAIL ❌ |

## Quality (F1)

| Metric | Baseline | LazyLLM | Delta |
|---|---|---|---|
| F1 mean | 0.0289 | 0.0304 | +0.0014 |
| F1 95% CI lo | 0.0093 | 0.0101 | — |
| F1 95% CI hi | 0.0542 | 0.0570 | — |
| Gate CI_lo ≥ baseline-2% | — | PASS ✅ | — |

## Paper Target (LLaMA-2-7B, multi_doc_qa)

| Metric | Paper | This run | Status |
|---|---|---|---|
| TTFT speedup | 2.34× | 1.30× | ❌ |
| F1 score | ≥22.0 | 3.0 | TBD |
