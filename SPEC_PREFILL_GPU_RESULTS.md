# Speculative-prefill GPU benchmark — Llama-3.1-70B + 8B draft

_Report generated: 2026-04-30 09:29 UTC_

## Setup

- **Main model:** `Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf` (llama 70B Q4_K - Medium, 39.59 GiB)
- **Draft model:** `llama-3.1-8b-instruct-q4_k_m.gguf`
- **Hardware:** NVIDIA L4, NVIDIA L4, NVIDIA L4, NVIDIA L4
- **CPU:** AMD EPYC 7R13 Processor
- **Backend:** CUDA, `n_gpu_layers=99`, `flash_attn=True`
- **llama-bench flags:** `--spec-chunk 1` (exact `kr` realization), `-r 5`, `-n 0` (prefill only), `--spec-lah 8`
- **Build:** `c7af45ae4` (#235)

## Methodology

- `kr` = requested speculative keep-ratio (`--spec-kr`). With `--spec-chunk 1` the realized `kept` count is exact: `kept = round(kr × n_prompt)`.
- `t/s` = prefill throughput (tokens / second), mean ± stddev over 5 reps.
- `ours Sx` = measured TTFT speedup (`meas_ttft_Sx` from llama-bench, the binary's native field). Computed as `dense_TTFT / spec_TTFT`.
- `dense (t/s ÷ Sx)` = implicit dense-prefill baseline = `spec_t/s @ kr=1` ÷ `Sx @ kr=1`. This recovers what the model would do without spec-prefill overhead.
- `Sx(paper)` = corresponding speedup reported in arXiv:2502.02789 for the closest comparable setting (see footnotes per row).
- Rows ordered with **`kr` descending** (1.0 → 0.1).
- Context labels follow llama-bench convention (`ppN` = prefill of N tokens).

## Per-context results

### pp128

| kind | kr | kept | t/s | ours Sx | Sx(paper) |
|---|---:|---:|---:|---:|---:|
| dense (t/s ÷ Sx) | — | — | **323.88** | 1.00 | — |
| spec | 1.0 | 128 | 210.52 ± 1.44 | 0.65 ± 0.01 | 1.00 |
| spec | 0.5 | 64 | 271.91 ± 1.03 | 0.84 ± 0.01 | — |
| spec | 0.25 | 32 | 289.53 ± 0.42 | 0.90 ± 0.00 | 2.54–6.54 |
| spec | 0.1 | 13 | 289.77 ± 0.41 | 0.90 ± 0.00 | ≤7.66 |

### pp256

| kind | kr | kept | t/s | ours Sx | Sx(paper) |
|---|---:|---:|---:|---:|---:|
| dense (t/s ÷ Sx) | — | — | **326.52** | 1.00 | — |
| spec | 1.0 | 256 | 248.15 ± 0.21 | 0.76 ± 0.00 | 1.00 |
| spec | 0.5 | 128 | 398.03 ± 2.95 | 1.23 ± 0.02 | — |
| spec | 0.25 | 64 | 509.46 ± 0.45 | 1.57 ± 0.01 | 2.54–6.54 |
| spec | 0.1 | 26 | 543.90 ± 0.52 | 1.68 ± 0.01 | ≤7.66 |

### pp512

| kind | kr | kept | t/s | ours Sx | Sx(paper) |
|---|---:|---:|---:|---:|---:|
| dense (t/s ÷ Sx) | — | — | **324.79** | 1.00 | — |
| spec | 1.0 | 512 | 269.57 ± 0.25 | 0.83 ± 0.00 | 1.00 |
| spec | 0.5 | 256 | 461.29 ± 0.86 | 1.43 ± 0.01 | — |
| spec | 0.25 | 128 | 710.24 ± 1.07 | 2.21 ± 0.00 | 2.54–6.54 |
| spec | 0.1 | 52 | 897.28 ± 1.03 | 2.79 ± 0.01 | ≤7.66 |

### pp1024

| kind | kr | kept | t/s | ours Sx | Sx(paper) |
|---|---:|---:|---:|---:|---:|
| dense (t/s ÷ Sx) | — | — | **361.23** | 1.00 | — |
| spec | 1.0 | 1024 | 314.27 ± 0.27 | 0.87 ± 0.01 | 1.00 |
| spec | 0.5 | 512 | 503.18 ± 0.80 | 1.40 ± 0.00 | — |
| spec | 0.25 | 256 | 827.35 ± 3.27 | 2.30 ± 0.02 | 2.54–6.54 |
| spec | 0.1 | 103 | 1268.86 ± 1.44 | 3.54 ± 0.01 | ≤7.66 |

### pp2048

| kind | kr | kept | t/s | ours Sx | Sx(paper) |
|---|---:|---:|---:|---:|---:|
| dense (t/s ÷ Sx) | — | — | **371.93** | 1.00 | — |
| spec | 1.0 | 2048 | 331.02 ± 0.55 | 0.89 ± 0.00 | 1.00 |
| spec | 0.5 | 1024 | 566.24 ± 0.77 | 1.53 ± 0.00 | — |
| spec | 0.25 | 512 | 875.23 ± 1.93 | 2.36 ± 0.00 | 2.54–6.54 |
| spec | 0.1 | 205 | 1426.88 ± 4.12 | 3.84 ± 0.01 | ≤7.66 |

### pp4096

| kind | kr | kept | t/s | ours Sx | Sx(paper) |
|---|---:|---:|---:|---:|---:|
| dense (t/s ÷ Sx) | — | — | **386.21** | 1.00 | — |
| spec | 1.0 | 4096 | 347.59 ± 1.30 | 0.90 ± 0.00 | 1.00 |
| spec | 0.5 | 2048 | 615.04 ± 1.66 | 1.61 ± 0.00 | — |
| spec | 0.25 | 1024 | 994.41 ± 0.83 | 2.62 ± 0.00 | 2.54–6.54 |
| spec | 0.1 | 410 | 1522.75 ± 4.19 | 4.02 ± 0.01 | ≤7.66 |

### pp8192

| kind | kr | kept | t/s | ours Sx | Sx(paper) |
|---|---:|---:|---:|---:|---:|
| dense (t/s ÷ Sx) | — | — | **371.79** | 1.00 | — |
| spec | 1.0 | 8192 | 334.61 ± 0.91 | 0.90 ± 0.00 | 1.00 |
| spec | 0.5 | 4096 | 618.40 ± 4.04 | 1.67 ± 0.01 | — |
| spec | 0.25 | 2048 | 1038.66 ± 4.39 | 2.81 ± 0.02 | 2.54–6.54 |
| spec | 0.1 | 820 | 1721.72 ± 6.02 | 4.66 ± 0.02 | ≤7.66 |

### pp16384

_Not measured (sweep was stopped before this context)._

## `Sx(paper)` sources

- **kr=1.0** → `1.00` — dense baseline / identity (Fig.6 short-ctx note)
- **kr=0.5** → `—` — no headline figure for this kr (see arXiv:2502.02789)
- **kr=0.25** → `2.54–6.54` — Fig.5 TTFT vs MInference (70B)
- **kr=0.1** → `≤7.66` — Fig.3 / §4.7.2 peak vs dense (405B-Instruct-FP8, ~10% tokens)

## Notes & known caveats

- **`kr=0.5` paper column is `—`** because arXiv:2502.02789 sweeps `kr ∈ {0.1, 0.25}` for headline figures; no published 70B Q4 number at `kr=0.5`.
- **`Sx(paper)` for `kr=0.1` (`≤7.66`) is the paper's *peak* (405B-Instruct-FP8, ~10% tokens, Fig.3 / §4.7.2)** — strictly higher than what a 70B-Q4 on L4×4 would achieve. Treat it as an upper bound, not a target.
- **`ours Sx` for `kr=1.0` is < 1.0** because the spec-prefill code path still runs lookahead + importance compute even when keeping all tokens. The gap (≈0.65 at pp=128, ≈0.90 at pp=8192) is the spec-prefill overhead amortized over a longer prefill.
- **pp16384 not measured** in this run (sweep stopped at user request after pp8192 completed). A single 16k cell at `r=5` takes ~15-18 minutes on this hardware.
- All runs used the `recovered` `llama-bench` source (`tools/llama-bench/llama-bench.cpp`, see `Cursor` history `zTT1.cpp` Apr 29 19:45). Same source built today gives reproducible numbers.
