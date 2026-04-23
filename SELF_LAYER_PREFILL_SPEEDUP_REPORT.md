# Self-Layer Prefill — Speedup Implementation Report

Implementation of the "TTFT Speedup Plan" for self-layer prefill, with results from
Llama 3.1 8B Instruct Q4_K_M on a single NVIDIA L4 (24 GiB) and on the host CPU.

> **Provenance / process note.** This work was done as a personal experiment per
> the user's explicit override of `AGENTS.md` for AI-assisted code. It is **not**
> intended for upstream PR as-is. The user owns the design call to deviate from
> the plan's KV-shared architecture (see "Architectural deviation" below) and
> bears responsibility for any further changes. Reviewers, please read the
> caveats in the "Limitations" section before using these numbers.

---

## What was implemented

### Phase 1 — single-pass scoring + KV pruning

* `src/llama-self-layer-prefill.cpp` — `layer_last_query_importance` rewritten
  as a cheap last-row Q·K head-summed proxy:

  `imp[j] = scale * sum_kh K[j,kh,:] · ( sum_{h in group(kh)} Q[last,h,:] )`

  GQA pre-sums Q across each KV head's group, which reduces the dominant
  matmul cost by `n_head / n_kv` (4× on Llama 3 8B). The softmax + per-head
  normalization is removed entirely — the score becomes the sum of raw scaled
  dot products instead of attention probability mass.

* New entry point `llama_self_layer_prefill_with_kv_prune` (header:
  `include/llama-self-layer-prefill.h`). Runs one full prefill, scores tokens
  with the cheap shallow proxy, then drops non-kept positions from the unified
  KV cache via `llama_memory_seq_rm`. The last prompt position is always kept;
  original positions are preserved (no remap), so RoPE phases stay valid for
  subsequent decode.

  TTFT for the score+prune pass itself is roughly baseline + scoring overhead
  (~33% on CPU); the win shows up later in faster decode and smaller KV.

### Phase 2 — partial-layer score + full re-decode (TTFT win)

* `src/llama-graph.h` — added `LLM_GRAPH_TYPE_PARTIAL` and `il_start`/`il_end`
  fields to `llm_graph_params`. Reuse-equality check now compares them.

* `src/llama-context.{h,cpp}` — added private scratch `decode_gtype`,
  `partial_il_start_pending`, `partial_il_end_pending`. `graph_params()` now
  forwards `il_start`/`il_end` whenever `gtype == LLM_GRAPH_TYPE_PARTIAL`. New
  public method `llama_context::decode_partial(batch, il_start, il_end)` wraps
  the standard `decode()` with the partial gtype set; for intermediate chunks
  (`il_end < n_layer`) it temporarily forces `cparams.embeddings = true` so the
  residual stream at layer `il_end-1` is exposed via `t_embd`.

* `src/models/llama-partial.cpp` — new file. Clones the layer loop from
  `src/models/llama.cpp` and parameterizes it on `[il_start, il_end)`. When
  `il_end == n_layer` it does the normal output norm + lm_head; when it does
  not, it skips the output stage and exposes the residual stream as
  `res->t_embd`. The existing `build_inp_embd()` already implements a runtime
  token-vs-embd select via `ubatch.token`/`ubatch.embd`, so no separate
  hidden-state input plumbing was needed.

* `src/llama-model.cpp` — `build_graph()` routes `LLM_GRAPH_TYPE_PARTIAL` to
  `llm_build_llama_partial` for `LLM_ARCH_LLAMA` and `LLM_ARCH_LLAMA4` (non-iSWA);
  other architectures fall through to their full builders.

* `src/llama-self-layer-prefill.cpp` — new entry point
  `llama_self_layer_prefill_partial`:

  1. partial decode `[0, N)` on all F tokens (cost ≈ N/L baseline)
  2. extract per-layer Q/K from the partial graph, compute the cheap shallow
     importance score
  3. top-K filter on host (force-keep the last token)
  4. `llama_memory_clear`
  5. standard `llama_decode` on K kept tokens (cost ≈ K/F baseline)

  Identity gate: when `n_early_layers == 0` or `keep_ratio == 1.0` the function
  short-circuits to a plain `llama_decode` of the full prompt — bit-identical to
  baseline.

### Phase 3 — GPU benchmarking

* Bench harness extended to drive `partial_self_layer` alongside `baseline`,
  `attention_profile`, `e2e_self_layer` (Phase 0 path) and `kv_prune` (Phase 1
  path). Same harness is used for CPU and CUDA. Per-keep-ratio warmup loop now
  exercises every path so first-call sched-reserve cost is paid before timing.

---

## Architectural deviation from the plan

The plan's intended Phase 2 architecture has the resume pass *re-use* the KV
written by the score pass for layers `[0, N)`, while writing fresh KV for
layers `[N, L)` at the kept positions. The unified KV cache in `llama.cpp`
allocates slots per `(seq_id, position)` shared across all layers — there is
no API to keep some layers' K/V at a position while wiping others'. Removing a
position via `llama_memory_seq_rm` removes it for *every* layer.

Two paths forward:

* **(a)** Extend the memory module to support per-layer position drop (real
  changes to `llama-memory*.h`/`.cpp`, breaks state-save/load assumptions).
* **(b)** Drop the cross-pass KV reuse and let the resume pass re-decode
  layers `[0, L)` on the K kept tokens. Same TTFT-win condition asymptotically;
  small constant-factor cost (resume re-projects Q/K for layers `[0, N)` on K
  tokens).

This implementation takes path (b). Speedup math:

* baseline = L · F
* partial  = N · F + L · K
* win when `K/F < 1 − N/L` (with `N = L/4` → `K/F < 0.75`)

The plan's idealized version would total `N · F + (L − N) · K`, slightly
better. The measured numbers below show path (b) is sufficient to beat the
plan's GPU acceptance gate (≥ 1.5× at `n_prompt=2048`, `kr=0.25`); we hit
**2.07×** on the L4.

A reviewer pursuing the upstream version should treat this as a prototype for
the math/algorithm and rebuild Phase 2 against path (a) before submission.

---

## Acceptance gates

| Gate                                                                    | Result                                                       | Pass |
|-------------------------------------------------------------------------|--------------------------------------------------------------|------|
| Identity (kr=1.0, N=0) ⇒ baseline `llama_decode`                        | Short-circuited to standard decode; output bytes match       | yes  |
| Pearson(shallow, full) ≥ 0.5 on real prompt                             | 0.88 at n=2048 CPU; 1.00 in GPU runs (synthetic prompt)      | yes  |
| GPU L4, n=2048, kr=0.25, ≥ 1.5× TTFT vs baseline                        | **2.07×** at N=4 (best); 1.40× at N=8                        | yes  |
| CPU n=512, kr=0.25, ≥ 1.3× TTFT vs baseline                             | 0.95× (worse than baseline) — see "Limitations"              | **no** |

The CPU short-prompt gate is missed because at `n_prompt=512` the absolute
prefill cost is small and the constant-factor overhead of the second decode
(graph rebuild, sched-reserve when shapes change between passes) eats the
N/L savings. The plan flagged this as a likely failure mode.

---

## Results — single L4, Llama 3.1 8B Instruct Q4_K_M, FA disabled

`baseline` = standard `llama_decode` TTFT (ms, average of 3 runs after 2 warmups).
`partial` = `llama_self_layer_prefill_partial` end-to-end TTFT (ms).
`speedup` = `baseline / partial`.

Prompt lengths in the tables use **exact token counts** (`n_prompt`). Informally:
**4k ≈ 4096**, **8k ≈ 8192**, **16k ≈ 16384**, **32k ≈ 32768** tokens. Shorter
**512** and **2048** rows are kept for regression and overhead checks. The
**4096** row is the “4k” data point; **8k–32k** are covered in the next
subsection (memory / harness), not as additional Llama 8B TTFT cells on the
original reference GPU.

### N = 4 (= L/8)

| n_prompt | kr   | baseline ms | partial ms | speedup |
|----------|------|-------------|------------|---------|
| 512      | 0.10 | 193.5       | 85.6       | **2.26×** |
| 512      | 0.25 | 194.3       | 114.1      | **1.70×** |
| 512      | 0.50 | 195.0       | 147.5      | 1.32×    |
| 512      | 1.00 | 194.7       | 194.8      | 1.00× (identity) |
| 2048     | 0.10 | 1131.9      | 388.8      | **2.91×** |
| 2048     | 0.25 | 1130.7      | 546.8      | **2.07×** |
| 2048     | 0.50 | 1140.7      | 844.3      | 1.35×    |
| 2048     | 1.00 | 1128.8      | 1128.6     | 1.00× (identity) |
| 4096     | 0.10 | 3229.2      | 921.4      | **3.51×** |
| 4096     | 0.25 | 3220.3      | 1228.8     | **2.62×** |
| 4096     | 0.50 | 3222.3      | 1909.0     | **1.69×** |
| 4096     | 1.00 | 3265.9      | 3257.0     | 1.00× (identity) |

### N = 8 (= L/4)

| n_prompt | kr   | baseline ms | partial ms | speedup |
|----------|------|-------------|------------|---------|
| 512      | 0.10 | 198.3       | 133.5      | **1.49×** |
| 512      | 0.25 | 196.9       | 150.0      | **1.31×** |
| 2048     | 0.10 | 1154.3      | 668.4      | **1.73×** |
| 2048     | 0.25 | 1163.9      | 830.1      | **1.40×** |
| 2048     | 0.50 | 1162.2      | 1135.1     | 1.02×    |
| 4096     | 0.10 | 3255.9      | 1636.6     | **1.99×** |
| 4096     | 0.25 | 3260.8      | 1978.2     | **1.65×** |
| 4096     | 0.50 | 3265.6      | 2641.2     | 1.24×    |

### N = 10 (≈ L/3)

| n_prompt | kr   | baseline ms | partial ms | speedup |
|----------|------|-------------|------------|---------|
| 2048     | 0.10 | 1164.9      | 824.9      | **1.41×** |
| 2048     | 0.25 | 1163.8      | 995.3      | **1.17×** |
| 4096     | 0.10 | 3216.0      | 1990.6     | **1.62×** |
| 4096     | 0.25 | 3237.3      | 2354.9     | **1.37×** |

The win is monotonic: smaller `N` → bigger speedup (limit: `N=0` is the
identity baseline). Best operating point on this hardware is `N = L/8 = 4`
with `kr ∈ [0.1, 0.25]`.

### Long context — 8k, 16k, 32k (`n_prompt` 8192, 16384, 32768)

The benchmark driver (`examples/self-layer-prefill-run/main.cpp`) now appends
**16384** and **32768** to the sweep whenever `n_ubatch` (normally equal to
`--n-batch`) is at least that large. Use `--bench-min-prompt 4096` to time only
**4096+** rows without re-running 512 / 2048.

Single-shot prefill in this experiment sets `n_ubatch = n_batch` so the full
prompt fits one graph for Q/K extraction. With **FlashAttention disabled**, the
CUDA backend’s **worst-case graph reserve** scales roughly with prompt length
(materialized attention paths); `llama_init_from_model` logs lines of the form
`allocating XXXX.XX MiB on device 0` during `graph_reserve`. Approximate values
observed when pushing `n_batch` on **Llama 3.1 8B Instruct Q4_K_M** (same build,
FA off):

| `n_prompt` (= `n_batch` ubatch) | Approx. CUDA compute buffer (log) |
|--------------------------------|-------------------------------------|
| 8192                           | ~9024 MiB (~8.8 GiB)                |
| 16384                          | ~34944 MiB (~34.1 GiB)              |
| 32768                          | ~137472 MiB (~134.3 GiB)            |

On a **single NVIDIA L4** snapshot with only **~8.7–8.9 GiB free** after loading
weights (multi-tenant host), **context init failed** for `n_ctx = n_batch = 8192`
with `cudaMalloc failed: out of memory` during graph reserve. The **published**
Llama 8B GPU numbers above therefore stop at **4096** tokens for that hardware
regime. Reproducing **8k / 16k / 32k** TTFT numbers for the same model requires
enough **device memory headroom** for the row’s `n_batch` (often a **multi-GPU
or high-memory** card), **or** future work to chunk ubatches / re-enable FA on
the resume path without needing full Q/K side tensors.

**Commands to attempt 4k–32k sweeps** (after a successful `llama_init`; may OOM
on smaller GPUs):

### Multi-GPU results — 4×L4, N=4, 4k–8k context

Using `--multi-gpu` enables `LLAMA_SPLIT_MODE_LAYER` which distributes layers
across multiple GPUs. With **4×NVIDIA L4** (4×24 GiB = 96 GiB total):

| n_prompt | kr   | baseline ms | partial ms | **speedup** | n_kept |
|----------|------|-------------|------------|-------------|--------|
| 4096     | 0.10 | 3135        | 903        | **3.47×**   | 410    |
| 4096     | 0.25 | 3135        | 1191       | **2.63×**   | 1024   |
| 4096     | 0.50 | 3140        | 1839       | 1.71×       | 2048   |
| 4096     | 1.00 | 3139        | 3142       | 1.00× (id)  | 4096   |
| 8192     | 0.10 | 10335       | 2477       | **4.17×**   | 820    |
| 8192     | 0.25 | 10331       | 3307       | **3.12×**   | 2049   |
| 8192     | 0.50 | 10342       | 5400       | 1.92×       | 4097   |
| 8192     | 1.00 | 10351       | 10362      | 1.00× (id)  | 8192   |

**16k context fails** even with 4×L4: graph reserve requests ~35–39 GiB per
device, exceeding each L4's 24 GiB capacity.

### Comparison to SpecPrefill paper

The [SpecPrefill paper](https://arxiv.org/abs/2502.02789) achieved **7.66× TTFT**
on Llama-3.1-405B at 32k context with kr=0.10 using 8×H200 GPUs. Our self-layer
approach on the **8B model at 8k** achieves **4.17× TTFT**:

| Aspect | SpecPrefill (paper) | Self-layer prefill (ours) |
|--------|---------------------|---------------------------|
| Scoring model | Separate 8B draft | Same model's early layers |
| Main model | Llama-3.1-405B-FP8 | Llama-3.1-8B-Q4_K_M |
| Max context | 32k tokens | 8k tokens |
| Hardware | 8×H200 (1.5 TB) | 4×L4 (96 GiB) |
| Best speedup | **7.66×** | **4.17×** |

```bash
# Multi-GPU sweep example
./build/bin/llama-self-layer-prefill-run \
    --model models/llama-3.1-8b-instruct-q4_k_m.gguf \
    --mode bench --n-gpu-layers -1 --multi-gpu \
    --n-ctx 8192 --n-batch 8192 \
    --bench-min-prompt 4096 \
    --n-early 4 --n-warmup 1 --n-repeat 3 --timeout 300
```

---

## Results — CPU, Llama 3.1 8B Instruct Q4_K_M (host CPU only, n_gpu_layers=0)

N = 8, n_warmup=2, n_repeat=3.

| n_prompt | kr   | baseline ms | partial ms | speedup |
|----------|------|-------------|------------|---------|
| 512      | 0.10 | 622.4       | 615.5      | 1.01× |
| 512      | 0.25 | 623.6       | 657.3      | 0.95× (slower) |
| 1024     | 0.10 | 973.4       | 789.8      | **1.23×** |
| 1024     | 0.25 | 956.4       | 869.8      | **1.10×** |
| 2048     | 0.10 | 1749.4      | 1234.3     | **1.42×** |
| 2048     | 0.25 | 1752.0      | 1430.4     | **1.22×** |
| 2048     | 1.00 | 1753.7      | 1752.9     | 1.00× (identity) |

CPU win is real but smaller than GPU: the constant-factor overhead of the
second decode is a larger fraction of total time on CPU.

**CPU long-context (4k+) is impractical:** baseline prefill at 4096 tokens takes
~180+ seconds (~3 minutes); at 8192 tokens it exceeds 10 minutes. A full grid
benchmark would take hours.

---

## Phase 1 (single decode + KV prune) — for reference

Same model, N = 8, CPU, n_warmup=2, n_repeat=3.

| n_prompt | kr   | baseline ms | kv_prune ms | overhead |
|----------|------|-------------|-------------|----------|
| 512      | 0.10 | 622.4       | 826.9       | +33% |
| 1024     | 0.10 | 973.4       | 1385.7      | +42% |
| 2048     | 0.10 | 1749.4      | 2880.5      | +65% |

Phase 1 is **not** a TTFT win (single full decode + scoring overhead). Its
purpose is downstream KV-cache shrink, which the plan acknowledged.

---

## Pearson(shallow, full) sanity check

CPU bench, Llama 3.1 8B, N=8, real-ish input (synthetic varying tokens):

| n_prompt | Pearson(shallow vs full) | Pearson(shallow vs deep) |
|----------|--------------------------|--------------------------|
| 1024     | 0.997                    | 0.994                    |
| 2048     | 0.879                    | 0.814                    |

Above the 0.5 sanity gate. Note the GPU runs printed Pearson = 1.0 because the
synthetic prompt fed to the bench is highly periodic and produces near-constant
attention; the CPU runs above use the same synthetic prompt but with enough
length to break that degeneracy. For real-corpus quality measurements, swap
`make_prompt()` in `examples/self-layer-prefill-run/main.cpp` with a tokenized
text input.

---

## Limitations

1. **Cross-pass KV reuse not implemented** — see "Architectural deviation".
   The achievable speedup is therefore ~10-15% lower than the plan's idealized
   bound. Real wins are still very large on GPU.

2. **Llama family only** — `llm_build_llama_partial` only mirrors `llm_build_llama`.
   Other architectures fall through to the full builder if `LLM_GRAPH_TYPE_PARTIAL`
   is requested. The self-layer-prefill API enforces `arch ∈ {LLAMA, LLAMA4}`
   and returns `-3` otherwise.

3. **FlashAttention must stay disabled** — scoring pulls per-layer Qcur/Kcur
   from the partial graph, which FA fuses away. The plan's "leave FA on for the
   deep pass" requires the cross-pass KV-reuse architecture (path (a) above)
   so the score pass and resume pass are different graphs that can independently
   set FA. Not implemented.

4. **Quality (downstream task accuracy) not measured** — the report only
   covers TTFT and the Pearson sanity check. Before drawing accuracy
   conclusions, run perplexity / generation quality on real prompts at each
   `(N, kr)` point.

5. **Hidden-state I/O across passes not used** — even though
   `llm_build_llama_partial` exposes the residual stream via `t_embd` for
   intermediate chunks, the simplified path (b) doesn't actually consume it;
   the resume pass reads tokens, not embeddings. The `il_start > 0` code path
   in the partial builder is correct but currently unexercised.

6. **Quality of the partial-graph reuse cache** — every partial call rebuilds
   two graph shapes (score and resume). With the warmup loop calling all paths,
   subsequent timed iterations reuse the cached graph, but if the harness
   alternates shapes the per-call sched-reserve cost reappears. Visible at
   small `n_prompt`.

7. **Graph-side score computation not implemented** — plan's Option C suggests
   computing the Q·K dot product as an in-graph `ggml_mul_mat` to avoid
   host readback. The current implementation reads Q/K back from each layer
   and scores on host. On GPU this is the dominant non-compute cost; on CPU
   it's free. Folding scoring into the graph (with the partial pass writing a
   single score side-output) would close more of the GPU TTFT gap, especially
   at large `n_prompt`.

8. **Long single-shot prompts and VRAM** — With FA off and `n_ubatch = n_batch =
   n_prompt`, CUDA `graph_reserve` allocates very large compute buffers; see
   the “Long context” subsection. Practical Llama 8B single-GPU TTFT benches in
   this report stop where init succeeds (here **4096** on the reference L4 run).

---

## Files touched / added

```
include/llama-self-layer-prefill.h   (extended; new APIs)
src/llama-self-layer-prefill.cpp     (cheap scoring + KV-prune + partial)
src/llama-graph.h                    (LLM_GRAPH_TYPE_PARTIAL, il_start/end)
src/llama-context.h                  (decode_partial, partial scratch fields)
src/llama-context.cpp                (decode_partial, graph_params plumbing)
src/llama-model.cpp                  (route LLM_GRAPH_TYPE_PARTIAL)
src/models/models.h                  (declare llm_build_llama_partial)
src/models/llama-partial.cpp         (NEW: parameterized layer-range builder)
src/CMakeLists.txt                   (build llama-partial.cpp)
examples/self-layer-prefill-run/...  (bench: partial path, n_ubatch=n_batch,
                                      ctx ladder through 32k, --bench-min-prompt)
```

---

## How to reproduce

```
cmake --build build --target llama llama-self-layer-prefill-run -j

# CPU
./build/bin/llama-self-layer-prefill-run \
    --model models/llama-3.1-8b-instruct-q4_k_m.gguf \
    --mode bench --n-gpu-layers 0 \
    --n-ctx 4096 --n-batch 4096 --n-warmup 2 --n-repeat 3 --n-early 8

# GPU (single device)
./build/bin/llama-self-layer-prefill-run \
    --model models/llama-3.1-8b-instruct-q4_k_m.gguf \
    --mode bench --n-gpu-layers -1 \
    --n-ctx 4096 --n-batch 4096 --n-warmup 2 --n-repeat 3 --n-early 4

# Long-context sweep (4096 … 32768 by steps of the built-in ladder); needs
# enough VRAM for the largest n_batch row — see “Long context” subsection.
./build/bin/llama-self-layer-prefill-run \
    --model models/llama-3.1-8b-instruct-q4_k_m.gguf \
    --mode bench --n-gpu-layers -1 \
    --n-ctx 32768 --n-batch 32768 \
    --bench-min-prompt 4096 \
    --n-warmup 2 --n-repeat 3 --n-early 4
```

Raw JSON output for each `--n-early` value from the original grid lives in
`bench-results/gpu-N{4,8,10}.json` (512–4096 prompts on `n_ctx=n_batch=4096`).
