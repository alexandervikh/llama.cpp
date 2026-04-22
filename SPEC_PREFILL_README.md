# Speculative Prefill in llama.cpp

Speculative prefill optimizes the **prompt processing phase** by selectively filtering tokens based on importance scores, reducing computation while preserving model quality.

based on: https://github.com/Jingyu6/speculative_prefill 

## How It Works

1. **Generate lookahead tokens** using a draft model
2. **Extract Q/K tensors** for attention computation
3. **Compute attention scores** between prompt tokens and lookahead
4. **Calculate token importance** based on attention patterns
5. **Filter tokens** (keep most important ones, discard less important)
6. **Process filtered prompt** through the base model

## Building & Running Tests

```bash
# Build the full test suite (7 binaries)
cmake --build build -j4

# Run unit tests
./build/bin/test-spec-prefill-unit --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf

# Run all spec-prefill test binaries
./build/bin/test-spec-prefill --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf
./build/bin/test-spec-prefill-extended --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf
./build/bin/test-spec-prefill-integration --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf
./build/bin/test-spec-prefill-parity models/qwen2.5-0.5b-instruct-q4_k_m.gguf
./build/bin/test-spec-prefill-quality models/qwen2.5-0.5b-instruct-q4_k_m.gguf

## C API Usage

```cpp
#include "llama.h"
#include "llama-spec-prefill.h"

// Initialize with base and draft model contexts
llama_spec_prefill_context * sp_ctx = llama_spec_prefill_init(ctx_base, ctx_spec);

// Run spec prefill on prompt
llama_token prompt[] = {1, 450, 2043, 338, 263};
int n_prompt = 5;
int n_lookahead = 8;
// keep_ratio defaults to 0.25 (25% kept) if not specified
int n_filtered = llama_spec_prefill(
    sp_ctx, prompt, n_prompt, n_lookahead, 0.25f
);

// Use filtered prompt in base model...

// Cleanup
llama_spec_prefill_free(sp_ctx);
```

## Key Functions

| Function | Description |
|----------|-------------|
| `llama_spec_prefill_init()` | Create spec prefill context |
| `llama_spec_prefill()` | Full pipeline: filter + process |
| `llama_spec_prefill_generate_lookahead()` | Draft token generation |
| `llama_spec_prefill_compute_importance()` | Calculate token importance |
| `llama_spec_prefill_filter_tokens()` | Select top-k important tokens |
| `llama_spec_prefill_process_base()` | Process filtered tokens in base model |
| `llama_spec_prefill_free()` | Free context |

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_lookahead` | Tokens to generate from draft | 8 |
| `keep_ratio` | Fraction of prompt to keep | 0.25 (25%) |
| `pool_kernel_size` | Importance smoothing kernel | 13 (vLLM default) |

## Requirements

- **Two model contexts**: Base + draft model
- **Same vocabulary**: Both models must share tokenizer
- **GGUF format**: Quantized model files

## Example Model Download

```bash
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct-GGUF \
  qwen2.5-0.5b-instruct-q4_k_m.gguf --local-dir models

huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  qwen2.5-1.5b-instruct-q4_k_m.gguf --local-dir models
```

## CLI Usage

Spec prefill is **not exposed via CLI** - it's only available through the C API. There is no `--spec-prefill` flag in llama-cli or llama-server.

The test binary (`test-spec-prefill`) is the only CLI tool that exercises spec prefill:

```bash
# Build spec prefill test
cmake --build build --target test-spec-prefill -j4

# Run spec prefill tests (exercises the full C API pipeline)
./build/bin/test-spec-prefill models/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

To use spec prefill in your own application, call the C API directly (see usage example above).

## Implementation Status

**POC — Proof of Concept**

This is a simplified POC implementation of the speculative-prefill algorithm:
- Q/K extraction: works via `get_gf_res_prev()` on spec model graph
- Attention: **NO-OP** — importance scoring uses perplexity proxy instead
- Chunked filtering: implemented with fallback to percentage strategy
- Position preservation: tokens re-indexed to 0..n_kept-1 (RoPE limitation noted)

**NOT production-ready**: The attention computation stub (no-op) means importance
estimation relies entirely on the draft model's own confidence, not base-model
attention. This is the primary gate for production release.

## Files

| File | Description |
|------|-------------|
| [`include/llama-spec-prefill.h`](include/llama-spec-prefill.h) | C API header (public) |
| [`src/llama-spec-prefill.cpp`](src/llama-spec-prefill.cpp) | Core implementation |
| [`examples/spec-prefill-run/main.cpp`](examples/spec-prefill-run/main.cpp) | CLI evaluation harness |
| [`tests/test-spec-prefill*.cpp`](tests/) | 7 test binaries, 36+ tests |
| [`tools/spec-prefill-*.py`](tools/) | 7 Python tool scripts |
| [`tools/spec-prefill-parity-mock.sh`](tools/spec-prefill-parity-mock.sh) | C++ vs Python parity mock |