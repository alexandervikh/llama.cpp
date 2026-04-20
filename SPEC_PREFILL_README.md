# Speculative Prefill in llama.cpp

Speculative prefill optimizes the **prompt processing phase** by selectively filtering tokens based on importance scores, reducing computation while preserving model quality.

## How It Works

1. **Generate lookahead tokens** using a draft model
2. **Extract Q/K tensors** for attention computation
3. **Compute attention scores** between prompt tokens and lookahead
4. **Calculate token importance** based on attention patterns
5. **Filter tokens** (keep most important ones, discard less important)
6. **Process filtered prompt** through the base model

## Building & Running Tests

```bash
# Build the test binary
cmake --build build --target test-spec-prefill -j4

# Run tests (8/8 pass)
./build/bin/test-spec-prefill models/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

### Test Suite:
```
[1] test_dual_model_loading
[2] test_spec_model_lookahead_generation
[3] test_attention_score_extraction
[4] test_attention_computation
[5] test_token_importance_aggregation
[6] test_token_filtering
[7] test_base_model_execution_with_filtered_prompt
[8] test_end_to_end_spec_prefill
```

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
float keep_ratio = 0.6f;  // keep 60% of tokens

int n_filtered = llama_spec_prefill(
    sp_ctx, prompt, n_prompt, n_lookahead, keep_ratio
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
| `keep_ratio` | Fraction of prompt to keep | 0.6 (60%) |
| `pool_kernel_size` | Importance smoothing kernel | 1 (disabled) |

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

**POC** - Simplified implementation:
- Q/K extraction is basic
- Attention computation is basic
- Full vLLM-style chunked filtering not yet implemented

## Files

- [`src/llama-spec-prefill.cpp`](src/llama-spec-prefill.cpp) - Core implementation
- [`src/llama-spec-prefill.h`](src/llama-spec-prefill.h) - API header
- [`tests/test-spec-prefill.cpp`](tests/test-spec-prefill.cpp) - Test suite
- [`tests/test-spec-prefill-bench.cpp`](tests/test-spec-prefill-bench.cpp) - Benchmarks