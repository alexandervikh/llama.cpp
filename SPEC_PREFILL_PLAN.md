# Speculative Prefill POC Implementation Plan

## Overview
Implement speculative prefill optimization for llama.cpp using TDD approach. Small model analyzes prompt attention patterns to filter tokens before base model processes them.

## Phase 0: Test Infrastructure (Week 1, Day 1-2)

### 0.1 Create Test Framework
**File**: `tests/test-spec-prefill.cpp`

**Tests to write (all should FAIL initially):**

1. `test_dual_model_loading()`
   - Load base model (e.g., TinyLlama 1.1B)
   - Load spec model (e.g., TinyLlama 1.1B, same for POC)
   - Assert both models loaded successfully
   - Check model compatibility (vocab size, architecture)

2. `test_spec_model_lookahead_generation()`
   - Input: prompt tokens [1, 2, 3, 4, 5]
   - Configure lookahead count: 3
   - Expected: spec model generates 3 lookahead tokens
   - Assert: output has correct shape

3. `test_attention_score_extraction()`
   - Run spec model with prompt + lookahead
   - Extract Q tensors from lookahead positions
   - Extract K tensors from prompt positions
   - Assert: tensors have correct dimensions (n_layers, n_heads, seq_len, head_dim)

4. `test_attention_score_computation()`
   - Given: Q_lookahead [n_heads, 3, head_dim], K_prompt [n_heads, 5, head_dim]
   - Compute: attention_scores = Q @ K^T (before softmax)
   - Expected: [n_heads, 3, 5] attention matrix
   - Assert: shape matches

5. `test_token_importance_aggregation()`
   - Given: attention scores across layers/heads
   - Apply: softmax -> max over lookahead -> mean over heads -> mean over layers
   - Expected: importance score per prompt token [5]
   - Assert: scores sum to ~1.0, all positive

6. `test_token_filtering()`
   - Given: prompt tokens [1, 2, 3, 4, 5], importance [0.3, 0.1, 0.4, 0.05, 0.15]
   - Filter: keep top 60% -> keep tokens [1, 3, 2] (sorted by importance)
   - Preserve: original position indices [0, 2, 1]
   - Assert: filtered count correct, positions preserved

7. `test_base_model_execution_with_filtered_prompt()`
   - Given: filtered tokens with position IDs
   - Execute: base model forward pass
   - Expected: KV cache only for filtered tokens at correct positions
   - Assert: generation continues normally

8. `test_end_to_end_spec_prefill()`
   - Input: long prompt (e.g., 128 tokens)
   - Run: full spec prefill pipeline
   - Output: base model completion
   - Assert: output is coherent text
   - Compare: latency vs regular prefill (should be less)

**Build config:**
```cmake
# Add to tests/CMakeLists.txt
add_executable(test-spec-prefill test-spec-prefill.cpp get-model.cpp)
target_link_libraries(test-spec-prefill PRIVATE llama common ggml)
```

### 0.2 Test Runner Setup
```bash
# Run tests
./build/bin/test-spec-prefill

# Expected initial output:
# [FAIL] test_dual_model_loading: function not implemented
# [FAIL] test_spec_model_lookahead_generation: function not implemented
# ... (all 8 tests fail)
```

---

## Phase 1: Core Data Structures (Week 1, Day 2-3)

### 1.1 Create Spec Prefill Context
**File**: `include/llama-spec-prefill.h`

```cpp
struct llama_spec_prefill_params {
    int lookahead_count = 8;           // Number of lookahead tokens to generate
    float keep_ratio = 0.5f;           // Keep top 50% of tokens
    int n_layers_to_analyze = -1;      // -1 = all layers
    bool verbose = false;
};

struct llama_spec_prefill_context {
    struct llama_context * ctx_base;   // Base model context
    struct llama_context * ctx_spec;   // Spec model context
    struct llama_spec_prefill_params params;
    
    // Intermediate buffers for attention analysis
    std::vector<std::vector<float>> q_lookahead;  // [n_layers][n_heads * lookahead * head_dim]
    std::vector<std::vector<float>> k_prompt;     // [n_layers][n_heads * prompt_len * head_dim]
    std::vector<float> token_importance;          // [prompt_len]
    std::vector<int> filtered_tokens;             // Selected token IDs
    std::vector<int> filtered_positions;          // Original positions
};

// API functions
struct llama_spec_prefill_context * llama_spec_prefill_init(
    struct llama_context * ctx_base,
    struct llama_context * ctx_spec,
    struct llama_spec_prefill_params params
);

void llama_spec_prefill_free(struct llama_spec_prefill_context * ctx);
```

**Tests that should now PASS:**
- `test_dual_model_loading()` - after implementing init/free

---

## Phase 2: Lookahead Generation (Week 1, Day 3-4)

### 2.1 Implement Spec Model Forward Pass
**File**: `src/llama-spec-prefill.cpp`

```cpp
// Generate lookahead tokens with spec model
int llama_spec_prefill_generate_lookahead(
    struct llama_spec_prefill_context * ctx,
    const llama_token * prompt_tokens,
    int n_prompt
) {
    // 1. Clear spec model KV cache
    llama_kv_cache_clear(ctx->ctx_spec);
    
    // 2. Create batch with prompt tokens
    struct llama_batch batch = llama_batch_get_one(
        (llama_token *)prompt_tokens, n_prompt
    );
    
    // 3. Process prompt
    if (llama_decode(ctx->ctx_spec, batch) != 0) {
        return -1;
    }
    
    // 4. Generate lookahead tokens
    for (int i = 0; i < ctx->params.lookahead_count; i++) {
        // Sample next token
        llama_token token = llama_sampler_sample(/* ... */);
        
        // Add to sequence
        batch = llama_batch_get_one(&token, 1);
        if (llama_decode(ctx->ctx_spec, batch) != 0) {
            return -1;
        }
    }
    
    return ctx->params.lookahead_count;
}
```

**Tests that should now PASS:**
- `test_spec_model_lookahead_generation()`

---

## Phase 3: Q/K Tensor Extraction (Week 1, Day 4-5)

### 3.1 Hook into GGML Graph for Tensor Capture

**Challenge**: Extract intermediate Q/K tensors during forward pass.

**Approach 1 (Naive Recomputation)**: Re-run attention layers to capture Q/K
**Approach 2 (Graph Inspection)**: Hook into ggml computation graph

For POC, use **Approach 1** (simpler):

```cpp
// Extract Q/K by recomputing attention with custom ops
int llama_spec_prefill_extract_qk(
    struct llama_spec_prefill_context * ctx,
    int n_prompt,
    int n_lookahead
) {
    // For each layer:
    //   1. Get KV cache tensors (already computed)
    //   2. Extract K tensor for prompt positions [0, n_prompt)
    //   3. Recompute Q for lookahead positions [n_prompt, n_prompt+n_lookahead)
    //   4. Store in ctx->k_prompt[layer] and ctx->q_lookahead[layer]
    
    // NOTE: This requires accessing llama internal KV cache
    // May need to add getter functions to llama.h
    
    return 0;
}
```

**Required additions to `llama.h`:**
```cpp
// Get pointer to KV cache tensors for a layer
struct ggml_tensor * llama_get_kv_key_tensor(
    struct llama_context * ctx,
    int layer_idx
);

struct ggml_tensor * llama_get_kv_value_tensor(
    struct llama_context * ctx,
    int layer_idx
);
```

**Tests that should now PASS:**
- `test_attention_score_extraction()`

---

## Phase 4: Attention Score Computation (Week 2, Day 1-2)

### 4.1 Compute Attention Scores Using GGML

```cpp
int llama_spec_prefill_compute_attention(
    struct llama_spec_prefill_context * ctx
) {
    // For each layer:
    //   Q_lookahead: [n_heads, lookahead_count, head_dim]
    //   K_prompt:    [n_heads, n_prompt, head_dim]
    //   
    //   attention = Q @ K^T  -> [n_heads, lookahead_count, n_prompt]
    //   
    //   Use ggml_mul_mat for efficient matrix multiplication
    
    struct ggml_context * ggml_ctx = /* create temporary context */;
    
    for (int layer = 0; layer < n_layers; layer++) {
        struct ggml_tensor * Q = /* wrap ctx->q_lookahead[layer] */;
        struct ggml_tensor * K = /* wrap ctx->k_prompt[layer] */;
        
        // Transpose K: [n_heads, head_dim, n_prompt]
        struct ggml_tensor * K_T = ggml_cont(ggml_ctx, ggml_transpose(ggml_ctx, K));
        
        // Matrix multiply: [n_heads, lookahead_count, n_prompt]
        struct ggml_tensor * scores = ggml_mul_mat(ggml_ctx, K_T, Q);
        
        // Build and execute graph
        struct ggml_cgraph * graph = ggml_new_graph(ggml_ctx);
        ggml_build_forward_expand(graph, scores);
        ggml_graph_compute_with_ctx(ggml_ctx, graph, /* n_threads */);
        
        // Store results
        // ...
    }
    
    return 0;
}
```

**Tests that should now PASS:**
- `test_attention_score_computation()`

---

## Phase 5: Token Importance & Filtering (Week 2, Day 2-3)

### 5.1 Aggregate Attention Scores

```cpp
int llama_spec_prefill_compute_importance(
    struct llama_spec_prefill_context * ctx,
    int n_prompt,
    int n_lookahead
) {
    // Aggregation pipeline:
    // 1. Softmax over prompt dimension for each (layer, head, lookahead_pos)
    // 2. Max over lookahead dimension -> [n_layers, n_heads, n_prompt]
    // 3. Mean over heads -> [n_layers, n_prompt]
    // 4. Mean over layers -> [n_prompt]
    
    ctx->token_importance.resize(n_prompt, 0.0f);
    
    // Implementation details...
    
    return 0;
}
```

### 5.2 Filter Tokens by Importance

```cpp
int llama_spec_prefill_filter_tokens(
    struct llama_spec_prefill_context * ctx,
    const llama_token * prompt_tokens,
    int n_prompt
) {
    // 1. Compute keep_count = ceil(n_prompt * keep_ratio)
    int keep_count = std::ceil(n_prompt * ctx->params.keep_ratio);
    
    // 2. Sort token indices by importance (descending)
    std::vector<std::pair<float, int>> scored_tokens;
    for (int i = 0; i < n_prompt; i++) {
        scored_tokens.push_back({ctx->token_importance[i], i});
    }
    std::sort(scored_tokens.rbegin(), scored_tokens.rend());
    
    // 3. Keep top-k, preserve original order
    std::vector<int> kept_indices;
    for (int i = 0; i < keep_count; i++) {
        kept_indices.push_back(scored_tokens[i].second);
    }
    std::sort(kept_indices.begin(), kept_indices.end());
    
    // 4. Store filtered tokens and positions
    ctx->filtered_tokens.clear();
    ctx->filtered_positions.clear();
    for (int idx : kept_indices) {
        ctx->filtered_tokens.push_back(prompt_tokens[idx]);
        ctx->filtered_positions.push_back(idx);
    }
    
    return keep_count;
}
```

**Tests that should now PASS:**
- `test_token_importance_aggregation()`
- `test_token_filtering()`

---

## Phase 6: Base Model Execution (Week 2, Day 3-4)

### 6.1 Process Filtered Prompt with Position IDs

```cpp
int llama_spec_prefill_process_base(
    struct llama_spec_prefill_context * ctx
) {
    // Clear base model KV cache
    llama_kv_cache_clear(ctx->ctx_base);
    
    // Create batch with filtered tokens and their original positions
    struct llama_batch batch = llama_batch_init(
        ctx->filtered_tokens.size(), 
        0, 
        1
    );
    
    for (size_t i = 0; i < ctx->filtered_tokens.size(); i++) {
        batch.token[i] = ctx->filtered_tokens[i];
        batch.pos[i] = ctx->filtered_positions[i];  // CRITICAL: use original position
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == batch.n_tokens - 1);  // Only last token needs logits
    }
    batch.n_tokens = ctx->filtered_tokens.size();
    
    // Decode with base model
    if (llama_decode(ctx->ctx_base, batch) != 0) {
        llama_batch_free(batch);
        return -1;
    }
    
    llama_batch_free(batch);
    return 0;
}
```

**Tests that should now PASS:**
- `test_base_model_execution_with_filtered_prompt()`

---

## Phase 7: End-to-End Integration (Week 2, Day 4-5)

### 7.1 Main API Function

```cpp
int llama_spec_prefill(
    struct llama_spec_prefill_context * ctx,
    const llama_token * prompt_tokens,
    int n_prompt
) {
    // 1. Generate lookahead with spec model
    if (llama_spec_prefill_generate_lookahead(ctx, prompt_tokens, n_prompt) < 0) {
        return -1;
    }
    
    // 2. Extract Q/K tensors
    if (llama_spec_prefill_extract_qk(ctx, n_prompt, ctx->params.lookahead_count) < 0) {
        return -1;
    }
    
    // 3. Compute attention scores
    if (llama_spec_prefill_compute_attention(ctx) < 0) {
        return -1;
    }
    
    // 4. Aggregate to token importance
    if (llama_spec_prefill_compute_importance(ctx, n_prompt, ctx->params.lookahead_count) < 0) {
        return -1;
    }
    
    // 5. Filter tokens
    if (llama_spec_prefill_filter_tokens(ctx, prompt_tokens, n_prompt) < 0) {
        return -1;
    }
    
    // 6. Process with base model
    if (llama_spec_prefill_process_base(ctx) < 0) {
        return -1;
    }
    
    return ctx->filtered_tokens.size();
}
```

**Tests that should now PASS:**
- `test_end_to_end_spec_prefill()`

---

## Phase 8: CLI Tool (Week 2, Day 5)

### 8.1 Create Example Binary
**File**: `examples/spec-prefill/spec-prefill.cpp`

```cpp
int main(int argc, char ** argv) {
    // Parse args: base_model, spec_model, prompt, lookahead_count, keep_ratio
    
    // Load models
    llama_model * model_base = llama_load_model_from_file(path_base, params);
    llama_model * model_spec = llama_load_model_from_file(path_spec, params);
    
    // Create contexts
    llama_context * ctx_base = llama_new_context_with_model(model_base, cparams);
    llama_context * ctx_spec = llama_new_context_with_model(model_spec, cparams);
    
    // Initialize spec prefill
    llama_spec_prefill_params sp_params;
    sp_params.lookahead_count = lookahead;
    sp_params.keep_ratio = keep_ratio;
    
    llama_spec_prefill_context * sp_ctx = llama_spec_prefill_init(
        ctx_base, ctx_spec, sp_params
    );
    
    // Tokenize prompt
    std::vector<llama_token> tokens = llama_tokenize(ctx_base, prompt, true, true);
    
    // Run spec prefill
    auto t_start = std::chrono::high_resolution_clock::now();
    int n_processed = llama_spec_prefill(sp_ctx, tokens.data(), tokens.size());
    auto t_end = std::chrono::high_resolution_clock::now();
    
    printf("Processed %d/%d tokens in %.2f ms\n",
        n_processed, (int)tokens.size(),
        std::chrono::duration<double, std::milli>(t_end - t_start).count()
    );
    
    // Continue generation
    for (int i = 0; i < n_predict; i++) {
        llama_token token = llama_sampler_sample(/* ... */);
        // ...
    }
    
    // Cleanup
    llama_spec_prefill_free(sp_ctx);
    llama_free(ctx_base);
    llama_free(ctx_spec);
    llama_free_model(model_base);
    llama_free_model(model_spec);
    
    return 0;
}
```

**Usage:**
```bash
./build/bin/llama-spec-prefill \
    --model-base models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    --model-spec models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    --prompt "Write a long essay about the history of France." \
    --lookahead 8 \
    --keep-ratio 0.5 \
    --n-predict 256
```

---

## Phase 9: Benchmarking (Week 3)

### 9.1 Compare Performance
**File**: `examples/spec-prefill/bench-spec-prefill.cpp`

Measure:
- Regular prefill latency vs spec prefill latency
- Token reduction percentage
- Generation quality (perplexity)
- Speedup ratio

```bash
# Benchmark script
for prompt_len in 128 256 512 1024; do
    for keep_ratio in 0.3 0.5 0.7; do
        ./build/bin/bench-spec-prefill \
            --prompt-len $prompt_len \
            --keep-ratio $keep_ratio
    done
done
```

---

## Success Criteria

✅ **POC Complete When:**
1. All 8 unit tests pass
2. CLI tool runs successfully on TinyLlama
3. Speedup measured: spec prefill faster than regular prefill for prompts >256 tokens
4. Output quality: generated text is coherent
5. Documentation: README with usage examples

---

## Files to Create/Modify

**New Files:**
- `tests/test-spec-prefill.cpp` (unit tests)
- `include/llama-spec-prefill.h` (public API)
- `src/llama-spec-prefill.cpp` (implementation)
- `examples/spec-prefill/spec-prefill.cpp` (CLI tool)
- `examples/spec-prefill/CMakeLists.txt` (build config)

**Modified Files:**
- `src/llama.cpp` (add KV cache accessor functions)
- `include/llama.h` (expose KV tensor getters)
- `tests/CMakeLists.txt` (add test target)

---

## Estimated Timeline

- **Week 1**: Test infrastructure + core implementation (Phases 0-3)
- **Week 2**: Integration + CLI tool (Phases 4-8)
- **Week 3**: Benchmarking + optimization (Phase 9)

**Total: ~2-3 weeks for working POC**

---

## Next Steps

1. ✅ Create test file with failing tests
2. Implement dual model loading
3. Implement lookahead generation
4. Implement Q/K extraction
5. ... (follow phases)
