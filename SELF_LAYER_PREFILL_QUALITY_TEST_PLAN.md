# Self-Layer Prefill — Quality Validation Test Plan

Based on the [SpecPrefill paper](https://arxiv.org/abs/2502.02789) methodology.

## Objective

Validate that self-layer prefill with token filtering (kr < 1.0) preserves
output quality compared to baseline full-context inference.

---

## Test Matrix

### Models
- Llama 3.1 8B Instruct Q4_K_M (primary)
- Optional: Llama 3.1 70B (if multi-GPU capacity allows)

### Parameters to sweep
| Parameter | Values |
|-----------|--------|
| `n_early_layers` (N) | 4 (L/8), 8 (L/4) |
| `keep_ratio` (kr) | 0.10, 0.25, 0.50, 1.00 (baseline) |
| `n_prompt` | 1024, 2048, 4096, 8192 |

### Metrics
1. **Task accuracy** (% correct vs baseline)
2. **Perplexity** (PPL on held-out text)
3. **Generation coherence** (human eval or LLM-as-judge)
4. **Exact match / F1** (for QA tasks)

---

## Phase 1: Perplexity Evaluation

### Goal
Measure how much filtering degrades the model's language modeling ability.

### Method
1. Use WikiText-2 or a subset of C4/OpenWebText as test corpus
2. For each (N, kr) configuration:
   - Run self-layer prefill on context window
   - Generate next-token probabilities
   - Compute perplexity over test set
3. Compare to baseline (kr=1.0)

### Implementation

```cpp
// Pseudocode for perplexity test
for each test_chunk in corpus:
    // Baseline
    baseline_logits = llama_decode(ctx, test_chunk)
    baseline_ppl = compute_perplexity(baseline_logits, test_chunk)
    
    // Self-layer prefill
    n_kept = llama_self_layer_prefill_partial(ctx, test_chunk, params)
    filtered_logits = get_logits(ctx)
    filtered_ppl = compute_perplexity(filtered_logits, test_chunk)
    
    // Record: ppl_ratio = filtered_ppl / baseline_ppl
```

### Acceptance criteria
| keep_ratio | Max PPL increase |
|------------|------------------|
| 0.50 | < 5% |
| 0.25 | < 15% |
| 0.10 | < 30% |

---

## Phase 2: LongBench-Style Task Evaluation

### Goal
Measure task-specific accuracy on long-context benchmarks.

### Tasks (subset of LongBench)

| Category | Task | Metric | n_prompt range |
|----------|------|--------|----------------|
| Single-doc QA | NarrativeQA | F1 | 2k-8k |
| Multi-doc QA | HotpotQA | F1 / EM | 4k-16k |
| Summarization | GovReport | ROUGE-L | 4k-8k |
| Few-shot | TriviaQA | EM | 2k-4k |
| Code | HumanEval | pass@1 | 1k-2k |
| Synthetic | PassKey | EM | 4k-32k |

### Implementation steps

1. **Download LongBench dataset**
   ```bash
   git clone https://github.com/THUDM/LongBench
   # Or use HuggingFace: datasets.load_dataset("THUDM/LongBench")
   ```

2. **Create evaluation harness**
   ```
   examples/self-layer-prefill-eval/
   ├── main.cpp           # Driver
   ├── tasks/
   │   ├── narrativeqa.cpp
   │   ├── hotpotqa.cpp
   │   ├── govreport.cpp
   │   └── passkey.cpp
   └── metrics.cpp        # F1, EM, ROUGE
   ```

3. **Run evaluation**
   ```bash
   ./llama-self-layer-prefill-eval \
       --model models/llama-3.1-8b-instruct-q4_k_m.gguf \
       --task narrativeqa \
       --n-early 4 --keep-ratio 0.25 \
       --output results/narrativeqa-N4-kr0.25.json
   ```

### Acceptance criteria (relative to baseline kr=1.0)

| keep_ratio | Min accuracy retention |
|------------|------------------------|
| 0.50 | ≥ 98% |
| 0.25 | ≥ 95% |
| 0.10 | ≥ 90% |

---

## Phase 3: PassKey Retrieval (Needle-in-Haystack)

### Goal
Test if filtering drops critical information hidden in long context.

### Method
1. Generate synthetic prompts with a "passkey" (random number) embedded at
   various positions (10%, 25%, 50%, 75%, 90% depth)
2. Ask model to retrieve the passkey
3. Measure exact match rate at each (kr, depth) combination

### Test cases
```
Prompt template:
"There is important information hidden in the following text. The passkey is {PASSKEY}.
[FILLER TEXT ~4000 tokens]
What is the passkey mentioned above?"
```

### Expected failure modes
- kr=0.10 may drop passkey if it appears in "unimportant" middle sections
- Shallow layers (N=4) may not capture long-range dependencies

### Implementation
```cpp
struct PassKeyTest {
    int passkey;
    float depth;  // 0.0 = start, 1.0 = end
    int n_total_tokens;
};

bool run_passkey_test(llama_context * ctx, PassKeyTest test, 
                      llama_self_layer_prefill_params params) {
    auto prompt = generate_passkey_prompt(test);
    int n_kept = llama_self_layer_prefill_partial(ctx, prompt, params);
    
    // Check if passkey position was kept
    bool passkey_kept = check_position_kept(test.depth * prompt.size());
    
    // Generate response and check for passkey
    std::string response = generate(ctx, "The passkey is");
    bool correct = response.contains(std::to_string(test.passkey));
    
    return correct;
}
```

### Acceptance criteria
| keep_ratio | Min passkey retrieval |
|------------|----------------------|
| 0.50 | ≥ 95% |
| 0.25 | ≥ 85% |
| 0.10 | ≥ 70% (may fail on middle positions) |

---

## Phase 4: Generation Quality (LLM-as-Judge)

### Goal
Use a stronger model to judge if filtered outputs are coherent/correct.

### Method
1. Generate responses to a set of prompts using:
   - Baseline (kr=1.0)
   - Self-layer prefill (kr=0.25)
2. Use GPT-4 or Claude to compare outputs

### Judge prompt template
```
You are evaluating two responses to the same question.

Question: {QUESTION}
Context: {CONTEXT}

Response A (baseline): {RESPONSE_A}
Response B (filtered): {RESPONSE_B}

Rate Response B compared to Response A:
1 = Much worse (missing critical information)
2 = Slightly worse (minor quality loss)
3 = Equal quality
4 = Slightly better
5 = Much better

Provide your rating and explanation.
```

### Test set
- 100 prompts from each LongBench category
- Mix of short (1k) and long (8k) contexts

### Acceptance criteria
- Average rating ≥ 2.5 for kr=0.25
- < 10% rated as "Much worse" (1)

---

## Phase 5: Attention Score Correlation (Already Implemented)

### Current implementation
We already compute Pearson correlation between:
- Shallow attention (layers 0..N)
- Full attention (all layers)
- Deep attention (layers N..L)

### Current results
| n_prompt | Pearson(shallow vs full) |
|----------|--------------------------|
| 1024 | 0.997 |
| 2048 | 0.879 |

### Additional analysis needed
1. **Per-position correlation**: Are certain positions systematically
   mis-ranked by shallow layers?
2. **Token-type analysis**: Do shallow layers correctly identify:
   - Question tokens (high importance)
   - Filler/connector words (low importance)
   - Named entities (variable importance)

---

## Implementation Priority

| Phase | Effort | Value | Priority |
|-------|--------|-------|----------|
| 1. Perplexity | Low | High | **P0** |
| 3. PassKey | Low | High | **P0** |
| 2. LongBench | High | High | **P1** |
| 4. LLM-as-Judge | Medium | Medium | **P2** |
| 5. Attention analysis | Low | Low | **P3** |

---

## Quick-Start: Minimal Quality Check

For a fast sanity check before full evaluation:

```bash
# 1. Perplexity on small test set
./build/bin/llama-perplexity \
    --model models/llama-3.1-8b-instruct-q4_k_m.gguf \
    --file test_data/wikitext-2-raw/wiki.test.raw \
    --ctx-size 2048

# 2. Manual inspection: generate with filtering
./build/bin/llama-self-layer-prefill-run \
    --model models/llama-3.1-8b-instruct-q4_k_m.gguf \
    --mode once --n-early 4 --keep-ratio 0.25

# Compare baseline vs filtered output visually
```

---

## Test Data Sources

| Dataset | URL | Size |
|---------|-----|------|
| WikiText-2 | https://huggingface.co/datasets/wikitext | 4.5 MB |
| LongBench | https://github.com/THUDM/LongBench | ~500 MB |
| RULER | https://github.com/hsiehjackson/RULER | Synthetic |
| Needle-in-Haystack | https://github.com/gkamradt/LLMTest_NeedleInAHaystack | Synthetic |

---

## Expected Results Summary

If self-layer prefill works correctly:

| keep_ratio | TTFT speedup | Expected quality |
|------------|--------------|------------------|
| 1.00 | 1.0× (baseline) | 100% |
| 0.50 | ~1.7× | 98%+ |
| 0.25 | ~2.5-3× | 95%+ |
| 0.10 | ~4× | 90%+ (task-dependent) |

If quality drops significantly (e.g., <85% at kr=0.25), investigate:
1. Is N (n_early_layers) too small?
2. Are shallow attention scores poorly correlated with importance?
3. Is the scoring algorithm (last-row Q·K sum) inadequate?

---

## Notes

- **SpecPrefill used a separate 8B model** to score tokens for 405B; we use
  the same model's early layers. This may affect quality differently.
  
- **Quantization matters**: Q4_K_M may have different attention patterns than
  FP16. Consider testing with higher precision if quality is poor.

- **Position bias**: Ensure the test evaluates passkey/answer positions across
  the full context, not just at the end (where the last token is always kept).
