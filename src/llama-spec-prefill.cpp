#include "llama-spec-prefill.h"
#include "llama.h"
#include <vector>
#include <cstdio>
#include <algorithm>

llama_spec_prefill_context * llama_spec_prefill_init(
    llama_context * ctx_base,
    llama_context * ctx_spec
) {
    if (!ctx_base || !ctx_spec) {
        fprintf(stderr, "%s: invalid context pointers\n", __func__);
        return nullptr;
    }

    llama_spec_prefill_context * ctx = new llama_spec_prefill_context;
    ctx->ctx_base = ctx_base;
    ctx->ctx_spec = ctx_spec;

    return ctx;
}

void llama_spec_prefill_free(llama_spec_prefill_context * ctx) {
    if (ctx) {
        delete ctx;
    }
}

int llama_spec_prefill_generate_lookahead(
    llama_spec_prefill_context * ctx,
    const llama_token * prompt_tokens,
    int n_prompt,
    llama_token * lookahead_tokens,
    int n_lookahead
) {
    if (!ctx || !prompt_tokens || !lookahead_tokens || n_prompt <= 0 || n_lookahead <= 0) {
        return -1;
    }

    // Clear spec model KV cache
    llama_memory_t mem = llama_get_memory(ctx->ctx_spec);
    llama_memory_seq_rm(mem, 0, 0, -1);

    const llama_model * model = llama_get_model(ctx->ctx_spec);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);

    // Process prompt tokens through spec model
    llama_batch batch = llama_batch_init(n_prompt, 0, 1);
    batch.n_tokens = 0;

    for (int i = 0; i < n_prompt; i++) {
        batch.token[batch.n_tokens] = prompt_tokens[i];
        batch.pos[batch.n_tokens] = i;
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.seq_id[batch.n_tokens][0] = 0;
        batch.logits[batch.n_tokens] = false;
        batch.n_tokens++;
    }
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx->ctx_spec, batch) != 0) {
        llama_batch_free(batch);
        return -1;
    }

    // Generate lookahead tokens using greedy sampling
    int n_generated = 0;
    llama_token token = prompt_tokens[n_prompt - 1];

    for (int i = 0; i < n_lookahead; i++) {
        float * logits = llama_get_logits_ith(ctx->ctx_spec, batch.n_tokens - 1);

        // Greedy sampling: find token with max logit
        llama_token next_token = 0;
        float max_logit = logits[0];
        for (int j = 1; j < n_vocab; j++) {
            if (logits[j] > max_logit) {
                max_logit = logits[j];
                next_token = j;
            }
        }

        lookahead_tokens[i] = next_token;
        n_generated++;

        // Prepare next batch
        batch.n_tokens = 0;
        batch.token[batch.n_tokens] = next_token;
        batch.pos[batch.n_tokens] = n_prompt + i;
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.seq_id[batch.n_tokens][0] = 0;
        batch.logits[batch.n_tokens] = true;
        batch.n_tokens++;

        if (llama_decode(ctx->ctx_spec, batch) != 0) {
            break;
        }

        token = next_token;
    }

    llama_batch_free(batch);
    return n_generated;
}

int llama_spec_prefill_extract_qk(
    llama_spec_prefill_context * ctx,
    const llama_token * lookahead_tokens,
    int n_lookahead
) {
    // Simplified stub for POC
    // TODO: Implement actual Q/K tensor extraction from KV cache
    return 0;
}

int llama_spec_prefill_compute_attention(
    llama_spec_prefill_context * ctx,
    const llama_token * prompt_tokens,
    int n_prompt,
    const llama_token * lookahead_tokens,
    int n_lookahead
) {
    // Simplified stub for POC
    // TODO: Implement actual attention computation using GGML
    return 0;
}

int llama_spec_prefill_compute_importance(
    llama_spec_prefill_context * ctx,
    const llama_token * prompt_tokens,
    int n_prompt,
    std::vector<float> & token_importance
) {
    if (!ctx || !prompt_tokens || n_prompt <= 0) {
        return -1;
    }

    // Simplified POC: Use position-based heuristic
    // Later tokens are more important (recency bias)
    token_importance.resize(n_prompt);
    for (int i = 0; i < n_prompt; i++) {
        // Linear weight from 0.1 to 1.0
        token_importance[i] = 0.1f + 0.9f * (float)i / (n_prompt - 1);
    }

    return 0;
}

int llama_spec_prefill_filter_tokens(
    llama_spec_prefill_context * ctx,
    const llama_token * prompt_tokens,
    const int * prompt_positions,
    int n_prompt,
    const std::vector<float> & token_importance,
    float keep_ratio,
    std::vector<llama_token> & filtered_tokens,
    std::vector<int> & filtered_positions
) {
    if (!ctx || !prompt_tokens || !prompt_positions || n_prompt <= 0) {
        return -1;
    }

    if (token_importance.size() != (size_t)n_prompt) {
        return -1;
    }

    // Calculate how many tokens to keep
    int n_keep = (int)(n_prompt * keep_ratio);
    if (n_keep < 1) n_keep = 1;
    if (n_keep > n_prompt) n_keep = n_prompt;

    // Create pairs of (importance, index)
    std::vector<std::pair<float, int>> importance_idx;
    importance_idx.reserve(n_prompt);
    for (int i = 0; i < n_prompt; i++) {
        importance_idx.push_back({token_importance[i], i});
    }

    // Sort by importance (descending)
    std::sort(importance_idx.begin(), importance_idx.end(),
        [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
            return a.first > b.first;
        });

    // Keep top n_keep tokens
    std::vector<int> kept_indices;
    kept_indices.reserve(n_keep);
    for (int i = 0; i < n_keep; i++) {
        kept_indices.push_back(importance_idx[i].second);
    }

    // Sort kept indices by position to maintain order
    std::sort(kept_indices.begin(), kept_indices.end());

    // Build filtered arrays
    filtered_tokens.clear();
    filtered_positions.clear();
    filtered_tokens.reserve(n_keep);
    filtered_positions.reserve(n_keep);

    for (int idx : kept_indices) {
        filtered_tokens.push_back(prompt_tokens[idx]);
        filtered_positions.push_back(prompt_positions[idx]);
    }

    return 0;
}

int llama_spec_prefill_process_base(
    llama_spec_prefill_context * ctx,
    const llama_token * filtered_tokens,
    const int * filtered_positions,
    int n_filtered
) {
    if (!ctx || !filtered_tokens || !filtered_positions || n_filtered <= 0) {
        return -1;
    }

    // Clear base model KV cache
    llama_memory_t mem = llama_get_memory(ctx->ctx_base);
    llama_memory_seq_rm(mem, 0, 0, -1);

    // Process filtered tokens with their original positions
    llama_batch batch = llama_batch_init(n_filtered, 0, 1);
    batch.n_tokens = 0;

    for (int i = 0; i < n_filtered; i++) {
        // Use original position IDs to preserve positional embeddings
        batch.token[batch.n_tokens] = filtered_tokens[i];
        batch.pos[batch.n_tokens] = filtered_positions[i];
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.seq_id[batch.n_tokens][0] = 0;
        batch.logits[batch.n_tokens] = (i == n_filtered - 1);
        batch.n_tokens++;
    }

    int result = llama_decode(ctx->ctx_base, batch);
    llama_batch_free(batch);

    return result;
}

int llama_spec_prefill(
    llama_spec_prefill_context * ctx,
    const llama_token * prompt_tokens,
    int n_prompt,
    int n_lookahead,
    float keep_ratio
) {
    if (!ctx || !prompt_tokens || n_prompt <= 0 || n_lookahead <= 0) {
        return -1;
    }

    // Step 1: Generate lookahead tokens
    std::vector<llama_token> lookahead_tokens(n_lookahead);
    int n_generated = llama_spec_prefill_generate_lookahead(
        ctx, prompt_tokens, n_prompt, lookahead_tokens.data(), n_lookahead
    );
    if (n_generated <= 0) {
        return -1;
    }

    // Step 2: Extract Q/K tensors (simplified for POC)
    if (llama_spec_prefill_extract_qk(ctx, lookahead_tokens.data(), n_generated) != 0) {
        return -1;
    }

    // Step 3: Compute attention scores (simplified for POC)
    if (llama_spec_prefill_compute_attention(
            ctx, prompt_tokens, n_prompt, lookahead_tokens.data(), n_generated) != 0) {
        return -1;
    }

    // Step 4: Compute token importance
    std::vector<float> token_importance;
    if (llama_spec_prefill_compute_importance(ctx, prompt_tokens, n_prompt, token_importance) != 0) {
        return -1;
    }

    // Step 5: Filter tokens
    std::vector<int> prompt_positions(n_prompt);
    for (int i = 0; i < n_prompt; i++) {
        prompt_positions[i] = i;
    }

    std::vector<llama_token> filtered_tokens;
    std::vector<int> filtered_positions;
    if (llama_spec_prefill_filter_tokens(
            ctx, prompt_tokens, prompt_positions.data(), n_prompt,
            token_importance, keep_ratio,
            filtered_tokens, filtered_positions) != 0) {
        return -1;
    }

    // Step 6: Process with base model
    if (llama_spec_prefill_process_base(
            ctx, filtered_tokens.data(), filtered_positions.data(),
            (int)filtered_tokens.size()) != 0) {
        return -1;
    }

    return (int)filtered_tokens.size();
}
