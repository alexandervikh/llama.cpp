#include "llama-spec-prefill.h"
#include "llama.h"
#include "llama-graph.h"
#include "llama-kv-cache.h"
#include "llama-model.h"
#include "llama-context.h"
#include <vector>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <fstream>

llama_spec_prefill_context * llama_spec_prefill_init(
    struct llama_context * ctx_base,
    struct llama_context * ctx_spec
) {
    llama_spec_prefill_params default_params;
    return llama_spec_prefill_init_with_params(ctx_base, ctx_spec, default_params);
}

llama_spec_prefill_context * llama_spec_prefill_init_with_params(
    struct llama_context * ctx_base,
    struct llama_context * ctx_spec,
    const llama_spec_prefill_params & params
) {
    if (!ctx_base || !ctx_spec) {
        fprintf(stderr, "%s: invalid context pointers\n", __func__);
        return nullptr;
    }

    llama_spec_prefill_context * ctx = new llama_spec_prefill_context;
    ctx->ctx_base = ctx_base;
    ctx->ctx_spec = ctx_spec;
    ctx->lookahead_stats.clear();
    ctx->params = params;
    ctx->actual_lookahead_cnt = 0;
    
    // Initialize default EOS tokens if not provided
    if (ctx->params.eos_tokens.empty()) {
        const llama_model * model = llama_get_model(ctx_spec);
        const llama_vocab * vocab = llama_model_get_vocab(model);
        llama_token eos = llama_vocab_eos(vocab);
        if (eos != LLAMA_TOKEN_NULL) {
            ctx->params.eos_tokens.push_back(eos);
        }
    }

    return ctx;
}

void llama_spec_prefill_set_dump_path(
    llama_spec_prefill_context * ctx,
    const char * path
) {
    if (ctx && path) {
        ctx->dump_path = path;
    }
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
    llama_memory_clear(llama_get_memory(ctx->ctx_spec), false);

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
        fprintf(stderr, "[spec-prefill] generate_lookahead initial decode failed n_prompt=%d\n", n_prompt);
        llama_batch_free(batch);
        return -1;
    }

    // Generate lookahead tokens using greedy sampling
    // Track logit statistics for importance scoring
    int n_generated = 0;
    llama_token token = 0;
    ctx->lookahead_stats.resize(n_lookahead);
    ctx->actual_lookahead_cnt = n_lookahead;

    for (int i = 0; i < n_lookahead; i++) {
        float * logits = llama_get_logits_ith(ctx->ctx_spec, batch.n_tokens - 1);

        // Greedy sampling: find token with max logit
        // Also track logit statistics for importance computation
        llama_token next_token = 0;
        float max_logit = logits[0];
        float sum_exp = 0.0f;
        
        // First pass: find max and compute exp sum for entropy
        for (int j = 0; j < n_vocab; j++) {
            if (logits[j] > max_logit) {
                max_logit = logits[j];
                next_token = j;
            }
            sum_exp += expf(logits[j] - max_logit);
        }
        
        // Compute entropy (measure of prediction uncertainty)
        float entropy = 0.0f;
        for (int j = 0; j < n_vocab; j++) {
            float prob = expf(logits[j] - max_logit) / sum_exp;
            if (prob > 1e-10f) {
                entropy -= prob * logf(prob);
            }
        }
        
        // Store statistics for this lookahead position
        ctx->lookahead_stats[i].max_logit = max_logit;
        ctx->lookahead_stats[i].entropy = entropy;
        ctx->lookahead_stats[i].position = n_prompt + i;

        lookahead_tokens[i] = next_token;
        n_generated++;
        
        // Check for EOS token (like vLLM's _get_actual_look_ahead_cnts)
        if (!ctx->params.ignore_eos) {
            bool is_eos = false;
            for (llama_token eos : ctx->params.eos_tokens) {
                if (next_token == eos) {
                    is_eos = true;
                    break;
                }
            }
            if (is_eos) {
                ctx->actual_lookahead_cnt = i + 1;
                break;
            }
        }

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

// Helper: check if a tensor name matches Q/K tensor patterns
static bool is_q_tensor_name(const char * name) {
    if (!name || !name[0]) return false;
    const char * p = name;
    // Match "Qcur", "Qcur_normed", "Qcur_rope", "qcur", etc.
    // Case-insensitive check for "Q" followed by "cur" or similar
    if (p[0] != 'Q' && p[0] != 'q') return false;
    // Must contain "cur" somewhere in the name
    const char * cur_pos = nullptr;
    for (const char * s = p; *s; ++s) {
        if (*s == 'c' || *s == 'C') {
            if ((*(s+1) == 'u' || *(s+1) == 'U') && (*(s+2) == 'r' || *(s+2) == 'R')) {
                cur_pos = s;
                break;
            }
        }
    }
    if (!cur_pos) return false;
    // The char before 'C' should be 'Q' or 'q' (possibly with underscores)
    for (const char * s = cur_pos - 1; s >= p; --s) {
        if (*s == '_') continue;
        if (*s == 'Q' || *s == 'q') return true;
        return false;
    }
    return false;
}

static bool is_k_tensor_name(const char * name) {
    if (!name || !name[0]) return false;
    const char * p = name;
    if (p[0] != 'K' && p[0] != 'k') return false;
    const char * cur_pos = nullptr;
    for (const char * s = p; *s; ++s) {
        if (*s == 'c' || *s == 'C') {
            if ((*(s+1) == 'u' || *(s+1) == 'U') && (*(s+2) == 'r' || *(s+2) == 'R')) {
                cur_pos = s;
                break;
            }
        }
    }
    if (!cur_pos) return false;
    for (const char * s = cur_pos - 1; s >= p; --s) {
        if (*s == '_') continue;
        if (*s == 'K' || *s == 'k') return true;
        return false;
    }
    return false;
}

static int extract_layer_index(const char * name) {
    if (!name) return -1;
    const char * p = name;
    while (*p) {
        if (*p == 'c' || *p == 'C' || *p == 'u' || *p == 'U' || *p == 'r' || *p == 'R') {
            p++;
            if (*p == '-') {
                p++;
                if (*p >= '0' && *p <= '9') {
                    return atoi(p);
                }
                break;
            }
        } else {
            p++;
        }
    }
    return -1;
}

struct found_tensor {
    const ggml_tensor * tensor;
    int graph_idx;
    bool is_q;
};

static void filter_best_per_layer(std::vector<found_tensor> & candidates, int n_layer) {
    // Keep the last candidate (highest graph index) per layer
    std::vector<found_tensor> best;
    best.resize(n_layer);
    for (auto & c : candidates) {
        int li = extract_layer_index(c.tensor->name);
        if (li >= 0 && li < n_layer) {
            if (best[li].tensor == nullptr || c.graph_idx > best[li].graph_idx) {
                best[li] = c;
            }
        }
    }
    candidates = std::move(best);
}

int llama_spec_prefill_extract_qk(
    llama_spec_prefill_context * ctx,
    const llama_token * lookahead_tokens,
    int n_lookahead
) {
    if (!ctx || !lookahead_tokens || n_lookahead <= 0 || !ctx->ctx_spec) {
        return -1;
    }

    ctx->ctx_spec->synchronize();
    ctx->q_tensors.clear();

    auto * gf_res = ctx->ctx_spec->get_gf_res_prev();
    if (!gf_res) {
        fprintf(stderr, "[spec-prefill] extract_qk: no graph result\n");
        return -1;
    }

    auto * gf = gf_res->get_gf();
    if (!gf) {
        fprintf(stderr, "[spec-prefill] extract_qk: no computation graph\n");
        return -1;
    }

    const int n_layer = (int)llama_model_n_layer(&ctx->ctx_spec->get_model());
    const int n_nodes = ggml_graph_n_nodes(gf);

    std::vector<found_tensor> q_candidates;
    std::vector<found_tensor> k_candidates;

    for (int j = 0; j < n_nodes; j++) {
        ggml_tensor * t = ggml_graph_node(gf, j);
        if (!t || !t->name[0]) continue;
        if (ggml_n_dims(t) < 2) continue;
        if (t->ne[0] < 16 || t->ne[0] > 4096) continue;
        if (t->ne[1] < 1 || t->ne[1] > 128) continue;

        if (is_q_tensor_name(t->name)) {
            q_candidates.push_back({t, j, true});
        } else if (is_k_tensor_name(t->name)) {
            k_candidates.push_back({t, j, false});
        } else if (strstr(t->name, "Kcur")) {
            fprintf(stderr, "[spec-prefill] extract_qk: found Kcur tensor but rejected: '%s' shape=[%ld,%ld,%ld] dims=%d is_k=%d\n",
                    t->name, (long)t->ne[0], (long)t->ne[1], (long)t->ne[2],
                    ggml_n_dims(t), (int)is_k_tensor_name(t->name));
        }
    }

    filter_best_per_layer(q_candidates, n_layer);
    filter_best_per_layer(k_candidates, n_layer);

    fprintf(stderr, "[spec-prefill] extract_qk: found %zu Q candidates, %zu K candidates\n",
            q_candidates.size(), k_candidates.size());

    const int n_q_to_use = (int)std::min((size_t)n_layer, q_candidates.size());
    if (n_q_to_use == 0) {
        fprintf(stderr, "[spec-prefill] extract_qk: no Q tensors found in graph (n_nodes=%d), proceeding with empty tensors\n", n_nodes);
        return 0;
    }

    int extracted = 0;
    for (int qi = 0; qi < n_q_to_use; qi++) {
        const ggml_tensor * t = q_candidates[qi].tensor;
        if (!t) {
            fprintf(stderr, "[spec-prefill] extract_qk: Q[%d] tensor ptr is NULL (graph may be stale)\n", qi);
            continue;
        }

        llama_spec_q_tensor q_info;
        q_info.layer_idx = qi;
        q_info.n_embd_head_q = t->ne[0];
        q_info.n_head = t->ne[1];
        q_info.n_tokens = t->ne[2];
        q_info.tensor_ptr = t;

        const int64_t n_elements = (int64_t)t->ne[0] * t->ne[1] * t->ne[2];
        if (n_elements <= 0) {
            fprintf(stderr, "[spec-prefill] extract_qk: Q[%d] invalid n_elements=%ld\n", qi, (long)n_elements);
            continue;
        }

        q_info.data.resize(n_elements);

        // Copy tensor data from GPU to CPU, then convert to float if needed
        const size_t nbytes = ggml_nbytes(t);
        std::vector<uint8_t> cpu_buf(nbytes);
        ggml_backend_tensor_get(t, cpu_buf.data(), 0, nbytes);

        if (t->type == GGML_TYPE_F32) {
            std::memcpy(q_info.data.data(), cpu_buf.data(), nbytes);
        } else {
            const auto * type_traits = ggml_get_type_traits(t->type);
            if (type_traits && type_traits->to_float != NULL) {
                type_traits->to_float(cpu_buf.data(), q_info.data.data(), n_elements);
            } else {
                fprintf(stderr, "[spec-prefill] extract_qk: unsupported type %d for Q il=%d\n", t->type, qi);
                continue;
            }
        }

        ctx->q_tensors.push_back(q_info);
        extracted++;
    }

    if (extracted == 0) {
        fprintf(stderr, "[spec-prefill] extract_qk: failed to extract any Q tensor data (all tensors NULL or invalid)\n");
        return -1;
    }

    fprintf(stderr, "[spec-prefill] extract_qk: successfully extracted %d Q tensors\n", extracted);
    return 0;
}

int llama_spec_prefill_compute_attention(
    llama_spec_prefill_context * ctx,
    const llama_token * prompt_tokens,
    int n_prompt,
    const llama_token * lookahead_tokens,
    int n_lookahead
) {
    // POC NO-OP: Returns 0. Importance scoring uses perplexity in
    // compute_importance() instead. Full implementation would use GGML
    // softmax(Q @ K^T / sqrt(d_k)) with base-model Q/K tensors.
    if (!ctx || !prompt_tokens || !lookahead_tokens || n_prompt <= 0 || n_lookahead <= 0) {
        return -1;
    }
    return 0;
}

// Apply average pooling to smooth importance scores (like vLLM's avg_pool1d)
void llama_spec_prefill_apply_pooling(
    std::vector<float> & importance,
    int kernel_size
) {
    if (kernel_size <= 1 || importance.empty()) {
        return;
    }
    
    int n = (int)importance.size();
    int half_kernel = kernel_size / 2;
    std::vector<float> smoothed(n);
    
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        int count = 0;
        
        // Window centered at position i
        int start = std::max(0, i - half_kernel);
        int end = std::min(n - 1, i + half_kernel);
        
        for (int j = start; j <= end; j++) {
            sum += importance[j];
            count++;
        }
        
        smoothed[i] = sum / count;
    }
    
    importance = smoothed;
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

    token_importance.resize(n_prompt, 0.0f);

    // Perplexity-based scoring: importance[i+1] = -log P(token[i+1] | context[0..i])
    // Processed in chunks of CHUNK_SIZE to match the n_ubatch=512 graph reservation.
    // Logits are read immediately after each chunk before the next chunk overwrites the buffer.
    const int PERPLEXITY_THRESHOLD = 4096;
    const int CHUNK_SIZE = 512;

    if (n_prompt <= PERPLEXITY_THRESHOLD && ctx->ctx_spec) {
        llama_memory_clear(llama_get_memory(ctx->ctx_spec), false);

        const llama_model * model = llama_get_model(ctx->ctx_spec);
        const llama_vocab * vocab = llama_model_get_vocab(model);
        int n_vocab = llama_vocab_n_tokens(vocab);

        token_importance[0] = 1.0f;  // BOS / first token always kept

        bool perplexity_ok = true;
        int chunk_start = 0;
        while (chunk_start < n_prompt && perplexity_ok) {
            int chunk_end = std::min(chunk_start + CHUNK_SIZE, n_prompt);
            int chunk_len = chunk_end - chunk_start;

            llama_batch batch = llama_batch_init(chunk_len, 0, 1);
            batch.n_tokens = 0;
            for (int i = chunk_start; i < chunk_end; i++) {
                batch.token[batch.n_tokens]     = prompt_tokens[i];
                batch.pos[batch.n_tokens]       = i;
                batch.n_seq_id[batch.n_tokens]  = 1;
                batch.seq_id[batch.n_tokens][0] = 0;
                batch.logits[batch.n_tokens]    = true;
                batch.n_tokens++;
            }

            if (llama_decode(ctx->ctx_spec, batch) != 0) {
                llama_batch_free(batch);
                perplexity_ok = false;
                break;
            }

            // Read logits for each token in the chunk immediately (before next chunk overwrites)
            for (int k = 0; k < chunk_len; k++) {
                int global_pos = chunk_start + k;
                if (global_pos + 1 >= n_prompt) break;  // no next token to predict
                float * logits = llama_get_logits_ith(ctx->ctx_spec, k);
                float max_logit = logits[0];
                for (int j = 1; j < n_vocab; j++)
                    if (logits[j] > max_logit) max_logit = logits[j];
                float sum_exp = 0.0f;
                for (int j = 0; j < n_vocab; j++)
                    sum_exp += expf(logits[j] - max_logit);
                llama_token next_tok = prompt_tokens[global_pos + 1];
                float log_prob = (logits[next_tok] - max_logit) - logf(sum_exp);
                token_importance[global_pos + 1] = -log_prob;  // surprise >= 0
            }

            llama_batch_free(batch);
            chunk_start = chunk_end;
        }

        if (perplexity_ok) {
            // Normalize to [0, 1]
            float min_imp = token_importance[0], max_imp = token_importance[0];
            for (float v : token_importance) {
                if (v < min_imp) min_imp = v;
                if (v > max_imp) max_imp = v;
            }
            float range = max_imp - min_imp;
            if (range > 1e-6f)
                for (float & v : token_importance) v = (v - min_imp) / range;

            if (ctx->params.pool_kernel_size > 1)
                llama_spec_prefill_apply_pooling(token_importance, ctx->params.pool_kernel_size);

            return 0;
        }
        // Decode failed — clear corrupted KV state then fall through to entropy proxy
        llama_memory_clear(llama_get_memory(ctx->ctx_spec), false);
    }

    // --- Entropy proxy fallback (long prompts or decode failure) ---
    if (ctx->lookahead_stats.empty()) {
        for (int i = 0; i < n_prompt; i++)
            token_importance[i] = 0.1f + 0.9f * (float)i / (n_prompt - 1);
        return 0;
    }

    float max_entropy = 1e-6f;
    for (const auto & stat : ctx->lookahead_stats)
        if (stat.entropy > max_entropy) max_entropy = stat.entropy;

    for (int i = 0; i < n_prompt; i++) {
        float importance = 0.0f;
        for (const auto & stat : ctx->lookahead_stats) {
            int distance = stat.position - i;
            if (distance > 0) {
                float confidence = 1.0f - (stat.entropy / max_entropy);
                importance += confidence / (1.0f + sqrtf((float)distance));
            }
        }
        token_importance[i] = 0.1f + 0.9f * importance;
    }

    float min_imp = token_importance[0], max_imp = token_importance[0];
    for (int i = 1; i < n_prompt; i++) {
        if (token_importance[i] < min_imp) min_imp = token_importance[i];
        if (token_importance[i] > max_imp) max_imp = token_importance[i];
    }
    float range = max_imp - min_imp;
    if (range > 1e-6f)
        for (int i = 0; i < n_prompt; i++)
            token_importance[i] = 0.1f + 0.9f * (token_importance[i] - min_imp) / range;

    if (ctx->params.pool_kernel_size > 1)
        llama_spec_prefill_apply_pooling(token_importance, ctx->params.pool_kernel_size);

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

int llama_spec_prefill_filter_tokens_chunked(
    llama_spec_prefill_context * ctx,
    const llama_token * prompt_tokens,
    const int * prompt_positions,
    int n_prompt,
    const std::vector<float> & token_importance,
    float keep_ratio,
    int chunk_size,
    std::vector<llama_token> & filtered_tokens,
    std::vector<int> & filtered_positions
) {
    // Chunk-based selection like vLLM's _get_kept_indices_from_token_importance
    // 1. Split tokens into chunks
    // 2. Compute average importance per chunk
    // 3. Keep top-k chunks
    // 4. Include all tokens from kept chunks

    if (!ctx || !prompt_tokens || !prompt_positions || n_prompt <= 0) {
        return -1;
    }

    if (token_importance.size() != (size_t)n_prompt) {
        return -1;
    }

    if (chunk_size <= 0) {
        chunk_size = 32;
    }

    // Calculate number of chunks
    int n_chunks = (n_prompt + chunk_size - 1) / chunk_size;

    // Compute average importance for each chunk
    std::vector<std::pair<float, int>> chunk_importance;
    chunk_importance.reserve(n_chunks);

    for (int c = 0; c < n_chunks; c++) {
        int start = c * chunk_size;
        int end = std::min(start + chunk_size, n_prompt);

        float avg = 0.0f;
        for (int i = start; i < end; i++) {
            avg += token_importance[i];
        }
        avg /= (end - start);

        chunk_importance.push_back({avg, c});
    }

    // Sort chunks by importance (descending)
    std::sort(chunk_importance.begin(), chunk_importance.end(),
        [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
            return a.first > b.first;
        });

    // Calculate how many chunks to keep
    int n_keep_chunks = (int)std::ceil(n_chunks * keep_ratio);
    if (n_keep_chunks < 1) n_keep_chunks = 1;
    if (n_keep_chunks > n_chunks) n_keep_chunks = n_chunks;

    // Calculate target number of tokens to keep (minimum guarantee)
    int n_target = (int)(n_prompt * keep_ratio);
    if (n_target < 1) n_target = 1;

    std::vector<int> kept_chunk_indices;
    kept_chunk_indices.reserve(n_keep_chunks);
    for (int i = 0; i < n_keep_chunks; i++) {
        kept_chunk_indices.push_back(chunk_importance[i].second);
    }

    std::sort(kept_chunk_indices.begin(), kept_chunk_indices.end());

    filtered_tokens.clear();
    filtered_positions.clear();
    std::vector<bool> token_kept(n_prompt, false);
    int tokens_kept = 0;

    for (int c : kept_chunk_indices) {
        int start = c * chunk_size;
        int end = std::min(start + chunk_size, n_prompt);

        for (int i = start; i < end; i++) {
            filtered_tokens.push_back(prompt_tokens[i]);
            filtered_positions.push_back(prompt_positions[i]);
            token_kept[i] = true;
            tokens_kept++;
        }
    }

    // Fallback: if chunks don't provide enough tokens, fill from remaining high-importance tokens
    if (tokens_kept < n_target) {
        std::vector<std::pair<float, int>> remaining;
        remaining.reserve(n_prompt - tokens_kept);
        for (int i = 0; i < n_prompt; i++) {
            if (!token_kept[i]) {
                remaining.push_back({token_importance[i], i});
            }
        }
        std::sort(remaining.begin(), remaining.end(),
            [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                return a.first > b.first;
            });

        int fill_count = n_target - tokens_kept;
        std::vector<int> fill_indices;
        fill_indices.reserve(fill_count);
        for (int i = 0; i < fill_count && i < (int)remaining.size(); i++) {
            fill_indices.push_back(remaining[i].second);
        }

        std::sort(fill_indices.begin(), fill_indices.end());

        std::vector<bool> is_fill(n_prompt, false);
        for (int idx : fill_indices) {
            is_fill[idx] = true;
        }

        filtered_tokens.clear();
        filtered_positions.clear();
        for (int i = 0; i < n_prompt; i++) {
            if (token_kept[i] || is_fill[i]) {
                filtered_tokens.push_back(prompt_tokens[i]);
                filtered_positions.push_back(prompt_positions[i]);
            }
        }
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
    llama_memory_clear(llama_get_memory(ctx->ctx_base), false);

    // Re-index positions to be contiguous [0..n_filtered-1].
    // llama.cpp's llama_decode requires contiguous positions within a batch.
    // The original non-contiguous positions (e.g., [0,1,2,3,4,5,6,7,16,17,...])
    // are not supported — llama_decode will fail with "positions are not
    // continuous" or GGML_ASSERT(n_tokens_all <= cparams.n_batch).
    //
    // Re-indexing is safe because:
    // 1. The spec model already generated lookahead tokens using the full
    //    prompt (generate_lookahead processes positions 0..n_prompt-1).
    // 2. The base model only needs the KV cache entries for the filtered
    //    tokens to continue generation — relative ordering is preserved.
    // 3. RoPE at re-indexed positions gives slightly different absolute
    //    embeddings but the same relative distances, which is fine for
    //    continued generation.
    llama_batch batch = llama_batch_init(n_filtered, 0, 1);
    batch.n_tokens = 0;

    for (int i = 0; i < n_filtered; i++) {
        batch.token[batch.n_tokens] = filtered_tokens[i];
        batch.pos[batch.n_tokens] = i;  // re-indexed to [0..n_filtered-1]
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.seq_id[batch.n_tokens][0] = 0;
        batch.logits[batch.n_tokens] = (i == n_filtered - 1);
        batch.n_tokens++;
    }

    int result = llama_decode(ctx->ctx_base, batch);
    if (result != 0) {
        fprintf(stderr, "[spec-prefill] process_base decode failed n_filtered=%d\n", n_filtered);
    }
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
    
    // Use parameters from context if available, otherwise use provided values
    int actual_lookahead = (n_lookahead > 0) ? n_lookahead : ctx->params.n_lookahead;
    float actual_keep_ratio = (keep_ratio > 0) ? keep_ratio : ctx->params.keep_ratio;

    // Step 1: Generate lookahead tokens
    std::vector<llama_token> lookahead_tokens(actual_lookahead);
    int n_generated = llama_spec_prefill_generate_lookahead(
        ctx, prompt_tokens, n_prompt, lookahead_tokens.data(), actual_lookahead
    );
    if (n_generated <= 0) {
        fprintf(stderr, "[spec-prefill] step1 generate_lookahead failed n_generated=%d n_prompt=%d\n", n_generated, n_prompt);
        return -1;
    }

    // Step 2: Extract Q/K tensors (simplified for POC)
    if (llama_spec_prefill_extract_qk(ctx, lookahead_tokens.data(), n_generated) != 0) {
        fprintf(stderr, "[spec-prefill] step2 extract_qk failed\n");
        return -1;
    }

    // Step 3: Compute attention scores — NO-OP in POC.
    // Importance scoring uses perplexity-based proxy in compute_importance().
    // Full implementation would compute softmax(Q @ K^T / sqrt(d_k)) with
    // base-model Q/K tensors.  llama_spec_prefill_compute_attention() is a
    // stub that always returns 0.
    llama_spec_prefill_compute_attention(
        ctx, prompt_tokens, n_prompt, lookahead_tokens.data(), n_generated
    );  // no-op, return value ignored

    // Step 4: Compute token importance (with pooling applied inside)
    std::vector<float> token_importance;
    if (llama_spec_prefill_compute_importance(ctx, prompt_tokens, n_prompt, token_importance) != 0) {
        fprintf(stderr, "[spec-prefill] step4 compute_importance failed\n");
        return -1;
    }

    // Step 5: Filter tokens (choose chunked or token-based)
    std::vector<int> prompt_positions(n_prompt);
    for (int i = 0; i < n_prompt; i++) {
        prompt_positions[i] = i;
    }

    std::vector<llama_token> filtered_tokens;
    std::vector<int> filtered_positions;
    
    if (ctx->params.use_chunking) {
        // Chunk-based selection like vLLM
        if (llama_spec_prefill_filter_tokens_chunked(
                ctx, prompt_tokens, prompt_positions.data(), n_prompt,
                token_importance, actual_keep_ratio, ctx->params.chunk_size,
                filtered_tokens, filtered_positions) != 0) {
            return -1;
        }
    } else {
        // Token-based selection
        if (llama_spec_prefill_filter_tokens(
                ctx, prompt_tokens, prompt_positions.data(), n_prompt,
                token_importance, actual_keep_ratio,
                filtered_tokens, filtered_positions) != 0) {
            return -1;
        }
    }

    // Step 6: Process with base model
    if (llama_spec_prefill_process_base(
            ctx, filtered_tokens.data(), filtered_positions.data(),
            (int)filtered_tokens.size()) != 0) {
        return -1;
    }

    int n_kept = (int)filtered_tokens.size();

    // Write parity trace if dump path is set
    if (!ctx->dump_path.empty()) {
        std::ofstream f(ctx->dump_path, std::ios::app);
        if (f) {
            f << "{\"n_prompt\":" << n_prompt
              << ",\"n_kept\":" << n_kept
              << ",\"keep_ratio\":" << actual_keep_ratio
              << ",\"n_lookahead\":" << actual_lookahead
              << "}\n";
        }
    }

    return n_kept;
}
