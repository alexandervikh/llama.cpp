#include "llama-self-layer-prefill.h"
#include "llama-spec-prefill.h"
#include "llama-arch.h"
#include "llama-context.h"
#include "llama-model.h"
#include "llama-memory.h"
#include "llama.h"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ============================================================================
// Look-ahead Token Generation for SpecPrefill-style scoring
// Uses early exit mechanism: runs only n_early_layers but applies output_norm
// and lm_head to get logits for speculative token generation.
// ============================================================================

// Generate look-ahead tokens using early exit (layers 0..n_early_layers with lm_head)
// Returns the generated tokens in `out_tokens`
static int generate_lookahead_tokens(
    llama_context * ctx,
    const llama_token * prompt_tokens,
    int n_prompt,
    int n_early_layers,        // number of early layers to use (e.g., 4)
    int n_lookahead,           // number of tokens to generate (e.g., 8)
    std::vector<llama_token> & out_tokens,
    float temperature = 0.f    // 0 = greedy (unused for now, always greedy)
) {
    GGML_UNUSED(temperature);
    
    const llama_model * model = llama_get_model(ctx);
    if (!model) return -1;
    
    const int n_vocab = (int)llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int n_layer_total = (int)llama_model_n_layer(model);
    
    // Validate n_early_layers
    if (n_early_layers <= 0 || n_early_layers > n_layer_total) {
        fprintf(stderr, "[lookahead] invalid n_early_layers=%d, n_layer=%d\n", 
                n_early_layers, n_layer_total);
        return -1;
    }
    
    out_tokens.clear();
    out_tokens.reserve((size_t)n_lookahead);
    
    // Clear KV cache
    llama_memory_clear(llama_get_memory(ctx), false);
    
    // Decode prompt using early exit (layers 0..n_early_layers with lm_head)
    llama_batch batch = llama_batch_get_one((llama_token *)prompt_tokens, n_prompt);
    
    // Use decode_partial with early_exit=true to get logits from early layers
    int r = ctx->decode_partial(batch, 0, n_early_layers, /*early_exit=*/true);
    
    if (r != 0) {
        fprintf(stderr, "[lookahead] initial decode_partial failed: %d\n", r);
        return -1;
    }
    
    // Generate look-ahead tokens one by one
    for (int gen = 0; gen < n_lookahead; gen++) {
        // Get logits for the last token (use -1 for last position)
        float * logits = llama_get_logits_ith(ctx, -1);
        if (!logits) {
            fprintf(stderr, "[lookahead] failed to get logits at gen=%d\n", gen);
            return -1;
        }
        
        // Greedy sampling: find argmax
        llama_token next_token = 0;
        float max_logit = logits[0];
        for (int v = 1; v < n_vocab; v++) {
            if (logits[v] > max_logit) {
                max_logit = logits[v];
                next_token = v;
            }
        }
        
        out_tokens.push_back(next_token);
        
        // Decode next token using early exit (if not last iteration)
        if (gen < n_lookahead - 1) {
            llama_batch single = llama_batch_get_one(&next_token, 1);
            
            r = ctx->decode_partial(single, 0, n_early_layers, /*early_exit=*/true);
            
            if (r != 0) {
                fprintf(stderr, "[lookahead] decode_partial token %d failed: %d\n", gen, r);
                return -1;
            }
        }
    }
    
    return 0;
}

// Graph tensor names are "Qcur-0", "Kcur-0"; with multi-backend copies they look like
// "CUDA0#Qcur-5#1" — match the stable infix, not the first character.
// Skip fused/view variants: "(reshaped)", "(view)", "(permuted)" have wrong layouts.
static bool is_q_tensor_name(const char * name) {
    return name && strstr(name, "Qcur-") && !strstr(name, "(");
}

static bool is_k_tensor_name(const char * name) {
    return name && strstr(name, "Kcur-") && !strstr(name, "(");
}

static int extract_layer_index(const char * name) {
    if (!name) {
        return -1;
    }
    const char * q = strstr(name, "Qcur-");
    const char * k = strstr(name, "Kcur-");
    const char * p = q ? q + 5 : (k ? k + 5 : nullptr);
    if (!p || (*p < '0' || *p > '9')) {
        return -1;
    }
    return int(atoi(p));
}

struct slp_found_tensor {
    const ggml_tensor * tensor = nullptr;
    int graph_idx = -1;
};

static void filter_best_per_layer(std::vector<slp_found_tensor> & best, const std::vector<slp_found_tensor> & candidates, int n_layer) {
    best.assign(n_layer, {});
    for (const auto & c : candidates) {
        int li = extract_layer_index(c.tensor->name);
        if (li >= 0 && li < n_layer) {
            if (!best[li].tensor || c.graph_idx > best[li].graph_idx) {
                best[li] = c;
            }
        }
    }
}

static int tensor_to_float(const ggml_tensor * t, std::vector<float> & out) {
    const int64_t n_elements = ggml_nelements(t);
    if (n_elements <= 0) return -1;
    out.resize((size_t) n_elements);
    const size_t nbytes = ggml_nbytes(t);
    std::vector<uint8_t> cpu_buf(nbytes);
    ggml_backend_tensor_get(t, cpu_buf.data(), 0, nbytes);
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(out.data(), cpu_buf.data(), nbytes);
        return 0;
    }
    const auto * traits = ggml_get_type_traits(t->type);
    if (!traits || !traits->to_float) return -1;
    traits->to_float(cpu_buf.data(), out.data(), n_elements);
    return 0;
}

struct slp_layer_qk {
    int layer = 0;
    int64_t d = 0;
    int64_t n_head = 0;
    int64_t n_kv = 0;
    int64_t n_tok = 0;
    std::vector<float> q;
    std::vector<float> k;
};

static int extract_qk_all_layers(llama_context & lctx, std::vector<slp_layer_qk> & layers) {
    lctx.synchronize();
    layers.clear();

    auto * gf_res = lctx.get_gf_res_prev();
    if (!gf_res || !gf_res->get_gf()) {
        fprintf(stderr, "[self-layer-prefill] extract_qk: no graph\n");
        return -1;
    }
    ggml_cgraph * gf = gf_res->get_gf();
    const int n_layer = (int) llama_model_n_layer(&lctx.get_model());
    const int n_nodes = ggml_graph_n_nodes(gf);

    std::vector<slp_found_tensor> q_cand;
    std::vector<slp_found_tensor> k_cand;
    for (int j = 0; j < n_nodes; j++) {
        ggml_tensor * t = ggml_graph_node(gf, j);
        if (!t || !t->name[0]) continue;
        if (ggml_n_dims(t) != 3) {
            continue;
        }
        if (t->ne[0] < 8 || t->ne[0] > 4096) continue;
        if (t->ne[1] < 1 || t->ne[1] > 256) continue;
        if (t->ne[2] < 1 || t->ne[2] > 131072) continue;
        if (is_q_tensor_name(t->name)) {
            q_cand.push_back({t, j});
        } else if (is_k_tensor_name(t->name)) {
            k_cand.push_back({t, j});
        }
    }

    std::vector<slp_found_tensor> q_best;
    std::vector<slp_found_tensor> k_best;
    filter_best_per_layer(q_best, q_cand, n_layer);
    filter_best_per_layer(k_best, k_cand, n_layer);

    for (int il = 0; il < n_layer; il++) {
        if (!q_best[il].tensor || !k_best[il].tensor) {
            continue;
        }
        const ggml_tensor * tq = q_best[il].tensor;
        const ggml_tensor * tk = k_best[il].tensor;
        if (tq->ne[2] != tk->ne[2]) {
            continue;
        }

        slp_layer_qk L;
        L.layer = il;
        L.d = tq->ne[0];
        L.n_head = tq->ne[1];
        L.n_kv = tk->ne[1];
        L.n_tok = tq->ne[2];
        if (L.n_head % L.n_kv != 0) {
            continue;
        }
        if (tensor_to_float(tq, L.q) != 0 || tensor_to_float(tk, L.k) != 0) {
            continue;
        }
        layers.push_back(std::move(L));
    }

    if (layers.empty()) {
        fprintf(stderr, "[self-layer-prefill] extract_qk: no usable Q/K pairs\n");
        return -1;
    }
    return 0;
}

static inline float q_at(const std::vector<float> & q, int64_t d, int64_t h, int64_t t, int64_t ne0, int64_t ne1) {
    return q[(size_t)(d + ne0 * (h + ne1 * t))];
}

static inline float k_at(const std::vector<float> & k, int64_t d, int64_t kh, int64_t t, int64_t ne0, int64_t ne1) {
    return k[(size_t)(d + ne0 * (kh + ne1 * t))];
}

// Phase 1 cheap scoring: per token j, compute importance from query position(s) to key position j.
// Two modes:
// - use_max_heads=false (original): SUM over heads
// - use_max_heads=true (SpecPrefill): MAX over heads
static void layer_query_importance(const slp_layer_qk & L, int qi, std::vector<float> & imp, bool use_max_heads = false) {
    const int64_t n_tok = L.n_tok;
    const int n_head = (int) L.n_head;
    const int n_kv = (int) L.n_kv;
    const int d = (int) L.d;
    const int group = n_head / n_kv;
    const float scale = 1.f / sqrtf(float(d));

    if (use_max_heads) {
        // MAX over heads (SpecPrefill style): pick the most salient head per token
        imp.assign((size_t) n_tok, -1e30f);
        for (int j = 0; j < (int) n_tok; j++) {
            for (int h = 0; h < n_head; h++) {
                const int kh = h / group;
                float t = 0;
                for (int di = 0; di < d; di++) {
                    t += q_at(L.q, di, h, qi, L.d, L.n_head)
                       * k_at(L.k, di, kh, j, L.d, L.n_kv);
                }
                t *= scale;
                if (t > imp[(size_t) j]) {
                    imp[(size_t) j] = t;
                }
            }
        }
    } else {
        // SUM over heads (original): average contribution across all heads
        std::vector<float> q_sum((size_t)(d * n_kv), 0.f);
        for (int kh = 0; kh < n_kv; kh++) {
            for (int hg = 0; hg < group; hg++) {
                const int h = kh * group + hg;
                for (int di = 0; di < d; di++) {
                    q_sum[(size_t)(di + d * kh)] +=
                        q_at(L.q, di, h, qi, L.d, L.n_head);
                }
            }
        }

        imp.assign((size_t) n_tok, 0.f);
        for (int j = 0; j < (int) n_tok; j++) {
            float s = 0;
            for (int kh = 0; kh < n_kv; kh++) {
                float t = 0;
                for (int di = 0; di < d; di++) {
                    t += q_sum[(size_t)(di + d * kh)]
                       * k_at(L.k, di, kh, j, L.d, L.n_kv);
                }
                s += t;
            }
            imp[(size_t) j] = s * scale;
        }
    }
}

// Wrapper for backward compatibility - uses single query position (last token)
static void layer_last_query_importance(const slp_layer_qk & L, int qi_last, std::vector<float> & imp, bool use_max_heads = false) {
    layer_query_importance(L, qi_last, imp, use_max_heads);
}

static void accumulate_range(const std::vector<slp_layer_qk> & layers, int il0, int il1, int qi_last, std::vector<float> & out, bool use_max = false) {
    out.clear();
    int count = 0;
    for (const auto & L : layers) {
        if (L.layer < il0 || L.layer >= il1) {
            continue;
        }
        std::vector<float> imp;
        layer_last_query_importance(L, qi_last, imp, use_max);
        if (out.empty()) {
            out = std::move(imp);
        } else {
            for (size_t i = 0; i < out.size(); i++) {
                if (use_max) {
                    // MAX over layers (SpecPrefill style)
                    if (imp[i] > out[i]) out[i] = imp[i];
                } else {
                    // SUM for later averaging (original)
                    out[i] += imp[i];
                }
            }
        }
        count++;
    }
    if (!use_max && count > 1) {
        for (float & v : out) {
            v /= float(count);
        }
    }
}

// SpecPrefill-style aggregation with multiple query positions:
// 1. MAX over heads (per layer, per query)
// 2. MAX over layers (per query)
// 3. MEAN over query positions
static void accumulate_range_multi_query(
    const std::vector<slp_layer_qk> & layers,
    int il0, int il1,
    const std::vector<int> & query_positions,  // multiple query positions
    std::vector<float> & out
) {
    out.clear();
    if (layers.empty() || query_positions.empty()) return;

    const size_t n_tok = layers[0].n_tok;
    
    // For each query position, compute max-max (over heads, over layers)
    std::vector<std::vector<float>> per_query_scores;
    per_query_scores.reserve(query_positions.size());
    
    for (int qi : query_positions) {
        if (qi < 0 || qi >= (int)n_tok) continue;
        
        std::vector<float> query_max(n_tok, -1e30f);  // max over layers for this query
        
        for (const auto & L : layers) {
            if (L.layer < il0 || L.layer >= il1) continue;
            
            std::vector<float> layer_imp;
            layer_query_importance(L, qi, layer_imp, true);  // MAX over heads
            
            // MAX over layers
            for (size_t j = 0; j < n_tok && j < layer_imp.size(); j++) {
                if (layer_imp[j] > query_max[j]) {
                    query_max[j] = layer_imp[j];
                }
            }
        }
        per_query_scores.push_back(std::move(query_max));
    }
    
    if (per_query_scores.empty()) return;
    
    // MEAN over query positions
    out.assign(n_tok, 0.f);
    for (const auto & qs : per_query_scores) {
        for (size_t j = 0; j < n_tok && j < qs.size(); j++) {
            out[j] += qs[j];
        }
    }
    const float inv_n = 1.f / float(per_query_scores.size());
    for (float & v : out) {
        v *= inv_n;
    }
}

static float pearson(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size() || a.size() < 2) {
        return 0.f;
    }
    double sa = 0, sb = 0;
    const size_t n = a.size();
    for (size_t i = 0; i < n; i++) {
        sa += a[i];
        sb += b[i];
    }
    sa /= (double) n;
    sb /= (double) n;
    double num = 0, da = 0, db = 0;
    for (size_t i = 0; i < n; i++) {
        const double xa = a[i] - sa;
        const double xb = b[i] - sb;
        num += xa * xb;
        da += xa * xa;
        db += xb * xb;
    }
    const double eps = 1e-20;
    if (da <= eps && db <= eps) {
        // both ~constant: 1 if same level, else 0
        return fabs(sa - sb) < 1e-6 * std::fmax(fabs(sa), 1.0) ? 1.f : 0.f;
    }
    if (da <= eps || db <= eps) {
        return 0.f;
    }
    const float r = float(num / sqrt(da * db));
    return std::isfinite(r) ? r : 0.f;
}

static int decode_full_prompt(llama_context * ctx, const llama_token * tokens, int n_prompt) {
    llama_memory_clear(llama_get_memory(ctx), false);
    llama_batch batch = llama_batch_init(n_prompt, 0, 1);
    batch.n_tokens = 0;
    for (int i = 0; i < n_prompt; i++) {
        batch.token[batch.n_tokens] = tokens[i];
        batch.pos[batch.n_tokens] = i;
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.seq_id[batch.n_tokens][0] = 0;
        batch.logits[batch.n_tokens] = false;
        batch.n_tokens++;
    }
    if (n_prompt > 0) {
        batch.logits[n_prompt - 1] = true;
    }
    const int r = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return r;
}

int llama_self_layer_prefill_attention_profiles(
    llama_context * ctx,
    const llama_token * prompt_tokens,
    int n_prompt,
    const llama_self_layer_prefill_params * params,
    std::vector<float> & out_shallow,
    std::vector<float> & out_deep,
    std::vector<float> & out_full,
    float * pearson_shallow_vs_full,
    float * pearson_shallow_vs_deep) {
    if (!ctx || !prompt_tokens || !params || n_prompt <= 0) {
        return -1;
    }
    if (n_prompt > (int) llama_n_batch(ctx)) {
        return -2;
    }

    llama_self_layer_prefill_params P = *params;
    if (P.n_early_layers < 1) {
        P.n_early_layers = 1;
    }

    if (decode_full_prompt(ctx, prompt_tokens, n_prompt) != 0) {
        return -1;
    }

    std::vector<slp_layer_qk> layers;
    if (extract_qk_all_layers(*ctx, layers) != 0) {
        return -1;
    }

    const int n_layer = (int) llama_model_n_layer(llama_get_model(ctx));
    if (P.n_early_layers > n_layer) {
        P.n_early_layers = n_layer;
    }

    const int qi_last = n_prompt - 1;
    const bool use_max = P.use_max_aggregation;
    accumulate_range(layers, 0, P.n_early_layers, qi_last, out_shallow, use_max);
    accumulate_range(layers, P.n_early_layers, n_layer, qi_last, out_deep, use_max);
    accumulate_range(layers, 0, n_layer, qi_last, out_full, use_max);

    if (out_shallow.empty() || out_deep.empty() || out_full.empty()) {
        return -1;
    }

    if (pearson_shallow_vs_full) {
        *pearson_shallow_vs_full = pearson(out_shallow, out_full);
    }
    if (pearson_shallow_vs_deep) {
        *pearson_shallow_vs_deep = pearson(out_shallow, out_deep);
    }
    return 0;
}

static int filter_topk(
    const llama_token * prompt_tokens, const int * pos, int n_prompt,
    const std::vector<float> & imp, float keep_ratio,
    bool use_chunk, int chunk_size,
    std::vector<llama_token> & ft, std::vector<int> & fp) {
    // filter_* require non-null ctx but only check pointer / sizes
    llama_spec_prefill_context dummy{};
    if (use_chunk) {
        return llama_spec_prefill_filter_tokens_chunked(
            &dummy, prompt_tokens, pos, n_prompt, imp, keep_ratio, chunk_size, ft, fp);
    }
    return llama_spec_prefill_filter_tokens(
        &dummy, prompt_tokens, pos, n_prompt, imp, keep_ratio, ft, fp);
}

static int process_filtered_decode(llama_context * ctx, const std::vector<llama_token> & ft) {
    llama_memory_clear(llama_get_memory(ctx), false);
    const int n = (int) ft.size();
    llama_batch batch = llama_batch_init(n, 0, 1);
    batch.n_tokens = 0;
    for (int i = 0; i < n; i++) {
        batch.token[batch.n_tokens] = ft[i];
        batch.pos[batch.n_tokens] = i;
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.seq_id[batch.n_tokens][0] = 0;
        batch.logits[batch.n_tokens] = (i == n - 1);
        batch.n_tokens++;
    }
    const int r = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return r;
}

// Phase 1 (Option C): single-decode TTFT-neutral path. Runs one full prefill, scores
// with the cheap shallow-only proxy, then prunes non-kept positions from the KV cache
// via llama_memory_seq_rm. The last prompt position is always kept (generation needs
// its hidden state). Positions of kept tokens are NOT remapped, so RoPE phases stay
// consistent with what the model already wrote into KV.
int llama_self_layer_prefill_with_kv_prune(
    llama_context * ctx,
    const llama_token * prompt_tokens,
    int n_prompt,
    const llama_self_layer_prefill_params * params) {
    if (!ctx || !prompt_tokens || !params || n_prompt <= 0) {
        return -1;
    }
    if (n_prompt > (int) llama_n_batch(ctx)) {
        return -2;
    }

    llama_self_layer_prefill_params P = *params;
    if (P.n_early_layers < 1) P.n_early_layers = 1;

    if (decode_full_prompt(ctx, prompt_tokens, n_prompt) != 0) {
        return -1;
    }

    std::vector<slp_layer_qk> layers;
    if (extract_qk_all_layers(*ctx, layers) != 0) {
        return -1;
    }
    const int n_layer = (int) llama_model_n_layer(llama_get_model(ctx));
    if (P.n_early_layers > n_layer) P.n_early_layers = n_layer;

    std::vector<float> imp;
    accumulate_range(layers, 0, P.n_early_layers, n_prompt - 1, imp, P.use_max_aggregation);
    if (imp.empty()) {
        return -1;
    }
    if (P.pool_kernel_size > 1) {
        llama_spec_prefill_apply_pooling(imp, P.pool_kernel_size);
    }

    int n_keep = (int) (n_prompt * P.keep_ratio);
    if (n_keep < 1) n_keep = 1;
    if (n_keep > n_prompt) n_keep = n_prompt;

    std::vector<std::pair<float, int>> ranked;
    ranked.reserve((size_t) n_prompt);
    for (int i = 0; i < n_prompt; i++) {
        ranked.push_back({imp[(size_t) i], i});
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const std::pair<float,int> & a, const std::pair<float,int> & b) {
                  return a.first > b.first;
              });

    std::vector<bool> keep((size_t) n_prompt, false);
    for (int i = 0; i < n_keep; i++) {
        keep[(size_t) ranked[(size_t) i].second] = true;
    }
    keep[(size_t)(n_prompt - 1)] = true;

    auto * mem = llama_get_memory(ctx);
    int n_kept = 0;
    int gap_lo = -1;
    for (int i = 0; i <= n_prompt; i++) {
        const bool is_keep = (i < n_prompt) && keep[(size_t) i];
        if (i < n_prompt && is_keep) n_kept++;
        if (!is_keep) {
            if (gap_lo < 0) gap_lo = i;
        } else if (gap_lo >= 0) {
            llama_memory_seq_rm(mem, /*seq_id*/ 0, /*p0*/ gap_lo, /*p1*/ i);
            gap_lo = -1;
        }
    }
    return n_kept;
}

// SpecPrefill-style scoring with generated look-ahead tokens:
// 1. Generate N look-ahead tokens using early exit (n_early layers + lm_head)
// 2. Decode prompt + look-ahead tokens together
// 3. Compute attention from each look-ahead token to all prompt positions
// 4. Aggregate: MAX over heads, MAX over layers, MEAN over look-ahead tokens
static int self_layer_score_with_lookahead(
    llama_context * ctx,
    const llama_token * prompt_tokens,
    int n_prompt,
    int n_early_layers,
    int n_lookahead,           // number of look-ahead tokens to generate (e.g., 8)
    std::vector<float> & out_imp,
    float lookahead_temp = 0.f // 0 = greedy
) {
    if (!ctx || !prompt_tokens || n_prompt <= 0 || n_early_layers < 1 || n_lookahead < 1) {
        return -1;
    }
    
    // Step 1: Generate look-ahead tokens using early exit
    std::vector<llama_token> lookahead_tokens;
    int gen_r = generate_lookahead_tokens(ctx, prompt_tokens, n_prompt, n_early_layers, 
                                           n_lookahead, lookahead_tokens, lookahead_temp);
    if (gen_r != 0 || (int)lookahead_tokens.size() != n_lookahead) {
        fprintf(stderr, "[specprefill] failed to generate look-ahead tokens: %d\n", gen_r);
        return -1;
    }
    
    // Step 2: Decode prompt + look-ahead tokens together using early layers
    const int n_total = n_prompt + n_lookahead;
    if (n_total > (int)llama_n_batch(ctx)) {
        fprintf(stderr, "[specprefill] prompt + lookahead (%d) exceeds batch size\n", n_total);
        return -2;
    }
    
    llama_memory_clear(llama_get_memory(ctx), false);
    
    llama_batch batch = llama_batch_init(n_total, 0, 1);
    batch.n_tokens = 0;
    
    // Add prompt tokens
    for (int i = 0; i < n_prompt; i++) {
        batch.token[batch.n_tokens] = prompt_tokens[i];
        batch.pos[batch.n_tokens] = i;
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.seq_id[batch.n_tokens][0] = 0;
        batch.logits[batch.n_tokens] = false;
        batch.n_tokens++;
    }
    
    // Add look-ahead tokens (positioned after prompt)
    for (int i = 0; i < n_lookahead; i++) {
        batch.token[batch.n_tokens] = lookahead_tokens[(size_t)i];
        batch.pos[batch.n_tokens] = n_prompt + i;
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.seq_id[batch.n_tokens][0] = 0;
        batch.logits[batch.n_tokens] = false;
        batch.n_tokens++;
    }
    
    const int r = ctx->decode_partial(batch, 0, n_early_layers);
    llama_batch_free(batch);
    
    if (r != 0) {
        fprintf(stderr, "[specprefill] decode failed: %d\n", r);
        return -1;
    }
    
    // Step 3: Extract Q/K from the decode and compute importance
    std::vector<slp_layer_qk> layers;
    if (extract_qk_all_layers(*ctx, layers) != 0) {
        fprintf(stderr, "[specprefill] failed to extract Q/K tensors\n");
        return -1;
    }
    
    // Step 4: For each look-ahead token, compute attention to prompt tokens
    // Then average (MEAN over look-ahead tokens)
    // Query positions are: n_prompt, n_prompt+1, ..., n_prompt+n_lookahead-1
    std::vector<int> query_positions;
    for (int i = 0; i < n_lookahead; i++) {
        query_positions.push_back(n_prompt + i);
    }
    
    // Use max-max-mean aggregation
    accumulate_range_multi_query(layers, 0, n_early_layers, query_positions, out_imp);
    
    // Only return importance for prompt tokens (first n_prompt)
    if ((int)out_imp.size() > n_prompt) {
        out_imp.resize((size_t)n_prompt);
    }
    
    if ((int)out_imp.size() != n_prompt) {
        fprintf(stderr, "[specprefill] importance size mismatch: %zu vs %d\n", out_imp.size(), n_prompt);
        return -1;
    }
    
    return 0;
}

// Phase 2: partial-layer score pass. Decodes only [0, N) on F tokens, extracts Q/K
// from the partial graph and computes the cheap last-row proxy importance score.
// Caller is responsible for memory_clear before invoking the resume decode.
//
// When n_query_positions > 1, uses SpecPrefill-style aggregation:
//   - MAX over heads, MAX over layers, MEAN over query positions
// The query positions are the last n_query_positions tokens of the prompt.
static int self_layer_score_partial(
    llama_context * ctx,
    const llama_token * prompt_tokens,
    int n_prompt,
    int n_early_layers,
    std::vector<float> & out_imp,
    bool use_max = false,
    int n_query_positions = 1) {
    if (!ctx || !prompt_tokens || n_prompt <= 0 || n_early_layers < 1) {
        return -1;
    }
    if (n_prompt > (int) llama_n_batch(ctx)) {
        return -2;
    }

    llama_memory_clear(llama_get_memory(ctx), false);

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
    const int r = ctx->decode_partial(batch, 0, n_early_layers);
    llama_batch_free(batch);
    if (r != 0) {
        fprintf(stderr, "[self-layer-prefill] partial score decode failed: %d\n", r);
        return -1;
    }

    std::vector<slp_layer_qk> layers;
    if (extract_qk_all_layers(*ctx, layers) != 0) {
        return -1;
    }
    
    if (n_query_positions > 1) {
        // SpecPrefill-style: use multiple query positions with max-max-mean
        std::vector<int> query_positions;
        int start_qi = std::max(0, n_prompt - n_query_positions);
        for (int qi = start_qi; qi < n_prompt; qi++) {
            query_positions.push_back(qi);
        }
        accumulate_range_multi_query(layers, 0, n_early_layers, query_positions, out_imp);
    } else {
        // Original: single query position (last token)
        accumulate_range(layers, 0, n_early_layers, n_prompt - 1, out_imp, use_max);
    }
    
    if ((int) out_imp.size() != n_prompt) {
        fprintf(stderr, "[self-layer-prefill] partial score: imp size %zu != n_prompt %d\n",
                out_imp.size(), n_prompt);
        return -1;
    }
    return 0;
}

int llama_self_layer_prefill_partial(
    llama_context * ctx,
    const llama_token * prompt_tokens,
    int n_prompt,
    const llama_self_layer_prefill_params * params) {
    if (!ctx || !prompt_tokens || !params || n_prompt <= 0) {
        return -1;
    }
    if (n_prompt > (int) llama_n_batch(ctx)) {
        return -2;
    }

    const llama_model * model = llama_get_model(ctx);
    const llm_arch arch = model->arch;
    if (arch != LLM_ARCH_LLAMA && arch != LLM_ARCH_LLAMA4) {
        return -3;
    }

    llama_self_layer_prefill_params P = *params;
    if (P.n_early_layers < 0) P.n_early_layers = 0;
    const int n_layer = (int) llama_model_n_layer(model);
    if (P.n_early_layers > n_layer) P.n_early_layers = n_layer;

    int n_keep = (int) (n_prompt * P.keep_ratio);
    if (n_keep < 1) n_keep = 1;
    if (n_keep > n_prompt) n_keep = n_prompt;

    std::vector<llama_token> resume_tokens;
    resume_tokens.reserve((size_t) n_keep);

    if (P.n_early_layers == 0 || n_keep == n_prompt) {
        // Identity gate: no scoring or no filtering. Resume = full decode on all tokens.
        resume_tokens.assign(prompt_tokens, prompt_tokens + n_prompt);
    } else {
        std::vector<float> imp;
        int rs;
        
        if (P.n_lookahead_tokens > 0) {
            // SpecPrefill-style: generate look-ahead tokens via early exit, then score
            rs = self_layer_score_with_lookahead(ctx, prompt_tokens, n_prompt, P.n_early_layers, 
                                                  P.n_lookahead_tokens, imp, P.lookahead_temp);
        } else {
            // Original: use last token(s) of prompt as query
            rs = self_layer_score_partial(ctx, prompt_tokens, n_prompt, P.n_early_layers, 
                                          imp, P.use_max_aggregation, P.n_query_positions);
        }
        
        if (rs != 0) {
            return rs;
        }
        if (P.pool_kernel_size > 1) {
            llama_spec_prefill_apply_pooling(imp, P.pool_kernel_size);
        }

        std::vector<std::pair<float,int>> ranked;
        ranked.reserve((size_t) n_prompt);
        for (int i = 0; i < n_prompt; i++) {
            ranked.push_back({imp[(size_t) i], i});
        }
        std::sort(ranked.begin(), ranked.end(),
                  [](const std::pair<float,int> & a, const std::pair<float,int> & b) {
                      return a.first > b.first;
                  });

        std::vector<bool> keep((size_t) n_prompt, false);
        for (int i = 0; i < n_keep; i++) {
            keep[(size_t) ranked[(size_t) i].second] = true;
        }
        keep[(size_t)(n_prompt - 1)] = true; // last token must remain (generation entry)

        for (int i = 0; i < n_prompt; i++) {
            if (keep[(size_t) i]) {
                resume_tokens.push_back(prompt_tokens[i]);
            }
        }
    }

    if (process_filtered_decode(ctx, resume_tokens) != 0) {
        return -1;
    }
    return (int) resume_tokens.size();
}

int llama_self_layer_prefill(
    llama_context * ctx,
    const llama_token * prompt_tokens,
    int n_prompt,
    const llama_self_layer_prefill_params * params) {
    if (!ctx || !prompt_tokens || !params || n_prompt <= 0) {
        return -1;
    }
    if (n_prompt > (int) llama_n_batch(ctx)) {
        return -2;
    }

    llama_self_layer_prefill_params P = *params;
    std::vector<float> shallow, deep, full;
    float p_sf = 0, p_sd = 0;
    if (llama_self_layer_prefill_attention_profiles(ctx, prompt_tokens, n_prompt, &P, shallow, deep, full, &p_sf, &p_sd) != 0) {
        return -1;
    }

    std::vector<float> imp = full;
    if (P.pool_kernel_size > 1) {
        llama_spec_prefill_apply_pooling(imp, P.pool_kernel_size);
    }

    std::vector<int> pos((size_t) n_prompt);
    for (int i = 0; i < n_prompt; i++) {
        pos[(size_t) i] = i;
    }

    std::vector<llama_token> ft;
    std::vector<int> fp;
    if (filter_topk(prompt_tokens, pos.data(), n_prompt, imp, P.keep_ratio, P.use_chunking, P.chunk_size, ft, fp) != 0) {
        return -1;
    }

    if (process_filtered_decode(ctx, ft) != 0) {
        return -1;
    }
    return (int) ft.size();
}

int llama_self_layer_prefill_init(struct llama_context * ctx, int n_early) {
    if (!ctx) {
        return -1;
    }
    return ctx->init_self_layer_prefill(n_early);
}

bool llama_self_layer_prefill_is_init(struct llama_context * ctx) {
    return ctx && ctx->self_layer_prefill_enabled();
}

int llama_self_layer_prefill_get_n_early(struct llama_context * ctx) {
    return ctx ? ctx->self_layer_prefill_n_early() : 0;
}
