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

// Phase 1 cheap scoring: per token j, sum over heads h of Q[last,h,:] . K[j,kh,:]
// (raw scaled dot, no softmax). GQA reduces cost by pre-summing Q within each KV group:
//   imp[j] = sum_kh K[j,kh,:] . ( sum_{h in group(kh)} Q[last,h,:] ) * scale
// Cost ~= n_kv * n_tok * d  vs. old n_head * n_tok * d + n_head * 2*n_tok softmax.
static void layer_last_query_importance(const slp_layer_qk & L, int qi_last, std::vector<float> & imp) {
    const int64_t n_tok = L.n_tok;
    const int n_head = (int) L.n_head;
    const int n_kv = (int) L.n_kv;
    const int d = (int) L.d;
    const int group = n_head / n_kv;
    const float scale = 1.f / sqrtf(float(d));

    std::vector<float> q_sum((size_t)(d * n_kv), 0.f);
    for (int kh = 0; kh < n_kv; kh++) {
        for (int hg = 0; hg < group; hg++) {
            const int h = kh * group + hg;
            for (int di = 0; di < d; di++) {
                q_sum[(size_t)(di + d * kh)] +=
                    q_at(L.q, di, h, qi_last, L.d, L.n_head);
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

static void accumulate_range(const std::vector<slp_layer_qk> & layers, int il0, int il1, int qi_last, std::vector<float> & out) {
    out.clear();
    int count = 0;
    for (const auto & L : layers) {
        if (L.layer < il0 || L.layer >= il1) {
            continue;
        }
        std::vector<float> imp;
        layer_last_query_importance(L, qi_last, imp);
        if (out.empty()) {
            out = std::move(imp);
        } else {
            for (size_t i = 0; i < out.size(); i++) {
                out[i] += imp[i];
            }
        }
        count++;
    }
    if (count > 1) {
        for (float & v : out) {
            v /= float(count);
        }
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
    accumulate_range(layers, 0, P.n_early_layers, qi_last, out_shallow);
    accumulate_range(layers, P.n_early_layers, n_layer, qi_last, out_deep);
    accumulate_range(layers, 0, n_layer, qi_last, out_full);

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
    accumulate_range(layers, 0, P.n_early_layers, n_prompt - 1, imp);
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

// Phase 2: partial-layer score pass. Decodes only [0, N) on F tokens, extracts Q/K
// from the partial graph and computes the cheap last-row proxy importance score.
// Caller is responsible for memory_clear before invoking the resume decode.
static int self_layer_score_partial(
    llama_context * ctx,
    const llama_token * prompt_tokens,
    int n_prompt,
    int n_early_layers,
    std::vector<float> & out_imp) {
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
    accumulate_range(layers, 0, n_early_layers, n_prompt - 1, out_imp);
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
        const int rs = self_layer_score_partial(ctx, prompt_tokens, n_prompt, P.n_early_layers, imp);
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
