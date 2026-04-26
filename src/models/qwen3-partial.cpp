#include "models.h"
#include "lazyllm-pool.h"

// Partial layer-range Qwen3 graph builder. Identical to qwen2-partial except
// Qwen3 adds QK-norm (attn_q_norm / attn_k_norm) applied before RoPE.
//
// When il_start > 0 the embedding lookup is bypassed: the caller must populate
// ubatch.embd with the residual-stream values from an earlier partial call.
//
// When il_end < n_layer the final output norm and lm_head are skipped; the
// residual stream after layer (il_end-1) is named "l_out-N" via the "l_out"
// callback and exposed as res->t_embd.

llm_build_qwen3_partial::llm_build_qwen3_partial(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    const int32_t il_start   = params.il_start < 0 ? 0 : params.il_start;
    const int32_t il_end_raw = params.il_end   < 0 ? (int32_t) n_layer : params.il_end;
    const int32_t il_end     = il_end_raw > (int32_t) n_layer ? (int32_t) n_layer : il_end_raw;
    GGML_ASSERT(il_start >= 0 && il_start <= il_end && il_end <= (int32_t) n_layer);

    const bool last_chunk  = (il_end == (int32_t) n_layer) || params.early_exit;
    const bool use_out_ids = last_chunk && !params.early_exit;

    ggml_tensor * cur  = nullptr;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    ggml_tensor * inp_pos  = build_inp_pos();
    auto        * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = use_out_ids ? build_inp_out_ids() : nullptr;

    for (int il = il_start; il < il_end; ++il) {
        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        {
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            cb(Qcur, "Qcur", il);

            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            cb(Kcur, "Kcur", il);

            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            // Qwen3: QK-norm before RoPE
            if (model.layers[il].attn_q_norm) {
                Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
                cb(Qcur, "Qcur_normed", il);
            }
            if (model.layers[il].attn_k_norm) {
                Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
                cb(Kcur, "Kcur_normed", il);
            }

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].bo,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                    1.0f / sqrtf(float(n_embd_head)), il);
            cb(cur, "attn_out", il);
        }

        if (last_chunk && il == (int32_t) n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }
    cur = inpL;

    if (last_chunk) {
        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);
        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        cur = build_lora_mm(model.output, cur);
        if (model.output_b != nullptr) {
            cur = ggml_add(ctx0, cur, model.output_b);
        }
        cb(cur, "result_output", -1);
        res->t_logits = cur;
    } else {
        res->t_embd = cur;
    }

    lazyllm_add_score_pool(ctx0, gf, il_start, il_end);
    ggml_build_forward_expand(gf, cur);
}
