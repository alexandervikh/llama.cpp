#include "models.h"
#include "lazyllm-pool.h"

// Partial layer-range OpenAI-MoE graph builder for LazyLLM.
// Mirrors llm_build_openai_moe_iswa but parameterised on [il_start, il_end).
//
// gpt-oss-20B uses LLM_ARCH_OPENAI_MOE with per-layer rope freqs and
// build_attn_inp_kv_iswa (interleaved sliding-window attention).
//
// When il_start > 0, the caller fills ubatch.embd with hidden states from
// a previous partial-score call (embedding injection path).
// When il_end < n_layer, lm_head is skipped; residual is exposed as t_embd.

llm_build_openai_moe_partial::llm_build_openai_moe_partial(
        const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {

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
    auto        * inp_attn = build_attn_inp_kv_iswa();

    ggml_tensor * inp_out_ids = use_out_ids ? build_inp_out_ids() : nullptr;

    for (int il = il_start; il < il_end; ++il) {
        const float freq_base_l  = model.get_rope_freq_base (cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL,
                model.layers[il].attn_norm, nullptr,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        {
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            cb(Qcur, "Qcur", il);
            if (model.layers[il].bq) {
                Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                cb(Qcur, "Qcur", il);
            }
            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            cb(Kcur, "Kcur", il);
            if (model.layers[il].bk) {
                Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                cb(Kcur, "Kcur", il);
            }
            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
            cb(Vcur, "Vcur", il);
            if (model.layers[il].bv) {
                Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                cb(Vcur, "Vcur", il);
            }

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_rot, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_rot, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_rot, n_head_kv, n_tokens);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                    ext_factor, attn_factor, beta_fast, beta_slow);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].bo,
                    Qcur, Kcur, Vcur, nullptr, model.layers[il].attn_sinks, nullptr,
                    1.0f / sqrtf(float(n_rot)), il);
            cb(cur, "attn_out", il);
        }

        if (last_chunk && il == (int32_t) n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = ffn_inp;
        cur = build_norm(cur,
                model.layers[il].attn_post_norm, nullptr,
                LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        cur = build_moe_ffn(cur,
                model.layers[il].ffn_gate_inp,    model.layers[il].ffn_gate_inp_b,
                model.layers[il].ffn_up_exps,     model.layers[il].ffn_up_exps_b,
                model.layers[il].ffn_gate_exps,   model.layers[il].ffn_gate_exps_b,
                model.layers[il].ffn_down_exps,   model.layers[il].ffn_down_exps_b,
                nullptr,
                n_expert, n_expert_used,
                LLM_FFN_SWIGLU_OAI_MOE, false,
                false, 0.0,
                LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT,
                il);
        cb(cur, "ffn_moe_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }
    cur = inpL;

    if (last_chunk) {
        cur = build_norm(cur,
                model.output_norm, nullptr,
                LLM_NORM_RMS, -1);
        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        cur = build_lora_mm(model.output, cur);
        cb(cur, "result_output", -1);
        res->t_logits = cur;
    } else {
        res->t_embd = cur;
    }

    lazyllm_add_score_pool(ctx0, gf, il_start, il_end);
    ggml_build_forward_expand(gf, cur);
}
