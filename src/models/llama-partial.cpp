#include "models.h"
#include "lazyllm-pool.h"

// Partial layer-range Llama graph builder. This is a clone of llm_build_llama<false>
// (src/models/llama.cpp) with the layer loop parameterised on [il_start, il_end).
//
// Intent (see SELF_LAYER_PREFILL_PLAN.md, Phase 2 / Option A):
//   - "score" pass: il_start = 0,           il_end = N (< n_layer)
//   - "resume" pass: il_start = N (> 0),    il_end = n_layer
//
// When il_start > 0 the embedding lookup is bypassed: the caller must populate
// ubatch.embd with the residual-stream values that fed layer (il_start) in the
// score pass. The existing build_inp_embd() already implements a runtime
// token-vs-embd select via ubatch.token / ubatch.embd, so we reuse it.
//
// When il_end < n_layer the final output norm and lm_head are skipped; the
// residual stream after layer (il_end-1) is exposed as res->t_embd. Callers
// can fetch it from the schedule and copy back to host (Phase 2 CPU path) or
// keep it on backend (Phase 3 GPU path).
//
// KV cache semantics: the layer loop calls build_attn(... il) which writes
// K/V at the current layer index. Layers outside [il_start, il_end) are
// simply not built and therefore not touched, which is exactly what we want.

llm_build_llama_partial::llm_build_llama_partial(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    const int32_t il_start = params.il_start < 0 ? 0 : params.il_start;
    const int32_t il_end_raw = params.il_end < 0 ? (int32_t) n_layer : params.il_end;
    const int32_t il_end = il_end_raw > (int32_t) n_layer ? (int32_t) n_layer : il_end_raw;
    GGML_ASSERT(il_start >= 0 && il_start <= il_end && il_end <= (int32_t) n_layer);

    // When early_exit is true, apply output_norm + lm_head even at intermediate layers
    const bool last_chunk  = (il_end == (int32_t) n_layer) || params.early_exit;
    const bool first_chunk = (il_start == 0);
    
    // For early_exit, we skip out_ids optimization to simplify buffer allocation.
    // The regular last_chunk uses out_ids to select specific output tokens.
    const bool use_out_ids = last_chunk && !params.early_exit;

    ggml_tensor * cur = nullptr;
    ggml_tensor * inpL;

    // build_inp_embd handles both token-id and ubatch.embd inputs at runtime.
    // For partial-resume (il_start > 0), the caller fills ubatch.embd with the
    // residual stream H_{il_start-1} produced by an earlier partial-score call.
    inpL = build_inp_embd(model.tok_embd);

    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    const float kq_scale = hparams.f_attention_scale == 0.0f
                            ? 1.0f / sqrtf(float(n_embd_head))
                            : hparams.f_attention_scale;

    ggml_tensor * inp_out_ids = use_out_ids ? build_inp_out_ids() : nullptr;

    for (int il = il_start; il < il_end; ++il) {
        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        {
            ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

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
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            if (hparams.use_kq_norm) {
                Qcur = ggml_rms_norm(ctx0, Qcur, hparams.f_norm_rms_eps);
                Kcur = ggml_rms_norm(ctx0, Kcur, hparams.f_norm_rms_eps);
                cb(Qcur, "Qcur_normed", il);
                cb(Kcur, "Kcur_normed", il);
            }
            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].bo,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            cb(cur, "attn_out", il);
        }

        // Only the last layer of the FULL model gathers logical outputs (last_chunk &&
        // last layer of the chunk). For an intermediate chunk we keep all rows so the
        // hidden state we expose covers every input token.
        if (last_chunk && il == (int32_t) n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        if (model.layers[il].ffn_gate_inp == nullptr) {
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                    model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, NULL,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, true,
                    false, 0.0,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il);
            cb(cur, "ffn_moe_out", il);
        }
        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "ffn_out", il);

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
        cb(cur, "result_output", -1);
        res->t_logits = cur;
    } else {
        // Intermediate chunk: expose the residual stream as t_embd so callers can copy
        // it back and feed it into the next partial-resume invocation.
        // NOTE: do NOT rename with cb() here — the tensor already carries the "l_out-N"
        // name from the layer loop above. Renaming it to "partial_h_out" would overwrite
        // that name and break tensor extraction by name in llama-lazyllm.cpp.
        res->t_embd = cur;
    }

    GGML_UNUSED(first_chunk);

    // GPU-side attention score pooling (Fix 1): reduces kq_soft_max to
    // per-key-position scores on the GPU, avoiding huge CPU↔GPU transfers.
    lazyllm_add_score_pool(ctx0, gf, il_start, il_end);

    ggml_build_forward_expand(gf, cur);
}
