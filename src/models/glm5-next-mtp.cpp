#include "models.h"

// GLM-5.3's Multi-Token Prediction module, run as a standalone speculative draft.
//
// The module is one MLA+DSA attention layer and one 144-expert MoE, preceded by the MTP entry
// point: the incoming token embedding and the *target model's* hidden state are each RMS-normed,
// concatenated, and projected back down to n_embd by eh_proj. Reference:
// vllm/model_executor/models/glm4_moe_mtp.py, Glm4MoeMultiTokenPredictorLayer.forward.
//
// Two details are easy to get backwards and both change the result:
//
//   * enorm normalises the token EMBEDDING and hnorm the incoming HIDDEN state - not the other
//     way round - and the concatenation order is [enorm(embed), hnorm(h_prev)], matching the
//     [2*n_embd, n_embd] shape of eh_proj.
//
//   * res->t_embd is set BEFORE the final norm, unlike every other architecture here. The value
//     the reference feeds back as previous_hidden_states on the next speculative step is the
//     module's raw output; shared_head.norm is applied only on the way to the logits. Exposing
//     the post-norm tensor instead would quietly corrupt drafting at depth > 1.
//
// Unlike the 45 transformer layers of the parent, this block has no mHC hyper-connections - it
// consumes an already-collapsed hidden state, so there are no parallel streams to mix - and
// therefore uses plain pre-norm residuals.

llm_build_glm5_next_mtp::llm_build_glm5_next_mtp(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params), model(model) {
    GGML_ASSERT(n_layer == 1 && "the MTP draft is a single block");

    const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v_mla = hparams.n_embd_head_v_mla();
    const int64_t kv_lora_rank      = hparams.n_lora_kv;
    const float   kq_scale_mla      = 1.0f / sqrtf((float) n_embd_head_k_mla);
    const int64_t n_head            = hparams.n_head();

    const int    il    = 0;
    const auto & layer = model.layers[il];

    ggml_tensor * cur;

    ggml_tensor * inp_embd = build_inp_embd(model.tok_embd);
    cb(inp_embd, "model.embed_tokens", -1);

    auto * inp_mtp = build_inp_mtp_hidden();

    // NoPE, like the parent's MLA layers: qk_rope_head_dim is 0, so there is no rotary tail.
    auto * inp_attn    = build_attn_inp_k();
    auto * inp_out_ids = build_inp_out_ids();

    // ---------------- MTP entry point ----------------
    {
        ggml_tensor * e = ggml_mul(ctx0, inp_embd, inp_mtp->emask);
        e = build_norm(e, layer.nextn.enorm, NULL, LLM_NORM_RMS, il);
        cb(e, "mtp_enorm", il);

        ggml_tensor * h = build_norm(inp_mtp->h_prev, layer.nextn.hnorm, NULL, LLM_NORM_RMS, il);
        cb(h, "mtp_hnorm", il);

        cur = ggml_concat(ctx0, e, h, 0);
        cur = ggml_mul_mat(ctx0, layer.nextn.eh_proj, cur);
        cb(cur, "mtp_eh_proj", il);
    }

    ggml_tensor * inpL = cur;

    // ---------------- attention ----------------
    cur = build_norm(inpL, layer.attn_norm, NULL, LLM_NORM_RMS, il);
    cb(cur, "attn_norm", il);

    {
        ggml_tensor * q_a = ggml_mul_mat(ctx0, layer.wq_a, cur);
        q_a = build_norm(q_a, layer.attn_q_a_norm, NULL, LLM_NORM_RMS, il);
        ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.wq_b, q_a);

        ggml_tensor * kv_cmpr = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);
        kv_cmpr = build_norm(kv_cmpr, layer.attn_kv_a_norm, NULL, LLM_NORM_RMS, il);

        if (layer.wk_b && layer.wv_b) {
            // Absorbed MLA: the query is folded through wk_b so attention runs against the
            // compressed latent, and wv_b expands the result on the way out.
            ggml_tensor * q_nope = ggml_reshape_3d(ctx0, Qcur, n_embd_head_k_mla, n_head, n_tokens);
            q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);                 // [hk, T, n_head]
            ggml_tensor * q_absorbed = ggml_mul_mat(ctx0, layer.wk_b, q_nope);
            q_absorbed = ggml_permute(ctx0, q_absorbed, 0, 2, 1, 3);         // [kv_lora, n_head, T]
            Qcur = ggml_cont(ctx0, q_absorbed);

            ggml_tensor * Kcur = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
            ggml_tensor * Vcur = Kcur;

            cur = build_attn(inp_attn, layer.wo, NULL, Qcur, Kcur, Vcur,
                             nullptr, nullptr, layer.wv_b, kq_scale_mla, il);
        } else {
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head_k_mla, n_head, n_tokens);
            ggml_tensor * kv = ggml_mul_mat(ctx0, layer.wkv_b, kv_cmpr);
            const int64_t kv_per_head = n_embd_head_k_mla + n_embd_head_v_mla;

            ggml_tensor * Kcur = ggml_view_3d(ctx0, kv, n_embd_head_k_mla, n_head, n_tokens,
                    ggml_row_size(kv->type, kv_per_head),
                    ggml_row_size(kv->type, kv_per_head * n_head), 0);
            ggml_tensor * Vcur = ggml_view_3d(ctx0, kv, n_embd_head_v_mla, n_head, n_tokens,
                    ggml_row_size(kv->type, kv_per_head),
                    ggml_row_size(kv->type, kv_per_head * n_head),
                    ggml_row_size(kv->type, n_embd_head_k_mla));
            Kcur = ggml_cont(ctx0, Kcur);
            Vcur = ggml_cont(ctx0, Vcur);

            cur = build_attn(inp_attn, layer.wo, NULL, Qcur, Kcur, Vcur,
                             nullptr, nullptr, nullptr, kq_scale_mla, il);
        }
        cb(cur, "mla_out", il);
    }

    ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
    cb(ffn_inp, "attn_residual", il);

    // ---------------- MoE ----------------
    cur = build_norm(ffn_inp, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
    cb(cur, "ffn_norm", il);

    {
        ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                layer.ffn_exp_probs_b,
                hparams.n_expert, hparams.n_expert_used,
                LLM_FFN_SWIGLU_CLAMPED, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il);
        cb(moe_out, "ffn_moe_out", il);

        // Same clamped SwiGLU as the parent's shared expert: the fused SiLU path has no limit,
        // and the clamp only bites once activations exceed it.
        const float limit = hparams.swiglu_limit;
        ggml_tensor * sg = ggml_clamp(ctx0, ggml_mul_mat(ctx0, layer.ffn_gate_shexp, cur), -INFINITY, limit);
        ggml_tensor * su = ggml_clamp(ctx0, ggml_mul_mat(ctx0, layer.ffn_up_shexp,   cur), -limit, limit);
        ggml_tensor * shexp = ggml_mul_mat(ctx0, layer.ffn_down_shexp,
                ggml_mul(ctx0, ggml_silu(ctx0, sg), su));
        cur = ggml_add(ctx0, moe_out, shexp);
    }
    cb(cur, "ffn_out", il);

    cur = ggml_add(ctx0, cur, ffn_inp);

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    // Pre-norm, deliberately: this is the tensor that becomes previous_hidden_states on the next
    // speculative step. See the note at the top of this file.
    cb(cur, "result_mtp_hidden", -1);
    res->t_embd = cur;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
