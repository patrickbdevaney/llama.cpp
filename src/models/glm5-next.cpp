#include "models.h"

#include "../llama-kv-cache.h"   // build_attn_dsa calls cpy_k/get_k on the cache context

#include "llama-memory-recurrent.h"

// GLM-5.3-Flash (glm5-next).
//
// The transformer body is Kimi-Linear's: 34 KDA linear-attention layers interleaved 3:1 with
// 11 NoPE MLA layers (plus the MTP block, which is loaded and not executed), sigmoid-routed MoE
// with a shared expert. That part is a near-transcription of src/models/kimi-linear.cpp.
//
// What is new is mHC. Instead of one residual stream there are `hc_mult` of them, and each of
// the two sites per layer (attention, FFN) does:
//
//     residual              = streams
//     post, comb, collapsed = mHC(streams)
//     y                     = sublayer(norm(collapsed))
//     streams               = post (x) y  +  comb^T @ residual
//
// `comb` is column-stochastic, produced by ggml_mhc_sinkhorn (see ggml.h for why the
// normalisation order matters and why it is fused). `post` is 2*sigmoid, range [0,2] - it is
// not a probability and must not be clamped to [0,1].
//
// Streams are carried as [n_embd, hc, n_tokens]: that makes the mHC input flatten a free
// reshape, and the one permutation per site is shared between the collapse and the mixing
// matmul.

// Causal Conv1d function for Q,K,V
// When qkv is 0, it is Q, 1 is K, 2 is V
static ggml_tensor * causal_conv1d(ggml_cgraph * gf, ggml_context * ctx0, ggml_tensor * conv_states_all, ggml_tensor * conv_state_all, int64_t qkv, ggml_tensor * x, ggml_tensor * proj_w, ggml_tensor * conv_w, int64_t d_conv, int64_t head_dim, int64_t n_head, int64_t n_seq_tokens, int64_t n_seqs, int64_t n_tokens, int64_t kv_head) {
    const int64_t d_inner = head_dim * n_head;
    const int64_t conv_state_size = (d_conv - 1) * d_inner;
    const int64_t n_embd_r_total = 3 * conv_state_size;  // Q + K + V

    // conv_state_all is [n_embd_r_total, n_seqs], split into Q, K, V
    // Each conv state is [(d_conv-1) * d_inner] per sequence, need to reshape to [d_conv-1, d_inner, n_seqs]
    // Memory layout: for each seq, Q state is first conv_state_size elements, then K, then V
    // conv_state_all has stride: nb[0] = element_size, nb[1] = n_embd_r_total * element_size
    // View Q conv state: offset 0, size conv_state_size per seq
    // conv_state_all is [n_embd_r_total, n_seqs] with memory layout:
    //   state[i + seq * n_embd_r_total] where i = conv_step + channel * (d_conv-1) + {0, conv_state_size, 2*conv_state_size} for Q/K/V
    // We want [d_conv-1, d_inner, n_seqs] view:
    //   nb1 = (d_conv-1) * element_size (stride between channels)
    //   nb2 = n_embd_r_total * element_size (stride between seqs)
    ggml_tensor * conv_state_x = ggml_view_3d(ctx0, conv_state_all, d_conv - 1, d_inner, n_seqs,
        (d_conv - 1) * ggml_element_size(conv_state_all),  // nb1: stride between channels
        n_embd_r_total * ggml_element_size(conv_state_all),  // nb2: stride between seqs
        qkv * conv_state_size * ggml_element_size(conv_state_all));

// Causal Conv1d function for Q,K,V
// When qkv is 0, it is Q, 1 is K, 2 is V
    // Step 1: Q, K, V projections -> [d_inner, n_tokens]
    ggml_tensor * x_proj = ggml_mul_mat(ctx0, proj_w, x);

    // Reshape input: {d_inner, n_tokens} -> {d_inner, n_seq_tokens, n_seqs}
    ggml_tensor * x_3d = ggml_reshape_3d(ctx0, x_proj, d_inner, n_seq_tokens, n_seqs);

    // Concat Q conv state and current input: {d_conv-1 + n_seq_tokens, d_inner, n_seqs}
    ggml_tensor * conv_x = ggml_concat(ctx0, conv_state_x, ggml_transpose(ctx0, x_3d), 0);

    // Save last (d_conv-1) columns back to Q conv state
    ggml_tensor * last_conv_x = ggml_view_3d(ctx0, conv_x, d_conv - 1, d_inner, n_seqs,
        conv_x->nb[1], conv_x->nb[2], n_seq_tokens * conv_x->nb[0]);
    ggml_build_forward_expand(gf,
        ggml_cpy(ctx0, last_conv_x,
            ggml_view_3d(ctx0, conv_states_all,
                d_conv - 1, d_inner, n_seqs,
                (d_conv - 1) * ggml_element_size(conv_states_all),           // nb1: contiguous within one channel's conv taps
                n_embd_r_total * ggml_element_size(conv_states_all),         // nb2: stride between sequences (skip over K,V states)
                (kv_head * n_embd_r_total + qkv * conv_state_size) * ggml_element_size(conv_states_all))));  // offset to first seq's Q/K/V state
    // Reshape conv weight: GGUF [d_conv, 1, d_inner, 1] -> ggml_ssm_conv expects [d_conv, d_inner]
    // GGUF stores as [d_conv, 1, d_inner, 1] with memory layout w[conv_step + channel * d_conv]
    // vLLM stores as [d_inner, d_conv] with memory layout w[channel * d_conv + conv_step]
    // ggml_ssm_conv computes: c[conv_step + channel * d_conv]
    // GGUF layout: [d_conv, 1, d_inner] or [d_conv, 1, d_inner, 1] -> reshape to [d_conv, d_inner]
    // Reshape conv weight from [d_conv, 1, d_inner, 1] to [d_conv, d_inner] for ggml_ssm_conv
    ggml_tensor * conv_weight = ggml_reshape_2d(ctx0, conv_w, d_conv, d_inner);

    // Apply conv1d
    // ggml_ssm_conv output: {d_inner, n_seq_tokens, n_seqs}
    ggml_tensor * Xcur = ggml_ssm_conv(ctx0, conv_x, conv_weight);
    // Reshape to 2D for bias add: {d_inner, n_tokens}
    Xcur = ggml_reshape_2d(ctx0, Xcur, d_inner, n_tokens);
    Xcur = ggml_silu(ctx0, Xcur);

    return ggml_reshape_4d(ctx0, Xcur, head_dim, n_head, n_seq_tokens, n_seqs);
}


// DSA index scores for the current batch.
//
// Validated formulation - tests/test-dsa-indexer.cpp checks exactly this against the
// transformers oracle at 2.18e-06. Four things here are easy to get wrong and invisible if you
// do (see scripts/dsa_reference.py):
//   * k_norm is a LAYERNORM with a bias, not RMS.
//   * The pool key is a PER-CHANNEL softmax over the kpool tokens of (gate + ape) - not a mean.
//   * relu sits between the per-head scores and the head-weighted sum.
//   * Pools start at the first real token; with no left padding that is slot 0.
//
// LIMITATION: this scores only the CURRENT batch, which is the prefill case. Decode needs the
// indexer key and gate of every cached token, which means widening the KV row on this layer -
// Phase 2b. Until then the scores are computed and discarded, so nothing consumes them.
ggml_tensor * llm_build_glm5_next::build_dsa_index_scores(
        ggml_tensor * x, ggml_tensor * q_a, const llama_layer & layer, int il) {
    const int64_t hd = hparams.indexer_head_size;
    const int64_t nh = hparams.indexer_n_head;
    const int64_t kp = hparams.indexer_kpool ? hparams.indexer_kpool : 4;
    const int64_t S  = x->ne[1];

    ggml_tensor * q = ggml_mul_mat(ctx0, layer.indexer_attn_q_b, q_a);
    q = ggml_reshape_3d(ctx0, q, hd, nh, S);

    ggml_tensor * k = ggml_mul_mat(ctx0, layer.indexer_attn_k, x);
    k = ggml_norm(ctx0, k, 1e-6f);                       // LayerNorm, eps 1e-6
    k = ggml_add(ctx0, ggml_mul(ctx0, k, layer.indexer_k_norm), layer.indexer_k_norm_b);

    ggml_tensor * gate = ggml_mul_mat(ctx0, layer.indexer_kpool_gate, x);

    const int64_t n_pools = (S + kp - 1)/kp;
    if (n_pools*kp != S) {
        // Ragged tail: the reference pads the final pool and masks the missing slots. Not yet
        // handled here, and silently mis-pooling would be worse than not running.
        return nullptr;
    }

    ggml_tensor * k3   = ggml_reshape_3d(ctx0, k,    hd, kp, n_pools);
    ggml_tensor * g3   = ggml_reshape_3d(ctx0, gate, hd, kp, n_pools);
    ggml_tensor * ape3 = ggml_reshape_3d(ctx0, layer.indexer_kpool_ape, hd, kp, 1);

    // soft_max reduces ne0, so bring kp there and put it back.
    ggml_tensor * lg = ggml_add(ctx0, g3, ape3);
    lg = ggml_cont(ctx0, ggml_permute(ctx0, lg, 1, 0, 2, 3));
    lg = ggml_soft_max(ctx0, lg);
    lg = ggml_cont(ctx0, ggml_permute(ctx0, lg, 1, 0, 2, 3));

    ggml_tensor * pk = ggml_mul(ctx0, lg, k3);
    pk = ggml_cont(ctx0, ggml_permute(ctx0, pk, 1, 0, 2, 3));
    pk = ggml_sum_rows(ctx0, pk);
    pk = ggml_reshape_2d(ctx0, pk, hd, n_pools);

    ggml_tensor * sc = ggml_mul_mat(ctx0, pk, q);
    sc = ggml_scale(ctx0, sc, 1.0f/sqrtf((float) hd));
    sc = ggml_relu(ctx0, sc);                            // load-bearing

    ggml_tensor * wgt = ggml_mul_mat(ctx0, layer.indexer_proj, x);
    wgt = ggml_scale(ctx0, wgt, 1.0f/sqrtf((float) nh));
    ggml_tensor * wgt3 = ggml_reshape_3d(ctx0, wgt, nh, 1, S);
    ggml_tensor * scp  = ggml_cont(ctx0, ggml_permute(ctx0, sc, 1, 0, 2, 3));
    ggml_tensor * idx  = ggml_mul_mat(ctx0, scp, wgt3);
    return ggml_reshape_2d(ctx0, idx, n_pools, S);
}

// DSA sparse attention over the KV cache (Phase 3).
//
// Selection happens at POOL granularity, and that is the whole trick that makes this tractable
// in ggml. Cache rows for consecutive tokens are contiguous, so the [D, n_kv] cache view can be
// re-viewed as [D*kpool, n_pools] and a single ggml_get_rows then gathers whole pools. No index
// arithmetic at all - no scaling pool ids into token ids, no I32 maths ggml has no ops for - and
// pool granularity is exactly DSA's own granularity, since select_k = indexer_top_k / kpool.
//
// Returns nullptr whenever the sparse path does not apply and the caller falls back to dense
// build_attn. Two of those bail-outs are load-bearing rather than laziness:
//
//   * n_pools <= select_k is the FREE DSA invariant. Every pool would be selected, so selection
//     is a no-op and sparse MUST equal dense exactly. Taking the dense path there makes that
//     identity structural instead of something to hope a test catches.
//   * n_tokens != 1 is prefill. top_k returns a per-token selection, but one gathered K/V can
//     only serve one selection, so prefill would need block-sparse machinery this does not have.
//     Decode is also where the win is: prefill is compute-bound, decode is KV-bandwidth-bound.
//
// KNOWN LIMITATION, and the reason this stays opt-in: pooling groups kpool CONSECUTIVE CACHE
// CELLS, and treats them as kpool consecutive sequence positions. Those coincide for a single
// sequence filling a fresh cache in order, which is the case this is written for. They stop
// coinciding under anything that reorders cells against positions - a second sequence sharing a
// unified cache, a context shift, defragmentation. Attention itself stays correct there, because
// the gathered mask travels with the gathered rows; what degrades is the SELECTION, which would
// pool unrelated positions and pick the wrong ones. That is a quality regression with no visible
// symptom, so this must not be enabled by default until the pooling reads positions rather than
// assuming them.
//
// The gathered-attention algebra itself (gather K/V, gather the mask through transpose ->
// get_rows -> transpose, then build_attn_mha) is the construction proved against dense attention
// in tests/test-dsa-attn.cpp at 1.19e-07.
ggml_tensor * llm_build_glm5_next::build_attn_dsa(
        llm_graph_input_attn_k * inp,
        ggml_tensor * wo,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * x,
        ggml_tensor * q_a,
        const llama_layer & layer,
        ggml_tensor * v_mla,
              float   kq_scale,
              int     il) {
    const int64_t hd = hparams.indexer_head_size;
    const int64_t nh = hparams.indexer_n_head;
    const int64_t kp = hparams.indexer_kpool ? hparams.indexer_kpool : 4;
    const int64_t T  = q_cur->ne[2];

    if (!hparams.dsa_enabled || !layer.indexer_attn_k || !layer.indexer_kpool_gate) {
        return nullptr;
    }
    if (T != 1) {
        return nullptr;                      // prefill: see above
    }

    // Every bail-out has to happen BEFORE the graph is touched. If this returned nullptr after
    // expanding the stores, the caller's dense build_attn would emit a SECOND cpy_k into the same
    // cache slots - so the guards below run against a get_k view that is created but not yet
    // expanded, which costs a tensor header and nothing else.
    const auto * mctx_cur = inp->mctx;

    ggml_tensor * k = mctx_cur->get_k(ctx0, il);          // [D, n_head_kv, n_kv, ns]

    const int64_t D       = k->ne[0];
    const int64_t n_kv    = k->ne[2];
    const int64_t kv_lora = v_cur->ne[0];

    // The [D*kp, n_pools] re-view below reads the cache as raw contiguous memory, so everything
    // that could make that untrue is a bail-out rather than an assert.
    if (k->ne[1] != 1 || k->ne[3] != 1)                  return nullptr;  // MQA/multi-seq only
    if (ggml_is_quantized(k->type))                      return nullptr;  // no fixed element size
    if (k->type != GGML_TYPE_F32 && k->type != GGML_TYPE_F16) return nullptr;
    if (D != kv_lora + 2*hd)                             return nullptr;  // row not widened
    if (n_kv % kp != 0)                                  return nullptr;  // ragged final pool

    const int64_t n_pools  = n_kv / kp;
    const int64_t select_k = hparams.indexer_top_k / kp;

    if (select_k <= 0 || n_pools <= select_k)            return nullptr;  // free DSA: dense == sparse

    // Committed to the sparse path: now it is safe to mutate the graph.
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);
    ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, inp->get_k_idxs(), il));

    const size_t es = ggml_type_size(k->type);

    // --- pooled indexer keys over the whole cache ------------------------------------------
    // k_norm was applied before the key was written into the cache, so the stored indexer key is
    // already normalised; re-normalising here would apply it twice.
    ggml_tensor * ik = ggml_cont(ctx0,
        ggml_view_2d(ctx0, k, hd, n_kv, (size_t) D*es, (size_t) kv_lora*es));
    ggml_tensor * ig = ggml_cont(ctx0,
        ggml_view_2d(ctx0, k, hd, n_kv, (size_t) D*es, (size_t) (kv_lora + hd)*es));
    if (k->type != GGML_TYPE_F32) {
        ik = ggml_cast(ctx0, ik, GGML_TYPE_F32);
        ig = ggml_cast(ctx0, ig, GGML_TYPE_F32);
    }

    ggml_tensor * k3   = ggml_reshape_3d(ctx0, ik, hd, kp, n_pools);
    ggml_tensor * g3   = ggml_reshape_3d(ctx0, ig, hd, kp, n_pools);
    ggml_tensor * ape3 = ggml_reshape_3d(ctx0, layer.indexer_kpool_ape, hd, kp, 1);

    // Per-channel softmax over the kp slots of (gate + ape) - not a mean. soft_max reduces ne0,
    // so kp has to be brought there and put back.
    ggml_tensor * lg = ggml_add(ctx0, g3, ape3);
    lg = ggml_cont(ctx0, ggml_permute(ctx0, lg, 1, 0, 2, 3));
    lg = ggml_soft_max(ctx0, lg);
    lg = ggml_cont(ctx0, ggml_permute(ctx0, lg, 1, 0, 2, 3));

    ggml_tensor * pk = ggml_mul(ctx0, lg, k3);
    pk = ggml_cont(ctx0, ggml_permute(ctx0, pk, 1, 0, 2, 3));
    pk = ggml_sum_rows(ctx0, pk);
    pk = ggml_reshape_2d(ctx0, pk, hd, n_pools);

    // --- score the pools against this token -------------------------------------------------
    ggml_tensor * q = ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, layer.indexer_attn_q_b, q_a), hd, nh, T);

    ggml_tensor * sc = ggml_mul_mat(ctx0, pk, q);                       // [n_pools, nh, T]
    sc = ggml_scale(ctx0, sc, 1.0f/sqrtf((float) hd));
    sc = ggml_relu(ctx0, sc);                                           // load-bearing

    ggml_tensor * wgt  = ggml_scale(ctx0, ggml_mul_mat(ctx0, layer.indexer_proj, x),
                                    1.0f/sqrtf((float) nh));
    ggml_tensor * scp  = ggml_cont(ctx0, ggml_permute(ctx0, sc, 1, 0, 2, 3));
    ggml_tensor * idx  = ggml_mul_mat(ctx0, scp, ggml_reshape_3d(ctx0, wgt, nh, 1, T));
    ggml_tensor * scores = ggml_reshape_2d(ctx0, idx, n_pools, T);      // [n_pools, T]

    ggml_tensor * kq_mask = inp->get_kq_mask();                         // [n_kv, T_pad]

    // A pool must not be selected if none of its tokens are visible, or top_k spends slots on
    // rows that attention will then mask to -inf. Under a causal mask a pool has a visible token
    // iff its FIRST token is visible, so slot 0 of each pool is the exact pool-level mask - and
    // -inf + finite is -inf, so adding it removes those pools from contention.
    ggml_tensor * pmask = ggml_cont(ctx0,
        ggml_view_3d(ctx0, kq_mask, 1, n_pools, T,
                     (size_t) kp*ggml_type_size(kq_mask->type), kq_mask->nb[1], 0));
    scores = ggml_add(ctx0, scores, ggml_reshape_2d(ctx0, pmask, n_pools, T));
    cb(scores, "dsa_pool_scores", il);

    ggml_tensor * sel = ggml_reshape_1d(ctx0, ggml_top_k(ctx0, scores, select_k), select_k);
    cb(sel, "dsa_sel", il);

    const int64_t n_sel = select_k * kp;

    // --- gather K, V and the mask -----------------------------------------------------------
    ggml_tensor * kpools = ggml_view_2d(ctx0, k, D*kp, n_pools, (size_t) (D*kp)*es, 0);
    ggml_tensor * ksel   = ggml_get_rows(ctx0, kpools, sel);            // [D*kp, select_k], F32
    ksel = ggml_reshape_4d(ctx0, ksel, D, 1, n_sel, 1);

    // V is the leading kv_lora channels of the gathered row. wv_b expands from the compressed
    // latent, so the indexer key and gate must not reach it.
    ggml_tensor * vsel = ggml_view_4d(ctx0, ksel, kv_lora, ksel->ne[1], ksel->ne[2], ksel->ne[3],
                                      ksel->nb[1], ksel->nb[2], ksel->nb[3], 0);

    // The mask is [n_kv, T_pad] and get_rows selects along ne1, so it has to be transposed,
    // gathered, and transposed back. This is the pattern proved in tests/test-dsa-attn.cpp.
    const int64_t T_pad = kq_mask->ne[1];
    ggml_tensor * mt   = ggml_cont(ctx0, ggml_transpose(ctx0, kq_mask));    // [T_pad, n_kv]
    mt = ggml_reshape_2d(ctx0, mt, T_pad*kp, n_pools);
    ggml_tensor * msel = ggml_get_rows(ctx0, mt, sel);                      // [T_pad*kp, select_k]
    msel = ggml_reshape_2d(ctx0, msel, T_pad, n_sel);
    msel = ggml_cont(ctx0, ggml_transpose(ctx0, msel));                     // [n_sel, T_pad]
    cb(msel, "dsa_mask_sel", il);

    ggml_tensor * cur = build_attn_mha(q_cur, ksel, vsel, nullptr, msel, nullptr,
                                       v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur);
    }
    return cur;
}

llm_build_glm5_next::mhc_site llm_build_glm5_next::build_mhc(
        ggml_tensor * streams, ggml_tensor * fn, ggml_tensor * base,
        ggml_tensor * scale, int il) {
    const int64_t hc       = hparams.hc_mult;
    const int64_t n_tokens = streams->ne[2];
    const float   eps      = hparams.hc_eps;

    // The mixing weights come from a (2+hc)*hc vector; the reference computes this whole path
    // in F32 even when the streams are BF16, because rounding before the softmax visibly moves
    // `comb`. ggml activations are already F32, so nothing extra is needed here - but do not
    // "optimise" this to a lower precision.
    ggml_tensor * flat = ggml_reshape_2d(ctx0, streams, n_embd*hc, n_tokens);
    flat = ggml_rms_norm(ctx0, flat, hparams.f_norm_rms_eps);   // unweighted, no gain tensor
    cb(flat, "mhc_flat", il);

    ggml_tensor * mix = ggml_mul_mat(ctx0, fn, flat);           // [(2+hc)*hc, n_tokens]
    cb(mix, "mhc_mix", il);

    // Views along ne0 of a contiguous tensor are row-strided, so make them contiguous before
    // anything reshapes or broadcasts them. 24 floats per token - not worth being clever about.
    ggml_tensor * pre_w  = ggml_cont(ctx0, ggml_view_2d(ctx0, mix, hc,    n_tokens, mix->nb[1], 0));
    ggml_tensor * post_w = ggml_cont(ctx0, ggml_view_2d(ctx0, mix, hc,    n_tokens, mix->nb[1], hc*mix->nb[0]));
    ggml_tensor * comb_w = ggml_cont(ctx0, ggml_view_2d(ctx0, mix, hc*hc, n_tokens, mix->nb[1], 2*hc*mix->nb[0]));

    ggml_tensor * pre_b  = ggml_view_1d(ctx0, base, hc,    0);
    ggml_tensor * post_b = ggml_view_1d(ctx0, base, hc,    hc*base->nb[0]);
    ggml_tensor * comb_b = ggml_view_1d(ctx0, base, hc*hc, 2*hc*base->nb[0]);

    ggml_tensor * s_pre  = ggml_view_1d(ctx0, scale, 1, 0);
    ggml_tensor * s_post = ggml_view_1d(ctx0, scale, 1, scale->nb[0]);
    ggml_tensor * s_comb = ggml_view_1d(ctx0, scale, 1, 2*scale->nb[0]);

    // pre = sigmoid(w*s + b) + eps
    ggml_tensor * pre = ggml_sigmoid(ctx0, ggml_add(ctx0, ggml_mul(ctx0, pre_w, s_pre), pre_b));
    pre = ggml_scale_bias(ctx0, pre, 1.0f, eps);

    // post = 2*sigmoid(w*s + b). Range [0,2]; NOT a probability.
    ggml_tensor * post = ggml_sigmoid(ctx0, ggml_add(ctx0, ggml_mul(ctx0, post_w, s_post), post_b));
    post = ggml_scale(ctx0, post, 2.0f);
    cb(post, "mhc_post", il);

    ggml_tensor * comb_l = ggml_reshape_3d(ctx0, comb_w, hc, hc, n_tokens);
    comb_l = ggml_add(ctx0, ggml_mul(ctx0, comb_l, s_comb),
                      ggml_reshape_3d(ctx0, comb_b, hc, hc, 1));
    ggml_tensor * comb = ggml_mhc_sinkhorn(ctx0, comb_l, hparams.hc_sinkhorn_iters, eps);
    cb(comb, "mhc_comb", il);

    // One permutation, used twice: for the pre-weighted collapse and for comb^T @ streams.
    ggml_tensor * streams_p = ggml_cont(ctx0, ggml_permute(ctx0, streams, 1, 0, 2, 3)); // [hc, n_embd, n_tokens]

    ggml_tensor * weighted = ggml_mul(ctx0, streams_p, ggml_reshape_3d(ctx0, pre, hc, 1, n_tokens));
    ggml_tensor * collapsed = ggml_reshape_2d(ctx0, ggml_sum_rows(ctx0, weighted), n_embd, n_tokens);
    cb(collapsed, "mhc_collapsed", il);

    return { post, comb, collapsed, streams_p };
}

ggml_tensor * llm_build_glm5_next::apply_mhc(const mhc_site & s, ggml_tensor * y, int il) {
    const int64_t hc       = hparams.hc_mult;
    const int64_t n_tokens = y->ne[1];

    // comb^T @ streams : each OUTPUT stream is a convex combination of the input streams.
    //
    // Index care, because getting this backwards is invisible: ggml's ne0 is torch's LAST dim,
    // so comb_ggml[n0, n1] == comb_torch[n1, n0]. The reference computes
    //     out[i, d] = sum_j comb_torch[j, i] * streams[j, d]
    // and `comb` is COLUMN-stochastic (sum_j comb_torch[j, i] == 1), so comb^T is row-stochastic
    // and the update is a convex combination. Contracting comb_ggml's ne0 directly would use
    // comb_torch[i, j] - the row sums, which are 0.98-1.02, not 1 - and produce a model that is
    // wrong by a few percent everywhere. Transposing first contracts the correct index.
    ggml_tensor * mixed = ggml_mul_mat(ctx0, s.streams_p, ggml_cont(ctx0, ggml_transpose(ctx0, s.comb)));
    cb(mixed, "mhc_mixed", il);

    ggml_tensor * y3 = ggml_repeat(ctx0, ggml_reshape_3d(ctx0, y, n_embd, 1, n_tokens), mixed);
    ggml_tensor * scaled = ggml_mul(ctx0, y3, ggml_reshape_3d(ctx0, s.post, 1, hc, n_tokens));

    return ggml_add(ctx0, scaled, mixed);
}

llm_build_glm5_next::llm_build_glm5_next(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    ggml_tensor * cur;

    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "model.embed_tokens", -1);

    const int64_t hc = hparams.hc_mult;

    // NoPE throughout - the MLA path has qk_rope_head_dim == 0, so there is no inp_pos.
    auto * inp_kv       = !hparams.is_mla() ? build_inp_mem_hybrid() : nullptr;
    auto * inp_k        =  hparams.is_mla() ? build_inp_mem_hybrid_k() : nullptr;
    auto * inp_rs       =  hparams.is_mla() ? inp_k->get_recr() : inp_kv->get_recr();
    auto * inp_attn_kv  = !hparams.is_mla() ? inp_kv->get_attn() : nullptr;
    auto * inp_attn_k   =  hparams.is_mla() ? inp_k->get_attn() : nullptr;

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const int64_t n_head       = hparams.n_head();
    const int64_t head_dim     = hparams.n_embd_head_kda;
    const int64_t d_conv       = hparams.ssm_d_conv;
    const int64_t d_inner      = n_head * head_dim;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v_mla = hparams.n_embd_head_v_mla();
    const int64_t kv_lora_rank      = hparams.n_lora_kv;
    const float   kq_scale_mla      = 1.0f / sqrtf((float) n_embd_head_k_mla);

    // The MTP block is a real layer in the file but is not part of the forward pass.
    const int n_transformer_layers = n_layer - hparams.nextn_predict_layers;

    // Streams start as hc copies of the embedding (reference: inputs_embeds.unsqueeze(2).expand).
    ggml_tensor * streams = ggml_repeat(ctx0,
            ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens),
            ggml_new_tensor_3d(ctx0, inpL->type, n_embd, hc, n_tokens));
    cb(streams, "mhc_streams_init", -1);

    for (int il = 0; il < n_transformer_layers; ++il) {
        const auto & layer = model.layers[il];

        // ---------------- attention site ----------------
        mhc_site site = build_mhc(streams, layer.hc_attn_fn, layer.hc_attn_base, layer.hc_attn_scale, il);

        cur = build_norm(site.collapsed, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);
        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recurrent(il)) {
            // === KDA Layer (Kimi Delta Attention) with Recurrent State ===
            // Reference: vLLM kda.py
            const auto * mctx_cur = inp_rs->mctx;
            const auto kv_head = mctx_cur->get_head();

            // Get conv states from r_l tensor (Q, K, V each have separate state)
            ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
            cb(conv_states_all, "conv_states_all", il);
            ggml_tensor * conv_state_all = build_rs(inp_rs, conv_states_all, hparams.n_embd_r(), n_seqs);
            ggml_tensor * Qcur = causal_conv1d(gf, ctx0, conv_states_all, conv_state_all, 0, cur, layer.wq, layer.ssm_q_conv, d_conv, head_dim, n_head, n_seq_tokens, n_seqs, n_tokens, kv_head);
            ggml_tensor * Kcur = causal_conv1d(gf, ctx0, conv_states_all, conv_state_all, 1, cur, layer.wk, layer.ssm_k_conv, d_conv, head_dim, n_head, n_seq_tokens, n_seqs, n_tokens, kv_head);
            ggml_tensor * Vcur = causal_conv1d(gf, ctx0, conv_states_all, conv_state_all, 2, cur, layer.wv, layer.ssm_v_conv, d_conv, head_dim, n_head, n_seq_tokens, n_seqs, n_tokens, kv_head);

            // g1 = -exp(A_log) * softplus(f_b(f_a(x)) + dt_bias)
            ggml_tensor * f_a = ggml_mul_mat(ctx0, layer.ssm_f_a, cur);
            ggml_tensor * g1 = ggml_mul_mat(ctx0, layer.ssm_f_b, f_a);
            cb(g1, "g1 f_b(f_a(cur))", il);
            g1 = ggml_add(ctx0, g1, layer.ssm_dt_b);
            g1 = ggml_reshape_3d(ctx0, g1, head_dim, n_head, n_tokens);

            // A_log shape is [1, n_head] or [1, n_head, 1, 1], need to broadcast to [head_dim, n_head, n_tokens]. No need to -exp(a_log) because it was done in convert_hf_to_gguf.py
            // Reshape to [1, n_head, 1] for broadcasting with g1 [head_dim, n_head, n_tokens]
            // GLM-5.3 forget gate:  g = bound * sigmoid(exp(A_log) * (w + dt_bias)).
            //
            // Kimi-Linear's is  g = -exp(A_log) * softplus(w + dt_bias)  - a different function
            // with a different sign convention, and the converter stores exp(A_log) rather than
            // -exp(A_log) to match. Using Kimi's form here is not a small error: it is bounded
            // vs unbounded decay, so the discrepancy compounds along the sequence and shows up
            // as logit error that grows monotonically with position.
            ggml_tensor * A = ggml_reshape_3d(ctx0, layer.ssm_a, 1, n_head, 1);
            g1 = ggml_sigmoid(ctx0, ggml_mul(ctx0, g1, A));
            g1 = ggml_scale(ctx0, g1, hparams.ssm_gate_lower_bound);
            cb(g1, "kda_g1", il);

            g1 = ggml_reshape_4d(ctx0, g1, head_dim, n_head, n_seq_tokens, n_seqs);

            // Compute beta (mixing coefficient)
            ggml_tensor * beta = ggml_mul_mat(ctx0, layer.ssm_beta, cur);
            beta = ggml_reshape_4d(ctx0, beta, 1, n_head, n_seq_tokens, n_seqs);
            cb(beta, "kda_beta", il);

            beta = ggml_sigmoid(ctx0, beta);

            // Reshape for KDA recurrence
            // {n_embd, n_tokens} -> {n_embd, n_seq_tokens, n_seqs}
            cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], n_seq_tokens, n_seqs);

            // Get SSM state and compute KDA recurrence using ggml_kda_scan
            ggml_tensor * ssm_states_all = mctx_cur->get_s_l(il);
            ggml_tensor * state = build_rs(inp_rs, ssm_states_all, hparams.n_embd_s(), n_seqs);
            state = ggml_reshape_4d(ctx0, state, head_dim, head_dim, n_head, n_seqs);

            // The Q/K L2 norm inside the delta rule uses 1e-6, NOT rms_norm_eps: transformers'
            // l2norm() defaults to eps=1e-6 to match FLA, while rms_norm_eps here is 1e-5.
            // Kimi-Linear's builder passes f_norm_rms_eps; copying that leaves a small flat
            // error on every KDA layer.
            Qcur = ggml_l2_norm(ctx0, Qcur, 1e-6f);
            Kcur = ggml_l2_norm(ctx0, Kcur, 1e-6f);

            // Choose between build_delta_net_chunking and build_delta_net_recurrent based on n_tokens
            auto attn_out = build_delta_net(Qcur, Kcur, Vcur, g1, beta, state, il);

            ggml_tensor * output = ggml_cont(ctx0, attn_out.first);
            ggml_tensor * new_state = attn_out.second;
            cb(output, "attn_output", il);
            cb(new_state, "new_state", il);

            // Update the recurrent states
            ggml_build_forward_expand(gf,
                                     ggml_cpy(ctx0, new_state,
                                              ggml_view_1d(ctx0, ssm_states_all, hparams.n_embd_s() * n_seqs,
                                                           kv_head * hparams.n_embd_s() * ggml_element_size(ssm_states_all))));

            // Output gating g2 = g_b(g_a(x))
            ggml_tensor * cur_2d = ggml_reshape_2d(ctx0, cur, cur->ne[0], n_seq_tokens * n_seqs);
            ggml_tensor * g_a = ggml_mul_mat(ctx0, layer.ssm_g_a, cur_2d);
            ggml_tensor * g2 = ggml_mul_mat(ctx0, layer.ssm_g_b, g_a);
            cb(g2, "g2 g_b(g_a(cur_2d))", il);
            g2 = ggml_reshape_3d(ctx0, g2, head_dim, n_head, n_seq_tokens * n_seqs);

            // Apply o_norm with sigmoid gating
            // Note: Kimi model uses sigmoid gating, not SiLU (despite FusedRMSNormGated default being swish)
            // Formula: output = RMSNorm(x) * sigmoid(g)
            ggml_tensor * attn_out_final = ggml_reshape_3d(ctx0, output, head_dim, n_head,  n_seq_tokens * n_seqs);
            ggml_tensor * normed = build_norm(attn_out_final, layer.ssm_o_norm, nullptr, LLM_NORM_RMS, il);
            cb(normed, "kda_normed", il);
            ggml_tensor * gate = ggml_sigmoid(ctx0, g2);
            ggml_tensor * gated = ggml_mul(ctx0, normed, gate);

            // Output projection
            gated = ggml_cont_2d(ctx0, gated, d_inner, n_tokens);
            cur = ggml_mul_mat(ctx0, layer.wo, gated);
            cb(cur, "kda_out", il);

        } else {
            // NoPE MLA. qk_rope_head_dim == 0, so there is no rotary tail to split off the
            // query and no k_pe to concatenate - the compressed KV is the whole key.
            ggml_tensor * q_a = ggml_mul_mat(ctx0, layer.wq_a, cur);
            q_a = build_norm(q_a, layer.attn_q_a_norm, NULL, LLM_NORM_RMS, il);
            ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.wq_b, q_a);

            // DSA indexer. Formulation validated against transformers in
            // tests/test-dsa-indexer.cpp (rel error 2.18e-06); see research/DSA_LLAMACPP.md.
            //
            // Gated OFF by default: consuming these scores means widening the KV row on this
            // layer to carry the indexer key, gate and valid flag, and the shipped GGUFs were
            // validated with these layers dense. Built here so the graph-side formulation lives
            // with the model rather than only in a test.
            if (hparams.dsa_enabled && layer.indexer_attn_k && layer.indexer_kpool_gate) {
                ggml_tensor * iscores = build_dsa_index_scores(cur, q_a, layer, il);
                if (iscores) {
                    cb(iscores, "dsa_index_scores", il);
                    ggml_build_forward_expand(gf, iscores);
                }
            }

            ggml_tensor * kv_cmpr = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);
            kv_cmpr = build_norm(kv_cmpr, layer.attn_kv_a_norm, NULL, LLM_NORM_RMS, il);

            if (layer.wk_b && layer.wv_b) {
                ggml_tensor * q_nope = ggml_reshape_3d(ctx0, Qcur, n_embd_head_k_mla, n_head, n_tokens);
                q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);                 // [hk, T, n_head]
                ggml_tensor * q_absorbed = ggml_mul_mat(ctx0, layer.wk_b, q_nope);
                q_absorbed = ggml_permute(ctx0, q_absorbed, 0, 2, 1, 3);         // [kv_lora, n_head, T]
                Qcur = ggml_cont(ctx0, q_absorbed);

                ggml_tensor * Kcur = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
                ggml_tensor * Vcur = Kcur;

                // DSA (opt-in): carry the indexer key and gate in the K row so decode can score
                // cached tokens. Q is zero-padded to match, so the extra dimensions contribute
                // exactly nothing to the attention scores.
                //
                // V is deliberately NOT widened and no longer aliases K: wv_b expands from the
                // compressed latent, so a wider V would feed it the indexer state as if it were
                // value content.
                if (hparams.dsa_enabled && layer.indexer_attn_k && layer.indexer_kpool_gate) {
                    const int64_t ihd = hparams.indexer_head_size;

                    ggml_tensor * ik = ggml_mul_mat(ctx0, layer.indexer_attn_k, cur);
                    ik = ggml_norm(ctx0, ik, 1e-6f);
                    ik = ggml_add(ctx0, ggml_mul(ctx0, ik, layer.indexer_k_norm),
                                  layer.indexer_k_norm_b);
                    ggml_tensor * ig = ggml_mul_mat(ctx0, layer.indexer_kpool_gate, cur);

                    ggml_tensor * extra = ggml_concat(ctx0, ik, ig, 0);          // [2*ihd, T]
                    extra = ggml_reshape_3d(ctx0, extra, 2*ihd, 1, n_tokens);
                    Kcur  = ggml_concat(ctx0, Kcur, extra, 0);                   // [kv_lora+2*ihd, 1, T]

                    ggml_tensor * pad = ggml_new_tensor_3d(ctx0, Qcur->type, 2*ihd, n_head, n_tokens);
                    pad = ggml_scale(ctx0, pad, 0.0f);
                    Qcur = ggml_concat(ctx0, Qcur, pad, 0);
                    cb(Kcur, "dsa_k_widened", il);
                }

                // Sparse selection when it applies, dense otherwise. build_attn_dsa returns
                // nullptr for every case it does not handle (prefill, short context, an
                // un-widened row), so the dense path stays the default and the fallback is a
                // plain null check rather than a duplicated condition that could drift.
                cur = build_attn_dsa(inp_attn_k, layer.wo, Qcur, Kcur, Vcur,
                                     cur, q_a, layer, layer.wv_b, kq_scale_mla, il);
                if (cur == nullptr) {
                    cur = build_attn(inp_attn_k, layer.wo, NULL, Qcur, Kcur, Vcur,
                                     nullptr, nullptr, layer.wv_b, kq_scale_mla, il);
                }
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

                cur = build_attn(inp_attn_kv, layer.wo, NULL, Qcur, Kcur, Vcur,
                                 nullptr, nullptr, nullptr, kq_scale_mla, il);
            }
            cb(cur, "mla_out", il);
        }

        streams = apply_mhc(site, cur, il);
        cb(streams, "mhc_after_attn", il);

        // ---------------- FFN site ----------------
        site = build_mhc(streams, layer.hc_ffn_fn, layer.hc_ffn_base, layer.hc_ffn_scale, il);

        cur = build_norm(site.collapsed, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            // Clamped SwiGLU, spelled out: build_ffn's fused SiLU path has no limit, and the
            // clamp only bites once activations exceed it - so a plain SILU dense FFN matches
            // on a small test and diverges on the real model.
            const float limit = hparams.swiglu_limit;
            ggml_tensor * g = ggml_mul_mat(ctx0, layer.ffn_gate, cur);
            ggml_tensor * u = ggml_mul_mat(ctx0, layer.ffn_up,   cur);
            g = ggml_clamp(ctx0, g, -INFINITY, limit);
            u = ggml_clamp(ctx0, u, -limit,    limit);
            cur = ggml_mul(ctx0, ggml_silu(ctx0, g), u);
            cur = ggml_mul_mat(ctx0, layer.ffn_down, cur);
        } else {
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

            const float limit = hparams.swiglu_limit;
            ggml_tensor * sg = ggml_clamp(ctx0, ggml_mul_mat(ctx0, layer.ffn_gate_shexp, cur), -INFINITY, limit);
            ggml_tensor * su = ggml_clamp(ctx0, ggml_mul_mat(ctx0, layer.ffn_up_shexp,   cur), -limit, limit);
            ggml_tensor * shexp = ggml_mul_mat(ctx0, layer.ffn_down_shexp,
                    ggml_mul(ctx0, ggml_silu(ctx0, sg), su));
            cur = ggml_add(ctx0, moe_out, shexp);
        }
        cb(cur, "ffn_out", il);

        cur = build_cvec(cur, il);

        streams = apply_mhc(site, cur, il);
        cb(streams, "l_out", il);
    }

    // Final collapse is an unweighted mean over the streams (reference:
    // Glm5NextTextHyperConnectionOutput.forward -> hidden_streams.mean(dim=2)).
    {
        ggml_tensor * sp = ggml_cont(ctx0, ggml_permute(ctx0, streams, 1, 0, 2, 3)); // [hc, n_embd, T]
        cur = ggml_reshape_2d(ctx0, ggml_sum_rows(ctx0, sp), n_embd, n_tokens);
        cur = ggml_scale(ctx0, cur, 1.0f/(float) hc);
        cb(cur, "mhc_collapse_out", -1);
    }

    // Token selection happens after the collapse rather than inside the last layer: the streams
    // are 3D and get_rows would have to index ne2. The saving that matters - not running the
    // vocab projection for non-output tokens - is preserved either way.
    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
