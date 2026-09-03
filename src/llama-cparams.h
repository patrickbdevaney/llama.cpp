#pragma once

#include "llama.h"

#include <cstdint>

#define LLAMA_MAX_SEQ 256

struct llama_cparams {
    uint32_t n_ctx;           // context size used during inference
    uint32_t n_ctx_seq;       // context for a single sequence
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    int32_t  n_threads;       // number of threads to use for generation
    int32_t  n_threads_batch; // number of threads to use for batch processing

    float rope_freq_base;
    float rope_freq_scale;

    uint32_t n_ctx_orig_yarn;
    // These hyperparameters are not exposed in GGUF, because all
    // existing YaRN models use the same values for them.
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;

    // Speculative MTP drafts (LLM_ARCH_GLM5_NEXT_MTP) consume the *target* model's hidden
    // state alongside the token being fed. It cannot ride in the ubatch: the batch carries
    // either tokens or embeddings, never both, and the draft needs its own token lookup. The
    // buffer lives in llama_context and is allocated once, so this pointer is stable for the
    // lifetime of the context and safe to capture at graph-build time.
    const float * mtp_h_prev;   // [n_embd, mtp_h_prev_n], row i belongs to batch token i
    uint32_t      mtp_h_prev_n;

    bool embeddings;

    // Extract embeddings only for the tokens that are already outputs, instead of promoting
    // every token in the batch to an output. Embedding models want the latter - they pool over
    // the whole sequence - but a speculative MTP draft only needs the hidden state of the token
    // it is about to draft from, and promoting a whole prompt would allocate an n_vocab logits
    // row per token (620 KB each at GLM-5.3's 154880 vocab) that nothing ever reads.
    bool embd_outputs_only;
    bool causal_attn;
    bool offload_kqv;
    bool flash_attn;
    bool auto_fa;
    bool fused_gdn_ar;       // use fused gated delta net (autoregressive)
    bool fused_gdn_ch;       // use fused gated delta net (chunked)
    bool auto_fgdn;
    bool no_perf;
    bool warmup;
    bool op_offload;
    bool kv_unified;
    bool pipeline_parallel;

    enum llama_pooling_type pooling_type;

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;
};
