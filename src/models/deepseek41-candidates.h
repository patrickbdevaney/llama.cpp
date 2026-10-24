#pragma once

// DeepSeek-V4.1 two-level top-k, level one: the candidate-block mask.
//
// Reference: inference/model.py, select_candidate_blocks() and Indexer.forward(). The candidate
// source layer (candidate_source_layer_id, 20) scores every compressed row with its indexer as
// usual, then groups the rows into blocks of candidate_block_size (8) and keeps the
// candidate_topk_blocks (2048) blocks with the highest best row. The block holding the query's
// newest row is always kept. Every later index source (24, 28, 32, 36) masks its own indexer score
// to those blocks before its top-k; the layers in between reuse that top-k (patch 0002), so they
// inherit the restriction. The source layer itself does not apply the mask.
//
// Only existing ggml ops are used (pool_1d, exp, pad, sub, scale, add, top_k, fill, set_rows), so
// every backend that runs the indexer already runs this. Kept in a header with no llama.cpp
// dependencies so tests/test-dsv41-candidates.cpp can build exactly this graph on the CPU.

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

static inline ggml_tensor * dsv41_cont(ggml_context * ctx, ggml_tensor * t) {
    return ggml_is_contiguous(t) ? t : ggml_cont(ctx, t);
}

// score: F32 [n_comp, n_q, 1, n_stream], the source layer's indexer score with the causal mask
//        already added (rows a query cannot reach are -inf).
// mask:  F32 [n_comp, n_q, 1, n_stream], that causal mask: 0 for row j < n_visible(query), else -inf.
//
// Returns F32 [n_blocks, n_q, 1, n_stream]: 0 for a kept block, -inf for a dropped one; add it to a
// later index source's score with dsv41_apply_candidate_blocks(). Returns nullptr when there are
// no more blocks than topk_blocks: every reachable block is then kept and the mask changes nothing
// (model.py: topk(min(topk_blocks, num_blocks)) then `top.values > -inf`), so the graph stays the
// one this tree had before the mask existed. `force` builds it anyway (tests only).
static inline ggml_tensor * dsv41_build_candidate_blocks(
        ggml_context * ctx,
        ggml_tensor * score,
        ggml_tensor * mask,
        int64_t topk_blocks,
        int64_t block_size,
        bool force = false) {
    GGML_ASSERT(score->type == GGML_TYPE_F32 && mask->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(score, mask));
    GGML_ASSERT(score->ne[2] == 1);
    GGML_ASSERT(topk_blocks > 0 && block_size > 0);

    const int64_t n_comp   = score->ne[0];
    const int64_t n_q      = score->ne[1];
    const int64_t n_stream = score->ne[3];

    // the compressed caches and their plans are padded to a multiple of 256 rows, so a block never
    // straddles the end; the loader only accepts block sizes that divide 256
    GGML_ASSERT(n_comp % block_size == 0);
    const int64_t n_blocks = n_comp / block_size;

    if (n_blocks <= topk_blocks && !force) {
        return nullptr;
    }
    const int64_t k = std::min(topk_blocks, n_blocks);

    // block score = its best row. An all -inf block comes out as -FLT_MAX (pool_1d starts from
    // -FLT_MAX on CPU and CUDA alike), which still ranks below every reachable block; and since
    // its rows are all -inf in every consumer's score too, keeping or dropping it is the same.
    ggml_tensor * bscore = ggml_pool_1d(ctx, dsv41_cont(ctx, score), GGML_OP_POOL_MAX, (int) block_size, (int) block_size, 0);

    // Pin the block holding the newest reachable row, (n_visible - 1) / block_size, as the reference
    // does with +inf: it is only partly filled and could otherwise lose to an older full block.
    // Visibility is monotone in the row (j < n_visible), so a block is visible iff its first row
    // is, and the pinned block is the last visible one: vis[b] - vis[b+1] is 1 there and 0
    // elsewhere. A query that sees nothing pins nothing (reference: last = -1).
    ggml_tensor * vis = ggml_reshape_4d(ctx, dsv41_cont(ctx, mask), block_size, n_blocks, n_q, n_stream);
    vis = ggml_view_4d(ctx, vis, 1, n_blocks, n_q, n_stream, vis->nb[1], vis->nb[2], vis->nb[3], 0);
    vis = ggml_reshape_4d(ctx, ggml_cont(ctx, vis), n_blocks, n_q, 1, n_stream);
    vis = ggml_exp(ctx, vis);                                   // 0 -> 1, -inf -> 0

    ggml_tensor * vis_next = ggml_pad(ctx, vis, 1, 0, 0, 0);    // [n_blocks + 1, ...], last = 0
    vis_next = ggml_view_4d(ctx, vis_next, n_blocks, n_q, 1, n_stream,
            vis_next->nb[1], vis_next->nb[2], vis_next->nb[3], ggml_element_size(vis_next));
    vis_next = ggml_cont(ctx, vis_next);

    // +1e30 instead of +inf: the pinned block's score is finite, so this ranks it above every
    // other block (index scores are O(1e2)) without inf arithmetic; only the kept set matters.
    ggml_tensor * pin = ggml_scale(ctx, ggml_sub(ctx, vis, vis_next), 1e30f);
    bscore = ggml_add(ctx, bscore, pin);

    ggml_tensor * top = ggml_cont(ctx, ggml_top_k(ctx, bscore, (int) k));   // I32 [k, n_q, 1, n_stream]

    // scatter 0 into a -inf block row at the picked blocks, one element per row (as
    // build_top_k_mask does for the index top-k)
    ggml_tensor * keep = ggml_fill(ctx, bscore, -INFINITY);
    keep = ggml_view_4d(ctx, keep, 1, n_blocks, n_q, n_stream, keep->nb[0], keep->nb[1], keep->nb[3], 0);

    ggml_tensor * top_3d = ggml_view_4d(ctx, top, k, n_q, n_stream, 1, top->nb[1], top->nb[3], n_stream*top->nb[3], 0);

    ggml_tensor * zeros = ggml_fill(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, k, n_q, n_stream), 0.0f);

    keep = ggml_set_rows(ctx, keep, zeros, top_3d);
    keep = ggml_view_4d(ctx, keep, n_blocks, n_q, 1, n_stream, keep->nb[2], keep->nb[3], keep->nb[3], 0);

    return keep;
}

// score: F32 [n_comp, n_q, 1, n_stream] of a later index source (causal mask already added).
// keep:  what dsv41_build_candidate_blocks() returned for the same plan (same n_comp, n_q).
// Every row of a dropped block becomes -inf (model.py: index_score.masked_fill(~candidates, -inf)).
static inline ggml_tensor * dsv41_apply_candidate_blocks(
        ggml_context * ctx,
        ggml_tensor * score,
        ggml_tensor * keep,
        int64_t block_size) {
    const int64_t n_comp   = score->ne[0];
    const int64_t n_q      = score->ne[1];
    const int64_t n_stream = score->ne[3];
    const int64_t n_blocks = keep->ne[0];

    GGML_ASSERT(score->ne[2] == 1);
    GGML_ASSERT(n_comp == n_blocks*block_size);
    GGML_ASSERT(keep->ne[1] == n_q && keep->ne[3] == n_stream);

    ggml_tensor * s = ggml_reshape_4d(ctx, dsv41_cont(ctx, score), block_size, n_blocks, n_q, n_stream);
    ggml_tensor * m = ggml_reshape_4d(ctx, dsv41_cont(ctx, keep), 1, n_blocks, n_q, n_stream);
    s = ggml_add(ctx, s, m);   // broadcast over the rows of each block

    return ggml_reshape_4d(ctx, s, n_comp, n_q, 1, n_stream);
}
