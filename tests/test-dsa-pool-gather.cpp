// DSA Phase 3: pool-granularity gather off the KV cache.
//
// build_attn_dsa does not gather individual tokens. It re-views the [D, n_kv] cache as
// [D*kpool, n_pools] and calls get_rows once with POOL ids, so one row of the re-view is a whole
// pool of kpool consecutive tokens. That removes all index arithmetic -- ggml has no ops for
// scaling I32 pool ids into token ids -- and pool granularity is DSA's own granularity, since
// select_k = indexer_top_k / kpool.
//
// The re-view is only legal because consecutive cache cells are contiguous in memory. That is
// true here (n_head_kv == 1, so n_embd_k_gqa == D and get_k's nb[2] is exactly D*es) but it is an
// assumption about layout rather than something the type system enforces, and if it were wrong
// the gather would return the right NUMBER of rows with the wrong contents -- which still decodes
// to fluent text. Hence this test.
//
// Three constructions are pinned down, all of which appear verbatim in build_attn_dsa:
//   1. K gather: [D, n_kv] -> [D*kpool, n_pools] -> get_rows(pool ids) -> [D, sel].
//   2. Mask gather: the mask is [n_kv, T] and get_rows selects along ne1, so it goes
//      transpose -> reshape to pool rows -> get_rows -> reshape -> transpose back.
//   3. Pool-level causal mask: slot 0 of each pool, taken as a strided ne0 view. Under a causal
//      mask a pool has a visible token iff its FIRST token is visible, so this is the exact
//      pool mask, and adding it to the scores keeps top_k from spending slots on dead pools.
//
// Values are chosen so that any off-by-one-pool or transposed gather produces a mismatch rather
// than a plausible-looking permutation.
#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int failures = 0;

static void check(bool ok, const char * what) {
    printf("%-58s %s\n", what, ok ? "OK" : "FAIL");
    if (!ok) failures++;
}

int main() {
    const int kv_lora  = 4;
    const int ihd      = 2;
    const int D        = kv_lora + 2*ihd;   // widened row: latent + indexer key + gate
    const int n_kv     = 24;
    const int kv_size  = 40;                // cache is BIGGER than n_kv, as at runtime
    const int kpool    = 4;
    const int n_pools  = n_kv / kpool;      // 6
    const int T        = 3;
    const int select_k = 3;
    const int sel      = select_k * kpool;  // 12 of 24 tokens survive

    // Deliberately not in ascending order: an implementation that ignores the ids and takes the
    // first select_k pools would pass an ascending list.
    const std::vector<int32_t> pool_ids = { 4, 0, 3 };

    ggml_init_params ip = { (size_t) 64*1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    // --- cache, and the strided [D, 1, n_kv, 1] view get_k would hand back -------------------
    ggml_tensor * cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, kv_size);
    float * cd = (float *) cache->data;
    for (int t = 0; t < kv_size; t++)
        for (int c = 0; c < D; c++)
            cd[t*D + c] = 1000.0f*t + c;          // token id is readable off any element

    const size_t es = ggml_type_size(cache->type);
    ggml_tensor * k = ggml_view_4d(ctx, cache, D, 1, n_kv, 1,
                                   (size_t) D*es, (size_t) D*es, (size_t) D*es*kv_size, 0);

    ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, select_k);
    memcpy(ids->data, pool_ids.data(), pool_ids.size()*sizeof(int32_t));

    // 1. K gather at pool granularity
    ggml_tensor * kpools = ggml_view_2d(ctx, k, D*kpool, n_pools, (size_t) (D*kpool)*es, 0);
    ggml_tensor * ksel   = ggml_get_rows(ctx, kpools, ids);
    ksel = ggml_reshape_4d(ctx, ksel, D, 1, sel, 1);

    // V is the leading kv_lora channels of the gathered row, exactly as in build_attn_dsa.
    ggml_tensor * vsel = ggml_view_4d(ctx, ksel, kv_lora, ksel->ne[1], ksel->ne[2], ksel->ne[3],
                                      ksel->nb[1], ksel->nb[2], ksel->nb[3], 0);
    ggml_tensor * vsel_c = ggml_cont(ctx, vsel);

    // --- mask ---------------------------------------------------------------------------------
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_kv, T);
    float * md = (float *) mask->data;
    for (int t = 0; t < T; t++)
        for (int j = 0; j < n_kv; j++)
            md[t*n_kv + j] = 100.0f*j + t;

    // 2. mask gather: transpose -> pool rows -> get_rows -> back
    ggml_tensor * mt = ggml_cont(ctx, ggml_transpose(ctx, mask));      // [T, n_kv]
    mt = ggml_reshape_2d(ctx, mt, T*kpool, n_pools);
    ggml_tensor * msel = ggml_get_rows(ctx, mt, ids);                  // [T*kpool, select_k]
    msel = ggml_reshape_2d(ctx, msel, T, sel);
    msel = ggml_cont(ctx, ggml_transpose(ctx, msel));                  // [sel, T]

    // 3. pool-level mask: slot 0 of each pool, strided along ne0
    ggml_tensor * pmask = ggml_cont(ctx,
        ggml_view_3d(ctx, mask, 1, n_pools, T, (size_t) kpool*es, mask->nb[1], 0));

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, ksel);
    ggml_build_forward_expand(gf, vsel_c);
    ggml_build_forward_expand(gf, msel);
    ggml_build_forward_expand(gf, pmask);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    // --- verify ------------------------------------------------------------------------------
    const float * kg = (const float *) ksel->data;
    bool ok_k = true;
    for (int s = 0; s < sel && ok_k; s++) {
        const int tok = pool_ids[s / kpool]*kpool + (s % kpool);
        for (int c = 0; c < D; c++)
            if (kg[s*D + c] != 1000.0f*tok + c) { ok_k = false; break; }
    }
    check(ok_k, "K: pool re-view gathers the right kpool tokens");

    const float * vg = (const float *) vsel_c->data;
    bool ok_v = true;
    for (int s = 0; s < sel && ok_v; s++) {
        const int tok = pool_ids[s / kpool]*kpool + (s % kpool);
        for (int c = 0; c < kv_lora; c++)
            if (vg[s*kv_lora + c] != 1000.0f*tok + c) { ok_v = false; break; }
    }
    check(ok_v, "V: leading kv_lora channels, same rows as K");

    const float * mg = (const float *) msel->data;
    bool ok_m = true;
    for (int t = 0; t < T && ok_m; t++)
        for (int s = 0; s < sel; s++) {
            const int tok = pool_ids[s / kpool]*kpool + (s % kpool);
            if (mg[t*sel + s] != 100.0f*tok + t) { ok_m = false; break; }
        }
    check(ok_m, "mask: transpose -> get_rows -> transpose is row-exact");

    const float * pg = (const float *) pmask->data;
    bool ok_p = true;
    for (int t = 0; t < T && ok_p; t++)
        for (int p = 0; p < n_pools; p++)
            if (pg[t*n_pools + p] != 100.0f*(p*kpool) + t) { ok_p = false; break; }
    check(ok_p, "pool mask: strided ne0 view picks slot 0 of each pool");

    // The gathered rows must be a SUBSET, not a reordering that happens to line up: pool 1, 2
    // and 5 were not selected and must not appear anywhere in the gather.
    bool ok_excl = true;
    for (int s = 0; s < sel; s++) {
        const int tok = (int) (kg[s*D] / 1000.0f);
        const int pool = tok / kpool;
        if (pool != pool_ids[s / kpool]) { ok_excl = false; break; }
    }
    check(ok_excl, "unselected pools are absent from the gather");

    ggml_free(ctx);
    printf("\n%s\n", failures ? "FAILED" : "PASS");
    return failures ? 1 : 0;
}
