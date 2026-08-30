// DSA Phase 3: attention over GATHERED cache rows must equal dense attention over the full cache
// masked to those same rows.
//
// This is the invariant that makes sparse attention safe to ship. softmax over a gathered subset
// and softmax over the full row with -inf everywhere outside the subset are the same function:
// both reduce to exp(s_i) / sum_{j in S} exp(s_j). If the gather, the mask gather, or the V
// alignment is wrong, the two disagree. Nothing else in the pipeline would catch it -- a
// mis-gathered attention still produces fluent text.
//
// Three specific hazards this pins down:
//   1. The cache is a STRIDED VIEW (kv_size > n_kv), so get_rows must read through nb1, not
//      assume a packed buffer. A packed-buffer assumption reads the right count of wrong rows.
//   2. The mask is [n_kv, T] and must be gathered along ne0, which get_rows cannot do -- it
//      selects along ne1. It has to go through a transpose, and a missing transpose silently
//      gathers along the token axis instead.
//   3. V is a VIEW of K at width kv_lora_rank. Gathered V rows must stay paired with the same
//      gathered K rows; any independent gather of V permutes value content against its weights.
#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

int main() {
    const int kv_lora = 4;
    const int ihd     = 2;
    const int kw      = kv_lora + 2*ihd;   // widened K row: latent + indexer key + gate
    const int n_head  = 2;
    const int T       = 3;                 // decode; >1 because MTP/DFlash spec decode submits several
                                           // tokens per step, and T==1 makes the mask transpose a no-op
    const int n_kv    = 24;
    const int kv_size = 40;                // cache is BIGGER than n_kv -> strided view
    const int kpool   = 4;
    const int n_pools = n_kv / kpool;      // 6
    const int select_k = 3;                // -> 12 of 24 tokens survive
    const int sel     = select_k * kpool;
    const float scale = 1.0f / std::sqrt((float) kv_lora);

    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    ggml_init_params ip = { (size_t) 256*1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    // ---- the cache, as llama_kv_cache actually lays it out: [kw, kv_size] ----
    ggml_tensor * cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kw, kv_size);
    for (int i = 0; i < kw*kv_size; ++i) ((float *) cache->data)[i] = nd(rng);

    // Query, already absorbed: [kv_lora, n_head, T], zero-padded out to kw so the indexer
    // dimensions contribute exactly nothing to the scores.
    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kw, n_head, T);
    memset(q->data, 0, ggml_nbytes(q));
    for (int h = 0; h < n_head; ++h)
        for (int t = 0; t < T; ++t)
            for (int d = 0; d < kv_lora; ++d)
                ((float *) q->data)[(t*n_head + h)*kw + d] = nd(rng);

    // ---- pool scores -> selected pools -> selected token indices ----
    std::vector<float> pool_scores(n_pools);
    for (auto & s : pool_scores) s = nd(rng);

    ggml_tensor * S = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_pools);
    memcpy(S->data, pool_scores.data(), ggml_nbytes(S));

    ggml_tensor * selp = ggml_top_k(ctx, S, select_k);                                  // I32
    ggml_tensor * self = ggml_cpy(ctx, selp, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, select_k));
    ggml_tensor * first = ggml_reshape_2d(ctx, ggml_scale(ctx, self, (float) kpool), 1, select_k);
    ggml_tensor * off   = ggml_reshape_2d(ctx, ggml_arange(ctx, 0.0f, (float) kpool, 1.0f), kpool, 1);
    ggml_tensor * toks  = ggml_add(ctx, ggml_repeat(ctx, first, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kpool, select_k)),
                                        ggml_repeat(ctx, off,  ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kpool, select_k)));
    ggml_tensor * idx = ggml_cpy(ctx, ggml_reshape_1d(ctx, toks, sel),
                                 ggml_new_tensor_1d(ctx, GGML_TYPE_I32, sel));

    // ================= GATHERED PATH =================
    // The cache as attention sees it, then gathered to the selected rows.
    ggml_tensor * k2d   = ggml_view_2d(ctx, cache, kw, n_kv, cache->nb[1], 0);
    ggml_tensor * k_sel = ggml_get_rows(ctx, k2d, idx);                                 // [kw, sel]
    ggml_tensor * kg    = ggml_reshape_3d(ctx, k_sel, kw, 1, sel);
    ggml_tensor * vg    = ggml_view_3d(ctx, kg, kv_lora, 1, sel, kg->nb[1], kg->nb[2], 0);

    ggml_tensor * qp  = ggml_permute(ctx, q,  0, 2, 1, 3);                              // [kw, T, n_head]
    ggml_tensor * kgp = ggml_permute(ctx, kg, 0, 2, 1, 3);                              // [kw, sel, 1]
    ggml_tensor * vgp = ggml_permute(ctx, vg, 0, 2, 1, 3);                              // [kv_lora, sel, 1]

    // The real decode mask is not all-visible: unoccupied cache cells and causal structure put
    // -inf inside pools the selector still picks. Gather it for real, through the transpose,
    // rather than assuming every selected row is visible.
    // Visibility VARIES BY TOKEN -- causal structure plus a few dead cache cells. If it did not
    // vary, every column of the mask would be identical and a gather along the wrong axis would
    // return the right numbers by accident.
    ggml_tensor * mask_base = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_kv, T);
    std::vector<std::vector<char>> vis(T, std::vector<char>(n_kv, 1));
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < n_kv; ++i) if (i > n_kv - T + t) vis[t][i] = 0;   // causal tail
        for (int i : {2, 3, 9, 17}) vis[t][i] = 0;                            // dead cells
        vis[t][(5 + t) % n_kv] = 0;                                           // token-specific
    }
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < n_kv; ++i)
            ((float *) mask_base->data)[t*n_kv + i] = vis[t][i] ? 0.0f : -INFINITY;

    // get_rows selects along ne1, but the mask needs gathering along ne0 -- hence the transpose
    // either side. Dropping either one gathers along the token axis and silently returns garbage
    // that is still finite and still the right shape.
    ggml_tensor * mask_g = ggml_cont(ctx, ggml_transpose(ctx,
                               ggml_get_rows(ctx, ggml_cont(ctx, ggml_transpose(ctx, mask_base)), idx)));

    ggml_tensor * kq_g = ggml_mul_mat(ctx, kgp, qp);                                    // [sel, T, n_head]
    kq_g = ggml_soft_max_ext(ctx, kq_g, mask_g, scale, 0.0f);
    ggml_tensor * out_g = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, vgp)), kq_g);

    // ================= DENSE PATH =================
    ggml_tensor * kd = ggml_reshape_3d(ctx, ggml_cont(ctx, k2d), kw, 1, n_kv);
    ggml_tensor * vd = ggml_view_3d(ctx, kd, kv_lora, 1, n_kv, kd->nb[1], kd->nb[2], 0);
    ggml_tensor * kdp = ggml_permute(ctx, kd, 0, 2, 1, 3);
    ggml_tensor * vdp = ggml_permute(ctx, vd, 0, 2, 1, 3);

    // -inf everywhere the selection did not land.
    std::vector<int> sel_pools(select_k);
    {
        std::vector<int> ord(n_pools); for (int i=0;i<n_pools;++i) ord[i]=i;
        std::partial_sort(ord.begin(), ord.begin()+select_k, ord.end(),
                          [&](int a,int b){ return pool_scores[a] > pool_scores[b]; });
        std::copy(ord.begin(), ord.begin()+select_k, sel_pools.begin());
    }
    std::vector<char> keep(n_kv, 0);
    for (int p : sel_pools) for (int j = 0; j < kpool; ++j) keep[p*kpool + j] = 1;

    ggml_tensor * mask_d = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_kv, T);
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < n_kv; ++i)
            ((float *) mask_d->data)[t*n_kv + i] = (keep[i] && vis[t][i]) ? 0.0f : -INFINITY;

    ggml_tensor * kq_d = ggml_mul_mat(ctx, kdp, qp);
    kq_d = ggml_soft_max_ext(ctx, kq_d, mask_d, scale, 0.0f);
    ggml_tensor * out_d = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, vdp)), kq_d);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out_g);
    ggml_build_forward_expand(gf, out_d);
    ggml_build_forward_expand(gf, idx);
    ggml_build_forward_expand(gf, mask_g);
    ggml_graph_compute_with_ctx(ctx, gf, 4);

    // ---- 1. the gather picked the pools the reference picked ----
    std::vector<int> want;
    for (int p : sel_pools) for (int j = 0; j < kpool; ++j) want.push_back(p*kpool + j);
    std::vector<int> got((int32_t *) idx->data, (int32_t *) idx->data + sel);
    std::vector<int> want_s = want, got_s = got;
    std::sort(want_s.begin(), want_s.end()); std::sort(got_s.begin(), got_s.end());
    if (want_s != got_s) {
        printf("FAIL: token selection mismatch\n  want:");
        for (int v : want_s) printf(" %d", v);
        printf("\n  got: ");
        for (int v : got_s) printf(" %d", v);
        printf("\n");
        return 1;
    }

    // ---- 2. the gathered mask carries the right visibility, in the right order ----
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < sel; ++i) {
            const float want_m = vis[t][got[i]] ? 0.0f : -INFINITY;
            const float got_m  = ((float *) mask_g->data)[t*sel + i];
            if (!((std::isinf(want_m) && std::isinf(got_m)) || want_m == got_m)) {
                printf("FAIL: gathered mask[t=%d][%d] (token %d) = %f, want %f\n",
                       t, i, got[i], got_m, want_m);
                return 1;
            }
        }
    }

    // ---- 3. gathered attention == dense attention masked to the same rows ----
    const int n = kv_lora * T * n_head;
    double maxerr = 0.0;
    for (int i = 0; i < n; ++i) {
        maxerr = std::max(maxerr, (double) std::fabs(((float *) out_g->data)[i] - ((float *) out_d->data)[i]));
    }

    printf("selected %d of %d tokens (%d of %d pools)\n", sel, n_kv, select_k, n_pools);
    printf("max |gathered - dense_masked| = %.3e\n", maxerr);

    // Not bit-exact: top_k returns the pools in no particular order, so the softmax denominator
    // and the V-weighted sum accumulate in a different order than the dense pass. The set is
    // identical, so the difference is pure summation order.
    if (!(maxerr < 1e-6)) { printf("FAIL: gathered attention diverges from dense\n"); return 1; }

    printf("OK\n");
    ggml_free(ctx);
    return 0;
}
