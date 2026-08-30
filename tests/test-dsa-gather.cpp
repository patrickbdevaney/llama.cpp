// Phase 3 building block: expand selected POOL indices into TOKEN indices, in ggml.
//
// This is the step that looked like a blocker. ggml_top_k returns I32 and ggml has no integer
// add or scale, so pool p -> tokens [p*kpool, p*kpool + kpool) seemed inexpressible. It is not:
// ggml_cpy converts I32 <-> F32, so the arithmetic goes through float and back.
//
// Checked here against a scalar reference because an off-by-one in this expansion selects
// neighbouring tokens - which still produces fluent output, and would be invisible until someone
// measured long-context retrieval carefully.
#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    const int n_pools  = 6;
    const int kpool    = 4;
    const int select_k = 3;

    // Scores over pools; the top-3 are pools 4, 1, 5 (values 9, 7, 6).
    const std::vector<float> scores = { 2.0f, 7.0f, 1.0f, 0.5f, 9.0f, 6.0f };

    ggml_init_params ip = { (size_t) 32*1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * S = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_pools);
    memcpy(S->data, scores.data(), ggml_nbytes(S));

    // 1. top-k -> I32 pool indices, "in no particular order"
    ggml_tensor * sel = ggml_top_k(ctx, S, select_k);

    // 2. I32 -> F32 so arithmetic is available at all
    ggml_tensor * self = ggml_cpy(ctx, sel, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, select_k));

    // 3. pool p -> first token p*kpool, then broadcast-add arange(0..kpool-1)
    ggml_tensor * first = ggml_scale(ctx, self, (float) kpool);
    first = ggml_reshape_2d(ctx, first, 1, select_k);
    ggml_tensor * ar = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kpool, 1);
    for (int i = 0; i < kpool; ++i) ((float *) ar->data)[i] = (float) i;

    ggml_tensor * base = ggml_repeat(ctx, first,
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kpool, select_k));
    ggml_tensor * tok = ggml_add(ctx, base, ar);            // [kpool, select_k]

    // 4. back to I32, flattened - this is what ggml_get_rows consumes
    ggml_tensor * flat = ggml_reshape_1d(ctx, tok, kpool*select_k);
    ggml_tensor * toki = ggml_cpy(ctx, flat,
        ggml_new_tensor_1d(ctx, GGML_TYPE_I32, kpool*select_k));

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, toki);
    ggml_build_forward_expand(gf, sel);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    const int32_t * pools = (const int32_t *) sel->data;
    const int32_t * got   = (const int32_t *) toki->data;

    std::vector<int> want;
    printf("selected pools:");
    for (int i = 0; i < select_k; ++i) {
        printf(" %d", pools[i]);
        for (int j = 0; j < kpool; ++j) want.push_back(pools[i]*kpool + j);
    }
    printf("   (expected top-3 of {2,7,1,0.5,9,6} = pools 4,1,5 in some order)\n");

    bool ok = true;
    printf("expanded tokens:");
    for (size_t i = 0; i < want.size(); ++i) {
        printf(" %d", got[i]);
        if (got[i] != want[i]) ok = false;
    }
    printf("\n");

    // The selection itself must be the right SET, order aside.
    std::vector<int> sp(pools, pools + select_k);
    std::sort(sp.begin(), sp.end());
    const std::vector<int> expect_pools = {1, 4, 5};
    if (sp != expect_pools) { printf("wrong pools selected\n"); ok = false; }

    ggml_free(ctx);
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
