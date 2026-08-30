// Checks ggml_mhc_sinkhorn against the torch reference that was itself validated bit-for-bit
// against transformers' Glm5NextTextHyperConnection on real checkpoint weights.
//
// The failure this exists to catch is not "the numbers are a bit off" - it is the normalisation
// ORDER. Symmetric Sinkhorn, or an extra/missing column pass, still yields a plausible matrix
// with sensible sums, and the model that results is wrong in a way no later test localises.
// The reference output here is COLUMN-stochastic (column sums 1.0, row sums 0.98-1.02); a
// doubly-stochastic result means the order is wrong.
#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>

int main(int argc, char ** argv) {
    const char * path = argc > 1 ? argv[1]
        : "/home/patrickd/glm-5.3-reap/vendor/mhc/sinkhorn_case.bin";
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 1; }

    int32_t hc, n, iters; float eps;
    if (fread(&hc, 4, 1, f) != 1 || fread(&n, 4, 1, f) != 1 ||
        fread(&iters, 4, 1, f) != 1 || fread(&eps, 4, 1, f) != 1) return 1;

    std::vector<float> logits((size_t) hc*hc*n), want((size_t) hc*hc*n);
    if (fread(logits.data(), 4, logits.size(), f) != logits.size()) return 1;
    if (fread(want.data(),   4, want.size(),   f) != want.size())   return 1;
    fclose(f);
    printf("case: hc=%d slices=%d iters=%d eps=%g\n", hc, n, iters, (double) eps);

    ggml_init_params ip = { (size_t) 64*1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hc, hc, n);
    memcpy(a->data, logits.data(), ggml_nbytes(a));

    ggml_tensor * out = ggml_mhc_sinkhorn(ctx, a, iters, eps);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    const float * got = (const float *) out->data;
    double num = 0.0, den = 0.0, maxabs = 0.0;
    for (size_t i = 0; i < want.size(); ++i) {
        const double d = (double) got[i] - want[i];
        num += d*d; den += (double) want[i]*want[i];
        maxabs = std::max(maxabs, std::fabs(d));
    }
    const double rel = std::sqrt(num/den);

    // Structural check: the reference is column-stochastic, not doubly stochastic.
    double col_err = 0.0, row_spread = 0.0;
    for (int s = 0; s < n; ++s) {
        for (int c = 0; c < hc; ++c) {
            double cs = 0.0;
            for (int r = 0; r < hc; ++r) cs += got[(size_t) s*hc*hc + r*hc + c];
            col_err = std::max(col_err, std::fabs(cs - 1.0));
        }
        for (int r = 0; r < hc; ++r) {
            double rs = 0.0;
            for (int c = 0; c < hc; ++c) rs += got[(size_t) s*hc*hc + r*hc + c];
            row_spread = std::max(row_spread, std::fabs(rs - 1.0));
        }
    }
    printf("rel error vs reference : %.3e   max abs %.3e\n", rel, maxabs);
    printf("max |col sum - 1|      : %.3e  (must be ~0: column-stochastic)\n", col_err);
    printf("max |row sum - 1|      : %.3e  (must be NON-zero: not doubly stochastic)\n", row_spread);

    ggml_free(ctx);
    const bool ok = rel < 1e-5 && col_err < 1e-4 && row_spread > 1e-3;
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
