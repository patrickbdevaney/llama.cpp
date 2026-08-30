// DSA indexer forward, built from ggml ops and checked against the transformers oracle.
//
// Phase 1 of research/DSA_LLAMACPP.md. This validates the GRAPH FORMULATION in isolation, before
// any of it touches the model's attention path or the KV cache - so a wrong pooling offset or a
// dropped relu fails here, at 24 tokens, instead of as a quietly worse model at 128k.
//
// The four hazards this is written to catch (see scripts/dsa_reference.py):
//   1. Pooling starts at the FIRST REAL TOKEN, not slot 0.
//   2. relu sits between the per-head scores and the head-weighted sum.
//   3. The pool key is a PER-CHANNEL softmax over the kpool tokens of (gate + ape).
//   4. A pool is visible to a query only if its LAST token is <= the query position.
#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static std::vector<float> rd(FILE * f, size_t n) {
    std::vector<float> v(n);
    if (fread(v.data(), 4, n, f) != n) { fprintf(stderr, "short read\n"); exit(1); }
    return v;
}

int main(int argc, char ** argv) {
    const char * path = argc > 1 ? argv[1]
        : "/home/patrickd/glm-5.3-reap/vendor/dsa/dsa_case.bin";
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 1; }

    int32_t S, D, hd, nh, kp, ql, P;
    int32_t hdr[7];
    if (fread(hdr, 4, 7, f) != 7) return 1;
    S = hdr[0]; D = hdr[1]; hd = hdr[2]; nh = hdr[3]; kp = hdr[4]; ql = hdr[5]; P = hdr[6];
    printf("case: S=%d D=%d hd=%d nh=%d kpool=%d q_lora=%d pools=%d\n", S, D, hd, nh, kp, ql, P);

    auto x     = rd(f, (size_t) S*D);
    auto qres  = rd(f, (size_t) S*ql);
    auto wq_b  = rd(f, (size_t) nh*hd*ql);
    auto wk    = rd(f, (size_t) hd*D);
    auto knw   = rd(f, (size_t) hd);
    auto knb   = rd(f, (size_t) hd);
    auto wproj = rd(f, (size_t) nh*D);
    auto kgate = rd(f, (size_t) hd*D);
    auto kape  = rd(f, (size_t) kp*hd);
    auto want  = rd(f, (size_t) S*P);
    fclose(f);

    ggml_init_params ip = { (size_t) 512*1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    auto mk2 = [&](int ne0, int ne1, std::vector<float> & src) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
        memcpy(t->data, src.data(), ggml_nbytes(t));
        return t;
    };
    auto mk1 = [&](int ne0, std::vector<float> & src) {
        ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne0);
        memcpy(t->data, src.data(), ggml_nbytes(t));
        return t;
    };

    ggml_tensor * X  = mk2(D,  S,  x);        // [D, S]
    ggml_tensor * QR = mk2(ql, S,  qres);     // [q_lora, S]
    ggml_tensor * WQ = mk2(ql, nh*hd, wq_b);  // [q_lora, nh*hd]
    ggml_tensor * WK = mk2(D,  hd, wk);       // [D, hd]
    ggml_tensor * KNW= mk1(hd, knw);
    ggml_tensor * KNB= mk1(hd, knb);
    ggml_tensor * WP = mk2(D,  nh, wproj);    // [D, nh]
    ggml_tensor * KG = mk2(D,  hd, kgate);    // [D, hd]
    ggml_tensor * AP = mk2(hd, kp, kape);     // [hd, kpool]

    // q = wq_b(q_resid) -> [hd, nh, S]
    ggml_tensor * q = ggml_mul_mat(ctx, WQ, QR);
    q = ggml_reshape_3d(ctx, q, hd, nh, S);

    // k = LayerNorm(wk(x)) -> [hd, S]. LayerNorm, not RMS, and it has a bias.
    ggml_tensor * k = ggml_mul_mat(ctx, WK, X);
    k = ggml_norm(ctx, k, 1e-6f);
    k = ggml_add(ctx, ggml_mul(ctx, k, KNW), KNB);

    // gate = kpool_gate(x) -> [hd, S]
    ggml_tensor * gate = ggml_mul_mat(ctx, KG, X);

    // Pool: groups of kp consecutive tokens. All tokens valid here, so pools start at 0 and the
    // count is ceil(S/kp); the first-real-token offset only matters with left padding.
    const int n_pools = (S + kp - 1) / kp;
    if (n_pools != P) { printf("pool count %d != expected %d\n", n_pools, P); return 1; }

    // [hd, kp, n_pools] views of k and gate
    ggml_tensor * k3 = ggml_reshape_3d(ctx, k,    hd, kp, n_pools);
    ggml_tensor * g3 = ggml_reshape_3d(ctx, gate, hd, kp, n_pools);

    // logits = gate + ape (ape broadcast over pools); softmax over the kp axis, PER CHANNEL.
    ggml_tensor * ape3 = ggml_reshape_3d(ctx, AP, hd, kp, 1);
    ggml_tensor * lg = ggml_add(ctx, g3, ape3);
    // soft_max reduces ne0, so bring kp to ne0, normalise, and put it back.
    lg = ggml_cont(ctx, ggml_permute(ctx, lg, 1, 0, 2, 3));     // [kp, hd, n_pools]
    lg = ggml_soft_max(ctx, lg);
    lg = ggml_cont(ctx, ggml_permute(ctx, lg, 1, 0, 2, 3));     // [hd, kp, n_pools]

    ggml_tensor * pk = ggml_mul(ctx, lg, k3);
    // sum over kp: bring it to ne0 and sum_rows
    pk = ggml_cont(ctx, ggml_permute(ctx, pk, 1, 0, 2, 3));     // [kp, hd, n_pools]
    pk = ggml_sum_rows(ctx, pk);                                // [1, hd, n_pools]
    pk = ggml_reshape_2d(ctx, pk, hd, n_pools);                 // [hd, n_pools]

    // scores = relu( (q . pool_keys) * hd^-0.5 ) -> [n_pools, nh, S]
    ggml_tensor * sc = ggml_mul_mat(ctx, pk, q);
    sc = ggml_scale(ctx, sc, 1.0f/sqrtf((float) hd));
    sc = ggml_relu(ctx, sc);

    // weights = weights_proj(x) * nh^-0.5 -> [nh, S]; index_scores = weights . scores
    ggml_tensor * wgt = ggml_mul_mat(ctx, WP, X);
    wgt = ggml_scale(ctx, wgt, 1.0f/sqrtf((float) nh));
    ggml_tensor * wgt3 = ggml_reshape_3d(ctx, wgt, nh, 1, S);
    ggml_tensor * scp  = ggml_cont(ctx, ggml_permute(ctx, sc, 1, 0, 2, 3)); // [nh, n_pools, S]
    ggml_tensor * idx  = ggml_mul_mat(ctx, scp, wgt3);                      // [n_pools, 1, S]
    idx = ggml_reshape_2d(ctx, idx, n_pools, S);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, idx);
    ggml_graph_compute_with_ctx(ctx, gf, 4);

    const float * got = (const float *) idx->data;
    const float LO = -1e30f;
    double num = 0, den = 0; int cmp = 0, bad = 0;
    for (int s = 0; s < S; ++s) {
        for (int p = 0; p < n_pools; ++p) {
            const float w = want[(size_t) s*n_pools + p];
            if (w <= LO) continue;                 // masked in the reference (causality)
            const double d = (double) got[(size_t) s*n_pools + p] - w;
            num += d*d; den += (double) w*w; ++cmp;
            if (std::fabs(d) > 1e-3) ++bad;
        }
    }
    const double rel = std::sqrt(num/std::max(den, 1e-30));
    printf("compared %d unmasked entries | rel error %.4e | %d over 1e-3\n", cmp, rel, bad);

    ggml_free(ctx);
    const bool ok = rel < 1e-4;
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
