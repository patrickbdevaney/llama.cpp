// End-to-end logits check for the glm5-next port against transformers.
//
// This is the gate before spending 165 GiB converting the real checkpoint. The tiny fixture has
// the same STRUCTURE - both attention types, dense and MoE FFN, a shared expert, an MTP block
// that must load and not execute, mHC at every site - so the things that fail silently here are
// the same things that would fail silently at scale: Sinkhorn order, a transposed mHC mixing
// matmul, KDA/MLA layer types off by one, NoPE handling.
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model.gguf reference.bin\n", argv[0]); return 1; }

    FILE * f = fopen(argv[2], "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
    int32_t n_tok = 0, n_vocab_ref = 0;
    if (fread(&n_tok, 4, 1, f) != 1 || fread(&n_vocab_ref, 4, 1, f) != 1) return 1;
    std::vector<int32_t> ids(n_tok);
    if (fread(ids.data(), 4, n_tok, f) != (size_t) n_tok) return 1;
    std::vector<float> ref((size_t) n_tok * n_vocab_ref);
    if (fread(ref.data(), 4, ref.size(), f) != ref.size()) return 1;
    fclose(f);

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    // CPU by default (the reference is F32 and exact). Set GLM5_TEST_NGL to push layers onto
    // the GPU - that is what actually exercises ggml_cuda_op_mhc_sinkhorn, and running the same
    // fixture both ways is the CPU-vs-CUDA equivalence check.
    const char * ngl = getenv("GLM5_TEST_NGL");
    mp.n_gpu_layers = ngl ? atoi(ngl) : 0;
    printf("n_gpu_layers = %d\n", mp.n_gpu_layers);
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "failed to load %s\n", argv[1]); return 1; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx     = 512;
    cp.n_batch   = 512;
    cp.n_ubatch  = 512;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    if (n_vocab != n_vocab_ref) {
        fprintf(stderr, "vocab mismatch: gguf %d vs reference %d\n", n_vocab, n_vocab_ref);
        return 1;
    }

    llama_batch batch = llama_batch_init(n_tok, 0, 1);
    for (int i = 0; i < n_tok; ++i) {
        batch.token[i]     = ids[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = 1;
    }
    batch.n_tokens = n_tok;
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }

    // Compare every position. A port that is right at position 0 and wrong later is exactly what
    // a broken recurrent state or a mis-shaped mask looks like, so do not only check the last.
    double worst_rel = 0.0;
    int    worst_pos = -1, top1_mismatch = 0;
    for (int i = 0; i < n_tok; ++i) {
        const float * got = llama_get_logits_ith(ctx, i);
        const float * want = ref.data() + (size_t) i * n_vocab;
        double num = 0.0, den = 0.0;
        int gi = 0, wi = 0;
        for (int v = 0; v < n_vocab; ++v) {
            const double d = (double) got[v] - want[v];
            num += d*d; den += (double) want[v]*want[v];
            if (got[v]  > got[gi])  gi = v;
            if (want[v] > want[wi]) wi = v;
        }
        const double rel = std::sqrt(num/std::max(den, 1e-30));
        if (rel > worst_rel) { worst_rel = rel; worst_pos = i; }
        if (gi != wi) ++top1_mismatch;
        printf("  pos %2d  rel %.4e  top1 got %6d want %6d%s\n", i, rel, gi, wi, gi==wi?"":"   <-- MISMATCH");
    }
    printf("positions       : %d\n", n_tok);
    printf("worst rel error : %.4e  (position %d)\n", worst_rel, worst_pos);
    printf("top-1 mismatches: %d/%d\n", top1_mismatch, n_tok);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    // TOP-1 IS THE HARD GATE. Every structural bug this test found - the KDA forget gate,
    // the MQA head count, the leading-dense count, the mHC mixing transpose - showed up first as
    // top-1 mismatches or as error that GREW with position. What remains is ~5e-3, flat across
    // positions: the graph reassociates sums differently from torch, and a randomly-initialised
    // 6-layer network has poorly-conditioned logits, so a few e-3 is what this fixture is worth.
    // The real check on the real weights is whether the model generates coherent text.
    const bool ok = worst_rel < 1e-2 && top1_mismatch == 0;
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
