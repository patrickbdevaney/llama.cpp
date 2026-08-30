#include "mhc.cuh"

// mHC Sinkhorn (GLM-5.3-Flash hyper-connections).
//
// Each [hc, hc] slice is independent and tiny - hc is 4, so 16 floats - and the work is the
// 2*iters reductions over it, not the data. So: one THREAD per slice, whole matrix in registers,
// no shared memory and no cross-thread reduction. A warp-per-slice layout would spend more on
// shuffles than the arithmetic costs.
//
// The normalisation order is load-bearing and is NOT symmetric Sinkhorn: one column pass, then
// (iters - 1) full (row, column) passes, leaving a COLUMN-stochastic matrix. This must stay
// bit-comparable with the CPU path in ggml-cpu/ops.cpp - expf, not __expf, for that reason.

#define MHC_MAX_HC 8

static __global__ void mhc_sinkhorn_f32(
        const float * __restrict__ src,
              float * __restrict__ dst,
        const int hc, const int nslices, const int iters, const float eps) {

    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nslices) {
        return;
    }

    const int n = hc*hc;
    float m[MHC_MAX_HC*MHC_MAX_HC];

    const float * s = src + (size_t) i*n;

    // softmax over ne0 (torch's last dim), then + eps
    for (int r = 0; r < hc; ++r) {
        const float * sr = s + r*hc;
        float mx = -INFINITY;
        for (int c = 0; c < hc; ++c) {
            mx = fmaxf(mx, sr[c]);
        }
        float sum = 0.0f;
        for (int c = 0; c < hc; ++c) {
            const float e = expf(sr[c] - mx);
            m[r*hc + c] = e;
            sum += e;
        }
        const float inv = 1.0f/sum;
        for (int c = 0; c < hc; ++c) {
            m[r*hc + c] = m[r*hc + c]*inv + eps;
        }
    }

    // COLUMN normalisation first
    for (int c = 0; c < hc; ++c) {
        float sum = 0.0f;
        for (int r = 0; r < hc; ++r) {
            sum += m[r*hc + c];
        }
        const float inv = 1.0f/(sum + eps);
        for (int r = 0; r < hc; ++r) {
            m[r*hc + c] *= inv;
        }
    }

    // then (iters - 1) full (row, column) passes
    for (int it = 1; it < iters; ++it) {
        for (int r = 0; r < hc; ++r) {
            float sum = 0.0f;
            for (int c = 0; c < hc; ++c) {
                sum += m[r*hc + c];
            }
            const float inv = 1.0f/(sum + eps);
            for (int c = 0; c < hc; ++c) {
                m[r*hc + c] *= inv;
            }
        }
        for (int c = 0; c < hc; ++c) {
            float sum = 0.0f;
            for (int r = 0; r < hc; ++r) {
                sum += m[r*hc + c];
            }
            const float inv = 1.0f/(sum + eps);
            for (int r = 0; r < hc; ++r) {
                m[r*hc + c] *= inv;
            }
        }
    }

    float * d = dst + (size_t) i*n;
    for (int k = 0; k < n; ++k) {
        d[k] = m[k];
    }
}

void ggml_cuda_op_mhc_sinkhorn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(src0->ne[0] == src0->ne[1]);
    GGML_ASSERT(src0->ne[0] <= MHC_MAX_HC);

    int   iters;
    float eps;
    memcpy(&iters, (const int32_t *) dst->op_params + 0, sizeof(int));
    memcpy(&eps,   (const float   *) dst->op_params + 1, sizeof(float));

    const int hc      = src0->ne[0];
    const int nslices = ggml_nelements(src0) / (hc*hc);

    const int block = 256;
    const int grid  = (nslices + block - 1)/block;

    mhc_sinkhorn_f32<<<grid, block, 0, ctx.stream()>>>(
        (const float *) src0->data, (float *) dst->data, hc, nslices, iters, eps);
}
