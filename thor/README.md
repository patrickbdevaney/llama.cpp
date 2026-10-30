# DeepSeek-V4.1-Flash REAP-192 on Jetson AGX Thor with this llama.cpp branch

This branch (`deepseek-v4.1-reap-thor`) is upstream llama.cpp [PR #28696](https://github.com/ggml-org/llama.cpp/pull/28696), which adds the `deepseek41` architecture, plus the commits needed to run [`patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-GGUF`](https://huggingface.co/patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-GGUF) fully on the GPU of one 128 GB Jetson AGX Thor, with image input and DSpark speculative decoding.

```bash
git clone -b deepseek-v4.1-reap-thor https://github.com/patrickbdevaney/llama.cpp && cd llama.cpp
thor/build.sh                       # CUDA, sm_110 (CUDA_ARCH=89 etc. for other GPUs)
hf download patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-GGUF --local-dir ~/models/DeepSeek-V4.1-Flash-1017-REAP192-GGUF
# Engram row files: see "Engram row files" below; put them at $MODEL_DIR/engram_rows or set ENGRAM_DIR
thor/start.sh                       # http://127.0.0.1:8080: text + images, 16K context
DSPARK=1 thor/start.sh              # + DSpark speculative decoding (text)
```

## Measured on Thor (2026-10-30)

These numbers come from `thor/start.sh` at its defaults on a Jetson AGX Thor (CUDA 13.0), over `/v1/chat/completions`, with greedy sampling and one stream.

| | `thor/start.sh` | `DSPARK=1 thor/start.sh` |
|---|---:|---:|
| decode, 256-token chat answers | 12.3–12.5 tok/s | **16.6–19.9 tok/s** (1.33–1.61×) |
| DSpark acceptance (3 drafted per step) | | 55–72% |
| prefill, 1.8K-token prompt | 159 tok/s | |
| prefill, 5.9K-token prompt | 147 tok/s | |
| image + question (215 prompt tokens) | 84 tok/s prompt, 12.4 tok/s answer | |
| memory left while serving | 25 GiB of 122 | 16 GiB |

- **Image input works.** For the [`test-1.jpeg`](../tools/mtmd/test-1.jpeg) newspaper page, the model reads the masthead, the date ("MONDAY, JULY 21, 1969"), the headline ("MEN WALK ON MOON"), the sub-headline and both column headlines correctly. A low-resolution 531×229 crop of the same page made it loop while second-guessing the masthead, so give it legible images.
- **DSpark output is not byte-identical to plain decoding.** Every token is still the target model's greedy choice. But the batched verify pass sums in a different order from single-token decode, so a near-tie can flip ("…with the previous one" vs "…with the last one…"). This is llama.cpp's usual speculative behaviour, not a drafter error.
- **The pure-CUDA server is about 2× faster.** [`patrickbdevaney/deepseek-v4.1-flash-cuda-server`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-cuda-server) runs the separate `thor-v1` build at 25.0 tok/s decode and 739 tok/s prefill on the same box. It is text only.

## `thor/start.sh` settings

| variable | default | |
|---|---|---|
| `MODEL_DIR` | `~/models/DeepSeek-V4.1-Flash-1017-REAP192-GGUF` | the 3 `L3` shards, plus the optional `mmproj-…-bf16.gguf` and `dspark-…-Q8_0.gguf` |
| `ENGRAM_DIR` | `$MODEL_DIR/engram_rows` | `engram_L1.v41rows`, `engram_L14.v41rows`, on local NVMe |
| `CTX` | `16384` | see "Context length" below |
| `VISION` | `1` (`0` with `DSPARK=1`) | load the mmproj if it is present |
| `DSPARK` | `0` | `1` adds `-md dspark-… --spec-type draft-dspark`; measured text-only, so `VISION` defaults to 0 in this mode |
| `DRAFT_N` | `3` | tokens drafted per step |
| `HOST`, `PORT` | `127.0.0.1`, `8080` | |
| `EXTRA` | | further `llama-server` flags |

The script runs `llama-server -ngl 999 -np 1 --jinja`. It refuses to start beside another model-loading process, because on Thor's unified memory two such processes can hard-reboot the box. It also refuses row files on removable media.

## Engram row files

V4.1's two Engram tables (about 101 GB each) are not in the GGUF. They are read per token with O_DIRECT from two `.v41rows` files, which you build once from base-model shards 47 and 48. Use [`ops/s09_engram_rowfile.py`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/ops/s09_engram_rowfile.py) `build`, then `verify --full`; the format is in [`ENGRAM_ROWFILE.md`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/llamacpp/ENGRAM_ROWFILE.md). The same files serve the CUDA server. `LLAMA_ENGRAM_ROWFILE_DIR` points the model at them, and `start.sh` sets it.

## Commits on top of PR #28696

| commit | |
|---|---|
| keep hyper-connection tensors F32 when quantizing | mHC mixing needs F32 |
| reader layers reuse their index source's top-k | V4.1 cross-layer index sharing |
| Engram O_DIRECT row-store input | `LLAMA_ENGRAM_ROWFILE_DIR` |
| REAP heal gain on the routed-expert sum | `deepseek41.reap.routed_gain` |
| export the V4.1 DSpark drafter (`--dspark`) + target-layer fix | `convert_hf_to_gguf.py --dspark`; V4.1 taps layer inputs |
| clamp `n_ctx_seq` to 16384 | until the candidate-block mask is validated |
| two-level candidate-block mask | lifts the clamp when `deepseek41.candidate.*` is set (validated at 32K, below) |
| indexer score transposes as two tiled copies | 32K prompts: from more than 2 h to 14.6 min, bit-identical |
| vision projector `deepseek41v` | from PR #30817; Engram off on image ubatches, `bias_vl` on image rows |
| mmproj export for `DeepseekV41ForCausalLM` | `convert_hf_to_gguf.py --mmproj` |

## Context length

Up to 16,384 tokens the model runs as published. Above that, V4.1 attends through a two-level candidate-block selection: 2048 blocks of 8 rows, chosen at layer 20. That selection is on this branch. The published GGUF predates the metadata keys that switch it on, so `CTX=32768 thor/start.sh` passes them with `--override-kv`. Without them, the context is clamped to 16,384.

**Validated at 32K (2026-10-30).** On a 32,767-token held-out row, teacher-forced against the CUDA engine as teacher:

| positions | mask on | mask off | on − off, 95% block-bootstrap CI |
|---|---|---|---|
| 12,287–16,382 | KL 0.0563 | KL 0.0563 | identical at every position |
| 16,384–32,766 | KL 0.1050, top-1 0.985 | KL 0.1044, top-1 0.984 | KL +0.00055 [−0.00001, +0.00122]; NLL +0.00024 [−0.00024, +0.00081] |

At 32K the mask is within noise of running without it. That is expected: it keeps 2048 of 4096 blocks, and the indexer's top-512 rows almost always fall inside them. It matches the reference model's semantics and should matter more at 64K and beyond, which has not been measured here.

Long prompts are slow on this branch. One 32K-token prompt takes about 14.6 minutes to process, against about 1 minute for 8K, because the indexer and sparse attention run as dense ggml ops. Before the transpose fix in `deepseek41 : indexer score transposes…`, a single 32K prompt did not finish in 2 hours. For long contexts, the CUDA server prefills at 580–740 tok/s.

## Validation

- **Text.** On 180,224 held-out tokens, teacher-forced against the unpruned model: KL 0.508, top-1 agreement 0.773 (details on the GGUF card).
- **Vision encoder.** Golden-tested on CPU and CUDA against the reference torch ViT + Aligner in f32. Every token row is at least as close as the reference's own bf16 path.
- **DSpark drafter.** All 78 tensors are byte-identical to the pinned reference drafter, and every reference key is present.
