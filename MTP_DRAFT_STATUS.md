# GLM-5.3-Flash (`glm5-next`) MTP draft support — status and honest outcome

Fork-only work against ggml-org/llama.cpp. **Not proposed upstream**, deliberately: see
"What upstream has since done" below.

Base: upstream `761797ffd`. Branch: `glm5_next` (15 commits).

## What this branch adds that upstream does not have

- **`glm5-next` architecture**: mHC operator, hybrid KDA/MLA graph, vision tower, DSA indexer
  and sparse attention (Phases 2b/3), GGUF conversion, and the extra EOG tokens.
  Upstream has no `src/models/glm5-next.cpp` at all — GLM-5.3-Flash is unsupported there.
- **`glm5-next-mtp`**: the model's native MTP block (`blk.45`) run as a standalone draft model,
  plus `scripts/make_mtp_draft.py`, a byte-preserving repack (2.28 GiB at IQ3_M).
- **`llama-embedding --pooling none`** dumping raw fp16 hidden states with token ids beside them,
  which is what makes drafter-training capture possible at all.

## What upstream has since done (checked at `465e49b9c`, 2161 commits past this fork's base)

- **`common_spec_rollback` here is SUPERSEDED.** Upstream now handles recurrent/hybrid targets
  directly: `COMMON_CONTEXT_SEQ_RM_TYPE_RS` ("the context supports bounded partial sequence
  removal", gated on `llama_n_rs_seq(ctx) > 0`) plus single-position checkpoints in
  `common/speculative.cpp`. Their mechanism, not this one, is what a rebase should keep.
- **MTP drafting is now a first-class upstream feature** — `spec_type_draft_mtp`, an `--mtp`
  sidecar model, and download plumbing in `common/arg.cpp`. A future rebase should port
  `glm5-next-mtp` onto that interface rather than carrying this branch's bespoke path.

So the general fix in this branch was solved independently upstream. The model-specific work
(`glm5-next`, and its MTP block as a drafter) is what still has no upstream equivalent.

## The measured outcome, which is a NEGATIVE result

Verified 2026-09-03 with the **native, un-fine-tuned** `blk.45` against the REAP-50 pruned body.
No draft head was ever fine-tuned; every number below is the stock MTP block.

- **Acceptance: 46.5% at depth 2, 58.5% at depth 3** (48-token runs). 32-token runs range
  11%–74%, so do not read a single short run.
- **End-to-end it is a NET SLOWDOWN.** IQ3_M, `-ngl 24`: baseline **2.99 t/s**, depth 2 **1.73**,
  depth 3 **1.85**.

The draft is not at fault: it costs 5.3 ms/token and the state checkpoint ~11 ms, both negligible
against a 334 ms decode. The loss is entirely that **the target's cost per token barely improves
with batch size below 32 tokens**, so verifying a K+1 batch costs nearly K+1 decodes. With
llama.cpp's `op_offload_min_batch_size = 32`, a partially-offloaded model gets no batching benefit
in exactly the width range speculation needs.

**Do not fine-tune a draft head expecting this to turn positive on GGUF.** The head is not the
bottleneck.

### A trap worth repeating

Before the compatibility guard was added, `llama_decode` returned -1 and
`llama-speculative-simple` never checked it. Generation continued from stale logits while
**reporting acceptance for tokens it had never verified** — a plausible-looking fake 18–52% and
1.4–1.5x. Only the server called `is_compat`. If you measure speculation, check the decode return
value first; a silent failure here looks exactly like a success.

## Related, and where the rest lives

Quants, imatrix, the applied patch and `make_mtp_draft.py` are published at
`patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF`. `blk.45` already ships inside all five quants
(29 tensors each), so the draft weights were never missing — only the means to run them.
