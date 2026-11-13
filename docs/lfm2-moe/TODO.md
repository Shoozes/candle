# LFM2-MoE selected delivery

- What: Implement native safetensors and mixed F32/Q4_K/Q6_K GGUF LFM2-MoE on CPU
  and CUDA, then add the explicit Edge profile from the accepted plan.
- Why: Existing dense LFM2 loaders reject `lfm2moe` and have no expert routing.
- When: Native CPU oracle precedes GGUF CPU, which precedes CUDA and Edge proof.
- Where: New model modules and this independent overlay; Edge owns its adapter,
  profile, tokenizer/template, resource admission and output contracts.
- How: Reuse the existing LFM2 token mixers. Preserve sigmoid selection,
  selection-only expert bias, normalized unbiased weights and configured scale.
  Keep quantized experts resident and bound prefill/cache/scratch ownership.
- Done when: Current native and GGUF fixtures pass; CPU/CUDA behavior and cleanup
  pass; the opt-in Edge profile enforces 48 KiB / 12,288 total / 2,048 output
  tokens including reasoning; exact-source CLI/API qualification and the selected
  production Q4 CPU/CUDA proof pass; reviewed dependency adoption is reconciled.
- Verification: Pinned Transformers tiny oracle, focused Rust tests, native
  CPU/CUDA checks, required neighboring regressions and repository gates.
- Held: Production runs need a fresh exact grant, finite packet and admission.
  Builds, publication, Edge pin adoption, package assembly, installation,
  activation and release creation remain separately gated. Preserve consumer
  pins and frozen releases. Full native/Q8 production, Metal and 131k context
  qualification are outside this delivery.

## Remaining gates in order

1. Prepare a fresh inert zero-based layer-2 CPU diagnostic.
   - **Why** One composed offline layer-1 replay matched all ten
     checkpoints on the same 37 frozen IDs. Layer 2 is the first
     attention and sparse-MoE block in this GGUF.
     [Composed replay](D:/Compact_Evidence/candle/runs/lfm2-moe-layer1-composed-offline-20261113-60ce38d4/summary.json).
   - **How** Bind the computed `l_out-1` entry, SHA-256
     `0df2dfda89526966365f5787f063ccb07e6f4ad250f0aab004fb23be63946462`,
     the 37 frozen IDs, pinned llama.cpp, and exact layer-2 sources and
     weights. Require exact entry before weighted operator RMSNorm. Compare
     Q/K/V, Q/K normalization and RoPE, attention output, FFN RMSNorm,
     routing, sparse FFN and final residual in execution order. Stop at
     the first mismatch or failure.
   - **Capture contract** Bind each checkpoint's label, dtype, shape and
     axis order. Normalize attention tensors to token-first order using
     their strides (32 query heads, 8 KV heads, width 64). Compare the
     ordered top-four expert IDs as integers. Capture router logits,
     selection scores and IDs, gathered unbiased weights and normalized
     weights separately. Candle follows Transformers' `sum + 1e-6` rule.
     Pinned llama.cpp clamps the sum to at least `6.103515625e-5`.
   - **Limits** Carry forward the native layer-1 CPU diagnostic envelope
     from [packet `8060130a`](D:/Compact_Evidence/candle/runs/lfm2-moe-layer1-inert-20261113-72a53981/packet.json).
     Allow one attempt, zero retries, at most four children, 180 seconds
     per child and 900 seconds total. Preserve the 4 GiB build and 32 GiB
     model Job limits, 16 GiB floors for available RAM and commit headroom,
     two CPU threads, 12,288 context tokens and 128-token prefill chunks.
   - **Done when** A fresh inert packet binds these sources, inputs,
     checkpoint contracts, stops and admission controls. Model-free checks
     cover strided attention capture, integer expert order and the
     first-mismatch stop. The attempt remains unconsumed. Execution requires
     separate authorization and fresh admission.
2. Resolve the CPU mismatch under the existing [QUALIFICATION](QUALIFICATION.md)
   criteria. After authorization and admission, verify the exact layer-2
   entry before comparing the ordered checkpoints. Stop at the first
   mismatch or failure. Replay the first differing operands, capture a
   deterministic regression and integrate the justified correction into
   the runtime.
   Bind a new inert packet to the reviewed candidate, a reference
   binary built from the continuation-reporting probe source, frozen inputs,
   original limits and four comparison commands. Keep the consumed packet and
   its receipts available for comparison.
3. After a new exact owner grant and fresh admission, run the reviewed finite
   packet once under [QUALIFICATION](QUALIFICATION.md).
   - Done when: The pinned CPU reference, Candle CPU and RTX 4090 complete the
     two frozen cases with exact teacher-forced selections, four frozen-input
     comparisons, finite logit differences, observed peaks and terminal cleanup.
     Stop on any admission or execution failure without retry or relaxed limits.
4. After supplier parity, explicitly select the bounded Edge qualification
   admission through its held candidate gate. Prove CLI/API chat, strict JSON,
   cancellation, unload/reload and a successful recovery request on both devices.
5. After confirmed Candle source publication and separately authorized Edge
   adoption, reconcile only Edge's
   reviewed Candle pin and affected locks, rerun its selected graph/gates and
   record adoption. Other consumers retain their exact qualified identities.

Local implementation, native/GGUF/CUDA fixtures, opt-in profile, tokenizer/output
contracts, typed memory admission and ownership routes are implemented. Their
current proof is in STATUS and the compact run record. Those local results do
not close the remaining production/adoption criteria or authorize new work for
completed D1. No mandatory package, installation or activation task is selected.

Standalone LFM3 text starts with an independent numerical oracle and an
offline runner. Shared ShortConv optimization follows the corresponding
correctness baseline. Device-resident MoE execution follows validation of
the indexed CUDA path against the grouped implementation on identical weights.
LFM3 GGUF, lower precision and the broader model family have separate gates.

---
AI-edited: 2026-11-13 | agent=Codex/root | task=lfm2-moe-layer1-recheck | change=bind layer-2 capture and diagnostic limits
