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

1. Separate the ShortConv C-slice and convolution contributions to the
   output-projection input.
   - What/why: The [matched output-projection replay](D:/Compact_Evidence/candle/runs/lfm2-moe-shortconv-out-replay-20261112-42deff05-proof/summary.json)
     attributed the row-16 channel-557 mixer gap to the incoming F32 input.
     Both the input and the matched dot affected downstream FFN scales.
   - Where/how: Cross the two retained middle projection slices with the
     two retained convolution outputs in F32. Keep all 37 rows in order.
     Reproduce both original products, then compare Q8_K scales and bytes at
     row 16. Hold one verified dot kernel fixed when measuring the effect
     of each crossed input at channel 557. Reuse the verified libraries and
     witnesses. Preserve the frozen weight and original limits, and admit
     any new native children separately.
   - Done when: Reproduce the two original projection inputs and their
     saved fixed-kernel outputs. Identify which C or convolution changes
     account for the row-16 scale changes and channel-557 input effect.
2. Resolve the CPU mismatch under the existing [QUALIFICATION](QUALIFICATION.md)
   criteria. Bind a new inert packet to the reviewed candidate, a reference
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

---
AI-edited: 2026-11-12 | agent=Codex/root | task=lfm2-moe-shortconv-out-replay | change=select the crossed ShortConv input operand replay
