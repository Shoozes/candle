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

1. Trace the corrected ShortConv mixer into layer-0 FFN normalization.
   - **Why** The [matched-gate replay](D:/Compact_Evidence/candle/runs/lfm2-moe-shortconv-out-proj-schedule-20261112-5648414d/summary.json)
     matched the entire output projection. The [matched-residual replay](D:/Compact_Evidence/candle/runs/lfm2-moe-ffn-norm-replay-20261112-42deff05/summary.json)
     showed both engines agree on each shared FFN residual. The stock
     mixer drift changed row-16 Q8_K scale blocks 0, 2, 3, 4, 5, 6 and 7.
   - **How** In an isolated candidate, compose the proven input and output
     Q4_K schedules with the reference three-tap accumulation on the same
     37 rows. Bind the frozen operands and run from the retained normalized
     input through the two projections and causal convolution. Use the
     identical retained embedding, FFN RMSNorm weight and epsilon. Compare
     the gate, mixer, residual, normalization and row-16 Q8_K blocks in
     order. Any native replay needs fresh admission under the revised
     component limits.
   - **Done when** Reproduce the retained controls and compare the full
     tensors through FFN RMSNorm. Report the first difference if present,
     and which of the seven row-16 scales recover the reference bits.
   - **Resume** The [prepared candidate](D:/Compact_Evidence/candle/runs/lfm2-moe-layer0-composed-replay-20261112-5648414d/summary.json)
     passed input binding. The attempt stopped before compilation because
     Cargo and Clippy were active. Reuse the bound source and controls in
     the next fresh admission.
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
AI-edited: 2026-11-12 | agent=Codex/root | task=lfm2-moe-layer0-composed-replay | change=record the verified inputs and next admission step
