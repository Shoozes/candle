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

1. Trace the first CPU prefill boundary before preparing a successor packet.
   - What/why: The retained arithmetic logits differ before forced decode.
     Position 2 still selects `1113` in the pinned reference and `6986` in
     Candle. The Q8_K half-tie witness in [STATUS](STATUS.md) gives a concrete
     hypothesis to test with actual production activations. Packet `00ab6c5e`
     is consumed.
   - Where/how: After separately authorized diagnostic preparation, use the
     same 37 frozen prompt tokens on both CPU engines. Compare embedding rows,
     layer 0 operator normalization and `blk.0.shortconv.in_proj` in order.
     Include the packed prompt rows and the single tail row. Record per-row
     deltas and retain the first unequal row. If the projection differs, replay
     common normalized input in the original four-row group or tail path.
     Compare Q8_K bytes and scales before comparing the Q4_K dot. If these
     boundaries agree, extend to block outputs to find the first divergent layer.
   - Done when: Paired observations identify the earliest divergent operation
     and distinguish input drift, activation quantization and dot arithmetic.
     Preserve the frozen inputs, binaries and qualification limits. Use separate
     diagnostic artifacts. Builds and further model loads remain held pending
     their own authorization.
2. Resolve the CPU mismatch under the existing [QUALIFICATION](QUALIFICATION.md)
   criteria. Bind a new inert packet to the reviewed candidate and its binaries,
   frozen inputs, original limits and four comparison commands. Keep the consumed
   packet and its receipts available for comparison.
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
AI-edited: 2026-11-11 | agent=Codex/root | task=lfm2-moe-cpu-isolation | change=select paired prefill tracing after retained-evidence isolation
