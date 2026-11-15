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

1. Produce the maintained Candle candidate for connected layer-2 CPU correctness.
   - **Next bounded step.** Seal an inert successor to unused `078aa139`
     with the verified SwiGLU and down controls. Preserve all 37 rows,
     groups, limits, exact entry, Q/K RoPE and top-K. Keep Candle's computed
     path to `layer_out` and both engines' weighting, merge and residual checks.
     Done when the seal verifies, a wrong digest fails, and cleanup is
     proven with zero probe dispatches. [Down replay receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-down-cross-20261115-62579e67/summary.json).
   - Connected execution needs exact authorization for the successor,
     fresh admission and the whole-model hash. Stop at the first
     unexplained difference or failure.
   - **Then integrate** Use the connected proof through `layer_out` to
     select demonstrated corrections in maintained code. Reuse operator proof
     where dtype, layout, width, group size and dispatch match. Preserve
     [both routing formulas](DECISIONS.md#layer-2-routing-comparison).
   - **Done when** The connected layer passes its declared comparisons, the
     maintained candidate passes its relevant regressions and fixture gates,
     and a new inert packet binds its exact source and binaries, the
     continuation-reporting reference binary, frozen inputs, original limits
     and four comparison commands. Preserve the consumed packet and receipts.
2. After a new exact owner grant and fresh admission, run the reviewed finite
   packet once under [QUALIFICATION](QUALIFICATION.md).
   - Done when: The pinned CPU reference, Candle CPU and RTX 4090 complete the
     two frozen cases with exact teacher-forced selections, four frozen-input
     comparisons, finite logit differences, observed peaks and terminal cleanup.
     Stop on any admission or execution failure without retry or relaxed limits.
3. After supplier parity, explicitly select the bounded Edge qualification
   admission through its held candidate gate. Prove CLI/API chat, strict JSON,
   cancellation, unload/reload and a successful recovery request on both devices.
4. After confirmed Candle source publication and separately authorized Edge
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
AI-edited: 2026-11-13 | agent=Codex/root | task=lfm2-moe-layer2-attempt | change=select inert successor with attempt marker after final reference admission
AI-edited: 2026-11-13 | agent=Codex/root | task=lfm2-moe-layer2-successor | change=route the held finite trace to the sealed successor
AI-edited: 2026-11-13 | agent=Codex/root | task=lfm2-moe-layer2-cpu-trace | change=queue the bounded query projection replay
AI-edited: 2026-11-13 | agent=Codex/root | task=lfm2-moe-layer2-query-offline | change=queue the isolated reference query schedules
AI-edited: 2026-11-13 | agent=Codex/root | task=lfm2-moe-layer2-query-schedule | change=bind the next replay-only packet to the existing build
AI-edited: 2026-11-13 | agent=Codex/root | task=lfm2-moe-layer2-query-replay | change=queue the bounded layer-2 key projection replay
AI-edited: 2026-11-13 | agent=Codex/root | task=lfm2-moe-layer2-key-replay | change=queue the bounded layer-2 value projection replay
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-value-replay | change=queue the import correction and new single-attempt value replay
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-value-recovery | change=queue the query and key normalization replay
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-qk-norm | change=queue the query and key RoPE replay
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-rope-review | change=bind the RoPE handoff to the admission receipt and fresh run directory
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-rope-successor | change=record the successor admission stop
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-rope-replay | change=queue the causal attention values replay
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-attn-values | change=queue the bounded attention output projection replay
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-attn-output | change=queue post-mixer residual and weighted FFN RMSNorm
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-ffn-norm | change=queue the layer-2 F32 router projection replay
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-router-review | change=bind the router control, exact matrix shape, and CPU dispatch
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-router-work | change=record the verified unsealed router preparer and next admission
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-closeout | change=make the verified progress and local replay authority explicit
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-router-admission-stop | change=record the consumed attempt and replace the stale replay command
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-router-successor-review | change=bind the inert successor to the retained reference build and original gates
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-router-successor | change=queue execution of the exact inert router successor
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-router-replay | change=queue the layer-2 sigmoid replay on matched router logits
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-router-sigmoid | change=queue expert selection bias on matched probabilities
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-router-sigmoid-review | change=bind the expert-selection bias pass-down to the verified GGUF range and capture
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-router-bias | change=queue ordered top-four expert IDs on the matched scores
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-router-bias-closeout | change=bind the queued top-four replay to the original limits and row-36 witness
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-router-top4 | change=queue the unbiased expert-weight gather on the matched IDs
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-router-gather | change=route the next gather admission check to the verified unused packet
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-gather-review | change=bind the next gather attempt to the verified checkout and controller
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-gather-replay | change=queue raw expert-weight sums before normalization
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-raw-sum | change=queue the raw sum witness and native attention before connected layer proof
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-maintained-reduction | change=record scoped CPU reduction and pending exact replay
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-router-replay | change=record exact raw sums and the classified denominator
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-router-normalized | change=record exact engine-specific normalization and queue expert projections
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-expert-projection | change=record the consumed harness compile stop
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-gate-witness | change=record the first grouped gate mismatch and queue the exact dot trace
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-expert0-gate-dot | change=record the compiler admission stop and retain the reference object
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-expert0-dot-resume | change=record exact Q8_K match and route the indexed reference discrepancy
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-expert0-indexed | change=record native CPU repack witness and matching Q8_K scratch
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-expert0-repack-arithmetic | change=queue a grouped expert projection replay with the traced schedule
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-expert-gate-up-repack | change=select the bounded expert SwiGLU replay
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-expert-swiglu | change=select the isolated Q6_K expert down replay
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-expert-q6k-down | change=select the isolated per-expert weighting replay
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-expert-weighting | change=select the isolated sparse expert merge
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-expert-merge | change=select maintained Candle attention before connected layer-2 parity
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-candle-attention-baseline | change=retain the bound attention probe for a fresh attempt
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-candle-attention-resume | change=select the first causal attention mismatch trace
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-attention-review | change=bind the attention trace to the failing layouts and output controls
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-attention-first-term | change=route the selected trace to the prepared successor
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-closeout | change=require a new revision binding before the unused offline attempt
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-attention-qk | change=queue the bounded F32 dot accumulation comparison
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-qk-dot | change=queue downstream attention on common raw scores
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-shared-scores | change=queue the connected comparison from exact layer entry
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-connected-prep | change=route the review to the inert connected CPU packet
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-connected-admission | change=retain the unused attempt and link its refused windows
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-connected-attempt | change=queue the first unexplained attention value on shared operands
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-attention-channel1 | change=queue a shared-score comparison across the full attention tensor
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-attention-all-shared | change=queue the first differing value accumulation on retained operands
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-value-reduction-channel8 | change=queue full value reduction proof on shared probabilities
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-value-reduction-all | change=link the prepared full-reduction successor
AI-edited: 2026-11-15 | agent=Codex/root | task=lfm2-moe-layer2-value-reduction-review | change=specify native control coverage and full accumulation gate
AI-edited: 2026-11-15 | agent=Codex/root | task=lfm2-moe-layer2-value-reduction-work | change=queue inert replay-only packet from retained builds
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-replay-only-prep | change=route the exact packet to review and authorization
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-value-replay | change=close the isolated comparison and select connected preparation
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-connected-prep | change=select review of the inert successor
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-connected-review | change=record local review and exact execution gate
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-connected-attempt | change=select the first mixer mismatch
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-mixer-replay | change=select connected successor with both verified mixer controls
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-mixer-review | change=record staged comparator and pending build
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-mixer-seal | change=select the inert successor
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-ffn-norm-stop | change=select matched-input FFN normalization replay
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-ffn-norm-cross | change=select inert connected packet with both normalization controls
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-connected-norm-prep | change=select review of the sealed normalization successor
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-connected-router-stop | change=queue a shared-input replay of the first router logit difference
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-router-cross | change=select an inert successor with both exact router controls
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-connected-router-prep | change=select review of the sealed router successor
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-connected-router-review | change=order the held connected attempt before maintained integration
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-connected-sigmoid-stop | change=select shared-input sigmoid proof before resuming connected layer 2
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-sigmoid-cross | change=select an inert successor with both sigmoid controls
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-sigmoid-dropin | change=route the sealed sigmoid successor to review and its held execution gate
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-connected-bias-stop | change=select the shared-input selection bias replay after the connected stop
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-selection-bias-cross | change=select an inert successor with both selection-score controls
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-bias-recheck | change=queue the shared-input native gather before the next connected successor
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-connected-up-seal | change=route the sealed gate-plus-up packet to review and held execution
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-connected-up-review | change=prioritize the offline SwiGLU control before dispatch
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-swiglu-cross | change=select the two-lane Q8_K and Q6_K expert down replay
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-down-cross | change=select one inert successor with both controls
