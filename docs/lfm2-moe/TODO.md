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

1. Once selected, replay the layer-2 raw expert-weight sums offline.
   - **Why** Packet `62ab7c06` matched all 148 unbiased gathered weights.
     The raw sum is the next checkpoint before either engine applies its
     normalization denominator.
     [Gather comparison](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-router-gather-20261114-91a27e18/comparison.json).
   - **Bound inputs** Verify the matched 37×4 F32 weights and the pinned
     llama.cpp `ffn_moe_weights_sum-2` capture of 37 raw F32 sums. Preserve
     each row's four inputs in their verified slot order.
     [Weights](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-router-gather-20261114-91a27e18/selected_weights.f32),
     [reference sums](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-successor-20261113-4df21393/reference/weight_sum.f32),
     and [capture hashes](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-successor-20261113-4df21393/reference-captures.json).
   - **Execution when selected** Compare the native CPU row sums in a
     fresh finite packet. Check quiet before sealing and require fresh
     admission under the [original limits](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-router-gather-20261114-91a27e18/packet.json)
     for one offline attempt. Keep Edge repin and push held.
   - **Verification and done when** Reproduce the 37 retained reference
     sums and compare all 37 Candle sums bit for bit, including row 36
     `0x3f9d6164`. Stop at the first mismatch or failure. For a mismatch,
     retain the row, its four operand bit patterns and both sum bit
     patterns before testing either denominator.
2. Resolve the CPU mismatch under the existing [QUALIFICATION](QUALIFICATION.md)
   criteria. Use the raw expert-weight sum result to choose the next
   diagnostic.
   Capture a deterministic regression and integrate only a justified correction.
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
