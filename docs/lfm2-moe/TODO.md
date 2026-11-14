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

1. Once selected, replay layer-2 query and key RoPE offline.
   - **Why** The per-head weighted RMSNorm outputs match pinned llama.cpp.
     [Q/K replay](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-qk-norm-20261114-9b0d5544/summary.json).
     Packet `4ddbcf69` verified all 32 bindings. Admission stopped before
     the build because Cargo, rustc and Python were active. The terminal
     receipt records zero dispatches and an unconsumed attempt.
     [Admission receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-qk-rope-20261114-a2f11de3/summary.json).
   - **Bound inputs** Reuse the matched normalized tensors, the original
     rotary configuration, positions and limits, and the retained
     `q_rope` and `k_rope` captures.
   - **Execution** Seal a successor in a fresh evidence directory. Preserve
     the prior terminal receipt because the controller uses a fixed terminal
     path within each run. Recheck quiet/resource admission before dispatch.
     Run one offline attempt if admitted. Keep Edge repin and push held.
   - **Verification and done when** Match all 75,776 query and 18,944 key
     rotated values bit for bit with pinned llama.cpp.
     Stop at the first mismatch or failure.
2. Resolve the CPU mismatch under the existing [QUALIFICATION](QUALIFICATION.md)
   criteria. Use the RoPE result to select the next diagnostic.
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
