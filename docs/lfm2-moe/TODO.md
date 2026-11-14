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

1. Prepare an inert successor for the zero-based layer-2 CPU diagnostic.
   - **Why** Packet `623dc38e` wrote its attempt marker after the initial
     census. The final reference census found two Python processes and
     blocked both probes before launch. [Terminal receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-controller-review-20261113-9cc7c853/terminal.json).
   - **Bound inputs** Reuse the computed `l_out-1` entry, SHA-256
     `0df2dfda89526966365f5787f063ccb07e6f4ad250f0aab004fb23be63946462`,
     the 37 frozen IDs, pinned llama.cpp, both probe binaries and 13 layer-2
     tensor ranges from [packet `623dc38e`](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-controller-review-20261113-9cc7c853/packet.json).
   - **How** Keep verified files locked through dispatch and preserve the
     consumed packet. Finish model identity verification and the final
     reference admission before writing the new attempt marker immediately
     ahead of the bounded launch. Preserve per-probe admission, cleanup
     evidence and the [routing comparison decision](DECISIONS.md#layer-2-routing-comparison).
   - **Limits** Carry forward the native layer-1 CPU diagnostic envelope.
     One attempt, zero retries and at most four children remain bound. The
     two prebuilt probes are the execution children. The packet preserves
     180 seconds per child, 900 seconds total, 4 GiB build and 32 GiB model
     Job caps, 16 GiB RAM and commit floors, two CPU threads, 12,288 context
     tokens and 128-token chunks.
   - **Done when** A fresh packet binds the same 27 typed checkpoints and
     two prebuilt probes. A synthetic census passes initially and refuses
     the final reference admission, leaving zero launches and no attempt
     marker. A synthetic admitted launch writes one marker and cannot be
     repeated. End with the real packet inert for review.
2. Resolve the CPU mismatch under the existing [QUALIFICATION](QUALIFICATION.md)
   criteria. After exact authorization and fresh admission, execute the
   reviewed layer-2 successor once. Require exact entry and stop on the
   first mismatch or failure. Retain available operands, Job exits, cleanup
   and post-run censuses. Use the terminal result to select an operand replay.
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
