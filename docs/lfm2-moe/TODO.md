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

1. Compose the zero-based layer-1 CPU replay offline.
   - **Why** Packet `8060130a` consumed its attempt and stopped at weighted
     RMSNorm after an exact layer-0 entry check. All ten layer-1
     checkpoints now match in separate offline runs. The composed replay
     must prove that the candidate can carry its own intermediates through
     layer 1. [Native terminal](D:/Compact_Evidence/candle/runs/lfm2-moe-layer1-inert-20261113-72a53981/terminal.json).
     [Final offline checkpoint](D:/Compact_Evidence/candle/runs/lfm2-moe-layer1-output-offline-20261113-8060130a/summary.json).
   - **How** Use the verified layer-0 output for the same 37 frozen IDs,
     the frozen layer-1 weights and epsilon, and the existing isolated CPU
     schedules. Verify the entry hash before executing. Feed each computed
     tensor into the next operator and use pinned llama.cpp tensors as
     comparison controls only. Allow one offline attempt and stop at the
     first mismatch or failure. Keep production inference and adoption held.
   - **Done when** The entry and all ten checkpoints through `l_out-1`
     match bit for bit in one replay, including rows 16 and 36. Bind source,
     executable, input and output hashes to the terminal receipt. If the
     replay stops, retain the first mismatching row, channel, values and
     available operands.
2. Resolve the CPU mismatch under the existing [QUALIFICATION](QUALIFICATION.md)
   criteria. After the composed replay, use its exact `l_out-1` as the entry
   control for zero-based layer 2, the first attention and sparse-MoE
   block in this GGUF. Prepare a fresh inert diagnostic that starts at its
   weighted operator RMSNorm and advances only on a match. Any production
   probe needs separate authorization and fresh admission. Replay the first
   differing operands, capture a deterministic regression and integrate
   the justified correction into the runtime.
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
AI-edited: 2026-11-13 | agent=Codex/root | task=lfm2-moe-layer1-review | change=select the composed layer-1 offline replay
