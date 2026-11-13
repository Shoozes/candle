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

1. Locate the first CPU difference in layer 1.
   - **Why** Both engines chose token `207` after the 37-token prefill,
     with a maximum sampled logit delta of `1.313808` despite exact layer-0
     output. The GGUF uses ShortConv and dense FFN at zero-based layer 1.
     [Full CPU prefill receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-full-cpu-prefill-20261113-bd3094dd-recheck/summary.json).
   - **Ready for review** Packet `cbdfeb0f` binds 74 inputs, the same
     37 frozen IDs and pinned llama.cpp `582eab71`. Rust metadata and
     pinned MSVC syntax checks passed. A missing or wrong packet hash
     rejected execution before dispatch. [Inert packet](D:/Compact_Evidence/candle/runs/lfm2-moe-layer1-inert-20261113-bd3094dd/packet.json).
     The controller requires HEAD `bd3094dd`. On a successor commit,
     verify the diagnostic build inputs are unchanged and bind a new
     packet to that HEAD before requesting authorization.
   - **How** After a fresh exact grant, require the layer-0 `l_out-0`
     bytes to match before comparing layer 1. Start with weighted RMSNorm
     (`model.layers.{}.operator_norm-1`), then compare the ShortConv and
     FFN stages only while the preceding checkpoints match. The reference
     graph labels its normalized FFN input `model.layers.{}.ffn_out-1`.
     Admit each child under the revised component limits. The one
     attempt allows two diagnostic builds and two CPU probes. Keep the
     production CPU/CUDA packet and Edge adoption held.
   - **Done when** The layer-0 entry control and all measured layer-1
     stages through `l_out-1` agree bit for bit, or the trace stops at the
     first mismatch and records its row, channel, values and operand
     identities. The one-prompt prefill result leaves forced decode open.
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
AI-edited: 2026-11-13 | agent=Codex/root | task=lfm2-moe-closeout | change=record the layer-1 packet source binding at publication
