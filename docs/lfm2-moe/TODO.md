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
- Held: Production runs need an exact finite packet and fresh admission.
  Final Edge pin adoption, package assembly, installation, activation and release
  creation remain separately gated. Current Candle source-progress publication
  is selected by closeout/publish. Other consumer pins and frozen releases
  are not changed. Full native/Q8 production, Metal and 131k context qualification
  are outside this delivery.

## Remaining gates in order

1. Review the committed-source inert Candle Q4 parity packet under
   [QUALIFICATION](QUALIFICATION.md):
   - What/why: Check the current source/binary/input binding before selecting
     production execution. [STATUS](STATUS.md) identifies the fresh packet and
     [HISTORY](HISTORY.md) records the completed model-free refresh.
   - Where/how: Review `packet/packet.json` and `binding-verification.json` under
     `D:/Compact_Evidence/candle/runs/lfm2-moe-publish-20261110-5f094821/`.
     Preserve the old packet/binaries, frozen prompts/model, consumer pins and
     all original limits. Require a new exact owner grant and fresh quiet/resource
     admission before one finite attempt; preparation supplies no execution grant.
   - Done when: The finalized source inventory, both binaries, all frozen inputs,
     original limits and four comparison commands match their recorded identities;
     the packet remains inert and the next execution boundary is explicit.
2. After a new exact owner grant and fresh admission, run the reviewed finite
   packet once under [QUALIFICATION](QUALIFICATION.md):
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

---
AI-edited: 2026-11-10 | agent=Codex/root | task=lfm2-moe-publish | change=save supplier-source progress while retaining runtime and consumer acceptance criteria
