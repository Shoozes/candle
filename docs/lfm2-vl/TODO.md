# LFM2.5-VL Active Backlog

Only incomplete work belongs here. Completed implementation and proof belong
in `HISTORY.md`; recurring hazards belong in `FAILURE_LOG.md`. All required
verification is local. Do not invoke, inspect, or depend on hosted CI.

## Active Candle backlog

No finite Candle-owned implementation item is active. The bounded hybrid
metadata preflight and its complete local verification are archived in
`HISTORY.md`; add a task only when its trigger, owner, files, completion
condition, and verification command are concrete.

## Sequencing holds

- Optional LFM2-VL captioning in SnapFlash waits for the diffusion runtime and
  numerical ControlNet boundary; it must use Candle's public hybrid loader and
  an application-owned retained/resource/proof contract.
- CUDA optimization waits for CPU/deterministic parity and a fresh quiet-host
  memory preflight.

## Deferred outside this product backlog

Gknome adoption, lower-bit vision quantization, generic VLM traits, video,
true text batching, converters, WebGPU, broad WSL replay, public signing, and
LTS remain separate repository or future-product work. They are not hidden
INT-5 requirements and must not be introduced without a scoped proposal and
acceptance contract.

### Conditional upstream maintenance — disabled-feature panic fallbacks

- What: sixteen unrelated upstream model modules still use `unimplemented!()`
  when flash attention is requested from a build without that feature.
- Why: those paths can convert an unsupported caller policy into a panic, but
  they do not belong to the LFM2-VL or SnapFlash overlay and project policy
  prohibits opportunistic edits to unrelated architectures.
- When: only if the owner selects a separate upstream-wide safety campaign.
- Where: the current inventory is produced by `rg -l -e 'unimplemented!\("compile
  with' --glob '*.rs' candle-transformers/src/models`; it spans Gemma, Granite,
  Llama/Mistral/Mixtral, Mimi, MMDiT, Phi3, StableLM, Voxtral, and Wuerstchen.
- How: take one model family per reviewable slice, return a typed Candle error
  from the disabled-feature helper, add a no-feature regression, and preserve
  the enabled kernel path unchanged.
- Done when: the inventory command returns no disabled-feature panic fallback
  and every affected family plus the locked/offline workspace gate passes.

---
AI-edited: 2026-08-13T13:36:16-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=ultra | task=repo-integrity-hardening | change=archived current fixes and bounded the unrelated upstream panic inventory
