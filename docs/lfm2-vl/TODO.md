# LFM2.5-VL Active Backlog

Only incomplete work belongs here. Completed implementation and proof belong
in `HISTORY.md`; recurring hazards belong in `FAILURE_LOG.md`. All required
verification is local. Do not invoke, inspect, or depend on hosted CI.

## Active Candle backlog

No finite Candle-owned implementation item is active. REL-8 is published and
archived in `HISTORY.md`; add a new task here only when its trigger, repository
owner, files, completion condition, and verification command are concrete.

## Sequencing holds

- Queued inpainting is application-owned and may proceed in SnapFlash against
  the published Candle pin; it does not require another Candle primitive.
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

---
AI-edited: 2026-08-13T12:43:56-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=ultra | task=rel-8-downstream-rollback | change=moved the published cross-repository proof into history and cleared the finite Candle queue
