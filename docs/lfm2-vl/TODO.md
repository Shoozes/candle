# LFM2.5-VL Active Backlog

Only incomplete work belongs here. Completed implementation and proof belong
in `HISTORY.md`; recurring hazards belong in `FAILURE_LOG.md`. All required
verification is local. Do not invoke, inspect, or depend on hosted CI.

## REL-8 — Prove downstream later-component LoRA write rollback

### What

Expose Candle's existing deterministic SDXL transaction fault only through a
non-default consumer-test feature, then prove from SnapFlash that failures at
the first write of text encoder 1 or text encoder 2 restore every prior live
tensor and preserve the active adapter.

### Why

Candle already proves its private rollback path, but downstream consumers
cannot invoke that failure deterministically. Copying transaction logic into
SnapFlash would create the drift this shared primitive was designed to remove.

### When

After INT-5C/D publication and before accepting a future Candle transaction
upgrade as independently downstream fault-injection-proven. This does not
require or authorize a model, CUDA, or live inference run.

### Where

- Candle `candle-transformers/Cargo.toml`,
  `stable_diffusion/mutable.rs`, and its external integration test.
- SnapFlash `source/src-tauri/Cargo.toml`, `Cargo.lock`,
  `engine/loader/head_swap.rs`, focused tests, current docs, and context route.

### How

- [x] Add Candle's non-default `test-utils` feature and one public method that
  injects failure at the selected component's first planned write while using
  the production snapshot/write/rollback path.
- [x] Add an external Candle integration test for text encoder 1, text encoder
  2, and the no-planned-write fail-before-mutation case.
- [ ] Publish the exact Candle revision after its full local and overlay gates.
- [ ] Repin SnapFlash and enable `test-utils` only for dev/test builds.
- [ ] Add SnapFlash component-2/component-3 tests that preserve every tensor,
  transaction revision/target inventory, and application `active_adapter`.
- [ ] Pass both repositories' focused and complete local gates, publish
  reviewed `main`, and move this item to `HISTORY.md`.

### Current blockers

- None in source. Publication and downstream pinning are ordered external Git
  steps; production weights and CUDA are outside this test-only slice.

### Done when

- Candle exposes no failure hook in default builds; opted-in tests can name a
  component and deterministically reach the production rollback path.
- Candle and SnapFlash both prove text-encoder-1 and text-encoder-2 failure
  restores every prior tensor, transaction state, and active adapter.
- Both exact pins, focused/full local gates, context/overlay checks, reviewed
  direct-main commits, and remote equality are green without duplicate
  transaction code, production weights, CUDA, or live inference.

### Verification

- Candle feature-enabled external test plus the existing private transaction
  suite.
- SnapFlash focused `head_swap` consumer tests through the published revision.
- Locked/offline format, check, warnings-denied Clippy, library/integrity,
  context, overlay, diff, staged-path, and guarded publication gates.

## Sequencing holds

- Queued inpainting is application-owned and may proceed only after REL-8 is
  closed or is deliberately sequenced around the exact dependency pin.
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
AI-edited: 2026-08-13T12:09:28-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=ultra | task=rel-8-downstream-rollback | change=archived completed INT-5C/D and made the public consumer rollback seam executable
