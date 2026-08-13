# LFM2.5-VL Active Backlog

Only incomplete work belongs here. Completed implementation and proof belong
in `HISTORY.md`; recurring hazards belong in `FAILURE_LOG.md`. All required
verification is local. Do not invoke, inspect, or depend on hosted CI.

## INT-5 — Prove differential ControlNet conditioning and residual parity

### What

Create a pinned tiny deterministic ControlNet fixture that proves the ordered
nine SDXL down residuals, the mid residual, text-conditioning influence, and
the final Candle UNet result against the pinned reference implementation.
Repin SnapFlash to published Candle Round 7
`95ac9ff815fbac4f252b4ef6780b5e4a7843f328` and consume the same public hook
without copying framework tensor checks.

### Why

Round 7 proves structural admission, not numerical equivalence. SnapFlash's
current ControlNet forward path still leaves text `_context` unused, so shape-
correct tensors alone cannot justify a conditioned-generation or inpainting
claim.

### When

Start only after REL-6/7 is clean and published. That condition is met. Finish
before promoting ControlNet-backed inpainting, enabling unattended real-weight
ControlNet use, or claiming application-level numerical parity.

### Where

- A new tiny generated fixture under Candle's SnapFlash-derived test boundary,
  with revision, package versions, dtype, device, seed, tensor inventory,
  shapes, and hashes recorded in its manifest.
- Candle Stable Diffusion residual tests and SnapFlash's ControlNet model,
  loader, dependency pin, deterministic tests, proof docs, and focused
  summary-bank route.

### How

1. Pin the official reference source and generate only tiny deterministic
   tensors; do not commit production weights or generated images.
2. Record all nine down residuals and the mid residual in exact application
   order, plus a final UNet output for zero control and nonzero control.
3. Prove text conditioning by changing only the context and requiring the
   reference and SnapFlash residual/output deltas to agree within declared
   per-dtype tolerances.
4. Test zero strength, start/end timing boundaries, malformed extra/missing
   residuals, wrong topology, and controlled loader failure before device
   allocation.
5. Keep image preprocessing, retained-file admission, catalogs, resource
   limits, queues, API schemas, and proof publication in SnapFlash; add no
   application model or request type to Candle.

### Current blockers

- The tiny reference fixture does not exist yet. Its source revision,
  generator environment, tensor inventory, and hashes must be fixed before
  implementation can make a numerical claim.
- SnapFlash still leaves ControlNet text `_context` unused. This is the exact
  behavioral gap the fixture must expose and close; shape-only tests are not a
  substitute.
- Production weights, CUDA, and live inpainting are intentionally outside this
  first deterministic slice. They are later evidence gates, not permission to
  broaden INT-5.

### Done when

- The fixture manifest pins its reference repository/revision, environment,
  generated inputs, tensor inventory, dtype/device, seed, and hashes.
- SnapFlash emits exactly nine ordered down residuals plus one mid residual;
  each shape/value and the final Candle UNet result matches the pinned
  reference within documented tolerances.
- Two different text contexts produce the expected distinct conditioned
  residuals/output; `_context` is no longer unused or falsely documented.
- Zero-control and timing boundary cases are exact, malformed topology fails
  before mutation/allocation, and the prior resident ControlNet restores after
  injected failure.
- Both repositories pass focused and complete local gates with no production
  model run, then publish reviewed direct-`main` checkpoints.

### Verification

- Pinned reference fixture generator and manifest verifier.
- Candle fixture-driven residual and final-forward tests.
- SnapFlash exact residual inventory/value/context/timing and transactional
  loader tests through the published Candle revision.
- Locked/offline format, check, strict Clippy, library/integrity, context,
  overlay, diff, staged-path, and guarded publication gates in both repos.

## Sequencing holds

- ControlNet-backed inpainting promotion waits for INT-5.
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
AI-edited: 2026-08-13T02:35:00-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=three-repo-round-7 | change=moved completed REL-6/7 to history and opened INT-5 as the sole active gate
