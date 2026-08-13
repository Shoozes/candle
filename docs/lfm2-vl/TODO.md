# LFM2.5-VL Active Backlog

Only incomplete work belongs here. Completed implementation and proof belong
in `HISTORY.md`; recurring hazards belong in `FAILURE_LOG.md`. All required
verification is local. Do not invoke, inspect, or depend on hosted CI.

## INT-5 — Prove differential ControlNet conditioning and residual parity

### What

Create a pinned tiny deterministic ControlNet fixture that proves the ordered
nine SDXL down residuals, the mid residual, text-conditioning influence, and
the final Candle UNet result against the pinned reference implementation.
Repin SnapFlash to the published Candle INT-5B.1 lower-precision checkpoint
and consume its public residual and `text_time` hooks without copying
framework tensor checks.

### Why

INT-5A/B prove fail-closed topology admission and the reusable base-UNet
addition input, not numerical equivalence. SnapFlash's current ControlNet
forward path still leaves text `_context` unused, so shape-correct tensors
alone cannot justify a conditioned-generation or inpainting claim.

### When

INT-5A and INT-5B are published. The INT-5B.1 cast-order implementation is
green locally and must be published before SnapFlash repins. Start INT-5C only
after that exact revision is available. Finish INT-5C/D before promoting
ControlNet-backed inpainting, enabling unattended real-weight ControlNet use,
or claiming application-level numerical parity.

### Where

- A new tiny generated fixture under Candle's SnapFlash-derived test boundary,
  with revision, package versions, dtype, device, seed, tensor inventory,
  shapes, and hashes recorded in its manifest.
- Candle Stable Diffusion residual tests and SnapFlash's ControlNet model,
  loader, dependency pin, deterministic tests, proof docs, and focused
  summary-bank route.

### How

Complete the remaining tasks in order. INT-5A/B are archived in `HISTORY.md`.

1. **INT-5C — faithful SnapFlash ControlNet graph.**
   - What: consume the official down-block types, transformer depths,
     cross-attention context, and `text_time` conditioning, plus correct SDXL
     pooled/penultimate CLIP outputs and time IDs.
   - Why: `_context` is presently unused and the installed attention/addition
     weights are ignored.
   - When: only after SnapFlash pins the Candle INT-5B.1 checkpoint.
   - Where: SnapFlash ControlNet, prompt conditioning, loader, sampling, and
     deterministic generated-weight tests.
   - How: reuse Candle's public Stable Diffusion blocks, keep the exact
     nine-plus-mid order, validate configuration and tensor coverage, and
     preserve retained revision/rollback ownership.
   - Done when: two contexts produce distinct residuals, every official
     required tensor is graph-owned, no checkpoint tensor family is silently
     ignored, and an injected replacement failure restores the prior resident
     revision.
2. **INT-5D — pinned differential fixture and publication.**
   - What: generate only tiny deterministic tensors from the pinned official
     Diffusers source and compare all nine residuals, the mid residual, and the
     final Candle UNet output.
   - Why: source completeness and shape checks are not numerical parity.
   - When: after INT-5B/C are green; before any production-weight or CUDA run.
   - Where: a small generated fixture and manifest in the SnapFlash-derived
     Candle overlay, the reference exporter/verifier, and SnapFlash consumers.
   - How: record revision, packages, dtype, device, seed, shapes, and hashes;
     test zero strength, nonzero strength, two contexts, and exact start/end
     timing boundaries within declared F32 tolerances.
   - Done when: both repositories pass their focused and complete local gates,
     SnapFlash pins the published Candle result, and reviewed `main` revisions
     are published without production weights, images, CUDA, or live models.

### Current blockers

- The official source is now identified: Diffusers tag `v0.39.0` peels to
  `a3608b512ed7248499a44c61d954965ed9bdae4d`; the two INT-5B behavior blobs
  are pinned, but the later tiny exporter, resolved package lock, tensor
  bundle, and manifest do not exist yet.
- The installed official Canny/Depth layouts require cross-attention and SDXL
  `text_time` addition embeddings. Candle now owns the generic addition
  primitive plus the pinned lower-precision cast order, but SnapFlash still
  lacks the faithful cross-attention graph, CLIP2 pooled projection, and
  time-ID policy. INT-5A continues to fail closed until INT-5C replaces that
  incomplete path.
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
AI-edited: 2026-08-13T04:34:15-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=int-5b-cast-order | change=made publication of the lower-precision checkpoint the exact INT-5C prerequisite
