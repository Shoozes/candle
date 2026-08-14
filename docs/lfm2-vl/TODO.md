# LFM2.5-VL Active Backlog

Only incomplete work belongs here. Completed implementation and proof belong
in `HISTORY.md`; recurring hazards belong in `FAILURE_LOG.md`. All required
verification is local. Do not invoke, inspect, or depend on hosted CI.

## Active Candle backlog

### [ ] Publish the immutable combined-overlay 0.2.0 snapshot

- What: Create annotated tag `candle-overlays-mvp-0.2.0`, attach the source
  contract and external receipt, and apply owner-selected branch/tag
  protection to the already-published source checkpoint.
- Why: Local and remote `main` now agree, but a moving branch alone does not
  identify or protect an immutable release.
- When: Only after separate explicit authorization for the tag, hosted
  release, and repository-rule changes.
- Where: Root lock/toolchain/workflow, both overlay manifests and verifiers,
  `docs/releases/CANDLE_OVERLAYS_MVP_0.2.0.md`, local `main`, `origin/main`, and
  the hosted `candle-overlays-mvp-0.2.0` release.
- How:
  - Completed source, lock/toolchain, acceptance, verifier, and receipt work is
    recorded in `HISTORY.md`; do not duplicate that evidence here.
  - [ ] Create and publish the annotated tag, emit the external identity
    receipt, create and verify the release assets, and apply the separately
    owner-managed immutability rules without moving the old tag.
- Done when: Commit, tree, remote `main`, annotated tag, release receipt,
  compiler/Cargo versions, lock hash, overlay inventories, and release assets
  all agree; `lfm2-vl-mvp-0.1.0` remains unchanged.
- Verification: The exact commands in
  `docs/releases/CANDLE_OVERLAYS_MVP_0.2.0.md`, focused LFM2-VL/LoRA gates,
  summary/layout/overlay verifiers, `git diff --check`, clean status, guarded
  remote equality, and annotated-tag peel/asset comparison.

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
- Where: the current inventory is produced by `rg -l -F 'unimplemented!("compile
  with' --glob '*.rs' candle-transformers/src/models`; it spans Gemma, Granite,
  Llama/Mistral/Mixtral, Mimi, MMDiT, Phi3, StableLM, Voxtral, and Wuerstchen.
- How: take one model family per reviewable slice, return a typed Candle error
  from the disabled-feature helper, add a no-feature regression, and preserve
  the enabled kernel path unchanged.
- Done when: the inventory command returns no disabled-feature panic fallback
  and every affected family plus the locked/offline workspace gate passes.
- Verification: Run one no-feature regression per changed family, its focused
  crate tests, warnings-denied Clippy, the selected workspace suite, and the
  exact inventory command above.

### Conditional upstream maintenance — malformed stable-diffusion VAE inputs

- What: Replace the unchecked `block_out_channels[0]`/`last().unwrap()`
  assumptions in `AutoEncoderKL::new` and the two-result unwrap in
  `DiagonalGaussianDistribution::new` with controlled validation errors.
- Why: Both constructors are public. An empty channel layout or a latent
  parameter tensor that cannot split into mean/log-variance halves can panic
  instead of returning the crate's normal `Result` error.
- When: After combined-overlay 0.2.0 publication, and only as an independently
  reviewed upstream safety slice; `vae.rs` is outside both frozen overlays.
- Where: `candle-transformers/src/models/stable_diffusion/vae.rs`, its nearest
  unit tests, and whichever overlay or upstream manifest explicitly adopts the
  change.
- How: Validate the channel inventory before constructing encoder/decoder
  blocks, validate the latent parameter channel count before splitting, keep
  valid SD 1.x/XL shapes unchanged, and avoid a new abstraction or dependency.
- Done when: Empty channel layouts and unsplittable latent tensors return
  actionable errors without mutation or panic, valid VAE construction and
  sampling remain compatible, and the owning manifest records the new path.
- Verification: Focused malformed/valid VAE tests, transformer tests,
  warnings-denied Clippy, formatting, the selected locked/offline workspace
  gate, and the affected overlay/union verifiers.

### Conditional upstream maintenance — reachable model/operator stub branches

- What: Replace externally reachable `todo!()` branches with the pinned
  implementation when a supported contract exists, or reject the unsupported
  option/model shape with a typed error before dispatch.
- Why: Valid Rust call paths can currently panic when ViT positional
  interpolation is requested, SNAC relative positions are configured, SAM
  relative-position tables need resizing, DeBERTa `z_steps > 1`, or ONNX
  `Gather` receives higher-rank indices.
- When: After combined-overlay 0.2.0 publication; take exactly one subsystem
  per change and obtain reference fixtures before implementing numerical math.
- Where: `candle-transformers/src/models/vit.rs`, `snac.rs`,
  `segment_anything/image_encoder.rs`, `debertav2.rs`, and
  `candle-onnx/src/eval.rs`.
- How: Add a failing edge-case regression first, distinguish advertised
  support from an unsupported configuration, use checked shapes/indexes, and
  preserve existing supported paths. Do not create one generic model adapter.
- Done when: The selected branch cannot panic on caller/model input, its
  supported result matches a pinned reference or its rejection is actionable,
  and no unrelated stub is claimed complete.
- Verification: Focused crate tests and fixture, warnings-denied Clippy,
  formatting, selected locked/offline workspace tests, and a refreshed exact
  stub inventory for the selected source file.

### Conditional core maintenance — fallible safetensors serialization

- What: Remove the device-to-host `convert_back(...).unwrap()` calls used by
  the `safetensors::View` adapters and provide a fallible serialization path.
- Why: A device transfer/allocation failure during public tensor saving can
  currently panic inside an API that otherwise returns `Result`.
- When: After combined-overlay 0.2.0 publication as an isolated Candle-core API
  safety change; decide compatibility before changing public signatures.
- Where: `candle-core/src/safetensors.rs` and its serialization tests.
- How: Materialize validated owned byte views through a fallible preparation
  step, then pass only infallible views to `safetensors`; keep CPU zero-copy or
  copy behavior explicit and avoid a new dependency.
- Done when: Synthetic device-copy/allocation failure is propagated as a
  Candle error, CPU and supported device round trips remain byte-compatible,
  and neither `View::data` implementation unwraps fallible work.
- Verification: Focused safetensors round-trip/error tests, core tests, strict
  Clippy, formatting, and the selected locked/offline workspace gate.

### Conditional core maintenance — unsupported dtype and dummy-backend panics

- What: Replace reachable integer-unary and feature-disabled dummy-backend
  `todo!`/`unimplemented!` dispatches with typed unsupported-operation errors,
  or make impossible backend storage states unconstructible outside the crate.
- Why: Public tensor unary operations can dispatch macro-generated floating
  math over integer storage, and public dummy CUDA/Metal storage values expose
  trait methods whose infallible `dtype`/`device` contracts currently panic.
- When: After combined-overlay 0.2.0 publication as one Candle-core API design
  slice; decide compatibility and trait invariants before implementation.
- Where: `candle-core/src/op.rs`, `candle-core/src/cpu_backend/mod.rs`,
  `candle-core/src/dummy_cuda_backend.rs`, and
  `candle-core/src/dummy_metal_backend.rs`.
- How: Add caller-level regressions first, reject unsupported dtype/op pairs
  before invoking scalar callbacks, and prefer preventing dummy storage
  construction over fabricating dtype/device values. Preserve supported
  integer operations and feature-enabled backends.
- Done when: Unsupported integer unary calls and disabled accelerator paths
  cannot panic from public API use, errors identify the dtype/operation or
  missing feature, and enabled CPU/CUDA/Metal behavior is unchanged.
- Verification: Focused core error tests, CPU tensor tests, feature-disabled
  compile/tests, enabled-backend checks where locally available, strict Clippy,
  formatting, and the selected locked/offline workspace gate.

### Conditional examples maintenance — accepted unsupported CLI combinations

- What: Reject or implement example CLI branches that currently accept an
  option/model combination and later reach `todo!` or `unimplemented!`.
- Why: These are user-facing binaries; unsupported Flux quantized-dev,
  LLaVA conversation modes, MusicGen decoder masking, and non-tiny quantized
  Whisper choices should fail during argument/config validation rather than
  after downloads or model setup.
- When: After combined-overlay 0.2.0 publication; take one example per slice,
  and require a pinned model/reference before implementing numerical behavior.
- Where: `candle-examples/examples/flux/main.rs`, `llava/main.rs`,
  `musicgen/musicgen_model.rs`, `whisper/main.rs`, and
  `whisper-microphone/main.rs`.
- How: Add a no-network validation regression for each accepted unsupported
  combination, move rejection ahead of API/download work, and implement only
  branches with authoritative fixtures. Do not share abstractions merely
  because the panic spelling is similar.
- Done when: Each selected CLI either performs its advertised mode correctly
  or returns an actionable pre-I/O error, and its source has no reachable
  placeholder branch for accepted input.
- Verification: Focused no-network argument/config tests, the selected example
  check/test, warnings-denied Clippy, formatting, and the locked/offline
  workspace gate.

---
AI-edited: 2026-08-13T20:21:47-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=ultra | task=repo-integrity | change=completed direct-main source publication and retained only the separately authorized immutable-release step
