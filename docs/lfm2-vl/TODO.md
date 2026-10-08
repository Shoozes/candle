# LFM2.5-VL Active Backlog

Only incomplete work belongs here. Completed implementation and proof belong
in `HISTORY.md`; recurring hazards belong in `FAILURE_LOG.md`. All required
verification is local. Do not invoke, inspect, or depend on hosted CI.

## Active Candle backlog

The owner-authorized Candle d1 publication closeout and its identity contract
are [recorded in HISTORY](HISTORY.md#2026-10-08-d1-publication-closeout).
GenEye adoption requires the guarded helper's matching remote-main receipt.

### [ ] GenEye adoption of the published reusable d1 API

- What: Hand GenEye the exact published Candle identity, public d1 contract and
  source-bound receipts, then integrate the library in its owned worker.
- Why: Consumer package provenance and process/resource behavior require proof
  in GenEye after Candle's library qualification.
- When: After Candle's guarded remote-equality receipt for the reviewed d1
  source checkpoint and separate authorization for GenEye's consumer
  integration and any new model study.
- Where: GenEye owns its six-package Candle pin, standalone lock, packaged-worker
  provenance, process isolation, resource admission and study orchestration.
  Candle owns `candle_vlm::lfm2_d1` and its tensor readout primitive.
- How: Use explicit local retained artifacts, sequential fresh prefills,
  identity calibration and immutable image-feature reuse through the public
  session. Preserve partial failures, actual forward counts and zero generated
  tokens in consumer receipts. Update pin, lock and package identity together.
- Done when: GenEye's worker consumes the exact published library revision;
  pin/lock/package provenance agrees, and its owned acceptance proves typed
  question handling, cancellation/cleanup and resource boundaries. Product
  default changes and production study admission remain separately scoped.
- Verification: GenEye's native consumer tests, package/source identity checks
  and owner-admitted study receipts. The consumed 42-request allowance remains
  closed; reuse Candle's retained proof without relabeling its source identity.

### [ ] Qualify a real LFM2.5-2.6B text adapter and Edge switch

- What: Admit an immutable trained text adapter for the accepted
  `LFM2.5-2.6B-Q8_0.gguf` base, then prove serving behavior and quality.
- Why: Candle's local synthetic LoRA mechanics do not establish competence,
  training provenance, or safe Edge session switching.
- When: After the focused Candle native adapter gate is green and GenUni has
  produced a real retained adapter artifact; before enabling product routes.
- Where: GenUni owns training, data/seed/config, held-out evaluation, and the
  target manifest. Edge owns serialized adapter size/SHA-256 verification,
  parsing, model lease, atomic switch, receipt, rollback, and API selection.
  Candle owns only the quantized LFM2 hook and admission logic.
- How: Bind the manifest to base SHA-256
  `1e22128dfa128bdfb684da167e74e072d0a056baa7d06d9f280291e2839b0fc9`
  and the exact tokenizer/conversion lineage. Provide canonical target names,
  `[rank,input]` and `[output,rank]` F32 tensors, alpha, immutable adapter
  filename/size/SHA-256, and training/evaluation provenance. Use one retained
  file handle for hash/parse, load the base with
  `from_gguf_with_base_sha256`, switch with `replace_lora_adapter`, and record
  `active_lora_identity` in Edge receipts. Reject mismatch without fallback.
- Done when: Same-base base/adapter/base sessions, cache reset, bounded
  repeated-session memory, cancellation/recovery, held-out quality, and
  unsupported paths have passing receipts on the selected native backend.
- Verification: Exact-byte artifact hashes, model-generated target inventory,
  focused Candle admission tests, Edge job/API replay, and independent held-out
  comparison. No CUDA LoRA or 2.6B quality claim follows from synthetic tests.
- Current disposition (2026-10-07): Candle's F32 coefficient prerequisite now
  rejects conversion overflow/underflow before adapter or cache mutation; its
  five local LoRA tests, 46 neighboring LFM tests, and native maintained CPU
  closeout pass. The operator-supplied `D:\huggingface-cache` contains base snapshots,
  including the 2.6B GGUF revision `84022ce711b28455e8c4fc364ce68c00cf995875`,
  but snapshot metadata inspection found no retained adapter, training manifest,
  or held-out receipt. GenUni must supply those inputs before the unchanged
  real-adapter/Edge Done When gate can be executed.
- Producer prerequisite (2026-10-08): No real trained 2.6B adapter, selected
  training/held-out plan or quality thresholds were found in the inspected
  GenUni roots. Its verified Q8/native HF bases and setup-only jobs do not
  supply those artifacts. Complete the stock diagnostic prerequisite before
  selecting any adapter investment; exposed development results cannot qualify
  unseen competence. Completed preparation identities and original proof are
  [in HISTORY](HISTORY.md#2026-10-08---genuni-and-edge-diagnostic-preparation).
- Actual Work attempt (2026-10-08): One authenticated API owner session stopped
  at support refusal before request creation/submission. Seven views remained
  unattempted, no model call or retry occurred. All eight slots stay in the
  denominator; import/scorer replay retain them unbound with no semantic grades.
  Evidence: `artifacts/adapter-coordination/20261008-stock-diagnostic-w1/`.
- Exact next boundary: Edge must reconcile its regular-file admission with the
  supplied HF snapshot symlink and required GGUF filename, and make the
  cleanup-copy hook idempotent across repeated client retirement. Preserve
  hashes, original failed receipts and storage/resource bounds; do not silently
  stage 2.8 GiB outside the 64 MiB budget.
  Prove readiness and refusal/cancellation cleanup, then prepare a new reviewed
  packet and fresh admission. The consumed first approval/run root cannot be
  reused. The frozen 384/2048 views, sampler policy and no-retry rule stay binding.
- Remaining Done When: Collect actual bound responses and review explanations/
  citations after the Edge gates pass. The real trained adapter and held-out
  quality criteria above are unchanged and incomplete.
- Cleanup proof: `artifacts/adapter-coordination/20261008-cleanup-c1/` records
  the reproduced frozen-input gap, stricter importer and 13 native protocol
  tests over all eight retained-file views. These are authored controls, no
  model, training or adapter qualification. Keep the six-Python catalog intact.
  The subsequent `20261008-config-bound-c2/` proof closes the first configuration
  read before its allocation guard: one regression plus the thirteen neighbors
  pass. Use the refreshed importer contract for any future exact approval.

The 2026-09-27 pinned upstream sync is independent of the older consumer
pin-adoption task below. Consumer pins and historical selection identities
must be reviewed in their owning repository; they do not block Candle sync.
Integration proof lives in `UPSTREAM_SYNC.md`.

### [ ] Edge pin-adoption of published S3/S4 SHA (not LFM2-VL C0)

- What: Adopt published Shoozes/candle `4c1feb82eccd60a14f9f00f84ef6c9bdedcddd06`
  in Edge together with `tokenizers` 0.23/`onig`. Do not silently unify 0.22
  and 0.23 `Tokenizer` types.
- Why: Overlay and consumer gates passed. `TokenizerFromGguf` is implemented
  for candle-core's tokenizers crate version, so Edge 0.22 and this SHA's
  0.23 cannot share one `Tokenizer` type.
- When: After this published SHA. Before the LFM2-VL/Edge C0 chain.
- Where: Edge `source/backend/Cargo.toml` tokenizers line and the four
  `candle-*` git revs. Keep historical VL selection `dca98495…`.
- How: Bind new Edge evidence to the new pin; do not rewrite pack-selection
  identity. Do not open the LFM2-VL C0 chain from the pin bump.
- Done when: Edge pin and tokenizers move together, historical VL selection
  stays `dca98495…`, and the LFM2-VL C0 chain remains unopened until that
  pin-adoption lands.
- Verification: Edge `cargo check --locked` against `4c1feb82…` with
  `tokenizers` 0.23/`onig`.

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

Completed GPT-OSS Tasks 2 and 3 are
[archived in HISTORY](HISTORY.md#2026-10-07---gpt-oss-evidence-and-handoff-cleanup).
Their current source-bound acceptance limits remain in
[GPT-OSS status](../gpt-oss/STATUS.md#accepted-capability-rows).
Task 4 and maintained product qualification remain separately gated.

### [ ] GPT-OSS Task 4: quantized text plus split dense MMProj

- What: Add the next ordered GPT-OSS execution boundary for quantized GGUF
  text with split dense MMProj while preserving the accepted packed CUDA path.
- Why: The ordered implementation sequence keeps quantized text/MMProj
  integration after native packed execution and prevents Q8/default changes
  from being inferred from the bounded synthetic CUDA proof.
- When: Only after the published Task 3 receipt and a separately scoped owner
  acceptance; the exact assembly proof does not substitute for quantized-text
  or split-MMProj numerical evidence.
- Where: GPT-OSS GGUF/runtime modules, CUDA-gated or quantized tests, and
  `docs/gpt-oss/` proof records. Do not touch Edge/Harmony integration.
- How: Retain packed ownership, admit split tensor identities before
  allocation, add an independent fixture, and keep numerical product claims
  closed until tokenizer and inference evidence is manifested.
- Done when: Quantized text and split dense MMProj component behavior,
  resource ownership, and cleanup are independently verified without changing
  maintained Q8/LFM2 defaults.
- Verification: Targeted quantized tests, locked native checks, formatting,
  diff, summary-bank/mod-manifest/overlay verifiers, and the guarded local
  publication gate.

### [ ] Close the native 3B and official 400M Q8 MMProj production proof gap

- Sequencing: outside the S3/S4 → C0 worker chain. Do not start this instead
  of S3.

- What: Acquire and verify the immutable `LiquidAI/LFM2.5-VL-3B` native
  snapshot and the official `LiquidAI/LFM2.5-VL-3B-GGUF` text/F16/Q8_0 MMProj
  artifacts, then produce the bounded native and hybrid evidence receipts.
- Why: Candle now has the config-driven architecture, direct-GGUF Q8
  execution, hash-bound artifact admission, and fixture regressions, but
  architecture support is not production compatibility evidence.
- When: The next owner-authorized production proof lane, after the locked
  external artifacts and compatible pinned oracle environment are available;
  CPU-F32 is first and CUDA is out of scope.
- Where: `tools/lfm2_vl/reference-lock.json`, the reference manifest/inspect/
  acquire/trace/comparator tools, `candle-examples/examples/lfm2-vl/`, and the
  external snapshot, artifact manifests, traces, and receipts. Never place
  weights, caches, model code, or generated evidence in this repository.
- How:
  - [x] Lock the native 3B files, official GGUF text/F16/Q8_0 files, sizes,
    hashes, memory bounds, tokenizer markers, processor limits, and config
    dimensions.
  - [x] Admit model-provided Python only for an exact locked snapshot after
    every code file is listed and rehashed; reject moving-branch, cache-only,
    and unlisted code. The current pinned snapshot has no such files.
  - [x] Validate actual 3B config values without loading weights and preserve
    the existing native trace schema.
  - [x] Publish a separate direct-GGUF `hybrid-trace` bundle with projected
    image embeddings, prefill/decode logits, input identities, execution mode,
    Q8 tensor count, generated IDs, and cache-reset evidence.
  - [ ] Acquire the external artifacts through the guarded owner path, create
    hash-only manifests, and revalidate every consumed file after inference.
  - [ ] Run the 3B native CPU-F32 oracle/Candle comparison through processor,
    prompt expansion, all 27 vision layers, post-layernorm, projector, merge,
    prefill, cached decode, generated IDs, cache reset, and cleanup.
  - [ ] Run identical-input dense/dequantized versus native-Q8 direct-GGUF
    comparisons and prove the resolved path retains Q8 tensors rather than
    silently dequantizing.
- Done when: Both external artifact manifests match the lock; all required
  component tensors, logits, generated IDs, input hashes, cache-reset replay,
  and cleanup checks pass at their documented tolerances; the hybrid receipt
  reports `dense-dequantized` versus `q8_0-native` with a positive retained Q8
  tensor count; no production claim remains without its receipt; and the
  native/Q8 rows in `STATUS.md` and `PARITY.md` are updated from Gated only
  after that evidence exists.
- Verification: `python -m pytest -q tools/lfm2_vl/reference`, focused
  `candle-transformers`, `candle-vlm`, and `lfm2-vl` example tests, locked
  native Windows/MSVC checks, WSL `scripts/lfm2-vl/verify-baseline.sh`,
  summary-bank/mod-manifest/overlay verifiers, `cargo fmt --all -- --check`,
  `git diff --check`, and the bounded external oracle/Candle receipt audit.

## Sequencing holds

Deferred GPT-OSS-focused optional acceleration research is maintained in
[GPT-OSS status](../gpt-oss/STATUS.md#deferred-optional-acceleration-research).
Profile existing safe cases before selecting one optimization; exact support
detection, explicit opt-in and version-bound parity/memory/performance proof
must precede admission. GenAce MVP/LTS and GenUni's current foundation retain
priority; this note does not reorder Task 4 or current product gates.

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
AI-edited: 2026-10-08 | agent=Codex/root | model=unknown | effort=unknown | task=d1-closeout-publish | change=archived the selected publication contract and kept GenEye adoption bounded
