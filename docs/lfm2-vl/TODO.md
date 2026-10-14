# Candle Fork Active Backlog

Only incomplete work belongs here. Completed narrative and superseded source
states are in [HISTORY](HISTORY.md#2026-10-14---independent-mvp-scope-freeze).
Local-only proof and original numerical/resource contracts remain mandatory.

## Release blockers

### [ ] Finish the five-overlay source release 0.2.0

- What: Repair native identity/receipt tooling, freeze the supported source
  scope, and produce a reproducible source archive and evidence index.
- Why: The unshipped August writer uses a retired WSL checkout and two-overlay
  inventory; a green ordinary source push does not prove a usable release.
- When: Current bounded work, before any new model, optimization or consumer
  activation. Each repository can finish independently.
- Where: `docs/releases/CANDLE_OVERLAYS_MVP_0.2.0.md`, release writer/test/frozen
  expectations, native gate, five manifests and existing status/routes.
- How / finite remaining blockers:
  - [x] Native single-checkout identity, fixed five-overlay expectations,
    package creation and failure-contract tests pass on PS7 and PS5.1.
  - [ ] Maintained native source and diffusion gates pass; reviewed clean
    checkpoint, externally frozen commit/tree, candidate source archive and
    compact evidence index agree. No production study is repeated.
  - [ ] Under separate explicit authority, guarded main publication,
    annotated `candle-overlays-mvp-0.2.0`, hosted source/receipt assets and remote
    identities/hashes agree; historical `lfm2-vl-mvp-0.1.0` remains unchanged.
- Done when: The source release has immutable identity, usable archive, honest
  support matrix, replay/recovery instructions and no scoped release blocker.
- Verification: The release contract's native commands, no-overwrite/identity/
  inventory/tag tests, maintained gate, source-package inspection and final
  remote/tag/asset comparison. Publication is not authorized by this Work.
- Stop: Maintenance after release. No new model, kernel, context length,
  benchmark, repin or LTS commitment enters this batch.

## Deferred outside the MVP release

These tasks retain their acceptance criteria and closed allowances. Their
historical sequencing does not gate the source MVP; later execution needs
separate selection. Edge owns Windows core-runtime 0.2.1 at `a0c70e1d`, and
Symbio owns manager 0.1.1 on Codex 0.154.0. Neither depends on GPT activation or
the newest Candle graph. Broader GenAce/VL, new engines/providers, fast precision,
voice/image promotion, broad portability, signing and LTS remain separate.

### [ ] Maintained GPT-OSS support through EdgeSymbio and symbio-code

- What: Finish explicit GGUF/packed local loading, measured CPU/CUDA
  optimization and qualified selectable GPT-OSS-20B consumer integration.
- Why: Existing short experimental receipts do not establish a maintained
  loader, 32k capacity, performance improvement or coding-provider readiness.
- When: Deferred outside MVP; baseline and repair allowances are closed.
  Each production comparison needs fresh source/binary binding and allowance.
- Where: GPT-OSS overlay; [current STATUS](../gpt-oss/STATUS.md) and
  [LOADING contract](../gpt-oss/LOADING.md). Edge/Symbio own their integrations.
- How: Local loading/config/session fixture gate first, then profile before
  optimization, CPU before CUDA, staged 8k/16k/32k qualification, strict F32
  default and independently proven BF16. Preserve Q8 defaults and GenEye pin.
- Done when: Each advertised artifact/device/mode/route has observed parity,
  bounded memory, context/output and recovery proof, measured improvement,
  and actual Responses/tool/coding integration after authorized publication.
- Verification: Native focused/all gates and source-bound retained proof.
  Target: 32 GiB host, 20e9 GPU bytes, 30,720 history/prompt + 2,048 generated
  tokens including reasoning. Responses remains90s; WorkerRuntime remains600s;
  one route's capacity never qualifies the other. No current BF16/32k claim.

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
- Retained serving predecessor: Edge must admit the same base as a regular file
  with the required GGUF filename and make repeated retirement/cleanup-copy
  idempotent. Then collect bound responses for all eight exposed public views
  and review explanations/citations; these are not unseen-competence proof.
  Preserve the frozen 384/2048 views, sampler and no-retry policy, all slots in
  the denominator and the consumed first allowance. GenUni's distinct 3B pilot
  is not the exact-base 2.6B handoff. The complete retained prerequisite and
  partial-result contract is the
  [prior TODO](https://github.com/Shoozes/candle/blob/63f0d2cd275e4653b268939b429cbb9af25aba1a/docs/lfm2-vl/TODO.md#qualify-a-real-lfm25-26b-text-adapter-and-edge-switch).

### [ ] Close the native 3B and official 400M Q8 MMProj production proof gap

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


### GenEye-owned Q8Rows portable handoff

Current boundary (2026-10-11): GenEye's Q8Rows adoption is published at
`8e97e33`, with package progress records at `0d552b0`; Candle remains pinned to
`3fa7075e`. Clean CPU/CUDA candidates pass model-free package and process-owner
checks. The new package's numerical proof stopped at CPU owner admission before
worker startup: zero model loads, requests or forwards, and all four slots
unstarted. That allowance is consumed and closed. See
[the source-bound package records](D1_LOADING.md#geneye-consumer-qualification).

- What: Finish fresh bounded CPU/CUDA numerical qualification of the clean
  portable candidate at the unchanged Candle pin. GenEye owns this remaining
  consumer gate.
- Why: The original four-request private-worker proof and model-free package
  checks do not qualify the new package's numerical execution.
- When: After a quiet admission window and explicit approval of a newly reviewed
  finite proof packet. All earlier allowances remain closed.
- Where: GenEye's active `docs/TODO.md`, its clean `q8rows-portable` checkout and
  retained `D:\GenEye-evidence\candle-q8rows-portable-20261011` records.
- How: Revalidate source, lock, packages, binaries, DLLs, models and frozen
  inputs; prepare the fresh finite allowance before execution. Use the existing
  bounded Job and worker/request limits, CPU first, CUDA only after CPU passes,
  stop on any failure and verify process release. Preserve shared workloads,
  existing receipts and the unchanged pin. LTS activation stays separate.
- Done when: Exact published source, Candle lock/worker/package identities and
  CPU/CUDA text/image numerical, native-dispatch, Q8Rows storage, counters,
  ordering and cleanup receipts agree at the admitted bounds. Existing consumer
  proof keeps its original source and image-case quality limitation.
- Verification: The freshly admitted package proof and identity/process-release
  recheck. Original numerical and model-free records remain explicitly separate.


### Conditional maintained-model qualification — Qwen recovery/projections

- What: Qualify the repaired Qwen recovery and optional Grouped projections
  against the retained exact 4B artifact and its existing pinned reference.
- Why: Authored bit-exact components and paired transfer timings do not prove
  retained-model behavior or full-model latency at the new source identity.
- When: After separately selecting a clean source/binary-bound packet and
  approving its exact finite model-work allowance. The completed d1 t01
  allowance is closed and cannot be reused.
- Where: The Qwen CPU/hybrid models, existing bounded parity/performance
  helpers and independently owned `docs/qwen35/` proof records.
- How: Freeze inputs, modes, work counts and resource limits before approval;
  preserve Sequential default, run CPU first, stop on failure and prove release.
- Done when: The selected existing Qwen reference/recovery criteria pass;
  matched Sequential/Grouped results preserve arithmetic and observer order;
  any latency claim has matched measurements, without a new speed threshold.
- Verification: Exact source/binary/artifact hashes, native bounded receipts
  and the accepted Qwen bounds/contracts in `docs/qwen35/STATUS.md` and
  `PERFORMANCE.md`. No Edge/provider activation or consumer repinning.

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
AI-edited: 2026-10-14 | agent=Codex/root | model=unknown | effort=unknown | task=overlay-mvp-release | change=separated finite release blockers from unchanged deferred acceptance
