# LFM2.5-VL Active Backlog

Only incomplete work belongs here. Completed implementation and proof belong
in `HISTORY.md`; recurring hazards belong in `FAILURE_LOG.md`. All required
verification is local. Do not invoke, inspect, or depend on hosted CI.

## REL-6/7 — Publish the bounded SnapFlash runtime and exact Candle residual contract

### What

Finish SnapFlash-Server Round 6 as one bounded, revision-retaining execution
owner, then publish Candle Round 7's exact additional-residual admission on the
same owner-reviewed direct-`main` workflow. Reconcile EdgeSymbio read-only at
its current proof-owner checkpoint while retaining the Round 5 acceptance
revision as historical lineage.

### Why

Green tensor or source tests do not make an application runtime safe by
themselves. Large request/response bytes, filesystem authority, model
revision identity, queue cancellation, engine mutation, and completion
publication must share one bounded lifecycle. Candle must then expose only the
generic residual contract that both applications can pin without absorbing
their HTTP, queue, path, resource, or proof policy.

### When

This is the current release gate. SnapFlash publishes first so Candle's status
can record its exact Round 6 consumer identity. Candle publishes second.
Do not start a production model, CUDA workload, Python oracle, llama.cpp
process, inpainting promotion, or later ControlNet numerical claim during this
gate.

### Where

- SnapFlash-Server's inference owner, immutable generation context, retained
  assets, cancellation/queue state, artifact-set publisher, encoded input and
  output wrappers, model catalog, path policy, API adapters, canonical docs,
  tests, and `summary_bank.json`.
- Candle's
  `candle-transformers/src/models/stable_diffusion/unet_2d.rs`,
  SnapFlash-derived overlay manifest/verifier, fork-overlay registry, status,
  history, decisions, changelog, and focused summary-bank route.
- EdgeSymbio only for clean `main == origin/main` reconciliation at
  `eb9c07127321bd7528786c4fa103b92f893991f5`; its Round 5 LoRA acceptance
  remains `633f774a3690df5a8a35b6cac000df4b390316d5`.
- SnapFlash-Server Round 6 implementation
  `d66c1c35158aca7b37e6e1d82e527334b209d93a` and final proof-record head
  `b83db70ba4027535e4e55f6509e6011feeead850`.

### How

1. Close SnapFlash's bounded runtime boundary.
   - Retain encoded request strings without a second full-size allocation;
     validate canonical base64 and aggregate limits before admission.
   - Decode and image-preflight inside the admitted blocking lane before
     retained revision binding or model, LoRA, and ControlNet mutation.
   - Encode generated output once before leaving that lane and retain a
     shallow immutable result for queue/API serialization.
   - Reject Windows drive-relative roots before filesystem access and retain
     rooted drive/UNC authority through publication.
   - Preserve one permit through kernel, postprocess, cancellation/commit
     arbitration, manifest-last publication, and result construction.
2. Prove and publish SnapFlash.
   - Run focused no-GPU tests, format, locked/offline check, strict Clippy,
     library tests, canonical integrity, docs/context/layout gates, and a
     complete diff/secret/path audit.
   - Update canonical app state, proof, review, regression guard, history,
     TODO, and summary-bank routes without duplicating progress logs.
   - Fetch and review `origin/main`, explicitly stage only reviewed paths,
     commit one coherent Round 6 checkpoint, publish with the ignored guarded
     helper, and prove clean local/remote equality.
3. Prove and publish Candle.
   - Retain the existing public residual method and `None` fast path; require
     the configuration-derived exact down inventory plus exact shape, dtype,
     and device for every down and mid tensor before addition.
   - Run focused 6/6 tests, the complete transformer crate, required four-
     crate check, strict all-target and workspace Clippy, the model-free
     workspace test gate, both overlay verifiers, union/context/layout/proof
     gates, format, and WSL Git diff review.
   - Record exact Round 6/7 revisions, explicitly stage only overlay-owned
     paths, publish through `.tools/gitpush.ps1`, and prove clean local/remote
     equality.

### Current blockers

- SnapFlash's three release-blocking memory/compatibility findings are closed
  and published. The implementation and later proof-record commits are clean
  locally/remotely and retain exact model-free verification evidence.
- Candle has no known implementation, API, or dependency blocker. Its focused,
  crate, required-check, workspace-test, targeted and full-workspace strict
  Clippy, overlay, context, layout, and preflight gates are green. Only final
  state reconciliation, lightweight release-gate replay, exact staging, and
  guarded publication remain.
- Full ControlNet numerical parity is intentionally not part of REL-6/7. The
  current SnapFlash `_context` gap and real-weight parity remain explicit
  blockers for INT-5 and later inpainting promotion.

### Done when

- SnapFlash direct, queued, and inpaint requests use the same bounded owner;
  malformed images cannot bind or mutate a model; exact retained base, LoRA,
  and ControlNet identities reach completion metadata.
- Cancellation cannot report failure after a completion manifest commits;
  success exposes an image, metadata, and manifest whose verified reader
  proves the exact retained set.
- Request and response payload clones are shallow; field/aggregate boundaries,
  Tokio heartbeat, queue admission, panic quarantine, model rollback, Windows
  rooted-path, and manifest-last failure seams pass deterministically.
- Candle rejects short, long, broadcastable, wrong-dtype, wrong-device, and
  malformed-mid residuals while preserving `None` and exact-zero behavior.
- SnapFlash and Candle complete their local gates, contain no staged secret,
  model, cache, runtime, or generated artifact, and have clean named `main`
  branches exactly equal to `origin/main` after guarded publication.
- EdgeSymbio remains clean at its published proof-owner identity while the
  Round 5 LoRA acceptance commit remains traceable. No live model, CUDA,
  llama.cpp, network, hosted runner, or PR is used for this release.

### Verification

- SnapFlash focused encoded-input, inference-owner, cancellation, queue,
  retained-asset, artifact-set, ControlNet/loader rollback, catalog, direct,
  and inpaint tests; locked/offline strict Clippy and no-GPU library suite;
  canonical local integrity and documentation/context/layout gates.
- Candle 6/6 residual tests; transformer 77/77 plus generation 5/5 and NMS
  8/8; strict targeted and full-workspace Clippy; required crate checks;
  model-free workspace tests excluding only the recorded live-HTTP dataset
  owner; overlay inventories 150/15/135 and 9/3/6; union 158/two/five;
  summary bank, module layout, preflight smoke, format, and diff checks.
- Exact staged-file review, guarded helper publication, and post-push local/
  remote revision equality in each publishing repository.

## INT-5 — Prove differential ControlNet conditioning and residual parity

### What

Create a pinned tiny deterministic ControlNet fixture that proves the ordered
nine SDXL down residuals, the mid residual, text-conditioning influence, and
the final Candle UNet result against the pinned reference implementation.
Repin SnapFlash to the published Round 7 Candle revision and consume the same
public hook without copying framework tensor checks.

### Why

Round 7 proves structural admission, not numerical equivalence. SnapFlash's
current ControlNet forward path still leaves text `_context` unused, so shape-
correct tensors alone cannot justify a conditioned-generation or inpainting
claim.

### When

Start only after REL-6/7 is clean and published. Finish before promoting
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
REL-6/7 or INT-5 requirements and must not be introduced without a scoped
proposal and acceptance contract.

---
AI-edited: 2026-08-13T02:10:00-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=three-repo-round-7 | change=recorded published SnapFlash Round 6 and narrowed REL-6/7 to Candle release
