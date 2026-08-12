# LFM2.5-VL Active Backlog

Only incomplete work belongs here. Completed Candle implementation and proof
belong in `HISTORY.md`; recurring hazards belong in `FAILURE_LOG.md`. All
required verification is local. Do not invoke, inspect, or depend on hosted CI.

## INT-2 — EdgeSymbio proof-only 450M consumer

### What

Pin EdgeSymbio to the exact published Candle Round 1 revision and add a
separate, CLI-only `Lfm2VlModel` for one-image LFM2.5-VL proof. Preserve the
existing text-only `LfmModel` unchanged.

### Why

Candle now owns complete local hybrid construction, but only a product consumer
can prove retained-file admission, resource leasing, cancellation, evidence
publication, and mutation detection across a real application boundary.

### When

Start after guarded publication proves the clean Candle `main` revision equals
`origin/main`. Finish the CPU/CPU F32 lane before Q8, CUDA, public API, UI, RAG,
captioning, or SnapFlash integration.

### Where

- Edge dependency and lock manifests under `source/backend/`.
- New `source/backend/src/lfm2_vl.rs` rather than adding multimodal state to
  the text-only model.
- `source/backend/src/models.rs`, proof dispatch, resource admission,
  retained-asset handling, and Edge-owned runtime-proof modules.
- Edge current-state, verification, and summary-bank documents.

### How

1. Dependency identity
   - Replace direct crates.io `candle-core`, `candle-nn`, and
     `candle-transformers` entries with one immutable Git `rev`.
   - Add `candle-vlm` at that same `rev`; keep feature selection consumer-owned.
   - Refresh both lockfiles through the repository's explicit dependency-update
     workflow, then return to locked/offline operation.
   - Add a metadata gate that rejects any crates.io Candle package, more than
     one Candle source, or differing Candle Git revisions.
2. Asset admission
   - Define one Edge-owned `Lfm2VlAssetSet` containing the exact compatible
     450M text GGUF, direct F16 MMProj GGUF, tokenizer, processor config, and
     fixed PNG/JPEG proof image.
   - Record repository revisions, filenames, byte sizes, SHA-256 values, and
     license/source policy before any production proof.
   - Open each file through expected-size/hash admission and retain the verified
     handles or equivalent identity guard through report emission.
   - Fail explicitly for a missing, changed, mismatched, or unapproved member;
     never discover or download an alternative.
3. Runtime boundary
   - Create a dedicated model wrapper around Candle's returned model,
     processor, prompt, and consumed-file inventory plus Edge's retained
     handles, resource lease, bundle identity, and cancellation state.
   - Admit one image, one prompt, PNG/JPEG only, at most 32 generated tokens,
     CPU/CPU F32, and exact `<image>` handling.
   - Reserve the combined text, vision, projector, processor, and decode budget
     before construction; release it on every success and failure path.
   - Reject any attempt to route this proof target through the normal Tauri or
     public product surface.
4. Proof and evidence
   - Add `model-proof lfm2-vl --json` as an explicit-only target.
   - Record Candle revision, every asset hash, source image identity and
     dimensions, crop/projected-token counts, image-token spans, component
     devices/dtypes, generated token IDs, stop reason, and cache-clear replay.
   - Require at least one generated token before `RealOutput`.
   - Compare the fixed CPU fixture against Candle's standalone example and
     require exact generated token IDs.
5. Failure and regression coverage
   - Prove missing MMProj/tokenizer/processor/image failures, text/MMProj
     mismatch, post-admission mutation detection, prompt/image-count mismatch,
     cancellation, resource-release cleanup, and unchanged text-only LFM2
     behavior.

### Current blocker

EdgeSymbio has no admitted first-proof bundle yet. Its existing local 450M Q4
text GGUF is only a presence observation and is not a confirmed compatible
official pairing. The F16 MMProj, standalone tokenizer, processor config, and
fixed image are absent from its tracked manifest. Code and tiny-fixture work may
proceed after the Candle revision is published; official inference must wait
for explicit asset acquisition/admission authority and a quiet-host preflight.

### Done when

- Both Edge lockfiles resolve one exact Git Candle revision with no crates.io or
  duplicate Candle graph.
- The separate CPU/F32 proof target loads the complete admitted bundle through
  the public Candle API and matches Candle's generated token IDs for the fixed
  image/prompt.
- Mutation, missing-member, malformed-input, cancellation, and resource-release
  tests fail safely and leave no resident model process or lease.
- Existing text-only LFM2 behavior and normal no-download/no-GPU verification
  remain unchanged.
- Edge current-state docs identify the proof-only boundary and exact next gate.

### Verification

- Edge's narrow unit/integration tests for dependency identity, admission,
  resource cleanup, proof dispatch, and LFM2-VL failure paths.
- Locked/offline CPU checks and full repository integrity gate.
- `cargo metadata --locked --offline` and `cargo tree -d` audits proving one
  Candle source/revision.
- Bounded CPU/F32 parity through the Edge process owner, followed by PID,
  memory, lease, and consumed-file revalidation.
- Exact staged diff, secret/path audit, and local direct-main publication gate
  only after every preceding condition passes.

## Sequencing hold

The next cross-repository slice is Candle's generic three-component SDXL LoRA
transaction. It does not become active here until INT-2 is green. SnapFlash
fork pinning, SnapFlash LoRA migration, Edge LoRA migration, SnapFlash runtime
hardening, ControlNet hooks, inpainting promotion, and optional captioning
remain ordered follow-ons in `docs/FORK_OVERLAYS.md`; none may bypass INT-2.

## Deferred outside this product backlog

Gknome adoption, lower-bit vision quantization, generic VLM traits, video,
true text batching, converters, WebGPU, broad WSL replay, public signing, and
LTS remain separate repository or future-product work. They are not release
gates for INT-2 and must not be introduced without a scoped proposal and
acceptance contract.

---
AI-edited: 2026-08-12T12:42:54-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=three-repo-round-1 | change=opened the exact Edge proof-only consumer gate with prerequisites and completion conditions
