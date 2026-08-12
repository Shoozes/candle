# LFM2.5-VL Active Backlog

Only incomplete work belongs here. Completed Candle implementation and proof
belong in `HISTORY.md`; recurring hazards belong in `FAILURE_LOG.md`. All
required verification is local. Do not invoke, inspect, or depend on hosted CI.

## INT-4 — SnapFlash-Server reconsumes Candle's LoRA transaction

### What

Pin SnapFlash-Server to the exact Candle Round 3 publication revision and
replace its local LoRA pair parsing, tensor merge, immutable-base replacement,
and three-component rollback internals with Candle's public Stable Diffusion
LoRA modules.

### Why

SnapFlash-Server is the complete behavior donor and therefore the strongest
first regression witness. Reconsumption proves that the framework boundary is
usable without copying application paths, licenses, catalogs, JSON reports, or
model-family key policy into Candle.

### When

Start only after guarded Candle publication proves clean local and remote
`main` equality for the LoRA promotion commit. Complete SnapFlash-Server before
EdgeSymbio's UNet-only implementation migrates or any ControlNet/inpainting
promotion begins.

### Where

- SnapFlash-Server Candle dependency entries and `Cargo.lock` under
  `source/src-tauri/`.
- `source/src-tauri/src/engine/loader/lora.rs` for its application-owned
  inspection, filename, mapping, and report adapter.
- `source/src-tauri/src/engine/loader/head_swap.rs` for replacement by
  `VarMapSwapTransaction`.
- `source/src-tauri/src/engine/loader/sdxl.rs` for initialization and swap
  wiring across UNet and both text encoders.
- SnapFlash-Server integrity tests, live proof scripts, app-state, TODO,
  history, review, and focused summary-bank route.

### How

1. Pin one framework graph.
   - Set `candle-core`, `candle-nn`, and `candle-transformers` to the same exact
     Git revision produced by Round 3.
   - Refresh the lockfile only through the repo's explicit dependency workflow.
   - Add or update a locked/offline metadata guard rejecting crates.io Candle,
     mixed revisions, or duplicate Candle graphs.
2. Replace only shared internals.
   - Feed loaded safetensor tensors to `parse_sdxl_lora_pairs` and
     `VarMapSwapTransaction`.
   - Wrap the existing direct/Kohya mapping in `SdxlLoraTargetResolver`.
   - Hold SnapFlash-Server's exclusive generation/model lease across plan
     preparation and apply; Candle does not serialize concurrent inference.
   - Keep safe adapter-name/path handling, license and catalog policy,
     inspection JSON, Tauri/API types, queues, caches, and resource policy in
     SnapFlash-Server.
   - Delete superseded local pair validation, delta math, base-copy, and
     rollback code only after parity passes.
3. Prove deterministic parity.
   - Use the same model tensors, adapter bytes, strength, and target resolver
     before and after migration.
   - Compare per-component target inventory and the canonical Candle base,
     effective-delta, and merged hashes.
   - Prove base -> adapter A -> adapter B -> exact base with no temporary
     merged model file.
   - Re-run injected component-2/component-3 rollback and invalid later-
     component tests through the public Candle API.
4. Preserve product behavior.
   - Keep direct, queued, img2img, inpaint, and current ControlNet behavior
     unchanged.
   - Run one bounded live same-seed regression only after tensor-level parity
     and a quiet-host memory preflight are green.

### Current blocker

There is no known Candle API, test, dependency, or memory blocker. The only
sequencing gate is publication of the exact Round 3 Candle revision; do not pin
SnapFlash-Server to an uncommitted worktree or moving branch.

### Done when

- SnapFlash-Server resolves exactly one Candle Git revision and no crates.io or
  duplicate Candle graph.
- Existing three-component LoRA tests call Candle rather than local tensor or
  transaction internals.
- The same adapter/model pair produces identical component targets and
  canonical base/delta/merged hashes.
- Base -> A -> B -> base and injected rollback proofs are green.
- Local duplicate LoRA math/transaction code is removed without changing
  filename, mapping, licensing, report, inpaint, ControlNet, queue, or API
  policy.
- Focused and full local verification pass, the worktree is clean, and guarded
  direct-main publication proves remote equality.

### Verification

- Focused SnapFlash-Server LoRA parser, inspector, transaction, and SDXL loader
  tests.
- Locked/offline Cargo metadata/source and duplicate-graph guards.
- Deterministic target/hash comparison plus A -> B -> base and component-2/3
  rollback tests.
- Existing inpaint, ControlNet, queue, direct-generation, API, and integrity
  gates.
- One owner-scoped bounded live regression only if the repository's admitted
  assets and memory policy permit it.
- Exact staged-file and secret/path audit followed by the guarded direct-main
  helper; no PR and no hosted CI.

## Sequencing hold

After INT-4, EdgeSymbio consumes the same Candle revision, replaces its
UNet-only generic LoRA internals, and retains mutable maps for both SDXL text
encoders. SnapFlash runtime-context hardening, ControlNet hooks, inpainting
promotion, and optional LFM2-VL captioning remain later ordered tasks in
`docs/FORK_OVERLAYS.md`.

## Deferred outside this product backlog

Gknome adoption, lower-bit vision quantization, generic VLM traits, video,
true text batching, converters, WebGPU, broad WSL replay, public signing, and
LTS remain separate repository or future-product work. They are not release
gates for INT-4 and must not be introduced without a scoped proposal and
acceptance contract.

---
AI-edited: 2026-08-12T16:04:00-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=three-repo-round-3 | change=opened exact SnapFlash reconsumption with the consumer-owned execution lease
