# LFM2.5-VL Current Status

## Release Identity

- Compatibility baseline: Candle 0.11.0 at
  `31f35b147389700ed2a178ee66a91c3cc25cc80d`.
- Upstream integration base: Candle main at
  `6f74e7c390c717f8fd34f23ce02aceb058173370`.
- Published combined-overlay source checkpoint: `e2c6565d2970de7a9e507b7759a608d3a2c827e7`,
  tree `18c1600fe0278754c697c83cbc6113cb69ab39bc`.
- Last verified app/source `main` head: `53b2f3b3a1e7a9e49126a57213c3abb74a65c698`,
  tree `b431e817325e8f3c12da30c4c3c34b9dfe7f5cfc`; the WSL replay and this
  follow-up state record are source-neutral documentation work.
- Immutable first-MVP tag: `lfm2-vl-mvp-0.1.0` peels to
  `ff885586f6d44a3d9b9ac1724032cdf5f0155384`; never move or reuse it.
- Proposed combined tag: `candle-overlays-mvp-0.2.0`. It does not yet exist
  and no hosted release is claimed.
- Exact published round/repository lineage lives in `docs/FORK_OVERLAYS.md`;
  completed implementation and proof narratives live in `HISTORY.md` and
  `PARITY.md`.

## Worktree And Authority

- `C:\DevStuff\candle-mods` is a WSL-owned linked worktree attached to local
  `main`. Use the `NVIDIA-Workbench` WSL Git backend for status, revision,
  staging, commit, merge, and publication checks.
- Native Windows/MSVC is the product and release-proof lane. WSL2/Linux is a
  secondary portability replay, not the product platform.
- The guarded helper published and remotely verified the source checkpoint on
  2026-08-13 and the closeout commits above directly on `main` in this
  session. No annotated tag, hosted release, repository-rule change, secret
  inspection, or hosted-CI invocation is included.
- Current active local slice: none. The LFM2 compatibility implementation,
  tests, TODO/history/decision/status updates, README command correction, and
  summary-bank cleanup are committed in the closeout head above.
- Models, caches, downloads, generated proof, Cargo output, and
  `.tools/.secrets/` remain ignored or external. Operator disk cleanup removed
  the local model/cache inputs; do not reconstruct or download them implicitly.

## Current Product State

- LFM2.5 text configuration, embedding prefill, cached decode, and reset are
  config-driven and compatibility-preserving. Maintained external
  configuration loading uses fallible `try_into_config`; the legacy
  infallible `into_config` API is deprecated without a signature change.
- SigLIP2 NaFlex, image preprocessing, crop/thumbnail metadata, prompt
  expansion, pixel unshuffle/projector, and multi-image feature insertion are
  implemented with checked limits and controlled malformed-input errors.
- Native safetensors, quantized GGUF text plus split dense MMProj, direct GGUF
  MMProj, and CPU-F32 native Q8_0 MMProj execution are implemented.
- `candle_vlm::lfm2_vl::load_lfm2_vl_hybrid` is the public local-only hybrid
  assembly boundary. The example remains a thin CLI/reporting adapter; Candle
  performs no discovery, download, retained-handle admission, resource lease,
  or application proof publication.
- Generic SDXL framework additions cover three-component LoRA transactions,
  exact residual admission, opt-in pooled-text/time-ID conditioning,
  lower-precision cast order, and a consumer-test-only rollback seam.
- The LFM2-VL overlay contains 156 paths (16 fork modifications, 140
  additions). The SnapFlash-derived overlay contains 20 paths (8
  modifications, 12 additions). Their registered union is 167 paths with 13
  shared paths.

## Latest Integrity Review (2026-08-20)

- No missing LFM2-VL module export, broken example import, production stub, or
  incomplete owned feature was found. The public example test binary compiles
  and exercises the reexport surface.
- Q8 MMProj source-policy validation now has one implementation shared by
  parse-time and resolved-runtime checks; focused tests cover valid GGUF/F32
  and invalid native, split, BF16, and F16 combinations.
- The stale SDXL attention TODO was replaced with the actual contiguous-layout
  invariant; no speculative kernel optimization was introduced.
- `summary_bank.json` separates the active linked-worktree hazard from the
  archived Gknome attempt and keeps the default orientation route focused at
  129.8/256 KiB. The text-only LFM2 route no longer repeats the composite VL
  configuration file; the current issue and workflow groups remain covered by
  the existing route set.
- `START_HERE.md` and `docs/FORK_OVERLAYS.md` no longer repeat completed
  lineage and parity narratives. This file holds current truth; `TODO.md`
  holds only active or explicitly deferred work; `HISTORY.md` holds completed
  detail.
- The legacy public `Lfm2Config::into_config` compatibility boundary is now
  explicitly deprecated without changing its signature. Maintained in-tree
  callers use `try_into_config`, and malformed parsed configuration is covered
  by the focused rejection regression. Direct deprecated callers remain
  responsible for validation and may still observe the historical panic; this
  limitation is recorded in D-0059/D-0060.
- `summary_bank.json` now has a focused
  `issue__lfm2_config_compatibility` route. The large failure log is no longer
  repeated in the reference-environment or linked-worktree groups; the
  dedicated containment route remains the owner for that history.
- A repository-wide incomplete-logic scan found additional upstream
  `todo!`/`unimplemented!` and unchecked serialization/configuration paths.
  They are shaped in `TODO.md` as post-0.2.0, one-subsystem-at-a-time work so
  the frozen candidate is not silently widened.
- No architect or research-inbox files are present in the repository; no
  unverified drop-in recommendations were promoted into current state or the
  backlog.

## Current Verification

- 2026-08-20 native Windows PowerShell with Rust/Cargo 1.97.1: `cargo fmt
  --all -- --check`, locked/offline checks for `candle-core`, `candle-nn`,
  `candle-transformers`, `candle-vlm`, and the `lfm2`, `quantized-lfm2`, and
  `lfm2-vl` examples passed.
- Focused tests passed: `candle-vlm` 37/37,
  `candle-transformers --lib lfm2` 36/36, and `candle-examples --example
  lfm2-vl` 32/32.
- The closing-session rerun also passed the locked/offline checks for all four
  affected libraries and all three LFM2 examples, workspace warnings-denied
  Clippy, and the workspace unit/integration/doc-test lane excluding the
  live-HTTP `candle-datasets` and testless `candle-pyo3` aggregate packages.
- 2026-08-21 WSL2 replay used the installed pinned `1.97.1-x86_64-unknown-linux-gnu`
  toolchain with `cargo`, `clippy`, `rustfmt`, and `rust-std`; the complete
  `scripts/lfm2-vl/verify-baseline.sh` passed, including all locked/offline
  crate/example checks, module-layout, diff gates, and the mod manifest.
  The verifier-only Cargo.lock SHA-256 was
  `9b7aa15899ae8acf7b1a09b951ddba2f16462137eee2fed0db863a9d84707175`.
- Summary-bank validation passed for 31 groups with a 129.8 KiB default
  union; the Python 3.13 module-layout verifier passed all registered splits.
- Both overlay manifests and the repository union passed at 156/20/167 with
  13 shared paths. The local/remote `main` heads and trees match exactly.
- `PYO3_NO_PYTHON=1 cargo check --locked --offline -j 2 --workspace`, the
  matching warnings-denied workspace Clippy gate, and the locked/offline
  workspace test/doc-test suite excluding `candle-datasets` and
  `candle-pyo3` all passed. The direct `candle-pyo3` package test also passed
  with Python 3.13 (0 tests discovered).
- Production model/CUDA parity and hosted CI were not rerun. Historical parity
  results remain in `HISTORY.md`; no new production-runtime claim is made here.

## Gaps And Blockers

- The combined-overlay implementation has no known owned source or local-check
  blocker. The last verified app/source closeout head is
  `53b2f3b3a1e7a9e49126a57213c3abb74a65c698`; the WSL documentation update
  does not alter application source.
- Production model inputs are absent after operator cleanup. Existing retained
  hash-bound parity remains historical evidence; no new live model/CUDA claim
  is made in this task.
- The upstream `candle-datasets` runtime test performs live HTTP even under
  Cargo offline mode. It remains an owner-scoped skip; crate check and Clippy
  still cover its source.
- `candle-pyo3` currently defines no runtime tests; its package test lane
  compiled and passed with Python 3.13. The live-HTTP `candle-datasets` test
  remains intentionally excluded from the offline gate.
- Lower-than-Q8 vision quantization, video, true text batching, generic VLM
  traits, WebGPU/WASM, public signing, and LTS are deferred product scope.
- Reachable panic/stub candidates outside both frozen overlays are not hidden:
  disabled-feature model attention, stable-diffusion VAE input assumptions,
  selected model/operator and example stub branches, core dtype/dummy-backend
  panics, and fallible safetensors serialization have explicit post-release
  tasks in `TODO.md`. The LFM2 configuration conversion boundary is now
  deprecated and tracked as a compatibility limitation rather than an active
  TODO item.
- The linked-worktree Git boundary is operational through `NVIDIA-Workbench`;
  the current local and remote refs were read successfully in this pass. No
  related Candle/LFM2/Cargo/Rust process was running after verification.
- WSL2/Linux is no longer blocked for the pinned CPU baseline: the required
  toolchain and components are installed and the full replay passed. Future
  WSL runs remain secondary portability evidence, not a replacement for the
  native Windows release lane.

## Exact Next Task

With separate explicit owner authority, the next release task is to create and
publish the annotated tag `candle-overlays-mvp-0.2.0`, emit the external
identity receipt, create the matching hosted release, and apply owner-selected
immutability rules. Do not combine those families with the post-release safety
tasks in `TODO.md`.

---
AI-edited: 2026-08-20T21:17:10-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=ultra | task=wsl-toolchain-replay | change=recorded the successful pinned Linux toolchain and portable baseline replay
