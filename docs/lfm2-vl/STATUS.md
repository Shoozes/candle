# LFM2.5-VL Current Status

## Release Identity

- Compatibility baseline: Candle 0.11.0 at
  `31f35b147389700ed2a178ee66a91c3cc25cc80d`.
- Live HF integration base: Candle main at
  `7c2e89295dad4aeebc6ef7a92c255360b6957c2c` (`#3950`). Local S3 merge
  commit: `25d676f5663f152cf9371b405236d75fe110d14f`. The earlier 0.2.0
  publication base `6f74e7c390c717f8fd34f23ce02aceb058173370` remains the
  frozen receipt identity, not the live union gate.
- Inventory: `docs/lfm2-vl/UPSTREAM_SYNC.md`. Reject later `#3952`
  fancy-regex, `#3953`, `#3959`, and `#3954` Remove ug. This SHA keeps
  `candle-ug` and `tokenizers` `onig`, and already includes `hf-hub 1.0.0`
  plus `tokenizers 0.23.1` (lock currently resolves `0.23.2` with `onig`
  still present). Consumer risk is not zero.
- Published combined-overlay source checkpoint: `e2c6565d2970de7a9e507b7759a608d3a2c827e7`,
  tree `18c1600fe0278754c697c83cbc6113cb69ab39bc`.
- Last verified app/source `main` head before this proof-gap slice:
  `226f8be21cb955efbbd65254db479ddd0a9504b2`; this task adds the bounded
  3B/Q8 proof contract on top of that clean head.
- Immutable first-MVP tag: `lfm2-vl-mvp-0.1.0` peels to
  `ff885586f6d44a3d9b9ac1724032cdf5f0155384`; never move or reuse it.
- Proposed combined tag: `candle-overlays-mvp-0.2.0`. It does not yet exist
  and no hosted release is claimed.
- Exact published round/repository lineage lives in `docs/FORK_OVERLAYS.md`;
  completed implementation and proof narratives live in `HISTORY.md` and
  `PARITY.md`.

## Worktree And Authority

- Canonical checkout is `C:\DevStuff\candle` on local `main` (WSL path
  `/mnt/c/DevStuff/candle`). Native Windows Git owns `main`. Publish with the
  Candle-owned `.tools/gitpush.ps1` plus ignored `.tools/.secrets/gt.txt`.
  The helper requires a clean reviewed commit and never stages or commits.
  `C:\DevStuff\candle-mods` is a retired linked worktree; do not recreate it.
  WSL is an optional `verify-baseline.sh` replay, not the Git backend.
- Native Windows/MSVC is the product and release-proof lane. WSL2/Linux is a
  secondary portability replay, not the product platform.
- The guarded helper published and remotely verified the source checkpoint on
  2026-08-13 and the closeout commits above directly on `main` in this
  session. No annotated tag, hosted release, repository-rule change, secret
  inspection, or hosted-CI invocation is included.
- Current active local slice for the maintained LFM2-VL product chain: **Edge
  pin-adoption** of published `4c1feb82eccd60a14f9f00f84ef6c9bdedcddd06`
  with `tokenizers` 0.23/`onig`. That immutable consumer target remains valid
  even though Candle `main` now contains later administrative and experimental
  commits. The LFM2-VL/Edge C0 chain remains held until the Edge pin and
  tokenizers move together; keep the Edge pin on `238cc176…` until then. A
  separate owner-authorized Candle-only GPT-OSS experiment has delivered its
  C0/C1 source, Task 2 synthetic CPU/resource proof, Task 3 packed CUDA proof,
  and bounded synthetic/exact GGUF-to-weights assembly; see
  `docs/gpt-oss/STATUS.md`. The CUDA result is feature-gated and limited to
  the digest-pinned synthetic fixture on the named local RTX 4090. The exact
  artifact was read in place only: no model bytes entered this checkout. The
  round does not claim tokenizer compatibility, forward-logit parity,
  production support, cross-device parity, or a Q8/default change. The 3B
  native / official Q8 MMProj production-proof gap, overlay 0.2.0 tagging, and
  live product CUDA/model parity remain outside this chain. No implicit 3B
  download.
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
- The architecture supports the 3B shape contract and the official 400M Q8_0
  MMProj path, but neither is a production-support claim yet. The 3B lock is
  `5a414ead75d45db003906d06fb62bd5b6846cec0`; the official GGUF lock is
  `3e0e828198e2abb75a957ad823f5d691c13f0f28`.
- Native trace publication remains unchanged. Direct-GGUF evidence is a
  separate `hybrid-trace` bundle containing projected image embeddings,
  prefill/decode logits, input identities, execution mode, Q8 tensor count,
  and exact cache-reset evidence.
- `candle_vlm::lfm2_vl::load_lfm2_vl_hybrid` is the public local-only hybrid
  assembly boundary. The example remains a thin CLI/reporting adapter; Candle
  performs no discovery, download, retained-handle admission, resource lease,
  or application proof publication.
- Generic SDXL framework additions cover three-component LoRA transactions,
  exact residual admission, opt-in pooled-text/time-ID conditioning,
  lower-precision cast order, and a consumer-test-only rollback seam.
- The LFM2-VL overlay contains 163 paths (17 fork modifications, 146
  additions). The SnapFlash-derived overlay contains 20 paths (8
  modifications, 12 additions). The separate GPT-OSS overlay currently
  registers 35 paths (26 overlay-owned and 9 shared).
  Their current registered union is 200 paths with 21 shared paths, including
  the shared CUDA build path.

## Separate GPT-OSS Experimental Handoff

- Current phase: Task 3 short exact-parity runner and CUDA prefill/decode
  boundary. Short parity is published on `main`; the separate performance
  harness and its 10-test model-free qualification suite are implemented. The
  owner-artifact characterization reached four of six cases before its
  declared deadline, so its real receipt remains pending. It does not change
  EdgeSymbio or symbio-code. Its supplied-artifact run passes real GGUF load,
  CUDA construction, cancellation rollback, bounded prefill/decode parity,
  reset/replay, and teardown.
- Current baseline before this slice: `main`
  `2abf29cd35b0633336cf67d55f8d5ec6ceb62ee1`, tree
  `bdebed85d31a6fe42d3410593fb841ea4709623a`.
- The accepted implementation is published at guarded checkpoint
  `ded81e9be2d6c13406507ecc74842a48f83f4162`; the prior short-parity
  checkpoint was `ea5900f614c35c47822b8363ff18ee676ae2159a`.
- The successful receipt executed candidate tree
  `69830f1d244ef0d13b734ee48d5bf7243366c371` before these post-proof status
  and history edits; the committed source checkpoint includes that later
  reconciliation without changing the receipt's executed source identity.
- Task 1, Task 2, and the prior Task 3 packed-executor checkpoints remain
  accepted. No model bytes entered the repository. The detailed evidence is
  recorded in `docs/gpt-oss/STATUS.md`.
- Delivery files currently under active work:
  `candle-transformers/src/models/gpt_oss/cuda.rs`, `gguf.rs`, `model.rs`,
  `mxfp4.rs`, `runtime.rs`, `candle-examples/examples/gpt-oss-short-parity.rs`,
  `scripts/gpt-oss/run-short-parity.ps1`,
  `candle-examples/examples/gpt-oss-performance.rs`, and
  `scripts/gpt-oss/run-performance.ps1`, and
  `scripts/gpt-oss/test-performance.ps1`.
- Proven: all-32-coordinate GGML wire normalization with mixed signs/scales;
  real serialized fused/split loader values and expert-contribution checks;
  one identity-verified retained file handle with chunk cancellation and
  Windows write/delete sharing denial; lease rejection while a model lives and
  release after teardown/construction failure; public CUDA config validation;
  and synchronized finite-output/final-cancellation cache commit guards.
- Known limitation: the owner-selected product GGUF with SHA-256
  `aab205256a9b6361e410c24de3086e30f907092ca6f9ba8cd4b22c8a2b025778` was
  read only from its external owner-provided path. The fixed tokenizer IDs,
  exact assembly/model construction, and short CUDA parity receipt pass. The
  raw prefill row retains a clipped-tail diagnostic outlier, so the accepted
  criterion is explicitly top-k/mean/total-variation based; Q8 remains the
  maintained product default and broad production claims remain unproven.
- Known blocker: the owner-artifact run in
  `artifacts/gpt-oss/performance/runs/20260921T071321303Z-3cd84d87e5b8`
  completed the 8, 512, 2048, and 8192-token autoregressive cases, then hit
  the explicit 45-minute deadline during the 16384-token case. Its terminal
  records report `timeout`, `4/6`, `report_written: false`, and
  `success_claim: false`; the runner exited with
  `performance overall deadline exceeded: GPT-OSS CUDA operation cancelled`.
  No post-unload recovery receipt or final throughput/TTFO report exists. The
  wrapper cleaned up the runner process and retained 2,414 monitor samples.
- Exact next task: if a performance receipt is still required, rerun
  `run-performance.ps1` with a deadline sufficient for the declared matrix,
  or explicitly declare a narrower matrix, and require the final report plus
  post-unload recovery evidence before changing the gated status. Keep Task 4
  quantized text and split dense MMProj work separately scoped.

## Latest Integrity Review (2026-08-21)

- No missing LFM2-VL module export, broken example import, production stub, or
  incomplete owned feature was found. The public example test binary compiles
  and exercises the reexport surface.
- Q8 MMProj source-policy validation now has one implementation shared by
  parse-time and resolved-runtime checks; focused tests cover valid GGUF/F32
  and invalid native, split, BF16, and F16 combinations.
- The stale SDXL attention TODO was replaced with the actual contiguous-layout
  invariant; no speculative kernel optimization was introduced.
- `summary_bank.json` routes current Git topology through the canonical
  `C:\DevStuff\candle` checkout group and keeps the rejected Gknome attempt
  archived. The default orientation route stays focused at 135.5/256 KiB. The
  text-only LFM2 route no longer repeats the composite VL configuration file;
  the current issue and workflow groups remain covered by the existing route
  set.
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
  repeated in the reference-environment or Git-topology groups; the
  dedicated containment route remains the owner for that history.
- The 3B/Q8 proof slice adds immutable native and official GGUF repository
  entries, exact 3B config/tokenizer/processor checks, bounded custom-code
  admission, direct-GGUF hybrid evidence, dense-versus-native-Q8 comparison,
  and fixture regressions. The current official 3B snapshot has no model
  Python files and an empty `auto_map`, so `trust_remote_code` remains false;
  the model-card custom-code/context claims are recorded as a source conflict
  rather than silently accepted.
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
  lfm2-vl` 33/33.
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
- 2026-08-21 focused proof-gap checks passed: Python reference tools `36 passed,
  4 skipped`; `cargo test --locked --offline -j 2 -p candle-examples
  --example lfm2-vl` passed `33/33`, including direct-Q8 hybrid evidence,
  retained Q8 selection, and exact cache-reset replay. No production model,
  GGUF, oracle trace, or model code was loaded.
- The warnings-denied Windows example Clippy lane passed after reducing the
  hybrid evidence writer boundary; the final WSL replay also passed the locked
  format/check/layout/diff/mod-manifest gate from `16:36:17Z` to `16:36:54Z`.
- Summary-bank validation passed for 31 groups with a 135.5 KiB default
  union; the Python 3.13 module-layout verifier passed all registered splits.
- Current overlay inventories are 163 LFM2-VL / 20 SnapFlash / 17 GPT-OSS,
  with a 183-path union and 20 registered shared paths. Live union baseline
  is `7c2e8929…`. The frozen
  unpublished 0.2.0 candidate still recorded 156/20/167.
- `PYO3_NO_PYTHON=1 cargo check --locked --offline -j 2 --workspace`, the
  matching warnings-denied workspace Clippy gate, and the locked/offline
  workspace test/doc-test suite excluding `candle-datasets` and
  `candle-pyo3` all passed. The direct `candle-pyo3` package test also passed
  with Python 3.13 (0 tests discovered).
- Production model/CUDA parity and hosted CI were not rerun. Historical parity
  results remain in `HISTORY.md`; no new production-runtime claim is made here.

## Gaps And Blockers

- The local implementation and focused checks have no known owned blocker. The
  clean starting head was `226f8be21cb955efbbd65254db479ddd0a9504b2`.
- Production model inputs are absent after operator cleanup. The locked 3B
  native snapshot requires 6,264,993,989 bytes and the official GGUF entry
  contains a 1,674,454,240-byte text file plus 853,993,088-byte F16 and
  583,109,120-byte Q8_0 MMProj files. No download is implicit. Native 3B and
  official 400M Q8 production claims remain Gated until their external
  manifests, bounded oracle traces, Candle receipts, and cleanup evidence
  exist.
- The current 3B model card advertises custom code and 32,768 context, but the
  pinned config has no `auto_map`/Python code and 128,000 text positions. A
  compatible pinned Transformers/oracle environment must resolve this before
  native inference is admitted; the lock currently follows the exact snapshot.
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
- Git identity is the `C:\DevStuff\candle` checkout. The retired
  `candle-mods` linked worktree is not part of current operations.
- WSL2/Linux is no longer blocked for the pinned CPU baseline: the required
  toolchain and components are installed and the full replay passed. Future
  WSL runs remain secondary portability evidence, not a replacement for the
  native Windows release lane.

## Exact Next Task

Acquire the exact locked 3B native snapshot and official 3B-GGUF text/F16/Q8_0
MMProj files only through the guarded external path, build artifact manifests,
run the pinned CPU-F32 oracle/Candle native trace and dense-versus-Q8 hybrid
comparison, then update the production rows only if every receipt and cleanup
condition is green. If acquisition or the pinned oracle remains unavailable,
leave the status Gated and retain the explicit blocker. The separate
`candle-overlays-mvp-0.2.0` publication task remains deferred until its own
authorization.

---
AI-edited: 2026-09-21T00:00:00-04:00 | agent=Codex | model=unknown | effort=high | task=gpt-oss-performance-qualification | change=recorded the hardened performance harness, model-free tests, and exact owner-artifact timeout evidence without claiming a final receipt
