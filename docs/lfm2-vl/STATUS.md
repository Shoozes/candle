# LFM2.5-VL Current Status

## Release Identity

- Compatibility baseline: Candle 0.11.0 at
  `31f35b147389700ed2a178ee66a91c3cc25cc80d`.
- Upstream integration base: Candle main at
  `6f74e7c390c717f8fd34f23ce02aceb058173370`.
- Published combined-overlay source checkpoint and verified remote `main`:
  `e2c6565d2970de7a9e507b7759a608d3a2c827e7`, tree
  `18c1600fe0278754c697c83cbc6113cb69ab39bc`.
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
  2026-08-13. This documentation-only state reconciliation is its direct
  fast-forward successor. No annotated tag, hosted release, repository-rule
  change, secret inspection, or hosted-CI invocation was authorized or used.
- Models, caches, downloads, generated proof, Cargo output, and
  `.tools/.secrets/` remain ignored or external. Operator disk cleanup removed
  the local model/cache inputs; do not reconstruct or download them implicitly.

## Current Product State

- LFM2.5 text configuration, embedding prefill, cached decode, and reset are
  config-driven and compatibility-preserving.
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

## Latest Integrity Review (2026-08-16)

- No missing LFM2-VL module export, broken example import, production stub, or
  incomplete owned feature was found. The public example test binary compiles
  and exercises the reexport surface.
- Q8 MMProj source-policy validation now has one implementation shared by
  parse-time and resolved-runtime checks; focused tests cover valid GGUF/F32
  and invalid native, split, BF16, and F16 combinations.
- The stale SDXL attention TODO was replaced with the actual contiguous-layout
  invariant; no speculative kernel optimization was introduced.
- `summary_bank.json` now separates the active linked-worktree hazard from the
  archived Gknome attempt, routes newly found upstream VAE/runtime panic work,
  and keeps the publication route at 103.7 KiB. The default orientation route
  is 130.9/256 KiB.
- `START_HERE.md` and `docs/FORK_OVERLAYS.md` no longer repeat completed
  lineage and parity narratives. This file holds current truth; `TODO.md`
  holds only active or explicitly deferred work; `HISTORY.md` holds completed
  detail.
- This pass found one additional owned edge case: the legacy public
  `Lfm2Config::into_config` method still panics on malformed input by contract,
  while `try_into_config` provides the safe path. It is now routed as a
  post-release compatibility task under decision D-0059 rather than changed
  during the frozen snapshot boundary.
- `summary_bank.json` now has a focused
  `issue__lfm2_config_compatibility` route. The large failure log is no longer
  repeated in the reference-environment or linked-worktree groups; the
  dedicated containment route remains the owner for that history.
- A repository-wide incomplete-logic scan found additional upstream
  `todo!`/`unimplemented!` and unchecked serialization/configuration paths.
  They are shaped in `TODO.md` as post-0.2.0, one-subsystem-at-a-time work so
  the frozen candidate is not silently widened.

## Current Verification

- 2026-08-16 native Windows PowerShell 7.6.4: `cargo fmt --all -- --check`,
  `cargo check --locked --offline -j 2 -p candle-core -p candle-nn
  -p candle-transformers -p candle-vlm`, and the three focused example checks
  passed.
- 2026-08-16 focused tests passed: `candle-vlm` 37/37,
  `candle-transformers --lib lfm2` 35/35, and `candle-examples --example
  lfm2-vl` 32/32.
- 2026-08-16 summary-bank validation passed for 31 groups with a 130.9 KiB
  default union; the bundled Python module-layout verifier passed all
  registered splits; local Markdown targets passed for 18/18 state files.
- 2026-08-16 bounded local harness checks passed: release receipt 22/22,
  preflight smoke, and bounded-oracle smoke. The system `python` command is
  absent from `PATH`; the module-layout result used the repository-session
  bundled Python executable explicitly.
- The remaining bullets in this section retain the 2026-08-13 published
  checkpoint's broader local evidence. They were not all replayed in the
  managed shell for this documentation/context pass.
- `cargo fmt --all -- --check`: passed.
- The complete `lfm2-vl` example suite passed 32/32, including the 17 focused
  argument-policy cases. Focused LoRA parser tests passed 3/3 and the
  disabled-feature attention regression passed 1/1.
- `PYO3_NO_PYTHON=1 cargo check --locked --offline -j 2 --workspace`: passed.
- `PYO3_NO_PYTHON=1 cargo clippy --locked --offline -j 2 --workspace
  --all-targets -- -D warnings`: passed.
- Summary-bank validation passed under PowerShell 7 and 5.1 for 30 groups at
  the 2026-08-13 checkpoint; the current PowerShell 7 replay above passes 31
  groups. A temporary negative fixture proved archived groups without
  `_archive_note`
  are rejected.
- Module layout passed for every registered include-based split using the
  bundled read-only Python runtime.
- Both overlay manifests and the repository union passed at 156/20/167 with
  13 shared paths.
- Local Markdown targets across project/fork/LFM2-VL docs passed 18/18 files.
- `PYO3_NO_PYTHON=1 cargo test --locked --offline -j 2 --workspace --exclude
  candle-datasets --exclude candle-pyo3` passed all selected unit,
  integration, and doc-test lanes.
- Release-receipt contract tests passed 22/22 assertions under PowerShell 7
  and 22/22 under Windows PowerShell 5.1.
- `git diff --check` and the complete manifest diff inspection passed.

## Gaps And Blockers

- The combined-overlay implementation has no known owned source or local-check
  blocker. The last committed source publication is complete; this closeout
  pass has five reviewed documentation/context edits pending explicit
  stage/commit authorization.
- Production model inputs are absent after operator cleanup. Existing retained
  hash-bound parity remains historical evidence; no new live model/CUDA claim
  is made in this task.
- The upstream `candle-datasets` runtime test performs live HTTP even under
  Cargo offline mode. It remains an owner-scoped skip; crate check and Clippy
  still cover its source.
- `candle-pyo3` tests cannot link in the cleaned local environment: the system
  Python launcher has no registered interpreter, the bundled Python is 3.12
  while the crate requires `abi3-py313`, and interpreter-free test linking
  cannot find `python3.lib`. Workspace check and warnings-denied Clippy cover
  the crate with `PYO3_NO_PYTHON=1`; its runtime tests remain an explicit
  environment skip rather than a source failure.
- Lower-than-Q8 vision quantization, video, true text batching, generic VLM
  traits, WebGPU/WASM, public signing, and LTS are deferred product scope.
- Reachable panic/stub candidates outside both frozen overlays are not hidden:
  disabled-feature model attention, stable-diffusion VAE input assumptions,
  selected model/operator and example stub branches, core dtype/dummy-backend
  panics, fallible safetensors serialization, and the legacy LFM2 configuration
  conversion boundary have explicit post-release tasks in `TODO.md`.
- The initial integrity pass could not run Git status/diff or the WSL replay
  because the managed environment denied WSL enumeration and the checkout's
  `.git` pointer targets a Linux worktree path. An elevated read-only retry
  reached the linked worktree: `main` and `origin/main` both resolve to
  `b4e1aacf4c531fe6e6e1844e4c74451ecef02fed`, while the five reviewed files
  remain modified. The guarded `gitpush.ps1 -DryRun -Yes` therefore refused
  the dirty worktree before any fetch, staging, commit, or push.

## Exact Next Task

After explicit commit authorization, review and commit the five listed
documentation/context edits, rerun the guarded publisher, and publish only if
`main` remains clean and no longer behind `origin/main`. If publication is
already complete, report that state without a redundant push. With separate
explicit owner authority, the later release task is to create and publish the
annotated tag `candle-overlays-mvp-0.2.0`, emit the external identity receipt,
create the matching hosted release, and apply owner-selected immutability
rules; do not combine those families.

---
AI-edited: 2026-08-16T00:20:34-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=ultra | task=closing-session | change=recorded focused verification, exact main parity, and the guarded publisher dirty-worktree blocker
