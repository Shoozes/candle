# LFM2.5-VL Current Status

## Release Identity

- Compatibility baseline: Candle 0.11.0 at
  `31f35b147389700ed2a178ee66a91c3cc25cc80d`.
- Upstream integration base: Candle main at
  `6f74e7c390c717f8fd34f23ce02aceb058173370`.
- Current combined-overlay candidate parent and `origin/main`:
  `dca9849584e377cebc1da40de966d050733f3bbf`.
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
- The owner authorized a scoped commit and guarded direct push to `main` for
  this source candidate on 2026-08-13. That authorization does not include an
  annotated tag, hosted release, repository-rule change, secret inspection,
  or hosted-CI invocation.
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

## Latest Integrity Review

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
  and trims the publication route from 227.8 KiB to 101.8 KiB. The default
  orientation route is 127.3/256 KiB.
- `START_HERE.md` and `docs/FORK_OVERLAYS.md` no longer repeat completed
  lineage and parity narratives. This file holds current truth; `TODO.md`
  holds only active or explicitly deferred work; `HISTORY.md` holds completed
  detail.
- A repository-wide incomplete-logic scan found additional upstream
  `todo!`/`unimplemented!` and unchecked serialization/configuration paths.
  They are shaped in `TODO.md` as post-0.2.0, one-subsystem-at-a-time work so
  the frozen candidate is not silently widened.

## Current Verification

- `cargo fmt --all -- --check`: passed.
- The complete `lfm2-vl` example suite passed 32/32, including the 17 focused
  argument-policy cases. Focused LoRA parser tests passed 3/3 and the
  disabled-feature attention regression passed 1/1.
- `PYO3_NO_PYTHON=1 cargo check --locked --offline -j 2 --workspace`: passed.
- `PYO3_NO_PYTHON=1 cargo clippy --locked --offline -j 2 --workspace
  --all-targets -- -D warnings`: passed.
- Summary-bank validation passed under PowerShell 7 and 5.1 for 30 groups; a
  temporary negative fixture proved archived groups without `_archive_note`
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
  blocker for the authorized direct `main` source publication. A clean-head
  replay remains required before push.
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
  panics, and fallible safetensors serialization have explicit post-release
  tasks in `TODO.md`.

## Exact Next Task

Complete the owner-authorized scoped source commit, replay the complete local
gate from its clean exact head, and publish reviewed `main` through
`.tools/gitpush.ps1`. Annotated tag `candle-overlays-mvp-0.2.0`, external
identity receipt, hosted release, and repository immutability remain separate
owner actions. After source publication, begin only one explicitly selected
post-release safety task from `TODO.md`; do not combine those families.

---
AI-edited: 2026-08-13T20:08:58-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=ultra | task=repo-integrity | change=reconciled full local verification, explicit environment skips, and owner-authorized source publication boundary
