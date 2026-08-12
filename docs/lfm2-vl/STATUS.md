# LFM2.5-VL Status

## Baseline and Publication

- Model and compatibility baseline: Candle 0.11.0 at
  `31f35b147389700ed2a178ee66a91c3cc25cc80d`.
- Upstream integration base: Candle main at
  `6f74e7c390c717f8fd34f23ce02aceb058173370`.
- Round 1 public-loader revision:
  `c0fb3a9fe098e50d07ec1b749c77015d7bd8d9a5`.
- EdgeSymbio Round 2 consumer revision:
  `d535a4f56f5a8e06407cb4b8f5be0df7f3121327`.
- Integration and publication branch: `main`; owner-reviewed work lands
  directly without a pull request.
- Historical implementation checkpoint:
  `c9b60f0b906fa8fe70423295e2e1164648a8fa53` on
  `feat/lfm2-vl-mmproj`.
- Immutable first-MVP snapshot: annotated tag `lfm2-vl-mvp-0.1.0` peels to
  `ff885586f6d44a3d9b9ac1724032cdf5f0155384`. Do not move or reuse it for the
  coordinated runtime.
- Current LFM2-VL overlay: 150 paths, exactly 15 fork-origin modifications and
  135 mod-owned additions. The SnapFlash-derived overlay is 8 paths (2
  fork-origin modifications and 6 additions); the repository-wide union is
  157 paths across both overlays.

## Worktree Boundary

- Native Windows/MSVC is the product and primary proof lane; WSL2/Linux is a
  secondary portability replay.
- `C:\DevStuff\candle-mods` is a WSL-owned linked worktree attached to local
  `main`. Use `NVIDIA-Workbench` WSL Git for status, staging, commits, and
  revision checks.
- Owner-reviewed work lands directly on `main`. Broad staging, force-push,
  implicit merge/rebase, hosted-CI evidence, PR creation, and secret inspection
  remain prohibited.
- `.tools/gitpush.ps1` is the only authorized publication path. It may publish
  an already-reviewed fast-forward `main`; it must not stage or commit.

## Current Phase

- Product phase: coordinated three-repository integration, Rounds 1 and 2
  published; Round 3 Candle LoRA promotion implemented and locally green.
- The reusable hybrid constructor now lives at
  `candle_vlm::lfm2_vl::load_lfm2_vl_hybrid`. It accepts explicit local text,
  tokenizer, processor, MMProj, dtype, device, and execution-policy inputs and
  returns the paired model, processor, prompt, and exact consumed-file list.
- The example is a thin CLI/reporting adapter. Candle performs no discovery,
  download, hidden fallback, retained-handle admission, hashing, resource
  leasing, or product-proof publication.
- Independent LFM2-VL and SnapFlash-derived manifests plus focused and union
  verifiers prevent one overlay from silently claiming another overlay's files
  or proof.
- EdgeSymbio now pins the Round 1 Candle revision and passes its bounded,
  CLI-only 450M CPU/F32 token-level proof. It remains intentionally absent from
  API, Tauri, model packs, release sweeps, and product vision claims.
- Candle now exposes validated three-component SDXL LoRA parsing, injected
  target resolution, canonical base/delta/merged hashes, and rollback-capable
  `VarMapSwapTransaction` replacement/clear under a consumer-owned exclusive
  model lease. SnapFlash-Server remains on its reviewed crates.io graph until
  this exact Round 3 candidate is published.

## Last Green Verification

### Round 3 SDXL LoRA promotion gate

- `cargo fmt --all -- --check`: passed after the focused implementation.
- `cargo test --locked --offline -j 2 -p candle-transformers
  stable_diffusion`: passed 12/12 focused LoRA tests, including fail-before-
  write revision exhaustion.
- `cargo test --locked --offline -j 2 -p candle-transformers`: passed 71/71
  library, 5/5 generation, and 8/8 NMS tests; one unrelated doc test remains
  intentionally ignored.
- `cargo clippy --locked --offline -j 2 -p candle-transformers --all-targets --
  -D warnings`: passed.
- `bash scripts/snapflash/verify-mod-manifest.sh`: passed at 8 total paths, 2
  fork modifications, and 6 additions.
- `bash scripts/verify-fork-overlays.sh`: passed at 157 union paths, two
  overlays, and five registered shared paths.
- `scripts/lfm2-vl/verify-summary-bank.ps1`: passed 24 groups; the focused
  LoRA transaction route is 72.2 KiB and defaults remain 121.9/256 KiB.
- `cargo test --locked --offline -j 2 --workspace --exclude
  candle-datasets`: passed every remaining workspace unit, integration, and
  doc-test lane on native Windows. `cargo check --locked --offline -p
  candle-datasets` passed, and strict full-workspace/all-target Clippy passed
  with `-D warnings`.
- The unexcluded workspace attempt and focused dataset replay both stopped only
  because the pre-existing `candle-datasets::hub::tests::test_dataset` performs
  live HTTP even under Cargo offline mode and `HF_HUB_OFFLINE=1`; F-0053 records
  the exact owner-scoped skip. No network access was granted.
- No model, CUDA workload, Python oracle, llama.cpp process, checkpoint, or
  network dependency was loaded.

### Round 1 public-loader release gate

- `cargo fmt --all -- --check`: passed.
- `cargo test --locked --offline -j 2 -p candle-vlm`: passed, 35/35 unit tests
  plus doc tests.
- `cargo test --locked --offline -j 2 -p candle-examples --example lfm2-vl`:
  passed, 32/32.
- Strict targeted Clippy passed for `candle-vlm --all-targets` and the
  `lfm2-vl` example with `-D warnings`.
- `cargo test --locked --offline -j 2 --workspace`: passed on native Windows,
  including core, NN, transformer 59/59, VLM 35/35, WASM, Python-binding build,
  integration, and doc-test lanes.
- `cargo clippy --locked --offline -j 2 --workspace --all-targets -- -D
  warnings`: passed.
- The workspace-wide commands selected the already-installed Python 3.13
  interpreter because the unrelated `candle-pyo3` ABI3 feature rejects the
  bundled Python 3.12 interpreter. No Python package installation or network
  access occurred.
- Public-loader fixtures prove split dense, direct dense GGUF, and direct Q8_0
  GGUF construction. Their four generated inputs are size/hash pinned and
  contain no production checkpoint bytes.
- The LFM2-VL manifest verifier passes at 150/15/135 with 13 text and six
  binary fixture files. The root overlay verifier passes at 153 paths, two
  overlays, and five registered shared paths. The summary bank passes with 23
  groups.
- No production checkpoint, Python oracle, llama.cpp process, CUDA inference,
  dependency download, or concurrent large-model run was started.

### Retained production evidence

- Admitted 450M model: `LiquidAI/LFM2.5-VL-450M` at revision
  `fc6221ca597f3315e4f82fc2df606783267b34ba`; artifact-manifest SHA-256
  `659c8421530586b6cc28c094cfcdc69719ea8626f2abc0efd9eec4ac2a68a984`.
- Six bounded 450M routes remain green: CPU/CPU F32, all-CUDA F32/BF16/F16,
  CPU-text/CUDA-vision F32, and CUDA-text/CPU-vision F32. Each produced exact
  IDs `[1098, 4646, 5251]`, reset cache, exited zero, and released its PID.
- Official 1.6B native Windows CPU-F32 parity remains green at 51/51 tensors;
  comparison SHA-256 is
  `9a0b16256a222678f9dce1282660e49fc6d19103cc6dd6a53c824bb58a6412c0`.
- Detailed production commands, thresholds, and evidence identities live in
  `PARITY.md` and `HISTORY.md`; they are not duplicated here.

## Proven Behavior

- Config-driven LFM2.5 text, embedding prefill, cached decode, and reset.
- SigLIP2 NaFlex, pixel unshuffle/projector, composite native model, checked
  raw-image processing, prompt expansion, and multi-image feature insertion.
- Native safetensors, quantized GGUF text plus split dense MMProj, direct GGUF
  MMProj, CPU-F32 native Q8_0 execution, strict inventory/provenance checks,
  and controlled malformed-input errors.
- Public local-only hybrid assembly with exact consumed-file inventory and
  deterministic construction tests for every supported hybrid form.
- Deterministic native/hybrid evidence, official 450M and 1.6B CPU parity,
  complete advertised 450M placement/dtype parity, and official GGUF
  same-artifact decoded-output agreement with pinned llama.cpp.
- Kill-on-close Windows Job containment, timeout/memory enforcement, exact PID
  cleanup, quiet-host admission, and no-clobber evidence publication.

## Known Gaps and Conflicts

- EdgeSymbio's token-level CPU/F32 proof matches exact generated IDs, text,
  image geometry, spans, stop reason, and in-process cache replay. Its observed
  prefill-logits hash differs from the standalone Candle executable; no
  source/dependency/feature drift was found, so this remains a recorded
  non-bitwise observation rather than a falsely explained numerical claim.
- Candle's upstream dataset smoke remains network-backed and has no local
  fixture. It is excluded only from the workspace test execution; the crate
  check and its all-target Clippy compile remain green. See F-0053.
- SnapFlash-Server still owns duplicate LoRA parsing/math/transaction code and
  resolves Candle from crates.io. It is intentionally unchanged until the
  Round 3 Candle revision is cleanly published.
- EdgeSymbio still owns a separate UNet-only LoRA transaction and has not yet
  retained mutable maps for both SDXL text encoders. Its migration follows the
  SnapFlash regression witness.
- Lower-than-Q8 vision quantization, video, true text batching, generic VLM
  traits, converters, WebGPU/WASM, broad WSL replay, public signing, and LTS are
  deferred future scope, not hidden MVP promises.
- The prior llama.cpp residency incident required a host restart. Exact cause
  remains unproven; `FAILURE_LOG.md` F-0008 containment is mandatory for every
  future model run.

## Blockers

- No Candle Round 3 implementation, test, dependency, import/export, memory,
  or focused verification blocker is known.
- The upstream network-backed dataset test is a disclosed owner-scoped skip,
  not a Round 3 blocker and not permission to enable network verification.
- SnapFlash-Server's only current gate is the exact published Round 3 Candle
  revision; it must not pin an uncommitted worktree or moving branch.
- Edge CUDA/F16, public LFM2-VL routes, and Edge LoRA migration are sequencing
  holds, not concealed failures.
- Hosted GitHub Actions state is intentionally not a blocker or verification
  dependency.

## Active Change Set

- LoRA source: `candle-transformers/src/models/stable_diffusion/lora.rs`,
  `mutable.rs`, and their `mod.rs` exports.
- Overlay proof: `docs/snapflash/MOD_MANIFEST.md`, its independent verifier,
  `docs/FORK_OVERLAYS.md`, the root union verifier, changelog, and focused
  `summary_bank.json` route.
- Current-state records: this file, `TODO.md`, `HISTORY.md`, `DECISIONS.md`,
  `START_HERE.md`, and F-0053 in `FAILURE_LOG.md`.
- Models, caches, downloads, generated proof logs, Cargo output, and
  `.tools/.secrets/` remain ignored or external.

## Exact Next Task

After the complete local Candle gate and guarded publication prove local
`main == origin/main`, pass the exact Round 3 commit to SnapFlash-Server. Pin
all of its Candle packages to that immutable revision, replace local generic
LoRA pair/math/transaction internals with the public Candle API, and prove
identical targets/hashes plus base -> A -> B -> base before deleting the
duplicate code. Keep SnapFlash filename, mapping, license, report, queue,
inpaint, ControlNet, and API policy application-owned. Do not start Edge LoRA
migration, ControlNet/inpainting promotion, or another model run before that
first consumer regression is green.

---
AI-edited: 2026-08-12T16:05:00-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=three-repo-round-3 | change=recorded audited LoRA proof, owner-scoped dataset skip, blockers, and exact SnapFlash handoff
