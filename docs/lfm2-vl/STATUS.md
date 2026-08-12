# LFM2.5-VL Status

## Baseline and Publication

- Model and compatibility baseline: Candle 0.11.0 at
  `31f35b147389700ed2a178ee66a91c3cc25cc80d`.
- Upstream integration base: Candle main at
  `6f74e7c390c717f8fd34f23ce02aceb058173370`.
- Round 1 release parent: `6a7a9ceec6be038b0b4df3c6b06d32597e2762bd`.
  The exact consumer revision is the clean published `Shoozes/candle:main`
  descendant containing this loader promotion; use live Git refs rather than
  copying a prose hash into a dependency manifest.
- Integration and publication branch: `main`; owner-reviewed work lands
  directly without a pull request.
- Historical implementation checkpoint:
  `c9b60f0b906fa8fe70423295e2e1164648a8fa53` on
  `feat/lfm2-vl-mmproj`.
- Immutable first-MVP snapshot: annotated tag `lfm2-vl-mvp-0.1.0` peels to
  `ff885586f6d44a3d9b9ac1724032cdf5f0155384`. Do not move or reuse it for the
  coordinated runtime.
- Current LFM2-VL overlay: 150 paths, exactly 15 fork-origin modifications and
  135 mod-owned additions. The repository-wide union is 153 paths across the
  LFM2-VL overlay and the SnapFlash-derived boundary scaffold.

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

- Product phase: coordinated three-repository integration, Round 1 complete at
  the local release gate.
- The reusable hybrid constructor now lives at
  `candle_vlm::lfm2_vl::load_lfm2_vl_hybrid`. It accepts explicit local text,
  tokenizer, processor, MMProj, dtype, device, and execution-policy inputs and
  returns the paired model, processor, prompt, and exact consumed-file list.
- The example is a thin CLI/reporting adapter. Candle performs no discovery,
  download, hidden fallback, retained-handle admission, hashing, resource
  leasing, or product-proof publication.
- Independent LFM2-VL and SnapFlash-derived manifests plus a union verifier now
  prevent one overlay from silently claiming another overlay's files or proof.
- Round 2 is the exact next gate: EdgeSymbio pins one published Candle revision
  and adds a separate proof-only 450M `Lfm2VlModel`. SnapFlash-Server remains on
  its reviewed crates.io Candle graph until the later shared LoRA API exists.

## Last Green Verification

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

- EdgeSymbio does not yet consume this fork or expose LFM2-VL. Its existing
  `LfmModel` remains text-only and must stay unchanged.
- EdgeSymbio currently lacks an admitted first-proof bundle containing an exact
  compatible 450M text GGUF, direct F16 MMProj GGUF, standalone tokenizer,
  processor configuration, and fixed image with revisions, sizes, hashes, and
  license records. Presence of an untracked file is not proof eligibility.
- SnapFlash-Server has the strongest three-component SDXL LoRA donor behavior,
  but no generic LoRA transaction has been promoted to Candle yet. It must not
  pin this fork merely for the boundary scaffold.
- Lower-than-Q8 vision quantization, video, true text batching, generic VLM
  traits, converters, WebGPU/WASM, broad WSL replay, public signing, and LTS are
  deferred future scope, not hidden MVP promises.
- The prior llama.cpp residency incident required a host restart. Exact cause
  remains unproven; `FAILURE_LOG.md` F-0008 containment is mandatory for every
  future model run.

## Blockers

- Candle Round 1 has no implementation, test, dependency, memory, or local
  verification blocker.
- EdgeSymbio Round 2 can begin dependency pinning, tiny-fixture integration,
  admission code, and controlled failure tests after Candle publication. The
  official CPU/F32 proof is blocked until the exact Edge-owned asset bundle is
  acquired and admitted; CUDA/F16 proof remains sequenced after CPU acceptance.
- SnapFlash-Server is intentionally held before fork pinning and runtime work,
  not blocked. Its next implementation starts only after Edge Round 2 passes
  and Candle publishes the generic three-component LoRA transaction.
- Hosted GitHub Actions state is intentionally not a blocker or verification
  dependency.

## Active Change Set

- Public loader: `candle-vlm/src/lfm2_vl/loading.rs`, its export, crate docs,
  exact loader fixtures, and the thin example adapter.
- Overlay boundary: `docs/FORK_OVERLAYS.md`, both overlay manifests, the root
  union verifier, `AGENTS.md`, `CHANGELOG.md`, and `summary_bank.json`.
- Current-state records: this file, `TODO.md`, `HISTORY.md`, `DECISIONS.md`,
  and `START_HERE.md`.
- Models, caches, downloads, generated proof logs, Cargo output, and
  `.tools/.secrets/` remain ignored or external.

## Exact Next Task

After guarded publication proves local `main == origin/main`, pass the exact
Candle commit to EdgeSymbio. In EdgeSymbio, replace every direct Candle package
with the same immutable Git `rev`, add `candle-vlm`, prove a single Candle
source with locked/offline metadata, and implement the separate CLI-only
CPU/F32 450M LFM2-VL proof lane described in `TODO.md`. Do not begin Candle
LoRA promotion, SnapFlash dependency migration, a public Edge route, CUDA
inference, or any large-model run before that CPU lane's prerequisites and
admission checks are green.

---
AI-edited: 2026-08-12T12:42:54-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=three-repo-round-1 | change=recorded public loader, overlay boundary, verification, consumer blocker, and exact Edge handoff
