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
- Candle Round 3 shared LoRA revision:
  `37584ecd2738ba1eb4ec4c1ab218667681f54973`.
- Candle Round 7 exact residual-contract revision:
  `95ac9ff815fbac4f252b4ef6780b5e4a7843f328`.
- Candle INT-5B SDXL `text_time` revision:
  `ba1e8acc142c4683995e4cdbc8b1d933c81e96c6`.
- SnapFlash-Server Round 4 LoRA consumer revision:
  `6e64320fe26e7c3be91262bc0dac99ce53f4c628`.
- SnapFlash-Server Round 6 bounded-runtime implementation revision:
  `d66c1c35158aca7b37e6e1d82e527334b209d93a`.
- Current SnapFlash-Server `main` after INT-5A fail-closed admission:
  `9bc58ccaef77e7ceac0ab4e75a1a4c93acc1cdff`.
- EdgeSymbio Round 5 LoRA consumer revision:
  `633f774a3690df5a8a35b6cac000df4b390316d5`.
- Current EdgeSymbio `main` after bounded proof-owner hardening:
  `eb9c07127321bd7528786c4fa103b92f893991f5`.
- Integration and publication branch: `main`; owner-reviewed work lands
  directly without a pull request.
- Historical implementation checkpoint:
  `c9b60f0b906fa8fe70423295e2e1164648a8fa53` on
  `feat/lfm2-vl-mmproj`.
- Immutable first-MVP snapshot: annotated tag `lfm2-vl-mvp-0.1.0` peels to
  `ff885586f6d44a3d9b9ac1724032cdf5f0155384`. Do not move or reuse it for the
  coordinated runtime.
- Current LFM2-VL overlay: 150 paths, exactly 15 fork-origin modifications and
  135 mod-owned additions. The SnapFlash-derived overlay is 10 paths
  (4 fork-origin modifications and 6 additions); the repository-wide union is
  159 paths across both overlays.

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

- Product phase: coordinated three-repository integration. Rounds 1 through 7
  are published. INT-5A fail-closed admission is published in SnapFlash and
  Candle INT-5B's generic SDXL `text_time` primitive is published. The narrow
  INT-5B.1 lower-precision cast-order follow-up is green locally and awaiting
  publication before INT-5C repins. No numerical ControlNet parity claim has
  been made.
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
- EdgeSymbio's current clean `main` adds a bounded Windows Job Object proof
  runner and deterministic small-process regressions. Its three-component live
  model replay and Candle-375 LFM2-VL reattestation remain owner-authorized
  runtime gates, not source-code failures or inferred green evidence.
- Candle exposes validated three-component SDXL LoRA parsing, injected target
  resolution, canonical base/delta/merged hashes, and rollback-capable
  `VarMapSwapTransaction` replacement/clear under a consumer-owned exclusive
  model lease. SnapFlash-Server and EdgeSymbio now consume that exact published
  framework revision without retaining duplicate generic transaction math.
- The existing Candle UNet additional-residual hook is sufficient for the
  application boundary. Published Round 7 makes its down-residual inventory
  configuration-derived and validates exact shape, dtype, and device before
  addition; it does not claim full ControlNet numerical parity.
- The opt-in Candle UNet conditioning route now accepts pooled text plus
  explicit SDXL size/crop time IDs, loads the official
  `add_embedding.linear_{1,2}` namespace, and adds its projection to the base
  timestep embedding. Existing constructors and forward methods remain
  compatibility wrappers; a public VarBuilder route allows retained-buffer
  opt-in without copying the private config. The base and addition sinusoidal
  projections now stay F32 and cast only before their learned MLPs, matching
  the pinned reference for F32, F16, and BF16 model tensors. SnapFlash still
  owns CLIP2 pooling, time-ID policy, the faithful attention graph, retained
  revisions, and runtime admission.

## Last Green Verification

### INT-5B.1 lower-precision cast-order candidate

- `cargo fmt --all -- --check`: passed on the source candidate.
- `cargo test --offline --locked -j 2 -p candle-transformers
  stable_diffusion --lib`: passed 29/29 focused tests. The complete tiny F16
  configured UNet executes on native Windows CPU; the F16 projection matches
  the F32-before-cast reference and differs from projecting in F16; F16 and
  BF16 addition inputs reach the learned MLP boundary in model dtype.
- `cargo test --offline --locked -j 2 -p candle-transformers`: passed 88/88
  library, 5/5 generation, and 8/8 NMS tests; the unrelated Smol doc test
  remains ignored.
- `cargo check --offline --locked -j 2 -p candle-core -p candle-nn -p
  candle-transformers -p candle-vlm`: passed.
- `cargo clippy --offline --locked -j 2 -p candle-transformers --all-targets
  -- -D warnings`: passed.
- `cargo test --offline --locked -j 2 --workspace --exclude
  candle-datasets --exclude candle-pyo3`: passed across the remaining cached
  native Windows workspace. The first complete-workspace attempt failed only
  because no system Python is currently registered; the bundled Codex Python
  is 3.12 while `candle-pyo3` requires the `abi3-py313` minimum. No package or
  network action was taken to replace the missing owner environment.
- `PYO3_NO_PYTHON=1 cargo clippy --offline --locked -j 2 --workspace
  --all-targets -- -D warnings`: passed across the complete cached workspace.
- Actual BF16 matrix execution is not claimed because the Windows CPU backend
  reports that dtype unsupported for matmul. No CUDA, model, checkpoint,
  Python oracle, network, or llama.cpp workload ran.

### INT-5B SDXL `text_time` publication gate

- `cargo fmt --all -- --check`: passed on the settled tree.
- `cargo test --offline --locked -j 2 -p candle-transformers
  stable_diffusion --lib`: passed 26/26 focused tests, including the 13 UNet
  conditioning/residual tests and the public VarBuilder construction route.
- `cargo test --offline --locked -j 2 -p candle-transformers`: passed 85/85
  library, 5/5 generation, and 8/8 NMS tests; the existing unrelated Smol doc
  test remains ignored.
- `cargo check --offline --locked -j 2 -p candle-core -p candle-nn -p
  candle-transformers -p candle-vlm`: passed.
- `cargo clippy --offline --locked -j 2 -p candle-transformers --all-targets
  -- -D warnings`: passed, followed by `PYO3_NO_PYTHON=1 cargo clippy
  --offline --locked -j 2 --workspace --all-targets -- -D warnings` across the
  complete cached workspace graph.
- `PYO3_PYTHON=<installed Python 3.13> cargo test --offline --locked -j 2
  --workspace --exclude candle-datasets`: passed every remaining native
  Windows unit, integration, and doc-test lane. The first compile-complete
  attempt without an interpreter failed only while linking the unrelated
  `candle-pyo3` test target because `python3.lib` was unavailable; selecting
  the already-installed interpreter required no package or network action.
- The SnapFlash manifest passed at 10/4/6, the LFM2-VL manifest at 150/15/135,
  and the cross-overlay union at 159 paths, two overlays, and five shared
  paths. The summary bank passed 24 groups with a 126.2 KiB consolidated SDXL
  route and 130.8/256 KiB defaults. Module-layout and `git diff --check`
  passed.
- No production model, checkpoint, Python oracle, CUDA workload, llama.cpp
  process, dependency download, hosted runner, PR, or secret inspection was
  used.

### Round 7 additional-residual publication gate

- `cargo fmt --all -- --check`: passed after applying canonical formatting to
  the focused UNet change.
- `cargo test --locked --offline -j 2 -p candle-transformers
  stable_diffusion::unet_2d::tests --lib`: passed 6/6 exact residual-contract
  tests.
- `cargo test --locked --offline -j 2 -p candle-transformers`: passed 77/77
  library, 5/5 generation, and 8/8 NMS tests; the existing unrelated Smol doc
  test remains ignored.
- `cargo clippy --locked --offline -j 2 -p candle-transformers --all-targets
  -- -D warnings`: passed.
- `PYO3_NO_PYTHON=1 cargo clippy --locked --offline -j 2 --workspace
  --all-targets -- -D warnings`: passed the complete cached compile-only
  workspace lane. The first local replay selected the desktop-bundled Python
  3.12 interpreter and stopped before project compilation because the
  unrelated `candle-pyo3` crate requires the `abi3-py313` floor; PyO3's
  supported no-interpreter mode then compiled the same all-target graph with
  warnings denied. The separate workspace test lane below already passed with
  installed Python 3.13.
- `cargo check --locked --offline -j 2 -p candle-core -p candle-nn -p
  candle-transformers -p candle-vlm`: passed.
- `cargo test --locked --offline -j 2 --workspace --exclude candle-datasets`:
  passed the complete remaining native Windows workspace test and doc-test
  lane after selecting the already-installed Python 3.13 interpreter required
  by the unrelated PyO3 crate. The first sandboxed attempt was an environment
  probe and failed before compilation because Python was absent from `PATH`;
  the second sandboxed probe proved the installed interpreter was denied by
  sandbox policy. No package installation or network access occurred.
- The SnapFlash-derived manifest passed at 9/3/6, the LFM2-VL manifest passed
  at 150/15/135 with 13 text and six binary fixture policies, and the root
  overlay union passed at 158 paths, two overlays, and five shared paths.
- The summary bank passed 25 groups; the focused residual route is 33.0 KiB
  and defaults remain 133.8/256 KiB. Module-layout and preflight smoke gates
  passed, and `git diff --check` is clean through the required WSL Git lane.
- No production model, CUDA workload, Python oracle, llama.cpp process,
  checkpoint, network dependency, or hosted runner was started.

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
- SnapFlash-Server Round 6 is published. Its Windows product lane now binds
  body admission to the materialized buffer lifetime, accounts for complete
  retained queue records including bounded terminal errors, rejects derived
  structured prompts before admission, preserves the required base64-result
  wire, and evicts whole terminal records from an independent owner. Its
  separate Unix/WSL hostile-directory authority item remains in the SnapFlash
  backlog and does not broaden the proven native-Windows boundary.
- SnapFlash-Server's current ControlNet forward path still leaves text
  `_context` unused. Nine-residual structural admission is proven, but
  end-to-end real-weight numerical parity is not.
- The installed official SDXL ControlNet inventories also require two
  cross-attention down blocks, a cross-attention mid block, and `text_time`
  addition embeddings. Candle now represents the reusable base-UNet addition,
  but SnapFlash still ignores the attention families and does not yet supply
  correct pooled CLIP2/time-ID inputs. Loading must remain fail-closed until
  INT-5C represents them.
- Lower-than-Q8 vision quantization, video, true text batching, generic VLM
  traits, converters, WebGPU/WASM, broad WSL replay, public signing, and LTS are
  deferred future scope, not hidden MVP promises.
- The prior llama.cpp residency incident required a host restart. Exact cause
  remains unproven; `FAILURE_LOG.md` F-0008 containment is mandatory for every
  future model run.

## Blockers

- REL-6/7 and INT-5A/B have no remaining source, test, dependency, or
  publication blocker.
  Candle implementations `95ac9ff815fbac4f252b4ef6780b5e4a7843f328` and
  `ba1e8acc142c4683995e4cdbc8b1d933c81e96c6` plus the exact SnapFlash
  revisions above are published.
- The upstream network-backed dataset test is a disclosed owner-scoped skip,
  not a Round 3 blocker and not permission to enable network verification.
- Edge CUDA/F16 and public LFM2-VL routes remain later product holds, not
  concealed failures; Edge LoRA migration itself is complete.
- Hosted GitHub Actions state is intentionally not a blocker or verification
  dependency.

## Active Change Set

- Active Candle files are the two Stable Diffusion source files and their
  focused state/provenance documents for the INT-5B.1 publication checkpoint.
- Models, caches, downloads, generated proof logs, Cargo output, and
  `.tools/.secrets/` remain ignored or external.

## Exact Next Task

Publish the reviewed INT-5B.1 Candle cast-order checkpoint, repin SnapFlash to
that exact implementation revision, and implement INT-5C's faithful SDXL
ControlNet attention graph, CLIP2 pooled projection, and time-ID policy. Do
not generate the numerical fixture, load production weights, run CUDA, or
promote inpainting before INT-5C is green.

---
AI-edited: 2026-08-13T04:34:15-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=int-5b-cast-order | change=recorded the green lower-precision candidate and its exact publication handoff
