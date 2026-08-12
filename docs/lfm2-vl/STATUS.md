# LFM2.5-VL Status

## Baseline and Publication

- Model and compatibility baseline: Candle 0.11.0 at
  `31f35b147389700ed2a178ee66a91c3cc25cc80d`.
- Upstream integration base: Candle main at
  `6f74e7c390c717f8fd34f23ce02aceb058173370`.
- Integration and publication branch: `main` at
  `https://github.com/Shoozes/candle.git`; no pull request is used.
- Historical implementation checkpoint:
  `c9b60f0b906fa8fe70423295e2e1164648a8fa53` on
  `feat/lfm2-vl-mmproj`.
- Current reviewed parent before this release slice:
  `95c067f7cc9a702575b5b7c0f400ca3aa3ff1386`. The final release commit and
  annotated tag own the post-slice identity.
- Current overlay relative to the upstream integration base: 141 allowlisted
  paths, exactly 14 fork-origin modifications and 127 mod-owned additions.

## Worktree Boundary

- Native Windows/MSVC is the product and primary proof lane; WSL2/Linux is a
  secondary portability replay.
- `C:\DevStuff\candle-mods` is a WSL-owned linked worktree attached to local
  `main`. Use `NVIDIA-Workbench` WSL Git for status, staging, commits, tags,
  and revision checks; do not attach `main` to another worktree.
- Owner-reviewed work lands directly on `main`. Broad staging, force-push,
  implicit merge/rebase, hosted-CI evidence, PR creation, and secret inspection
  remain prohibited.

## Current Phase

- Product phase: post-Phase 7 production stabilization.
- Release posture: feature-complete MVP release candidate, not LTS.
- NR-5B official 450M CPU-F32 component parity, official-base GGUF
  same-artifact output, official 1.6B CPU-F32 component parity, public device
  placement, tiny distinct-device CUDA proof, official 450M CUDA parity,
  synchronized diagnostics, the complete resolved device/dtype matrix, and
  the isolated generation baseline are green.
- CPU components are F32-only. BF16/F16 on any resolved CPU component fail
  before model loading; all-CUDA F32/BF16/F16 and both mixed F32 placements
  are production-proven on the official 450M checkpoint.
- `TODO.md` has no active MVP implementation, parity, safety, or performance
  task. Future scope must be promoted through a new acceptance contract.

## Last Green Verification

### Production evidence

- Admitted model: `LiquidAI/LFM2.5-VL-450M` at revision
  `fc6221ca597f3315e4f82fc2df606783267b34ba`; eight regular files totaling
  902,236,184 bytes; artifact-manifest SHA-256
  `659c8421530586b6cc28c094cfcdc69719ea8626f2abc0efd9eec4ac2a68a984`.
- Source-matching CUDA release executable: 65,076,736 bytes, SHA-256
  `7a9261f6808b09ffab0963f5c015661c515534c6b949ac4893f4fa8cbe0023a2`.
- Six sequential bounded routes passed: CPU/CPU F32, CUDA/CUDA
  F32/BF16/F16, CPU-text/CUDA-vision F32, and CUDA-text/CPU-vision F32.
  Every route consumed the exact artifact/image/prompt, expanded 64 image
  tokens, generated `[1098, 4646, 5251]`, preserved the CPU prefill top-five
  ID order, reset cache exactly, exited 0, and left its PID absent. Peak Job
  memory was 2,412,109,824–3,474,620,416 bytes.
- The isolated all-CUDA F32 generation benchmark passed 10 warm-ups and 30
  measurements with median 458.0633 ms, MAD 9.82745 ms, relative MAD 2.1454%,
  exact IDs, 3,475,435,520 peak Job bytes, and exact cleanup. No candidate met
  the 10% retention threshold, so no speculative optimization remains.
- Official 1.6B native Windows CPU-F32 parity remains green at 51/51 tensors
  with exact reset and comparison SHA-256
  `9a0b16256a222678f9dce1282660e49fc6d19103cc6dd6a53c824bb58a6412c0`.
  No model-math, processor, tokenizer, loading, or tensor-layout path changed
  after that proof, so the expensive Python/native traces were not repeated.

### Current source tree

- Clean admission record:
  `C:\DevStuff\candle-oracle\evidence\release-closeout-preflight-20260812T025925Z.json`,
  SHA-256
  `697e623d9a836e94be69d22999ae6ec22eb36ba6c2255faef862daf3ed26d910`;
  `status=review`, zero model/build/Python processes, 42,920,747,008 available
  physical bytes, 46,284,562,432 bytes commit headroom, and 23,865 MiB GPU
  memory free.
- `cargo fmt --all -- --check`: passed.
- Locked/offline two-job checks: `candle-core`, `candle-nn`,
  `candle-transformers`, `candle-vlm`, and the `lfm2`, `quantized-lfm2`, and
  `lfm2-vl` examples passed.
- `cargo test --locked --offline -j 2 -p candle-examples --example lfm2-vl`:
  36/36 passed. The CUDA-gated resolved-device matrix regression passed 1/1.
- Complete affected default-feature tests passed: core library/integration/doc
  lanes, transformer 59/59 plus generation 5/5 and NMS 8/8, and VLM 29/29.
  CUDA cast and dense non-contiguous-linear regressions each passed 1/1.
- Exact workspace strict Clippy passed with `PYO3_NO_PYTHON=1`:
  `cargo clippy --locked --offline -j 2 --workspace --tests --examples
  --benches -- -D warnings`. The environment flag is required because the
  unrelated Python binding enables ABI3 Python 3.13 while no 3.13 interpreter
  is installed; no package installation or network access was used.
- Native module-layout verification passed all nine wrappers. PowerShell 7 and
  Windows PowerShell 5.1 summary-bank verification passed with every route
  below 256 KiB; `summary_bank.json` SHA-256 is
  `b76d04195a30899e4c4c10699c4b8a3d2f98d73e835cdf7aa3fce0dcc98b7c15`.
- Preflight smoke passed under both PowerShell versions. Relative links pass
  across all 23 mod-owned Markdown files. WSL mod-manifest verification passed
  at 141/14/127, and `git diff --check` is clean.
- The pinned Python reference suite remains 82/82 green. It was not rerun
  because this slice did not change the oracle, processor, tokenizer, loader,
  comparator, or trace schema.

## Proven Behavior

- Config-driven LFM2.5 text, embedding prefill, cached decode, and reset.
- SigLIP2 NaFlex, pixel unshuffle/projector, composite native model, checked
  raw-image processing, prompt expansion, and multi-image feature insertion.
- Native safetensors, quantized GGUF text plus split dense MMProj, direct GGUF
  MMProj, CPU-F32 native Q8_0 execution, strict inventory/provenance checks,
  and controlled malformed-input errors.
- Deterministic native/hybrid evidence, official 450M and 1.6B CPU parity,
  complete advertised 450M native placement/dtype parity, and official GGUF
  same-artifact decoded-output agreement with pinned llama.cpp.
- Kill-on-close Windows Job containment, timeout/memory enforcement, exact PID
  cleanup, quiet-host admission, and no-clobber evidence publication.

## Known Gaps and Conflicts

- Lower-than-Q8 vision quantization, video, true text batching, generic VLM
  traits, converters, WebGPU/WASM, broad WSL replay, public signing, and LTS are
  deferred future scope, not hidden MVP promises.
- Official config context is 128,000 while model cards advertise 32,768;
  construction follows config and production policy remains unresolved.
- Official MMProj headers omit tiling metadata; pinned processor configuration
  or documented architecture defaults remain required.
- The prior llama.cpp residency incident required a host restart. Exact cause
  remains unproven; F-0008 containment is mandatory for every future run.
- Gknome adoption remains fail-closed on mature-repository authority conflicts
  and is outside the LFM2-VL product backlog.

## Blockers

- No implementation, parity, safety, test, or performance blocker remains for
  the defined MVP.
- Remote hygiene is not complete: `agent/lfm2-vl-backlog-closeout` still
  exists. Its remote head `52342156dcc20d8351a96ac8901293e972681bbb` has tree
  SHA-1 `cf30d53a81248fba4a5f0ab30fca7fec7d0aacc0`, exactly matching integrated
  main commit `6ea6aef5`; deletion therefore risks no unique file state, but
  the managed approval layer requires a fresh exact authorization naming that
  branch. Retain `feat/lfm2-vl-mmproj`.
- The complete local diff and verification gate are reviewed. Remote main/tag
  publication is the remaining release action. The main-only
  `.tools/gitpush.ps1` must not be weakened to delete branches or publish
  unrelated refs.
- Hosted GitHub Actions state is intentionally not a blocker or verification
  dependency.

## Active Files

- The working release slice contains resolved-device dtype validation/tests,
  synchronized diagnostics, the isolated benchmark and regression coverage,
  removal of the temporary workflow, release discoverability/support docs,
  quiet-host preflight hardening, verifier portability fixes, context routes,
  and this consolidated handoff.
- Models, downloads, caches, logs, owner evidence, generated traces, and
  `.tools/.secrets/` remain external or ignored.

## Exact Next Task

Publish the reviewed fast-forward release commit on `main` through
`.tools/gitpush.ps1`, create and publish annotated tag `lfm2-vl-mvp-0.1.0`,
and verify both remote refs resolve to the same commit. Delete only the
already-proven temporary remote branch after fresh exact owner authorization.
Then stop; do not start another product phase implicitly.

---
AI-edited: 2026-08-11T23:22:00-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=release-closeout | change=consolidated current state around the final local MVP gate
