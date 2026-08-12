# LFM2.5-VL Status

## Baseline and Publication

- Model and compatibility baseline: Candle 0.11.0 at
  `31f35b147389700ed2a178ee66a91c3cc25cc80d`.
- Upstream integration base: Candle main at
  `6f74e7c390c717f8fd34f23ce02aceb058173370`.
- Integration and publication branch: `main` at
  `https://github.com/Shoozes/candle.git`; no pull request is used.
- Fixture-portability release parent:
  `2a01aaf0e874c4d6a57ff3276cd24be7af0656e3`. The maintenance commit is the
  next direct descendant on `main`; use live Git refs for its current local and
  remote identity rather than inferring publication from prose.
- Historical implementation checkpoint:
  `c9b60f0b906fa8fe70423295e2e1164648a8fa53` on
  `feat/lfm2-vl-mmproj`.
- Immutable first-MVP snapshot: annotated tag `lfm2-vl-mvp-0.1.0` peels to
  `ff885586f6d44a3d9b9ac1724032cdf5f0155384`. Reviewed maintenance may
  advance `main`; the snapshot tag must not be moved or reused.
- Current maintenance overlay relative to the upstream integration base: 142
  allowlisted paths, exactly 14 fork-origin modifications and 128 mod-owned
  additions.

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

- Product phase: post-Phase 7 production stabilization; the defined first MVP
  remains feature-complete and production-proven, not LTS.
- Maintenance posture: hash-pinned deterministic fixture bytes are stable
  across native Windows Git checkouts without weakening byte-exact loader or
  manifest verification. M-1 is complete.
- The repository-wide source audit found no production placeholder, unfinished
  LFM2-VL branch, broken public export, or new async/run-condition defect that
  warrants a source change. Test-only panics remain test assertions; the
  documented legacy infallible config adapter remains paired with its checked
  API.
- `TODO.md` has no active implementation, parity, safety, performance,
  environment, or maintenance release task.

## Last Green Verification

### Current fixture-portability worktree

- A disposable native Windows Git 2.54 checkout at the reviewed parent with
  `core.autocrlf=true` reproduced the defect exactly: canonical
  `processor_config.json` changed from 524 LF bytes and SHA-256
  `97b79ebfc8eae3a5bcbeb8f1494c1decdbade5d20d3204739143d17b460906f2`
  to 553 CRLF bytes and SHA-256
  `09150e818ebe443d2df9009b78c46ef5aaa4aed17ebc4b20cf55eefb8f01e53f`.
- A fresh native Windows clone containing the maintenance fix and the same
  `core.autocrlf=true` setting retained all 10 fixture JSON/Markdown files as
  LF with zero carriage returns. All three safetensors files resolve
  `text=unset`. The split bundle retained exact SHA-256 values
  `b932d4e6c58224d6d97182b0aa969c701beafb0130e2f6031bba189cf9d04f39`,
  `97b79ebfc8eae3a5bcbeb8f1494c1decdbade5d20d3204739143d17b460906f2`,
  and `b6aef395937e6ce1dbc1fe110438b19db82e87c9351edc61fca7b27a72a287d3`.
- The two exact split-MMProj identity/hybrid tests passed. The full native
  Windows locked/offline workspace test gate passed, including transformer
  59/59 and VLM 29/29. Strict workspace Clippy passed with `-D warnings`.
- The final release audit repeated both exact regressions, exercised an
  ephemeral unknown-extension probe and observed the required controlled
  verifier failure, removed the probe, and passed the real inventory again.
  The complete `cargo test --locked --offline -j 2 --workspace` gate and strict
  workspace Clippy then exited zero. The first workspace attempt was blocked
  only by managed-sandbox denial of the installed Python interpreter; the
  identical approved-boundary rerun passed without a download or file change.
- The broad gate selected the installed Python 3.13 interpreter explicitly for
  the unrelated `candle-pyo3` ABI3 crate; no dependency installation or network
  access was used. The prior statement that Python 3.13 was absent is stale.
- The mod-manifest verifier now discovers every committed fixture text/binary
  payload and enforces LF/no-CR or `-text` attributes. It passes at
  142/14/128 with 10 text and three binary fixtures.
- Current-root formatting, locked/offline checks for the four affected crates
  and all three LFM2 examples, PowerShell 7 and Windows PowerShell 5.1 context
  verification, all nine module-layout wrappers, 23 Markdown documents with 50
  relative links, 11 JSON files, and 16 Python files pass. No production
  TODO/FIXME/todo!/unimplemented! marker was found. Current
  `summary_bank.json` SHA-256 is
  `d91bf698d4a9561081197a619dc3bcde6d34268a1d20752e65aa7bcfb4906502`.
- No production checkpoint, Python oracle, llama.cpp process, CUDA inference,
  or concurrent large-model run was started.

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
- The isolated all-CUDA F32 baseline remains 30 measurements with median
  458.0633 ms and relative MAD 2.1454%. No candidate met the 10% retention
  threshold, so no speculative optimization is retained.
- Detailed production commands, thresholds, and evidence identities live in
  `PARITY.md` and `HISTORY.md`; they are not duplicated here.

## Proven Behavior

- Config-driven LFM2.5 text, embedding prefill, cached decode, and reset.
- SigLIP2 NaFlex, pixel unshuffle/projector, composite native model, checked
  raw-image processing, prompt expansion, and multi-image feature insertion.
- Native safetensors, quantized GGUF text plus split dense MMProj, direct GGUF
  MMProj, CPU-F32 native Q8_0 execution, strict inventory/provenance checks,
  and controlled malformed-input errors.
- Deterministic native/hybrid evidence, official 450M and 1.6B CPU parity,
  complete advertised 450M placement/dtype parity, and official GGUF
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

- No implementation, fixture, test, memory, dependency, environment,
  publication-policy, or remote-history blocker remains for the defined MVP or
  this maintenance slice.
- Owner authorization for direct-main commit and guarded publication was
  provided. Live local/remote ref equality remains the only authority for
  whether a particular checkout has received the maintenance commit.
- Hosted GitHub Actions state is intentionally not a blocker or verification
  dependency.

## Active Files

- No source or documentation file remains under active implementation. The
  completed fixture-portability release changed `.gitattributes`,
  `scripts/lfm2-vl/verify-mod-manifest.sh`, the split fixture README,
  `summary_bank.json`, and the seven current/decision/history/provenance docs.
- Models, caches, downloads, generated proof clones/logs, Cargo output, and
  `.tools/.secrets/` remain ignored or external.

## Exact Next Task

If live `origin/main` differs from the clean local `main`, stop and determine
whether this checkout is merely behind or has unreviewed divergence; do not
merge or overwrite automatically. Otherwise no release task remains. Any
deferred feature, new backend, LTS effort, or further maintenance must begin
with a new focused acceptance contract. Do not move
`lfm2-vl-mvp-0.1.0`; a new maintenance tag requires a separate explicit
request.

---
AI-edited: 2026-08-12T11:16:23-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=fixture-portability-release | change=closed the audited fixture-portability maintenance gate
