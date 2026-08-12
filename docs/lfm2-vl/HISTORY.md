# LFM2.5-VL Completed Work History

This file preserves completed implementation and verification evidence. Any present-tense phase, blocker, worktree, or next-task statement below its dated section is historical. Use `STATUS.md` for current truth and `TODO.md` for active work.

## 2026-08-11 — P3.3 official 1.6B bounded Python component trace

- Rehashed the exact pinned regular-file snapshot immediately before tracing;
  the eight-file total remained 3,198,084,631 bytes and the artifact-manifest
  SHA-256 remained
  `b8d582c40214a1a8df82f21ece21fb683a5e5377c7c03b4fba0e97feb865e585`.
  Reused `trace-gradient-256.png` (SHA-256
  `f902f8d2e47e53eafac86831cfc692001dc15870eb81d57abc3128f048d2efca`), the
  prompt `Describe this image.`, CPU F32, seed 0, and three cached decode
  steps.
- Ran the pinned Python executable (SHA-256
  `b2c836c52cdf063180b9ee76f67ac42946101b79ac457f3494035a67c090d961`)
  through `run-bounded-oracle.ps1` with a 24 GiB Job Object and 7,200-second
  timeout. PID 28560 exited 0 in 28,505 ms; peak Job memory was
  14,482,644,992 bytes and exact PID cleanup was recorded in
  `C:\DevStuff\candle-oracle\evidence\p3-1.6b-python-trace-owner-20260811T210231Z.json`.
  The 762-byte combined log SHA-256 is
  `a85229de763b4ac459100d03fdbd6165a5fa99a2247eb52ec6ff1bc8c6ba973c`.
- External bundle
  `C:\DevStuff\candle-oracle\evidence\python-trace-1.6b-20260811T210231Z`
  validates with 51 tensors, 182,528,392 safetensors bytes, payload SHA-256
  `184d62de07a1b72c8e6a0190b05ef15ff7361c2a029fe5fc2c04a0e17ebbb2f2`, 80
  input tokens, 64 projected image tokens, exact cache reset (`max_abs=0`),
  unchanged artifact manifest, and `weights_serialized=false`. Pinned
  reference tests passed 81/81.
- A recreating Codex-owned build task was resolved by identity-checking and
  stopping its PowerShell owner shell; the final clean postflight at
  `C:\DevStuff\candle-oracle\evidence\p3-1.6b-python-trace-postflight-clean-20260811T210659Z.json`
  recorded zero model/build families, 43.5 GiB available physical memory,
  47.4 GiB commit headroom, and 23,438 MiB GPU free. P3.4 native trace is the
  next active gate.

## 2026-08-11 — P3.2 official 1.6B bounded load-only admission

- Rehashed the exact pinned regular-file snapshot immediately before the native
  run. The eight-file total remained 3,198,084,631 bytes and the artifact
  manifest SHA-256 remained
  `b8d582c40214a1a8df82f21ece21fb683a5e5377c7c03b4fba0e97feb865e585`.
- A fresh census found a Codex-owned Cargo/Tauri tree. PID 24676 was verified
  by name, stable MSVC executable path, creation time, parent chain, and
  descendants before `taskkill /PID 24676 /T /F` stopped only that exact tree.
  Codex, ChatGPT, PowerShell, unrelated helpers, and model processes were not
  targeted; every captured build PID was absent after three seconds.
- The recorded release executable was
  `C:\DevStuff\candle-mods\target\release\examples\lfm2-vl.exe`, 10,230,272
  bytes, SHA-256
  `338ebcbf02dbac13fabf6ce9115bdb3a91fc3316a84a9c23e1ad304fbd900d9a`.
  Direct CPU load-only execution through `run-bounded-oracle.ps1` used a
  12 GiB Job Object, 7,200-second timeout, executable-scoped concurrency, and
  suspended assignment. PID 15792 exited 0 in 2,264 ms; peak Job memory was
  6,433,579,008 bytes and the PID was absent after cleanup.
- The loader reported 589 tensors, one shard, CPU F32 vision/text, expected
  vision/projector/language roots, tied output, and tokenizer image token 396.
  No inference or trace payload was generated. External owner evidence is
  `C:\DevStuff\candle-oracle\evidence\p3-1.6b-native-load-owner-20260811T204250Z.json`;
  its 668-byte combined log has SHA-256
  `8c8395c2da88d76848fc66830a50c42bfee02b88e291bb27592808ae8acaee3e`.
- Postflight found no Cargo/rustc/rustup/cargo-tauri/llama/LFM2-VL process,
  50,953,560,064 bytes available physical memory, 55,951,667,200 bytes commit
  headroom, and 23,430 MiB GPU free. P3.2 is complete; P3.3 Python component
  tracing is the next active gate.

## 2026-08-11 — P3.1 official 1.6B snapshot acquisition and admission

- After explicit owner approval, replayed the no-network acquisition plan for
  `LiquidAI/LFM2.5-VL-1.6B@919fde3d022e3f90a4716006f993938ee8c2eb97`.
  It re-established schema 2, eight files, 3,198,084,631 total bytes, the
  public/no-token and Xet-disabled serial-transfer policy, 213,124,534,272
  bytes free, and absent snapshot, cache, manifest, and staging paths.
- Waited for a sustained zero-Cargo/rustc/llama window without terminating
  another task, then ran the exact pinned Python through the Windows Job Object
  owner with a 2,147,483,648-byte ceiling, 7,200-second timeout, and
  executable-scoped concurrency. PID 22940 exited 0 in 129,190 ms, peaked at
  75,395,072 Job bytes, and was absent after cleanup.
- Atomically published the external regular-file snapshot at
  `C:\DevStuff\candle-oracle\lfm2-vl-1.6b-919fde3d` plus its acquisition
  manifest. An independent full-file pass rehashed all eight direct regular
  files and proved the 3,193,334,216-byte `model.safetensors` SHA-256
  `7fc7458e4382fc6e558cfdda45857fbf9ab5b40a8bf199c9cd073003b14ac26d`.
  No partial snapshot, manifest stage, or incomplete cache file remained.
- Retained the 4,818-byte acquisition manifest at SHA-256
  `a080891c8d1099d58a01377af258ef04898f808eed0fcf4fbe718d4698f4b732`,
  the 4,958-byte combined log at SHA-256
  `0d4357d9c532ba943ec8ad5c495c733734652f7cd38cc4ca5d0de101ae16b1f3`,
  and the 1,724-byte owner record at SHA-256
  `631fb14581ef89f53c983ae2c77ff444f889d16a42bc8b5c3dede52c760a9380`.
  Evidence records `network_policy=permitted-cache-aware`,
  `network_used=null`, atomic publication, and `model_loaded=false`.
- Postflight retained 46,482,870,272 available physical bytes,
  48,465,375,232 bytes commit headroom, 199,416,389,632 bytes disk free, and
  23,523 MiB GPU memory free. No llama or exact acquisition interpreter
  remained. P3.2 load-only inspection was not started.

## 2026-08-11 — Direct-main integration and integrity release closeout

- Attached the formerly detached Windows-linked edit worktree to local `main`
  through WSL Git. The historical `feat/lfm2-vl-mmproj` branch remains owned by
  `/home/workbench/code/candle-lfm2-vl` at checkpoint
  `c9b60f0b906fa8fe70423295e2e1164648a8fa53`.
- Fetched GitHub main at `6f74e7c390c717f8fd34f23ce02aceb058173370`.
  Its nine post-0.11 upstream commits changed 29 paths and did not overlap the
  mod's nine fork-origin files. Checkpoint
  `a83acf13d2b6bff6528e8b8c87209500f6fbc85c` captured the reviewed integrity
  slice; merge checkpoint `2b1d9e80de06b251b2fe5f25e51c17d56db86591`
  then preserved both histories without conflict or force.
- Made `main` the single owner-reviewed publication line with no PR. Added an
  ignored WSL-aware `.tools/gitpush.ps1` that refuses dirty, detached, behind,
  diverged, wrong-remote, tracked-token, or non-main state and never stages,
  commits, merges, rebases, creates a repository, or force-pushes.
- Rebased `MOD_MANIFEST.md` and its verifier on the exact integrated upstream
  main commit. The overlay remains 91 paths: 9 fork-origin modifications and 82
  mod-owned additions; inherited upstream changes are no longer misclassified
  as mod files.
- The first PowerShell 7 bounded-owner replay exposed an owner-exit test race:
  PID visibility preceded Job Object assignment, so the test itself could kill
  the wrapper too early and strand its suspended 2 MiB fixture child. The exact
  PID was verified and terminated, the smoke gained a child-written
  post-resume handshake plus exact failure cleanup, and PowerShell 7.6.4 and
  Windows PowerShell 5.1 both passed the corrected bounded-owner and preflight
  suites. Failed-test temp directories were removed after PID absence.
- Fixed both actionable strict-Clippy `manual_contains` findings in mod-owned
  VLM validation. Strict Clippy then passed the affected libraries and
  LFM2-VL example with only compatibility-sensitive
  `manual-is-multiple-of` and indexing-clarity `needless-range-loop` allowed.
- Post-merge native Windows verification passed formatting; locked/offline
  checks for core, NN, transformers, VLM, and all three LFM2 examples; all core
  integration/doc lanes; transformer 58/58, generation 5/5, NMS 8/8, VLM
  29/29; and LFM2-VL example 29/29. The exact Windows oracle environment and
  81-test Python reference suite also passed in this release task.
- No production model, llama.cpp process, CUDA workload, package installation,
  hosted CI, PR, broad stage, force-push, or secret-value inspection occurred.
  The separately approved 1.6B acquisition and subsequent CPU proof remain the
  product next task.

## 2026-08-11 — Guarded 1.6B snapshot acquisition made source-complete

- Added a fail-closed acquisition owner for the eight exact files at
  `LiquidAI/LFM2.5-VL-1.6B@919fde3d022e3f90a4716006f993938ee8c2eb97`.
  It admits 3,198,084,631 bytes only after a 12 GiB disk check, uses serial
  public/no-token Hub requests, disables the installed Xet parallel-chunk
  backend before Hub import, verifies pinned Git-blob or LFS identities while
  streaming, and atomically publishes a clean external regular-file snapshot
  plus manifest without replacing a path that appeared after planning.
- Kept resumable provider data in a separate caller-owned cache and rejected
  repository-local, secret-tree, root, linked/reparse, nested, or pre-existing
  publication paths. Failed identity or manifest writes leave no published
  snapshot.
- Removed public downloader and artifact-builder callbacks after diff review
  showed they could bypass the enforced transfer/verifier path while emitting
  production-shaped evidence. Tests now replace private module boundaries.
- Versioned acquisition evidence to schema 2 so network permission is distinct
  from observation: plans prove disabled/false, while cache-aware execution
  reports permitted/unknown instead of claiming a request occurred.
- Added fail-closed stale-stage detection, a PID-bearing staging name,
  stale manifest-stage detection, post-cache path revalidation,
  returned-source containment inside the named cache, resumable-cache
  preservation on caught transfer failure, and an actionable diagnostic when
  manifest publication and snapshot rollback both fail.
- Replaced check-then-`os.replace` publication with exclusive primitives: the
  snapshot uses Windows no-replace rename or Linux
  `renameat2(RENAME_NOREPLACE)`, while the flushed manifest uses an atomic
  no-clobber hard link. Race-created owner paths are preserved, duplicate
  verifier records fail before publication, and rollback uses the same
  no-clobber directory primitive.
- Sanitized downloader failure propagation to filename plus exception class and
  suppressed the provider cause so programmatic tracebacks cannot retain signed
  transfer URLs.
- Defined the separately approved real transfer as a direct invocation of the
  existing Job Object owner with a 2 GiB ceiling, 7,200-second timeout,
  executable-scoped concurrency, retained external log/evidence, and resumable
  cache on termination; any unmatched snapshot-stage, manifest-stage, output,
  or manifest state is non-admissible and blocks retry for inspection.
- Split package responsibility at the real runtime boundary: acquisition
  planning is stdlib-only, the real downloader checks only
  `huggingface-hub==1.5.0`, and the full Torch/Transformers oracle lock remains
  mandatory only for model loading, trace generation, and tensor comparison.
- The native Windows read-only plan passed with all three destinations absent,
  eight exact files, no network, no model load, the explicit Xet-disabled HTTP
  transfer policy, schema-2 disabled/false network evidence, and
  243,618,676,736 bytes free.
  The focused offline acquisition suite passed 27/27; the complete pinned
  reference suite passed 75/75 in 18.79 seconds with the exact 42-distribution
  lock and compileall green. A native Windows manifest race regression and a
  WSL Linux no-clobber smoke passed. No model payload was downloaded or loaded.

## 2026-08-11 — Native Windows oracle pin made installable

- Found a real setup contradiction before package installation: the local
  per-user interpreter is Python 3.10.10, while the single 3.10.12 oracle pin
  has no official Windows binary distribution.
- Split only the interpreter pin by supported platform: official Python
  3.10.11 for native Windows and the already-proven Python 3.10.12 resolved
  lock for Linux. All Torch, TorchVision, safetensors, Transformers VCS, Hub,
  tokenizer, regex, Pillow, and pytest pins remain unchanged.
- Added `verify_environment.py`, which reports exact installed and expected
  versions plus runtime/test mismatches without importing Torch or loading a
  model. The existing 3.10.10 interpreter was directly probed and correctly
  reports every production dependency missing.
- Downloaded the official Python 3.10.11 x64 installer from Python.org,
  verified its valid Python Software Foundation Authenticode signature and
  SHA-256 `d8dede5005564b408ba50317108b765ed9c3c510342a598f9fd42681cbe0648b`,
  and installed it side by side without PATH changes or replacing 3.10.10.
- Created the ignored project `.venv`, installed only the exact CPU oracle/test
  pins, and verified every version plus the Transformers commit. The complete
  reference suite passed 40/40 in 23.17 seconds with tiny fixtures only.
- Promoted the exact `pip freeze --all` result to
  `requirements-reference-windows.txt`. No model weight was downloaded or
  production checkpoint loaded; the bounded P1-C dry load remains next.

## 2026-08-11 — Shared snapshot identity enforced at comparison

- Closed a production-proof gap where the Python oracle could be pinned to an
  external snapshot while the comparator still accepted missing native model
  identity fields.
- Production tracing now requires and hashes the external regular-file
  `--model-dir` before model import, loads model and processor from that path
  with local-only resolution, refuses download mode, and records the artifact
  manifest in the trace.
- The comparator now requires Candle's consumed config, processor, tokenizer,
  safetensors index, and weight evidence to match the oracle manifest by direct
  filename, byte count, and SHA-256 before tensor files are opened. Ambiguous,
  missing, unexpected, duplicate, or content-mismatched inputs fail closed.
- Both lanes now hash their model inputs again after inference and refuse
  evidence output if the snapshot changed during the trace.
- Corrected the cross-lane prompt contract: the Python oracle records original
  user text separately and promotes the official rendered chat-template text
  to the compared `prompt`; native replay must consume that exact value. This
  removes a would-be token-ID mismatch between templated Python input and raw
  Candle input before the first production run.
- Added the native all-ones attention-mask tensor and made the complete
  `stage.*` inventory an exact comparator contract. Optional configured stages
  can no longer vanish merely because the comparator previously intersected
  bundle inventories.
- Verification used bundled-Python compile/import and dependency-free matching,
  mismatch, and revalidation regressions; native formatting/checks; the
  LFM2-VL example suite at 28/28; and the complete core/transformer/VLM suite.
  No production weights, model inference, download, or package installation
  occurred; the official 450M numerical gate remains open.

## 2026-08-11 — Artifact identity ambiguity hardening

- Hardened `tools/lfm2_vl/reference/inspect_artifact.py` so malformed pinned
  file lists, duplicate paths, NUL-containing names, invalid indexed tensor
  names, and shard path traversal fail before any checkpoint bytes are hashed.
- Verification: bundled Python import/compile checks and dependency-free
  artifact identity regressions passed. No production checkpoint was read,
  installed, or downloaded.

## 2026-08-11 Reference bundle path-identity hardening

- Closed a validator gap in `tools/lfm2_vl/reference/tensor_dump.py`: manifest
  filenames are now required to be direct regular files in the bundle root.
  Absolute paths, traversal/nested names, directories, and symlinks are
  rejected before metadata or tensor bytes are hashed or opened.
- Malformed non-object manifest and metadata JSON now returns an actionable
  validation error instead of falling through to an attribute failure.
- Added stdlib-only `tools/lfm2_vl/reference/inspect_artifact.py` for the next
  P1-B gate. It requires explicit production opt-in, hashes only the pinned
  local regular files, supports a bounded indexed-shard path, and writes a
  small external manifest without serializing payloads.
- Added a dependency-free regression for manifest path escape and routed the
  owning manifest/tensor/test files through `workflow__reference_fixtures`.
- Verification: bundled Python 3.12.13 imports and AST parsing passed; the
  path-escape regression passed directly; no Torch, model, network, or
  production trace was used. The exact pinned pytest lane remains owner-managed
  and unavailable on this host. A disposable eight-file snapshot produced a
  valid 450M artifact manifest and atomic external write.

## 2026-08-11 Cross-version summary-bank verifier repair

- Fixed `scripts/lfm2-vl/verify-summary-bank.ps1` so its no-argument entry
  point does not evaluate `$PSScriptRoot` during parameter binding and its JSON
  parser works on both PowerShell 5.1 and 7 without the newer `-AsHashtable`
  parameter.
- Verification: the verifier passed under native PowerShell 7.6.4 and Windows
  PowerShell 5.1, with all route budgets below the configured 256 KiB ceiling.

## 2026-08-11 Cross-version disk census repair

- Fixed `scripts/lfm2-vl/preflight.ps1` where Windows PowerShell 5.1 exposed
  zeroed `Get-PSDrive` counters. It now falls back to
  `System.IO.DriveInfo`, records the selected source, and keeps the existing
  read-only/admission contract.
- Extended `test-preflight.ps1` to require a recognized disk source and a
  positive free-space value. Native PowerShell 7.6.4 and 5.1 smoke runs pass;
  no model or inference process was launched.

## 2026-08-10 Gknome inventory-boundary integration replay

- Extended the separate Gknome adoption source to prune generic `artifacts/`
  and `.pytest_cache/` trees before hashing, added its Candle-shaped regression,
  and refreshed its derived template manifest after the reviewed template-test
  edit made the previous metadata stale.
- The fixed source passed 49 adoption assertions, 53-file/77-link documentation
  integrity, 150 current-state assertions, and a 22-group context audit. The
  refreshed Candle dry run `20260811T032224Z-4a87c2b8` returned blocked with zero
  applied files, `existing_repository=true`, exactly four project-authority
  conflicts, 13,064 inventoried files, 12,563 included files, and zero included
  `artifacts/` or `.pytest_cache/` paths. No Candle files were applied.

## 2026-08-11 Gknome state reconciliation

- Reconciled the current Gknome source identity to clean commit `c93be64` and
  corrected the Candle entry point/backlog to distinguish the inventory-clean
  dry run from its four remaining authority conflicts.
- A fresh adoption test attempt was not promoted to proof: the managed task
  can read the external OneDrive checkout but cannot write its ignored
  `.artifacts/proof/adoption.json`, so the test exited before assertions. The
  prior 49/53/150/22 proof remains the last recorded result; no source or
  proof file was modified.

## 2026-08-11 Reference-environment contract clarification

- Documented the purpose of every direct Python oracle pin and separated
  runtime, compatibility, transitive, and test-only responsibilities. The
  README now states plainly that these packages are not Candle Rust runtime
  dependencies and that config-only/header/bundle checks remain available
  without the heavy oracle stack.
- Corrected the requirements comment that implied a Python 3.10–3.14 range;
  the production guard and resolved lock require Python 3.10.12 exactly.
- Verification: bundled Python `py_compile` passed; stdlib-only 450M config
  inspection passed; the production guard refused the unpinned environment at
  exit 2 before model import; summary-bank verification passed. The focused
  pytest suite could not run because the bundled Python has no pytest, and the
  mod-manifest verifier stopped at exit 2 because this Windows checkout cannot
  resolve its WSL-owned baseline Git metadata; neither gap was masked.

## 2026-08-11 Bounded memory-probe fallback

- Hardened `run-bounded-oracle.ps1` so the host-memory ceiling no longer
  depends exclusively on `Get-CimInstance Win32_ComputerSystem`. Restricted
  Windows hosts now use `GlobalMemoryStatusEx`, record the selected source in
  evidence, and still fail closed if both probes fail.
- Extended the harmless wrapper smoke assertion to require a recognized
  physical-memory source. A bounded copied-`cmd.exe` probe exercised the
  fallback path with `GlobalMemoryStatusEx`, recorded 68,438,708,224 bytes,
  exited normally, and proved the child PID absent after cleanup.
- Verification at `2026-08-11T00:29:38-04:00`: PowerShell parsing, the full
  bounded smoke, Cargo formatting, the LFM2-VL example (27/27), and the
  locked/offline core/transformer/VLM suite (transformer 56, generation 5,
  NMS 8, VLM 29, plus core integration/doc tests) passed with bounded build
  concurrency. No model process was launched.
- A PS5.1 replay initially exposed pre-existing C# `nameof`, modern SHA-256
  helpers, `ProcessStartInfo.ArgumentList`, and `Process.Kill(bool)` usage.
  These compatibility paths are now framework-safe; PS5.1 and PS7 smoke runs
  both pass.

## 2026-08-11 Native resource/PID admission preflight

- Added `scripts/lfm2-vl/preflight.ps1` as a read-only Windows admission census
  separate from the bounded child launcher. It records Git/worktree identity,
  CIM or `GlobalMemoryStatusEx` physical memory, commit counters, repository
  drive space, optional NVIDIA state, and matching llama/Python/build-tool
  processes with parent identity and memory fields. Command lines and secrets
  are deliberately omitted; an unusable linked-worktree Git result is retained
  as data rather than treated as a runtime failure.
- Added `test-preflight.ps1` to prove JSON schema, redaction fields, atomic
  output, overwrite refusal, and explicit replacement under both PowerShell
  5.1 and PowerShell 7. The current census found no `llama*` process and
  returned `review`; no model or large worker was launched.
- Hardened native stderr capture for the linked-worktree Git diagnostic and
  widened commit-counter conversion to `UInt64` after a real >2 GiB overflow
  exposed the PowerShell `Math.Max` overload trap. The current restricted host
  uses `GlobalMemoryStatusEx` for physical RAM and records commit counters;
  `Win32_Process` parent details remain explicitly unavailable when CIM access
  is denied.
- Native Windows verification from `2026-08-11T00:57:29.9555922-04:00` to
  `00:57:50.5885410-04:00` passed the affected core/transformer/VLM test suite
  and the LFM2-VL example (27/27), with no model process resident.
- The release `lfm2-vl.exe` build passed from `2026-08-11T01:06:18.7519929-04:00`
  to `01:11:36.6534217-04:00`; the 10,224,128-byte executable hashes to
  `fb7a56cfbc4d3cf7dc4b23634cf8f6ee1b0e4a902ec3709d2eb1957815919ff7`. A
  bounded no-model `--help` identity run then exited 0 with Job Object
  assignment-before-resume and exact PID absence. No production input or model
  was loaded.
- Extended the preflight smoke contract to require stable JSON arrays for
  tracked/llama processes, probe errors, and GPU inventories, preventing a
  one-item PowerShell serialization shape change from breaking report
  consumers.
- Tightened admission semantics so an unavailable committed-memory counter
  now returns `blocked` rather than `review`; owner approval cannot override a
  missing safety measurement.
- Changed the census to derive the complete `llama_processes` inventory before
  truncating the general process list to 64 records, so a low-memory model
  server cannot be hidden by compiler/Python fan-out.
- This closes the reusable P1-A implementation slice but not owner approval,
  the pinned Python environment, or official 450M numerical parity.

## 2026-08-10 Bounded Native Production-Trace Surface

- Added explicit native `--trace-output <external-dir>` support to the LFM2-VL example. The lane requires `--cpu`, explicit/implicit F32, one image, and one non-tiled crop; it keeps ordinary generation unchanged and performs a separate cache replay so decode tensors align with the Python oracle's first cached input and fixed step count.
- Added public trace-returning model APIs for vision/projector stages, merged text embeddings, prefill hidden/logit tensors, and cached decode tensors. Ordinary `encode_images`, `prefill`, and `decode` behavior remains unchanged at the contract boundary.
- Added an external-only safetensors bundle writer with input/image tensors, stage inventories, SHA-256 metadata, atomic staging/publish, and an explicit no-weights contract. The tiny native fixture test writes and reloads the bundle successfully.
- Added `tools/lfm2_vl/reference/compare_traces.py`, which validates the trace schema/mode, matching metadata/manifest contract, no-weights claim, both bundle hashes/inventories, and required tensors one pair at a time with exact integer inputs and recorded CPU-F32 tolerances. No production trace or numerical comparison was claimed because the pinned owner-managed Torch/Transformers environment remains unavailable on native Windows.
- Added comparator regression coverage to the reference harness and extended the local-build ignore contract with `/target-native/` after the publication gate caught Cargo runtime markers. The manifest now passes at 84 total paths (9 fork-origin modifications, 75 mod-owned additions).
- Added a no-heavy-import pin guard for production loading: Python/package versions and the Transformers VCS commit must match `requirements-reference.txt` before a production model/config import; the available Windows Torch/Transformers environment is rejected as unpinned.
- The live production-metadata guard probe returned exit 2 with the exact mismatch list and left its external output path absent, proving the refusal occurs before model/config loading.
- Tightened the NR-5B runbook to require the existing Job Object wrapper around each inference process, with the native executable built before wrapping and `cargo run` excluded from the bounded model step to keep compiler memory out of the OOM ceiling.

Verification for this slice: native LFM2-VL example tests 27/27, native example strict Clippy, `cargo fmt --all`, summary-bank validation, Python compile/CLI guards, the tiny trace writer/reload test, and the dependency-free reference harness 10 passed/4 skipped under the pin guard. The pinned production runtime remains owner-managed and unavailable for an official trace; no model process was launched.

## 2026-08-10 Repository Integrity Slice

- Replaced the feature-off LFM2 flash-attention `unimplemented!()` panic with an actionable Candle error and added a feature-off regression.
- Replaced the 57 KB current-status document and 40 KB completed bootstrap entry point with compact live documents; retained all detailed evidence in this history file and `history/BOOTSTRAP_AND_PHASE_GUIDE.md`.
- Added an active-only backlog with What/Why/When/Where/How/Done-when/Verification contracts, and updated the parity/failure documents to route current versus historical evidence without duplication.
- Added the missing `lfm2-vl` example README from the actual implemented CLI contract, including native/split/GGUF modes, deterministic JSON, dtype/device rules, Q8 restrictions, input bounds, and memory safety.
- Added `summary_bank.json` plus a strict PowerShell verifier. The default seven-file orientation route is 89.4 KiB; all 12 focused groups are below the 256 KiB ceiling; excluded secret/model/build/runtime paths, missing files, wildcards, duplicate members, archived defaults, excess fan-out, and path escapes fail closed.
- Added an offline Bash mod-manifest verifier and wired it into the baseline gate. It proves the current 81-path baseline delta is exactly 9 fork-origin modifications plus 72 mod-owned additions and rejects local secret/model/build/runtime paths.
- Recorded the WSL-owned `.git` file/detached-worktree topology as F-0009. Exact WSL Git evidence proves the Windows edit tree and feature branch share checkpoint `c9b60f0b906fa8fe70423295e2e1164648a8fa53`; Git mutation remains owner-run in an intentional WSL branch/worktree.
- Production-surface search found no remaining TODO/FIXME/unimplemented path. The remaining LFM2 `into_config` panic is the documented legacy infallible compatibility API; the other panic is test-only. Large-file seams were ranked in `TODO.md` and deliberately deferred until the production parity gate supplies a change-driven boundary.

Verification for the uncommitted integrity slice:

- `cargo test --locked --offline -p candle-transformers lfm2 -- --nocapture`: 33/33 passed, including the new controlled feature-off error.
- `cargo test --locked --offline -p candle-examples --example lfm2-vl`: 26/26 passed.
- `cargo test --locked --offline -p candle-core -p candle-transformers -p candle-vlm`: passed all library, integration, and doc-test lanes; transformer 54/54, generation 5/5, NMS 8/8, VLM 29/29, and core suites green.
- Strict scoped Clippy passed for core/transformer/VLM libraries and the LFM2-VL example with `-D warnings` plus the five documented Rust 1.97 compatibility allowances.
- `scripts/lfm2-vl/verify-baseline.sh` passed again on the consolidated final tree from `2026-08-10T20:32:51Z` to `2026-08-10T20:33:07Z`, including formatting, all required locked/offline checks, both diff gates, and the new exact manifest gate; ignored verifier-only Cargo.lock SHA-256 `7292957b78b688fe2d8d0f61ba5987b92638d6138a0faa9a13db014d09b06a26`.
- `test-bounded-oracle.ps1` first stopped at sandbox-denied CIM access and retained its safe temp evidence; the exact elevated rerun passed all harmless process-tree cases, and the failed-run temp directory was path-checked and removed. No model was loaded.
- PowerShell parsing, Bash parsing, Python compileall, all scoped JSON parsing, both dependency-free checkpoint config inspections, Markdown relative-link validation, summary-bank validation, and `lfm2-vl --help` passed.
- The prior pinned Python pytest environment is no longer present, and system Python has no `pytest`; the 23-test reference suite was truthfully skipped without installing or downloading dependencies. No Python source changed in this slice.

## Archived status snapshot

# LFM2.5-VL Status

## Baseline

- Upstream: Hugging Face Candle
- Base version: 0.11.0
- Baseline commit: `31f35b147389700ed2a178ee66a91c3cc25cc80d`
- Working branch: `feat/lfm2-vl-mmproj`
- Bootstrap checkpoint: `4a6b30a124abb32b4b275ea8c343ce7ef3ac8be7`
- Source-lock checkpoint: `f007daec10e5751e5899676a7c58098183ec1256`
- Reference-harness checkpoint: `a9594101c97589f6deabe7a2dddaaffeb5471a94`
- Phase 1 checkpoint: `f660b8e3f2b4560f133356864e012be83f29d9c0`
- Phase 2 checkpoint: `74e109aec5f9801cfead3eeb27fe3f93ac646b84`
- Phase 3 checkpoint: `37264b49cf74d0cf7697317eda0183f084db6ff8`
- Phase 4 checkpoint: `8d1bbe471404848730685c98e7dd56b13a457eb4`
- Phase 5 checkpoint: `1535a0a5fef09f243811b83553b9c75baad78ee2`
- Phase 6 checkpoint: `dc6321cde3a9b8a019c1f0fd82780d90afa046df`
- Phase 7 checkpoint: `1a9cf291ba9a1bbac1de83029f9d8f057aca00b5`
- Tags: `lfm2-vl-baseline-candle-0.11.0`, `lfm2-vl-phase-0-bootstrap`, `lfm2-vl-phase-0-reference`, `lfm2-vl-phase-1-text`, `lfm2-vl-phase-2-siglip2`, `lfm2-vl-phase-3-native-composite`, `lfm2-vl-phase-4-native-e2e`, `lfm2-vl-phase-5-hybrid`, `lfm2-vl-phase-6-gguf`, `lfm2-vl-phase-7-q8`

## Current Phase

- Phase: Post-Phase 7 Stabilization — Bounded Oracle Complete; Official 450M CPU-F32 Next
- Task: Publish the locally verified NR-4/NR-5A checkpoint for review, then execute official 450M native CPU-F32 component parity
- Scope: deterministic native/split/direct inference evidence, immutable llama.cpp bundle identity, suspended Job Object containment, and local-only verification; no new model execution, hosted verification, GitHub Actions, PR, or secret inspection in NR-5A
- Status: The runner implementation, exact file-level native/split/direct provenance, GGUF EOS metadata, 26/26 focused example tests, 32/32 focused transformer/LFM2 tests, full core/transformer/VLM tests, strict scoped Clippy, full baseline, exact mod/fork classification, local same-artifact decoded output, and the harmless bounded-oracle process-tree smoke suite are green. The exact pinned llama.cpp b10335 owner build passes a bounded no-model identity probe. Official-base payload parity remains unclaimed. The designated independent NR-4 worker could not start because its Codex agent loop died; the manager audit closed the direct-provenance coverage gap, but no independent NR-4 verdict is claimed.

## Source-Lock Results

- Transformers: `fd12552d770f745fdbe41031ff4daa688f5ed57e`
- LiquidAI 450M: `fc6221ca597f3315e4f82fc2df606783267b34ba`
- LiquidAI 1.6B: `919fde3d022e3f90a4716006f993938ee8c2eb97`
- mistral.rs: `8010b6a0578e416120b590ed72fd46ed5f24ee85`
- llama.cpp: `74ce15741b420b8d6f12e720398458b576c51c2c`
- MLX-VLM: `ffd7aeff0bd213c31534a969e0003d49451eef39`
- Transformers.js: `353007be131c2e44d16d46ba49b9a56f2955dfd8`
- Official safetensors metadata: 349 tensors for 450M and 589 for 1.6B; header-only Range reads; zero tensor payload bytes
- Official 450M MMProj GGUF: `166cd80bbe157dc86d65f964eb8cc6a2cede62ca`; F16 and Q8_0 headers each 12,736 bytes, 32 metadata records, 201 tensors, zero retained tensor payload bytes
- Production weight payloads or complete GGUF files downloaded by this project: none. NR-4 downloaded only the pinned official 450M `tokenizer.json`; the complete GGUF files under `C:\llamacpp` pre-existed this worktree.

## Source-Lock Verification

- Date: 2026-08-09 23:11 EDT (`2026-08-10T03:11:06Z` to `2026-08-10T03:11:09Z`)
- Environment: fresh detached Linux-home worktree `/home/workbench/code/candle-lfm2-vl-source-lock-verify`; WSL2 `NVIDIA-Workbench`; CPU-only lane
- Verification HEAD: `4a6b30a124abb32b4b275ea8c343ce7ef3ac8be7` with exactly the six source-lock paths staged
- Command: from `/tmp`, with the existing Linux build cache selected through `CARGO_TARGET_DIR`, `bash /home/workbench/code/candle-lfm2-vl-source-lock-verify/scripts/lfm2-vl/verify-baseline.sh`
- Results: passed `cargo fmt --all -- --check`; locked/offline checks for `candle-core`, `candle-nn`, `candle-transformers`, `lfm2`, and `quantized-lfm2`; staged and unstaged diff checks
- JSON: PowerShell semantic validation and Linux `python3 -m json.tool` passed
- Local-only lockfile SHA-256: `4e059ffe6035520ca6553303932173eba562f4985f82931a90129eea9849ce54`
- Retained log: `artifacts/verification/source-lock/baseline-final.log`; SHA-256 `563a6c97ebf416a7f85ba77296cdc73464b79a73d09238f9dc521d197d94eb6a`

## Reference Harness Verification

- Date: 2026-08-09 23:53–23:58 EDT
- Environment: detached Linux-home worktree `/home/workbench/code/candle-lfm2-vl-reference-verify`; WSL2 `NVIDIA-Workbench`; Python 3.10.12 virtual environment; CPU-only
- Direct pins: Torch `2.8.0+cpu`, safetensors `0.8.0`, Transformers `5.15.0.dev0` from `fd12552d770f745fdbe41031ff4daa688f5ed57e`, huggingface-hub `1.5.0`, tokenizers `0.22.2`, regex `2025.10.22`, Pillow `11.3.0`, pytest `8.4.1`
- Dependency lock: `requirements-reference.txt` contains the complete resolved environment and matches `python -m pip freeze --all` exactly after comments and index directives are removed
- Python tests: `.venv/bin/python -m pytest -q tools/lfm2_vl/reference/test_reference_tools.py` passed, 9 tests in 12.23 seconds
- Config-only proof: both CLIs passed under bare `/usr/bin/python3` with user packages disabled; the inspector also normalized the real pinned 450M `config.json` and `processor_config.json` without downloading weights
- Fixture proof: two independent seed-1234 exports were byte-identical; 87 tensors; raw source-image SHA-256 `08359b108fa567f5dcf319fa3434da6abbc1d595f426372666447f09cc5a87dc`
- Fixture files: `tensors.safetensors` 61,072 bytes, SHA-256 `d4ccbd62ebd8afdecb6207fe341a0880e74c5dda8deea680a08366ece4ec96c3`; `metadata.json` 2,422 bytes, SHA-256 `3add6fa29206fe2b404f3e0959d3c51d69046eacab3f1ebb19dac43964cb0199`; `manifest.json` 8,485 bytes, SHA-256 `c5461dadb0edfd920b20f308650c59676977110a1cc2f199e317dea7d75bdd7b`
- Fixture coverage: official class construction, padded packed patches, exact resized positions, both vision encoder layers, post-LN, factor-2 pixel unshuffle, optional projector LayerNorm, both projector linears, image-placeholder replacement, multimodal prefill logits, and three cached decode steps
- Rust baseline: locked/offline CPU verification passed from `2026-08-10T03:53:09Z` to `2026-08-10T03:53:12Z`, including formatting, core crates, dense LFM2, quantized LFM2, and staged/unstaged diff checks
- Retained logs: `artifacts/verification/reference-harness/python-final.log`, SHA-256 `fb46403e3315b49dcfb424b07fc062f0b2366f45ec627aa1bdd64b006bcf2c93`; `artifacts/verification/reference-harness/baseline-final.log`, SHA-256 `b08863fe6c0e6a33d86812e8d95d601e51654bcea65d4f51f6aefadb59e1d3d8`

## Text Compatibility Verification

- Date: 2026-08-10 00:31–00:32 EDT
- Environment: detached Linux-home worktree `/home/workbench/code/candle-lfm2-vl-text-verify`; WSL2 `NVIDIA-Workbench`; CPU-only, locked, offline lane; staged Phase 1 candidate based on `a9594101c97589f6deabe7a2dddaaffeb5471a94`
- Implementation scope: config aliases and normalization, checked `try_into_config` plus the preserved `into_config` API, standalone/nested dense roots, tied/explicit output heads, dense hidden/logit APIs, quantized embedding-driven forwarding, and cache clearing
- Focused command: `cargo test --locked --offline -p candle-transformers lfm2 -- --nocapture` passed all 5 focused tests plus filtered integration binaries
- Focused proof: official 450M/1.6B effective FFN widths, legacy aliases/fallback/precedence, standalone and nested roots, explicit and tied heads, dense token-ID versus embedding forwarding, committed-fixture merged prefill parity, three cached decode steps, cache-reset determinism, and quantized embedding-driven equivalence
- Maximum absolute errors: token embeddings `0`; token-ID versus embedding-driven output `0`; prefill hidden states `2.38418579e-7`; prefill logits `2.98023224e-8`; cached decode steps `1.86264515e-8`, `2.98023224e-8`, and `1.49011612e-8`; reset prefill hidden states `2.38418579e-7`; reset decode `1.86264515e-8`
- Focused log: `artifacts/verification/text-compatibility/focused-tests.log`; SHA-256 `022e7e7fed20f0b255424d04334108c419b9aee49e722132a6563dff4b67d034`
- Broader library command: `cargo test --locked --offline -p candle-transformers --lib` passed all 18 tests; retained log `artifacts/verification/text-compatibility/candle-transformers-lib.log`; SHA-256 `cc8faae1c192569ba3952e14e9bbcbc2a5ad3a24132e0cf20424eb12b0606db3`
- Full baseline: `scripts/lfm2-vl/verify-baseline.sh` passed from `2026-08-10T04:32:23Z` to `2026-08-10T04:32:51Z`, including formatting, `candle-core`, `candle-nn`, `candle-transformers`, the `lfm2` and `quantized-lfm2` examples, and staged/unstaged diff checks
- Baseline log: `artifacts/verification/text-compatibility/baseline-final.log`; SHA-256 `c72eccd8b77689878689f7e720c46a040c26f3cee8060b17727392f392862f46`
- Local-only lockfile SHA-256: `4e059ffe6035520ca6553303932173eba562f4985f82931a90129eea9849ce54`; no production weights or GGUF files were downloaded

## SigLIP2 NaFlex Verification

- Environment: manager Linux-home WSL2 `NVIDIA-Workbench`; CPU-only F32 lane
- Focused command: `cargo test --locked --offline -p candle-transformers siglip2 -- --nocapture`
- Result: 7/7 focused tests passed
- Broader library command: `cargo test --locked --offline -p candle-transformers --lib`; 25 passed, 0 failed
- Broader library log: `artifacts/verification/siglip2/candle-transformers-lib.log`; SHA-256 `b4c91d4bd6a0c1850a66d9cc27d61776b5ec96c152783e6c13f23a0cfcdf5197`
- Focused log: `artifacts/verification/siglip2/focused-tests.log`; SHA-256 `d09ec6bdf1d110711f347b89bba353eeb8ebb6172e8a45d19d3edd1aeb254645`
- Full baseline: passed from `2026-08-10T05:14:56Z` to `2026-08-10T05:15:08Z`, including `cargo fmt`, `candle-core`, `candle-nn`, `candle-transformers`, `lfm2`, `quantized-lfm2`, and both diff checks
- Full baseline log: `artifacts/verification/siglip2/baseline-final.log`; SHA-256 `727e0d8a029f121a7225d3d35a53addd480791323b6e0c501576408cc6460d52`
- Phase 2 checkpoint/tag: complete at commit `74e109aec5f9801cfead3eeb27fe3f93ac646b84`, annotated tag `lfm2-vl-phase-2-siglip2`
- Gate: positional max absolute error `<=2e-5`; vision cosine similarity `>=0.99999`
- Stage evidence: patch projection max abs `5.960464478e-8`, cosine `0.999999940`; resized positions `2.980232239e-8`, cosine `0.999999940`; embedding sum `1.192092896e-7`, cosine `1.000000119`; encoder layer 0 `4.768371582e-7`, cosine `0.999999881`; encoder layer 1 `1.192092896e-6`, cosine `0.999999881`
- Final evidence: returned post-LN `7.152557373e-7`, cosine `1.000000119`; post-LN hook matched the same result; padding-key isolation max abs `0`, cosine `1`
- Proven behavior: packed patch projection with bias, CPU F32 separable antialiased positional interpolation and per-shape cache, bidirectional key masking, F32 score/softmax with original-dtype value matmul, configured encoder activation, post-LN, checked malformed-input handling, and controlled exclusion of the vision pooling head
- Production weights or GGUF files downloaded: none

## Phase 3 Verification

- Phase 2 checkpoint: complete at commit `74e109aec5f9801cfead3eeb27fe3f93ac646b84`, annotated tag `lfm2-vl-phase-2-siglip2`
- Implementation proven in scope: dynamic top-level config, factor-N official pixel-unshuffle, optional projector LayerNorm, linear/GELU/linear projection, crop unpadding/ranges/order, strict one-span-per-image exact-length merge, multimodal prefill, ordinary cached decode, cache reset, and `EncodedImages`
- Focused Phase 3 gate: 11/11 passed; retained log `artifacts/verification/native-composite/focused-tests.log`; SHA-256 `7d727e1b8558f1f242ce940c8af36d44a3e292f4ffa023d1ff124ccf2cc13638`
- Maximum absolute errors: projector stages `<=5.960464478e-8`; encoded and merged embeddings `<=6.519258022e-9`; prefill logits `<=4.470348358e-8`; cached decode `<=2.980232239e-8`
- SigLIP2 repeated-crop regression: 8/8 passed; retained log `artifacts/verification/native-composite/siglip2-regression.log`; SHA-256 `5684568b060c6338f3e5d8bc94361d37bc64ddf84584ad4a5e05915acc275f38`
- Runtime defect resolved: batched SigLIP2 attention received a non-contiguous transposed left-hand operand and failed with `MatMulUnexpectedStriding`; `split_heads` now materializes a contiguous tensor, protected by the repeated-crop regression
- `candle-transformers` library gate: 37/37 passed; retained log `artifacts/verification/native-composite/candle-transformers-lib.log`; SHA-256 `0f36d6a8d54f77abfe9c5031075b7174cff83859315d0997f60a1a399f475497`
- Full locked/offline CPU baseline: passed `2026-08-10T05:48:07Z`–`2026-08-10T05:48:10Z` against pre-Phase-3-checkpoint HEAD `74e109aec5f9801cfead3eeb27fe3f93ac646b84`; retained log `artifacts/verification/native-composite/baseline-final.log`; SHA-256 `47d984dd3afe7b92b6a72bcdb93e7d9da99bd8673e5c1067b8f1fac7ed2b8b45`
- Cargo.lock SHA-256: `4e059ffe6035520ca6553303932173eba562f4985f82931a90129eea9849ce54`
- Phase 3 checkpoint/tag: complete at commit `37264b49cf74d0cf7697317eda0183f084db6ff8`, annotated tag `lfm2-vl-phase-3-native-composite`
- Not claimed: production-checkpoint parity, CUDA, GGUF, raw-image preprocessing, tokenizer/chat template, or CLI support

## Phase 4 Verification

- Phase 3 checkpoint/tag: complete at commit `37264b49cf74d0cf7697317eda0183f084db6ff8`, annotated tag `lfm2-vl-phase-3-native-composite`
- Implementation proven in scope: parsed processor JSON, `explicit > processor JSON > GGUF metadata > model config > defaults` precedence, fixed padding including explicit `max_num_patches`, RGB/grayscale/RGBA conversion, checked smart resize and tile selection, row-major crops, optional thumbnail, TorchVision-compatible byte resize, normalization, patchification, masks/shapes, image/crop metadata, tokenizer-resolved special tokens, sentinel preservation, exact per-crop spans, multiple images, context checks, and controlled invalid-input errors
- Rust processor/prompt gate: 24/24 passed; retained log `artifacts/verification/processor-prompt/candle-vlm-tests.log`; SHA-256 `ddf2b1b311ccd849a9491342bb1c5a82b5c528fac3bde826e6cebd6ffcc66702`
- Processor fixture: all 12 required cases passed; integer masks, spatial shapes, image grids/sizes, crop ranges/order/kinds, and prompt IDs/spans were exact; worst pixel-value maximum absolute error `1.192092896e-7`, cosine similarity `1.0`
- Direct resize regression: the complete 7×5 RGB to 8×4 byte output matches pinned TorchVision in all 96 channel values; boundary cases and unrepresentable allocation requests are also covered
- Real-dimension gate: all 10 pinned cases assert smart dimensions, large-image decision, selected grid, tile canvas, and whole/tile/thumbnail order
- Fixture reproduction: a fresh pinned-oracle export is byte-identical to the checked-in fixture. Manifest SHA-256 `2fb787e378f5fd1ddfa147913aadccd07add9a1045b8bb0f693ca2c2f564959c`; metadata `aca7f4d5e5e4ef0e4872adeb227b56cf3960d87b353c40162af97660783f2327`; tensors `a25932fc57f3e78f48a1a8f558216521c7ae3e8659fcf0a389cd0a4ebe0ab3f6`
- Fixture logs: regeneration SHA-256 `fbb7a0bbb247d29f8bd8725d8b48cd971ed1a3768a2dacb75300b19f3696b955`; checked-in/regenerated hash comparison `84aa570f6da7995977e424ae8d61491ec07cdfbdad0af7847d77604d74a492eb`
- Pinned Python reference tests: 9/9 passed in 11.83 seconds; retained log SHA-256 `00f0fac4bd730862e95945ef357d402ee88473b61fce0fff6b6614930eb615ee`
- `candle-transformers` regression: 37/37 passed, including exact encoded image/crop range-union validation; retained log SHA-256 `78fa7ede014c72ca5d7defd6833c8152be2e3b0455d635ad79b122788452b560`
- Independent worker re-audit: five actionable findings were resolved and the initially reported resize mismatch was withdrawn after exact reproduction showed it used no matching source/coordinate convention
- Full locked/offline CPU baseline: passed `2026-08-10T08:35:14Z`–`2026-08-10T08:35:25Z` against pre-Phase-4-checkpoint HEAD `37264b49cf74d0cf7697317eda0183f084db6ff8` with exactly the Phase 4 candidate staged; retained log `artifacts/verification/processor-prompt/baseline-final.log`; SHA-256 `fb16481302c9bdf15f4b04df0250e45bf8c9a2126b92b09a787f5360cc3a3140`
- Verifier-only Cargo.lock SHA-256 after adding `candle-vlm`: `a4f4379f73d38db1a148f96538e6b868d1ef148a8417069807bae451c9766fb4`
- Phase 4 checkpoint/tag: complete at commit `8d1bbe471404848730685c98e7dd56b13a457eb4`, annotated tag `lfm2-vl-phase-4-native-e2e`
- Not claimed: production-checkpoint numerical parity, GGUF/mmproj loading, CUDA, generated-caption parity, or CLI support

## Phase 5 Verification

- Phase 4 checkpoint/tag: complete at commit `8d1bbe471404848730685c98e7dd56b13a457eb4`, annotated tag `lfm2-vl-phase-4-native-e2e`
- Split artifact contract: deterministic `mmproj.safetensors`, versioned `mmproj.json`, and canonical `processor_config.json`; exact config-derived inventory; immutable source model/revision; SHA-256 coverage; atomic per-file writes; overwrite refusal; no production weights
- Loader contract: manifest, processor, architecture, text width/layer count, vision layer inventory, patch/factor, tokenizer image ID, tensor names/shapes/dtypes/byte counts, header size, tensor count, offsets, overlaps, gaps, and payload coverage are checked before tensor construction
- File identity and allocation: the weights file is opened and buffered once; the same bytes are hashed, inspected, and consumed by `VarBuilder`; the maximum allocation is derived with checked arithmetic from the validated manifest payload plus the bounded header and prefix
- Quantized path: deterministic real GGUF bytes are written from the committed tiny text tensors, hash-pinned as `8fbd510aeea4715547c57975a7adcb91c148a8bc5e8d869d9617b69af6a006b1`, parsed through `gguf_file::Content::read`, and loaded through `ModelWeights::from_gguf`; block-aligned matrices use Q8_0 and small unalignable tensors remain F32
- Numerical evidence: split versus unified image features max abs `0`; quantized hybrid prefill logits max abs `4.457309842e-5`; cached decode steps `2.650916576e-5`, `2.175569534e-5`, and `1.309439540e-5`; cache reset max abs `0`
- Python reference/exporter suite: 19/19 passed from `2026-08-10T10:01:59Z` to `2026-08-10T10:02:12Z`; retained log `artifacts/verification/hybrid-mmproj/python-final.log`; SHA-256 `96e17862334d3fc7877596cafb3daa743c58904dcca11a8b7e205138d40dab85`
- `candle-transformers`: 42/42 library tests plus 5 generation and 8 NMS integration tests passed from `2026-08-10T10:02:23Z` to `2026-08-10T10:02:26Z`; retained log `artifacts/verification/hybrid-mmproj/candle-transformers-tests.log`; SHA-256 `aba483c8bc2c01b4fd6ecc6c511dfc93831280933596b0ffad9660a3bd22b529`
- `candle-vlm` and example: 25/25 tests and `cargo check -p candle-examples --example lfm2-vl` passed from `2026-08-10T10:02:39Z` to `2026-08-10T10:02:48Z`; retained log `artifacts/verification/hybrid-mmproj/vlm-example-final.log`; SHA-256 `b431fa113f047ec0981f93e9a3ce581dc3ac9638b0185f4b57fa81aab0780eb7`
- Clippy: transformer, VLM, and example gates passed with only the recorded pre-existing newer-Clippy allowances; retained log `artifacts/verification/hybrid-mmproj/clippy-final.log`; SHA-256 `d2e79ade9bab2e9115a4ce526c1ed16f72622f3230f98939132c5e82124752d6`
- Distinct-device lane: source-complete CUDA-vision/CPU-text test verifies that raw packed inputs remain caller-owned, projected features stay on the vision device until merge, logits return on the text device, and prefill agrees within `1e-4`; local execution was skipped at the `cudarc` build because `nvcc` is absent despite an RTX 4090 driver at compute capability 8.9; retained log `artifacts/verification/hybrid-mmproj/cuda-distinct-device-skipped.log`; SHA-256 `6c16f84a925c9bb2d856b18919cc9fae45a69b38bcf7d4db65b44396f011eb97`
- Independent worker re-audit: all nine implementation findings are resolved; the worker confirmed no code blocker remains and classified the CUDA result as an environment-owned evidence gap
- Full locked/offline staged CPU baseline: passed `2026-08-10T10:07:11Z`–`2026-08-10T10:07:19Z` against pre-Phase-5-checkpoint HEAD `8d1bbe471404848730685c98e7dd56b13a457eb4` with exactly the Phase 5 candidate staged; retained log `artifacts/verification/hybrid-mmproj/baseline-final.log`; SHA-256 `594932158f4a99702cedd47ced1fe0ffd8e4aa18835346e65a0f9003963bc369`
- Verifier-only Cargo.lock SHA-256: `acd9419056b786da820b5120db8e78be06902721689c39b55f29445abdddaffc`
- Phase 5 checkpoint/tag: complete at commit `1535a0a5fef09f243811b83553b9c75baad78ee2`, annotated tag `lfm2-vl-phase-5-hybrid`
- Not claimed at Phase 5: production-checkpoint numerical parity, production GGUF numerical parity, direct GGUF mmproj compatibility, executed CUDA parity, generated-caption parity, or quantized vision execution

## Phase 6 Verification

- Phase 5 checkpoint/tag: complete at commit `1535a0a5fef09f243811b83553b9c75baad78ee2`, annotated tag `lfm2-vl-phase-5-hybrid`
- Official header evidence: `LiquidAI/LFM2.5-VL-450M-GGUF@166cd80bbe157dc86d65f964eb8cc6a2cede62ca`; exact range `bytes=0-12735`; header end 12,708; aligned tensor-data offset 12,736; 32 metadata records; 201 tensors; tensor-name SHA-256 `45e3f6cf0b51dc9f5e458b8af3375d368cc59daff70b79e2938c7490a94df828`; zero retained payload bytes
- Official dtype evidence: F16 file 75 F16 plus 126 F32 tensors, prefix SHA-256 `338099d49dd803963c9496cfbba56ab46a425ca7895c5edf59010337ae4436ac`; Q8_0 file 74 Q8_0 plus 127 F32 tensors, prefix SHA-256 `7a4f0f1e168d52b70a03f2773f0f20b9f65d1692f8e973aa0cf9ecee25e43d1c`
- Loader contract: one stable handle; required `clip/mmproj/lfm2` metadata; exact config-derived 201-tensor production inventory; paired optional input LayerNorm/projector biases; supported target dtypes F32/F16/BF16; no Phase 7 quantized operator
- Security boundary: caller-specific GGUF limits apply before allocation (16,384 tensor, metadata, and array records; 1 MiB strings; 16 MiB aligned header), followed by checked 8 GiB file, retained-dense, and conservative peak-allocation bounds plus alignment/offset/overlap/truncation validation
- Orientation: official headers prove every non-patch matrix is already in Candle `[out,in]`; only patch `[V,3,P,P]` is converted with `permute(0,2,3,1)`, contiguous, and reshape to `[V,3P²]`
- Processor/prompt boundary: absent GGUF tiling keys retain pinned official architecture defaults (2–10 tiles, thumbnail, 512 tile size, 64–256 image tokens, effective 1,024 packed patches); direct GGUF uses image markers and resolves image token ID through the tokenizer
- Deterministic dense MMProj GGUF SHA-256: `7361b57e6d9dbf2d7809d4f446944fdc7325b368e4444fee2bc3497376695256`
- Numerical evidence: dense direct/native image features max abs `0`; Q8_0-dequantized/dense image features `8.463021368e-5`; direct prefill `4.457309842e-5`; cached decode `2.650916576e-5`, `2.175569534e-5`, and `1.309439540e-5`; cache reset `0`
- Full Python reference suite: 23/23 passed `2026-08-10T11:46:49Z`–`2026-08-10T11:47:03Z`; retained log `artifacts/verification/gguf-mmproj/python-final.log`; SHA-256 `508b999476bf3dac595479f29d41f3831698c23395ab68f02f17c43c537ae998`
- Full Rust gate: `cargo test --locked --offline -p candle-core -p candle-transformers -p candle-vlm` passed `2026-08-10T11:43:47Z`–`2026-08-10T11:44:06Z`; this includes 21/21 core library tests, all core integration/doc tests, 47/47 transformer library tests, 5/5 generation, 8/8 NMS, and 26/26 VLM tests; retained log `artifacts/verification/gguf-mmproj/rust-tests-final.log`; SHA-256 `e8aa97169506331362d5a0446de9d5d8837f78e9963fca4636f87fd9a277b9ec`
- Strict scoped Clippy: core/transformer/VLM libraries and the `lfm2-vl` example passed with `-D warnings` plus the five recorded pre-existing Rust 1.97 allowances (`useless_borrows_in_formatting`, `manual_filter`, `manual_is_multiple_of`, `needless_range_loop`, `manual_contains`); library log SHA-256 `4c994e3ab472ac4b39abac4750304ca1c74da31581d65c3cdf8530d114f5adc6`; example log SHA-256 `96ff6e913ed4421d0e5111f4311c605409b5b978250ef0cb2eb21ac2dbc50c4d`
- Full locked/offline staged CPU baseline: passed `2026-08-10T11:51:12Z`–`2026-08-10T11:51:24Z` against pre-Phase-6-checkpoint HEAD `1535a0a5fef09f243811b83553b9c75baad78ee2` with exactly the 19 Phase 6 paths staged and no unstaged delta; retained log `artifacts/verification/gguf-mmproj/baseline-final.log`; SHA-256 `1f4c755e4271da48ea7906f4804181e75a5a0b4b61e5e0db37cdc1fd95bdedd3`
- Audit: the initial assigned-worker audit found no P0; processor markers/tokenizer ID, pre-allocation header bounds, transient memory accounting, official lock coverage, range negatives, required `general.type`, and CLI ID compatibility were addressed. The final bounded re-audit found no remaining P0/P1 defect; exact dtype-distribution assertions were added from its only actionable test-polish note.
- Phase 6 checkpoint/tag: complete at commit `dc6321cde3a9b8a019c1f0fd82780d90afa046df`, annotated tag `lfm2-vl-phase-6-gguf`
- Not claimed: production-checkpoint numerical parity, production MMProj payload execution, llama.cpp runtime numerical parity, executed CUDA parity, generated-caption parity, or native quantized vision execution

## Phase 7 Verification

- Phase 6 checkpoint/tag: complete at commit `dc6321cde3a9b8a019c1f0fd82780d90afa046df`, annotated tag `lfm2-vl-phase-6-gguf`
- Operator boundary: `LinearOp` keeps the existing dense `candle_nn::Linear` path and stores native Q8_0 weights directly as `QMatMul::QTensor`; the constructor deliberately bypasses the environment-sensitive eager-dequantization helper; a unit test pattern-matches the retained Q8_0 storage
- Tensor roles: native Q8_0 is accepted only for vision Q/K/V/out, vision MLP up/down, and projector linear 1/2; patch projection, positional embeddings, LayerNorms, and all biases remain dense; dense eligible matrices remain supported for mixed-width checkpoints
- Loader boundary: `from_gguf`/`load_gguf` remain the explicit Phase 6 dense compatibility APIs; `from_gguf_q8`/`load_gguf_q8` require at least one valid Q8_0 linear; `*_auto` selects native Q8 on F32 Q8 artifacts and propagates invalid Q8 role/alignment errors rather than silently dequantizing
- Dtype/device boundary: native Q8 activation dtype is currently F32; automatic F16/BF16 loading uses the dense compatibility path and the CLI prints the selected execution mode plus native-Q8 tensor count; native CUDA execution remains unverified and unclaimed
- Comprehensive two-layer fixture: all 14 eligible attention/MLP/projector matrices are Q8_0; GGUF SHA-256 `241f59dc92c033c9877654261cf538dc107087eab5834920bd4b0e52cbdcc056`; native versus dequantized-Q8 operator max abs `3.734588623e-3`; native versus dense max abs `5.300968885e-3`; cosine `0.999923348`
- Committed hybrid fixture: Q8_0 MMProj GGUF SHA-256 `225241e57bc84c62d097aab6daa9466a75e920dbb858daf4cba4cc18ef8bb3f0`; native-Q8 image-feature max abs `1.533385366e-4`; prefill max abs `1.650899649e-4`; cached decode `7.853843272e-5`, `6.113573909e-5`, and `4.052370787e-5`; cache reset `0`
- Negative coverage: dense-only strict-Q8 request, BF16 strict-Q8 activation, Q4_0, Q8_0 dense roles, Q8_0 patch role, non-block-aligned Q8_0 input width, malformed metadata/inventory/ranges/payload, and cross-artifact pairing mismatches return controlled errors
- Full Python reference suite: 23/23 passed at 2026-08-10 08:40 EDT; retained log `artifacts/verification/q8-mmproj/python-final.log`; SHA-256 `798ecb8a3cdc1cd63b635e249b230b9d536aa55e1b5fddcab632366e3d7cfd24`
- Full Rust gate: `cargo test --locked --offline -p candle-core -p candle-transformers -p candle-vlm` passed 2026-08-10 08:40 EDT; this includes 21/21 core library tests, all core integration/doc tests, 49/49 transformer library tests, 5/5 generation, 8/8 NMS, and 26/26 VLM tests; retained log `artifacts/verification/q8-mmproj/rust-tests-final.log`; SHA-256 `1638e2f86728770a4a46da4fc39d4b5cca9dcd2af165408189a18d7de1e2f68e`
- Strict scoped Clippy: core/transformer/VLM libraries and the `lfm2-vl` example passed with `-D warnings` plus the five recorded pre-existing Rust 1.97 allowances; library log SHA-256 `efc623c3b62b1114ffbfe3dd08ae83ef03a9d7baf17a24887580fc73451bd910`; example log SHA-256 `c24c4549a12dabcabd8f67e75b4349881b84bd1aa8778a9ed1fbdfb19f598238`
- Full locked/offline staged CPU baseline: passed `2026-08-10T12:49:32Z`–`2026-08-10T12:49:53Z` against pre-Phase-7-checkpoint HEAD `dc6321cde3a9b8a019c1f0fd82780d90afa046df` with exactly the 12 Phase 7 paths staged and no unstaged delta; retained log `artifacts/verification/q8-mmproj/baseline-final.log`; SHA-256 `ff46cc0b23a28050ffe856be2cb81ef7144667977587021f1d3cd221e00ed330`
- Verifier-only Cargo.lock SHA-256: `acd9419056b786da820b5120db8e78be06902721689c39b55f29445abdddaffc`
- Audit: the assigned worker verified that eligible weights remain `QMatMul::QTensor`, dense Phase 6 APIs remain intact, F32 auto-selection propagates validation failures, CLI diagnostics are explicit, the two-layer fixture covers 14 linears, and no P0/P1 defect remains in the initial CPU-F32 scope
- Phase 7 checkpoint/tag: complete at commit `1a9cf291ba9a1bbac1de83029f9d8f057aca00b5`, annotated tag `lfm2-vl-phase-7-q8`
- Not claimed: production-checkpoint numerical parity, production MMProj payload execution, llama.cpp runtime numerical parity, executed native-Q8 CUDA parity, generated-caption parity, or lower-bit native vision execution

## Vision Safety Limits Verification

- Candidate base HEAD: `f14a46a6967c38e84d99c08801234fd98aa2203a`; WSL2 `NVIDIA-Workbench`; CPU-only, locked, offline lane; no network, model payload, hosted runner, commit, tag, push, or PR
- Shared contract: default and hard ceilings are 67,108,864 pixels per source/derived image surface, 16 images, 11 crops per image, 64 total crops, 1,024 patches per crop, and 65,536 projected tokens; configuration can only tighten them
- Pre-allocation order: raw images are checked before RGB conversion/resizing/cropping/patchification; external prompt batches are revalidated before expansion; packed shapes, ranges, resized surfaces, spatial values, masks, projected counts, and vision batch size are checked before MMProj device transfer
- Focused transformer command: `cargo test --locked --offline -p candle-transformers lfm2_vl -- --nocapture`; 27/27 passed
- Focused VLM command: `cargo test --locked --offline -p candle-vlm lfm2_vl -- --nocapture`; 24/24 passed; all existing processor fixture tensors retain maximum absolute error `1.192092896e-7`
- Full Rust command: `cargo test --locked --offline -p candle-core -p candle-transformers -p candle-vlm`; passed 21/21 core library tests plus all core integration/doc tests, 53/53 transformer library tests, 5/5 generation tests, 8/8 NMS tests, and 29/29 VLM tests
- Checks: locked/offline `cargo check` passed for `candle-core`, `candle-nn`, `candle-transformers`, and `candle-vlm`; the `lfm2`, `quantized-lfm2`, and `lfm2-vl` examples all passed
- Strict scoped Clippy: core/transformer/VLM libraries and the `lfm2-vl` example passed with `-D warnings` plus the five recorded pre-existing Rust 1.97 allowances
- Full baseline: `bash scripts/lfm2-vl/verify-baseline.sh` passed from `2026-08-10T14:11:01Z` to `2026-08-10T14:12:00Z`, including formatting, affected crates, all three examples, and staged/unstaged diff checks; verifier-only Cargo.lock SHA-256 `acd9419056b786da820b5120db8e78be06902721689c39b55f29445abdddaffc`
- Independent re-audit: the assigned GPT-5.6 Luna worker found no remaining P0/P1 or concrete P2; it confirmed hard ceilings, resized-surface coverage, the non-exhaustive pre-release API boundary, full pre-transfer validation, and removal of the merge coverage allocation
- Not claimed: pre-decode image-file header enforcement, production-payload parity, llama.cpp runtime parity, executed CUDA parity, generated-caption parity, or lower-bit native vision execution

## Example Execution Policy Verification

- Candidate base HEAD: `f14a46a6967c38e84d99c08801234fd98aa2203a`; WSL2 `NVIDIA-Workbench`; CPU-only, locked, offline lane; no network, model payload, hosted runner, commit, tag, push, or PR
- Compatibility: both original positional split-MMProj loading and explicit path flags remain accepted; `--processor-config`, `--cpu`, and `--vision-cpu` retain their prior meanings
- Dtype policy: absent `--dtype` remains F32 on CPU and BF16 on CUDA; all canonical and long-form F32/BF16/F16 spellings are covered; diagnostics distinguish requested/defaulted from resolved dtype
- Execution policy: split input resolves dense; direct GGUF auto/dense/Q8 maps to `load_gguf_auto`/`load_gguf`/`load_gguf_q8`; strict Q8 rejects split input and non-F32 activations before any model/tokenizer file is opened
- Focused command: `cargo test --locked --offline -p candle-examples --example lfm2-vl`; 10/10 parser, placement-policy, routing-matrix, pre-I/O, help, and controlled-error tests passed
- Affected check: `cargo check --locked --offline -p candle-examples --example lfm2-vl`; passed
- Strict scoped Clippy: the `lfm2-vl` example passed with `-D warnings` plus the five recorded pre-existing Rust 1.97 allowances
- Full baseline: `bash scripts/lfm2-vl/verify-baseline.sh` passed from `2026-08-10T14:39:59Z` to `2026-08-10T14:40:33Z`, including formatting, all required library and example checks, and staged/unstaged diff checks; verifier-only Cargo.lock SHA-256 `acd9419056b786da820b5120db8e78be06902721689c39b55f29445abdddaffc`
- Independent final re-audit: the assigned GPT-5.6 Luna worker confirmed requested/defaulted and resolved dtype diagnostics, all six dtype spellings, the tested device policy consumed by `main`, loader routing, and pre-I/O Q8 rejection; no P0/P1/P2 defect remains
- Not claimed: successful production-file loading, production numerical parity, executed CUDA behavior, or llama.cpp runtime parity

## Native Unified Checkpoint Verification

- Candidate base HEAD: `f14a46a6967c38e84d99c08801234fd98aa2203a`; WSL2 `NVIDIA-Workbench`; local-only, locked, offline CPU lane; no network, model payload, hosted runner, commit, tag, push, or PR
- File contract: require exactly one `model.safetensors` or `model.safetensors.index.json`; canonicalize every file under the model directory; bound index/header/file/aggregate sizes, shard/tensor counts, shapes, offsets, overlaps, gaps, and payload coverage before memory mapping
- Model contract: derive the complete expected inventory from normalized configuration; accept the official `model.vision_tower.vision_model` root and committed-fixture direct root; support tied embeddings or explicit `lm_head`; reject missing, unexpected, or shape-incompatible tensors before payload construction
- Pairing contract: require local `config.json`, `processor_config.json`, and `tokenizer.json`; apply explicit processor override precedence; require processor patch/downsample compatibility and exact tokenizer/model image-token ID agreement
- Placement contract: explicit dtype applies to both native components; default dtype resolves independently per text and vision device, and distinct dtype builders are not silently shared
- Focused command: `cargo test --locked --offline -p candle-examples --example lfm2-vl`; 19/19 passed, covering actual tiny single/sharded safetensors, canonical/direct roots, tied/explicit heads, exact official 349/589 inventories, independent component dtypes, wrong index mappings, bad `total_size`, duplicate shard tensors, traversal, missing files, and pairing failures
- Official header contract: bounded pinned Range reads consumed 46,864 and 82,400 header bytes and zero payload bytes; canonical sorted name/BF16/shape SHA-256 values are `08f544b4495804ed842a37acf0936544ec88aa5d947bef8304a47816fee5b1a7` for 450M and `24728d0ed10229e788c5b9baf25e0cc6c92c93b9cdb12ebb252a3c140a861703` for 1.6B; the test also asserts raw FFN 6,656/12,288 normalization to 4,608/8,192
- Full Rust command: `cargo test --locked --offline -p candle-core -p candle-transformers -p candle-vlm`; passed all core library/integration/doc tests, 53/53 transformer library tests, 5/5 generation tests, 8/8 NMS tests, and 29/29 VLM tests
- Strict scoped Clippy: core/transformer/VLM libraries and the `lfm2-vl` example passed with `-D warnings` plus the five recorded pre-existing Rust 1.97 allowances
- Full baseline: `bash scripts/lfm2-vl/verify-baseline.sh` passed from `2026-08-10T15:42:32Z` to `2026-08-10T15:43:14Z`, including formatting, all required library/example checks, and both diff gates; verifier-only Cargo.lock SHA-256 `7292957b78b688fe2d8d0f61ba5987b92638d6138a0faa9a13db014d09b06a26`
- Launcher note: the first non-login WSL invocation at `2026-08-10T15:29:39Z` stopped before the formatting step because `cargo` was absent from that shell's `PATH`; the corrected login-shell command above is the verification result
- Integrity boundary: checkpoint files are an immutable local snapshot from header inspection through the returned model lifetime, as required by memory-mapped safetensors
- CUDA source boundary: the feature-gated native test covers distinct-device construction/loading only; the earlier hybrid test owns projected-feature transfer and forward coverage. Executed native CUDA inference is not claimed.
- Independent final re-audit: the assigned GPT-5.6 Luna worker confirmed the full official inventory digests, correct raw-to-effective FFN normalization, independent dtype policy, indexed-shard defenses, roots, tied/explicit heads, pairing, pre-payload rejection, CLI routing, mmap precondition, and honest CUDA scope; no P0/P1/P2 finding remains
- Reference-suite note: the ad hoc system-Python discovery command could not import `pytest`, and the previously documented repo `.venv` is not present in this checkout. No dependency was installed; the changed lock JSON was parsed directly and the exact digests are enforced by the green Rust test.
- Not claimed: production-payload construction or numerical parity, generated output, local llama.cpp runtime parity, executed native CUDA inference, or lower-bit native vision execution

## Deterministic Runner and Local llama.cpp Verification

- Candidate base HEAD: `f14a46a6967c38e84d99c08801234fd98aa2203a`; local-only CPU execution; no hosted runner, commit, tag, push, PR, production-weight download, or secret inspection
- Runner contract: `candle-lfm2-vl-inference-v1`; bounded prompt/image/generation inputs; deterministic greedy selection with lower-token-ID tie breaking; finite-logit enforcement; full F32-logit SHA-256 plus top-5 per step; exact expanded prompt, token IDs, image spans, crop metadata, packed tensor shapes, EOS provenance, decoded forms, and two-run cache-reset equality
- Artifact evidence: native, split, and direct loaders provide the exact config, tokenizer, processor, index/shard, manifest, and weight files they consumed. The runner canonicalizes, deduplicates, sizes, and SHA-256 hashes each regular file and rejects directory-only evidence. Input files must remain immutable from loader open through report emission.
- Focused command: `cargo test --locked --offline -p candle-examples --example lfm2-vl`; 26/26 passed. The real hybrid runner regression constructs deterministic text GGUF bytes from the committed tiny tensors, loads the committed split MMProj bundle, processes a generated 8x4 PNG, performs prefill plus three cached decode steps twice, resolves EOS from GGUF metadata, hashes all five consumed files, and serializes one-line JSON. A pure source-list regression asserts exact split, direct-GGUF, bundled-processor deduplication, and explicit-override inputs.
- Focused transformer command: `cargo test --locked --offline -p candle-transformers lfm2 -- --nocapture`; 32/32 focused tests passed with the established deterministic GGUF hashes and numerical tolerances unchanged. Optional `tokenizer.ggml.eos_token_id` parsing and validation are covered.
- Affected check: `cargo check --locked --offline -p candle-examples --example lfm2-vl`; passed in the WSL2 locked/offline lane.
- Full Rust command: `cargo test --locked --offline -p candle-core -p candle-transformers -p candle-vlm`; passed all library, integration, and doc-test lanes with 53/53 transformer library tests, 5/5 generation tests, 8/8 NMS tests, and 29/29 VLM tests.
- Strict scoped Clippy: core/transformer/VLM libraries and the `lfm2-vl` example passed with `-D warnings` plus the five recorded pre-existing Rust 1.97 allowances. The final example pass followed a small `GenerationInputs` grouping fix and the 26/26 test replay.
- Full baseline: `bash scripts/lfm2-vl/verify-baseline.sh` passed from `2026-08-10T19:42:13Z` to `2026-08-10T19:42:46Z`, including formatting, every required library/example check, and both diff gates; verifier-only Cargo.lock SHA-256 `7292957b78b688fe2d8d0f61ba5987b92638d6138a0faa9a13db014d09b06a26`.
- Provenance classification: relative to untouched Candle `31f35b147389700ed2a178ee66a91c3cc25cc80d`, the current 72-path delta is exactly nine fork-origin modifications plus 63 mod-owned additions. There are no unexpected baseline edits and no changed paths absent from `MOD_MANIFEST.md`.
- Independent audit boundary: the designated GPT-5.6 Luna task failed to start with `agent loop died unexpectedly`. No replacement user task was created. The manager's local audit added the missing direct-GGUF source-list regression and found no remaining P0/P1/P2 defect in the NR-4 scope.
- Resource incident: the completed local llama.cpp proof left PID 32000 resident with up to `131,549,319,168` private bytes. PowerShell, exact `taskkill`, Task Manager, and native termination attempts failed, timed out, or were denied. The PID later disappeared, but host performance and memory pressure did not recover sufficiently; an operator restart was required. WER recorded `RADAR_PRE_LEAK_64` and Windows recorded low-virtual-memory events. The legacy b9981 bundle is coherent and Defender/Code Integrity checks found no llama-related block. Exact root cause remains unproven; F-0008 records the evidence and closes the operational mystery with containment rather than a speculative attribution.
- Bounded owner proof: `pwsh -NoProfile -NonInteractive -File scripts/lfm2-vl/test-bounded-oracle.ps1` passes harmless normal-exit, timeout/descendant, owner-exit, concurrent-name-refusal, suspended-start, assign-before-resume, and exact-PID-absence cases. The wrapper defaults to a 24 GiB ceiling, rejects limits above 75% of physical RAM, defaults CUDA graphs off, and writes atomic JSON evidence.
- Installed oracle: `C:\llamacpp\llama-mtmd-cli.exe`, build b9981 / `(34558825a)`, 82,944 bytes, SHA-256 `01e191f9dd389b6e3b091eeaa8b6142784bd0e1b0e19ed7c67039afc6626ae1d`
- Manual current comparison: `C:\llamacpp\tools-b10344\llama-mtmd-cli.exe`, 84,480 bytes, SHA-256 `78ef208334fec62d62068cfd242a0b7358c602211125ceead3dd82b1347f717c`; inventoried read-only and not used as the pinned authority.
- Pinned owner oracle: ignored bundle `artifacts/llama-oracle/74ce1574-cuda-sm89/bundle`, exact source `74ce15741b420b8d6f12e720398458b576c51c2c`, CUDA 13.3/SM89/MSVC 19.33, executable SHA-256 `848e638069699149210b70945bdbb422494d7d03b8a18d7fb31a240d10e8abd0`. The parallel-1 build exited 0 with peak Job Object memory `7,889,661,952` bytes. Its bounded no-model probe reports `version: 10335 (74ce15741)`, peaks at `291,172,352` Job bytes, and verifies suspended assignment, exit 0, and PID absence. `bundle-manifest.json` records the complete EXE/DLL closure and hashes; no model was loaded.
- Text GGUF: local fine-tuned SFT derivative, 219,310,432 bytes, SHA-256 `84540fa23696ab9000f4a670b72e3405962264a920c3b7582d0e5a38b978abae`; exact bounded header ends at byte 2,387,296, contains 27 metadata records and 148 tensors, and reports LFM2 hidden width 1,024, 16 layers, vocabulary 65,536, context 128,000, image token 396, and EOS 7
- MMProj GGUF: 102,815,168 bytes, SHA-256 `ebfc428baa37efad8bae93864f914b2634a09009f91ad59f974fe1a1565d8561`. Its size and complete-file hash exactly match `LiquidAI/LFM2.5-VL-450M-GGUF@166cd80bbe157dc86d65f964eb8cc6a2cede62ca`, proving the local Q8_0 MMProj is byte-for-byte official.
- Tokenizer: pinned `LiquidAI/LFM2.5-VL-450M@fc6221ca597f3315e4f82fc2df606783267b34ba/tokenizer.json`, 4,733,040 bytes, SHA-256 `f3910942aa907c48b0cc20ec426ee38bfa8dcda8feecf035ced981918cb30f14`; image token 396. The unrelated local 2.6B tokenizer was rejected because its image token 124,907 is outside this model's 65,536-token vocabulary.
- Image: `candle-examples/examples/yolo-v8/assets/bike.jpg`, 182,991 bytes, SHA-256 `317e4a9d2d2be7859ba0ab8726a526f5ece9d77daf92857f3e93fb7b367824c1`, source dimensions 800x556
- Aligned structure: llama.cpp and Candle used the same text/MMProj/tokenizer/image, deterministic greedy settings, 4,096-token context, eight requested tokens, and equivalent chat framing. Both produced 608x416 vision input, 247 projected tokens, and 268 prompt tokens; Candle recorded the image span as `[5, 252)`.
- Exact output agreement: both runtimes decoded `A group of cyclists race on a road`. Candle generated IDs `[542, 2514, 803, 62480, 7736, 884, 768, 6671]` and tokens `["A", "Ġgroup", "Ġof", "Ġcyclists", "Ġrace", "Ġon", "Ġa", "Ġroad"]`; cache reset replay was exact. Candle prefill full-logit SHA-256 was `b2f21e55e855d162ecb3a4a91fdda102728c76c5e39c489377fb6ddae66d287b`.
- Claim boundary: this proves same-artifact preprocessing structure and greedy decoded-sequence/output behavior for the local fine-tuned text GGUF plus official MMProj. It does not prove official-base text parity, installed-build identity with the pinned llama.cpp commit, or component/logit equality because `llama-mtmd-cli` exposes no stable intermediate/logit dump.

## Bootstrap Proof

- Date: 2026-08-09 22:35 EDT (`2026-08-10T02:35:09Z` to `2026-08-10T02:35:12Z`)
- Environment: WSL2 `NVIDIA-Workbench`; Ubuntu 22.04.5 LTS; Linux `6.6.87.2-microsoft-standard-WSL2`; CPU-only lane
- Verification HEAD: `31f35b147389700ed2a178ee66a91c3cc25cc80d` with bootstrap paths staged in the detached Linux verification worktree
- Command: from `/tmp`, `bash /home/workbench/code/candle-lfm2-vl-verify/scripts/lfm2-vl/verify-baseline.sh`
- Results: passed formatting; locked/offline checks for `candle-core`, `candle-nn`, `candle-transformers`, `lfm2`, and `quantized-lfm2`; staged and unstaged diff checks
- Local-only lockfile SHA-256: `4e059ffe6035520ca6553303932173eba562f4985f82931a90129eea9849ce54`
- Retained baseline log SHA-256: `a4f77d1b007eb267865be01ef1c239754ac0e093dd1c27ad457d77242b614f22`
- Retained environment log SHA-256: `5f4fd70b4dd5ca6a956c9678d386598ca2ff6bcdb2e75ef3ba3aa6a10775e4d8`

## Environment Snapshot

- WSL2 distribution: `NVIDIA-Workbench`
- Linux home: `/home/workbench`
- Rust compiler: `rustc 1.97.1 (8bab26f4f 2026-07-14)`
- Cargo: `cargo 1.97.1 (c980f4866 2026-06-30)`
- Python: `Python 3.10.12`
- CMake: `cmake 3.22.1`
- Ninja: missing; optional for current gates
- Reference Python: isolated `.venv` with a complete checked-in CPU-lane dependency lock

## Proven

- The untouched Candle 0.11 baseline and existing dense/quantized LFM2 examples compile in the locked, offline CPU lane.
- Every external implementation and model reference is pinned to an immutable revision with path, purpose, authority, license, and adaptation boundary.
- Both official checkpoint tensor namespaces and representative shapes were read from safetensors headers without reading tensor payloads.
- The 450M effective FFN width is 4,608 and the 1.6B width is 8,192; the production headers confirm both.
- Both checkpoints omit `lm_head.weight`, confirming the required tied-output loading path.
- The source-lock patch changes no Candle Rust source, Cargo manifest, lockfile policy, or runtime dependency.
- The pinned official Transformers LFM2, SigLIP2, and LFM2-VL classes execute deterministically in the tiny-random harness, including multimodal prefill and incremental cache reuse.
- Config-only mode is stdlib-only and production model loading remains guarded by explicit production, load, and download flags.
- The tiny fixture deliberately omits the tied `lm_head.weight` duplicate, preserving the production checkpoints' missing-head loading contract.
- The complete Phase 1 text gate passes in the Linux-home CPU/offline lane: all 5 focused LFM2 tests, all 18 `candle-transformers` library tests, both existing LFM2 examples, and the full locked/offline baseline are green.
- Phase 1 is checkpointed at `f660b8e3f2b4560f133356864e012be83f29d9c0` and tagged `lfm2-vl-phase-1-text`.
- The focused Phase 2 SigLIP2 gate is green: all 7 tests pass with the exact stage errors recorded above; the Phase 2 checkpoint/tag is complete at `74e109aec5f9801cfead3eeb27fe3f93ac646b84` / `lfm2-vl-phase-2-siglip2`.
- The Phase 3 focused gate is green at 11/11, the SigLIP2 repeated-crop regression is green at 8/8, and the `candle-transformers` library gate is green at 37/37. Phase 3 is checkpointed at `37264b49cf74d0cf7697317eda0183f084db6ff8` and tagged `lfm2-vl-phase-3-native-composite`.
- The Phase 4 Rust-native raw-image and prompt path is green at 24/24 against all required pinned fixtures. Packed integer metadata, exact prompt strings/IDs/spans, crop ordering, and fixture regeneration are exact; normalized pixel values differ by at most `1.192092896e-7`.
- Phase 5 is checkpointed at `1535a0a5fef09f243811b83553b9c75baad78ee2` / `lfm2-vl-phase-5-hybrid`; split/native image features are exact and deterministic quantized-text hybrid prefill/decode remains within `4.457309842e-5`.
- The Phase 6 direct GGUF dense compatibility loader has exact dense/native image features, Q8_0-dequantized feature error `8.463021368e-5`, and direct-hybrid prefill/decode equal to the Phase 5 deterministic bounds.
- Official F16 and Q8_0 MMProj physical shapes, dtype placement, metadata, and names are locked from exact zero-payload header ranges; only the patch tensor requires a layout inverse.
- The Phase 7 native-Q8 path retains eligible GGUF weights as `QMatMul::QTensor`; the two-layer all-linear fixture reaches cosine `0.999923348`, and the committed hybrid fixture stays within `1.650899649e-4` prefill drift with exact cache reset.
- Shared request-wide vision limits now reject zero, one-over, overflow, above-hard-ceiling, malformed packed metadata, and oversized derived surfaces before expensive allocation or MMProj device transfer while preserving every existing tiny fixture result.
- The `lfm2-vl` example now exposes explicit dtype and MMProj execution intent, preserves original path and device flags, rejects invalid strict-Q8 policy before file I/O, and reports requested versus resolved policy.
- The example now loads an unmodified local unified Hugging Face directory from single or indexed safetensors, validates the entire normalized tensor contract before mapping, honors tied output weights, pairs processor/tokenizer/config inputs, and resolves text/vision dtype independently by device.
- The deterministic runner covers native and hybrid image prefill, cached decode, exact reset replay, finite full-logit hashes, stable top-k/token evidence, one-line JSON, and bounded external image/prompt/model-file diagnostics.
- Native, split-MMProj, and direct-GGUF inference evidence identifies and hashes every exact consumed file; directory paths are not accepted as artifact identity.
- The local fine-tuned text GGUF and byte-for-byte official Q8_0 MMProj produce the exact same eight-token caption under aligned deterministic Candle and llama.cpp execution.
- The Windows bounded-oracle smoke suite proves suspended Job Object assignment before resume, timeout and owner-exit tree cleanup, concurrency refusal, and exact PID absence without loading a model.
- The pinned llama.cpp b10335 CUDA/SM89 bundle is source-, build-, and file-identified and passes a 512 MiB/30-second bounded no-model identity probe with no residual process.
- Tiny-fixture dense parity is within `2.38418579e-7` for hidden states and `2.98023224e-8` for logits; production-checkpoint and GGUF numerical parity remain unclaimed.

## Known Conflicts

- Official config context is 128,000 while model cards advertise 32,768; construction follows config and production policy remains unresolved.
- Numeric IDs for image wrapper, row/column, and thumbnail marker strings must be exported by the tokenizer harness; only image placeholder ID 396 is config-explicit.
- llama.cpp PR #25524 for reading LFM2 tiling parameters from GGUF metadata is open and unmerged; official processor config remains authoritative.
- The official MMProj headers omit all three tiling metadata keys; direct loading therefore depends on pinned architecture defaults or an explicit processor document.
- The local WSL verifier exposes an RTX 4090 through the driver but has no Linux CUDA toolkit or `nvcc`; the committed distinct-device test remains an owner-scoped execution gap.
- The legacy `C:\llamacpp` runtime is build `b9981` / `(34558825a)` and the manual `tools-b10344` bundle is a newer current-master comparison. Neither substitutes for the exact pinned b10335 owner build.
- The only local text GGUF is a fine-tuned game-QA SFT derivative, not the official base checkpoint. Its pairing with the pinned official tokenizer and byte-identical official MMProj is proven for same-artifact execution, but it cannot serve as official-base text parity evidence.
- `llama-mtmd-cli` exposes deterministic sampling controls but no stable logits or intermediate-tensor dump contract, so local llama.cpp can currently prove same-artifact prompt/token/output behavior, not component-tensor equality by itself.
- A completed `llama-mtmd-cli` run remained attached under Codex with approximately 131.5 GB private memory, while normal exact-PID termination was denied or timed out. PID disappearance did not restore usable host performance; restart was required. WER leak evidence, virtual-memory pressure, related upstream CUDA/MTMD reports, and a possible Codex token/job boundary are recorded, but none is proven as the unique cause; see F-0008.

## Blockers

- None for starting official 450M native CPU-F32 work or the pinned llama.cpp lane: containment and the no-model identity probe are green. Git publication still needs an accessible `origin`; this worktree currently has none, and local `gh` is unauthenticated. Official-base production payload parity, the 1.6B checkpoint, executed native-Q8 CUDA, and lower-bit evidence remain incomplete and unclaimed.

## Active Files

- `candle-transformers/src/models/lfm2.rs`
- `candle-transformers/src/models/quantized_lfm2.rs`
- `candle-transformers/src/models/siglip2.rs`
- `candle-transformers/src/models/lfm2_vl/`
- `candle-transformers/src/models/lfm2_vl/gguf.rs`
- `candle-transformers/src/models/lfm2_vl/linear.rs`
- `candle-core/src/quantized/gguf_file.rs`
- `candle-vlm/`
- `candle-vlm/src/lfm2_vl/config.rs`
- `candle-vlm/src/lfm2_vl/processor.rs`
- `candle-vlm/src/lfm2_vl/prompt.rs`
- `candle-examples/examples/lfm2-vl/`
- `candle-examples/examples/lfm2-vl/runner.rs`
- `tools/lfm2_vl/reference/inspect_gguf_header.py`
- `tools/lfm2_vl/reference/test_gguf_header.py`
- `tools/export_lfm2_vl_mmproj.py`
- `scripts/lfm2-vl/run-bounded-oracle.ps1`
- `scripts/lfm2-vl/test-bounded-oracle.ps1`
- `tests/fixtures/lfm2_vl_mmproj_tiny/`
- `candle-transformers/src/models/mod.rs`
- `candle-examples/examples/lfm2/main.rs`
- `tests/fixtures/lfm2_vl_processor_tiny/`
- `tools/lfm2_vl/reference/export_processor_fixture.py`
- `docs/lfm2-vl/DECISIONS.md`
- `docs/lfm2-vl/PARITY.md`
- `docs/lfm2-vl/MOD_MANIFEST.md`
- `docs/lfm2-vl/FAILURE_LOG.md`
- `docs/lfm2-vl/STATUS.md`

## Next Task

NR-5B — obtain the pinned official `LiquidAI/LFM2.5-VL-450M@fc6221ca597f3315e4f82fc2df606783267b34ba` native files only through the guarded production path. Record repository, revision, filename, size, and SHA-256; preflight host commit/physical/GPU memory; run CPU F32 first; and compare selected processor, vision, projector, merge, prefill, and cached-decode tensors against the pinned Transformers oracle. Use the bounded llama.cpp owner only for a later same-artifact comparison. Done when every selected production tensor is within the specified tolerance, the deterministic trace replays exactly, all consumed files are identified, the process tree is absent, and post-run host memory is healthy. Do not start 1.6B or CUDA inference before this gate is green.

## 2026-08-10 Windows-First Verification Handoff

The owner clarified that native Windows is the intended product/runtime platform and WSL is a useful OS-agnostic replay. The platform decision was recorded as D-0029; the current WSL-owned `.git` pointer remains only a checkout-specific Git constraint.

Native preflight found Cargo/Rust `1.91.0` on `stable-x86_64-pc-windows-msvc`, Visual Studio 2022 Build Tools, CMake, and Ninja. No `llama*` process was resident, and `cargo fmt --all -- --check` passed. The ignored local lockfile was 189,832 bytes with SHA-256 `7292957b78b688fe2d8d0f61ba5987b92638d6138a0faa9a13db014d09b06a26`.

The bounded command `CARGO_NET_OFFLINE=true`, `CARGO_BUILD_JOBS=2`, `cargo test --locked --offline -p candle-transformers models::lfm2::tests::flash_attention_without_feature_returns_an_error -- --exact` stopped before compilation: the Windows Cargo cache offered `thiserror` through `2.0.19`, but the lockfile requires `2.0.20`. No dependency, model, or runtime was downloaded. This is an environment-blocked Windows lane, not a failed source test; TODO P0 owns the deliberate cache hydration and offline rerun.

The same focused command was deliberately re-run at `2026-08-10T20:42:27-04:00` with `cargo 1.91.0`, `rustc 1.91.0`, `CARGO_NET_OFFLINE=true`, and `CARGO_BUILD_JOBS=2`. Cargo reproduced the identical locked-resolution error before compilation (`thiserror ^2`, locked to `2.0.20`; cached candidates through `2.0.19`). No process, model, lockfile, or dependency state changed. The next action remains an owner-authorized, bounded cache hydration; no implicit network fetch was attempted.

The continuation integrity closeout at `2026-08-10T20:43:40-04:00` passed native formatting, summary-bank validation, JSON parsing, and documentation whitespace checks. Windows Git could not inspect the linked WSL-owned worktree (`fatal: not a git repository: (NULL)`), and the direct non-elevated WSL retry returned `E_ACCESSDENIED`; the prior WSL diff/manifest result remains the last green Git-owned check. No source, model, dependency, or process state changed.

## 2026-08-10 Native Windows P0 Completion and LFM2 Safety Hardening

Owner-authorized hydration made the exact locked `x86_64-pc-windows-msvc` dependency set available without changing the ignored local lockfile or downloading a model. Native Windows used Cargo/Rust `1.91.0`, `CARGO_NET_OFFLINE=true`, `CARGO_BUILD_JOBS=2`, and the bounded target directory `target-native`.

The current-tree verification window ran from `2026-08-10T21:00:24-04:00` through `2026-08-10T21:01:04-04:00`. `cargo test --locked --offline -p candle-core -p candle-transformers -p candle-vlm --quiet` passed every library, integration, and doc-test lane; transformer unit tests were 56/56, generation 5/5, NMS 8/8, and VLM 29/29. `cargo test --locked --offline -p candle-examples --example lfm2-vl --quiet` passed 26/26. Offline `cargo check` passed for `lfm2`, `quantized-lfm2`, and `lfm2-vl`; formatting and the focused LFM2 suite also passed. No `llama*` process was resident after the run.

The source hardening closes an external-input safety gap: `Config::validate` now rejects zero or incompatible attention dimensions, invalid rotary/norm values, zero convolution cache, unrepresentable positions, and shape arithmetic overflow before model/cache construction. Cache creation uses the validated position range, sequence-position addition is checked, and model forwarding rejects ranges beyond the rotary cache. Two focused regressions cover malformed dimensions and position/index overflow. This preserves valid 450M/1.6B normalization while converting previously panic-prone or silently truncated inputs into actionable errors.

The pinned official 450M snapshot was already present in the local Hugging Face cache at revision `fc6221ca597f3315e4f82fc2df606783267b34ba`; no production download was needed. Its Windows symlinked snapshot was rejected by the native immutable-inventory loader because the symlink targets live outside the supplied model directory. A regular-file copy was made under the system temporary directory, loaded with `--cpu` and no prompt, then removed after the proof. The run passed from `2026-08-10T21:06:37-04:00` through `2026-08-10T21:06:42-04:00` with 349 tensors, 897,484,568 model bytes, 12 vision layers, 16×1024 text, image token 396, processor max patches 1024, tied output embeddings, and F32 CPU placement. Exact source file sizes and SHA-256 values are retained in `PARITY.md`; no component parity or production inference claim is made.

The LFM2 safety slice added checked configuration, cache-position, and sequence-position validation with focused malformed-input regressions; the native Windows focused LFM2 lane is now 7/7 and the affected full transformer lane is 56/56. The LFM2-VL example parser now boxes `ParseOutcome::Run(Args)`, preserving the CLI contract while satisfying the current `large-enum-variant` lint. Current-tree strict scoped Clippy passed at `2026-08-10T21:16:29-04:00` for core/transformer/VLM libraries and the LFM2-VL example under Rust 1.91.0 with `-D warnings` and the five valid allowances: `needless-borrows-for-generic-args`, `manual-filter`, `manual-is-multiple-of`, `needless-range-loop`, and `manual-contains`.

Added the guarded `tools/lfm2_vl/reference/production_trace.py` path for the remaining NR-5B evidence. It requires explicit production and model-load opt-in, an external regular-file image and output directory, a non-empty prompt, CPU F32, deterministic single-thread execution, bounded input/patch/decode sizes, and a bit-exact prefill cache-reset check. It records processor inputs, vision hooks, projector input/output, merged embeddings, prefill logits, greedy cached-decode logits, and file/package identity without serializing weights. Compile-only and fail-closed CLI checks passed; no production trace ran because the pinned Torch/Transformers environment is still unavailable on native Windows.

## 2026-08-10 Gknome Adoption Gate

The Gknome worker exhausted its usage allowance before final handoff. Its current source was copied into a fresh secret-free disposable directory, initialized as a clean local `main` repository, and verified independently. `pwsh -NoProfile -File tests/Test-Adoption.ps1 -AsJson` passed 45 assertions in 52.3 seconds. The suite covered WSL `.git` recognition/refusal, compatible bank preservation, generated context/project tests, unsafe/incompatible bank conflicts, zero-mutation rollback backup preflight, tampered-plan refusal, transaction rollback, reparse refusal, secret pruning, and zero secret-value exposure.

The subsequent Candle dry run returned `status=blocked` and applied zero files. It identified exactly four root-authority conflicts (`.gitignore`, `AGENTS.md`, `README.md`, and `summary_bank.json`) while leaving the bank byte-identical at SHA-256 `67594972afc2bfdde4279b7708aa0a25bba94f7a258ff21d9b6cc20fe27f1e9e`. Review also found 3,384 ignored `artifacts/` paths and five `.pytest_cache/` paths in the hashed adoption inventory. No apply or repair followed; F-0010 and TODO C1 own the unresolved integration.

After recording the new Gknome issue route, the current bank hash became `2eeba50fcdbb2865120468b892c277422450aad814265e4c50dcbae4c52f2321` and its verifier remained green. Native formatting, Markdown relative links, JSON/PowerShell/Bash syntax, trailing whitespace, absence of `.gknome` residue, and absence of `llama*` processes also passed. A final WSL replay could not launch because the approval reviewer exhausted its own usage allowance; the prior WSL gate remains the last green result on the unchanged Rust/source tree, and no bypass was attempted.

## 2026-08-11 NR-5B Official 450M Native Windows CPU-F32 Parity

NR-5B completed against `LiquidAI/LFM2.5-VL-450M@fc6221ca597f3315e4f82fc2df606783267b34ba` on native Windows CPU F32. No network access, production-weight serialization, CUDA execution, commit, push, PR, or secret inspection occurred.

- Reference environment: official Python 3.10.11 x64 in the ignored `.venv`; 42-distribution Windows lock SHA-256 `d4e045f5577a67ffae3132082c933c6ea6f5a7bb27b2204bb40307acfc784286`; exact runtime/test/VCS verifier green; complete reference suite 43/43.
- Artifact identity: an external eight-file regular-file snapshot totaling 902,236,184 bytes. Its hash-only manifest is 1,981 bytes with SHA-256 `659c8421530586b6cc28c094cfcdc69719ea8626f2abc0efd9eec4ac2a68a984`. Both model lanes reverified their inputs after execution.
- P1-C dry load: the pinned Transformers class and processor loaded locally with no download and no tensor output. Wrapper exit was 0, PID cleanup was exact, peak Job memory was 2,875,105,280 bytes, and the load manifest SHA-256 was `fbb5ef8f8088089b102093d9deae37c7641e874fc9fa339d50489e067ee36351`.
- Deterministic input: a 572-byte 256x256 RGB PNG with SHA-256 `f902f8d2e47e53eafac86831cfc692001dc15870eb81d57abc3128f048d2efca`; user text `Describe this image.`; official rendered prompt retained separately and replayed verbatim; three greedy decode steps.
- Python oracle: 36-tensor bundle manifest SHA-256 `41f97daf914bd2c3eea81065ca87f1b002e869dd0dcedf010bba229646529d06`; wrapper evidence SHA-256 `08871cfb03442daebd0995ed427778ae5a609ccbc1007cf6af0c3aacc14236c6`; exit 0; PID absent; peak Job memory 4,966,543,360 bytes.
- Native replay: release executable 10,230,272 bytes, SHA-256 `338ebcbf02dbac13fabf6ce9115bdb3a91fc3316a84a9c23e1ad304fbd900d9a`; 8 GiB Job ceiling; exit 0; PID 33624 absent; peak working set 2,320,310,272 bytes; peak Job memory 2,120,413,184 bytes. Native manifest SHA-256 `286bc3c453188de38ac12a9553e60515a17aad61a57d03086c350b0f2d013345`.
- Comparison: `comparison-256-v4.json` is 11,241 bytes with SHA-256 `caaae9ad159ec8370007169bd7c486ccff96f8b547ea6a113685f0c8703bbbac`; `passed=true`, 36 tensors compared, zero failures, exact integer/input tensors, matching 12-layer stage inventory, exact cache reset, and exact artifact identity for every native-consumed config/processor/tokenizer/weight file. The largest maximum absolute delta was `0.0189208984375` at vision encoder layer 11; that tensor passed its recorded CPU-F32 allclose contract.
- Resource recovery: the final census found zero llama/model/Cargo/rustc processes, 46,049,075,200 available physical bytes, 49,654,607,872 bytes of commit headroom, and 23,420 MiB GPU memory free. The post-run report SHA-256 is `3133dc6060c2512448402bbd6bc9443de3b408a786cee890e5f15f84a71fc9c1`.

The production pass exposed and fixed three evidence-path defects before closure: process counters above `Int32::MAX` crashed wrapper sampling; native trace inventories used implementation dtype abbreviations and omitted consumed-file evidence; and the comparison CLI returned 0 even when its report contained a failed tensor. The projector range was also corrected from post-projector token ranges to the oracle's pre-projector valid-patch range.

The final current-tree Windows gate passed locked/offline formatting, checks for the four affected libraries and three examples, the complete affected core/transformer/VLM test lanes, 28/28 LFM2-VL example tests, strict scoped Clippy, Python compileall and environment-lock verification, and cross-version wrapper/preflight smoke. The pinned reference suite was 43/43 at NR-5B closure and is 47/47 after the subsequent P2 GGUF-inspection regressions. The reorganized summary bank passed PowerShell 7 and 5.1 at SHA-256 `03e6abd351579360b0ba11fe980f6f74532a3b8d9217e6bf9c7cb6d9a9b3c119`, and all 20 mod-owned Markdown files passed relative-link validation. Generated Python caches were removed after verification.

This gate proves the selected official 450M CPU-F32 component and deterministic decode contract. It does not prove the 1.6B checkpoint, official-base GGUF same-artifact behavior, CUDA, lower-bit MMProj execution, or WSL replay.

## 2026-08-11 P2 Official 450M GGUF Artifact Identity

Completed the no-inference artifact subgate for the official same-artifact GGUF comparison. The official Hugging Face file page ties `LFM2.5-VL-450M-Q4_0.gguf` to `LiquidAI/LFM2.5-VL-450M-GGUF@166cd80bbe157dc86d65f964eb8cc6a2cede62ca`. Its pre-existing local regular blob is 219,311,264 bytes with SHA-256 `6d2757dd0f0b98aea7dc90477bb5b3a0df1089be85ef92943f8cecb05121ccbf`.

The new bounded full-file inspector mode read and hashed only the exact 2,388,128-byte header prefix through the tensor-data offset. Header SHA-256 is `bdb33b992b136a77b4d807b84319a7daa43ebac15144e6336c0d9b9ef1e8ed2e`; it contains 39 metadata records, 148 tensors, dtype counts F32 55/Q4_0 92/Q6_K 1, and a declared extent exactly equal to the physical file. Selected metadata matches the official 450M text contract: LFM2, 16 layers, width 1024, FFN 4608, 128,000 context, 65,536-token GPT-2 tokenizer, BOS 1, EOS 7, and pad 0. The full UTF-8 report is retained under ignored `artifacts/official-base-gguf/` and is not a publication path.

The official Q8_0 MMProj blob and the existing `C:\llamacpp` copy are byte-identical at 102,815,168 bytes and SHA-256 `ebfc428baa37efad8bae93864f914b2634a09009f91ad59f974fe1a1565d8561`. The previously used game-QA SFT text derivative is separately 219,310,432 bytes with SHA-256 `84540fa23696ab9000f4a670b72e3405962264a920c3b7582d0e5a38b978abae` and a different 27-record/header identity, so it cannot be confused with the P2 artifact.

Inspection exposed a Windows console defect before execution: unescaped tokenizer Unicode made the full JSON print fail under CP1252. Stdout is now ASCII-escaped, retained reports remain UTF-8, and `--output ... --quiet` prevents multi-megabyte tokenizer inventories from duplicating into wrapper logs. Eight focused GGUF-inspector tests cover prefix mode, bounded full-file mode, exact lock facts, Unicode stdout, quiet output, and malformed inputs. No GGUF tensor was decoded, no model was loaded, and no inference process started.

## 2026-08-11 P2 Official 450M GGUF Same-Artifact Runtime Comparison

Completed the official-base replay sequentially under the native Windows bounded owner. Both runtimes consumed the exact official Q4_0 text GGUF (`6d2757dd0f0b98aea7dc90477bb5b3a0df1089be85ef92943f8cecb05121ccbf`), official Q8_0 MMProj (`ebfc428baa37efad8bae93864f914b2634a09009f91ad59f974fe1a1565d8561`), and 256x256 deterministic image (`f902f8d2e47e53eafac86831cfc692001dc15870eb81d57abc3128f048d2efca`). No model or build process overlapped either run.

- Candle: the already-built release executable (`338ebcbf02dbac13fabf6ce9115bdb3a91fc3316a84a9c23e1ad304fbd900d9a`) ran CPU F32 hybrid GGUF/MMProj, exited 0 in 3,449 ms, and left PID 23832 absent. Its 8 GiB Job peaked at 918,130,688 bytes. The runner retained exact artifact evidence, 256x256 preprocessing, one whole crop, 64 projected image tokens, image span `[5,69)`, 80 input IDs, generated IDs `[1098, 4646, 5251]`, prefill-logit SHA-256 `aa2e0aa2132cb67fc33cb57523e73dee1c0cabac9362d7adbb22ea1a871d5280`, decoded `The image features`, and exact cache reset.
- llama.cpp: pinned `llama-mtmd-cli` build 10335 at `74ce15741b420b8d6f12e720398458b576c51c2c` (`848e638069699149210b70945bdbb422494d7d03b8a18d7fb31a240d10e8abd0`) ran CPU text and MMProj, exited 0 in 1,318 ms, and left PID 6124 absent. Its 8 GiB Job peaked at 1,777,582,080 bytes. It decoded exactly `The image features` and logged one MTMD chunk. The experimental CLI exposes no stable generated IDs, preprocessing dimensions, projected-token count, logits, component tensors, or cache-reset replay, so those fields remain explicitly unavailable.
- Prompt equivalence: Candle consumed the 92-byte official rendered prompt. llama.cpp received the same raw `Describe this image.` user content and applied the GGUF-embedded official template. The embedded template is the standalone pinned template plus one trailing LF; their trimmed contents are exact.
- Context boundary: Candle reports the artifact capacity of 128,000; llama.cpp was deliberately capped at 4,096 to contain KV memory. The actual sequence occupies only 83 positions (80 input plus three generated), below both limits. This is a bounded operational difference, not a claim of identical configured capacity.
- Comparison evidence: external `official-gguf-comparison-256-v1.json`, 7,026 bytes, SHA-256 `2c54cd790aef5ddcf8b053923a7ebb18ef055e9b06b6b580abd2a1eb9b92f6fd`; `passed=true`, verdict `pass_with_bounded_differences`. It records exact matches, the prompt/context explanations, every unavailable field, evidence hashes, and both cleanup records.
- Recovery: the final postflight found no llama/model process, 46,353,580,032 available physical bytes, 50,132,758,528 bytes commit headroom, and 23,422 MiB GPU free.

The first attempt to nest the wrapper behind another `pwsh -File` invocation expanded the runner argument array before the wrapper could bind it. It failed before child creation and produced no model residency. The successful runs invoke the wrapper directly in the current PowerShell process with the tested named-parameter array binding; F-0031 records this operator pitfall.

## 2026-08-11 P3 Official 1.6B No-Model Admission Forecast

Completed P3's resource-planning slice without downloading or loading the absent checkpoint. The pinned target remains `LiquidAI/LFM2.5-VL-1.6B@919fde3d022e3f90a4716006f993938ee8c2eb97`. Official HEAD responses at that revision identify one 3,193,334,216-byte `model.safetensors` object with expected LFS SHA-256 `7fc7458e4382fc6e558cfdda45857fbf9ab5b40a8bf199c9cd073003b14ac26d`; local acquisition must rehash the complete file before that identity is accepted. The seven small pinned files bring the expected eight-file regular snapshot to 3,198,084,631 bytes. The local 1.6B Hugging Face cache was absent, so no implicit network fetch followed.

The already-locked payload-free header provides 82,400 header bytes, 589 tensor records, and canonical tensor-inventory SHA-256 `24728d0ed10229e788c5b9baf25e0cc6c92c93b9cdb12ebb252a3c140a861703`. Applying the official 2,048 text width, 1,152 vision width, 27 vision layers, and 4,608 projector input to the exact selected 450M trace inventory projects 51 tensors, 182,523,192 tensor-data bytes, and an approximately 182,530,856-byte safetensors trace. This is a deterministic shape forecast, not a generated target trace.

The exact 1.6B-to-450M safetensors ratio is 3.558093732. Scaling measured 450M Job peaks and then applying a separate 1.35 safety factor yields the following center/bound pairs: Python dry load 10,229,894,076 / 13,810,357,003 bytes; Python trace 17,671,426,800 / 23,856,426,180; native trace 7,544,628,860 / 10,185,248,961. The accepted first-attempt ceilings are stage-specific: 16 GiB, 24 GiB, and 12 GiB. They must run sequentially and cannot rise automatically after a limit termination. The Python trace requires at least 32 GiB of both available physical memory and commit headroom, plus zero llama/model/Cargo/rustc processes.

Disk planning allows for both a download cache and regular snapshot, two projected traces, and 1 GiB miscellaneous evidence margin: 7,834,972,798 bytes. Acquisition therefore requires at least 12 GiB free. The planning census retained 45,540,925,440 available physical bytes, 49,234,640,896 bytes commit headroom, 243,863,638,016 bytes disk free, 23,422 MiB GPU free, and zero model/build processes. External `p3-1.6b-resource-forecast-v1.json` is 5,587 bytes with SHA-256 `0c8f3cd31cea807591356d90aa442a2a02421e86a58215c01b4bcecc12659a59`; its verdict is `ready_for_guarded_acquisition_not_inference`.

## 2026-08-11 Repository Integrity and Evidence-Publication Closure

The current-tree review reconciled stale operator documentation without moving
the P3 product gate. `START_HERE.md` now points Gknome work to TODO C2,
`PARITY.md` records the completed P2 runtime result and 29-test example gate,
and the tokenizer config-only path can retain the remaining official image
wrapper/thumbnail/row-column IDs after the 1.6B snapshot is acquired. Marker
inspection rejects missing grid markers, conflicting or aliased marker IDs,
and IDs outside the model vocabulary; runtime IDs remain config-driven.

F-0036's acquisition lesson was applied to every reviewed durable writer.
Shared Python JSON/report publication, split-MMProj export, native trace
directories, and both PowerShell evidence writers now refuse existing or
racing destinations by default. Intentional replacement remains explicit.
This removed repeated temporary-file logic while preserving the existing CLI
and evidence schemas. No dependency, model payload, network action, secret
access, commit, push, or PR was added.

`summary_bank.json` was split at measured seams: native model math versus
checkpoint loading, reference fixture generation versus environment locking,
and GGUF artifact inspection versus production parity. A focused recurring
issue route now owns exclusive publication. PowerShell 7 and 5.1 validate all
routes below 256 KiB with no path fanning out to more than four groups.

The final bounded proof is 81/81 pinned Python reference tests, 29/29 native
LFM2-VL example tests, offline checks for the four affected libraries and three
examples, Python compileall/environment-lock validation, and PowerShell 7/5.1
bounded-owner and preflight smokes. The installed `NVIDIA-Workbench` WSL lane
has no `cargo`; an offline Windows-hosted Linux-target check also stopped at
`openssl-sys` because no Linux OpenSSL sysroot is configured. The new
Linux-specific native-trace collision regression is therefore truthfully
deferred to TODO C3; no network or toolchain install was substituted.

## 2026-08-11 — P3.4/P3.5 official 1.6B native CPU-F32 parity

- Reused the immutable eight-file snapshot, deterministic 256x256 gradient
  image (`f902f8d2e47e53eafac86831cfc692001dc15870eb81d57abc3128f048d2efca`),
  official rendered prompt, CPU F32, single crop, and three decode steps. The
  corrected release executable was 10,791,424 bytes with SHA-256
  `1f21125cdfe107a42a608920703755c499c7c75cae637b834724d78b175887e0`.
- The bounded native owner record
  `C:\DevStuff\candle-oracle\evidence\p3-1.6b-native-trace-f32-corrected-owner-20260811T214952Z.json`
  records PID 4788, exit 0, 29,486 ms, peak Job memory 6,845,521,920 bytes
  under the 12 GiB ceiling, and exact cleanup. The 786-byte combined log has
  SHA-256 `8da2c7137c0f5234bd5e46ca9621dbf5b0f6db75e220eb7cad2b69dc224991ac`.
- The external native bundle contains 51 tensors and 182,528,392 safetensors
  bytes, exact inputs, 80 input tokens, 64 projected image tokens, exact cache
  reset, and generated IDs `[1098, 4646, 40027]` (`The image depicts`).
- The phase-contract comparison report
  `C:\DevStuff\candle-oracle\evidence\comparison-1.6b-contract-v3-20260811T220300Z.json`
  has SHA-256 `9a0b16256a222678f9dce1282660e49fc6d19103cc6dd6a53c824bb58a6412c0`,
  `passed=true`, 51/51 tensors, and zero failures. CPU F32 acceptance keeps
  exact integer/input checks, uses a `<=2e-5` resized-position bound, a
  `<=1e-3` prefill-logit bound, and allclose-or-cosine (`>=0.99999`) for
  vision/projector/hidden-state stages. This records cross-kernel reduction
  drift explicitly; it is not an unbounded tolerance relaxation.
- The localized source fix routes SigLIP2 encoder pre-norms through Candle's
  stable two-pass F32 LayerNorm implementation, with a large-offset regression
  test. The F64 experiment was rejected because it increased the valid trace's
  failures and is not part of the retained source.
- The final clean postflight
  `C:\DevStuff\candle-oracle\evidence\p3-close-final-postflight-clean-20260811T221143Z.json`
  has SHA-256 `1f4399ed6bfbbbf6c6b400054c0cbfebac6fcc8c28ef4d204fdcefbb6fdc4030`,
  zero tracked model/build processes, 44,067,688,448 available physical
  bytes, 49,131,601,920 bytes commit headroom, and 23,463 MiB GPU free.
- Local focused verification passed: `cargo fmt --all -- --check`,
  `cargo test --locked --offline -j 2 -p candle-transformers siglip2 --
  --nocapture` (9/9), affected `cargo check` for `candle-transformers` and
  the `lfm2-vl` example, and pinned `pytest tools/lfm2_vl/reference -q`
  (82/82). P3 is closed; P4.1 is the next active task.

## 2026-08-11 — P4.1 public CPU-text device policy

- Added `--text-cpu` to the native and hybrid LFM2-VL example command forms.
  The existing `main.rs` policy consumer now resolves CPU text with the normal
  selected vision device, while `--cpu` remains authoritative for both
  components.
- The documented matrix is accelerator/accelerator by default,
  accelerator/CPU with `--vision-cpu`, CPU/accelerator with `--text-cpu`, and
  CPU/CPU with `--cpu`. Existing flags and report schemas remain unchanged.
- Added focused parser/policy coverage for all four placements, `--cpu`
  precedence, help exposure, and the controlled trace-lane rejection when
  `--text-cpu` is used without `--cpu`.
- The example README now documents the CPU-text/accelerator-vision command and
  placement matrix. The focused argument suite passed 12/12 and the locked,
  offline LFM2-VL example check passed. No model or CUDA runtime was started.
- P4.1 is complete when the public route is selectable, prior flags retain
  behavior, tests and help agree, and the affected example compiles; those
  conditions are now met. P4.2 is the next active gate.

## 2026-08-11 — P4.2 bounded native CUDA/distinct-device proof

- Native Windows CUDA identity was recorded as `nvcc` 13.3.33, Cargo/rustc
  1.91.0, RTX 4090 driver `32.0.16.1088`, and an MSVC target. The first bounded
  compile exposed a CUDA 13.3 CCCL requirement for MSVC's conforming
  preprocessor; `candle-kernels/build.rs` now passes
  `-Xcompiler /Zc:preprocessor` to both PTX and static-library builds.
- The corrected bounded native-loader owner passed the existing CUDA-gated
  distinct-device test 1/1. Owner evidence is
  `C:\DevStuff\candle-oracle\evidence\p4-2-native-cuda-distinct-owner-20260811T183000Z.json`
  (SHA-256 `57bd3b15081c61c3b1e64ff24d0dabbb2c344dac033446751daad0849d237de7`),
  with log SHA-256
  `bcc59ec9dca523955aefdcfdd1d1668317e2a2dd6b0d5e3b9db2006d35de6cd4`,
  2,691,182,592-byte peak Job memory under 16 GiB, and exact PID cleanup.
- The companion transformer test
  `split_vision_cuda_text_cpu_transfers_only_projected_features` passed 1/1,
  proving CUDA vision, CPU text, projected-feature-only transfer, and hybrid
  prefill agreement (`max_abs=4.456564784e-5`). Its owner SHA-256 is
  `ca73e16f06d30396497e2500061229e59ea1b93e5c849339d73fed04839f7227` and
  log SHA-256 is
  `1b1806537b1bbb4838cdbad16b7f88f02122537951bc065aed64d6c1e88dd3e6`.
- Final postflight
  `C:\DevStuff\candle-oracle\evidence\p4-2-postflight-clean-20260811T183800Z.json`
  (SHA-256 `76b33d493f2b82cd00b253acc74bd31a21478cf0a70daef746e767707abcf7aa`)
  recorded zero tracked/llama processes, 43,408,338,944 available physical
  bytes, 47,523,815,424 bytes commit headroom, and 23,421 MiB GPU free.
- P4.2 is complete when the toolkit builds, both distinct-device tests pass,
  only projected features cross devices, the bounded owner exits cleanly, and
  host/GPU resources recover; those conditions are met. P4.3 is the next gate.

## 2026-08-11 — P4.3 official 450M CUDA parity

- What: Proved the admitted official 450M native checkpoint on all-CUDA F32,
  CPU-text/CUDA-vision F32, and all-CUDA BF16 placements under sequential
  bounded Windows owners.
- Why: Close the production CUDA parity gate after the tiny distinct-device
  proof, while preserving a truthful CPU-BF16 boundary and exact cleanup.
- When: 2026-08-11, after the P4.2 green gate and final executable rebuild.
- Where: `candle-kernels/src/cast.cu`,
  `candle-core/tests/custom_op_tests.rs`,
  `candle-transformers/src/models/lfm2_vl/linear.rs`, and
  `candle-examples/examples/lfm2-vl/{main.rs,args.rs}`.
- How: Registered the missing CUDA `cast_i32_f32` kernel, materialized dense
  linear inputs before CUDA matmul, made `--text-cpu` create CUDA vision
  independently, and rejected explicit BF16 on CPU components before model
  loading. The final executable SHA-256 is
  `5b147767e5c45074035d884eaa0b1111ee0ebc6dbf5ed098ee8f120539a8a669`.
- Evidence: all-CUDA F32 exited 0 with peak Job memory 3,474,706,432 bytes;
  CPU-text/CUDA-vision F32 exited 0 at 3,241,332,736 bytes; all-CUDA BF16
  exited 0 at 2,783,182,848 bytes. Both F32 routes generated
  `[1098, 4646, 5251]`, projected 64 image tokens, matched all baseline top-k
  IDs, and reset cache exactly. Guarded CPU-BF16 rejection exited 1 before
  model load with peak Job memory 772,743,168 bytes. Every owner PID was
  absent after cleanup.
- Done when: Official 450M all-CUDA F32/BF16 and CPU-text/CUDA-vision F32
  execute with parity evidence, the unsupported CPU-BF16 case fails early,
  and no model/build process remains. These conditions are met; P4.4 is next.
- Verification: bounded owner records/log hashes are recorded in `STATUS.md`
  and `PARITY.md`; targeted CUDA cast/linear regressions passed 1/1 each;
  argument tests passed 13/13; final CUDA release build passed; no secrets or
  model weights were added to the repository.

## 2026-08-11 — P4.4 diagnostic timing baseline

- What: Added an opt-in `--timings` diagnostic and captured a bounded,
  sequential all-CUDA F32 timing baseline for the official 450M checkpoint.
- Why: P4.4 requires measured optimization; generation must be isolated from
  model load, preprocessing, vision, and the intentional cache-reset replay
  before changing a hot path.
- When: 2026-08-11, after the green P4.3 CUDA parity gate.
- Where: `candle-examples/examples/lfm2-vl/{args.rs,main.rs,runner.rs,README.md}`
  and the P4.4 status/parity records.
- How: Report stage durations only to stderr, preserve the versioned JSON
  evidence, rebuild under a 16 GiB Job ceiling, and run the exact 450M fixture
  three times sequentially. Model load was 435.885–452.501 ms, vision
  38.037–39.186 ms, first generation 446.959–469.142 ms, cache-reset replay
  419.065–444.422 ms, and total inference 1,388.056–1,462.027 ms.
- Done when: The diagnostic is documented, help-visible, parser/full-example
  tests pass, the bounded artifact and owner evidence are recorded, and JSON
  output remains schema-stable. These conditions are met; the optimization
  itself is not claimed until the decode/cache microbenchmark is repeatable.
- Verification: `cargo fmt --all -- --check`; full LFM2-VL example tests
  32/32; bounded offline CUDA release rebuild exit 0; `--help` smoke; three
  sequential all-CUDA F32 owners exit 0 with exact PID cleanup; no model
  weights, caches, or secrets entered the repository.

## 2026-08-11 — P4.4 benchmark audit and rejected candidate

- What: Audited a one-line-per-token ShortConv weight-view cache candidate
  against the exact official prompt and reverted it when the comparison was
  not cleanly attributable.
- Why: A performance change is not complete until the measured improvement is
  reproducible under the same prompt, artifact, host, and resource contract.
- When: 2026-08-11, during the first P4.4 decode/cache probe.
- Where: Candidate seam was `candle-transformers/src/models/lfm2.rs`; the
  retained code is unchanged. The pitfall is recorded as F-0047.
- How: The initial long-token series accidentally used a literal PowerShell
  backtick-newline prompt. The corrected official prompt preserved generated
  IDs and the prefill-logit hash, but concurrent EdgeSymbio Cargo/rustc work
  contaminated the timing window and the post-change sample was slower.
- Done when: The candidate is either proven by a quiet exact-prompt series or
  removed, and the active TODO states the remaining proof. Removal and the
  explicit rejection are complete; P4.4 itself remains active.
- Verification: `cargo fmt --all -- --check`; source diff confirms no retained
  `lfm2.rs` change; corrected owners exited 0 with exact PID cleanup; F-0041
  and F-0047 record the prompt/host controls.

## 2026-08-11 — C3 WSL trace-publication replay

- What: Replayed the Linux-specific native trace destination-race regression in
  the explicit `NVIDIA-Workbench` WSL2 distribution.
- Why: Close the secondary no-clobber portability task without installing a
  toolchain or treating hosted CI as evidence.
- When: 2026-08-11, after the WSL distribution exposed an existing Rust
  toolchain and the native Windows 32-test example lane was green.
- Where: `candle-examples/examples/lfm2-vl/trace.rs`, using the existing
  `trace::tests::trace_publication_does_not_replace_a_racing_directory` test.
- How: Ran `cargo test --locked --offline -j 2 -p candle-examples --example
  lfm2-vl trace::tests::trace_publication_does_not_replace_a_racing_directory`
  with `CARGO_TARGET_DIR=/home/workbench/code/candle-lfm2-vl/target`. Cargo and
  rustc were 1.97.1 on Linux 6.6.87.2-WSL2; the test passed 1/1 and no
  temporary `candle-*` directory remained.
- Done when: The exact Linux test compiles and passes without replacing the
  competing directory, temporary output is absent, and the native Windows
  example gate remains green. These conditions are met; C3 is complete.
- Verification: WSL exact test exit 0; temporary-path inventory; native
  Windows `cargo test --locked --offline -j 2 -p candle-examples --example
  lfm2-vl` 32/32; WSL manifest verification; `git diff --check`.

## 2026-08-11 — C1 modular source layout

- What: Split the largest LFM2-VL production files into bounded same-module
  source units while keeping wrappers, private name resolution, public APIs,
  evidence schemas, and tests stable.
- Why: Reduce context and merge pressure without introducing a generic VLM
  abstraction or changing runtime behavior.
- When: 2026-08-11, after the existing P3/P4 parity gates and during the
  owner-reviewed main integration.
- Where: LFM2 text/cache, SigLIP2, GGUF, weights, native composite, processor,
  prompt, runner, and native-loading wrappers plus their `include!` parts.
- How: Moved only proven responsibility seams into the new source units,
  retained tests in wrappers, added `MODULE_LAYOUT.md` and
  `verify-module-layout.py`, routed the new files through `summary_bank.json`,
  and preserved our stable LayerNorm and timing diagnostics at their new
  seams.
- Done when: Every wrapper/part stays within the documented size limits,
  include inventories match, compilation and focused/full tests remain green,
  and the mod manifest accounts for every new path. These conditions are met;
  C1 is complete.
- Verification: module-layout verifier passed; Windows locked/offline library
  and example checks passed; the LFM2-VL example suite passed 32/32; the
  focused SigLIP2 stable-LayerNorm test passed 1/1; summary-bank and
  mod-manifest verifiers passed.

## 2026-08-11 — S4 release-integrity closeout slice

- What: Closed the self-mutating one-shot backlog workflow from the publication
  tree, synchronized the opt-in timing diagnostic around CUDA device work, and
  moved the remaining performance experiment to a future PERF-1 backlog item.
- Why: A write-enabled workflow that edits and pushes project state is not a
  safe release control, and unsynchronized stage timings can include queued
  work without making that boundary visible.
- Where: `.github/workflows/lfm2-vl-backlog-harness.yml`,
  `candle-examples/examples/lfm2-vl/runner/{run,runtime}.rs`, `main.rs`,
  `docs/lfm2-vl/{TODO,STATUS,PARITY,START_HERE}.md`, and the release docs.
- How: Deleted the workflow rather than repairing or disabling it; added
  resolved-device synchronization before and after timed model stages and
  labeled the output `sync=cuda-device-complete`; kept no speculative source
  optimization after the noisy P4.4 measurements.
- Done when: No current source references the workflow, timing output states
  its synchronization boundary, JSON evidence is unchanged, and P4.4 has no
  unproven optimization claim. These conditions are met locally.
- Verification: `cargo fmt --all -- --check`; locked/offline package and
  example checks; the LFM2-VL example suite 34/34; PowerShell summary-bank;
  WSL mod-manifest; WSL `git diff --check`; and strict example Clippy all pass.
  The temporary branch head tree is byte-identical to squashed main commit
  `6ea6aef5`; it was not deleted because the guarded helper is intentionally
  main-only and remote deletion needs its own explicit authorization path.

## 2026-08-11 — Resolved device/dtype guard

- What: Changed the LFM2-VL CLI guard to validate the actual resolved vision
  and text `Device` values and reject both BF16 and F16 on CPU, with tests for
  BF16, F16, and accelerator-helper CPU fallback.
- Why: Policy flags describe intent, but a requested accelerator can resolve to
  CPU. A policy-only guard could therefore let an unsupported low-precision
  dtype reach a CPU matmul path.
- Where: `candle-examples/examples/lfm2-vl/args.rs` and `main.rs`.
- How: Pass the resolved devices into one validation function, use
  `Device::is_cpu()`, and return an actionable component/dtype error before any
  model load.
- Done when: CPU F32 remains accepted, CPU BF16/F16 fail before loading, and
  fallback is covered by a deterministic unit test. The source/tests are
  complete; official CUDA F16 and CUDA-text/CPU-vision F32 parity remain P4.5.
- Verification: Focused argument tests 16/16 and the full LFM2-VL example
  suite 34/34 pass; no production model run was started.

## 2026-08-12 — P4.5 complete resolved device/dtype matrix

- What: Closed every advertised native 450M placement/dtype route: CPU/CPU
  F32, all-CUDA F32/BF16/F16, CPU-text/CUDA-vision F32, and
  CUDA-text/CPU-vision F32. CPU components remain explicitly F32-only.
- Why: CUDA F16 and CUDA-text/CPU-vision F32 were the last unproven public
  routes; policy flags alone could not establish resolved-device behavior.
- When: After artifact rehash and an exclusive, clean host census with no
  model, Python, Cargo, rustc, or llama process.
- Where: The native `lfm2-vl` release example, resolved-device guards,
  bounded owner records under `C:\DevStuff\candle-oracle\evidence`, and the
  support-matrix documentation.
- How: Rebuilt the CUDA release offline under a 16 GiB Job limit, then ran all
  six routes sequentially under 12 GiB limits over the pinned 450M artifact,
  exact rendered prompt, and deterministic image. Every route produced IDs
  `[1098, 4646, 5251]`, the same prefill top-5 ID order, 64 image tokens, exact
  cache replay, exit 0, and PID absence. Peak Job memory ranged from
  2,412,109,824 to 3,474,620,416 bytes.
- Done when: Every advertised route has parity evidence or an explicit early
  rejection, CPU BF16/F16 fallback is guarded, and resource cleanup is exact.
  These conditions are met; P4.5 is complete.
- Verification: Artifact manifest SHA-256
  `659c8421530586b6cc28c094cfcdc69719ea8626f2abc0efd9eec4ac2a68a984`;
  release executable 65,076,736 bytes / SHA-256
  `7a9261f6808b09ffab0963f5c015661c515534c6b949ac4893f4fa8cbe0023a2`;
  six audited owner/log pairs; exact model/image/prompt/token-span checks; and
  owner-exit cleanup for every run.

## 2026-08-12 — PERF-1 isolated generation baseline

- What: Added and executed a dedicated `--benchmark-generation` lane for the
  native runner without changing `candle-lfm2-vl-inference-v1`.
- Why: End-to-end stage diagnostics included load/evidence noise and could not
  justify retaining the earlier speculative ShortConv change.
- When: After P4.5 on an empty model/build/Python census.
- Where: `runner/benchmark.rs`, CLI/runner guards and tests, the example
  README, and external bounded evidence
  `perf-1-cuda-f32-baseline-20260812T020500Z*`.
- How: Ran 10 warm-ups and 30 measured all-CUDA F32 direct
  prefill/greedy/cached-decode iterations, synchronized devices, excluded
  cache reset and evidence/tokenizer work from the timed region, and required
  exact IDs plus no more than 5% relative MAD.
- Done when: A reproducible baseline plus unchanged parity/resource evidence
  either proves a candidate's at least 10% gain or rejects it. Median was
  458.0633 ms, MAD 9.82745 ms, relative MAD 2.1454%, IDs remained
  `[1098, 4646, 5251]`, and no candidate met the retention threshold; the
  baseline is retained and PERF-1 is complete.
- Verification: Benchmark log SHA-256
  `2660209b7adc9b26eb204d50a165d78f85145b84532f8922b6fbdadc72a6e541`;
  owner SHA-256
  `a6f15d637ed340de9c7aa2188ae9e64c2e13964708244e7c0afcb7a017dd9066`;
  exit 0, 3,475,435,520 peak Job bytes, PID absent, and inference cache replay
  exact.

## 2026-08-12 — Cross-platform integrity-verifier closeout

- What: Fixed native Windows module-layout verification and reduced one
  context route below its existing 256 KiB ceiling.
- Why: `Path` stringification emitted backslashes on Windows while the
  canonical include inventory uses repository-style slashes; the
  exclusive-publication route also loaded the entire pitfall ledger and grew
  to 258 KiB.
- When: During the final post-PERF integrity audit, before publication.
- Where: `scripts/lfm2-vl/verify-module-layout.py` and `summary_bank.json`.
- How: Normalize discovered repository-relative paths with `as_posix()` and
  route the exclusive-publication group to the smaller accepted D-0042
  decision contract instead of the complete failure ledger.
- Done when: Native Windows module-layout verification passes, both PowerShell
  summary-bank lanes pass, and all routes remain below 256 KiB. These
  conditions are met; the affected route is 215.5 KiB.
- Verification: Native bundled Python module-layout verifier passed all nine
  split wrappers; PowerShell 7 and Windows PowerShell 5.1 summary-bank
  verifiers passed with the 139.0 KiB default orientation pack.

## 2026-08-12 — MVP release discoverability and support contract

- What: Completed the public LFM2.5-VL entry, crate-level orientation, exact
  device/dtype matrix, proof-level boundary, and zero-active-task handoff.
- Why: The implementation and evidence were complete, but users should not
  need historical parity documents to discover the example or distinguish
  proven, fixture-protected, rejected, and deferred behavior.
- When: After the complete P4.5 production matrix and PERF-1 baseline.
- Where: Root `README.md`, `candle-vlm/README.md`, the detailed example
  README, `START_HERE.md`, `STATUS.md`, `PARITY.md`, `TODO.md`, and
  `summary_bank.json`.
- How: Added the exact native/GGUF/MMProj discovery entry, documented the
  crate's no-network and checked-allocation boundary, published the 4x3
  text/vision/dtype matrix, kept future features outside the MVP backlog, and
  routed benchmark code only through parity/CUDA context groups.
- Done when: Public docs agree on one feature-complete MVP release candidate,
  CPU low precision fails early, all advertised production routes are proven,
  deferred work is not a release promise, and TODO has no active MVP item.
  These conditions are met.
- Verification: Relative links pass across all 23 mod-owned Markdown files;
  summary-bank routes pass both PowerShell versions below 256 KiB; the
  141-path mod manifest and WSL `git diff --check` pass.

## 2026-08-12 — Final local MVP gate and status consolidation

- What: Ran the final native Windows source gate, closed the review's exact
  CUDA device/dtype unit matrix, fixed the only mod-owned workspace Clippy
  finding, and reduced `STATUS.md` to current handoff state.
- Why: The release tag must follow a warning-clean, locally proven source tree,
  while completed phase narratives belong in `HISTORY.md`/`PARITY.md` rather
  than the live status document.
- When: After the complete six-route production matrix and isolated PERF-1
  baseline, without rerunning unchanged 1.6B model math.
- Where: The LFM2-VL example tests/benchmark, native Windows workspace gates,
  `STATUS.md`, and external preflight evidence
  `release-closeout-preflight-20260812T025925Z.json`.
- How: Added CUDA-gated assertions for all-CUDA F16 acceptance and BF16/F16
  rejection on either resolved CPU side; used `is_multiple_of(2)` for the
  benchmark median helper; ran the exact workspace Clippy target with
  `PYO3_NO_PYTHON=1` because the unrelated ABI3 binding requires Python 3.13;
  and kept only current truth in status.
- Done when: Formatting, full workspace strict Clippy, affected checks/tests,
  full example tests, CUDA regressions, module/context/manifest/link gates, and
  diff review are green with no active MVP product task. The local source gate
  is complete; remote branch deletion and main/tag publication remain separate
  explicit remote-hygiene steps.
- Verification: Example 36/36; CUDA device/dtype matrix 1/1; transformer 59/59
  plus generation 5/5 and NMS 8/8; VLM 29/29; complete core lanes; CUDA cast
  1/1; CUDA dense-linear 1/1; full workspace Clippy with `-D warnings`;
  module-layout nine wrappers; cross-version summary/preflight smoke; 23-file
  relative-link audit; 141/14/127 mod manifest; and clean `git diff --check`.

## 2026-08-12 — Guarded annotated release-tag publication

- What: Extended the ignored direct-main publication helper with a narrow
  annotated LFM2-VL MVP tag mode and recorded its durable policy.
- Why: The release contract requires an annotated tag, while raw Git or a
  second ad hoc credential path would bypass the existing remote, ancestry,
  cleanliness, and secret-containment checks.
- When: After implementation closeout commit `7601b766` was published and
  verified on `origin/main`.
- Where: `.tools/gitpush.ps1` locally; durable policy in `AGENTS.md`,
  `START_HERE.md`, `DECISIONS.md`, `STATUS.md`, and this history record.
- How: Permit only `lfm2-vl-mvp-X.Y.Z`; require a clean named `main`, exact
  expected HEAD, remote `main` equality, an annotated local tag that peels to
  HEAD, and an absent-or-identical remote tag; push one exact tag ref without
  force and verify both its remote object and peeled commit. Branch deletion
  remains unsupported.
- Done when: Main and tag use one authenticated helper, conflicting or
  lightweight tags fail closed, no unrelated ref can be selected, and the
  remote tag verifies at the final main commit.
- Verification: PowerShell 7/5.1 syntax, guarded main/tag dry runs, exact local
  tag type/peel checks, local documentation/context/manifest gates, and final
  remote-ref verification.

## 2026-08-12 — Final remote hygiene and MVP snapshot

- What: Removed the proven temporary agent branch, reconciled current-state
  documentation, and recreated the annotated MVP tag at final clean `main`.
- Why: The release contract requires no temporary closeout infrastructure and
  requires the remote clean head and immutable tag to identify the same state.
- When: After all implementation, six-route production parity, PERF-1, local
  source gates, release documentation, and exact owner authorization were
  complete.
- Where: Remote branch `agent/lfm2-vl-backlog-closeout`, annotated tag
  `lfm2-vl-mvp-0.1.0`, `STATUS.md`, `FAILURE_LOG.md`, and this history record.
- How: Re-fetched and pinned branch head `52342156`, proved tree `cf30d53a`
  equal to integrated commit `6ea6aef5`, deleted only that branch, retained
  `feat/lfm2-vl-mmproj`, verified the old annotated tag object before its
  approved removal, then fast-forwarded reviewed docs and recreated the same
  annotated tag at exact final `main` without force.
- Done when: The temporary branch and workflow are absent; the historical
  branch remains; TODO has zero active release items; worktree and remote main
  are clean; and the annotated tag peels to exact remote main. These conditions
  are met after final publication.
- Verification: Identity-pinned dry runs, exact approval-phrase guards,
  cross-version summary-bank checks, relative-link and mod-manifest gates,
  staged/committed diff checks, guarded fast-forward main/tag publication, and
  final `ls-remote` equality/absence checks.

## 2026-08-12 — Post-snapshot benchmark and context integrity

- What: Hardened the isolated generation benchmark and reconciled current
  entry-point, backlog, status, and context-bank state after MVP publication.
- Why: A request for two tokens does not guarantee two generated IDs when EOS
  occurs first, so the old lane could report a prefill-only timing as a cached
  decode benchmark. Zero-duration samples could also be labeled stable, and
  `START_HERE.md` still described the already-tagged snapshot as a candidate.
- When: During the first post-snapshot repository-integrity pass.
- Where: The LFM2-VL runner benchmark and tests, example README,
  `START_HERE.md`, `STATUS.md`, `TODO.md`, `HISTORY.md`, `FAILURE_LOG.md`, and
  `summary_bank.json`.
- How: Require the baseline to contain at least two generated IDs before any
  warm-up, accept only positive finite durations, keep the output schema
  unchanged, replace stale release wording, move existing context ownership
  instead of adding an overlapping group. Recheck the ignored pinned Python
  environment outside the managed sandbox after the sandbox denied access to
  its external base interpreter.
- Done when: Early-EOS and zero-duration cases fail explicitly; full example
  and affected library tests, formatting, strict targeted Clippy, import and
  module checks, both summary-bank lanes, script smokes, links, JSON, manifest,
  shell syntax, and diff checks pass; no model process is launched. These
  conditions are met for the Rust/documentation slice.
- Verification: Benchmark regression 1/1; LFM2-VL example 36/36; transformer
  59/59; VLM 29/29; core 21/21; locked/offline affected checks and strict
  example Clippy; nine module wrappers; PowerShell preflight and bounded-owner
  smokes; 23 Markdown documents/50 relative links; nine JSON files; and WSL
  141/14/127 manifest proof. The exact native Windows reference-environment
  verifier passed Python 3.10.11, all 42 locked distributions, and the pinned
  Transformers revision; the complete pytest suite passed 82/82 in 38.80s.

## 2026-08-12 — Hash-pinned fixture checkout portability

- What: Reproduced and fixed Windows checkout-time mutation of deterministic
  fixture text, extended the manifest verifier, and completed a focused
  repository-integrity audit around the affected provenance boundary.
- Why: A clone with `core.autocrlf=true` converted LF JSON to CRLF and changed
  an exact split-MMProj hash. That made a valid repository revision fail its
  own loader on a normal supported Windows configuration.
- When: Post-MVP maintenance at reviewed parent `2a01aaf0`; no product phase,
  model inference, remote ref, or immutable tag changed.
- Where: Root `.gitattributes`, all three committed LFM2-VL fixture families,
  `verify-mod-manifest.sh`, the split fixture README, current state/backlog,
  D-0050, F-0052, `MOD_MANIFEST.md`, and `summary_bank.json`.
- How: Pin fixture JSON/Markdown to LF and safetensors to `-text`; keep runtime
  and manifest hashing exact; dynamically enumerate and validate 10 text plus
  three binary fixtures; reproduce the old bytes in one native clone and prove
  the fix in a second fresh native clone using the same Git setting.
- Done when: Clean Windows checkout bytes and split hashes are exact; malformed
  identity still fails rather than being normalized; exact split tests and the
  broad local source gate pass; the owner-reviewed direct-main release is
  clean locally and remotely; no production stub, broken export, overlapping
  context route, active maintenance task, or temporary proof file remains.
- Verification: Unfixed 524-byte LF to 553-byte CRLF reproduction; fixed 10/10
  LF text and 3/3 `-text` binary inventory; exact split identity and hybrid
  tests; transformer 59/59; VLM 29/29; complete native Windows locked/offline
  workspace tests; strict workspace Clippy with `-D warnings`; current-root
  formatting and affected checks; both summary-bank lanes; nine module-layout
  wrappers; 23-document/50-link, 11-JSON, and 16-Python syntax audits; an
  unknown-extension negative verifier probe followed by a clean positive
  inventory; guarded helper dry run and publication; exact remote-main
  equality; no model load.

## 2026-08-12 — Three-repository Round 1 consumer boundary

- What: Established independent Candle fork overlays and promoted complete
  local LFM2-VL hybrid construction from the example into the public
  `candle-vlm` library.
- Why: EdgeSymbio must consume a stable framework API without copying example
  code, while later SnapFlash-derived diffusion work must remain independently
  reviewable and must not contaminate the immutable LFM2-VL release history.
- When: Before EdgeSymbio dependency pinning or proof-only 450M integration and
  before any shared SDXL LoRA promotion.
- Where: `candle-vlm/src/lfm2_vl/loading.rs`, the LFM2-VL example adapter,
  deterministic loader fixtures, `docs/FORK_OVERLAYS.md`, both overlay
  manifests, the root union verifier, current-state docs, and
  `summary_bank.json`.
- How: Added explicit local-only source/options/result types; retained split,
  direct dense, and native Q8 execution policy; returned exact consumed paths;
  moved all unique hybrid assembly into the library; kept hashing, retained
  handles, resource admission, proof JSON, discovery, and download policy in
  applications. Registered shared overlay paths and made union completeness a
  publication gate while preserving an independently runnable LFM2-VL
  verifier.
- Done when: The example owns no duplicate loader, all three hybrid forms load
  through the public API, generated inputs are byte/hash pinned, LFM2-VL and
  SnapFlash-derived paths are independently attributable, and the complete
  local Candle gate remains green. These conditions are met.
- Verification: Native Windows formatting; VLM 35/35; LFM2-VL example 32/32;
  strict targeted and full-workspace Clippy with `-D warnings`; complete
  locked/offline workspace tests and doc tests; LFM2-VL manifest 150/15/135;
  root overlay union 153 paths/two overlays/five shared paths; summary-bank 23
  groups; exact diff checks. The expanded workspace gate used the installed
  Python 3.13 required by the unrelated ABI3 crate and performed no download,
  model inference, llama.cpp execution, or concurrent large-model work.

## 2026-08-12 — EdgeSymbio Round 2 CPU/F32 consumer acceptance

- What: EdgeSymbio pinned all Candle packages to Round 1 commit
  `c0fb3a9fe098e50d07ec1b749c77015d7bd8d9a5`, added an isolated CLI-only
  LFM2-VL 450M adapter, admitted five exact official/owner fixture files, and
  published the completed consumer at
  `d535a4f56f5a8e06407cb4b8f5be0df7f3121327`.
- Why: A product consumer had to prove the public hybrid loader together with
  retained-file identity, resource leasing, cancellation, evidence, exact
  cache reset, and no hidden product/API exposure.
- When: After Candle Round 1 and before shared SDXL LoRA promotion, CUDA,
  public chat attachments, RAG captioning, or SnapFlash integration.
- Where: EdgeSymbio's backend LFM2-VL adapter and runtime proof, explicit CLI
  registry, exact proof manifest, dependency/lock guards, state docs, pitfall
  ledger, review, and summary bank.
- How: Restricted the proof to CPU/F32, one 256x256 PNG, the official rendered
  prompt, and three generated tokens; ran it inside the 8 GiB/300-second Candle
  Windows Job Object; compared exact token/text/image/span/stop behavior and
  two generations around `clear_cache()`; retained the reference and
  Edge-observed prefill hashes as distinct evidence rather than treating
  independently linked binaries as a bitwise-logit acceptance contract.
- Done when: Exact IDs `[1098, 4646, 5251]`, text `The image features`, one
  whole crop, 64 projected tokens, span `[5,69)`, max-token stop, and exact
  in-process reset replay pass; every PID and lease is released; normal
  text-only/API/Tauri/release surfaces remain unchanged; local and remote main
  are equal. These conditions are met.
- Verification: Corrected release proof exited in 8.510 seconds with 1.111 GiB
  peak Job memory and PID cleanup. Edge's observed prefill hash
  `460fc9d4e2be1ad2687d066faf935455f96124d014f146c753bdfd0e3a803610`
  differed from Candle's reference
  `f84844259d6001d3701df6e3a9602fb9cbc2e6db03e3c27cefab81ca7daec2d7`;
  no source/dependency/feature drift was found, so the numerical cause remains
  unproven and the mismatch is recorded without a bitwise claim. The complete
  local gate passed backend 805, API 208, CLI 115/115, desktop 22, fetch 5,
  Playwright 16, Python 111, and context checks twice. No third model run was
  performed.

## 2026-08-12 — Candle Round 3 three-component SDXL LoRA transaction

- What: Added generic public LoRA parsing, component/target evidence, and one
  rollback-capable mutable transaction across SDXL UNet, text encoder 1, and
  text encoder 2.
- Why: SnapFlash-Server and EdgeSymbio duplicated pair parsing, delta math,
  immutable-base replacement, and rollback. Candle is the reusable owner, but
  application naming, files, licensing, reports, and orchestration must remain
  in the consumers.
- When: After Edge CPU/F32 acceptance and before either consumer's LoRA
  dependency migration or any ControlNet/inpainting promotion.
- Where: `stable_diffusion/lora.rs`, `stable_diffusion/mutable.rs`, their module
  export, the SnapFlash-derived manifest/verifier, overlay registry, decision,
  changelog, current state/backlog, and focused summary-bank route.
- How: Parse every safetensor name into an explicit component and paired
  up/down/alpha record; inject target resolution; validate shapes, devices,
  dtypes, finite values, rank, alpha, strength, target uniqueness, and effect;
  retain independent bases; construct revision-bound plans from base; snapshot
  and revalidate all live targets; apply all writes; roll back in reverse on
  failure; reject revision exhaustion before mutation; require the application
  to retain its exclusive model lease; hash shape plus canonical F32 values
  for base/delta/merged evidence.
- Done when: UNet-only, both text-only, and mixed adapters pass; A -> B is from
  base; clear is exact; failures in components 2 and 3 restore all prior
  writes; malformed, zero, unsupported, duplicate/unmatched, non-finite, and
  stale inputs and revision exhaustion fail closed; BF16 1x1 and evidence
  hashes pass; all local and overlay gates pass. These implementation
  conditions are met.
- Verification: Focused Stable Diffusion 12/12; complete transformer 71/71
  plus 5 generation and 8 NMS tests; strict all-target transformer Clippy;
  independent SnapFlash-derived manifest 8/2/6; root overlay union 157 paths,
  two overlays, five registered shared paths; summary bank 24 groups with a
  72.2 KiB focused LoRA route. The native workspace gate passed with only the
  pre-existing live-HTTP dataset test explicitly excluded; its crate check
  passed, its exact socket denial is F-0053, and strict full-workspace Clippy
  remained green. The blocked probe transferred no network data; no model,
  CUDA workload, Python oracle, llama.cpp process, or production checkpoint was
  loaded.

---
AI-edited: 2026-08-12T16:05:00-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=three-repo-round-3 | change=archived Edge acceptance, the audited LoRA implementation, and its local workspace gate
