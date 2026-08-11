# LFM2.5-VL Status

## Baseline and Publication

- Model and compatibility baseline: Candle 0.11.0 at `31f35b147389700ed2a178ee66a91c3cc25cc80d`.
- Upstream integration base: Candle main at `6f74e7c390c717f8fd34f23ce02aceb058173370`.
- Integration and publication branch: `main` at `https://github.com/Shoozes/candle.git`.
- Historical feature checkpoint: `c9b60f0b906fa8fe70423295e2e1164648a8fa53` on `feat/lfm2-vl-mmproj`.
- Current code checkpoint: `a83acf13d2b6bff6528e8b8c87209500f6fbc85c`; upstream-preserving merge checkpoint: `2b1d9e80de06b251b2fe5f25e51c17d56db86591`.
- Pull request: none.
- Current mod overlay relative to the upstream integration base: 91 allowlisted paths, exactly 9 fork-origin modifications and 82 mod-owned additions. The 29 inherited post-0.11 upstream paths remain outside the mod overlay.

## Worktree Boundary

- Native Windows/MSVC is the product and primary proof lane; WSL2/Linux is a secondary portability replay.
- `C:\DevStuff\candle-mods` is a WSL-owned linked worktree attached to local `main`; its `.git` file points to Linux-owned metadata under `/home/workbench/code/candle-lfm2-vl`.
- Windows Git cannot resolve that pointer. Use the explicit `NVIDIA-Workbench` WSL distribution for all Git operations. The historical feature branch remains in the Linux-home worktree and must not be attached here.
- Owner-reviewed work lands directly on `main` without a PR. Broad staging, force-push, implicit merge/rebase, and secret inspection remain prohibited; the ignored `.tools/gitpush.ps1` only verifies and pushes an already clean fast-forward branch after explicit approval.

## Current Phase

- Product phase: post-Phase 7 production stabilization.
- NR-5B official 450M native Windows CPU-F32 component parity is green.
- P2 official-base GGUF same-artifact comparison is green at every stable field exposed by both runtimes.
- Current product gate: P3 official 1.6B native Windows CPU-F32 component parity.
- P3's no-model remote artifact inventory, memory/trace forecast, guarded acquisition owner, and read-only acquisition plan are green. The exact next step is the separately approved external download; no 1.6B model run is admitted before local full-file hashes and a fresh preflight pass.
- Following gate: native Windows CUDA/distinct-device execution. WSL replay remains optional and secondary.

## Last Green Verification

### Official 450M production gate

- Model: `LiquidAI/LFM2.5-VL-450M@fc6221ca597f3315e4f82fc2df606783267b34ba`.
- Reference environment: official Windows Python 3.10.11; exact 42-distribution lock; runtime/test/VCS verifier green; reference suite 43/43.
- Artifact: external eight-file regular snapshot, 902,236,184 bytes; hash-only manifest SHA-256 `659c8421530586b6cc28c094cfcdc69719ea8626f2abc0efd9eec4ac2a68a984`.
- Python trace: 36 tensors; manifest SHA-256 `41f97daf914bd2c3eea81065ca87f1b002e869dd0dcedf010bba229646529d06`; exit 0; exact PID cleanup; peak Job memory 4,966,543,360 bytes.
- Native trace: release executable SHA-256 `338ebcbf02dbac13fabf6ce9115bdb3a91fc3316a84a9c23e1ad304fbd900d9a`; manifest SHA-256 `286bc3c453188de38ac12a9553e60515a17aad61a57d03086c350b0f2d013345`; exit 0; PID 33624 absent; peak Job memory 2,120,413,184 bytes under an 8 GiB ceiling.
- Comparison: SHA-256 `caaae9ad159ec8370007169bd7c486ccff96f8b547ea6a113685f0c8703bbbac`; `passed=true`; 36/36 tensors; zero failures; exact inputs, artifact identity, stage inventory, decode IDs, and cache reset. Largest max abs was `0.0189208984375` at vision layer 11 and passed the recorded CPU-F32 allclose contract.
- Post-run census: no llama/model/Cargo/rustc process; 46,049,075,200 available physical bytes; 49,654,607,872 bytes commit headroom; 23,420 MiB GPU free.

### Official 450M GGUF artifact gate

- Source: `LiquidAI/LFM2.5-VL-450M-GGUF@166cd80bbe157dc86d65f964eb8cc6a2cede62ca`; the official file page identifies the Q4_0 text artifact at that immutable commit.
- Text GGUF: `LFM2.5-VL-450M-Q4_0.gguf`, 219,311,264 bytes, SHA-256 `6d2757dd0f0b98aea7dc90477bb5b3a0df1089be85ef92943f8cecb05121ccbf`.
- Text header: exact payload-free prefix 2,388,128 bytes, SHA-256 `bdb33b992b136a77b4d807b84319a7daa43ebac15144e6336c0d9b9ef1e8ed2e`; 39 metadata records, 148 tensors, physical size equal to declared extent, official 16x1024/FFN-4608/tokenizer-65536 contract.
- MMProj: official Q8_0 file 102,815,168 bytes, SHA-256 `ebfc428baa37efad8bae93864f914b2634a09009f91ad59f974fe1a1565d8561`; the Hugging Face blob and existing `C:\llamacpp` copy are byte-identical.
- Separation proof: the prior game-QA derivative is 219,310,432 bytes, SHA-256 `84540fa23696ab9000f4a670b72e3405962264a920c3b7582d0e5a38b978abae`, with a different 27-record/header identity. It is not an admissible P2 text artifact.
- Discovery used bounded header inspection and sequential hashing only. No GGUF tensor was decoded, no inference process started, and the post-discovery model-process census was clear.

### Official 450M GGUF runtime gate

- Exact artifact set: the official Q4_0 text GGUF and Q8_0 MMProj above, plus the 572-byte 256x256 deterministic image with SHA-256 `f902f8d2e47e53eafac86831cfc692001dc15870eb81d57abc3128f048d2efca`.
- Candle replay: release executable SHA-256 `338ebcbf02dbac13fabf6ce9115bdb3a91fc3316a84a9c23e1ad304fbd900d9a`; CPU F32 hybrid GGUF/MMProj; exit 0; PID 23832 absent; exact reset replay; 64 projected image tokens; generated IDs `[1098, 4646, 5251]`; decoded `The image features`; peak Job memory 918,130,688 bytes under an 8 GiB ceiling.
- Pinned llama.cpp replay: build 10335 at `74ce15741b420b8d6f12e720398458b576c51c2c`; executable SHA-256 `848e638069699149210b70945bdbb422494d7d03b8a18d7fb31a240d10e8abd0`; CPU text and MMProj; exit 0; PID 6124 absent; decoded `The image features`; peak Job memory 1,777,582,080 bytes under the same ceiling.
- Prompt framing: Candle consumed the official rendered prompt. llama.cpp consumed the same raw user text through an embedded official template whose only difference from the standalone pinned file is one trailing LF.
- Bounded difference: Candle reports the artifact's 128,000-token capacity; llama.cpp used a deliberate 4,096-token KV cap. The 80 input IDs plus three generated IDs occupy 83 positions, so the differing ceilings do not affect this replay.
- Claim boundary: llama.cpp exposes no stable generated IDs, preprocessing dimensions, projected-token count, logits, component tensors, or cache-reset replay. Those fields are explicitly unavailable, not inferred as matches.
- Machine comparison: external `official-gguf-comparison-256-v1.json`, 7,026 bytes, SHA-256 `2c54cd790aef5ddcf8b053923a7ebb18ef055e9b06b6b580abd2a1eb9b92f6fd`; `passed=true`, verdict `pass_with_bounded_differences`.
- Recovery: both child trees were absent after cleanup. Final llama.cpp postflight retained 46,353,580,032 available physical bytes, 50,132,758,528 bytes commit headroom, and 23,422 MiB GPU memory free.

### Official 1.6B no-model admission gate

- Target: `LiquidAI/LFM2.5-VL-1.6B@919fde3d022e3f90a4716006f993938ee8c2eb97`; local Hugging Face cache absent; no payload downloaded or model loaded.
- Locked structure: one 3,193,334,216-byte `model.safetensors` file; expected LFS SHA-256 `7fc7458e4382fc6e558cfdda45857fbf9ab5b40a8bf199c9cd073003b14ac26d`, subject to mandatory local full-file rehash; 82,400-byte payload-free header; 589 tensors; inventory SHA-256 `24728d0ed10229e788c5b9baf25e0cc6c92c93b9cdb12ebb252a3c140a861703`.
- Snapshot forecast: eight regular files totaling 3,198,084,631 bytes. A cache plus regular copy, two projected trace files, and 1 GiB miscellaneous margin require about 7.30 GiB; acquisition admission requires at least 12 GiB free.
- Trace projection: applying the locked 1.6B widths and 27 vision layers to the exact 450M selected tensors yields 51 tensors, 182,523,192 data bytes, and an estimated 182,530,856-byte safetensors file. This is a shape-derived estimate, not generated evidence.
- Memory forecast: exact model-byte ratio 3.558093732. Measured 450M peaks scale to 10,229,894,076 bytes for Python dry load, 17,671,426,800 for Python trace, and 7,544,628,860 for native trace. A further 1.35 safety factor yields 13,810,357,003 / 23,856,426,180 / 10,185,248,961 bytes.
- Stage ceilings: 16 GiB Python dry load, 24 GiB Python trace, and 12 GiB native trace. Each is below 75% of host RAM; none may rise automatically after a limit termination. Python-trace admission requires at least 32 GiB available physical and commit headroom plus zero competing model/build processes.
- Forecast evidence: external `p3-1.6b-resource-forecast-v1.json`, 5,587 bytes, SHA-256 `0c8f3cd31cea807591356d90aa442a2a02421e86a58215c01b4bcecc12659a59`; `passed=true`, verdict `ready_for_guarded_acquisition_not_inference`.

### Guarded 1.6B artifact-acquisition gate

- The checked-in contract owns eight exact filenames, byte counts, Git-blob/LFS identities, immutable revision, public/no-token policy, 3,198,084,631-byte snapshot total, and 12 GiB disk minimum.
- `acquire_snapshot.py --plan` passed on native Windows with 243,618,676,736 bytes free. Schema 2 reported `network_policy=disabled`, `network_used=false`, `model_loaded=false`, and `transfer_policy=serial-files-resumable-http-xet-disabled`; the snapshot, cache, and manifest paths remained absent afterward.
- The approved execution path will download serially into a resumable external Hub cache with Xet disabled before Hub import, stream and verify a clean staging snapshot, then atomically publish the snapshot and acquisition manifest without replacing a destination that appeared after planning. Stale snapshot or manifest staging blocks retry; failure rolls publication back.
- The download-only path checks `huggingface-hub==1.5.0`; it does not require Torch, TorchVision, Transformers, Pillow, or the other numerical-oracle packages. Those remain mandatory before any model load or trace.
- Evidence schema 2 records `network_policy=disabled` / `network_used=false` for planning and `network_policy=permitted-cache-aware` / `network_used=null` for execution; a pinned cache hit is never misreported as observed network traffic.
- Focused offline acquisition tests are green at 27/27, including site-packages-free planning, stale snapshot/manifest-stage refusal, post-cache path revalidation, returned-source cache containment, resumable-cache preservation, destination-race no-clobber behavior, duplicate verifier rejection, cleanup/rollback diagnostics, Xet-enabled pre-import refusal, and an exact public signature with no downloader/verifier bypass. Native Windows proved hard-link manifest publication against a racing writer; WSL proved Linux `renameat2(RENAME_NOREPLACE)` success and collision refusal. The real pinned Hub API was inspected offline with Xet disabled and every supplied keyword present. No 1.6B payload was downloaded, copied, or loaded.

### Current source regressions

- `cargo fmt --all -- --check`: green after the final trace fixes.
- Locked/offline `cargo check -j 2` is green for `candle-core`, `candle-nn`, `candle-transformers`, and `candle-vlm`, plus the `lfm2`, `quantized-lfm2`, and `lfm2-vl` examples.
- Locked/offline `cargo test -j 2` is green after upstream integration for the affected core/transformer/VLM workspace lanes: transformer 58/58, generation 5/5, NMS 8/8, VLM 29/29, and all core integration/doc lanes.
- `cargo test --locked --offline -j 2 -p candle-examples --example lfm2-vl`: 29/29 green, including destination-race preservation for native trace publication.
- Scoped strict Clippy is green for the affected libraries and the LFM2-VL example with only `manual-is-multiple-of` and `needless-range-loop` allowed for compatibility/indexing clarity. Two mod-owned `manual_contains` findings were fixed.
- Exact pinned `pytest tools/lfm2_vl/reference -q`: 81/81 green after shared no-clobber report publication, tokenizer marker/range validation, split-MMProj race preservation, guarded acquisition coverage, and prior GGUF/reference regressions.
- Exact pinned Python compileall and environment/lock verification are green; generated repository Python caches were removed afterward.
- The retained `.venv` cannot launch inside the managed sandbox because its Python 3.10.11 base executable lives under user AppData. The approved native execution replay passed the exact environment/lock verifier, 81/81 tests, and compileall without reinstalling any package.
- Bounded-wrapper smoke: green under PowerShell 7.6.4 and Windows PowerShell 5.1, including suspended assignment, timeout/tree cleanup, owner-exit cleanup, name/executable concurrency, combined logging, and a synthetic process counter above `Int32::MAX`. The owner-exit regression now waits for a child-written post-resume handshake so the test cannot kill its owner before job assignment; failure cleanup also terminates the exact test child.
- Resource-preflight smoke is green under both PowerShell versions. The summary bank also verifies under both versions at SHA-256 `91a0ccf11512bc71617ecc251922ecf652164391c0f197a3d398cea66892a507`; all routes remain below 256 KiB after separating native loading, reference environment, GGUF inspection, production parity, exclusive-publication, and direct-main provenance context.
- Relative links pass across all 20 mod-owned Markdown files. A repository-wide diagnostic separately found three pre-existing malformed upstream Qwen example links; they are outside the LFM2-VL publication allowlist and were not changed.
- Current WSL Git inspection is green through the explicit `NVIDIA-Workbench` distribution. GitHub main at `6f74e7c390c717f8fd34f23ce02aceb058173370` was merged without conflict or force; the release overlay remains exactly 91 allowlisted paths, 9 fork-origin modifications, and 82 mod-owned additions.
- The Linux-specific native trace collision test could not launch in the installed WSL distribution because `cargo` is absent. An offline Windows-hosted Linux-target check reached cached compilation but stopped at `openssl-sys` because no Linux OpenSSL sysroot is configured. Windows proved the same public no-clobber contract; TODO C3 retains the truthful secondary-lane replay instead of installing tooling implicitly.

## Proven

- LFM2.5 dense and quantized embedding prefill, cached decode, and reset behavior.
- SigLIP2 NaFlex, pixel unshuffle, projector, native composite, raw-image processing, prompt expansion, split dense MMProj, direct GGUF MMProj, and CPU-F32 native Q8 MMProj on deterministic fixtures.
- Strict native/HF and hybrid loaders with exact regular-file evidence and controlled malformed-input failures.
- Deterministic native/hybrid runner evidence and local fine-tuned-text GGUF plus official MMProj decoded-output agreement with pinned llama.cpp.
- Official 450M native Windows CPU-F32 processor, vision, projector, merge, prefill, cached-decode, artifact, deterministic replay, and cleanup parity against pinned Transformers.
- Official-base Q4_0 text GGUF plus Q8_0 MMProj exact-artifact, prompt-semantic, three-token decoded-output, bounded-resource, and cleanup agreement with pinned llama.cpp.
- Windows inference containment with kill-on-close Job Objects, 64-bit resource accounting, timeout/memory termination, exact executable concurrency, retained combined logs, and exact PID cleanup.
- Default no-clobber publication for Python reports and split bundles, PowerShell owner evidence, and native Windows trace directories; replacement now requires an explicit overwrite/force option where it is allowed.

## Known Gaps and Conflicts

- The official 1.6B payload is not local and its dry load, component traces, comparison, and cleanup proof remain incomplete. Native CUDA execution, lower-bit MMProj execution, and current WSL portability replay are also incomplete.
- Official config context is 128,000 while model cards advertise 32,768; construction follows config and production policy remains unresolved.
- Official MMProj headers omit tiling metadata; pinned processor configuration or documented architecture defaults remain required.
- The prior llama.cpp residency incident required a host restart. Exact root cause is unproven; F-0008 containment remains mandatory.
- The Linux-specific native trace publication regression has not run locally because WSL has no Rust toolchain and Windows cross-compilation has no Linux OpenSSL sysroot; this secondary portability check is tracked as TODO C3.
- Gknome adoption remains fail-closed on four mature-repository authority conflicts: `.gitignore`, `AGENTS.md`, `README.md`, and `summary_bank.json`.

## Blockers

- None for local source work.
- P3 artifact acquisition requires a separate guarded network/write action for an expected 3,198,084,631-byte external snapshot; no implicit multi-gigabyte download was performed.
- P3 inference remains blocked until the acquired regular snapshot passes exact local hashes and a fresh preflight satisfies the stage-specific forecast thresholds.
- Native Windows CUDA/distinct-device proof remains sequencing-blocked until P3 CPU-F32 parity is green.
- The Linux native-trace no-replace regression remains environment-blocked because the installed WSL distribution has no Cargo and the Windows cross target has no Linux OpenSSL sysroot. This is secondary portability debt, not a Windows release blocker.
- Gknome apply remains blocked until its dry plan has zero unresolved authority conflicts; repair/bypass is not authorized.

## Active Files

- No source file remains under uncommitted release work after the direct-main closeout.
- The next guarded acquisition task owns `tools/lfm2_vl/reference/acquire_snapshot.py`, `tools/lfm2_vl/reference-lock.json`, and external ignored evidence paths only; it must stop before model load.

## Exact Next Task

With separate approval for the multi-gigabyte production download, invoke the documented `--allow-production-download` argument array through `run-bounded-oracle.ps1` directly from the current PowerShell process. Use the exact `.venv` Python, 2 GiB Job ceiling, 7,200-second timeout, executable-scoped concurrency, and external log/owner evidence. Acquire the eight pinned 1.6B files into the named external regular-file snapshot, retain repository/revision/size/hash evidence, and verify `model.safetensors` is exactly 3,193,334,216 bytes with SHA-256 `7fc7458e4382fc6e558cfdda45857fbf9ab5b40a8bf199c9cd073003b14ac26d`. Stop before model load; dry load remains the next independent bounded task.

---
AI-edited: 2026-08-11T09:59:19-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=release | change=recorded direct-main integration, post-merge verification, and the owner-exit smoke race fix
