# LFM2.5-VL Status

## Baseline and Publication

- Model and compatibility baseline: Candle 0.11.0 at `31f35b147389700ed2a178ee66a91c3cc25cc80d`.
- Upstream integration base: Candle main at `6f74e7c390c717f8fd34f23ce02aceb058173370`.
- Integration and publication branch: `main` at `https://github.com/Shoozes/candle.git`.
- Historical feature checkpoint: `c9b60f0b906fa8fe70423295e2e1164648a8fa53` on `feat/lfm2-vl-mmproj`.
- Current reviewed release base: `b10c3a0c335050c066d8e02fd9f528f6b490fa39` (`fix(lfm2-vl): harden direct-main release verification`); upstream-preserving merge checkpoint: `2b1d9e80de06b251b2fe5f25e51c17d56db86591`. The live branch tip is authoritative through `git rev-parse main`; this durable file records the source/evidence base it describes rather than attempting an impossible self-referential commit SHA.
- Pull request: none.
- Current mod overlay relative to the upstream integration base: 94 allowlisted paths, exactly 12 fork-origin modifications and 82 mod-owned additions. The 29 inherited post-0.11 upstream paths remain outside the mod overlay.

## Worktree Boundary

- Native Windows/MSVC is the product and primary proof lane; WSL2/Linux is a secondary portability replay.
- `C:\DevStuff\candle-mods` is a WSL-owned linked worktree attached to local `main`; its `.git` file points to Linux-owned metadata under `/home/workbench/code/candle-lfm2-vl`.
- Windows Git cannot resolve that pointer. Use the explicit `NVIDIA-Workbench` WSL distribution for all Git operations. The historical feature branch remains in the Linux-home worktree and must not be attached here.
- Owner-reviewed work lands directly on `main` without a PR. Broad staging, force-push, implicit merge/rebase, and secret inspection remain prohibited; the ignored `.tools/gitpush.ps1` only verifies and pushes an already clean fast-forward branch after explicit approval.

## Current Phase

- Product phase: post-Phase 7 production stabilization.
- Release posture: feature-complete MVP release candidate, not LTS. Core native, split-MMProj, direct-GGUF, Q8_0, processor, prompt, trace, bounded-runtime, and admitted native CUDA placement paths are implemented; remaining product work is measured CUDA optimization and lower-bit production CUDA. The exact Linux trace-publication replay is green; broader WSL replay remains secondary.
- NR-5B official 450M native Windows CPU-F32 component parity is green.
- P2 official-base GGUF same-artifact comparison is green at every stable field exposed by both runtimes.
- Current product gate: P4.4 measured CUDA optimization; P3 official 1.6B native Windows CPU-F32 component parity, P4.1 public device policy, P4.2 tiny native CUDA/distinct-device proof, and P4.3 official 450M CUDA parity are green and archived.
- P3.1 through P3.5 are green: the complete pinned 1.6B regular-file snapshot, config/tokenizer/processor admission, Python/native load-only, both 51-tensor traces, exact reset, phase-specific comparison, and bounded cleanup all pass. The native source fix is localized stable F32 LayerNorm for SigLIP2 encoder pre-norms; the comparator now records the written CPU-F32 phase contract (allclose-or-cosine for vision/projector/hidden states and max-abs `1e-3` for prefill logits).
- Exact next step: P4.4, use the captured stage baseline to measure the decode/cache hot path with repeated warm-up and an explicit variance bound, then retain at most one measured optimization that preserves CPU/CUDA parity. Lower-bit CUDA and broader WSL replay remain later.
- Hosted CI is neither invoked nor accepted as release evidence. Required checks run locally on native Windows, with WSL2/Linux used only as an explicitly labeled secondary replay.

## Last Green Verification

### P4.1 public device-policy gate

- Added `--text-cpu` to all three LFM2-VL loading forms. The resolved matrix is accelerator/accelerator by default, accelerator/CPU with `--vision-cpu`, CPU/accelerator with `--text-cpu`, and CPU/CPU with `--cpu`; `--cpu` remains authoritative when combined with either component flag.
- The existing `main.rs` policy consumer now resolves the new CPU-text route without a second device abstraction. Native and hybrid reports already expose the resolved text and vision devices, so no evidence schema changed.
- Focused parser/policy tests passed 12/12, including help exposure, the four policy combinations, `--cpu` precedence, and the trace lane's controlled rejection when only `--text-cpu` is supplied. `cargo check --locked --offline -j 2 -p candle-examples --example lfm2-vl` passed, and formatting was normalized with `cargo fmt --all`.
- The example README documents the CPU-text/accelerator-vision command and the complete placement matrix. No model or CUDA workload was started for this CLI-only gate.

### P4.2 bounded native CUDA/distinct-device gate

- Native toolchain identity: `nvcc` 13.3.33, Cargo 1.91.0, rustc 1.91.0, MSVC target, RTX 4090 driver `32.0.16.1088`. The first bounded compile failed before test execution because CUDA 13.3 CCCL requires MSVC `/Zc:preprocessor`; the repository fix passes `-Xcompiler /Zc:preprocessor` through both `candle-kernels/build.rs` PTX and static-library builders.
- The corrected bounded owner `p4-2-native-cuda-distinct-owner-20260811T183000Z.json` records Cargo PID 28056, exit 0, 113,852 ms, peak Job memory 2,691,182,592 bytes under a 16 GiB ceiling, and exact PID cleanup. Its log SHA-256 is `bcc59ec9dca523955aefdcfdd1d1668317e2a2dd6b0d5e3b9db2006d35de6cd4`; owner SHA-256 is `57bd3b15081c61c3b1e64ff24d0dabbb2c344dac033446751daad0849d237de7`.
- The native loader test passed 1/1: CUDA vision and CPU text were constructed on distinct devices. The companion `candle-transformers` test passed 1/1 and proved projected-feature-only transfer, hybrid CPU-text prefill, and CUDA/CPU agreement (`max_abs=4.456564784e-5`); its owner SHA-256 is `ca73e16f06d30396497e2500061229e59ea1b93e5c849339d73fed04839f7227` and log SHA-256 is `1b1806537b1bbb4838cdbad16b7f88f02122537951bc065aed64d6c1e88dd3e6`.
- Final clean postflight `p4-2-postflight-clean-20260811T183800Z.json` has SHA-256 `76b33d493f2b82cd00b253acc74bd31a21478cf0a70daef746e767707abcf7aa`, zero tracked/llama processes, 43,408,338,944 available physical bytes, 47,523,815,424 bytes commit headroom, and 23,421 MiB GPU free. No production model was loaded.

### P4.3 official 450M CUDA parity gate

- The final release executable is `C:\DevStuff\candle-mods\target\release\examples\lfm2-vl.exe`, 65,032,192 bytes, SHA-256 `5b147767e5c45074035d884eaa0b1111ee0ebc6dbf5ed098ee8f120539a8a669`. The admitted model is `LiquidAI/LFM2.5-VL-450M@fc6221ca597f3315e4f82fc2df606783267b34ba`; artifact-manifest SHA-256 is `659c8421530586b6cc28c094cfcdc69719ea8626f2abc0efd9eec4ac2a68a984`; deterministic image SHA-256 is `f902f8d2e47e53eafac86831cfc692001dc15870eb81d57abc3128f048d2efca`; and the rendered prompt is unchanged from the green CPU-F32 baseline.
- The final CUDA release rebuild used the bounded Rustup owner `p4-3-cuda-release-rebuild-owner-20260811T193500Z.json` (owner SHA-256 `349724f10bccb8e1f923ee4969d88d515c4089ed294582ba544e7ee18f7b4515`, log SHA-256 `e9958f8de2a361dbfe10a84d4b1591a85235e959ee733e9900f8fc0ecb37b83f`), exited 0 in 15,817 ms, peaked at 890,474,496 Job bytes, and left PID 25852 absent.
- Sequential all-CUDA F32 evidence is `p4-3-all-cuda-f32-owner-20260811T194000Z.json` (owner SHA-256 `346925d6be44621b5b03d5c10ca1b69d06c5b279bfb63fabd0d8351ebc82de77`) with log SHA-256 `2991fedc58de944dbe7065cf0a824ec8e6460dce68a4d2ee601a17768c30c076`; it exited 0, peaked at 3,474,706,432 Job bytes, and left PID 14684 absent. CPU-text/CUDA-vision F32 evidence is `p4-3-text-cpu-f32-owner-20260811T194100Z.json` (owner SHA-256 `42340a9659dbfa1b889715fd9abd94556c2b1e55b9f2b38d7635e7fb7912d63a`) with log SHA-256 `f92322fd8ce307701220767da0da0c974bcfe62d0d38f63c811b973ac33d05b0`; it exited 0, peaked at 3,241,332,736 Job bytes, and left PID 28720 absent. Both runs were sequential, bounded, and resource-clean.
- Both F32 routes generated IDs `[1098, 4646, 5251]`, expanded the same prompt, projected 64 image tokens, and reset the cache exactly. Top-k IDs matched the CPU baseline at all three steps; maximum displayed top-k drift was approximately `3.960059e-5` for all-CUDA and `2.660059e-5` for CPU-text/CUDA-vision.
- All-CUDA BF16 also passed: `p4-3-all-cuda-bf16-20260811T194200Z-owner.json` (owner SHA-256 `b01977d99ea6dd5fb64ad8d552bc4a932d3e6654356fc706a755024f4a174e94`), log SHA-256 `c143193787e869864e8c071abc9fa72b4a580ea93d3d75657dfa7e040cec8764`, exit 0, peak Job memory 2,783,182,848 bytes, PID 27048 absent. Explicit BF16 on a CPU component is unsupported by Candle CPU matmul and is now rejected before model load; guarded evidence `p4-3-text-cpu-bf16-guarded-20260811T194300Z-owner.json` (owner SHA-256 `2317d3bc44650f0b35862ed139eee4d86be10cc07f5541e58d7eb8b8c194f465`) has log SHA-256 `e1a4285ea5bcc186d869a214defc8b04754414ba1714a7dddbf53c5a21e6f78`, child exit 1, peak Job memory 772,743,168 bytes, and exact PID cleanup.
- P4.3 is green for the admitted native 450M all-CUDA F32/BF16 and CPU-text/CUDA-vision F32 placements. The CUDA fixes are a missing `cast_i32_f32` kernel and dense projector linear input materialization; the public `--text-cpu` route now creates CUDA vision independently instead of cloning CPU text's device. P4.4 is the active optimization gate.
- Post-change local checks are green: `cargo fmt --all -- --check`; LFM2-VL argument tests 14/14; full example tests 32/32; CUDA cast regression 1/1; CUDA dense-linear regression 1/1; bounded offline checks for `candle-core`, `candle-nn`, `candle-transformers`, and `candle-vlm`; bounded offline checks for `lfm2`, `quantized-lfm2`, and `lfm2-vl`; PowerShell 7/5.1 summary-bank verification; WSL manifest verification; and WSL `git diff --check`.

### P4.4 diagnostic timing baseline

- The opt-in `--timings` flag now reports model-load, image-load, processor, prompt, vision, first-generation, cache-reset replay, and total inference durations to stderr without changing the versioned JSON evidence. `--help` exposes the flag, and the full example test suite remains green at 32/32, including rejection of `--timings` without `--prompt` before model loading.
- The bounded release rebuild owner `p4-4-timing-rebuild-owner-20260811T193400Z.json` exited 0, peaked at 890,847,232 Job bytes, and left its PID absent. Owner SHA-256 is `7345c2bf2401c126ea828abdfb60f845ff50eab2f07e9d53e8aff44265f956fa`; log SHA-256 is `755aac9083bf0eafce60c8ef7da83cf8f0fbc599815968f6b505878549c596d3`. The resulting executable is 65,035,264 bytes with SHA-256 `b25984aac5332f2655ba91478dec5a46b4fe6e538760891c2a6591816c54d81a`.
- After reverting the unproven ShortConv candidate, the retained source was rebuilt under `p4-4-final-rebuild-20260811T204000Z-owner.json`: exit 0, 99,263 ms, peak 2,066,661,376 Job bytes, PID absent, owner SHA-256 `92580761caa18396fe6407ac733eef2f2e259b25c3d9ef31c2ecf926505ee7e2` and log SHA-256 `94e156b7ca2ab15a1c5b5333b6c28e39c209f729f7c276261258bac69abd4d7a`. The current source-matching executable is 65,035,264 bytes with SHA-256 `9cd51ffefbae6e5c80907629817e0a27a854fc3325e61aa041f79eec9c7998c8`; rebuilds are not byte-identical, so each evidence record carries its own artifact hash.
- Three sequential all-CUDA F32 runs over the exact official 450M checkpoint and deterministic image exited 0, left their PIDs absent, and peaked at 3,475,668,992, 3,474,984,960, and 3,476,611,072 Job bytes. Across the runs, model load was 435.885–452.501 ms, vision 38.037–39.186 ms, first generation 446.959–469.142 ms, cache-reset replay 419.065–444.422 ms, and total inference 1,388.056–1,462.027 ms. Owner/log hash pairs are retained externally as `d5a7a5622cbd9f867f14d39f3632f780faf17f00fa7a22cdf191e85734d99a85`/`25f49f9c5b89789c2ae70b94888723b87f93e4dc37a205861c574e7473b3c21f`, `89b3535a3bafd84e839bbd4f7bd3faa351161ba1563fdeda910d097c67f1def7`/`a1527db845ada92acc5a08e48b8fdceb1f88d1ed76b4facecea29dbfc1760405`, and `795ab08a33c5556e4c6cd9785915791497bb03db45c5aee7baab169fbfa03aad`/`6ec4f82dc8a0cbe30ace447c963345dbee7757c78d4d9cdfe18904d3c2cb9f0a` under `C:\DevStuff\candle-oracle\evidence`.
- Narrowing runs at zero, one, and three requested tokens showed model-load and replay variance large enough that the timing data does not justify an optimization claim yet. Generation is the largest stable stage; the next proof is a repeated-warm-up decode/cache microbenchmark with an explicit variance bound, followed by one measured change and parity replay.
- A ShortConv weight-view candidate was audited and reverted. Its first comparison used a literal backtick-newline prompt; the corrected official-prompt replay preserved IDs and the prefill-logit hash but overlapped an unrelated EdgeSymbio Cargo build and was slower/noisy. No source optimization is retained without a quiet-host, exact-prompt before/after series.

### Review-ingestion gate

- WSL Git established a clean `main` baseline with `HEAD`, local `main`, and `origin/main` all at `b10c3a0c335050c066d8e02fd9f528f6b490fa39` before the three documentation edits.
- Source inspection confirmed the public device-policy matrix now has all four routes, while CUDA-gated loader/model tests cover vision CUDA with text CPU. The reviewed LFM2-VL production paths contain no `todo!` or `unimplemented!`; root `README.md` has no LFM2/`candle-vlm` entry and `candle-vlm/README.md` is absent.
- `cargo fmt --all -- --check` passed. The exact existing device-policy test passed 1/1, and locked/offline two-job checks passed for `candle-core`, `candle-nn`, `candle-transformers`, `candle-vlm`, and the `lfm2`, `quantized-lfm2`, and `lfm2-vl` examples.
- PowerShell 7.6.4 and Windows PowerShell 5.1 both passed the summary-bank verifier with every route below 256 KiB. All active task contracts contain What/Why/When/Where/How/Done-when/Verification; 20 mod-owned Markdown files, nine JSON files, trailing whitespace, the 94-path mod manifest, and `git diff --check` are green.
- No model or hosted runner was used. Final preflight found no llama/Cargo/rustc/LFM2-VL process, 41,429,434,368 available physical bytes, 45,314,990,080 bytes commit headroom, and 23,384 MiB GPU memory free. Native Windows Git remains expectedly unavailable through the Linux `.git` pointer; the owning WSL Git checks passed.

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

### Official 1.6B artifact and no-model admission gate

- Target: `LiquidAI/LFM2.5-VL-1.6B@919fde3d022e3f90a4716006f993938ee8c2eb97`; the external snapshot is locally admitted, while model load and numerical execution remain unstarted.
- Locked structure: one 3,193,334,216-byte `model.safetensors` file with locally confirmed SHA-256 `7fc7458e4382fc6e558cfdda45857fbf9ab5b40a8bf199c9cd073003b14ac26d`; 82,400-byte payload-free header; 589 tensors; inventory SHA-256 `24728d0ed10229e788c5b9baf25e0cc6c92c93b9cdb12ebb252a3c140a861703`.
- Snapshot forecast: eight regular files totaling 3,198,084,631 bytes. A cache plus regular copy, two projected trace files, and 1 GiB miscellaneous margin require about 7.30 GiB; acquisition admission requires at least 12 GiB free.
- Trace projection: applying the locked 1.6B widths and 27 vision layers to the exact 450M selected tensors yields 51 tensors, 182,523,192 data bytes, and an estimated 182,530,856-byte safetensors file. This is a shape-derived estimate, not generated evidence.
- Memory forecast: exact model-byte ratio 3.558093732. Measured 450M peaks scale to 10,229,894,076 bytes for Python dry load, 17,671,426,800 for Python trace, and 7,544,628,860 for native trace. A further 1.35 safety factor yields 13,810,357,003 / 23,856,426,180 / 10,185,248,961 bytes.
- Stage ceilings: 16 GiB Python dry load, 24 GiB Python trace, and 12 GiB native trace. Each is below 75% of host RAM; none may rise automatically after a limit termination. Python-trace admission requires at least 32 GiB available physical and commit headroom plus zero competing model/build processes.
- Forecast evidence: external `p3-1.6b-resource-forecast-v1.json`, 5,587 bytes, SHA-256 `0c8f3cd31cea807591356d90aa442a2a02421e86a58215c01b4bcecc12659a59`; `passed=true`, verdict `ready_for_guarded_acquisition_not_inference`.

### Guarded 1.6B artifact-acquisition gate

- The checked-in contract owns eight exact filenames, byte counts, Git-blob/LFS identities, immutable revision, public/no-token policy, 3,198,084,631-byte snapshot total, and 12 GiB disk minimum.
- The final `acquire_snapshot.py --plan` replay passed immediately before acquisition with 213,124,534,272 bytes free. Machine assertions rechecked schema 2, eight files, the 3,198,084,631-byte total, immutable revision, exact 3,193,334,216-byte weight identity, public/no-token policy, Xet-disabled serial transfer, `network_policy=disabled`, `network_used=false`, and `model_loaded=false`; snapshot, cache, manifest, and staging paths remained absent afterward.
- After explicit owner approval and a sustained zero-Cargo/rustc/llama window, the production acquisition ran through the Windows Job Object owner with the exact pinned Python, a 2,147,483,648-byte limit, and a 7,200-second timeout. PID 22940 exited 0 in 129,190 ms, peaked at 75,395,072 Job bytes, and was absent after cleanup; the 1,724-byte owner record has SHA-256 `631fb14581ef89f53c983ae2c77ff444f889d16a42bc8b5c3dede52c760a9380`.
- The external snapshot at `C:\DevStuff\candle-oracle\lfm2-vl-1.6b-919fde3d` contains exactly eight direct regular files totaling 3,198,084,631 bytes. An independent full-file pass matched every acquisition and artifact record and rehashed the 3,193,334,216-byte weight as `7fc7458e4382fc6e558cfdda45857fbf9ab5b40a8bf199c9cd073003b14ac26d`. Atomic publication is true; no snapshot/manifest stage or incomplete cache file remains.
- The 4,818-byte acquisition manifest has SHA-256 `a080891c8d1099d58a01377af258ef04898f808eed0fcf4fbe718d4698f4b732`; the 4,958-byte combined log has SHA-256 `0d4357d9c532ba943ec8ad5c495c733734652f7cd38cc4ca5d0de101ae16b1f3`. Evidence records `network_policy=permitted-cache-aware`, `network_used=null`, public/no-token access, Xet-disabled serial transfer, and `model_loaded=false`.
- Final postflight retained 46,482,870,272 available physical bytes, 48,465,375,232 bytes commit headroom, 199,416,389,632 bytes disk free, and 23,523 MiB GPU memory free. No llama or exact acquisition interpreter remained; unrelated Cargo/rustc work had resumed, so P3.2 still requires a fresh quiet-host admission.
- The download-only path checks `huggingface-hub==1.5.0`; it does not require Torch, TorchVision, Transformers, Pillow, or the other numerical-oracle packages. Those remain mandatory before any model load or trace.
- Evidence schema 2 records `network_policy=disabled` / `network_used=false` for planning and `network_policy=permitted-cache-aware` / `network_used=null` for execution; a pinned cache hit is never misreported as observed network traffic.
- Focused offline acquisition tests replayed green at 27/27 in 0.48 seconds, including site-packages-free planning, stale snapshot/manifest-stage refusal, post-cache path revalidation, returned-source cache containment, resumable-cache preservation, destination-race no-clobber behavior, duplicate verifier rejection, cleanup/rollback diagnostics, Xet-enabled pre-import refusal, and an exact public signature with no downloader/verifier bypass. Native Windows proved hard-link manifest publication against a racing writer; WSL proved Linux `renameat2(RENAME_NOREPLACE)` success and collision refusal. The real pinned Hub API was inspected offline with Xet disabled and every supplied keyword present. The test replay itself did not download, copy, or load the production payload.

### Official 1.6B non-load P3.2 admission

- The exact admitted snapshot at `C:\DevStuff\candle-oracle\lfm2-vl-1.6b-919fde3d` still contains eight direct regular files totaling `3,198,084,631` bytes. The fresh hash-only audit manifest is external at `C:\DevStuff\candle-oracle\evidence\p3-1.6b-artifact-audit-20260811T162000Z.json` (1,983 bytes, SHA-256 `b8d582c40214a1a8df82f21ece21fb683a5e5377c7c03b4fba0e97feb865e585`), and its reported model identity/revision and eight-file total match the locked contract.
- The stdlib-only config audit is external at `C:\DevStuff\candle-oracle\evidence\p3-1.6b-config-audit-20260811T162000Z.json` (7,835 bytes, SHA-256 `39fac83ce04986a2c14ea2e3b423eb81d34197db817a59e9b02b9d7ccfeee596`). It proves `Lfm2VlForConditionalGeneration`, 589 tensors, text/vision widths `2048/1152`, effective FFN `8192`, image token `396`, patch `16`, downsample `2`, tied output embeddings, and unique in-range marker IDs.
- The exact Windows reference environment verifier passed with all 42 distributions, including Python `3.10.11`, CPU Torch `2.8.0`, Transformers revision `fd12552d770f745fdbe41031ff4daa688f5ed57e`, and no lock/test mismatches. The artifact audit reported the pinned revision `919fde3d022e3f90a4716006f993938ee8c2eb97` and `model_loaded=false`; no component inference or trace occurred, and no model process was retained.
- The preflight immediately before Python was `review` with no llama/Cargo/rustc workload; its bounded owner supplied the explicit load boundary. A later fresh census found a Codex-owned Cargo/Tauri tree, so the native run was held until its exact root and descendants could be identity-checked and stopped without touching Codex or unrelated processes.
- The bounded Python dry-load is now green. The exact pinned interpreter loaded the admitted snapshot through `export_fixtures.py --mode production --load-model` without inference or tensor serialization. External output `C:\DevStuff\candle-oracle\evidence\p3-1.6b-python-load-20260811T201413Z` has `weights_loaded=true`, `tensor_payload_generated=false`, and manifest/metadata SHA-256 values `d2ef2505b2b92fa0b2f0d00e048a13bd02ed09ae2d48cbc31b5130643444ee88` / `3654dd86dde140d8345979eec16e13cad66b7f349132f7a08ff11d67f2dabff7`. The bounded owner record `p3-1.6b-python-load-owner-20260811T201413Z.json` records PID 18452, exit 0, 19,289 ms, peak Job memory 7,475,851,264 bytes under 16 GiB, and PID absence; its SHA-256 is `bcb23def782a6c2ca804e6ad45ed6cebb111d5efd14aaa1efa6d9a1693addf38`.
- Python postflight retained 56,785,924,096 available physical bytes, 59,669,757,952 bytes commit headroom, 203,312,377,856 bytes disk free, and 23,482 MiB GPU free with no llama process.
- The verified Cargo/Tauri root was PID 24676 at the stable MSVC `cargo.exe` path, with `cargo-tauri.exe` PID 13752 beneath it and compiler descendants. Exact `taskkill /PID 24676 /T /F` terminated only that tree; Codex, ChatGPT, PowerShell, and unrelated helpers were not targeted. A three-second postflight found every captured PID and all Cargo/rustc/rustup/cargo-tauri processes absent.

### Official 1.6B bounded native load-only gate

- The immediate pre-load artifact rehash at `C:\DevStuff\candle-oracle\evidence\p3-1.6b-native-preload-artifact-20260811T204235Z.json` passed with the same eight-file, 3,198,084,631-byte snapshot and manifest SHA-256 `b8d582c40214a1a8df82f21ece21fb683a5e5377c7c03b4fba0e97feb865e585`.
- The recorded release executable is `C:\DevStuff\candle-mods\target\release\examples\lfm2-vl.exe`, 10,230,272 bytes, SHA-256 `338ebcbf02dbac13fabf6ce9115bdb3a91fc3316a84a9c23e1ad304fbd900d9a`. The direct command was `lfm2-vl.exe --model-dir C:\DevStuff\candle-oracle\lfm2-vl-1.6b-919fde3d --cpu` under `run-bounded-oracle.ps1`, a 7,200-second timeout, executable-scoped concurrency, suspended assignment, and a 12 GiB Job Object limit.
- Native PID 15792 exited 0 in 2,264 ms. The external owner record is `C:\DevStuff\candle-oracle\evidence\p3-1.6b-native-load-owner-20260811T204250Z.json`; its combined log is `C:\DevStuff\candle-oracle\evidence\p3-1.6b-native-load-20260811T204250Z.log` (668 bytes, SHA-256 `8c8395c2da88d76848fc66830a50c42bfee02b88e291bb27592808ae8acaee3e`). Peak private memory was 6,425,124,864 bytes, peak working set 8,763,895,808 bytes, and peak Job memory 6,433,579,008 bytes; the PID was absent after cleanup.
- The loader reported text width 2,048, 27 vision layers, patch 16, factor 2, image token 396, 589 tensors, one shard, F32 vision/text on CPU, the expected vision/projector/language roots, and tied output. No inference or trace payload was generated. Postflight found no Cargo/rustc/rustup/cargo-tauri/llama/LFM2-VL process, 50,953,560,064 bytes available physical memory, 55,951,667,200 bytes commit headroom, and 23,430 MiB GPU free.
- Post-load local verification passed: `cargo fmt --all -- --check`; `cargo test --locked --offline -j 2 -p candle-examples --example lfm2-vl` (29/29); PowerShell `verify-summary-bank.ps1`; WSL `git diff --check`; and WSL `verify-mod-manifest.sh` (91 allowlisted paths, 9 fork-origin modifications, 82 mod-owned additions). No source files changed in this load-only task.

### Official 1.6B bounded Python component trace (P3.3)

- The immediate pre-trace rehash at `C:\DevStuff\candle-oracle\evidence\p3-1.6b-python-trace-artifact-20260811T210231Z.json` preserved the eight-file, 3,198,084,631-byte snapshot and artifact-manifest SHA-256 `b8d582c40214a1a8df82f21ece21fb683a5e5377c7c03b4fba0e97feb865e585`. The deterministic source image is `C:\DevStuff\candle-oracle\inputs\trace-gradient-256.png`, SHA-256 `f902f8d2e47e53eafac86831cfc692001dc15870eb81d57abc3128f048d2efca`; the user prompt was `Describe this image.`; the Python package/runtime identity is recorded in the bundle.
- The exact command used `export_fixtures.py --mode production --model 1.6b --model-dir C:\DevStuff\candle-oracle\lfm2-vl-1.6b-919fde3d --allow-production --load-model --trace --image ... --prompt "Describe this image." --max-new-tokens 3 --output <external>`, wrapped by `run-bounded-oracle.ps1` with a 7,200-second timeout, CPU F32, executable-scoped concurrency, suspended assignment, and a 24 GiB Job Object limit.
- The Python owner record is `C:\DevStuff\candle-oracle\evidence\p3-1.6b-python-trace-owner-20260811T210231Z.json`; PID 28560 exited 0 in 28,505 ms with peak Job memory 14,482,644,992 bytes (under 24 GiB) and exact PID cleanup. The combined log is 762 bytes, SHA-256 `a85229de763b4ac459100d03fdbd6165a5fa99a2247eb52ec6ff1bc8c6ba973c`.
- The external trace bundle is `C:\DevStuff\candle-oracle\evidence\python-trace-1.6b-20260811T210231Z`; its manifest/metadata validate through the pinned reference validator, contain 51 tensors and a 182,528,392-byte safetensors payload with SHA-256 `184d62de07a1b72c8e6a0190b05ef15ff7361c2a029fe5fc2c04a0e17ebbb2f2`, record 80 input tokens and 64 projected tokens, and prove `cache_reset_exact=true`, `cache_reset_prefill_max_abs=0`, `artifact_manifest_reverified=true`, and `weights_serialized=false`.
- The clean postflight is `C:\DevStuff\candle-oracle\evidence\p3-1.6b-python-trace-postflight-clean-20260811T210659Z.json`, 718 bytes, SHA-256 `fdc6034e85a208a170016c39bae18f52ba258fe9aed378901eeb156e40289853`. After stopping the recreating Codex-owned build owner, it recorded zero model/build families, 43.5 GiB available physical memory, 47.4 GiB commit headroom, 207.8 GB disk free, and 23,438 MiB GPU free. Pinned reference tests passed 81/81.

### Official 1.6B bounded native component parity (P3.4–P3.5)

- The corrected native trace used the same pinned snapshot, deterministic image (`f902f8d2e47e53eafac86831cfc692001dc15870eb81d57abc3128f048d2efca`), official rendered prompt, CPU F32, single-crop contract, and three decode steps. The release executable was 10,791,424 bytes, SHA-256 `1f21125cdfe107a42a608920703755c499c7c75cae637b834724d78b175887e0`.
- The bounded owner record `C:\DevStuff\candle-oracle\evidence\p3-1.6b-native-trace-f32-corrected-owner-20260811T214952Z.json` records PID 4788, exit 0, 29,486 ms, peak Job memory 6,845,521,920 bytes under the 12 GiB ceiling, and exact PID cleanup. The combined log SHA-256 is `8da2c7137c0f5234bd5e46ca9621dbf5b0f6db75e220eb7cad2b69dc224991ac`.
- The native bundle `C:\DevStuff\candle-oracle\evidence\native-trace-1.6b-f32-corrected-20260811T214952Z` contains 51 tensors and 182,528,392 safetensors bytes. It records exact input identity, 80 input tokens, 64 projected tokens, exact cache reset, and generated IDs `[1098, 4646, 40027]` / `The image depicts`.
- The phase-contract comparison `C:\DevStuff\candle-oracle\evidence\comparison-1.6b-contract-v3-20260811T220300Z.json` has SHA-256 `9a0b16256a222678f9dce1282660e49fc6d19103cc6dd6a53c824bb58a6412c0`, `passed=true`, 51/51 tensors, and zero failures. Prefill max abs is `0.0009407997131347656` against the `1e-3` CPU-F32 bound; vision layer 26's `0.022125244` elementwise drift is accepted by the documented `>=0.99999` cosine floor after allclose fails, while layer 9 remains accepted by allclose. Exact integer/input stages, structural pixel-unshuffle, projector, post-LN, decode logits, and output IDs pass.
- The final clean postflight `C:\DevStuff\candle-oracle\evidence\p3-close-final-postflight-clean-20260811T221143Z.json` has SHA-256 `1f4399ed6bfbbbf6c6b400054c0cbfebac6fcc8c28ef4d204fdcefbb6fdc4030`, zero tracked model/build processes, 44,067,688,448 available physical bytes, 49,131,601,920 bytes commit headroom, and 23,463 MiB GPU free. P3 is complete; no CUDA claim is made.

### Current source regressions

- `cargo fmt --all -- --check`: green after the final trace fixes.
- Locked/offline `cargo check -j 2` is green for `candle-core`, `candle-nn`, `candle-transformers`, and `candle-vlm`, plus the `lfm2`, `quantized-lfm2`, and `lfm2-vl` examples.
- Locked/offline `cargo test -j 2` is green after the parity fix for the affected core/transformer/VLM workspace lanes: transformer 59/59, generation 5/5, NMS 8/8, VLM 29/29, and all core integration/doc lanes.
- `cargo test --locked --offline -j 2 -p candle-examples --example lfm2-vl`: 29/29 green, including destination-race preservation for native trace publication.
- Scoped strict Clippy is green for the affected libraries and the LFM2-VL example with only `manual-is-multiple-of` and `needless-range-loop` allowed for compatibility/indexing clarity. Two mod-owned `manual_contains` findings were fixed.
- Exact pinned `pytest tools/lfm2_vl/reference -q`: 82/82 green after the phase-specific CPU-F32 comparator regression, shared no-clobber report publication, tokenizer marker/range validation, split-MMProj race preservation, guarded acquisition coverage, and prior GGUF/reference regressions.
- Exact pinned Python compileall and environment/lock verification are green; generated repository Python caches were removed afterward.
- The retained `.venv` cannot launch inside the managed sandbox because its Python 3.10.11 base executable lives under user AppData. The approved native execution replay passed the exact environment/lock verifier, 82/82 tests, and compileall without reinstalling any package.
- Bounded-wrapper smoke: green under PowerShell 7.6.4 and Windows PowerShell 5.1, including suspended assignment, timeout/tree cleanup, owner-exit cleanup, name/executable concurrency, combined logging, and a synthetic process counter above `Int32::MAX`. The owner-exit regression now waits for a child-written post-resume handshake so the test cannot kill its owner before job assignment; failure cleanup also terminates the exact test child.
- Resource-preflight smoke is green under both PowerShell versions. The summary bank also verifies under both versions at SHA-256 `96881cdd368b7990ab8bc447e99c3fee6253d748e22165d3bc0d1f5aeb571263`; all routes remain below 256 KiB after separating native loading, reference environment, GGUF inspection, production parity, exclusive-publication, and direct-main provenance context.
- Relative links pass across all 20 mod-owned Markdown files. A repository-wide diagnostic separately found three pre-existing malformed upstream Qwen example links; they are outside the LFM2-VL publication allowlist and were not changed.
- Current WSL Git inspection is green through the explicit `NVIDIA-Workbench` distribution: local `main`, `HEAD`, and `origin/main` were all `b10c3a0c335050c066d8e02fd9f528f6b490fa39` before this task. Upstream Candle main at `6f74e7c390c717f8fd34f23ce02aceb058173370` was merged without conflict or force; the release overlay is now exactly 94 allowlisted paths, 12 fork-origin modifications, and 82 mod-owned additions.
- The exact Linux-specific native trace collision test is now green in the `NVIDIA-Workbench` WSL2 lane. Broader Linux build/replay coverage remains secondary and is not required for the native Windows product gate.

### S4 integrity audit

- Native Windows `cargo fmt --all -- --check`, locked/offline two-job checks for `candle-core`, `candle-nn`, `candle-transformers`, `candle-vlm`, and the `lfm2`, `quantized-lfm2`, and `lfm2-vl` examples passed. The full `lfm2-vl` example test lane passed 32/32, including the new timing-without-prompt guard.
- PowerShell 7 and Windows PowerShell 5.1 summary-bank verification passed; the mod-owned Markdown link audit found zero missing links; the WSL manifest verifier and `git diff --check` passed. The focused Python reference files compile under WSL's Python 3.10.12 into a temporary cache; the dependency-backed pytest suite was not rerun in this managed shell because neither the system runtime nor the repository `.venv` can launch its pinned Python. The previously recorded pinned 82/82 reference proof remains the applicable functional evidence.
- The exact WSL2 Linux replay for `trace::tests::trace_publication_does_not_replace_a_racing_directory` passed 1/1 with Cargo/rustc 1.97.1 under `NVIDIA-Workbench`; no temporary test directory remained. The review began with local `main` and `origin/main` at the same base commit, and this audit added no unreviewed files.

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

- The official 1.6B artifact, Python/native dry-loads, Python/native component traces, phase-specific cross-runtime comparison, and official 450M native CUDA F32/BF16 gate are green. Lower-bit production CUDA execution and broader WSL portability replay are incomplete.
- Explicit BF16 on a CPU component is unsupported by Candle CPU matmul and is rejected before model load; CPU F32 remains the supported mixed-placement fallback.
- Official config context is 128,000 while model cards advertise 32,768; construction follows config and production policy remains unresolved.
- Official MMProj headers omit tiling metadata; pinned processor configuration or documented architecture defaults remain required.
- The prior llama.cpp residency incident required a host restart. Exact root cause is unproven; F-0008 containment remains mandatory.
- The exact Linux-specific native trace publication regression is green; broader WSL replay remains optional secondary coverage.
- Gknome adoption remains fail-closed on four mature-repository authority conflicts: `.gitignore`, `AGENTS.md`, `README.md`, and `summary_bank.json`.

## Blockers

- No blocker remains for the retained local source/docs slice; the timing
  diagnostic and rejected candidate are fully audited.
- A clean P4.4 optimization proof is currently environment-blocked by an
  unrelated, recurring EdgeSymbio Cargo/rustc tree under
  `C:\Users\jc816\AppData\Local\Temp\es-*`. It is not this worktree and is
  not safe to terminate by identity from this task. Do not start the next
  timing series until a fresh census shows no Cargo/rustc, model, or llama
  process; the existing bounded owners already enforce memory ceilings and
  exact PID cleanup for their own children. The latest identity-filtered
  quiet wait timed out after 120 seconds because the external owner respawned;
  no model run was started by that attempt. See F-0048.
- No blocker remains for the completed P3 CPU-F32 gate. Future model/CUDA runs still require a fresh quiet-host preflight and exact bounded owner; do not overlap any model or build workload.
- Native Windows CUDA/distinct-device proof and official 450M CUDA parity are green. P4.4 optimization must remain sequential, bounded, and preceded by a fresh artifact/executable rehash.
- Hosted GitHub Actions state is intentionally not a blocker or verification dependency under the repository's local-only policy.
- Gknome apply remains blocked until its dry plan has zero unresolved authority conflicts; repair/bypass is not authorized.

## Active Files

- The working tree contains the reviewed SigLIP2 parity fix, phase-contract comparator/test, CUDA cast and contiguous-linear fixes, independent text/vision placement wiring, and documentation/context edits. Snapshots, caches, manifests, logs, owner evidence, and trace output remain external and ignored.
- P3.1–P3.6, P4.1–P4.3, and C1 modular source layout are archived; P4.4 owns the next measured optimization proof.

## Exact Next Task

TODO P4.4 is the exact next task: run the bounded repeated-warm-up decode/cache
microbenchmark against the captured all-CUDA F32 baseline, change one measured
generation bottleneck, replay CPU/CUDA parity, and retain the optimization only
when correctness, variance, and resource contracts remain green.

---
AI-edited: 2026-08-11T22:45:00-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=s4-integrity-audit | change=closed the now-green WSL trace-publication replay and recorded current local verification limits
