# Qwen3.5 Compatibility Status

## Scope and current state

Pre-overlay Candle baseline: `main` at
`0c0bc3ecb7eed3eb22c8fa08a190741c84f2bb56`. Active files are `candle-transformers/src/models/qwen35/`,
the scoped Q8_0 CPU/CUDA core and CUDA kernel paths, model registry, fixture,
bounded runners, and overlay records. Bounded Q8_0/F32 admission, the native
CPU reference gate, and the exact-artifact GPU-resident-projection/host-state
hybrid gate pass. No Edge product activation or 27B/IQ/MTP support is claimed.

The pinned Edge checkout is
`C:\Users\jc816\OneDrive\Desktop\Gen-App\EdgeSymbio` at
`37d475efa873841cd31751a29b3e29d7804197fa`. Its retained proof root is
`runtime-data/proofs/candle-qwen35-20260930`. The local GGUF was verified at
4,482,403,200 bytes and SHA-256
`fd6dcc1ea357b483ae2517b482bf328181e5b371b65661c734aa133407ae2572`
before the user's storage cleanup. It was subsequently deleted from
`C:\llamacpp\models\Codename-B-8B-Instruct.Q4.gguf`. With explicit owner
approval, the exact public replacement was downloaded from
`mradermacher/Darkidol-Ballad-4B-GGUF` revision
`0fc1207385c1655f27e2bb93c72115ee8ac12369` to
`D:\CodexStorage\models\qwen35\Darkidol-Ballad-4B.Q8_0.gguf`. The complete
download measured 4,482,403,200 bytes and the same SHA-256. Its published LFS
identity matches; this does not establish the deleted file's original download
history. `Artifact::open` rehashes the whole file before each model test.

The five retained Edge proof hashes matched the handoff on this PC:

| File | SHA-256 |
| --- | --- |
| `codename-b.inspection.json` | `a9de899468afd3b407c3091680cf80a0403902f5e3388a435350330db9343d1c` |
| `qwen3-8-27b.inspection.json` | `16fed57be7297aa1b2b671d6761bea39ec7779e4f1630def96cdabb140091d1e` |
| `reference.json` | `248ab522e2ef472ff436db5fed65256125f8cd0d390bf7a7130e60cf8117572c` |
| `reference-job.json` | `a759f8aad29bb45fa9a47e988aea2b085af04355744b058c1716de30ceedc9b5` |
| `capture_reference.py` | `5a21f64ab1942c74c86936b65562b4e476dfb1b5fe4b0cdf0e7c8a3106e00b57` |

The copied Candle fixture has SHA-256
`0c5315e7289e719a9500c94a0faaa4ef7d42c8a4429d1a702ca88429ead121a9`,
matching the pinned Edge checkout byte-for-byte.

## Proven behavior and latest verification

- Admission rejects wrong architecture, missing/extra tensors, invalid
  ranges, shapes, dtypes, and resource budgets. The hash-pinned external
  inventory has 426 tensors: 249 Q8_0 and 177 F32.
- The CPU path uses token-sequential prefill, converted V-head ordering with
  tiled Q/K repetition, recurrent convolution FIFO and matrix state,
  full-attention KV state, and position reset. Its Q8_0 matmul call is scoped
  to GGML-compatible 32-value activation arithmetic; other Candle callers
  retain their existing path.
- A pinned llama.cpp `ubatch=1` 8-token layer capture and a bounded Candle
  token-7 capture match bit for bit through all 32 layer outputs after the
  GGML F32 attention score, softmax, and value-dot arithmetic changes. This is
  a diagnostic component result for that token, not broader full-vector
  qualification. Evidence is under ignored `artifacts/qwen35/` with label
  `layers-scoped-token-7-kqv-layer19` and reference label
  `ubatch-1-tokens-8-attn27`.
- The strict exact-artifact fixture run passed on native Windows CPU:
  49-token prefill, cached decode `[19, 248046]`, EOS, identical top-10 token
  sets at both positions, and maximum absolute raw-logprob error
  `0.00005283917236376112` across 20 compared token IDs (limit `0.01`).
  After unrelated `[10,20,30]` input, reset exactly replayed the first
  Candle raw-logit vector. The runner completed in `335.94` seconds with
  `4,548,972,544` sampled peak private bytes and `40,570,953,728` bytes
  free on C:. Its deadline was 480 seconds, with sampled memory and disk
  floors; those checks are not a hard memory cap. Evidence:
  `artifacts/qwen35/parity-parity-scoped-attention-arithmetic-{job.json,stdout.log,stderr.log}`.
  That saved report's legacy `cpu_path=vnni` names the default process mode;
  its `model_q8_matmul=scoped_ggml_q8_0` records the actual model path. The
  runner now reports `cpu_mode=process_default` to avoid that ambiguity.
- `cargo test --locked --offline -p candle-transformers qwen35 --lib` passed:
  5 tests, 3 external-model tests ignored. The exact header inventory passed
  separately after 85.96 seconds; the strict fixture passed under the bounded
  runner. `cargo test --locked --offline -p candle-core --lib` passed 27/27;
  `cargo test --locked --offline -p candle-transformers --lib` passed 134/134
  with 4 external-model tests ignored.
- `cargo fmt --all -- --check`, locked offline `cargo check` for `candle-core`,
  `candle-nn`, `candle-transformers`, `candle-vlm`, and the
  `quantized-lfm2` example passed. The summary-bank verifier, LFM2-VL
  mod-manifest verifier at its declared upstream baseline, and repository
  fork-overlay verifier at the task's rolling baseline passed. `git diff
  --check` passed. The complete local source and document diff was inspected.
  An initial LFM2-VL manifest invocation with the task's rolling baseline
  failed its fixed 17/146 upstream-delta count; using that verifier's declared
  upstream baseline passed 17/146. This was a baseline argument error, not a
  source or test failure.

## Release CPU performance and CUDA assessment

The bounded native release benchmark is documented in `PERFORMANCE.md` with
exact commands, warm-up, sample inputs, artifact hash, thread settings,
loading phases, throughput variation, sampled memory, and report paths. The
unprofiled 16-token prefill and 16-token cached-decode control averaged 9.52
and 9.45 tokens/s over three repeats. All 23,904 measured projection calls
over those repeats used Q8_0 weights and accounted for about 7.417 of 10.09
inference seconds in the profile. The benchmark's fixed decode IDs and short
prompt are for throughput measurement; the pinned acceptance fixture remains
unchanged. The static Q8_0 scheduling and recurrent scratch reuse candidates
were reverted after matched runs did not establish a safe throughput gain.
Unrelated game work on the PC was not measured during those runs, so the
timing differences are not an isolated code-effect estimate.

The final reverted-source release test build passed in the bounded D: target.
Its strict post-benchmark parity replay passed in 18.48 seconds: 49-token
prefill, cached decode `[19, 248046]`, EOS, identical top-10 sets, maximum
absolute raw-logprob error `0.0000528391723637611` across 20 IDs, and exact
reset replay after unrelated input. Sampled peak private memory was
4,545,609,728 bytes. Evidence is
`artifacts/qwen35/parity-parity-release-final-performance-{job.json,stdout.log,stderr.log}`.
The release binary's targeted Qwen3.5 tests passed 5/5 with four external
tests ignored. The serial locked offline native check passed for `candle-core`,
`candle-nn`, `candle-transformers`, `candle-vlm`, and the `quantized-lfm2`
example in the D: target. Its first attempt was interrupted by an execution
transport disconnect after compiling dependencies and had no usable
completion result; the resumed command completed successfully in 1m 36s.
The summary-bank verifier, LFM2-VL manifest verifier at its declared upstream
baseline, and fork-overlay verifier at the rolling baseline passed after the
performance document and route changes. `cargo fmt --all -- --check` and
`git diff --check` passed. The RTX 4090 feasibility assessment found
generic Q8_0 CUDA kernels but no Qwen3.5 device-resident forward path; no CUDA
build or execution was attempted. Cancellation is not exposed by this
synchronous CPU model, and the short repeated benchmark only establishes a
sampled peak memory, not a long-run leak bound.

## 2026-10-01 CUDA correctness checkpoint

- The Windows CUDA 13.3/MSVC release build passed in the bounded D: target.
  `HybridGpuModel` is gated by the `cuda` feature and a CUDA device. It keeps
  Q8_0 projection weights on device and scalar/recurrent/attention state on
  host. The generic CUDA Q8_1 arithmetic failed the unchanged full fixture;
  its first recurrent QKV mismatch was traced to two activation bytes and
  reduction order. The scoped Q8_0 CUDA kernel now reproduces the pinned
  AVX2/FMA CPU order. The first token's 32 layer outputs were bit identical.
- The final exact 4B CUDA fixture passed 49-token prefill, cached decode
  `[19, 248046]`, EOS, identical top-10 sets, and maximum absolute raw-logprob
  error `0.00005269050598144531` across the reference comparisons (limit
  `0.01`). Reset and post-cancellation replay matched the first CUDA logits
  exactly. Cancellation was observed at prefill check eight. CUDA free memory
  remained `19,646,119,936` bytes at the sampled loaded/replay points; the
  bounded job recorded `5,407,723,520` peak private bytes over 19.40 s.
  The final fail-closed source replay passed the same gate in 19.87 s with
  `5,407,240,192` sampled peak private bytes. Evidence: ignored
  `artifacts/qwen35/cuda-{recovery,failclosed}-parity-1.*`.
- The final projection test matched all 9,216 CPU outputs bit for bit, and
  rejected the unsupported two-row scoped call and forced dequantization in
  the scoped call. The generic forced fallback remains a diagnostic path.
  Evidence: `artifacts/qwen35/cuda-failclosed-projection-1.*`. The native CPU
  strict fixture passed separately in 16.60 s under
  `artifacts/qwen35/cuda-final-cpu-parity-1.*`.
- The CUDA build's synthetic Qwen tests passed 5/5 (seven external tests
  ignored), and the shared-FFI GPT-OSS focused tests passed 46/46 (one owner
  artifact test ignored) after the final fail-closed source edit.
  Locked/offline CPU checks passed for `candle-core`,
  `candle-nn`, `candle-transformers`, `candle-vlm`, and the `quantized-lfm2`
  example; locked/offline CUDA checks passed for `candle-core` and
  `candle-transformers`, with the final CUDA-only edit checked again under
  `artifacts/qwen35/cuda-failclosed-cuda-check-1.json`.
  `cargo fmt --all -- --check`, `git diff --check`,
  and `scripts/verify-fork-overlays.sh --rolling-baseline
  0c0bc3ecb7eed3eb22c8fa08a190741c84f2bb56` passed. WSL Bash was
  denied by the host, so the overlay verifier used the installed Git Bash
  with process-local `/usr/bin:/mingw64/bin` PATH. No publication occurred.

## 2026-10-01 sealed context latency checkpoint

The separate Edge owner completed a distinct short-lab Qwen4B CPU/CUDA
consumer path and a symbio-code diagnostic receipt pass. Its initial native
1,024-token context replay passed tokens/logits but took 176.392 s generation,
above the 90 s Responses deadline. The normal provider remains disabled.
Those application results are owner handback evidence, not checks executed in
this Candle task; see `INTEGRATION.md` for the exact handoff boundary.

On the same hash-pinned GGUF and sealed Edge token IDs, the new Candle release
profile first measured 115.846 s prefill. The measured host attention and
recurrent state loops, intermediate vocabulary projection work, and the
scoped Q8_0 CUDA kernel were optimized in small cuts while retaining the
original 64-lane FMA reduction order. The final scoped projection proof is
bit exact across all 9,216 values. Three final-source context runs measured
50.751, 50.630, and 50.621 s prefill plus 68-72 ms cached decode. All matched
the Edge selected IDs `[19, 248046]`, top-10 token sets, and a maximum checked
raw-logprob error of 0.0002502 (limit 0.01). One additional full reset replay
and post-token-512 cancellation replay were exact. CUDA free memory stayed at
19,646,119,936 bytes across sampled context sessions. Exact commands,
conditions, per-stage evidence, and caveats are in `PERFORMANCE.md`.

The final-source strict 49-token GPU trace, reset, and cancellation gate
passed under `artifacts/qwen35/cuda-eight-lane-short-1.*`; the native CPU
strict trace passed under `cuda-context-optimized-cpu-proof-1.*`. Qwen unit
tests passed 7/7 (four external tests ignored); shared GPT-OSS CUDA regression
passed 46/46 (one owner-artifact test ignored). Locked/offline native CPU
checks passed for core, nn, transformers, VLM, and the `quantized-lfm2`
example; CUDA checks passed for core and transformers.
`cargo fmt --all -- --check`, `git diff --check`, and the fork-overlay verifier at rolling
baseline `0c0bc3ecb7eed3eb22c8fa08a190741c84f2bb56` passed. The shared
GPT-OSS regression and those checks were executed on final source; only
documentation was updated afterward. No commit or publication occurred.

## 2026-10-01 source closeout

The complete native locked/offline repository gate passed after a bounded
strict-Clippy repair: private observer aliases and equivalent iterator
spelling preserve arithmetic and order. The transformer suite passed 138
tests with five external-model tests ignored; all maintained library/example
checks, strict Clippy, formatting, summary/layout/overlay verifiers, and
whitespace passed. The log is retained outside Git as
`%TEMP%/codex-candle-closeout-20261001-gate-final.log`.

This gate is source/unit-test evidence. The retained exact-artifact CPU/CUDA
and context receipts above keep their original tested source identities;
no production model, CUDA, or application qualification was rerun here.
Separate local overlay checkpoints are prepared for review. Publication is
blocked on a supported secure authentication route for the owned helper.

## Known limits and exact next task

This pass qualifies only the exact Q8_0/F32 4B artifact, strict 49-token
reference and one sealed 1,024-token context input. The first-token layer
outputs and one Q8_0 projection have full-vector GPU/CPU equality; broader
component vectors, other contexts, fully device-resident recurrent/attention
execution, and long-running memory remain unqualified. Cancellation is at
token boundaries. EdgeSymbio/symbio-code have a distinct short-lab integration
and diagnostic pass, but normal provider activation, tool/coding/output
capacity, and an end-to-end 90-second application gate on this optimized
Candle source remain unqualified. The exact next milestone is an Edge owner
rebuild and sealed 1,024-token Responses rerun under the existing 90-second
deadline with resource/receipt checks, followed by symbio-code consumer
qualification. The owner authorized this closeout and publication round; the qualified
source checkpoint remains local until its secure publication route is available.
