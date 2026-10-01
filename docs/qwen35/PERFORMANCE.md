# Qwen3.5 4B CPU performance and CUDA assessment

## Reproducible CPU baseline, 2026-09-30

This is a release-profile **test binary** of local Candle `main` at
`0c0bc3ecb7eed3eb22c8fa08a190741c84f2bb56` plus the uncommitted Qwen3.5
overlay. It is not a measurement of the earlier 335.94-second debug parity
job. The exact model is
`D:\CodexStorage\models\qwen35\Darkidol-Ballad-4B.Q8_0.gguf`, 4,482,403,200
bytes, SHA-256
`fd6dcc1ea357b483ae2517b482bf328181e5b371b65661c734aa133407ae2572`.
The runner hashes the file before execution and `Artifact::open` hashes it
again during admission. The external fixture and strict 49-token acceptance
trace are unchanged.

The host is Windows, AMD Ryzen 9 7950X (16 physical/32 logical cores),
`rustc 1.97.1`. The runner sets `CANDLE_NUM_THREADS=4`,
`RAYON_NUM_THREADS=4`, and `OMP_NUM_THREADS=4`; the first of these controls
Candle's quantized barrier pool. Release tests are built offline with two
Cargo jobs and incremental compilation disabled. The target and temp directory
are on D: under `D:\CodexStorage\build\candle-qwen35-release`.

The representative sample uses the first 16 token IDs from the pinned prompt,
then 16 fixed teacher-forced cached-decode IDs (`10..25`). It warms up with
four prompt tokens and two decode tokens, resets, and repeats the 16+16 sample
three times using the same loaded model. Decode IDs are fixed to prevent early
EOS from shortening the timing sample; this is a benchmark-only input change
from the 49-token/two-token acceptance fixture. Prefill remains token-sequential
in this implementation. Each repeat checks that reset reproduces the complete
final Candle raw-logit vector exactly.

The unprofiled control is
`artifacts/qwen35/performance-baseline-unprofiled-1.json` with adjacent stdout
and stderr logs. Mean prefill is **9.52 tokens/s** (three samples 9.505–9.522);
mean cached decode is **9.45 tokens/s** (9.392–9.478). Admission including a
full SHA-256 took 3.685 s and model materialization 1.276 s in this warm
filesystem-cache run; cold-storage loading was not measured. Sampled peak
private memory was 4,541,743,104 bytes. The profiled independent run,
`performance-baseline-profile-2.json`, measured 9.54 prefill and 9.48 decode
tokens/s. It attributed 7.417 s of roughly 10.09 s of sampled inference to
Q8_0 projections, including 1.094 s in the 248,320-output vocabulary head.
Profiling includes tensor creation and output conversion at the projection
call. It does not separate quantized dot time from allocation or scheduling.
The PC may also have been running unrelated game work; the runner did not
record that work's CPU, memory-bandwidth, or disk use. These are reproducible
inputs and observed timings, not an isolated idle-host throughput guarantee.
An exploratory first model run completed but its runner failed while reading
Windows CPU metadata (`Get-CimInstance: Access denied`), leaving only its raw
logs. The runner now uses a registry/environment fallback; all tabulated
measurements above have complete reports. A first candidate rebuild was
blocked from the D: Cargo lock by the execution sandbox and was rerun with
the authorized D: write escalation; the blocked attempt compiled nothing.

Build and benchmark commands (PowerShell from the canonical checkout):

```powershell
& .\scripts\qwen35\run-bounded-performance.ps1 -Mode build -TargetDir 'D:\CodexStorage\build\candle-qwen35-release' -TempDir 'D:\CodexStorage\build\candle-qwen35-release\tmp' -OutputLabel baseline-build
& .\scripts\qwen35\run-bounded-performance.ps1 -Mode benchmark -TargetDir 'D:\CodexStorage\build\candle-qwen35-release' -TempDir 'D:\CodexStorage\build\candle-qwen35-release\tmp' -TestExe 'D:\CodexStorage\build\candle-qwen35-release\release\deps\candle_transformers-7d682dd44d99b7a3.exe' -ModelFile 'D:\CodexStorage\models\qwen35\Darkidol-Ballad-4B.Q8_0.gguf' -OutputLabel baseline-unprofiled-1 -Threads 4
```

The runner samples process memory and drive free space at 500 ms intervals,
with a 900-second model deadline, 12 GiB sampled private-memory ceiling,
16 GiB system-available-memory floor, C: 100 GiB reserve, D: 132 GiB floor
(100 GiB reserve plus a 32 GiB WSL swap allowance), 16 GiB new D: output
ceiling, and 10 MiB log ceiling. These are sampled limits, not hard OS caps.
Reports and logs remain under ignored `artifacts/qwen35/` and are bounded to
less than 100 MiB for this pass.

## First optimization cut

Two reversible candidates were measured with the same exact artifact, release
profile, four threads, warm-up, and 16+16 repeated sample. Background game
load was not controlled, so the differences cannot be assigned solely to the
code edits:

| Candidate | Evidence | Prefill mean | Decode mean | Result |
| --- | --- | ---: | ---: | --- |
| Existing scoped Q8_0 path | `performance-baseline-unprofiled-1.json` | 9.52/s | 9.45/s | Retained |
| Static row ranges for scoped Q8_0 single-row matmul | `performance-scoped-static-unprofiled-1.json`, `performance-scoped-static-profile-1.json` | 8.84/s | 9.18/s | Reverted; observed projection time rose from 7.417 to 7.774 s in profiled runs |
| Reuse recurrent delta scratch across value heads | `performance-delta-reuse-unprofiled-1.json`, `performance-delta-reuse-unprofiled-2.json` | 9.46/s across six samples | 9.39/s across six samples | Reverted; no repeatable end-to-end gain |

The Q8_0 scheduler change followed the existing compact Q6K single-row
pattern, but these runs do not support replacing the dynamic cursor. The
recurrent buffer change eliminated repeated small allocations but its measured
effect was within run-to-run variation. Neither candidate is retained in the
production path. The next useful CPU experiment should separate quantized dot
work, tensor wrapping/copying, and barrier scheduling within the projection
hotspot before changing kernels. An optimized short-input throughput run must
continue to use a sustained non-EOS decode sequence and independently pass the
strict acceptance trace.

## CUDA feasibility and smallest proof

`nvidia-smi` reported an RTX 4090, 24,564 MiB total / 23,401 MiB free at
assessment time, driver 616.92, compute capability 8.9. NVIDIA lists the
[RTX 4090 as a 24 GB card](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/)
and [capability 8.9](https://developer.nvidia.com/cuda/gpus). The exact
4.48 GB GGUF should fit in that VRAM for a short proof, subject to measured
weight layout, workspace, and cache overhead; no Qwen GPU allocation or CUDA
build was attempted in this pass.

The fork already has generic CUDA quantized embedding, Q8_0 dequantization,
Q8_0 matrix-vector and matrix-matrix kernels in
`candle-core/src/quantized/cuda.rs`. Dense F32 CUDA tensor operations are
available through Candle's generic backend. They are **not connected to this
Qwen3.5 model**: `CpuModel` loads weights on `Device::Cpu`, the projection
wrapper creates a CPU tensor and converts every result to a host `Vec<f32>`,
and convolution FIFO, recurrent state updates, normalization, RoPE,
full-attention KV and softmax all run in host loops. The current model has no
device-resident state, CUDA dispatch, or end-to-end GPU parity proof. The
generic CUDA Q8_0 matrix-vector default quantizes its activation to Q8_1;
the proven CPU path uses scoped GGML-compatible Q8_0 activation arithmetic.
That numerical difference needs direct component evidence before GPU parity
can be assumed.

Offloading each of the 249 measured projections per token separately while the
other operations stay on CPU would force frequent host/device transfers and
kernel launches. NVIDIA's [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
recommends minimizing such transfers and retaining intermediates on device.
The 248,320-element vocabulary output alone is about 0.99 MiB of F32 data
per token if copied whole. This is a feasibility inference from source, not a
measured CUDA bottleneck.

The smallest GPU proof is a **single pinned Q8_0 projection**, without
changing the CPU model: load one exact weight to GPU, use a captured CPU
activation, compare its full output vector to the pinned CPU result, and
measure model transfer, H2D input, kernel, and D2H output separately. Compare
the default Q8_1-activation kernel and the existing dequantize-matvec fallback
to see which, if either, can meet the component numerical requirement. Check
VRAM, memory growth, and behavior across short repeated calls. If that passes
with a useful measured margin, the next implementation boundary is one hybrid
layer with its recurrent or attention state kept on device and component
parity at each operation. Full model CUDA, long context, and application
activation require separate scope after that proof. The current CPU fixture's
top-10 and 0.01 raw-logprob acceptance threshold must remain unchanged.

### First CUDA projection proof (2026-10-01)

The offline native Windows CUDA release build completed in 230.15 seconds in
`D:\CodexStorage\build\candle-qwen35-release`. The bounded ignored proof
reverified the exact 4,482,403,200-byte GGUF SHA-256
`fd6dcc1ea357b483ae2517b482bf328181e5b371b65661c734aa133407ae2572`,
captured the first fixture token's layer-zero post-attention normalization,
and compared all 9,216 values of its Q8_0 FFN gate projection to the scoped
CPU path. `artifacts/qwen35/cuda-projection-{build,proof}-1.{json,stdout.log,stderr.log}`
contains the bounded reports and raw test output.

| CUDA path | Max absolute output error | RMS error | Three synchronized kernel calls |
| --- | ---: | ---: | --- |
| Default Q8_1 activation | 0.000000298 | 0.0000000387 | 0.0285, 0.0275, 0.0206 ms |
| Forced dequantized matvec | 0.004300 | 0.001013 | 0.0280, 0.0287, 0.0258 ms |

One weight upload took 8.932 ms, input H2D 0.034 ms, and output D2H about
0.05 ms per selected path. The job completed in 11.19 seconds with 5.61 GB
sampled peak process private memory. The RTX 4090 had 26-28% unrelated
utilization, so timings are contended observations. No long-run memory-growth,
whole-model CUDA, reset, or cancellation qualification follows from this
single-projection result. The default CUDA component path is now guarded by a
`1e-5` maximum absolute error assertion; the forced fallback remains a
diagnostic comparison. This component threshold does not alter the original
top-10 and 0.01 raw-logprob acceptance fixture.

### Scoped CPU-order CUDA result

The direct whole-model hybrid attempt with generic CUDA Q8_1 activation
failed the unchanged fixture: maximum raw-logprob error was 0.2342, and the
second top-10 token set differed. Layer-zero recurrent QKV was the first
divergence. Its activation had two byte differences under the generic CUDA
quantizer. Scoped nearest-even quantization reduced the QKV max error from
0.001682 to 0.00000477, but the full trace still missed with 0.2332 maximum
raw-logprob error. The scoped eight-lane CPU-order FMA CUDA kernel made the
first token's QKV and all 32 layer outputs bit identical to CPU. These failed
diagnostics are retained under ignored `artifacts/qwen35/` with labels
`hybrid-parity-1`, `component-first-token-1`, `qkv-compare-1`,
`activation-quant-diff-1`, and `scoped-hybrid-parity-1`.

The final scoped projection proof, `cuda-projection-final-1.*`, was bit exact
against the full 9,216-value CPU gate output. Weight H2D was 6.687 ms,
activation H2D 0.036 ms, three synchronized CPU-order kernel calls 0.126,
0.126, and 0.119 ms, and output D2H 0.040 ms. The forced dequantized fallback
remained at 0.00430 maximum error and about 0.025-0.028 ms per call. The
earlier generic Q8_1 kernel was also about 0.021-0.029 ms, so exact CPU-order
arithmetic has a clear cost for this projection. Background PC work varied;
these calls are observations, not a clean isolated throughput comparison.

The whole-model GPU-resident-projection/host-state hybrid passed the unchanged
49-token prefill and cached-decode reference: IDs `[19, 248046]`, EOS,
identical top-10 sets, maximum absolute raw-logprob error
`0.00005269050598144531` (limit `0.01`), and exact reset replay. The final
bounded `cuda-recovery-parity-1.*` job took 19.40 s with 5,407,723,520
sampled peak private bytes. Cancellation at prefill check eight reset partial
state, and the next full prompt reproduced the first logits. CUDA free memory
reported the same 19,646,119,936 bytes after model load, after replay, and
after cancellation replay; this is evidence of no growth across those short
sessions, not a long-running leak guarantee. The hybrid rejects a CPU device,
and the scoped Q8_0 kernel rejects an unsupported two-row call. The
final-source CPU fixture separately passed in 16.60 s under
`cuda-final-cpu-parity-1.*`.

This hybrid keeps quantized projections resident on CUDA, but performs
recurrent state updates and attention KV on CPU. Each projection transfers
its input and output, so transfer/launch overhead remains a design limit.
It does not yet establish a fully device-resident hybrid layer or an Edge
product runtime. No further throughput claim is made from the acceptance
jobs; the earlier 335.94-second verification was likewise not a benchmark.

The final capability cut rejects the global forced-dequantization switch
inside the scoped Q8_0 CUDA call, alongside unsupported shapes. Its rerun
`cuda-failclosed-projection-1.*` kept the 9,216 outputs bit exact, while the
generic unspecialized forced fallback remained a separate diagnostic with
0.00430 maximum error. `cuda-failclosed-parity-1.*` passed the same strict
trace, reset, cancellation, and short-session memory gate in 19.87 s. The
shared GPT-OSS focused suite passed again (46 passed, one owner-artifact test
ignored), and the final locked/offline CUDA core/transformers check passed.

## Sealed 1,024-token Edge latency pass, 2026-10-01

The Edge owner supplied a sealed 1,024-ID input at
`D:\CodexStorage\build\edgesymbio-qwen35-short-20261001\proofs\context-gate-1\prompt-token-ids.json`
(SHA-256 `fceba3897357a86a2e9063b38e528549ac5df93661487f2f7fda0789e014fbdb`)
and `gate.json` (SHA-256
`8b1996d6347c936b4410f393b87732ad21e19ef56d2a1054d1df284d622ada4e`).
The Edge handback measured 176.392 s generation with a separate 11.310 s
load, above its 90 s response deadline. That was a different application
binary/run, so it is a motivation and acceptance input, not a matched Candle
performance baseline. The bounded Candle runner rehashes both inputs and the
exact GGUF before every context proof. It admits 1,088 positions and
134,217,728 state bytes. The unchanged 49-token reference remains a separate
strict gate.

Native Windows release builds used CUDA 13.3, an RTX 4090, four Candle/Rayon/
OMP threads, offline Cargo with two build jobs, and the existing D: target.
The test times model admission, materialization, 1,024-token prefill, and
one cached decode separately. Its stage counters are test-only and nested:
`mixer.total` includes projection calls, while `q8_0` includes input transfer,
kernel dispatch, and output readback/synchronization. The `projection.kernel`
counter measures dispatch/enqueue, not an isolated GPU kernel duration; the
blocking readback includes device work. The standalone projection proof uses
explicit synchronization for its per-call kernel comparison. Unrelated game
work occupied the GPU during these runs; the measurements are contended.

| Release source cut | Prefill | Tokens/s | Stage evidence |
| --- | ---: | ---: | --- |
| Existing hybrid, 1,024 `decode` calls | 115.846 s | 8.84 | Q8_0 calls 65.441 s; attention score/value loop 21.530 s |
| Token-major attention value accumulation | 104.589 s | 9.79 | Attention loop 10.334 s; projections 65.490 s |
| Last-token-only prefill vocabulary output | 101.560 s | 10.08 | Vocabulary projections fell from 1,025 to 2 including cached decode; recurrent state 22.316 s |
| Row-major recurrent matrix-vector access | 89.272 s | 11.47 | Recurrent state 10.062 s; projections 62.318 s |
| Eight-thread-per-row scoped Q8_0 CUDA kernel | 50.751, 50.630, 50.621 s | 20.18-20.23 | Projections 23.907 s in first run; independent reset replay 50.580 s |

The first baseline used individual `decode` calls because the original
`prefill` implementation invoked `decode` for each token. The final test uses
the public `prefill` method, which now skips vocabulary output until the last
token. All cuts used the same sealed token sequence and model. The attention
and recurrent cuts read contiguous rows but keep GGML's 64-lane FMA and
reduction order; deterministic unit tests compare every output bit against the
original dot calculation for multiple dimensions and sequence lengths. The
scoped CUDA kernel assigns eight threads to one row, one per independent FMA
lane, then shuffles values into the unchanged final reduction. Its 9,216-value
projection proof was bit exact; three synchronized calls took 0.040, 0.042,
and 0.045 ms versus about 0.12 ms for the earlier one-thread-per-row proof.

All final context runs produced `[19, 248046]` with the same top-10 token
sets as the Edge gate. Maximum checked raw-logprob error was 0.0002502 for
the first selection and 0.0000133 for the second, below the unchanged 0.01
limit. Cached decode measured 68-72 ms. In a bounded repeated-session run,
the complete first and cached-decode raw-logit vectors replayed exactly after
reset; cancellation at check 512 cleared state, and the next complete prompt
replayed exactly. Sampled CUDA free memory was 19,646,119,936 bytes after
the first, replay, and cancellation-recovery sessions. The runner's peak
private memory was 5,476,909,056 bytes across that job. This is a bounded
three-session observation, not a long-term leak guarantee.

Reports and logs are ignored under `artifacts/qwen35/` with labels
`cuda-context-profile-proof-1`, `cuda-context-attention-values-1`,
`cuda-context-final-head-1`, `cuda-context-recurrent-row-1`,
`cuda-context-eight-lane-{1,2}`, and `cuda-context-eight-lane-repeat-1`.
The final 9,216-value component proof is `cuda-eight-lane-projection-1`;
the short strict fixture/reset/cancellation gate is
`cuda-eight-lane-short-1`. The CPU exact fixture passed under
`cuda-context-optimized-cpu-proof-1`. The final-source Qwen unit tests passed
7/7 with four external tests intentionally ignored; the shared GPT-OSS CUDA
regression passed 46/46 with one owner-artifact test ignored.

Example bounded profile command from PowerShell at the canonical checkout:

```powershell
& .\scripts\qwen35\run-bounded-cuda.ps1 -Mode proof -TargetDir 'D:\CodexStorage\build\candle-qwen35-release' -TempDir 'D:\CodexStorage\build\candle-qwen35-release\tmp' -OutputLabel context-eight-lane-repeat-1 -TestFilter hybrid_cuda_sealed_context_profile -TestExe 'D:\CodexStorage\build\candle-qwen35-release\release\deps\candle_transformers-6573b7fa83f47a17.exe' -ModelFile 'D:\CodexStorage\models\qwen35\Darkidol-Ballad-4B.Q8_0.gguf' -ContextTokensFile 'D:\CodexStorage\build\edgesymbio-qwen35-short-20261001\proofs\context-gate-1\prompt-token-ids.json' -ContextGateFile 'D:\CodexStorage\build\edgesymbio-qwen35-short-20261001\proofs\context-gate-1\gate.json' -ContextRepeat
```

The runner checks 25 GiB C: and 100 GiB D: floors, 16 GiB available system
memory, 12 GiB sampled process private memory for proofs, 16 GiB D: output
growth, 10 MiB combined logs, and a 600 s proof deadline. These are sampled
limits, not hard OS process caps. The relevant Edge owner must rebuild its
application against these local Candle changes and rerun the real 90-second
Responses gate, including measured application overhead and resource cleanup.
No normal provider selection follows from the Candle timing result alone.
