# GPT-OSS maintained-session baseline profiling

The first maintained baseline characterizes the public `GptOssSession` on CPU
then CUDA, using F32 activations and retained GGUF MXFP4 experts. The historical
CUDA-only performance example, wrapper and receipts stay at their original
identities. This new runner is preparation for a fresh finite allowance; local
fixture tests never authorize production inference.

## Frozen first workload

Each device runs one fresh load and one raw-token request: eight prompt tokens
from `reference_prefix_token_ids` in the pinned
`tests/fixtures/gpt_oss_task3_short_parity/reference.json`, followed by sixteen
greedy outputs. There are no warmups, retries, downloads or provider activation.
The first output comes from prefill, then fifteen decode calls: sixteen public
forwards and 23 internal token steps per device, with a 24-token admission
ceiling. The two-device allowance is at most two loads/two requests, 32 public
forwards, 46 token steps and 32 generated tokens. A failure stops the batch.
These raw IDs do not require tokenizer I/O and establish no Harmony qualification.

The retained model remains `C:\llamacpp\models\gpt-oss-20b-mxfp4.gguf`,
13,792,638,656 bytes, expected SHA-256
`aab205256a9b6361e410c24de3086e30f907092ca6f9ba8cd4b22c8a2b025778`.
Presence/size inspection is preparation only. The admitted retained handle
rehashes the model under the approved native Job before parsing or execution.
Official packed-model production qualification remains separate.

## Measurement contract

`prefill_profiled`/`decode_profiled` return device-native logits and successful
forward wall time. Normal session calls do not read a profiling clock. CUDA
also measures the final synchronize/host-transfer/finite-check block before
cache commit. That block includes waiting for outstanding GPU work; its duration
is not isolated DMA latency or a per-kernel measurement. The cache commit,
finite validation and cancellation order are preserved. CPU reports this CUDA
field as absent.

`gpt-oss-session-profile` separately records caller `into_cpu` readback and
argmax time, identity-bound consumed files, construction/unload time, static
residency, cache position and work counts. Failed forward attempts retain an
attempted-token upper bound; they do not claim all partial internal work was
observed. A completed forward remains counted if later readback fails. Output
is a characterization record, with no invented speed or accuracy threshold.

## Native launch and release

`scripts/gpt-oss/run-session-profile.ps1` requires a reviewed packet binding
exact clean source commit/tree, CPU/CUDA executable identities, per-device
input files and the Candle bounded helper identity. Release builds are required;
the runner rejects debug execution before artifact I/O. It verifies bindings before
launching and creates only a fresh `artifacts/` subdirectory. CPU executes first;
CUDA launches only after CPU has complete work and release receipts.

The existing `run-bounded-oracle.ps1` starts the model process suspended and
assigns a native kill-on-close Job before resume, with 32 GiB host memory and a
1,800-second limit per device. CUDA also has an absolute 20,000,000,000-byte
physical GPU monitor with one-second sampling. CPU requires no NVIDIA tool or
library. The GPU monitor can miss peaks between samples; logical accounting
does not replace physical observation. Monitor failure terminates only the owned controller; closing its Job
kills the owned model process. Owner state retains PID/creation time/executable,
inputs, output root, lifetime and shutdown route. Complete Job and registry
release records are required; a forced stop cannot become a success claim.

Build and test with locked/offline Cargo and the native model-free script.
Production measurements need a newly approved packet at a clean checkpoint.
No baseline has executed yet. Later measured optimizations, BF16, staged
8k/16k/32k envelopes and Edge/Symbio route qualification retain their own gates;
Responses remains 90 seconds and WorkerRuntime remains 600 seconds.

---
AI-edited: 2026-10-12 | agent=Codex/root | model=unknown | effort=unknown | task=gpt-oss-profile-preparation | change=specified bounded common-session measurement without production execution
