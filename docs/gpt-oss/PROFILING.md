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
## First baseline outcome, 2026-10-13

The direct-launch packet at clean `3b16430b`, SHA-256
`36eeaf372ae7f50cfb24a80834d5e15153166af6ca98da64c117dca7a7638dda`,
completed CPU before admitting CUDA. Its retained output root is
`artifacts/gpt-oss/session-profile-3b16430b-direct-4ac8e8b2a67d4690abcb442cb5ad58ed/`.
CPU completed one load/request, sixteen forwards, 23 token steps and sixteen
outputs. Load was 26.960 seconds, eight-token prefill 103.885 seconds and
single-token decode approximately 12.79–13.32 seconds. Total including unload
was 325.672 seconds; peak Job memory was 17,682,280,448 bytes. Job cleanup and
zero registry entries pass. Readback/argmax are small; whole-forward timing
does not attribute this cost to a particular layer or expert kernel.

CUDA's one load failed with `CUDA_ERROR_OUT_OF_MEMORY`; no CUDA request or
forward ran. Registry release and process absence pass. Peak Job memory was
34,353,344,512 bytes against 34,359,738,368. The one-second whole-device samples
peaked at 9,381,609,472 bytes; this cannot identify the failed allocation or
exclude an unsampled peak. CUDA loaded DLL paths were not captured before exit.
CPU's observed runtime paths are retained with the preparation proof.
There is no CUDA timing, speedup, cross-device or consumer qualification.

This allowance is closed with two load attempts, one request, sixteen forwards,
23 token steps and sixteen outputs. The earlier packet `893bb1df…f2994` also
remains closed: an added host observer failed during CPU loading, with one load
attempt and zero requests/forwards; its forced-stop Job/unload receipts are
incomplete. The corrected direct launcher does not reuse that allowance.

The local upload repair consumes CPU weights and releases each source layer
after a completed upload. Fixture proof can establish numerical/ownership
behavior, but cannot prove it resolves the retained-model allocation failure.
The native, fixture, check and strict Clippy repair gates pass locally.
Next: obtain a scoped checkpoint authorization,
bind its release binary and request fresh finite CUDA proof. CPU component
profiling precedes CPU optimization. Later BF16, staged 8k/16k/32k and
Edge/Symbio route qualification retain their own gates. Responses remains
90 seconds and WorkerRuntime remains 600 seconds.

---
AI-edited: 2026-10-13 | agent=Codex/root | model=unknown | effort=unknown | task=gpt-oss-upload-ownership | change=recorded CPU baseline and closed CUDA load failure without a speedup claim
