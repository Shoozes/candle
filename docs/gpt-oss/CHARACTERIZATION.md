# GPT-OSS common-session characterization

This diagnostic contract is implemented by `gpt-oss-characterize` and
`scripts/gpt-oss/run-characterization.ps1`. It preserves the historical
eight-token/sixteen-output profiling contract and receipts unchanged. It is
preparation for a fresh finite allowance, not consumer or provider qualification.

## Frozen work

Each backend constructs one retained GGUF/F32 `GptOssSession`. Cold timing runs
first: eight pinned tokens, then fifteen greedy decodes, with sixteen outputs.
Reset, prefill the first five pinned tokens and decode fixed tokens six and
seven. Reset and replay that same 5+1+1 sequence. Numerical segments sample no
output tokens. Capture six full F32 rows, compare the first three against the
independent oracle and the second three against the first segment.

| Work | Per backend | CPU then CUDA total |
| --- | ---: | ---: |
| Loads | 1 | 2 |
| Request segments | 3 | 6 |
| Public forwards | 22 | 44 |
| Processed input token steps | 37 | 74 |
| Sampled outputs | 16 | 32 |
| Captured rows / resets | 6 / 2 | 12 / 4 |
| Warmups / retries | 0 / 0 | 0 / 0 |

The original reference fixture fixes positions 5/6/7, vocabulary 201,088,
top-1 agreement, top-eight-union log-probability error at most 0.25, full-row
mean error at most 0.01 and probability total variation at most 0.01. Replay
requires maximum raw-logit error at most `1e-5`. The shared decoder retains the
reference's max-minus-16 tail clamp. Row descriptors bind prefix IDs/digest,
position, vocabulary, F32 little-endian dtype, source/model identity, byte count
and digest. Missing, truncated, nonfinite or mismatched rows fail.
`--verify-progress` rereads the rows and recomputes both comparisons without
loading a model; producer summary numbers alone cannot qualify a run.

Historical timing, independent numerical agreement and same-runtime replay
are separate evidence. CUDA validation time remains nested inside forward
time. Load/unload, readback, argmax, capture/comparison, reset and observer
overhead are distinct. One run supplies descriptive timing, not a causal
speedup estimate or consumer deadline/capacity proof.

## Admission, observation and release

Packet version `candle.gpt-oss.characterization-packet.v1` binds exact clean
commit/tree, release binaries, build records/source closure, lock/toolchain,
inputs, retained model, oracle, helper/probe/watcher bytes and expected loaded
DLL paths/digests. Draft packets are rejected. Preparation builds CPU then CUDA
inside separate suspended-before-resume native Jobs, each 16 GiB/900 seconds.
Two additional no-model startup/codec probes use 2 GiB/15-second native Jobs.
The CUDA executable's DLL imports commit about 294 MB before model work,
exceeding the small observer Job; that rejected probe remains model-free evidence.
Execution requires a newly reviewed final packet and separate owner allowance.

The model retains the existing 32 GiB/1,800-second native Job and sampled
20,000,000,000-byte whole-GPU ceiling. CPU completion, numerical checks and
Job/registry/watcher release precede CUDA admission. An observation, comparison,
deadline or release failure stops the batch. Failure evidence stays incomplete.

GPU and process/module probes are separate kill-on-close 256 MiB native Jobs,
assigned before resume, with five-second deadlines and independently capped
stdout/stderr. GPU output is limited to 4 KiB; module output to 128 KiB and 256
modules. Concurrent drains avoid redirected-pipe deadlock. These additional
model-free probes and row verification make zero model forwards and report
their overhead separately; they can still affect descriptive timing. One-second
GPU sampling may miss peaks. The watcher has a finite 1,860-second lifetime;
its exact process handle, identity and shutdown route are recorded.

Row revalidation uses the CPU release executable on both devices, with the
same F32 helper. CUDA metadata is accepted for host verification; CPU builds
still reject actual CUDA execution before weight I/O. Each CUDA lane temporarily
prepends its declared DLL directories and restores PATH after success or error.

Runtime qualification requires observed loaded paths, not installed candidates.
The selected Windows/MSVC CPU binary imports `VCRUNTIME140.dll`; CUDA additionally
requires `nvcuda.dll`, `cublas64_13.dll`, `cublasLt64_13.dll` and
`curand64_10.dll`. Required module sets and byte identities are checked before
launch and against actual observations. OS modules outside this selected
closure remain inventory rather than a claim that every DLL was packaged.

## Short outcome at `e3e1afcb`, 2026-10-13

Fresh packet SHA-256
`521d4295f226c03b02e14ebba02296ee6d7e564b163ead1a69bc9f31063ee33b`
completed at clean `e3e1afcb9732268170d04bdbeb4b0c368bfa28c0`, tree
`85d83b28ac96d6dccf1622d5b55af887ef7a6cba`. CPU numerical and complete release
checks passed before CUDA admission. Actual work equals the frozen proposal:
two loads/six segments/44 forwards/74 steps/32 outputs, zero warmups/retries.
Both Jobs exited naturally; registries are zero and all six model/controller/
watcher owners are absent. The allowance is closed; no inference follows.

| Measurement | CPU | CUDA |
| --- | ---: | ---: |
| Load / unload | 27.525s / 734.472ms | 29.419s / 90.688ms |
| Cold eight-token prefill | 17.460s | 689.292ms |
| Mean of fifteen decode forwards | 2.187s | 71.655ms |
| Numerical / replay segments | 15.267s / 15.147s | 550.704ms / 496.173ms |
| Total including new segments | 108.963s | 32.353s |
| Peak Job bytes, limit 34,359,738,368 | 17,685,397,504 | 20,569,268,224 |
| Sampled whole-GPU bytes, limit 20e9 | Not applicable | 18,935,185,408 |

All three contexts have exact oracle top-1 IDs 1072/290/29082. Maximum
top-eight-union errors are 0.131067/0.131048, full-row mean errors
0.00205427/0.00205423 and probability TV 0.00485888/0.00485843 (CPU/CUDA).
These pass the original 0.25/0.01/0.01 bounds. All six replay comparisons are
raw-logit exact. Sixteen cold output IDs match CPU/CUDA and the retained 5a run.
Captured rows were independently revalidated by the bound host executable.

Loaded-module inventories contain 6/57 entries and the declared DLL paths/digests
pass. CPU/CUDA module probes consume 57.067s/17.669s over 202/59 calls; CUDA's 33
GPU probes consume 2.010s. These overlap model execution and are not added to
forward/total times; sampling can miss peaks and observers change run conditions.
Compared with historical 5a CPU prefill/decode 103.463s/17.479s, the new values
are lower. CUDA's historical 675.021ms/67.382ms is close to the current values.
One run per source and different observation/work prevent a causal estimate.

The next CUDA attribution target is the nested final-validation boundary:
mean decode 32.105ms versus 0.114ms caller readback and 0.526ms argmax. Split
synchronization, host materialization and finite scan in a future bounded
profile, preserving arithmetic/order/cache/cancellation. Synchronization waits
for queued GPU work; this does not identify a kernel bottleneck. Any additional
model calls require a newly reviewed source/binary/helper packet and allowance.

Proof is retained under
`artifacts/gpt-oss/characterization-e3e1afcb-d5726687beae43a0b3f9f0eb3544b83f/`.
`report.json` binds compact receipts; the approved helper bytes are preserved
once under `source-bindings/`. Raw rows retain verifier byte/path contracts.

Secondary parent owner records have null controller executable paths on both
lanes and a null parent CUDA watcher path. Watcher self-observations and native
model/Job/binary identities are complete; held handles and all six owners are
released. The raw receipts stay unchanged. A subsequent model-free repair
records bounded live PID/UTC-tick/executable identity instead of the nullable
adapted parent `Process.Path`, with wrong-executable/creation rejection. It does
not retrospectively qualify the changed helper or any consumer. Raw-token short
F32 proof supplies no Harmony, BF16/32k, route deadline or activation sign-off.

## Closed entry-point attempt and repair, 2026-10-13

Local checkpoint `03228e54595cf783e6186ba86f11e55a0fd63696` passed the clean
native gate, release preparation and final-packet review. The owner approved
packet SHA-256 `6fd228404aa00ca2e1245b1d87046b26735c4838472db5452d929eece36680ff`
for the frozen CPU-first workload. Its invocation exited zero without creating
the model output root: the imported legacy helper's `AsLibrary` parameter
overwrote the caller's same-named switch, causing an early library return.
Zero loads, segments, forwards, token steps or outputs ran; CUDA never launched.
An empty successful invocation is not a qualification receipt. The attempt and
allowance are closed, with original packet/invocation bytes retained alongside
`qualification-attempt-closeout.json` in the preparation proof root below.

The correction preserves the caller's library intent in a distinct private
value before dot-sourcing. Model-free child `pwsh -File` tests now exercise
missing-argument and malformed-packet refusal, output-root absence, and explicit
library mode; direct helper tests alone cannot prove executable admission.
The correction requires its own clean checkpoint and reviewed source/helper
binding before another finite owner allowance. No retry of the closed packet,
consumer activation or publication follows from this repair.

## Oracle recovery, 2026-10-13

The historical Temp oracle was missing. On the owner's explicit regeneration
request, retained llama.cpp b11379 / commit
`1537a0a8b2f8711d840878b0a0677ab2213c882c` produced one eight-token chunk,
three rows and zero outputs, warmups or model retries. An earlier wrapper
preflight stopped before any model load; its failure remains separately closed.
The successful native Job released worker and child. Its exact 1,206,604 bytes
match the original fixture SHA-256
`dafa573496953142bb637bd009f1a0e38a9e36fbd10255ee98fc9bfbc6dbfc3f`;
codec/prefix/top-1 verification passes. No fixture or historical receipt changed.

Recovery proof and the single payload are retained under
`artifacts/gpt-oss/characterization-prep-02326623-20261013/reference-b11379-direct-d29b21d98d5145edb981bda2f246bddd/`.
The raw supervisor terminal remains incomplete because the producer omitted a
layer-offload count. `recovery.json` establishes exact artifact recovery, not
CUDA layer execution or fresh Candle parity. It binds the new producer,
observed DLLs, native Job, codec verification and post-run census. Peak Job
memory was 14,066,053,120 bytes and sampled whole-GPU use 14,471,397,376 bytes.
No model processes remained; the post-run census reported 48.99 GB available
physical memory, 28.48 GB commit against a 73.38 GB limit and 1.146 GB GPU use.
Keep these new producer records separate even though their output digest matches.

---
AI-edited: 2026-10-13 | agent=Codex/root | model=unknown | effort=unknown | task=gpt-oss-short-qualification | change=recorded measured numerical runtime release proof and its precise limits
