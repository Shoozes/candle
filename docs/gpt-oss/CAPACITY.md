# GPT-OSS fixed-token capacity qualification

The first selected stage is an 8,192-token synthetic stress through the public
`GptOssSession` CPU/CUDA APIs. It is implemented by `gpt-oss-capacity-stress`
and the capacity preparation/launch/watcher helpers. The separately approved
CUDA stress completes at `e3ef7034`; its allowance is closed. The short proof at `3db05cd8` remains
its CPU-first numerical/runtime predecessor at its original source identity.

## Frozen work and scope

One 32 KiB little-endian u32 file repeats the pinned eight reference-prefix IDs
1,024 times. The input binds its exact size/digest and the retained GGUF identity;
token admission finishes before any model opens. Unknown input fields, malformed
counts, different prefix bytes, truncated/oversized files and invalid identities
fail. `--check-input` is model-free and accepts absent model paths.

| Work | Count |
| --- | ---: |
| Loads / request segments | 1 / 1 |
| Prefill | 6,144 tokens in 48 calls of 128 |
| Fixed decode | 2,048 calls/tokens |
| Total public forwards / processed steps | 2,096 / 8,192 |
| Sampled outputs / warmups / retries | 0 / 0 / 0 |

Fixed decode inputs exercise the generated-token capacity reserve, but do not
sample generated text. No Harmony, coding quality or independent long-context
numerical oracle follows. This is not Responses 90s or WorkerRuntime 600s proof.
16k/32k stages require later measured decisions and separate finite allowances.
Small-context timings cannot establish the 8k deadline; in particular their
sequential CPU extrapolation exceeds the native 1,800-second bound.

## Observation and recovery

Each prefill/decode uses the existing profiled session forward once, drops its
logits and never samples. CUDA's finite scan and commit/cancellation boundary
remain in the model. The constant-size progress record distinguishes attempted
and completed calls/steps, committed cache position, logical/allocated cache
bytes, and prefill/decode timing. Cache maxima are observed at public-call
boundaries, not a claim of every transient allocation peak. CUDA validation
time is nested in forward time. Failures retain partial progress; session drop
and the load registry establish release. `--verify-progress` rechecks completed
counts, input/consumed identities, storage and timing without model loading.

CPU model-free controls precede CUDA synthetic controls. Tiny deterministic
CPU/CUDA tests compare 128-token chunks with single-token execution across
64-row pages and the 128-token sliding window, subsequent decode and cancelled
snapshot recovery. CUDA chunk/single results are bit-exact; CPU/CUDA component
comparison retains absolute `1e-4`. These fixtures are not retained-model stress.
The original preparation fixtures pass six CPU and seven CUDA tests; 24 PowerShell
controls pass. The subsequent review adds explicit relative-path refusal before
token/model I/O. Local prefill projection work and synthetic profiling live in
[PROFILING.md](PROFILING.md#model-free-attribution-at-0994bfe4-2026-10-14);
they do not requalify this source-bound result.
Both release drafts and model-free input startup Jobs pass and
release their owners. The draft packet remains non-executable; clean-source
binding is checkpointed at `3b7868b3`/`e3ef7034`. The final source-bound packet
and its distinct execution allowance are complete; draft packets remain blocked.

## Observed 8k result, 2026-10-14

Clean source `e3ef70341026ceff030402bc2466b26c8457373d`, tree
`fa7953ea91815ec28e0a3cc41a18441d57b1fe24`, executes final packet
`e18779a4da7c3beecfaa791d5440c1464304548207824ad91d8071468f4c1641`
(8,119 bytes). Direct human approval was verified in Symbio's source chat.
Actual work exactly matches the frozen table: one load/request, 2,096 forwards,
8,192 processed/committed tokens, no sampling, warmup or retry.

| Measurement | Observed |
| --- | ---: |
| Load / unload | 34.106s / 117.768ms |
| Prefill / fixed decode | 462.014s / 175.204s |
| Mean fixed decode forward | 85.549ms |
| Total model time | 672.257s |
| Static payload | 15,053,650,176 B |
| Final / peak cache capacity | 408,944,640 / 411,992,064 B |
| Peak native Job, 32 GiB ceiling | 21,662,621,696 B |
| Peak sampled whole GPU, 20e9 ceiling | 17,283,678,208 B |

The native entry and model Job exit zero naturally. Registry count is zero;
model, controller, watcher and outer invocation identities are complete and
all four PIDs are absent. CPU host verification adds zero model forwards.
Actual required DLL identities pass across 57 observed modules. There are 632
GPU samples; their cadence can miss peaks. Original receipts remain unchanged.

Proof is retained under
`artifacts/gpt-oss/capacity8k-e3ef7034-25d56b01550f40eb8fe29ce87d73f995/`.
Its 7,933-byte `report.json` has SHA-256
`dd9c57f890e9e1c382db27e90ccfe97e60a05e18abd78648db83bda16298ae3a`,
binding eleven original execution/measurement/owner receipts. A quoting error
in the outer metadata command occurred before all helper/claim execution;
its separate diagnostic records zero model work. One native attempt followed.

Forward work alone is 637.218s, exceeding Responses 90s and WorkerRuntime 600s
for this input even with load excluded. The run proves this physical fixed-token
8k stress, without generated semantics, an independent long-context oracle,
16k/32k or consumer readiness. Timing is one descriptive observation, not a
controlled speed comparison. Next prepare model-free bottleneck attribution
and consumer workload admission; any further model run needs a fresh packet
and finite allowance. Q8, Codex and the disabled provider remain fixed.

The new coordinator uses the existing native bounded lane/probe helpers with
capacity-specific input/completion validators. A separate passive watcher
requires the capacity schema and observes actual loaded DLL paths. Both watcher
and controller record PID, creation time, executable, lifetime and cleanup route;
exact held handles own shutdown. The original short runner/helpers stay intact.

## Binding and execution gates

`prepare-capacity.ps1` builds CPU then CUDA release binaries in separate
16 GiB/900-second native Jobs, then checks both inputs in 2 GiB/15-second Jobs.
Preparation reads only GGUF path/size metadata; the approved model Job must
rehash its retained file. One immutable token stream is shared by both lanes.
The CPU binary performs model-free receipt verification; only one CUDA model
lane belongs to the proposed first stress. Required DLL file identities are
bound in preparation, while actual loaded-module observations require execution.

The final packet requires exact clean commit/tree, release binaries/build Jobs,
source closure, compiler/features/flags, helper/probe/watcher bytes, token/model
identities, and the original completed short predecessor/receipt identities.
Draft packets cannot execute. Preparation must finish completely and release
all owned Jobs. Execution needs separate owner review of its exact finite packet.

Each proposed model Job remains 32 GiB/1,800 seconds; sampled whole-GPU ceiling
is 20,000,000,000 bytes. GPU/module observers are independently bounded and make
zero model forwards. One-second samples can miss peaks. Any model, observation,
deadline, comparison or release failure stops the stage; no retry or borrowed
short allowance is allowed. Original incomplete receipts remain unchanged.
Q8 defaults, GenEye `3fa7075e`, Codex `0.154.0` and disabled GPT-OSS remain fixed.

---
AI-edited: 2026-10-14 | agent=Codex/root | model=unknown | effort=unknown | task=gpt-oss-prefill-review | change=rechecked published fixes and scoped model-free prefill attribution
