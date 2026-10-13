# GPT-OSS maintained-session baseline profiling

The next characterization uses a separate [common-session contract](CHARACTERIZATION.md)
and executable. It adds fixed-context numerical rows, reset/replay and observed
DLL validation without changing this frozen historical eight/16 workload.
Its local preparation checkpoint is `03228e54`; the missing pinned oracle was
recovered with exactly matching bytes. The approved `6fd22840…` final packet
exited before dispatch because a dot-sourced library switch replaced CLI intent.
It performed zero model work and is closed without retry. The scoped entry-point
repair passes 145 model-free controls including real `pwsh -File` refusal;
fresh measurements require a new clean source/helper-bound packet and allowance.

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

Completion requires all aggregate timings and positive bounded static-weight
bytes, null producer/Job errors, a resumed Job and natural exit. Timing checks
use a fixed `0.001 ms` rounding allowance per independently reported field.
The sequential sum of load, unload and each call's forward/readback/argmax
must not exceed total time by more than `0.051 ms` for the frozen 16-call
workload (50 component fields plus total). CUDA final validation is nested in
forward time: compare it with forward using `0.002 ms` allowance, and never
add it again to the sequential sum. Nonfinite accumulated sums fail. Total
need not equal the sum because registry work and progress I/O are also timed.
These tolerances address reported-number rounding, not performance acceptance.
Failed/forced-stop records may remain partial; they cannot pass completion.

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

## Upload-repair qualification at `5a7e392a`

The separately approved packet at clean
`5a7e392a7192978733e5844d7f7b25667a7e963b`, tree
`1dea5c734e94d838409143d850d00102d639ff76`, is 15,411 bytes with SHA-256
`a3a66d6e6cd223af9f239cf8a5926b22f503f5aabd0605a135b25ebb95a270a4`.
It uses the same retained model and frozen raw IDs, with CPU completion/release
before CUDA. Both lanes complete under the 32 GiB/1,800-second Job bounds.
Its retained root is
`artifacts/gpt-oss/session-profile-5a7e392a-fb2ea0b94ac44c7cb4f9d5a8b5629ffd/`;
`closed-batch.json` binds the source/tree, per-lane receipts and release.

| Measurement | CPU | CUDA |
| --- | ---: | ---: |
| Load | 25.408s | 30.778s |
| Eight-token prefill | 103.463s | 675.021ms |
| Mean of fifteen decode forwards | 17.479s | 67.382ms |
| Total including unload | 391.647s | 32.567s |
| Peak Job memory, bytes | 17,682,251,776 | 20,567,097,344 |
| Sampled whole-GPU peak, bytes | Not applicable | 18,903,728,128 |

CUDA's 35 one-second samples remain below 20,000,000,000 bytes, but may miss
peaks. Its mean final-validation interval is 28.022ms including pending device
work; mean caller readback is 0.111ms. These are distinct measurements.
All sixteen greedy IDs agree across CPU/CUDA and with the previous CPU run;
no full-logit comparison or general accuracy proof was collected. Single-run
CPU timing variability does not prove a performance regression. Successful
completion after progressive source release does not identify the cause of
the earlier allocation failure.

Live CUDA DLL paths were not captured during the short run. Installed candidate
DLL hashes are retained, but are not observed loaded-runtime provenance.
This remains a qualification limitation. Both Jobs exit zero, owned process
trees are absent and registry counts return to zero. The allowance is closed:
two loads/two requests, 32 public forwards, 46 token steps, 32 generated tokens,
zero warmups/retries. Earlier failed allowances remain closed.

## Ordered CPU packed-dot optimization

The measured whole-forward CPU cost does not identify its layer fractions.
A frozen synthetic primitive comparison isolates `PackedMxfp4::matmul_expert`
without production weights, model construction or token forwards. The candidate
computes one E8M0 multiplier and sixteen-value lookup per 32-value block,
then multiplies/accumulates low and high nibbles in the original column order.
It preserves F32 arithmetic, packed resident weights and scalar reduction order.

The retained recipe and source are in `artifacts/gpt-oss/cpu-dot-20261013/`.
`bench.rs` SHA-256 is
`5cfb549506118e319d8f88253ae04503d8347623414c8414537e8a5913f1e1e3`.
Both variants use release optimization/native CPU targeting, two experts,
expert 1 and one input row. Bytes are `(index*73+41) mod 256`, scales
`116+index%9`, and inputs `(index%31-15)*0.01F32`. Sampling was frozen before
measurement: one process per variant, small/medium/wide order, two warmups and
five measured calls per case, in separate 8 GiB/90-second native Jobs.
Warmups here are model-free primitive calls; no production allowance is used.

| Output rows × input width | Baseline mean | Candidate mean | Ratio |
| --- | ---: | ---: | ---: |
| 64 × 256 | 0.08686ms | 0.00996ms | 8.72x |
| 1,024 × 2,880 | 15.69418ms | 1.79990ms | 8.72x |
| 5,760 × 2,880 | 89.27012ms | 10.09476ms | 8.84x |

Every sample is finite and repeats its F32-bit checksum; baseline and candidate
checksums agree for all three cases. Baseline uses clean `5a7e392a` CPU library
`84e12fece74851c6`; the candidate uses the dirty source-bound CPU library
`8abd1c39d6c73ef6`. Recipe/binding records retain their full hashes, native flags,
default-only features, source file hashes and executable identities. The
candidate is not a clean-checkpoint or production latency qualification.
This one controlled result has no confidence estimate or general speed threshold.

The first attempted candidate link incorrectly reused the baseline library.
Its artifacts are retained and explicitly excluded in
`invalid-first-candidate-binding.json`. Only `candidate-bound-*` records bind
the measured implementation; the frozen harness and sampling did not change.
Both qualified synthetic Jobs exit zero and release their process owners.
CPU 53/CUDA 61 fixtures (one production ignore each) include independent model
oracles, cache/cancellation recovery, exact scalar-bit comparisons across widths,
routes/batches/scale extremes and controlled malformed-span errors.

The complete native `.tools/verify-before-push.ps1` gate passes, including
formatting, locked/offline maintained crate/example checks and tests, strict
CPU Clippy, layout, typed receipt, Summary Bank, overlay and whitespace checks.
Separate focused CUDA fixtures/check/strict Clippy and the GPT-OSS manifest
pass. Commands and exit codes remain in the same proof root. WSL was not
repeated because its previously observed offline dependency is still missing.

The owner approved the nine-file local checkpoint at `270bfc86`, tree
`e1f153647c928e5be82ff87d3d2392a549181895`, without push or inference. Its
clean native gate and locked/offline CPU/CUDA release builds pass. The prepared
18,210-byte packet has SHA-256
`9591dc7c4af55ad17df68ef1e04dbd1959173ad4c10c0bf8d1ecfde22edcf653`.
It freezes the same two-load/two-request/32-forward/46-step/32-output workload
and a separately armed read-only DLL watcher with 18 harmless-process controls.
The watcher never launches/stops model processes or their native Jobs. Its
observations must include the required modules per lane and hashes of observed
CUDA/MSVC files; installed candidate identities alone cannot qualify runtime
provenance. Greedy equality and descriptive single-run timing remain its
comparison scope. No model run consumed this proposed packet.

## Receipt review and scoped repair at `270bfc86`

Read-only review/recheck recovered the prior nine-file identities and original
component checksums. Packed-dot arithmetic, bounds and cache/cancellation
ownership remain unchanged. In-memory controls reproduced a host receipt
false-pass: PowerShell converted boolean load counts and fractional forward
counts to integers, treated the string `"false"` as a successful release flag,
and accepted nonnumeric generated IDs and coerced timings.

The scoped launcher repair requires nonnegative integer JSON values for input,
work/cache/PID/byte fields, bounded UInt32 token IDs and actual boolean Job flags.
Timing values must be finite nonnegative numbers; strings and booleans fail.
Existing equality, CPU-first, memory, identity and process-release gates remain
in force. Added model-free controls exercise malformed producer records and
valid JSON round trips. This changes proof admission, not model arithmetic or
the frozen production allowance. The retained `5a7e392a` receipts contain valid
typed fields; revalidating those existing records adds no model work or new
source qualification.

Fresh cleanup verification passes: `pwsh -NoProfile -File
scripts/gpt-oss/test-session-profile.ps1` has 139 model-free controls, including
independent consumed-file/expected-identity objects and valid JSON round trips.
The repaired completion validator accepts the original CPU/CUDA `5a7e392a`
progress and Job receipts. `cargo fmt --all -- --check`, the complete native
`.tools/verify-before-push.ps1` gate, GPT-OSS manifest, existing Summary Bank
routes and whitespace checks pass. Exact results and stale-packet disposition
are retained under
`artifacts/gpt-oss/cpu-dot-20261013/receipt-review-270bfc86/`. There are zero
new production loads, forwards or outputs; that review added no commit or push.

The earlier `9591dc7c…edcf653` packet is preserved unchanged but source/helper-
stale after this repair. Do not execute it, including if its pending approval
arrives later. The owner selected Closeout/Publish for the verified six-file
repair and related GPT-OSS checkpoint history. The guarded helper's canonical
receipt binds the delivered source/tree, final native gate and remote equality.
Compact closeout/publication proof and the preserved `f1278f94` receipt remain
in the existing `receipt-review-270bfc86/publication/` root. Model receipts keep
their executed identities; publication does not promote them to a new source.

Next: bind/review a new clean-source packet and exact binaries/helper/watcher
before fresh finite production approval. Repair-run receipts
retain `5a7e392a`; no prior allowance may be reused.
Later BF16, staged 8k/16k/32k and Edge/Symbio route qualification retain their
own gates. Responses remains 90 seconds and WorkerRuntime remains 600 seconds.

## Published receipt recheck at `ced33ca5`

The six published file identities and canonical publication receipt match
the retained closeout. Prior typed-field repairs remain closed. Read-only
in-memory controls found that aggregate load/unload/total timings and static
weight bytes were still unchecked. Completed records also accepted non-null
producer/Job errors, an unresumed Job or a timeout termination reason. The
outer controller exit check catches ordinary launch failures; these defects
affected persisted-record admission, not an observed failed-model bypass.

Cleanup requires finite nonnegative aggregate times, positive integer static
weight bytes within the CPU 32 GiB or CUDA 20e9-byte budget, null errors and a
boolean resumed Job with `termination_reason="exited"`. Existing work, identity,
per-call timing, release and CPU-first gates remain in force. The native script
now passes 178 model-free controls; original `5a7e392a` CPU/CUDA receipts pass
revalidation with unchanged bytes and no new model work. Proof is retained at
`artifacts/gpt-oss/receipt-review-ced33ca5-20261013/`. No model arithmetic,
sampling, allowance, consumer pin or qualification bound changes.

Current cleanup also passes `cargo fmt --all -- --check` through the complete
native `.tools/verify-before-push.ps1` gate, including maintained checks/tests,
strict Clippy, Summary Bank/layout/overlay gates and `git diff --check HEAD`.
The GPT-OSS and LFM manifest gates pass; inventories remain GPT44/overlay299,
with 27 shared paths. Final handoff-only updates pass routing/whitespace checks.
CUDA production and WSL lanes were not rerun for this host-script repair.

That review left checkpoint/publication for separate authorization before
fresh clean-source packet binding. The unchanged `9591dc7c…edcf653` proposal
remains forbidden to execute; all earlier model allowances remain closed.

## Supplied-feedback Work, 2026-10-13

The aggregate-field/error repair above was already present in the local
candidate when the drop-in arrived. Accepted remaining work adds the explicit
timing inequalities and rounding bounds, rejects finite-field sum overflow,
and exercises missing/malformed aggregate fields and JSON round trips on
both CPU and CUDA shapes. All 241 native model-free controls pass, including
partial failure preservation and actual harmless Job-tree cleanup. Original
`5a7e392a` records pass without inference: CPU/CUDA aggregate slack is
6.9700/6.0479 ms; receipt bytes and executed identities remain unchanged.

The current complete native locked/offline gate, both manifests and existing
Summary Bank/layout/overlay checks pass. Exact commands, runtime, source-file
identities and retained-record revalidation are kept in
`artifacts/gpt-oss/delivery-work-ced33ca5-20261013/`. Work left seven scoped
files uncommitted. There are zero new production loads/forwards/output tokens;
WSL and new production CUDA qualification were not run.

The delivery order, owner results and reviewed tuple are in
[LOADING.md](LOADING.md#coordinated-delivery-boundary). Whole-model speed/full-
logit/DLL/context/consumer proof is still pending; none is supplied by these
controls. New source delivery, packet approval and native execution remain
separate events. No stale proposal or closed allowance can be reused.

## Guarded progress closeout

The owner selected Closeout/Publish for these seven files. The guarded helper's
canonical receipt binds the delivered commit/tree, final native gate and exact
remote equality. Compact proof and the preserved `ced33ca5` publication receipt
are in `artifacts/gpt-oss/delivery-work-ced33ca5-20261013/publication/`.
The 241-control result, retained-record compatibility and owner coordination
are scoped progress; production numerical/speed/runtime/context/provider
criteria remain open. Build source-bound inputs and review a new packet before
requesting its finite allowance. No consumer edits, model runs or release tags
belong to this publication.

---
AI-edited: 2026-10-13 | agent=Codex/root | model=unknown | effort=unknown | task=gpt-oss-cli-admission | change=closed zero-work invocation without relabeling historical measurements
