# GPT-OSS local loading and session contract

The current implementation is a local F32 loading/session candidate. Synthetic
CPU/CUDA proof does not qualify production weights, performance, a 32k provider
or BF16 execution. Existing production receipts retain their original source
identities and 2,080-total-token boundary.

## Public API

`candle_transformers::models::gpt_oss` exports `GptOssSession`,
`GptOssLoadOptions`, `GptOssSource`, `GptOssDevice` and `GptOssLogits`.
Callers supply the exact local file identities, device, sequence/cache/device
limits, weight budget, shared load registry and cancellation token. There is
no cache discovery, download, tokenizer rendering, sampling or provider policy
in this session.

- `GptOssSource::Gguf` takes a `GptOssFileIdentity`: local path, byte size and
  expected SHA-256. Historical GGUF helpers still require the original selected
  artifact; `open_with_sha256` is the explicit alternative.
- `PackedSafetensors` takes `GptOssSafetensorsSource`: explicit Original or
  Transformers layout, config/tokenizer identities, optional index identity
  and exact shard identities. Multiple shards require their index. The index
  controls the shard set; filename numbering is not used to infer it.
- `GptOssSession::load` returns a CPU or feature-gated CUDA session. `config`,
  `device`, `activation_dtype`, `consumed_files` and `resource_usage` expose
  execution and provenance. Unsupported CUDA selections fail before artifact
  I/O in CPU builds.
  `static_weight_bytes` reports assembled payload residency. Both backends
  admit combined static/cache/conservative-workspace bytes before forwarding;
  CPU rejects a budget that cannot admit one token before weight assembly.
  CUDA bounds source assembly separately, selects lossless table storage and
  checks exact one-token device admission before opening the device.
- `prefill` and `decode` return device-native `GptOssLogits`. `into_cpu` is a
  separate fallible readback; readback failure is not a rolled-back forward.
  Inference cancellation/errors preserve the previously committed cache prefix.
  `reset_cache` releases owned cache buffers.

The CUDA executor still synchronizes and reads its full final logit row on the
host for finite-output validation before cache commit. Failure in that internal
validation preserves the old cache. A later caller-requested `into_cpu` failure
leaves the completed forward committed. Returning a CUDA tensor does not yet
eliminate the executor's validation transfer; profile that transfer before
changing the safety boundary or claiming a latency improvement.
Optional `prefill_profiled`/`decode_profiled` also return `GptOssForwardTimings`.
They separate forward wall time from the CUDA final validation block; ordinary
calls do not read a profiling clock. The validation interval includes pending
GPU work. Caller readback remains separate. See [PROFILING.md](PROFILING.md).

Both loaders preserve MXFP4 expert blocks/scales. Original safetensors store
adjacent low/high nibbles and alternating gate/up rows; only GGUF wire bytes
need their existing layout normalization. Non-expert F32/F16/BF16 weights, and
GGUF Q8_0 weights, normalize to F32 CPU values. CUDA embedding/head tables may
retain lossless BF16 storage with F32 activations and arithmetic; other dense
parameters retain F32. See [the memory candidate](MEMORY.md) for exact selection,
reduction and admission boundaries. There is no native-Q8 non-expert storage or
FastBf16 readiness claim.

## Admission and ownership

SHA-256 and size admission use the retained file handle. Windows read-sharing
denies writes/deletion while that handle is retained. An artifact owns its file
handles; assembled model weights hold the registry lease through CPU/CUDA model
lifetime. Failed/cancelled loading releases ownership for a subsequent load.

Safetensors bounds include 128 explicit shards, 16 MiB per header, 32 MiB total
headers, 4,096 total tensors and rank 1..4 positive shapes. Duplicate names,
unsupported dtypes, gaps/overlaps, truncated/trailing payloads, index mismatch,
unconsumed tensors and arithmetic overflow produce errors. Dense conversion
uses at most 1 MiB raw chunks rather than a second whole-tensor copy.

The weight budget bounds final assembled residency; metadata, conversion,
QKV concatenation and backend upload also require temporary host memory.
Physical host/GPU ceilings remain enforced by the owner launch and measured
qualification, not inferred from logical cache accounting.

CUDA construction consumes the assembled CPU weights. Embedding/norm vectors
move through owned tensor storage; source layers remain alive through their
uploads and synchronization, then drop individually. The source head drops
after its upload. The model retains the registry lease. This reduces source
retention overlap without changing F32 values, packed expert bytes or public
construction signatures. The first maintained CUDA load at `3b16430b` failed
near the host Job ceiling. Fresh CPU/CUDA qualification at `5a7e392a` completes
under the declared bounds; it does not establish the cause of the earlier
allocation error. Counts, sampled memory and the incomplete live CUDA DLL
provenance are recorded in [PROFILING.md](PROFILING.md).

CPU expert dot products decode each 32-value block's scale/lookup once. They
retain the packed blocks/scales, F32 activations and increasing-column scalar
accumulation order. There is no dense expert repack, new reduction or additional
retained weight storage. Exact-bit synthetic tests include exponent extremes,
multiple widths, routes and batches. The frozen component benchmark measures
this primitive; it does not qualify end-to-end model latency.

Configuration accepts original and Transformers aliases, reconciles equal
duplicate aliases, and rejects conflicting expert/YaRN fields. Explicit layer
types and RMS epsilon drive both executors. Missing layer types preserve the
legacy alternating sliding/full schedule. Checkpoint capacity is distinct from
the smaller supplied execution limits and any qualified provider envelope.
Existing JSON configuration remains compatible. Rust callers constructing
`GptOssConfig` with a struct literal must now supply `layer_types`,
`rms_norm_eps` and `max_position_embeddings`; repository callers are updated.

## Remaining maintained-support gates

The accepted target is GPT-OSS-20B on native Windows Ryzen 7950X and RTX 4090,
with 32 GiB host and 20,000,000,000 GPU bytes, and 32,768 total tokens split into
30,720 prompt/history plus 2,048 generated tokens including private reasoning.
120B configuration compatibility is distinct from production qualification.

Next: clean source/binary-bound CPU optimization qualification, then measured
CUDA prefill/decode optimizations, separately qualified optional BF16, and staged
8k/16k/32k qualification. No old model allowance may be reused. Full native
production qualification also requires retained official packed artifacts;
metadata inspection does not establish their local presence or permission to
download them.

EdgeSymbio owns Harmony, sampling, the existing in-process model slot and the
linked artifact/device/mode/route qualification. The maintained alias is
`edgesymbio.local.gpt-oss-20b.v1`; its runtime and Responses identities are
`edgesymbio.candle-gpt-oss.runtime.v1` and
`edgesymbio.codex-responses.candle-gpt-oss.v1`. Those are prepared identities,
not enabled capabilities. Preserve the experimental identity and Q8 default.

Responses retains its 90-second total deadline. Longer work uses WorkerRuntime
within its existing 600-second bound. A WorkerRuntime 32k receipt cannot qualify
Responses. Strict v1 receipts stay unchanged: their linked capacity-report
digest must bind the actual source/build/artifact/device/mode/format/settings,
route/deadline, observed envelope, physical memory, work counts and release.
Packed weights use a separately identified canonical consumed-file manifest;
its digest must never be presented as a single model-file hash.

Symbio preserves Codex's engine/tool loop and requires observed Edge
qualification. Pins/locks/provenance advance only after separately authorized
Candle publication, followed by Edge qualification and Symbio consumption.
GenEye remains pinned to `3fa7075e`.

## Coordinated delivery boundary

The selected 2026-10-13 batch is the supplied feedback's delivery reconciliation
and model-free completion gate (Tasks 1/2). It preserves one dependency order:
verified Candle delivery/publication, then Edge graph adoption and native
route qualification, then Symbio consumption, then a separately selected
release candidate. Publishing source is not runtime qualification or activation.

Reviewed heads, independently rechecked locally:

| Owner | Source | Current boundary |
| --- | --- | --- |
| Candle | `0232662335b393da3c8f5b64d08302c5e3a92a47` | Published receipt repair; common-session characterization implementation is local, with final clean packet and inference allowance held. |
| EdgeSymbio | `5d34c7bb773a9929bf46b03679cdeb92a9fea12f` | Twelve-file local backing-report/held-launch/recovery preparation verified; four Candle pins remain `0c0bc3ec`, tokenizers locked 0.23.2/onig and maintained profile disabled. |
| symbio-code | `e43eeeaccee1bae782eddb4faf677c0d9226ea59` | Twelve-file failure/replay preparation published and byte-verified; Codex 0.154.0 and pins retained. Native coding held. |

The old `4c1feb82`/tokenizers pin-hop task is superseded: `4c1feb82` is an
ancestor of Edge's actual `0c0bc3ec` pin, and the tokenizer upgrade is present
(0.23.1 declaration; all three canonical locks resolve 0.23.2/onig).
The next adoption is one reviewed four-dependency graph at the delivered
Candle revision, with canonical locks/provenance and regressions. Historical
VL selection/receipts are preserved; no intermediate obsolete hop is queued.

| Order and owner | Observable result | Predecessor |
| --- | --- | --- |
| Candle, completion proof | Complete typed metrics, coherent timing and success/release records; model-free CPU/CUDA controls and native gate. | Delivered at `02326623`; no reopened repair. |
| Candle, production characterization | Clean source/build/lock/features/compiler/helper/watcher binding; fresh finite CPU-first whole-model timing, applicable full-logit comparison, loaded DLL paths/hashes and release proof. | Completion repair delivered and a newly approved packet. |
| Candle, one CUDA optimization at a time | Measured bottleneck improvement and numerical/recovery proof, then Short/8k/16k/32k capacity stages; stop at the first failed stage. | Trustworthy current-source baseline. |
| Edge, native qualification | One resident owner/broker lease, safe post-commit readback recovery, exact tokenizer/Harmony admission, trusted backing report and profile-specific launch proof. | Model-free backing reader and held launch path implemented; native adoption/execution follows qualified delivery and separate authority. |
| Symbio, coding consumption | Exact Responses evidence/policy, independently applicable verified patch, failure/replay controls and contained cleanup using the retained engine. | Edge's matching maintained Responses row; model-free preparation may proceed now. |
| Each release owner | Reproducible immutable tuple, installed lifecycle/rollback and explicit support exclusions. | Selected native and consumer rows proven; separate release authority. |

Support rows stay distinct:

| Artifact/device/mode/route | Current proof | Gate remaining |
| --- | --- | --- |
| Retained GGUF, CPU/CUDA, F32, raw session | Implemented and component-tested; retained short observations at `5a7e392a` only. | Current optimized-source full-model comparison and actual CUDA DLL provenance. |
| Original/indexed packed safetensors, CPU/CUDA F32 | Implemented with deterministic fixtures. | Retained official-artifact production qualification. |
| Edge Responses, maintained GPT-OSS F32 | Disabled model-free policy; 90-second total route. | Observed route-specific limits and trusted backing report. Symbio requires 32,768 context/2,048 output. |
| Edge WorkerRuntime, maintained GPT-OSS F32 | Disabled model-free policy; 600-second route. | Independently measured envelope; never substitutes for Responses proof. |
| BF16 and release/LTS rows | Not qualified. | Independent parity/capacity or immutable release/support proof. |

Edge's general launcher remains 8 GiB. Its proposed GPT-only envelope is
32 GiB host/20e9 GPU bytes and must be reviewed before native admission;
Candle's 1,800-second diagnostic Job does not prove either consumer deadline.
Q8 defaults and their frozen successful/failed coding controls remain separate.
Post-commit caller readback failure requires slot reset, quarantine or explicit
reconciliation before reuse; repeating the same input would advance it again.
Consumers keep readback, sampling and recovery inside the resident lease;
post-release finalization may handle only completed host results.
Qualification digests need Edge-validated actual backing reports; digest-only
Symbio controls supply no native capacity. No new wire schema or signing service
is selected. Optional accelerators, engine upgrades, new downloads, voice/vision,
adapter training and consumer activation are outside this batch.

The current Edge return is 13,889 bytes, SHA-256
`8343d2e0fad3f088721755c3cd796fc07dfacd33f5ecdafe3eaa60490ff5fb44`, at
`runtime-data/proofs/gpt-oss-task3-return-20261013-01/return.json` in Edge's
checkout. All twelve named file identities and its proof journal are verified.
Its owner-run result is 1,085 backend, two API and 58 Python tests, with three
backend ignores/five Python skips; Candle did not rerun those tests. Symbio's
published candidate SHA is `b5e2df7e…d817`, with all twelve files verified and
publication closeout `0129feed…c41` matched. The current reconciliation is
`artifacts/gpt-oss/characterization-prep-02326623-20261013/consumer-reconciliation.json`.
These are model-free preparation; native cleanup, capacity, coding and activation
remain held. The 32 GiB GPT-only launch path refuses execution pending approval;
the general launcher remains 8 GiB. No consumer is repinned by this Candle work.

Prior model-free owner returns were verified against their then-current file
identities: Edge's six-path preparation adds recovery quarantine (13 lane
tests; 1,078 backend tests, three ignores; strict Clippy and context checks).
Symbio's twelve-path preparation adds failure/replay controls and the exact
maintained fixture policy (256 Rust tests, three native ignores; both
PowerShell lanes, Clippy and foundation/context gates). All three boards agree
on the order above. Compact receipt identities and unchanged holds are bound
in `artifacts/gpt-oss/delivery-work-ced33ca5-20261013/closeout.json`; owner proof
stays in its originating repository. These local preparations do not supply
native readiness, coding quality, E2 reconciliation or shared-provider drain.
The owner selected Candle publication of the seven-file successor; its exact
commit/tree and clean-source gate belong to `artifacts/publication/last-push.json`.
Consumer source/pin adoption and native qualification remain separate. No old
model receipt or allowance is rebound to the delivered revision.

---
AI-edited: 2026-10-13 | agent=Codex/root | model=unknown | effort=unknown | task=gpt-oss-characterization | change=reconciled current Edge and Symbio preparation identities
