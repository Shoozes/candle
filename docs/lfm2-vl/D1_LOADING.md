# Optional d1 loading qualification

## Explicit text-only loading, 2026-10-12

The maintained-model batch at `4c2dfaf3` adds `D1TextLoadOptions` and
`load_d1_text_q8[_with_embedding_mode]`. DenseF32 remains the default;
Q8Rows is explicit. The private session backend consumes only the language
GGUF and tokenizer, constructs no processor/vision/projector, reports
`vision_device="none"` and zero projector inventory, and rejects images and
literal/resolved image sentinels before forwards. Existing hybrid entry points
retain their defaults. The CLI selects this path with `--text-only`, which
conflicts with projector/processor flags and preflights every image-bearing
case before image I/O or model loading. Template bytes remain provenance.

Authored CPU fixtures pass 24 tests/two ignores; CUDA passes 31/two ignores
after CPU. Both embedding modes have bit-exact text/hybrid prompts, token IDs,
logits, answers, ordering and language counts. Tests exercise missing vision
files, malformed inputs, context admission, partial failure, cancellation,
cache reset/replay and nonatomic image markers. These component results do
not substitute for the separately approved retained `t01` numerical packet.

The accepted twelve-load protocol was frozen before measurement. All samples
use the same release binary, Q8Rows, retained hash-checked files, 4,096-token
ceiling, fresh processes, 16 GiB Jobs and 900-second deadlines; CPU precedes
CUDA. There were zero forwards/tokens, warmups or retries; all owners exited.

| Device/path | Constructor ms: first; subsequent; subsequent | Median startup s | Median peak Job bytes |
| --- | --- | ---: | ---: |
| CPU hybrid | 2693.060; 2712.643; 2616.311 | 7.612 | 4,099,100,672 |
| CPU text-only | 2493.764; 2482.665; 2422.149 | 6.498 | 3,252,051,968 |
| CUDA hybrid | 2675.587; 2729.242; 2732.695 | 7.584 | 4,757,794,816 |
| CUDA text-only | 2330.362; 2446.760; 2322.120 | 6.383 | 3,900,882,944 |

Median total startup fell 14.6%/15.8% (CPU/CUDA). CUDA free-memory observations
before/loaded/after-release show 3,758,096,384 hybrid versus 2,885,681,152
text-only retained device bytes, a reduction of 872,415,232 bytes; every
after-release observation equals before-load. These are retained allocation
observations, not peak VRAM. Job/private/working-set measurements are host
memory, recorded separately. OS caches were not flushed and ordinary desktop
and an independently owned graphics workload were present; no cold-disk or
isolated-host claim follows. Compact raw samples, source/binary/input hashes
and release receipts live under `artifacts/maintained-models/20261012/`.
The later literal-sentinel admission regression changes request validation
only; these measurements remain bound to their original constructor source.

## Historical Q8Rows qualification

The selected Work batch started at published `ebf2f85e96d264c3d78f7f35b22da3cc0a9abcaf`
and preserved the separately proven example-output repairs. The owner separately
published that progress at `7c9067bc494e1cc98bdd2a83151b056757897c1c` before
approving fresh production parity. One optional loading mode is implemented and
qualified; existing constructors and default execution remain unchanged. No
lower-bit conversion, new weights, training, consumer edits or publication belong
to the completed inference batch. The owner separately authorized closeout and
guarded publication of these parity records.

## Candidate comparison and frozen measurements

The retained d1 GGUF has a tied Q8_0 token/output table with shape
`[128000, 2048]`. Its bounded 8,240,736-byte header has SHA-256
`6cd942c246c9adefd2a594eb328fc5a407859fc3271b8d62336735aebf5b2de2`.
The table occupies 278,528,000 bytes (265.625 MiB); the DenseF32 lookup copy
adds 1,048,576,000 bytes (1,000 MiB). Whole-file identities remain those of
`tools/lfm2_vl/d1/reference-lock.json` and are rechecked for each process.

- Early quant-inventory admission would reject unsupported files sooner, but
  does not remove allocations when these retained supported artifacts load.
- Lazy vision loading could avoid the retained 556 MiB projector file on text
  requests; it would also need a new session lifecycle and image admission
  contract. Its precise resident savings require a separate implementation.
- Retained Q8 embedding rows can reuse the existing tied output allocation and
  avoid the 1,000 MiB F32 copy. Existing CPU/CUDA row lookup produces F32 results.

The following protocol was frozen before measurement. Before changing the model
loader, use the example `--load-only` harness for three sequential fresh-process
loads on CPU and three on CUDA. After implementing
the selected mode, repeat the same six loads. Each process uses the retained five
inputs, a 4,096-token limit, a 16 GiB native Job and a 900-second deadline. There
are zero language/vision forwards, zero output tokens and no retries. Record
synchronized constructor time, peak Job/private/working-set memory, process and
binary identities, retained input rechecks and verified process release.

Label the first observed load separately from the two subsequent fresh-process
loads. OS caches are not flushed; these measurements do not establish cold-disk
latency. Report individual timings and medians without a speed threshold. GPU
host Job memory is not VRAM. Retained historical studies remain bound to their
original source identities.

Freeze numerical acceptance before candidate measurement: CPU Q8-row lookup and
authored model logits must match full-table F32 lookup bit for bit; typed answers,
question order, cache resets and actual forward/dispatch accounting must agree.
Then test CUDA rows against dense CUDA lookup and CPU under the existing last-logit
maximum absolute bound 0.02 and readout bound 1e-6. Check repeated/strided/offset,
empty, negative/out-of-vocabulary and unsupported-dtype inputs before device row
dispatch. Production forwards require a fresh, separately approved bounded
allowance; no consumed study allowance can be reused.

## Selected construction and error contract

Retained Q8 embedding rows were selected first. The baseline's first CPU
constructor took 2,920.682 ms with 5,149,970,432 bytes peak Job memory. The tied-table header
establishes a removable 1,000 MiB allocation, larger than the retained vision
file. Early full-model quant inventory and lazy vision remain separate tasks.

`quantized_lfm2::Lfm2EmbeddingLoadMode::{DenseF32, Q8Rows}` selects storage for
`ModelWeights::from_gguf_with_embedding_mode(content, reader, device, mode)`.
`from_gguf` and the hash-bound LoRA constructor default to `DenseF32`.
`load_lfm2_vl_hybrid_with_embedding_mode(options, mode)` and
`load_d1_q8_with_embedding_mode(options, mode)` expose the option without adding
required fields to existing options structs. Existing entry points retain their
defaults.

The explicit `Q8Rows` path admits only a positive, block-aligned Q8_0 embedding
matrix on CPU/CUDA, with checked sizes and metadata width/EOS bounds, before
reading any text tensor payload. It retains an immutable `Arc<QTensor>` and
shares it with the output head when the source is tied and output execution
stays quantized. Only requested rows are dequantized to F32 through Candle's
existing row primitive. All major Q8 linears, the projector's original native
mix, cache semantics, prompt/tokenizer behavior and consumed-file inventory are
preserved.
The whole vision model still loads; this mode changes only token-table storage.

Unsupported embedding storage/devices, overflow, unsupported ID dtypes,
negative IDs and out-of-vocabulary IDs return controlled errors. IDs are checked
before CUDA's row gather, which has no device-side bounds check. Empty ID inputs
return an empty F32 tensor of the corresponding shape without device dispatch.

`Lfm2EmbeddingStorage` is available through model/composite/session getters. It
reports mode, vocabulary/hidden width, lookup-table bytes, whether output shares
the same allocation, and additional bytes relative to the output head. This keeps
the existing decision execution/usage schema unchanged. The standalone example
adds `--text-embedding dense-f32|q8-rows`; `--load-only` emits a separate verified
receipt and performs no request forward.

## Observed local proof

All twelve retained-artifact loads completed with zero forwards and verified
native Job release. Each Q8Rows load reports 278,528,000 token-table bytes,
`shared_with_output: true` and zero additional lookup-table bytes. The mode
removes the 1,048,576,000-byte F32 copy on both devices. Host peak Job medians:

| Device/mode | Constructor samples, ms (first; subsequent; subsequent) | Median peak Job bytes |
| --- | --- | ---: |
| CPU DenseF32 | 2920.682; 2866.906; 2874.851 | 5,149,970,432 |
| CPU Q8Rows | 2682.530; 2671.690; 2638.188 | 4,097,708,032 |
| CUDA DenseF32 | 2808.036; 2680.160; 2773.935 | 5,806,395,392 |
| CUDA Q8Rows | 2733.935; 2705.233; 2708.846 | 4,757,143,552 |

CPU median constructor time is 2,874.851 to 2,671.690 ms. CUDA medians are
2,773.935 to 2,708.846 ms with overlapping sample ranges; no CUDA speedup is
qualified. These small fresh-process samples include construction only,
excluding retained-file hashing, and do not measure inference or cold disk I/O.
Full Job wall times, including admission, final rehash and release, have medians
of 7,699 to 7,618 ms on CPU and 7,684 to 7,648 ms on CUDA. Their ranges overlap;
these observations qualify memory savings, not a full-startup speed threshold.

Four focused CPU storage/ID/header/buffer tests and the CPU d1 fixture pass.
The bounded CUDA test executables pass five storage controls and two CPU/CUDA
d1 fixtures. Row results and all authored d1 logits are bit-exact between modes,
including strided/offset IDs, text/image/text cache replay, all question types,
partial failure, cancellation and context admission. Dispatch and actual work
counts agree. Both CUDA Jobs exit zero and release their processes.

Commands and exact source/binary identities are retained once under
`artifacts/lfm2-d1/20261010-loading/`. `baseline-execution-source.json` binds
the pre-model-change load-only harness; `candidate-execution-source.json` binds
the opted-in implementation. `run-loads.ps1 -Phase baseline|candidate` invokes
`scripts/lfm2-vl/run-bounded-oracle.ps1` for the frozen samples. The referenced
historical production receipts are unchanged.

The final native gate passes 528 Rust tests, 13 explicit ignores and six Python
controls, including formatting, locked/offline checks, strict Clippy, routes,
module layout and overlay union. Strict CUDA library/example Clippy passes.
The exact LFM manifest passes at 208 paths (20 modified/188 added), with 281
paths across five overlays and 27 shared paths. A new loader fixture initially
omitted its tiny processor limits; supplying that configuration fixes the test.
That post-measurement change is confined to `cfg(test)`; measured runtime code
and all frozen load receipts remain unchanged. WSL replay is unrun because
Codex-Compat still has no cached pinned `accelerate-src`. The existing nonfatal
MSVC LIBCMT link warning remains.

Compact local proof: `artifacts/lfm2-d1/20261010-loading/verification.json`.
All fourteen owned load/test Jobs are released. Source, inputs and referenced
evidence remain retained; no weights were copied. The owner authorized guarded
progress publication on 2026-10-10. That saved the locally verified option and
runner repairs before the production qualification below. The exact saved
commit/tree/remote identity is retained in
`artifacts/lfm2-d1/20261010-loading-closeout/publication-receipt.json` only after
helper success. That producer checkpoint did not perform consumer mode adoption
or pin/package proof; the later GenEye qualification is recorded below.

## Retained production parity

The owner approved the fresh bounded proposal on 2026-10-10. The source is
published `7c9067bc494e1cc98bdd2a83151b056757897c1c`; release binary SHA-256 is
`893a36f3e588d3c9108d4e43edfa198fbafb8986cfc3ad390790414fb084bdb5`.
The frozen `t01`/`i01` requests retain SHA-256
`932966c0115d584a51eef8d0e0deaaf4af01f2a0aca366a186c13b1c9d1d6432`.
All five pinned local artifacts are rehashed before and after execution.

DenseF32 and Q8Rows complete on CPU first; CUDA begins only after the CPU
comparison passes. Each device compares six 128000-element F32 logit vectors.
Both devices are bit-exact between modes, including every typed answer, prompt,
expanded ID sequence, question/option order, vision tensor and execution count.
Q8Rows still reports tied 278,528,000-byte storage and zero extra lookup bytes.
The original 167 text Q8 linears and 137 Q8/27 F16-source projector mix remain;
native major-matmul counts are 501 for `t01` and 638 for `i01` in each run.

CPU/CUDA Q8Rows maximum last-logit error is 0.000011444091796875 (bound 0.02).
Projected-feature cosine is 0.999999999989375 (minimum 0.9999). Independent
pinned official prompt/token-alias replay passes 24 questions, 12 text token
sequences and 24 typed answer checks. Maximum independent readout error is
1.1102230246251565e-16 (bound 1e-6). This replays retained Candle logits; it
does not create a new independent full-model oracle or 20-case accuracy study.

Actual allowance use is eight requests, 24 language and four vision forwards,
zero output tokens and no retries. All four 16 GiB/1,800-second native Jobs
exit zero and release their PIDs; source/binary/input rechecks and tensor staging
release pass. The allowance is consumed and closed. The load-memory/startup
measurements above retain their original source identities; this parity batch
makes no additional inference-speed claim. WSL's offline gap remains unrun.

Exact source, commands, verifier/bounds, trace and Job bindings are retained in
`artifacts/lfm2-d1/20261010-loading/production-parity.json`. The approved launcher
is `run-parity.ps1` in that directory. Read-only comparators are `compare-modes.py`,
`replay-reference.py` and `tools/lfm2_vl/d1/compare_runs.py`; they perform zero
model forwards. The selected Candle loading gate is complete. The inference
batch made no consumer edits, commits or publication. Its immutable receipt has
SHA-256 `35c1be0bb8abf5ff0a0a3e4f483a20ef4d80fa603f5e0b833cb37db74e46cf0c`.

The subsequent documentation closeout uses Candle's `.tools/gitpush.ps1 -Yes`
and its required native locked/offline gate. Exact saved commit/tree/remote
identity and the rechecked production proof are retained in
`artifacts/lfm2-d1/20261010-q8rows-parity-closeout/publication-receipt.json`
only after helper success. Runtime source and numerical receipts stay bound to
`7c9067bc`; documentation publication performs no new production forwards.

## GenEye consumer qualification

GenEye's local worker selects the additive Q8Rows constructor at Candle
`3fa7075e536e66b024d14cee3f6f423d81d67162`. Its lock and packaging policy bind
that revision. Private CPU/CUDA workers report shared 278,528,000-byte Q8 storage
with zero additional lookup bytes. The separately approved two-load/four-request
text/image proof preserves all 52 numeric answer values exactly against each
matching-device producer, native dispatch, key order and complete process release.
That allowance is closed. The original private-worker proof retains its source
and binary identities; the retained image-case quality limitation is unchanged.
Its exact source and receipts are in
[HISTORY](HISTORY.md#2026-10-11---d1-review-recheck-and-consumer-handoff-cleanup).

GenEye's Q8Rows adoption is now published at
`8e97e3327a82be9b09a1eb1903aaad2a3efbaa68`; package progress records are published
at `0d552b0ec3db47afe62c373e5eefa283c6aac602`. The clean CPU/CUDA candidates bind
the same Candle revision, lock and worker provenance. Their 33-file package
inventories and model-free checks pass; the changed Linux process owner has
focused Windows/WSL consumer proof. These checks do not relabel the earlier
numerical receipt or qualify the new package's model execution.

The fresh approved package attempt stopped at CPU owner admission for competing
workloads before private-worker startup. It loaded no model, made no request or
forward, generated no token and left all four CPU/CUDA slots unstarted. The
allowance is consumed and closed; CUDA was not started. Process-release proof
passes. GenEye owns a quiet admission window and a newly reviewed finite
allowance before rerunning CPU first, then CUDA only after CPU passes.
LTS activation remains separately gated.

Retained package records are in
`D:\GenEye-evidence\candle-q8rows-portable-20261011`: `checks-cpu.json`,
`checks-cuda.json`, `portable-terminal.json`, `publication.json` and
`package-model-proof/process-release.json`. This review rehashes thirteen bound
packet/package/terminal objects; all match. The terminal receipt is SHA-256
`56a97b9607e187e56636b3d65bb8e4e43135d35cd40ff18afbf5ac9fe3efdaa7`.

---
AI-edited: 2026-10-12 | agent=Codex/root | model=unknown | effort=unknown | task=maintained-models | change=implemented and verified scoped model improvements
