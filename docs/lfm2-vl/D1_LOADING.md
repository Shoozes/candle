# Optional d1 loading qualification

The selected Work batch starts at published `ebf2f85e96d264c3d78f7f35b22da3cc0a9abcaf`
and preserves the separately proven, unpublished example-output repairs.
Only one loading option will be implemented. Existing constructors and default
execution remain unchanged. No lower-bit conversion, new weights, training,
consumer edits or publication belong to this batch.

## Candidate comparison and frozen measurements

The retained d1 GGUF has a tied Q8_0 token/output table with shape
`[128000, 2048]`. Its bounded 8,240,736-byte header has SHA-256
`6cd942c246c9adefd2a594eb328fc5a407859fc3271b8d62336735aebf5b2de2`.
The table occupies 278,528,000 bytes (265.625 MiB); the current F32 lookup copy
adds 1,048,576,000 bytes (1,000 MiB). Whole-file identities remain those of
`tools/lfm2_vl/d1/reference-lock.json` and are rechecked for each process.

- Early quant-inventory admission would reject unsupported files sooner, but
  does not remove allocations when these retained supported artifacts load.
- Lazy vision loading could avoid the retained 556 MiB projector file on text
  requests; it would also need a new session lifecycle and image admission
  contract. Its precise resident savings require a separate implementation.
- Retained Q8 embedding rows can reuse the existing tied output allocation and
  avoid the 1,000 MiB F32 copy. Existing CPU/CUDA row lookup produces F32 results.

Before changing the model loader, use the new example `--load-only` harness for
three sequential fresh-process loads on CPU and three on CUDA. After implementing
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

Select retained Q8 embedding rows first. The baseline's first CPU constructor
took 2,920.682 ms with 5,149,970,432 bytes peak Job memory. The tied-table header
establishes a removable 1,000 MiB allocation, larger than the retained vision
file. Early full-model quant inventory and lazy vision remain separate tasks.

Add `quantized_lfm2::Lfm2EmbeddingLoadMode::{DenseF32, Q8Rows}` and
`ModelWeights::from_gguf_with_embedding_mode(content, reader, device, mode)`.
Keep `from_gguf` and the hash-bound LoRA constructor on `DenseF32` by default.
Add `load_lfm2_vl_hybrid_with_embedding_mode(options, mode)` and
`load_d1_q8_with_embedding_mode(options, mode)` without adding required fields
to existing options structs. Existing entry points retain their defaults.

The explicit `Q8Rows` path admits only a positive, block-aligned Q8_0 embedding
matrix on CPU/CUDA, with checked sizes and metadata width/EOS bounds, before
reading any text tensor payload. Retain its immutable `Arc<QTensor>` and share
it with the output head when the source is tied and output execution stays
quantized. Dequantize only requested rows to F32 through Candle's existing row
primitive. Preserve all major Q8 linears, the projector's original native mix,
cache semantics, prompt/tokenizer behavior and exact consumed-file inventory.
The whole vision model still loads; this mode changes only token-table storage.

Reject unsupported embedding storage/devices, overflow, unsupported ID dtypes,
negative IDs and out-of-vocabulary IDs with controlled errors. Validate IDs
before CUDA's row gather, which has no device-side bounds check. Empty ID inputs
return an empty F32 tensor of the corresponding shape without device dispatch.

Expose `Lfm2EmbeddingStorage` through model/composite/session getters. Report
mode, vocabulary/hidden width, lookup-table bytes, whether output shares the
same allocation, and additional bytes relative to the output head. This keeps
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
progress publication on 2026-10-10. This saves the locally verified option and
runner repairs while keeping production qualification open. The exact saved
commit/tree/remote identity is retained in
`artifacts/lfm2-d1/20261010-loading-closeout/publication-receipt.json` only after
helper success. Consumer mode adoption and pin/package proof remain separate.

Production qualification is pending a fresh allowance for the frozen two-case
`t01`/`i01` input (SHA-256
`932966c0115d584a51eef8d0e0deaaf4af01f2a0aca366a186c13b1c9d1d6432`).
The proposed comparison executes both modes on CPU first, then both on CUDA:
eight requests, 24 language and four vision forwards, zero output tokens and
no retries, using 16 GiB/1,800-second Jobs. Stop if CPU parity fails. No new
production forwards have occurred in this Work batch.

---
AI-edited: 2026-10-10 | agent=Codex/root | model=unknown | effort=unknown | task=d1-loading-closeout | change=recorded verified progress publication and retained production qualification boundary
