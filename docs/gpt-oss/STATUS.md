# GPT-OSS Experimental Candle Status

## Assignment state

- Current phase: Task 3 correctness correction / packed MXFP4 CUDA executor
  plus bounded GGUF-to-`GptOssWeights` assembly, with the published boundary
  preserved; this slice also repairs the repository-wide overlay verifier and
  carries its isolated temporary-repository regression suite, all pending owner
  review.
- Current baseline: clean published `main` at
  `2ac78ba5160ad939f70374fdbef966c91fd6e426`, tree
  `18b70631609c969ba2d82972b6c287f84f2b1cee`.
- Task 1 and Task 2 remain accepted checkpoints. The Task 3 publication
  commit, tree, and remote receipt are recorded by the guarded publication
  helper in `artifacts/publication/last-push.json`.
- Product artifact identity: the owner-selected GGUF SHA-256 is
  `aab205256a9b6361e410c24de3086e30f907092ca6f9ba8cd4b22c8a2b025778`.
  The external file remains outside this checkout; it was read in place only
  at the owner-provided path and was not copied, downloaded, or committed.

## Accepted capability rows

| Capability | State | Evidence boundary |
| --- | --- | --- |
| Exact artifact admission | Accepted | `GptOssGgufArtifact::open` hashes before parsing and admits only the selected SHA-256. |
| GGUF directory/config normalization | Accepted | Strict v2/v3 parser, `gpt-oss` to `gpt_oss` normalization, YaRN/context and dimension checks. |
| Packed MXFP4 ownership | Accepted | CUDA keeps expert blocks/scales as device U8 tensors and dispatches a dedicated packed kernel; no dense expert fallback exists. |
| Packed expert/router/attention execution | Accepted | CUDA component trace and direct packed output match the independent oracle at absolute tolerance `1e-4`. |
| Expert contribution/residual contract | Accepted | `Mxfp4ExpertOperation::forward_contribution` returns only weighted expert output; CPU and CUDA add it directly to the post-attention hidden state. |
| Post-attention/post-MoE traces | Accepted | CPU and CUDA expose both hidden states in the two-layer independent fixture, including sliding/full attention, sinks, unequal routing, and position 3. |
| CUDA forward/cache execution | Accepted | Uncached logits and cached prefill/decode match the oracle; exact logical cache usage is reported. |
| Explicit CUDA selection | Accepted | Public feature-gated config selects device index and rejects non-F32 activation dtype with typed errors. |
| Static/runtime resource admission | Accepted | Logical static weight bytes are checked before device creation; token and exact KV-byte admission happen before forward mutation. |
| Cancellation/rollback/eviction | Accepted | Cancellation before allocation and at admitted work boundaries leaves the retained cache unchanged; eviction releases logical cache state and retry succeeds. |
| CUDA narrowed-view equivalence | Accepted | Padded/narrowed CUDA views match the materialized-copy path; pre-launch device/dtype/shape checks run before dispatch and cudarc pointer guards remain alive through launch. |
| GGUF-to-weights assembly | Accepted | Fused and converter-style split synthetic fixtures assemble and execute on CPU; the exact owner-selected 13,792,638,656-byte artifact assembles under the 32 GiB resident bound. |
| Load/process ownership | Accepted | The combined loader retains one bounded file session, returns a load lease, and releases registry ownership after the exact model object is dropped; no worker process or hidden download is created. |
| Product artifact numerical parity | Deferred | Exact assembly and model construction are proven, but tokenizer integration, forward logits, and production numerical parity remain outside this bounded round. |
| Quantized text/split dense MMProj | Deferred | The ordered post-CUDA task; no Q8/LFM2 defaults or Edge/Harmony integration were changed. |

## Proven evidence

- Independent fixture: `tests/fixtures/gpt_oss_task2/oracle.json`, fixture ID
  `synthetic-gpt-oss-task2-oracle-v1`, 2,690 raw bytes, SHA-256
  `db79441a688bce500217635051372270fce02b7e6f8f03d769855ac290d4ab04`.
  The pinned standalone Python 3.13.3 / NumPy 2.3.3 scalar reference covers
  packed expert output, router indices/weights, attention transitions,
  uncached logits, and cached prefill/decode logits at absolute tolerance
  `1e-4`.
- Independent two-layer correction fixture:
  `tests/fixtures/gpt_oss_task3_two_layer/oracle.json`, fixture ID
  `synthetic-gpt-oss-task3-two-layer-oracle-v1`, 10,525 raw bytes, SHA-256
  `0530081353cc7268d796ae070c4168ec4e6bb036f62d54670b975a43556bc173`.
  Standalone Python 3.13.3 / NumPy 2.3.3 values cover final logits and
  attention, router, expert-contribution, post-attention, and post-MoE traces
  for a four-token, two-layer sliding/full-attention sequence at absolute
  tolerance `1e-4`.
- CUDA proof ran on native Windows/MSVC with Rust/Cargo 1.97.1, CUDA/nvcc
  13.3, NVIDIA driver 616.92, and an NVIDIA GeForce RTX 4090 with 24,564 MiB.
  The focused feature-gated GPT-OSS run passed `34/34` tests: 29 CPU tests
  plus 5 CUDA-gated tests, including the zero-expert residual regression and
  the two-layer CPU/reference/CUDA trace comparison.
- The synthetic CUDA model admits `19,416` logical static weight bytes, of
  which `3,264` are packed MXFP4 block/scale bytes. Its exact cache contract
  remains `128` logical bytes per token and `512` bytes at the four-token
  bound. The tests cover the exact bound and one-over rejection.
- The CUDA component trace compares attention values, router values, sorted
  expert indices, normalized expert weights, post-attention hidden states,
  expert contributions, and post-MoE hidden states. The executor uses F32
  dense transformer tensors plus U8 packed expert tensors; it does not
  silently dequantize MXFP4 weights.
- The CUDA packed-kernel narrowed-view regression compares padded non-zero-offset
  views with a fully materialized copy. Non-contiguous or narrowed inputs are
  copied explicitly before launch, and `DevicePtr`/`DevicePtrMut` guards live
  through the kernel call.
- The synthetic GGUF assembly regression covers fused experts plus converter-
  style split gate/up experts and split Q/K/V tensors, then runs one CPU
  forward. The owner-selected artifact test used only
  `C:\llamacpp\models\gpt-oss-20b-mxfp4.gguf`, verified the pinned SHA, assembled
  `GptOssWeights`, constructed `GptOssModel`, and returned the registry count to
  zero after release. No model bytes entered the repository.
- Pinned behavioral sources remain OpenAI `gpt-oss` commit
  `7b583341fe16729127f6d5b94a7b09ccae97e1a1` and llama.cpp commit
  `f072b103714dfa1eee531f80b24512faf38e3dd2`; see `SOURCES.md`. The CUDA
  kernel itself is a fresh Candle-native implementation.
- The maintained Q8/default path is unchanged. No Edge-owned Harmony/profile
  integration, training, hidden download, exact product model load, or
  tokenizer work was added.

## Public boundary

`GptOssGgufArtifact::load_weights` and
`load_gpt_oss_weights_with_cancellation` assemble admitted dense F32/BF16
text tensors and packed MXFP4 expert tensors through one bounded retained file
session. The public combined loader retains a registry lease until the caller
drops the returned handle.

`GptOssCudaModel` is an opt-in, `cuda`-feature-gated executor over admitted
synthetic/native `GptOssWeights`. `GptOssCudaConfig` makes the device index,
F32 activation dtype, static weight budget, and existing token/KV limits
explicit. `GptOssCudaError` distinguishes unsupported dtype, unavailable
device, resource limit, overflow, cancellation, invalid input, backend, and
kernel failures. Forward work stages a cloned cache and commits it only after
all checkpoints pass; Candle tensor/device handles provide cleanup ownership.

The CUDA path is not wired into the GGUF product loader, tokenizer, generation
CLI, or maintained Q8/LFM2-VL defaults. Q8_0 dense text assembly remains
explicitly deferred. The exact artifact result proves bounded weight assembly
and model construction, not tokenizer or production numerical support.

## Known limitations and blockers

- The selected product GGUF is an owner-admitted external input, not a
  checked-out file. Its exact admission, bounded weight assembly, and model
  construction now pass; tokenizer compatibility, forward-logit parity, and
  production support remain unproven.
- Only F32 activations are accepted by this executor. Quantized text and split
  dense MMProj execution are intentionally deferred to the next ordered task.
- The CUDA receipt is a synthetic fixture result on one Windows RTX 4090
  lane; it is not a cross-device matrix or a production throughput claim.
- The existing Windows linker warning `LNK4098` remains an environment/build
  warning in the CUDA test binary; it did not fail the executed tests.
- The rolling repository-wide overlay gate now passes with the live union
  baseline, including committed, staged, unstaged, and untracked candidate
  paths. The isolated `scripts/tests/test-verify-fork-overlays.sh` regression
  suite covers allowed dirty paths, an unowned path, rename/deletion ownership,
  invalid and missing baselines, and the rolling/upstream distinction.
- `--upstream-baseline 6f74e7c` remains an exact-delta frozen-receipt check. It
  correctly rejects this current rolling candidate because that older delta
  contains unrelated upstream paths outside the registered overlays; this is
  not a blocker for the live rolling gate.

## Last green verification

Native Windows/MSVC, locked dependencies for this correction and assembly
slice:

- `cargo fmt --all -- --check`: passed.
- `pwsh -NoProfile -File .tools/verify-before-push.ps1`: passed the complete documented local
  format/check/example/test/summary/module-layout/rolling-overlay/whitespace gate.
- `cargo check --locked -p candle-core`: passed.
- `cargo check --locked -p candle-nn`: passed.
- `cargo check --locked -p candle-transformers`: passed.
- `cargo check --locked -p candle-vlm`: passed.
- `cargo check --locked --features cuda -p candle-transformers`: passed.
- `cargo clippy --locked -p candle-transformers --lib -- -D warnings` and
  `cargo clippy --locked --features cuda -p candle-transformers --lib -- -D
  warnings`: passed.
- `cargo test --locked -p candle-transformers --lib models::gpt_oss`:
  passed `30/30` CPU-focused tests, with one ignored exact-artifact test.
- `cargo test --locked --features cuda -p candle-transformers --lib models::gpt_oss`:
  passed `36/36`, with one ignored exact-artifact test, including direct packed-kernel parity, CUDA attention/router
  component traces, the zero-expert residual regression, two-layer
  CPU/reference/CUDA traces, uncached/cached oracle parity, cancellation
  rollback, exact admission, narrowed-view/materialized-copy equivalence,
  eviction/retry, typed dtype rejection, and pre-device static-budget
  rejection.
- `cargo test --locked -p candle-transformers --lib models::gpt_oss::gguf::tests::assembles_owner_selected_product_artifact_when_requested -- --ignored --exact --nocapture`, with
  `CANDLE_GPT_OSS_GGUF=C:\llamacpp\models\gpt-oss-20b-mxfp4.gguf` and
  `CANDLE_GPT_OSS_MAX_RESIDENT_BYTES=34359738368`: passed in `379.65s`.
  The test verified the pinned SHA, assembled the exact artifact, constructed
  a model, and released the registry lease.
- `git diff --check`: passed. The final guarded helper before publication also
  reran the repository's required CPU checks. The current correction slice also
  passed the summary-bank verifier, the GPT-OSS manifest verifier with
  baseline `2ac78ba5`, and the LFM2-VL manifest verifier with its recorded
  baseline `7c2e8929`. The rolling fork-overlay verifier passed with the
  live `d830e030` baseline, and the isolated seven-check verifier regression
  suite passed.
- `bash scripts/tests/test-verify-fork-overlays.sh`: passed seven isolated
  temporary-repository checks, including the dirty candidate path count.
- `bash scripts/verify-fork-overlays.sh --rolling-baseline d830e030...`: passed
  with 193 registered union paths and 21 shared paths.

## Exact next task

The verifier files under active work are scripts/verify-fork-overlays.sh,
scripts/tests/test-verify-fork-overlays.sh, docs/FORK_OVERLAYS.md, and the
GPT-OSS overlay manifest.

Owner review and, if authorized, publication of this correction and bounded
assembly from the clean `2ac78ba5` baseline. After that acceptance, Task 4 may
add quantized GPT-OSS text execution with the split dense MMProj boundary.
Preserve this Task 3 executor and loader as opt-in paths; do not infer
tokenizer, forward-logit, Q8/default, or production parity from the exact
assembly result.

---

AI-edited: 2026-09-20; agent=Codex; task=gpt-oss-task3-cuda-assembly; change=added narrowed-view CUDA equivalence and bounded synthetic/exact GGUF-to-weights assembly
