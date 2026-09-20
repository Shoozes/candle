# GPT-OSS Experimental Candle Status

## Assignment state

- Current phase: Task 3 / packed MXFP4 CUDA executor, accepted on the bounded
  synthetic fixture and published separately from the maintained product path.
- Task 3 starting baseline: clean published `main` at
  `eed8ef6594a9012f3ed61a5c1a1d06f2af0ee068`, tree
  `38fc50f0c3cbfe6e3d775633cbbf782c2479da96`.
- Task 1 and Task 2 remain accepted checkpoints. The Task 3 publication
  commit, tree, and remote receipt are recorded by the guarded publication
  helper in `artifacts/publication/last-push.json`.
- Product artifact identity: the owner-selected GGUF SHA-256 is
  `aab205256a9b6361e410c24de3086e30f907092ca6f9ba8cd4b22c8a2b025778`.
  That external file is not present in this checkout and was not downloaded.

## Accepted capability rows

| Capability | State | Evidence boundary |
| --- | --- | --- |
| Exact artifact admission | Accepted | `GptOssGgufArtifact::open` hashes before parsing and admits only the selected SHA-256. |
| GGUF directory/config normalization | Accepted | Strict v2/v3 parser, `gpt-oss` to `gpt_oss` normalization, YaRN/context and dimension checks. |
| Packed MXFP4 ownership | Accepted | CUDA keeps expert blocks/scales as device U8 tensors and dispatches a dedicated packed kernel; no dense expert fallback exists. |
| Packed expert/router/attention execution | Accepted | CUDA component trace and direct packed output match the independent oracle at absolute tolerance `1e-4`. |
| CUDA forward/cache execution | Accepted | Uncached logits and cached prefill/decode match the oracle; exact logical cache usage is reported. |
| Explicit CUDA selection | Accepted | Public feature-gated config selects device index and rejects non-F32 activation dtype with typed errors. |
| Static/runtime resource admission | Accepted | Logical static weight bytes are checked before device creation; token and exact KV-byte admission happen before forward mutation. |
| Cancellation/rollback/eviction | Accepted | Cancellation before allocation and at admitted work boundaries leaves the retained cache unchanged; eviction releases logical cache state and retry succeeds. |
| Load/process ownership | Accepted | Task 2 RAII registry tests return active/loaded ownership to zero; Task 3 creates no worker process and owns device state only through Candle RAII handles. |
| Product artifact load/parity | Excluded | No external artifact receipt or real tensor execution was available. |
| Quantized text/split dense MMProj | Deferred | The ordered post-CUDA task; no Q8/LFM2 defaults or Edge/Harmony integration were changed. |

## Proven evidence

- Independent fixture: `tests/fixtures/gpt_oss_task2/oracle.json`, fixture ID
  `synthetic-gpt-oss-task2-oracle-v1`, 2,690 raw bytes, SHA-256
  `db79441a688bce500217635051372270fce02b7e6f8f03d769855ac290d4ab04`.
  The pinned standalone Python 3.13.3 / NumPy 2.3.3 scalar reference covers
  packed expert output, router indices/weights, attention transitions,
  uncached logits, and cached prefill/decode logits at absolute tolerance
  `1e-4`.
- CUDA proof ran on native Windows/MSVC with Rust/Cargo 1.97.1, CUDA/nvcc
  13.3, NVIDIA driver 616.92, and an NVIDIA GeForce RTX 4090 with 24,564 MiB.
  The focused feature-gated GPT-OSS run passed `30/30` tests: 27 retained
  Task 2 CPU tests plus 3 Task 3 CUDA tests.
- The synthetic CUDA model admits `19,416` logical static weight bytes, of
  which `3,264` are packed MXFP4 block/scale bytes. Its exact cache contract
  remains `128` logical bytes per token and `512` bytes at the four-token
  bound. The tests cover the exact bound and one-over rejection.
- The CUDA component trace compares attention values, router values, sorted
  expert indices, and normalized expert weights for tokens 1 and 3, then
  compares packed output and final cached/uncached logits. The executor uses
  F32 dense transformer tensors plus U8 packed expert tensors; it does not
  silently dequantize MXFP4 weights.
- Pinned behavioral sources remain OpenAI `gpt-oss` commit
  `7b583341fe16729127f6d5b94a7b09ccae97e1a1` and llama.cpp commit
  `f072b103714dfa1eee531f80b24512faf38e3dd2`; see `SOURCES.md`. The CUDA
  kernel itself is a fresh Candle-native implementation.
- The maintained Q8/default path is unchanged. No Edge-owned Harmony/profile
  integration, training, hidden download, exact product model load, or
  tokenizer work was added.

## Public boundary

`GptOssCudaModel` is an opt-in, `cuda`-feature-gated executor over admitted
synthetic/native `GptOssWeights`. `GptOssCudaConfig` makes the device index,
F32 activation dtype, static weight budget, and existing token/KV limits
explicit. `GptOssCudaError` distinguishes unsupported dtype, unavailable
device, resource limit, overflow, cancellation, invalid input, backend, and
kernel failures. Forward work stages a cloned cache and commits it only after
all checkpoints pass; Candle tensor/device handles provide cleanup ownership.

The CUDA path is not wired into the GGUF product loader, tokenizer, generation
CLI, or maintained Q8/LFM2-VL defaults. It is a bounded execution proof, not
an exact-model or production-support claim.

## Known limitations and blockers

- The selected product GGUF is an owner-admitted external input, not a
  checked-out file. Exact-model loading, tokenizer compatibility, and
  production numerical parity remain unproven.
- Only F32 activations are accepted by this executor. Quantized text and split
  dense MMProj execution are intentionally deferred to the next ordered task.
- The CUDA receipt is a synthetic fixture result on one Windows RTX 4090
  lane; it is not a cross-device matrix or a production throughput claim.
- The existing Windows linker warning `LNK4098` remains an environment/build
  warning in the CUDA test binary; it did not fail the executed tests.

## Last green verification

Native Windows/MSVC, locked offline dependencies:

- `cargo fmt --all -- --check`: passed.
- `cargo check --locked --offline -p candle-transformers`: passed.
- `cargo check --locked --offline -p candle-transformers --features cuda`:
  passed.
- `cargo clippy --locked --offline -p candle-transformers --features cuda --lib -- -D warnings`:
  passed.
- `cargo test --locked --offline -p candle-transformers --features cuda gpt_oss`:
  passed `30/30`, including direct packed-kernel parity, CUDA attention/router
  component traces, uncached/cached oracle parity, cancellation rollback,
  exact admission, eviction/retry, typed dtype rejection, and pre-device
  static-budget rejection.
- `git diff --check`: passed before publication; the final guarded helper also
  reran the repository's required CPU checks, summary-bank verifier,
  GPT-OSS manifest verifier, LFM2-VL manifest verifier, and fork-overlay
  verifier.

## Exact next task

Task 4: add quantized GPT-OSS text execution with the split dense MMProj
boundary. Preserve this Task 3 executor as an opt-in feature-gated path; do
not start exact-product loading, tokenizer integration, Q8/default changes, or
CUDA optimization beyond this bounded proof without their own acceptance gate.

---

AI-edited: 2026-09-20; agent=Codex; task=gpt-oss-task3; change=accepted packed CUDA executor and bounded native proof
