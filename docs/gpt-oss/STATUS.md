# GPT-OSS Experimental Candle Status

## Assignment state

- Current phase: Task 1 / C2a exact GGUF admission and configuration/tensor
  normalization.
- Baseline: clean published `main` at
  `d830e03078a29d39a1aacd741620475eb33b7609`.
- Immutable source revision: `67a7fe605194384c1505df96b536c3238ad408e1`.
- Source tree: `fa35de0a6438992e5aa6bc6b4429bb18ed44005b`.
- Delivery status: source checkpoint is committed; this status and the shared
  handoff records are the follow-up delivery commit. Publication is governed by
  the repository's guarded helper; no exact-model run is implied.
- Product artifact identity: the owner-selected GGUF SHA-256 is
  `aab205256a9b6361e410c24de3086e30f907092ca6f9ba8cd4b22c8a2b025778`.
  That external file is not present in this checkout and was not downloaded.

## Accepted capability rows

| Capability | State | Evidence boundary |
| --- | --- | --- |
| Exact artifact admission | Accepted | `GptOssGgufArtifact::open` hashes before parsing and admits only the selected SHA-256. |
| GGUF directory/config normalization | Accepted | Strict v2/v3 parser, `gpt-oss` to `gpt_oss` normalization, YaRN/context and dimension checks. |
| Tensor ownership | Accepted | Complete fused or converter-style split inventory with role, logical shape, dtype, aligned offset, and bounded byte length. |
| Raw payload ownership | Accepted | On-demand reads re-check file size and SHA-256; no dequantization or dense residency is introduced. |
| Product artifact load/parity | Excluded | No external artifact receipt or real tensor execution was available. |
| Token/cache resource proof | Next | Task 2 remains required before any executor work. |
| CUDA executor | Excluded | Task 3 is ordered behind Task 2. |

## Proven evidence

- Synthetic fixture ID: `synthetic-gpt-oss-gguf-c2a-v1`.
  The in-source GGUF builder covers the fused inventory and the pinned
  llama.cpp converter-style split expert inventory, including canonical alias
  names and MXFP4 type 39 byte lengths.
- Rejection cases: wrong SHA identity before parsing, truncated payload,
  malformed magic, incomplete tensor identity, renamed tensor, and config/tensor
  shape mismatch. The successful fixtures prove normalized config values,
  tensor roles, logical shapes, dtype ownership, and the 17-byte/32-value
  MXFP4 raw block contract.
- Pinned source references: OpenAI `gpt-oss` commit
  `7b583341fe16729127f6d5b94a7b09ccae97e1a1` and llama.cpp commit
  `f072b103714dfa1eee531f80b24512faf38e3dd2`; see `SOURCES.md`.
- The maintained Q8/default path is unchanged. No Edge-owned Harmony/profile
  integration, training, hidden download, live generation, or CUDA work was
  added.

## Public boundary

`GptOssGgufArtifact` is an execution-free, local-path admission boundary. It
parses only bounded GGUF metadata, rejects unsupported dtypes and malformed or
overlapping ranges, normalizes the selected configuration into
`GptOssConfig`, and exposes only owned tensor descriptors plus raw payload reads.
The existing `GptOssCheckpoint`, packed `Mxfp4`, and synthetic CPU/cache
interfaces remain separate and unchanged in behavior.

## Known limitations and blockers

- The selected product GGUF is an owner-admitted external input, not a checked-
  out file, so this task proves the loader contract and rejection behavior but
  does not claim exact-model load or numerical parity.
- Task 2 must add independent numerical fixtures not computed by the loader,
  explicit token/cache byte bounds, cancellation/rollback, and no-duplicate or
  load-leak evidence.
- Only after Task 2 is accepted may the packed CUDA executor be implemented.

## Last green verification

Native Windows/MSVC, locked offline dependencies:

- `cargo test --locked --offline -p candle-transformers gpt_oss`: **19/19**
  focused GPT-OSS tests passed.
- `cargo check --locked --offline -p candle-transformers`: passed.
- `cargo clippy --locked --offline -p candle-transformers --lib -- -D warnings`:
  passed.
- `cargo fmt --all -- --check` and `git diff --check`: passed.
- `pwsh -NoProfile -File scripts/lfm2-vl/verify-summary-bank.ps1`: passed;
  GPT-OSS route 12 files / 241.7 KiB, max 256 KiB.
- Git-for-Windows `scripts/gpt-oss/verify-mod-manifest.sh`: passed.
- Git-for-Windows `scripts/lfm2-vl/verify-mod-manifest.sh`: passed with
  17 fork-origin modifications and 146 additions.
- Git-for-Windows `scripts/verify-fork-overlays.sh`: passed against rolling
  baseline `d830e03078a29d39a1aacd741620475eb33b7609`, 184 registered paths,
  3 overlays, 20 shared paths.
- `pwsh -NoProfile -File .tools/verify-before-push.ps1`: passed. Its broader
  native gate passed 21 Candle-core unit tests and 164 active Candle-core
  integration tests (bilinear 12, conv 9, custom-op 4, display 3, GGUF 10,
  grad 9, indexing 4, layout 2, matmul 11, pool 4, PTH 3, quantized 39,
  serialization 3, tensor 51; cutile and Metal suites had 0 active tests),
  110 transformer unit tests, 5 generation tests, 8 NMS tests, 37 VLM tests,
  and 33 LFM2-VL example tests; doc tests passed with one documented ignored
  Candle-core test and one documented ignored transformer test.

## Exact next task

Task 2: add independent numerical GGUF fixtures and prove token/cache byte
bounds, cancellation/rollback, and no-duplicate/load-leak behavior. Do not
start the packed CUDA executor or claim exact-model parity until that gate is
accepted.

---

AI-edited: 2026-09-20; agent=Codex; task=gpt-oss-c2a; change=hash-pinned GGUF admission and normalized tensor ownership
