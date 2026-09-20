# GPT-OSS Experimental Candle Status

## Assignment state

- Current phase: Task 2 / C2b independent numerical and bounded resource proof.
- Baseline: clean published `main` at
  `f9c51b4fb2717342644d75806f99d92da78d3f9a`.
- Immutable Task 1 source revision: `67a7fe605194384c1505df96b536c3238ad408e1`.
- Task 2 delivery: implementation, fixture, and handoff records are governed
  by the repository's guarded publication helper; no exact-model run is
  implied.
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
| Independent numerical oracle | Accepted | `synthetic-gpt-oss-task2-oracle-v1`, raw SHA-256 `db79441a688bce500217635051372270fce02b7e6f8f03d769855ac290d4ab04`, absolute tolerance `1e-4`. |
| Token/cache byte admission | Accepted | Exact synthetic configuration requires 128 logical KV bytes/token; exact four-token and one-over sequence/byte cases pass before mutation. |
| Cancellation and rollback | Accepted | Load cancellation releases its in-progress lease; prefill cancellation after partial cache work restores logical state; cancelled decode and injected forward failure preserve the retained cache. |
| Duplicate/load ownership | Accepted | RAII registry rejects active/loaded duplicates, releases cancelled and failed loads for retry, and drops completed ownership to zero. |
| Product artifact load/parity | Excluded | No external artifact receipt or real tensor execution was available. |
| CUDA executor | Excluded | Task 3 is separately reviewed and remains ordered behind this CPU gate. |

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
- Independent Task 2 oracle: `tests/fixtures/gpt_oss_task2/oracle.json`, fixture
  ID `synthetic-gpt-oss-task2-oracle-v1`, 2,690 raw bytes, SHA-256
  `db79441a688bce500217635051372270fce02b7e6f8f03d769855ac290d4ab04`.
  The pinned standalone Python 3.13.3 / NumPy 2.3.3 scalar reference covers
  packed expert output, router indices/weights, attention transitions,
  uncached logits, and cached prefill/decode logits at absolute tolerance
  `1e-4`; the Rust test verifies the raw digest before comparison.
- Resource evidence: the tiny normalized config retains 128 logical KV-cache
  bytes per token. Four-token / 512-byte admission passes at the exact bound;
  sequence one-over, byte one-over, and one-byte-under cases fail before cache
  mutation. Partial cancellation leaves zero logical cache bytes, reset drops
  the owned capacity to zero, and registered cancelled/failed/successful load
  leases return active/loaded ownership counts to zero.
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
`GptOssResourceLimits`, `GptOssCancellationToken`, and
`GptOssLoadRegistry` are opt-in CPU proof contracts; `GptOssModel` retains the
existing `forward` behavior while adding bounded prefill/decode seams. No
production checkpoint executor is exposed.

## Known limitations and blockers

- The selected product GGUF is an owner-admitted external input, not a checked-
  out file, so this task proves the loader contract and rejection behavior but
  does not claim exact-model load or numerical parity.
- The selected product GGUF remains absent, so the independent oracle is only a
  synthetic CPU receipt and cannot establish exact-model parity.
- The packed CUDA executor remains unimplemented and requires a separate owner
  acceptance after this CPU proof gate.

## Last green verification

Native Windows/MSVC, locked offline dependencies:

- `cargo test --locked --offline -p candle-transformers gpt_oss`: **19/19**
  focused GPT-OSS baseline tests passed before Task 2; the Task 2 focused run
  passed **27/27**.
- `cargo check --locked --offline -p candle-transformers`: passed.
- `cargo clippy --locked --offline -p candle-transformers --lib -- -D warnings`:
  passed.
- `cargo fmt --all -- --check` and `git diff --check`: passed.
- `cargo test --locked --offline -p candle-transformers gpt_oss`: **27/27**
  focused Task 2 tests passed, including the digest-pinned independent oracle,
  exact-boundary resource admission, cancellation/reset release, and
  registered-load retry/duplicate checks.
- `pwsh -NoProfile -File scripts/lfm2-vl/verify-summary-bank.ps1`: passed;
  GPT-OSS Task 2 proof route 9 files / 147.5 KiB, max 256 KiB.
- Git-for-Windows `scripts/gpt-oss/verify-mod-manifest.sh`: passed.
- Git-for-Windows `scripts/lfm2-vl/verify-mod-manifest.sh`: passed with
  17 fork-origin modifications and 146 additions.
- Git-for-Windows `scripts/verify-fork-overlays.sh`: passed against rolling
  baseline `d830e03078a29d39a1aacd741620475eb33b7609`, 187 registered paths,
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

Task 3: separately review and, only with owner acceptance, implement a packed
CUDA executor. Keep exact-model parity, production artifact loading, and CUDA
claims excluded until their own artifact and numerical gates are published.

---

AI-edited: 2026-09-20; agent=Codex; task=gpt-oss-task2; change=independent numerical oracle and bounded resource proof
