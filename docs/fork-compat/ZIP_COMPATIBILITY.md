# ZIP dependency compatibility

## Selected maintenance boundary

The verified repair starts at clean Candle main
`b70f5f3264d88413ee77e7e7c9827644590de3c7`. It moves the workspace ZIP requirement
from `8.6.0` to `9.0.4`, retaining `default-features = false` and the existing
CPU/optional-accelerator contracts. Core inherits this requirement; there is no
consumer patch, alternate registry, vendoring or application dependency.
The root lock changes only ZIP and its required BSD/macOS `nix` dependency.
The cached sources suffice; no package/model download is part of this batch.

RustSec DB `999ec89d5d3f10f20be14750ec8c2d8b44e0ac9b` marks ZIP versions
below 9.0.0 affected by
[RUSTSEC-2026-0367](https://rustsec.org/advisories/RUSTSEC-2026-0367.html),
[RUSTSEC-2026-0368](https://rustsec.org/advisories/RUSTSEC-2026-0368.html) and
[RUSTSEC-2026-0369](https://rustsec.org/advisories/RUSTSEC-2026-0369.html).
These concern extraction permissions and symlink handling. Candle's NPZ/PyTorch
readers read named members as tensor data; they do not extract archive paths.
The dependency repair does not add extraction or enable compression features.

## Reader adaptation

ZIP 9's `name`/`file_names` return fallible decoded names. Propagate their errors
through the existing Candle error boundary. The private NPZ-owned name-index
adapter is shared by these two tensor formats: UTF-8/raw lookup remains the
constant-time path; legacy CP437 names fall back to a decoded-name scan only
when raw lookup misses. NPZ all/named/lazy and PyTorch metadata/storage reads
retain decoded keys. Missing tensors retain their previous error/None behavior.
No public Candle function, DTO, archive policy or model arithmetic changes.
The underlying ZIP error type in public `Error::Zip` now comes from ZIP 9.
Consumers explicitly naming ZIP 8's concrete error type must adapt accordingly.

The focused CPU gate is:

```powershell
cargo test --locked --offline -j 2 -p candle-core --test archive_tests --test serialization_tests --test pth_tests
```

It covers existing PyTorch optional-key/Fortran fixtures, NPZ writing and
all/named/lazy loading, Unicode and CP437 keys, empty tensors, truncated archives,
malformed headers/payloads/metadata/storage, unsupported compression and caller
file preservation. Parent-directory member names remain keys and never become
filesystem writes. Fixtures use bounded tiny payloads and scoped temporary
folders; no Python/model runtime is required. A CP437 named-read regression
failed after the initial ZIP 9 API adaptation and passed after decoded lookup.
ZIP 9 decodes non-UTF8 names as CP437; this repair preserves those names
instead of imposing a new strict-UTF8 rejection policy.

## Consumer and audit proof

Local proof and the exact dirty candidate hashes are retained under
`D:/Compact_Evidence/candle/runs/zip-parent-20261017/summary.json`.
The standalone consumer has its own workspace/lock and references all four
Candle crates through local paths, with no direct ZIP dependency or patch.
It exercises an NPZ `VarBuilder` and a tiny retained PyTorch fixture on CPU.
This is provisional local dependency/API proof, not immutable Git-dependency
or Edge adoption acceptance. Native/model qualification is separate.

The full baseline lock audit reports these ZIP findings plus five unrelated
findings in h2 0.4.15 and rustls 0.23.43: RUSTSEC-2026-0258, -0285, -0346,
-0348 and -0349. Keep these visible; no ignore list, target filter, exclusion
or advisory suppression is introduced. Post-repair audits: ZIP findings are gone; the five unrelated findings remain
(workspace exit 1). The consumer has zero vulnerabilities (exit 0) and retains
the paste unmaintained warning.

## Historical adoption envelope from Edge's previous pin

At the original ZIP review, Edge held Candle
`a0c70e1d632e1f5989ab05ef9294d00bb259ccf2`. Compared with that pin, the
pre-repair base `b70f5f32` has no changes to root Cargo.toml/Cargo.lock, any
of the four consumer crate manifests, or candle-core/candle-nn/candle-vlm
runtime source.
The intervening runtime changes are confined to GPT-OSS:

- Seven transformer runtime files: cache.rs, cuda.rs, cuda_storage.rs,
  memory.rs, mod.rs, model.rs and session.rs under models/gpt_oss.
- Two CUDA kernel/FFI files: candle-kernels/src/gpt_oss_mxfp4.cu and src/ffi.rs.
- The remaining transformer delta is safetensors_tests.rs, a test file.

Those changes add lossless CUDA table admission/storage, paged transactional
KV/memory accounting and final-token-only prefill output projection, with
associated session budget/timing changes. They are material GPT-OSS runtime
changes, not documentation-only maintenance. Other intervening paths are
capacity examples, verification/observer/release tooling, tests and records;
the complete named-path envelope is retained in the compact summary.
LFM/Q8/Nomic model implementations are unchanged by that intervening delta.
Adopting a successor from current main still needs explicit consumer review
of the whole revision; this ZIP proof does not grant GPT-OSS acceptance.

## Published delivery and remaining consumer acceptance

The 12-file repair was published through Candle's guarded helper at
`38d90519f8ed10ddbbe160e5482b0d0089efbaad`. The immutable source/tree,
clean native gate and confirmed remote identity are preserved in
`D:/Compact_Evidence/candle/runs/zip-parent-20261017/publication.json`, including
its hashed helper-receipt snapshot. `artifacts/publication/last-push.json` is
only the mutable latest-publication pointer. Historical release tags/assets
and numerical receipts remain unchanged.

Edge `86e1eb10` has adopted this exact revision and ZIP 9.0.4 in all four
consumer lock graphs. Its nine packet-bound locks rehash unchanged. The fresh
October 22 attempt passed setup, nine Rust audits, GenAce 27+2, Harmony 2,
both Clippy lanes and backend 1,110 tests (three ignored). The nested canonical
verifier then reached its 1,800-second deadline (exit 124) during the next graph
compilation; the aggregate/controller returned exit 1. Both Jobs verify empty
membership and cleanup without forced cleanup; owner processes and disposable
checkout are absent. Original receipts are summarized in
`D:/Compact_Evidence/edgesymbio/runs/cargo-isolation-20261018/summary.json`,
section `native86Execution20261022`.

The closed allowance cannot retry. Edge must first resolve the canonical
deadline bottleneck within its retained contract, then prepare and obtain fresh
exact approval for any successor qualification. Calibration/core proofs,
installed Q8/Nomic/RAG/recovery, aggregate acceptance and package attestation
remain open; no installed acceptance or release is inferred. GPT-OSS remains
separately held. The owner separately authorized scoped publication of these
Candle documentation records through the guarded main helper. This does not
qualify Edge or authorize consumer edits, model/GPU work, tags or releases.

---
AI-edited: 2026-10-22T14:34:21-04:00 | agent=Codex/root | task=edge86-docs-closeout | change=bound reviewed qualification records to authorized guarded publication
