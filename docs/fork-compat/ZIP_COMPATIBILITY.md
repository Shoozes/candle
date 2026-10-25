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

Edge's local checkpoint `3512099d44f5f3406f5a7be19b4f9846ddfce3f8` retains
this revision and ZIP 9.0.4 in all four consumer graphs. Its nine packet-bound
locks rehash unchanged. Clean canonical proof passed 493 Python tests (seven
skips) and all 16 browser cases; CPU package/bundle/health and 16 packet controls
passed. The nested browser owner, fail-fast and bounded trace-retention repair
also has model-free full-suite and real failure/cleanup proof. These results
remain bound to that local checkpoint and do not establish installed acceptance.

Its exact packet
`2a718d73190e6c8043da24b30575929f69c6c4898834a789da8b540325818f4b`
was refused at the first actual admission: three Cargo processes and one rustc
process competed for the host. Memory and desktop checks passed. No reservation,
native dispatch, model request, fetch, install or aggregate attempt occurred.
The admission process exited, no persistent children or outer/aggregate Jobs
were created, normal/installed roots were absent, and the execution hold was
released. No foreign process was stopped. Original packet/admission/outcome
records and restoration maps are retained in
`D:/Compact_Evidence/edgesymbio/runs/cargo-isolation-20261018/native-3512099d-20261023/`;
the separate preparation receipt is in `browser-review-cleanup-20261023/`.
Earlier failures remain closed; the missing original e9 browser traces are not
recreated by the later retention proof.

That closed approval cannot be resampled or retried. The later clean Edge
checkpoint `10e970ce50adbb64f9919c6dae1e1d69687ebd9e` ran exact packet
`2df06d3eca1507e91a2101bc346666f2fa78a4750a34f3a841f1aa4b603820fe`
once. Setup, nine Rust audits and 27 GenAce backend conformance tests passed;
the nested GenAce v2 API build then failed writing Windows `full.rmeta` in the
actual C: temporary checkout (error 112; Cargo 101, handshake/controller 1).
Its allowance is closed. The light sweep ran Nomic work, with zero
positive-token-cap text requests; total model calls were not measured.
All six Jobs were empty, three `vctip` helpers were producer-terminated, the
owned temporary checkout was removed and all four quiet holds were released.
The source-bound outcome and restoration map are retained in
`native-10e970ce-20261023/summary.json` under that same Edge run collection.

Edge now owns bounded model-free scratch-placement/capacity proof, preserving
same-volume shared-cache identity, pins, CPU defaults and existing limits.
Renewed exact-source canonical/CPU/package evidence and a fresh exact packet
with new approval must precede another native attempt. Calibration/core proofs,
installed Q8/Nomic/RAG/recovery, aggregate acceptance and package attestation
remain open; no installed acceptance or release is inferred. GPT-OSS remains
separately held. Candle's earlier documentation publication is bound to
`efe80886332ac5e5893370490cf46047173a8c75` in
`D:/Compact_Evidence/candle/runs/edge86-docs-closeout-20261022/summary.json`.
The owner separately selected closeout and guarded publication of these Candle
documentation records. Its candidate, native gate and confirmed remote identity
are recorded after helper success in
`D:/Compact_Evidence/candle/runs/edge-handoff-closeout-20261023/summary.json`.
Consumer acceptance and later model/GPU, tag or release work retain their
separate approval boundaries.

## GenEye ZIP-only backport and consumer qualification

Published `81e1aa282922110569176464f86f54640b8daed5` has sole parent
`3fa7075e536e66b024d14cee3f6f423d81d67162`. Its five-file scope is Cargo.toml,
Cargo.lock, the NPZ/PyTorch readers and archive tests. This carries the same
ZIP 9 compatibility repair without importing intervening overlay work into
GenEye's frozen lineage. Published main integration
`f1b002c3733da283c57aebddd317346f5751330b` preserves the preceding main file tree.
The exact source/parent/tree, guarded gate and remote confirmation are retained in
`D:/Compact_Evidence/candle/runs/geneye-zip-backport-20261024-27d9b8c1/publication.json`.

GenEye's clean `3200310f` source resolves all six Candle packages to `81e1aa28`.
Its exact sealed CPU/CUDA packages passed the separately admitted `caa0c311`
finite private d1 proof. [D1_LOADING](../lfm2-vl/D1_LOADING.md#geneye-consumer-qualification)
owns the numerical/work/cleanup scope, closed allowance and remaining release
gates. Ordinary CUDA dense VLM and LTS activation are separate from private q8 d1.

At review intake, published Candle main's lock matched the backport's lock byte for byte. The retained
audit at DB `21d909e4` closes ZIP findings but retains eight findings in h2,
rustls and certificate-root dependencies. Consumer lock audits do not establish
a clean producer workspace audit. The independent local batch is now recorded
in [completed security work](../lfm2-vl/HISTORY.md#2026-10-25---compatible-root-lock-security-closure);
the older audit counts above retain their original database scope. The
compatible repair and full-lock audit are recorded in
[dependency security](DEPENDENCY_SECURITY.md); the published ZIP-only source
and its original audit remain unchanged.

---
AI-edited: 2026-10-25 | agent=Codex/root | task=root-lock-security | change=preserve the published ZIP cohort and route current root-lock security work
