# LFM2.5-VL Current Status

## Current phase and source

Current phase (2026-10-09): Complete with verified limitations. Candle's reusable
d1 implementation is published at `3300bef20be161e63816b8820c845264acb1942d`
(tree `e45304a19f74d5e094f9adaed19e4b485cfb872f`). GenEye's consumer integration
is published at `48bf4de746ebbfe4226cb8bb79fd37e3c8e96e4e` with that exact Candle
pin; its producer API P2 finding is closed. GenEye's LTS release approval was
not issued. Candle implementation files under active work: none. This local
recheck updates only `D1.md`, `START_HERE.md`, `TODO.md`, `HISTORY.md`, and this
status handoff.

Candle's guarded publication receipt and the independent 65-file source binding
remain in `artifacts/lfm2-d1/20261008-closeout-publish-p1/`. The receipt records
the clean-head native gate (500 Rust tests, eight explicit ignores, six Python
receipt controls), d1-focused tests (14 passed, one explicit ignore), strict
CUDA d1 Clippy, fixture byte-identity guard and Summary Bank verification. These
are retained checks from the exact published source, not rerun during this
documentation pass. The 120 retained typed answers replay with no new production
forwards; numerical proof remains bound to the original implementation snapshot.
This documentation pass ran the Summary Bank verifier and `git diff --check`;
it did not run Cargo tests or model inference.

GenEye's published consumer acceptance used its separate closed 44-request
allowance with zero retries, 132 language forwards, ten vision forwards and zero
generated tokens. CPU and CUDA each returned all 20 corrected study results;
the observed counts remain 13/20 choices, 14/20 booleans and 10/20 rubric
ranges, with no invented accuracy threshold. The initial CPU batch ended with a
control-channel stop failure after results completed; separate graceful-stop
and parent-exit cleanup checks passed. The existing GenEye 0.5.0 LTS remains
unchanged; a future LTS needs a selected clean published candidate, existing
vision-profile qualification and fresh exact-package release evidence. See
[D1.md](D1.md#consumer-handoff) and GenEye's retained evidence root for details.

The previous Candle closeout repaired CRLF in three tiny d1 fixture JSON files,
corrected their manifest/tokenizer hashes and made the Windows grep byte guard
literal. Its native fixture-identity regression reproduced the defect, then all
14 focused d1 tests passed (one explicit ignore). Golden data and model/kernel
math remain unchanged. Known limitations are the optional `Codex-Compat` WSL
cache missing `accelerate-src` and the existing CUDA MSVC LNK4098 warning. The
first remaining Candle task is the conditional real-adapter qualification in
`TODO.md`; it still depends on its recorded producer artifacts and Edge gates.

Previous phase (2026-10-08): Closeout of the stock CPU diagnostic progress
snapshot; runtime and real-adapter acceptance remain blocked at an Edge-owned
integration boundary. The native `main` closeout baseline is
`4f2a068e65cd1a5eb7d3bd931c8705d2e460f27e`. The owner authorized publication of
`STATUS.md`, `TODO.md` and `HISTORY.md` through `.tools/gitpush.ps1 -Yes`.
Rust source and public APIs are unchanged. Exact published commit/tree and the
fresh native gate belong to `artifacts/publication/last-push.json`; this
closeout's proof is retained in
`artifacts/adapter-coordination/20261008-publication-p2/`.

The current Work request admitted the documented eight-view CPU/C-output batch.
Fresh 330-pin/input/resource checks passed. One API session authenticated, then
capabilities returned `q8_artifact_missing_or_wrong_size` and
`q8_context_unavailable`. No request was created or submitted and no model
inference occurred. Seven views remain unattempted; no retry occurred.

The cached HF snapshot is a SymbolicLink to the verified 2,874,779,648-byte blob.
Edge's `worker_artifact_blocker` uses symlink metadata and regular-file/size
admission, so this cache path cannot qualify. Edge owns the compatible retained
artifact binding or placement decision. Copying a 2.8 GiB model outside the 64 MiB
runtime budget or weakening admission is not an implicit workaround.

The invocation also copies client cleanup evidence with exclusive creation
on both retirement calls. Both Job reports show verified cleanup and empty
active-process sets, but the original session receipt remains
`cleanup_evidence_incomplete`. Independent runtime checks found all recorded
API/Job/child PIDs absent and port 53760 without a listener. No process stop was
needed. Failed session/temporary evidence remains retained.

Candle's import kept all eight slots unbound. The pinned GenUni scorer and exact
report replay pass, with no semantic grades. Its `missing_attempt` means no
bound scoring row, not no owner session. No model competence, accuracy or adapter
quality follows from this refusal. Current evidence, actual counters and remaining
Done When criteria are in
`artifacts/adapter-coordination/20261008-stock-diagnostic-w1/execution-result.json`.
The original approval/run root is consumed and cannot be reused.
GenUni's completed read-only recheck confirmed both Edge defects, the complete
denominator and the observed process/endpoint release.

Prior F32 admission, request clocks and frozen binding fixes remain source-bound
to their retained proof. Closeout reran all 14 authored protocol controls,
verified 17 contract pins and all eight frozen inputs, and passed exact scorer
replay and the mod-manifest check. Fresh proof is in the publication closeout
directory; original first-read proof remains in `20261008-config-bound-c2/`.
No retained trained 2.6B adapter, selected training/held-out plan or quality receipt
was supplied. Cached Q8/native HF bases remain separate from that producer gate.
Completed preparation is
[recorded in HISTORY](HISTORY.md#2026-10-08---genuni-and-edge-diagnostic-preparation).
The selected work record and its runtime/real-adapter criteria remain unchanged.
Referenced code, baseline snapshots and failure evidence are deliberately retained.

Previous local closeout (2026-10-07): native CPU adapter-admission closeout is
complete; real adapter quality and Edge qualification remain externally gated.
The closeout baseline is native Windows `main` at
`b274902312a39cbe1b0028141276f43e64b56096`; the candidate adds the named task changes.
The quantized LFM2 LoRA admission path now rejects a scale that becomes
infinity or zero in F32 before active-adapter/cache mutation. Three new tests
cover overflow, underflow, preserved cached continuation, and representable
F32 boundaries. Five focused LoRA tests, 46 neighboring LFM tests, and the
broader native CPU gate pass. Changed source files are `quantized_lfm2.rs`
and `quantized_lfm2/lora.rs`, with the existing
GPT-OSS documentation cleanup and LFM status/backlog/history/decision records.
No kernel, dependency, checkpoint format, or consumer application is changed.
No implementation work remains active in this local batch. The owner authorized
scoped publication through `.tools/gitpush.ps1 -Yes`; its fresh gate and exact
commit/tree/remote identity are recorded in `artifacts/publication/last-push.json`.

The real adapter/quality/Edge milestone remains externally gated: inspection
of the supplied `D:\huggingface-cache` found the base-model snapshots but no
adapter, training manifest, or held-out evaluation receipt. No production
weights were read or downloaded, and no production/CUDA inference was run.

Previous documentation batch complete (2026-10-07): native Windows `main` at
`b274902312a39cbe1b0028141276f43e64b56096` plus documentation changes.
Review/recheck cover the existing research note, GPT-OSS evidence descriptions,
and the prior LFM cache findings. Cleanup reconciles qualification wording and
archives completed backlog entries. Changed files are the GPT-OSS module's
documentation header, `docs/gpt-oss/{STATUS,SOURCES,MOD_MANIFEST}.md`, and
`docs/lfm2-vl/{STATUS,TODO,HISTORY}.md`. No executable logic is changed.
At that documentation recheck, the LFM source hashes and all nine retained
receipt/log pairs matched the sealed prior gate. That slice ran documentation
and format checks; it did not rerun runtime tests or model/CUDA inference.
The later LoRA change and its new source-bound proof are described above.

Previous local batch complete (2026-10-07): review, scoped repair, and recheck of LFM2
cached forwarding on native Windows `main` at
`ee1139b15b9364676ae208ab3ccd24437666d87a`. The starting checkout was clean.
The reviewed batch comprises
`candle-transformers/src/models/lfm2.rs`,
`candle-transformers/src/models/lfm2/layers.rs`, and
`candle-transformers/src/models/quantized_lfm2.rs`, with status, decision,
and history records. No implementation work remains active in this batch.
Both towers preserve convolution history across multi-token continuations
and restart it at position zero. Quantized position overflow and context
excess fail before mask allocation or cache mutation. The exact component,
official dense fixture, quantized prefix, and rejected-span recovery checks
pass. Public forwarding signatures and LoRA admission remain intact.

The native locked/offline baseline passed 38 LFM2 tests; five new regressions
bring the focused gate to 43 passing tests. Final maintained tests passed,
including 143 transformer tests (five external-model tests ignored), 37 VLM
tests, and all 33 LFM2-VL example tests. Core/nn/transformer/VLM and all three
LFM2 example checks, strict transformer Clippy, formatting, layout, Summary
Bank, LFM manifest, overlay union, and whitespace checks passed. Exact commands
and failure-to-pass evidence are in the dated `HISTORY.md` entry. The sealed
summary is `artifacts/lfm2-cache/20261007-e7e19aff/verification-summary.json`,
SHA-256 `f29bfb6cd565d6d75a015f47c46a946b296ca280bf68a45b0bd2cb4c42f8c4b3`.
All nine owned Cargo process trees exited and their root PIDs were absent.
There is no current code/proof blocker. CUDA, production-model, and WSL replay
were not run. The owner authorized closeout and scoped publication on
2026-10-07. The guarded helper independently records the exact clean commit,
tree, fresh native gate, and remote-main identity in
`artifacts/lfm2-lora-scale/20261007-d56162f4/prior-publication.json`.
This retained copy of the earlier `last-push.json` records successful
publication of `b274902312a39cbe1b0028141276f43e64b56096` with native gate exit
code zero and matching remote tip at publication time.
Referenced task evidence and the existing D: build cache are deliberately
retained; no task-owned service or endpoint remains running.

Upstream stabilization: integrated pinned Hugging Face main
`aebc405d2b4bf42808387e0ca597bf7dad9b565f` while retaining the fork's public
contracts. Starting clean local/remote main:
`8f27ddfbee47957c274341fd6d32ccabd4767f9f`.
The sync inventory and exact verification results are owned by
`UPSTREAM_SYNC.md`.

Canonical checkout is `C:\DevStuff\candle`, native Windows main. Publish only
with `.tools/gitpush.ps1 -Yes`; secrets remain ignored and unread by the agent.
WSL is a secondary verification lane and never owns Git main.

Compatibility baseline remains Candle 0.11.0
`31f35b147389700ed2a178ee66a91c3cc25cc80d`. Frozen 0.2.0 receipt base
`6f74e7c390c717f8fd34f23ce02aceb058173370` and immutable first-MVP tag
`lfm2-vl-mvp-0.1.0` are unchanged. Historical receipt identities are not
rewritten to the sync revision.

## Proven behavior and ownership

- Config-driven LFM2.5 text, embedding prefill, cached decode, and reset.
  Fallible try_into_config is maintained; deprecated into_config retains its
  original compatibility behavior and requires caller validation.
- SigLIP2 NaFlex, bounded image processing, prompt expansion, pixel unshuffle,
  projection, and multi-image feature replacement.
- Native safetensors, split dense MMProj, direct GGUF MMProj and CPU F32
  native Q8_0 MMProj with deterministic fixtures and cache-reset proof.
- Generic SDXL three-component LoRA transactions/rollback, residual validation,
  and opt-in pooled-text/time-ID conditioning.
- Experimental GPT-OSS short parity and 2,080-token performance acceptance
  remain tied to original source `4a4699981ed55bb11e857c58923c67133559145d`
  and `docs/gpt-oss/STATUS.md`. Larger-context diagnostics are not acceptance.
  The sync does not establish new production parity or change the Q8 default.
- Retained ug public APIs/backend wiring and onig tokenizer selection are
  explicit compatibility differences, independently registered in
  `docs/fork-compat/MOD_MANIFEST.md`.
- Complete ownership is enforced by `docs/FORK_OVERLAYS.md` and its verifier;
  inherited upstream files are not claimed as overlay additions.

## Last green verification and active files

The 2026-10-07 LoRA closeout passed these native Windows/MSVC CPU checks,
all with exit code zero:

- `cargo fmt --all -- --check` (no build or inference).
- `cargo test --locked --offline --release -p candle-transformers --lib lfm2_lora`:
  five passed; the two new rejection tests first failed on the prior implementation.
- `cargo test --locked --offline --release -p candle-transformers --lib lfm2`:
  46 passed.
- `cargo check --locked --offline -p candle-core -p candle-nn -p candle-transformers -p candle-vlm`.
- `cargo check --locked --offline -p candle-examples --example lfm2 --example quantized-lfm2 --example lfm2-vl`.
- `cargo clippy --locked --offline -p candle-transformers --lib -- -D warnings`.
- `cargo test --locked --offline -j 2 -p candle-core -p candle-transformers -p candle-vlm`:
  146 transformer library tests passed with five external-model ignores,
  37 VLM tests passed, and 45 core doctests passed with one existing ignore.
- `cargo test --locked --offline -j 2 -p candle-examples --example lfm2-vl`:
  all 33 passed.

The first maintained test run failed with Cargo exit 101 in 44 core doctests
(E0460/E0462 dependency-loading errors) while the 8 GiB job reached
8,760,901,632 bytes. A single failing doctest and the complete gate passed
after setting process-local `RUST_TEST_THREADS=2`; the complete retry peaked
at 1,399,197,696 bytes. No cache artifacts, pins, or core sources were changed.
Exact commands, logs, source hashes, and all eleven receipt/log pairs are
sealed in `artifacts/lfm2-lora-scale/20261007-d56162f4/verification-summary.json`.
Summary SHA-256: `5d94423b97fda9e577bb5d6b99fe6d6ebfc5b93838460e20d63ccc45b932cbcc`.
All launched Cargo root PIDs were absent after owned Job Object teardown.
Retain this evidence and the existing owner-managed D: build cache. Summary
Bank routes and overlay path inventories remain unchanged. Native Summary
Bank, module layout, both GPT/LFM manifests, the 236-path overlay union,
whitespace, and local link/anchor checks pass. The guarded helper repeats
the maintained integrity gate on the final clean commit.
The review/recheck archive and this admission result are in the dated
`HISTORY.md` entries. CUDA, production-model, and WSL proof were not run.

The starting maintained native gate and merged core, nn, transformer, VLM,
test-utils rollback, example, ONNX, strict CPU/CUDA Clippy and bounded GPU
regressions passed. Exact commands, evidence and platform limits are in
`UPSTREAM_SYNC.md`. The guarded helper replays the maintained gate on the
clean delivery commit and records the exact remote identity in ignored
`artifacts/publication/last-push.json`.

For the checkpointed text-adapter slice, the native release
`quantized_lfm2` test filter passed 3/3 and `lfm2_vl` passed 27/27 on
2026-10-01. Locked/offline CPU checks passed for candle-core, candle-nn,
candle-transformers, candle-vlm, and the quantized-LFM2 example; the bounded
report is `artifacts/qwen35/cuda-lfm-lora-final-cpu-check.json`.
`cargo fmt --all -- --check`, summary-bank, LFM manifest, and rolling overlay
verifiers passed. The first compile failed on an implementation borrow error
and was corrected; the initial non-elevated D: build could not open Cargo's
lock under the sandbox, then the approved D: build passed. The first CPU
check script invocation omitted its required filter and started no job.

Checkpointed LFM2-VL text-adapter support: `candle-transformers/src/models/quantized_lfm2.rs`
and `quantized_lfm2/lora.rs`, with owned manifest, decision, status, backlog,
history, summary-bank, and shared overlay records. A bounded synthetic text
LoRA path retains the GGUF QMatMul base, admits canonical linear targets and
hash-bound base identity, and clears convolution/attention caches on successful
switch. The accepted 2.6B Q8_0 base file at Edge was rehashed locally as
`1e22128dfa128bdfb684da167e74e072d0a056baa7d06d9f280291e2839b0fc9`
(2,874,779,648 bytes); this is an input identity check, not a trained-adapter
or full-model LoRA proof. The independent Qwen3.5 compatibility overlay currently
owns `candle-transformers/src/models/qwen35/`, its artifact-specific fixture,
and the shared registry/overlay records; see `docs/qwen35/STATUS.md` for its
exact 4B CPU/CUDA trace and sealed 1,024-token context evidence. Edge's normal
provider gate remains independently owned. The compatibility manifest accounts
for retained APIs, the new cast regression and the strict-Clippy CUDA
launch-count correction.

## 2026-10-01 closeout verification

The native locked/offline `.tools/verify-before-push.ps1` gate passed on the
complete candidate source: maintained core/nn/transformer/VLM and all three
example checks, strict transformer Clippy, maintained tests (138 transformer
tests passed, five external-model tests ignored), 33 LFM2-VL example tests,
formatting, summary-bank, module-layout, overlay union, and Git whitespace.
The focused LFM manifest separately passed 164 paths (17 fork modifications,
147 additions); the rolling overlay gate passed at the pre-overlay baseline.
The log is retained outside Git as
`%TEMP%/codex-candle-closeout-20261001-gate-final.log`.

Local checkpoint preparation preserves the original model/CUDA receipt source
identities. This closeout ran no production model or CUDA proof. Strict Clippy
required only private observer aliases and equivalent iterator spelling in
the independent Qwen overlay. The later guarded publication receipt described
above resolves the source publication boundary; the October 1 verification
remains historical.

## Known limitations and blockers

- WSL Codex-Compat lacks its Cargo registry cache (accelerate-src offline
  resolution failed); no current Linux replay claim. Metal/AArch64 are unrun.
- Retained ug needs process-local CUDA 13.0 API selection with installed CUDA
  13.3 libraries; exact build settings and rejected alternatives are documented
  in UPSTREAM_SYNC. Maintained non-ug CUDA passed with normal detection.
- Native 3B and official 400M Q8 MMProj production acceptance remains Gated
  pending separately authorized external artifacts/oracles and cleanup proof.
  The locked revisions and exact contracts remain in TODO/PARITY.
- Consumer repinning, production downloads, release tagging, GPT-OSS 32k,
  broader model/backend support, and post-release inherited panic/stub cleanup
  remain separate tasks. No consumer mutation follows implicitly from sync.
- A trained, immutable LFM2.5-2.6B text adapter and held-out competence
  evidence still need to be supplied for this gate. Candle's adapter SHA label
  is caller supplied; Edge
  must independently hash the retained serialized adapter bytes and own its
  provenance, session lease, receipt, and recovery checks. CUDA LoRA and
  real-model adapter numerical parity are unrun.

Documentation-only acceleration research (2026-10-07, reviewed at
`b274902312a39cbe1b0028141276f43e64b56096`) is maintained in
[GPT-OSS status](../gpt-oss/STATUS.md#deferred-optional-acceleration-research),
with references in this file and `TODO.md`. It establishes no speedup or new
product qualification. GenAce MVP/LTS and GenUni's current foundation retain
priority; existing runtime receipts and the next adapter gate remain unchanged.

## Exact next task

Edge owns the next bounded repair: provide a consumer-compatible regular
artifact binding with the required GGUF filename while preserving the base
hash/storage policy, and make the invocation's cleanup-copy hook idempotent.
Prove readiness with the exact cached base and repeated retirement/refusal/
cancellation cleanup. Then review a new
pinned invocation and fresh resource/call admission; preserve the consumed first
batch and never reuse its approval or silently retry it.

Only after those gates pass should we collect/import the eight public views and
review actual explanations/citations. These exposed cases do not establish
unseen competence. If training is selected, GenUni must still provide the real
immutable adapter, provenance and held-out evidence required by TODO.

---
AI-edited: 2026-10-09 | agent=Codex/root | model=unknown | effort=unknown | task=geneye-d1-recheck-cleanup | change=recorded published GenEye consumer acceptance and the separate LTS gate
