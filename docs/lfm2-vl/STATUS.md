# Candle Fork Current Status

## Current phase and baseline

The accepted November 10 [LFM2-MoE implementation](../lfm2-moe/STATUS.md) is the
active capability batch. It is independent from completed D1 and earlier audio
maintenance. Its own [TODO](../lfm2-moe/TODO.md) owns native/GGUF, CPU/CUDA and
opt-in Edge acceptance. Candle source progress was published at `95df99a0`.
Packet `00ab6c5e` consumed its attempt and stopped at the first CPU comparison.
After the comparator repair, retained logs show sampled logits already differ
at prefill and top-1 differs at zero-based position 2. The paired CPU trace
found exact embeddings, first drift at layer-0 RMSNorm and five Q8_K half-tie
byte differences on common normalized inputs. Offline replay isolated their
layer-0 effects. The isolated CPU diagnostic validated both controls and all
three arithmetic variants in two bounded attempts. Only the RMSNorm-only
variant restored position 2. With both switches the first projection nearly
matched while later sampled prefill drift grew. The complete layer-0 CPU pair
measured its largest increase at the dense FFN down projection. The matched
replay attributed that increase to incoming SwiGLU drift crossing a Q8_K
rounding threshold. The SwiGLU replay isolated the retained up operand as the
factor that changes the rounded byte. The matched FFN up replay measured
both dot-kernel and input-scale effects. Matched-residual FFN RMSNorm tied
the seven row-16 scale changes to the ShortConv mixer output. The matched
output replay reproduced both mixer tensors and isolated the peak row-16
difference to the projection input. Dot differences also affected downstream
scales. The crossed replay measured a larger channel-557 shift from the
convolution on either C background. The shared-input convolution replay
reproduced both controls and traced the larger row-16 channel-306 shift to
the incoming `B × X` product. The completed input-projection replay matched
all Q8_K operands and attributed that product drift to the dot kernels.
The completed dot trace reproduced the six B/X values and localized the
first differences to the F32 schedules in block 0. The arithmetic replay
matched the reference witnesses with combined integer scale/min folding and
separate F32 accumulation. The isolated candidate now matches all 36 GEMM
rows and the final GEMV row bit for bit. The full mixer replay matched
`B × X` and found the first remaining difference in convolution. The tap
replay reproduced both F32 accumulation orders and the row-16 gate. The
isolated output projection matched all 37 rows on the reference gate. The
composed replay matched all 37 rows through FFN RMSNorm and recovered all
seven row-16 Q8_K scales. The FFN up trace localized the remaining witness
differences to F32 accumulation. The full FFN up projection matched all
435,712 reference values with the corrected schedule. The SwiGLU replay
recovered the original down-projection rounding witness. Reference SiLU
recovered all 470 remaining Q8_K scales. The offline FFN down trace
matched both native values at row 16, channel `1649`, and found the first
F32 contribution difference in block 0. The isolated Q6_K replay
matched all 75,776 reference projection outputs, including the final
row, and reproduced the stock Candle control. The offline final residual
addition matched all 37 rows of the reference layer-0 output. After
fixing four Q4_K packing widths, one isolated Candle run from the
frozen IDs matched all checked layer-0 stages and the full final output.
A fresh full-model CPU prefill reproduced the verified layer-0 output
and matched llama.cpp's final-position top token `207`. The nine
sampled logits had a maximum absolute delta of `1.313808`. The review
rechecked the retained logits. Packet `8060130a` at published Candle
`72a53981` consumed its native attempt after matching the layer-0
entry and finding the first mismatch in layer-1 weighted RMSNorm.
One composed offline replay at `60ce38d4` matched all ten layer-1
checkpoints, including the full `l_out-1`. The MoE overlay owns the next
inert layer-2 diagnostic. Its square ShortConv repair and CPU/RTX 4090
fixture coverage were published at `60ce38d4`.
The production qualification inventory now binds the example
package's build script and its local inputs.
[MoE status](../lfm2-moe/STATUS.md).
The local qualification comparator requires both engines to
report the frozen continuation. The next packet needs a reference binary from
the updated probe source.
Builds, further production model execution, deployment and consumer adoption
remain held.

The preceding upstream-sync publication is `5f094821fa7612bdeea156f729ba6de35cda3e5d`,
including local merge `311d267b` of Candle 0.11.1 at
`cfcbd7186d13271843ce7eb41d25659bd0835850`, from clean fork `7b9b9a94`.
It imports the five official commits, preserves ug/onig/CP437 and existing
overlay behavior, and limits third-party lock version changes to ZIP 9.1.3
and its required typed-path 0.13.0. [UPSTREAM_SYNC](UPSTREAM_SYNC.md) owns the
current source/verification contract. Exact local merge identity, commands and
final gate results are retained in
`D:/Compact_Evidence/candle/runs/upstream-sync-20261103-cfcbd718/summary.json`.
Local merge `311d267b` preserves both the fork and upstream histories. It passed
the maintained native gate (609 Rust tests, 14 ignored), 197 release assertions
on each PowerShell host, 65 targeted CPU tests, strict CPU library Clippy and
all overlay/manifest checks. These are retained local-integration results.

The guarded publication delivered this upstream scope and four completion
records. Its exact delivery commit/tree, fresh clean-commit gate and confirmed
remote result belong to
`D:/Compact_Evidence/candle/runs/upstream-sync-publish-20261103-311d267b/publication.json`.
Only that receipt establishes publication. Model runs, package builds,
installation and activation remain held. Metal runtime/parity is unverified
on Windows; frozen release inputs and consumer qualification/pins are unchanged.

The November 8 owned-cleanup pass removed the exact previously inventoried
`target/debug/incremental` generation (13,400 files) and ten redundant local
audio logs. It removed 10,546,274,008 logical bytes; C: free space increased by
9,394,388,992 bytes during deletion, with concurrent activity limiting attribution.
Existing Cargo/cache locks were acquired and released, and all 77 protected
artifact hashes matched before and after. The logs remain byte-exact members of
the historical audio `software-proof.zip`; the cleanup manifest maps each removed
path to its retained member. The result and final checks belong to
`D:/Compact_Evidence/candle/runs/owned-cleanup-20261108-5f094821/summary.json`.
Unique diagnostics, compiled outputs, dependency caches, reference inputs, model
assets and retained publication/runtime proof are preserved. The cataloged
`D:/CodexStorage/build/candle-zip-parent-20261017` build exception was already
absent at inspection; that pass removed no external files. Its original STATUS
and TODO edits remain preserved within the current uncommitted capability work.

The independent native [LFM3-Audio source](../lfm3-audio/STATUS.md) and subsequent
WAV repair are published through `a213f04f1b404c5b404172c84223ca1776136ec2`.
The review/recheck kept the WAV finding closed and corrected stale
publication/consumer handoff records. That five-file documentation delivery is
recorded separately in the audio STATUS. Its own TODO/source map owns audio
acceptance. D1 remains maintenance-complete and consumer pins stay frozen.

SymbioVoice's portable receipt/catalog handoff is now verified closed. No
Candle audio implementation or catalog repair remains; the next bounded voice
coverage/source review and model-free preflight plan belong to the consumer
queues. Audio STATUS/HISTORY bind the closure without new native qualification.

D1 pixel-cap hardening is published at
`27f24e9fc9e9d46fd25549655c3a6617eea75e4a`, from baseline
`443f67c57a37178dd1bf39d7fc06eebaa1a2cc57`. The D1 component is
maintenance-complete for the current tested integration. Consumer packages and
release inputs remain frozen. The exact commit/tree, clean-commit gate and
confirmed remote result belong to
`D:/Compact_Evidence/candle/runs/d1-pixel-cap-20261025-d64863b3/publication.json`.

The October 26 Closeout/Publish saved the reconciled GenEye acceptance and held
adviser handoff in Candle's STATUS/TODO/HISTORY. That review confirmed the
published D1 code and admitted verdict were unchanged. Its cleanup separated
the one-attempt finish line from later deployment/usefulness gates and preserved
unproven runtime criteria after failure. No D1 implementation was reopened.
The exact delivery commit/tree, clean-commit native gate and confirmed remote
result belong to the existing D1 run's `handoff-publication-20261026.json`.

Published `443f67c5` closed compatible root-lock security maintenance from
baseline `f1b002c3733da283c57aebddd317346f5751330b`. The exact clean
commit/tree and guarded publication result belong to
`D:/Compact_Evidence/candle/runs/root-lock-security-20261025-c24a6353/publication.json`.
The reviewed ZIP-only backport `81e1aa282922110569176464f86f54640b8daed5`
has sole parent `3fa7075e536e66b024d14cee3f6f423d81d67162`; its non-force
integration preserves the preceding main file tree. Exact source, integration,
native gate and remote confirmation are retained in
`D:/Compact_Evidence/candle/runs/geneye-zip-backport-20261024-27d9b8c1/publication.json`.
GenEye's corrected finite d1 package qualification is complete at that pin.
Edge's independently adopted ZIP baseline remains
`38d90519f8ed10ddbbe160e5482b0d0089efbaad`; its storage preparation and
installed/aggregate acceptance remain under Edge ownership.
Live upstream is `cfcbd7186d13271843ce7eb41d25659bd0835850`;
released 0.2.0 tag/assets remain at `9a7c65c2ebca67e7997ae1a73b652e38d0a3781f`.

## Last green verification and proven behavior

The published clean commit `27f24e9f` passed the guarded native gate: 592 Rust
tests, 14 ignored, all eight image-cap tests and 197 release assertions on each
PowerShell host. The complete guarded helper finished in 83,009 ms. Its exact
commands, runtime and remote confirmation are in the publication receipt above.
The October 26 receipt-reconciliation pass reused that unchanged Rust/bank proof
without new native or model qualification. Current guarded publication requires
a fresh native software gate against its clean commit; model qualification stays
outside this delivery.

The October 25 D1 native gate passed on Rust 1.97.1/MSVC in 122,672 ms:
592 Rust tests passed, 14 ignored, and 197 release assertions passed on each
PowerShell host. Formatting, library/example checks, strict Clippy, model-free
controls, routing, module layout, overlay ownership and whitespace passed.
Eight focused image-cap tests pass; four behavioral checks failed before the
repair. Seven independently generated Pillow 11.3 RGB cases match, including
normal landscape/portrait output, and all five original fixture hashes remain
unchanged. The checked pure helper covers 567 dimension/cap combinations without
allocating extreme images. The bicubic kernel itself is unchanged.
Exact commands, code inputs, final record checks and source identities belong to
`D:/Compact_Evidence/candle/runs/d1-pixel-cap-20261025-d64863b3/summary.json`.

The subsequent read-only Review/Recheck matched all eight candidate files to
that receipt, inspected the session caller and neighboring processor tests, and
found no new D1 defect. Native test results above are reused at unchanged source;
they were not rerun for documentation cleanup. Only STATUS, HISTORY and TODO
change in this follow-up. Its exact scope and final checks are retained beside
the original summary in `review-recheck-cleanup.json`.

The source-bound October 24 Candle publication gate at `f1b002c3` passed 585 Rust tests
(14 ignored) and 197 release assertions on each PowerShell host, with native
formatting, locked/offline checks, Clippy and documentation gates. The October
17 ZIP repair separately retains its 14 focused archive tests and archive
Clippy proof; those tests are included in its native total.
ZIP 9.0.4 retains disabled default features and decoded CP437 NPZ/PyTorch keys.
The independent local consumer built all four crates and loaded tiny NPZ/PTH
fixtures without an override. These are retained source-bound results;
this documentation cleanup does not relabel them as new runtime proof.
The earlier Edge documentation delivery remains bound to
`D:/Compact_Evidence/candle/runs/edge-handoff-closeout-20261023/summary.json`.

The local root-lock security candidate passed the unmodified native gate on
October 25 with Rust 1.97.1/MSVC: 585 Rust tests passed, 14 ignored, and 197
release assertions passed on each PowerShell host. Four-library and example
checks, strict Clippy, model-free controls, formatting, layout/routing and
overlay ownership passed. Candidate lock SHA-256 is
`b7c0c561aaf0b193909317f41cb88910c04f7257d8fea4a71fa67708c836e735`.
That retained working-tree proof preceded the successful clean-commit guarded
publication at `443f67c5`; its exact receipt remains in the security run above.

The first guarded attempt at `d72b045b` stopped before pushing when the
model-free watcher test read its empty reservation as an armed record. The
test now follows the existing production readiness handshake, retaining the
five-second arm bound and original worker limits. All 17 focused native
controls pass, including empty/partial JSON refusal. Runtime code is unchanged;
the repaired publication candidate subsequently passed its full guarded gate.

GenEye's exact clean package source `3200310fba0d0c81ed9fd5ab8acff0e78c0dffa1`
and Candle `81e1aa28` passed reviewed packet `caa0c311` once on October 25:
CPU t01/i01 and cleanup preceded CUDA t01/i01. Actual work was two loads,
four requests, twelve language forwards, two vision forwards and zero output
tokens/retries. Independent same-device replay had zero answer delta; order,
Q8Rows storage, native dispatch and release of all five owned processes passed.
The allowance is consumed and closed. The
[consumer contract](D1_LOADING.md#geneye-consumer-qualification) owns exact
receipts, the unchanged eight criteria and retained image-case limitation.
Full LTS, ordinary VLM and caller usefulness remain separately owned.

GenEye's later canonical supplemental collection closes CPU LTS at the same
`3200310f`/Candle `81e1aa28`: independent index and package verification both
report 20/20. Its `controller-20261025/supplemental-closure.json` remains under
the frozen cohort's `lts-preparation/feedback-adviser-20261025/` evidence root.
GenEye has now admitted the original inline acceptance for CPU adviser packet
`78d63154` as an exact 808-byte copy, with original chat/message provenance.
The review step is closed. The review path alone exists; the other twelve future
paths, including grant, consumption and attempt outputs, remain absent. The next
step is a separately authorized finite CPU adviser attempt. Caller usefulness,
plugin 110 deployment and ordinary dense CUDA VLM/CUDA LTS remain separately
owned. [The admission record](HISTORY.md#2026-10-26---geneye-original-review-admission-and-owned-handoff)
preserves the original verdict; no installation/activation is repeated.

Edge's local model-free checkpoint
`3512099d44f5f3406f5a7be19b4f9846ddfce3f8` retains Candle `38d90519` and
ZIP 9.0.4 in all four consumer graphs. All nine packet-bound locks rehash
unchanged. Its clean canonical gate passed 34 checks plus final completion,
including 493 Python tests (seven skips) and all 16 browser cases. CPU package,
bundle/health checks and 16 packet controls passed with empty Jobs and no
forced cleanup. These are preparation results, not installed acceptance.

The model-free repair owns Vite inside the Playwright worker, stops at the first
unexpected browser failure and retains bounded result/trace artifacts before
checkout cleanup. A full nested 16-case run and a real deliberate failure prove
that path. It preserves the existing process, memory, startup and browser limits.

## Qualification failure, cleanup and limitations

Earlier Edge packet
`2a718d73190e6c8043da24b30575929f69c6c4898834a789da8b540325818f4b`
was refused at its first actual admission on October 23: three Cargo processes
and one rustc process were observed. Memory and interactive-desktop checks passed;
the failure was competing compilation. Their repository/worker owners were not
established by that census. No allowance reservation, native dispatch, model
request, fetch, installation or aggregate attempt occurred. The approval is closed
with no resample, retry or reusable remainder.

The admission process exited; no persistent owned processes or outer/aggregate
Jobs were created. At that closeout, normal-setup and installed roots were absent,
no foreign process was stopped, and Edge released the execution quiet hold. Original packet,
admission/outcome bytes and their restoration map remain under
`D:/Compact_Evidence/edgesymbio/runs/cargo-isolation-20261018/native-3512099d-20261023/`.
The browser repair and package proof remain separately bound in
`browser-review-cleanup-20261023/summary.json` under that same run collection.

The later clean Edge checkpoint
`10e970ce50adbb64f9919c6dae1e1d69687ebd9e`, tree
`03416a27865988e4a83aaa407a95a9def9f11b58`, ran exact packet
`2df06d3eca1507e91a2101bc346666f2fa78a4750a34f3a841f1aa4b603820fe`
once. Normal setup passed, but its nested canonical GenAce v2 API build could
not write the Windows crate's `full.rmeta` under the actual C: temporary checkout
(Windows error 112; Cargo exit 101, handshake/controller exit 1). Nine Rust
audits and 27 GenAce backend conformance tests passed before that failure; API
conformance, calibration, core/installed qualification and final aggregate
acceptance were not reached. The allowance is closed with no reusable remainder.
This establishes storage exhaustion, not a GenAce semantic or quota regression.

The setup light sweep executed Nomic work: 16 proofs reported 11 real outputs,
one missing asset and four gated results. There were zero positive-token-cap
text requests; total model calls were not independently measured. No installed
attempt or push occurred. All six Jobs were empty without forced Job cleanup;
three `vctip` helpers were explicitly terminated by their producer. The owned
temporary checkout was removed, all four quiet holds were released, and no
foreign process was stopped. Original failures, cleanup and restoration records
remain under `native-10e970ce-20261023/summary.json` in the same Edge collection.

Earlier `86e1eb10` deadline, `594ad6f9` Cargo-source and `e9d1003f` browser
failures remain historical and closed. Original e9 browser traces were lost;
the retained raw log/journal and later retention repair do not recreate them.

The historical ZIP-backport audit at RustSec DB `21d909e4` retains eight
h2/rustls/root-certificate findings. The published compatible root-lock repair
closes all eight plus Tokio's existing Unix I/O-safety warning at the same
database, with zero vulnerabilities/unsoundness warnings and five unchanged
unmaintained notices. The [security contract](../fork-compat/DEPENDENCY_SECURITY.md)
owns exact patches, caller scope and passing native regression proof. The
selected Done When is complete; GenEye's locks and frozen `81e1aa28` are unchanged.
The older October 17 five-finding
audit and missing-WSL-disk observation retain their historical scope; this
review does not claim a fresh advisory database or Candle WSL replay.
Q8/Nomic defaults, disabled GPT profiles and immutable releases are preserved.
The preceding security and image-cap publications and their reviews are
complete. The October 26 handoff publication covered only three reviewed records
on Candle `origin/main`; it grants no further publication authority. GenEye's CPU adviser
attempt, package assembly, further model runs, installation, activation and
plugin changes remain held.

## Active files and exact next task

The active implementation is the separate [LFM2-MoE capability](../lfm2-moe/STATUS.md).
Its model modules, shared crate-private mixers, tiny fixtures, local example and
owning records are under active work. Its STATUS/TODO bind current proof and the
exact next gate. The prior owned-residue cleanup remains complete. Use cached
dependencies, `CARGO_INCREMENTAL=0`, `CARGO_BUILD_JOBS=2` and locked/offline checks.
The current Candle source publication is bound to the MoE STATUS and its guarded
receipt. Production runs, package assembly, installation, activation and final
Edge pin adoption remain held; consumer-owned planning below grants no execution.
The finite LFM3-Audio implementation
and portable catalog handoff are
closed, as recorded in [its STATUS](../lfm3-audio/STATUS.md) and
[TODO](../lfm3-audio/TODO.md). No further audio implementation is selected; the
remaining voice gap/source review and preflight planning stay consumer-owned.
The eight-path image-cap batch is already published; its implementation and
fixtures remain unchanged. Existing security pins and consumer locks are
preserved; audio adds only a direct edge to the already-pinned RustFFT dependency.
Dimensions are positive, source-bounded and checked against the requested cap;
every valid legacy size and the original resampler are preserved. Oversized
results alone shorten their longer axis. D-0074 keeps the historical invalid
`9x1`, cap-six golden separate from corrected cap-six output. A focused image-cap
route keeps the existing 256 KiB context ceiling. Proof is retained under
`D:/Compact_Evidence/candle/runs/d1-pixel-cap-20261025-d64863b3/`.

The root-lock security batch remains complete, with its original source-bound
claims. No package build, consumer repin or new model qualification is selected
by this review/recheck/cleanup pass.
The image-cap item is closed; no further D1 implementation batch is selected.
Full-resolution RGB conversion and the horizontal bicubic intermediate
remain unchanged. The cap limits returned pixels, not peak temporary memory;
no allocation optimization or measured memory-regression claim is included.
The completed image-cap delivery and October 26 handoff used `.tools/gitpush.ps1`
and confirmed their exact remote commits. The audio source was also delivered
through that guarded helper at `6a46737c`; its separate
[STATUS](../lfm3-audio/STATUS.md) binds that receipt and retained qualification
limits. The seven-file parser repair was published at `a213f04f`; its receipt
does not renew model or consumer qualification. Current TODO reflects the
implemented SymbioVoice adapter, closed portable traceability and separate
consumer-owned coverage planning.
GenEye owns its receipt admission and current-state/TODO updates.
The original review, native and publication receipts remain unchanged. A live
CPU adviser attempt still needs an exact grant and fresh admission under its
existing limits.
The immediate handoff ends with one terminal attempt and its evidence. Any
failed runtime criterion remains open; later plugin/usefulness acceptance stays
separate and supplies no retry authority.
The supplemental datasets unit/doctests passed
but invoke the Hub; their network/cache activity was not measured and is not
claimed as deterministic offline proof. Preserve the published `81e1aa28` and
consumer package cohorts; a local main dependency repair does not repin consumers.

Edge owns `R-ES-CORE-RELEASE-PREP-001`: prove actual temporary/model/Cargo
placement and sufficient capacity, preserving same-volume shared-cache identity,
pins, CPU defaults and existing limits. Use bounded model-free proof, then renew
exact-source canonical/CPU/package evidence and prepare a fresh packet. A native
successor requires new exact approval; earlier allowances cannot be retried.
Preserve local owner edits and the held documentation patch; existing proof
remains bound to its original clean checkpoint. Calibration/core proofs,
installed-core/RAG/recovery, aggregate acceptance, package attestation and
release remain open. Candle's
published ZIP repair and adopted consumer graphs do not need another repin.

---
AI-edited: 2026-11-13 | agent=Codex/root | task=lfm2-moe-layer1-compose | change=record composed layer-1 proof and route layer 2
