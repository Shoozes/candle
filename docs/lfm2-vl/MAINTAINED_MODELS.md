# Maintained-model implementation, 2026-10-12

## Guarded source closeout

The owner selected Closeout/Publish for the reviewed 35-file batch on native
main. Fresh public origin/main and local main both equal implementation
baseline `4c2dfaf3` at intake; all 35 source identities match the review
closeout, with no unrelated dirty or staged paths. The publication separates
d1 loading, Qwen recovery/projections, VAE admission and shared records into
scoped forward commits. Its exact final identity and native verification are
owned by `artifacts/publication/last-push.json`, checked against current HEAD.
Existing original/review receipts retain their source identities and original
authorization statements. Current publication proof belongs under
`artifacts/maintained-models/20261012-publication/`.

Model-free readiness rechecks pass seven comparator controls and three pinned
t01 prompt/token/alias plans, with token counts 264/222/293 and zero model loads
or forwards. Only Qwen runtime and comparator hashes in the original proposal
are stale. This publication preserves proven source progress; it does not
close retained-model qualification. Rebuild/rebind the packet after the clean
checkpoint, then obtain fresh approval. Closed allowances stay closed, and
GenEye remains pinned to `3fa7075e`.

## Review/recheck and scoped repair

The 34-file candidate and ten saved logs match closeout receipt
`a4529186775efcdff314e3abc0a6b87fa4f8a0ca3438f3ecdc7d8f608715a809`
at review intake. Main stays `4c2dfaf3`; the full working candidate was reviewed,
not HEAD alone. Prior nonatomic/resolved image sentinel and common-mode prompt/
artifact-reference findings are closed. Independent Qwen/VAE inspection found
no new arithmetic, admission/cache recovery or VAE defect. Two new controls
reproduced a Qwen partial-trace regression and d1 receipt-accounting gap.

Qwen now emits each Sequential projection immediately. Grouped emits its
readable completed prefix if a later kernel fails, preserving the original
error without re-executing kernels. Failure tests cover FFN-up and recurrent-z,
empty/nonempty cache recovery and exact replay on CPU/CUDA. CPU passes nine/
four ignores and bounded CUDA twelve/ten ignores. The unchanged authored
sampling protocol replays at median 0.14620/0.09210 ms (Sequential/Grouped
recurrent) and 0.08135/0.06810 ms (FFN), 97/100 and 93/100 faster pairs;
all compared outputs remain bit exact. These are component results only.

The d1 comparator now derives input totals from all three token traces,
requires integer work/admission/dispatch counts and zero inactive-device
dispatch. Seven authored controls pass, including common incorrect totals,
boolean/float/null/string counters and existing pinned-reference rejection.
No production requests were used. The stale entry-point ordering, VAE route
description and original CPU count below are corrected. Review proof is
`artifacts/maintained-models/20261012-review/`; the two prior runtime sources
are retained once compressed by digest, preserving historical proof recovery.

The repaired source passes the complete native `.tools/verify-before-push.ps1`
gate: formatting, four locked/offline libraries and examples, strict CPU
Clippy, maintained and example tests, Summary Bank, module layout, typed
receipt controls, overlay union and whitespace. Strict affected CUDA Clippy
also passes (`cargo clippy --locked --offline -j 2 -p candle-transformers
--features cuda --lib -- -D warnings`). LFM/diffusion inventories
remain 214/21 paths; the union remains 290. Both retained runtime source
archives recover their exact original bytes. No WSL replay was repeated
because the prior offline `accelerate-src` dependency failure was unchanged;
that lane supplies no proof for this repair.

The original t01 proposal below remains immutable planning evidence but is
source-stale after these repairs. It cannot authorize execution. The next
selected task is a separately authorized clean checkpoint, rebuilt/rebound
packet and fresh bounded approval. GenEye stays pinned to `3fa7075e`.

## Original implementation proof

The accepted three-slice plan started on clean native Candle main at
`4c2dfaf3bd7a8e519c70db026a79c32e271c3a61`. Each focused gate preceded the
next slice. All changes remain local; commit/publication need separate task
authorization. GenEye retains Candle `3fa7075e` and its consumer/package gate.
No model downloads, training, provider activation or consumer edits occurred.
The complete native `.tools/verify-before-push.ps1` gate passed: formatting,
locked/offline four-crate and four-example checks, strict Clippy, maintained
tests, example tests, Summary Bank, layout, receipt controls and overlay union.
The independently affected LFM/diffusion inventory gates also passed.

D1 now has explicit text-only construction with DenseF32 default and explicit
Q8Rows. It consumes only language GGUF/tokenizer, constructs no vision stack,
rejects supplied/literal/resolved image sentinels before forwarding, and reuses
prompt/readout/counters/cache cleanup. CPU fixtures pass 24/two ignores,
CUDA 31/two ignores, CLI eight; both embedding modes match hybrid text
prompts, token IDs, logits, ordered answers and language work bit for bit.
The twelve source-bound Q8Rows load-only Jobs all completed/released with
zero forwards/tokens and no retries. Median total startup improves 14.6%
CPU/15.8% CUDA; peak Job memory falls about 808/817 MiB (CPU/CUDA) and retained CUDA
allocation by 832 MiB. See [individual measurements](D1_LOADING.md).

Qwen validates complete token/context admission before mutation, preserves
state on rejection and clears all recurrent/attention caches after admitted
execution failure or token-boundary cancellation. CPU cancel now shares the
hybrid contract. Sequential loading stays default; Grouped is opt-in and
shares input/output transfers while preserving individual Q8/F32 arithmetic,
observer order and reduction order. CPU tests pass eight/four ignores; bounded
CUDA ten/ten ignores. Every matched authored timing pair has bit-exact
outputs, with median improvements 37.4% recurrent/15.1% FFN. See
[the independent Qwen protocol](../qwen35/PERFORMANCE.md).

The diffusion overlay explicitly adopts the VAE path. Construction validates
nonempty positive channel layouts, normalization groups and checked latent/
decoder arithmetic. Gaussian parameters need a channel axis of positive even
width; splitting returns controlled errors. Three CPU tests cover malformed
inputs, tiny valid construction/encode/decode and unchanged mean/std/sampling.
Frozen combined-release receipts retain their original identities.

Owned proof is `artifacts/maintained-models/20261012/`: source/binary-bound
CUDA Jobs, frozen load/projection protocols, raw measurements once, process
release and the native gate log. Five authored comparator controls reject
wrong counts, nonfinite/different logits, changed/common-mode prompts, aliases
and artifact identities. The retained t01 packet performs no forwards during
preparation and needs a separately authorized clean checkpoint and fresh
approval before its eight-request/24-language-forward matrix can execute.
The pinned-reference comparator checks official prompt/alias/token inputs and
artifact admission; it does not claim an independent full-model logit oracle.
Qwen retained-model qualification has its own fresh-allowance requirement.

The pending packet is `d1-parity-packet/proposal.json` under that proof root:
16,137 bytes, SHA-256
`8b091f5f5102faafa0d63707e539d8cd678b0de0769b6b878c58d166778f747b`.
Preparation rehashes the five retained artifacts, binds the final release
binary and launcher, and verifies three official t01 prompts/IDs/alias groups
without executing a model. After the full native gate, supplemental CPU/CUDA
controls also pass exact context admission and cancellation after one question;
the current runtime sources are unchanged by that test-only addition.

WSL Codex-Compat has Rust 1.97.1, but the relevant offline replay is unavailable:
`cargo test --locked --offline -j 2 -p candle-transformers --lib qwen35`
fails resolving cached `accelerate-src` before compilation. No dependency
fetch or hosted substitute was used. Native Windows/MSVC is the primary lane.

---
AI-edited: 2026-10-12 | agent=Codex/root | model=unknown | effort=unknown | task=maintained-models-publish | change=closed out verified scoped progress while retaining numerical qualification gates
