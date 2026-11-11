# LFM2-MoE completed work

## 2026-11-11 - CPU prefill isolation

The frozen arithmetic logs place the first observed difference at the 37-token
prefill output. Sampled logit delta reaches `0.70701456` before forced decode.
Both existing synthetic prefill tests passed using the prebuilt executable.
A controlled Q8_K half-tie witness shows different integer rounding on identical
inputs and a resulting Q4_K dot difference. Four common fixture activation rows
produce equal quants. The source arithmetic witness does not identify the cause
of the production token mismatch. [STATUS](STATUS.md) links the compact receipt
and the next paired checkpoint trace.

## 2026-11-11 - Q4 comparator log repair

Packet `00ab6c5e` passed admission and stopped on its first CPU comparison after
three successful child exits. The reference's tokenizer diagnostic ended with
an incomplete UTF-8 character. The comparator now filters byte lines before
decoding JSON as strict UTF-8. Thirteen protocol tests passed. Offline replay
of the retained arithmetic logs reached the numerical gate and reported the
first differing top-1 token at zero-based position 2. The attempted run and
offline repair receipts are linked from [STATUS](STATUS.md).

## 2026-11-10 - Supplier source-progress closeout

The owner's paired closeout/publish selects the reviewed 49-path Candle scope:
native/GGUF model support, shared private mixers, example, deterministic fixtures,
qualification tools and independent ownership/context records. Earlier owner
cleanup notes remain preserved in the root STATUS/TODO. Edge's checkout, its
owner edits, normal dependency pin and locks remain outside this publication.
Fresh focused closeout proof passes nine CPU MoE tests, eleven protocol tests,
strict library/example Clippy, formatting, overlay ownership and Summary Bank.

The exact candidate's native gate and confirmed remote result are retained in
`D:/Compact_Evidence/candle/runs/lfm2-moe-publish-20261110-5f094821/summary.json`.
The guarded helper alone confirms delivery. Commit-bound inert preparation in
that collection reuses the verified binary through unchanged Rust build-input
hashes and retains every earlier packet/binary. Saving source progress does not
qualify production Q4 execution, close Edge CLI/API acceptance or adopt a new
consumer dependency. [TODO](TODO.md) preserves those remaining gates.

## 2026-11-10 - Model-free qualification artifact refresh

The reviewed comparator/EOS cleanup now has a fresh CUDA-enabled qualification
binary in the separately owned D: build directory recorded in [STATUS](STATUS.md).
The locked/offline release build passed in 513 seconds. All 788 build inputs
matched before and after compilation and at packet preparation. Its bounded
Windows Job peaked at 3,390,283,776 bytes and released its children. An LNK4098
library-conflict warning is retained; production CPU/CUDA behavior is unproven.

Fresh proof passed 11 protocol tests, the selected locked/offline crate/example
check, header-only inspection, all four pinned tokenizer cases and the unchanged
reference's no-model version check. Header inspection reports 428 admitted tensors
and the retained 16,035,996,672-byte estimate at context 12,288/chunk 128; this is
not an observed inference peak. The existing 14,526,258,464-byte model was hashed
in place and matches the frozen SHA-256. No model payload was loaded.

The new inert packet binds the finalized source inventory, fresh binary, retained
reference, two frozen prompts/eight positions each and four explicit frozen-case
comparisons. Eight negative binding controls reject modified bounds, identities,
commands or execution permission. The old `1e4449ab` packet and both old binary
paths remain byte exact. No consumer pin, model, lock or release identity changed.
Production dispatches, commits, pushes, package assembly, installation and
activation are zero for this batch.

Compact proof and exact commands:
`D:/Compact_Evidence/candle/runs/lfm2-moe-artifact-refresh-20261110-5f094821/summary.json`.
The shared catalog protects the D: build output while this packet depends on it;
review its retention only after the packet is closed or replaced. [TODO](TODO.md)
owns packet review, separately granted finite production qualification, Edge
acceptance and separately authorized publication/adoption. D1 stays closed.

---
AI-edited: 2026-11-11 | agent=Codex/root | task=lfm2-moe-cpu-isolation | change=record prefill boundary and quantization witness
