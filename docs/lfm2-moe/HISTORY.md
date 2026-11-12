# LFM2-MoE completed work

## 2026-11-12 - Matched-input ShortConv output projection replay

Both engines reproduced their retained mixer outputs. Q8_K matched on common
inputs. At row 16, channel 557, switching the input accounted for the entire
mixer difference. Matched dots differed elsewhere, and the verified offline
FFN propagation showed both effects in the downstream scale witnesses.
[STATUS](STATUS.md) links the native and offline results.

## 2026-11-12 - Frozen reference continuation check

The model-free review reproduced a false pass when the reference record used a
different forced sequence with the same length. The reference probe now emits
the schedule, and the comparator requires both engines to match the frozen IDs.
The original qualification record remains archived with its original format.

## 2026-11-12 - Matched-residual FFN RMSNorm replay

Reconstructed both F32 post-mixer residuals and replayed the frozen weighted
normalization. Both engines reproduced their retained outputs and agreed on
each shared residual. Switching only the residual reproduced the exact seven
row-16 Q8_K scale changes with each engine and quantizer. [STATUS](STATUS.md)
links the witness.

## 2026-11-12 - Matched-input FFN up projection replay

Replayed the single Q4_K up weight with both retained FFN-normalized tensors
and their original CPU groupings. The Q8_K quantizers agreed on each shared
input before dot work. Both the dot kernel and the retained normalization
scales contributed to the row-16 up differences. The offline witness rounded
to `9` with the reference dot and reference input, and to `10` after either
factor changed. [STATUS](STATUS.md) links the measurements.

## 2026-11-12 - Matched-input SwiGLU replay

Separated CPU SiLU from multiplication on retained gate and up inputs. Both
engines reproduced their full retained SwiGLU outputs. SiLU differed by one
ULP at each selected feature. Multiplication agreed on all shared operands
in the row-16 block. Switching the up tensor moved the threshold lane across
`9.5` for both gate inputs and SiLU engines. [STATUS](STATUS.md) links the replay.

## 2026-11-12 - Matched-input FFN down projection replay

Replayed the single Q6_K weight with both retained 37-row SwiGLU tensors.
The two Q8_K quantizers matched on each identical input before dot work.
The retained input difference crossed the row-16 rounding threshold and
reproduced the projection gap. Both engines' dots on identical bytes differed
by at most `2.2351742e-8`. [STATUS](STATUS.md) links the exact witness.

## 2026-11-12 - Complete CPU layer-0 trace

Built the pinned CPU reference observer and locked Candle diagnostic, then
captured 11 checkpoints per engine on the 37 frozen tokens with both Candle
switches enabled. The maximum delta rose from `2.0861626e-7` at SwiGLU to
`1.4526770e-5` at the FFN down projection and persisted into the final
layer-0 output. Both Jobs exited and passed the post-run resource checks.
[STATUS](STATUS.md) links the source-bound result.

## 2026-11-12 - Three CPU arithmetic variants

Reused the isolated diagnostic and retained controls under fresh admission.
Reference RMSNorm alone restored position 2. Ties-to-even alone narrowed its
wrong-token margin. The combined arm matched layer-0 RMSNorm and brought its
ShortConv input projection within `1.4305115e-6`, while the position-2 mismatch
persisted. All three Jobs exited and the final census was quiet. [STATUS](STATUS.md)
links the exact results and original limits.

## 2026-11-12 - CPU diagnostic baseline

Built one isolated CPU diagnostic with independent RMSNorm and Q8_K rounding
switches. The pinned reference and Candle baseline reproduced the retained
arithmetic results through all eight positions. A Python process observed by
the post-baseline quiet-host check stopped the remaining arms before dispatch.
Both model Jobs exited and the terminal census was quiet. [STATUS](STATUS.md)
links the source-bound diagnostic record.

## 2026-11-12 - Offline CPU arithmetic isolation

Replayed the 37 retained layer-0 rows with four combinations of normalization
and Q8_K rounding. Both native RMSNorm results and both common-input activation
buffers were reproduced byte for byte. With the same projection weights and
reduction in every arm, ties-to-even alone reduced the layer-0 projection gap
from `3.411e-4` to `2.176e-6` against the arm with both reference rules.
[STATUS](STATUS.md) links the compact receipt.

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
AI-edited: 2026-11-12 | agent=Codex/root | task=lfm2-moe-shortconv-out-replay | change=record the completed matched ShortConv output replay
