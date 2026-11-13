# LFM2-MoE completed work

## 2026-11-13 - Layer-1 CPU isolation and offline checkpoints

The one-shot native packet `8060130a` at Candle `72a53981` matched all
37 layer-0 entry rows and stopped at layer-1 weighted RMSNorm. Stock Candle
reproduced the row-0/channel-0 mismatch after two offline harness failures.
The corrected RMSNorm and subsequent ShortConv and dense FFN replays matched
the ten reference checkpoints. The final F32 sum matched `l_out-1`, including
all values in rows 16 and 36. Production arithmetic was outside this batch.
[Native terminal](D:/Compact_Evidence/candle/runs/lfm2-moe-layer1-inert-20261113-72a53981/terminal.json).
[Final offline checkpoint](D:/Compact_Evidence/candle/runs/lfm2-moe-layer1-output-offline-20261113-8060130a/summary.json).

## 2026-11-13 - Layer-1 diagnostic preparation

Packet `cbdfeb0f` binds Candle `bd3094dd` and the pinned CPU reference.
Model-free checks verified its input hashes, diagnostic syntax and digest
rejections. Qualification packet preparation now also binds the example
build script, download module and CUDA kernel sources. The native layer-1
trace ran under the successor packet recorded above.
[Preparation receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer1-inert-20261113-bd3094dd/summary.json).

## 2026-11-13 - Full-model CPU prefill with verified layer 0

One admitted run reused the bound candidate on the 37 frozen IDs.
Layer 0 matched all 75,776 output values. The final-position top token
was `207` in both engines, with a maximum absolute delta of `1.313808`
and RMS of `0.612766` across the nine frozen logit probes.
Both bounded Jobs released their processes.
[Full CPU prefill receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-full-cpu-prefill-20261113-bd3094dd-recheck/summary.json).

## 2026-11-13 - Single-path layer-0 replay

The four explicit Q4_K packing widths fixed the earlier compilation
failure. One fresh attempt ran isolated Candle from the 37 frozen IDs.
Fifteen F32 stages and four Q8_K buffers matched llama.cpp. The stock
FFN down control reproduced, and the final output matched all 75,776
values, including rows 16 and 36. Both Jobs released their processes.
[Layer-0 replay receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer0-single-path-20261113-bd3094dd-widths/summary.json).

## 2026-11-13 - Final layer-0 residual check

One offline F32 addition of the corrected Q6_K FFN down output and the
retained post-mixer residual matched all 37 rows of llama.cpp `layer_out`.
All 75,776 values and the complete output SHA-256 matched.
[Final residual receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer0-output-offline-20261113-bd3094dd/summary.json).

## 2026-11-13 - Full Q6_K FFN down projection replay

One admitted isolated Candle replay reproduced the stock control and
matched all 75,776 reference outputs on the recovered Q8_K buffer.
Row 16, channel `1649` matched `3c08db15` and all 2,048 channels in
row 36 matched. Both bounded Jobs released their processes.
[Full projection receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-ffn-down-schedule-20261113-bd3094dd/summary.json).

## 2026-11-13 - Offline Q6_K FFN down dot trace

The reconstruction verified the retained 37-row controls and matched both
native outputs at row 16, channel `1649`. Block 0's F32 contribution was
the first difference after matching the integer dot. This run read retained
bytes and launched zero native probes.
[Offline dot receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-ffn-down-dot-offline-20261113-bd3094dd/summary.json).

## 2026-11-13 - Reference SiLU replay

The two row-16 SiLU traces first differed at the exponent. The pinned
reference SiLU and corrected up tensor recovered all 470 Q8_K scales
across the 37 rows. Both quantizers matched the reference buffer.
[Reference SiLU receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-silu-reference-20261113-bd3094dd/summary.json).

## 2026-11-13 - Corrected-input SwiGLU replay

Both CPU paths recovered Q8_K byte `9` at row 16, block 38, lane 250
on the corrected up output and reference gate. SiLU was the first
differing stage across the 37 rows and left 470 Q8_K scale differences.
[Corrected-input SwiGLU receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-swiglu-corrected-up-20261113-bd3094dd/summary.json).

## 2026-11-13 - Full FFN up projection replay

The isolated GEMM/GEMV candidate matched all 435,712 llama.cpp outputs
on the matched Q8_K input, including both row-16 witnesses. The stock
projection and Q8_K controls reproduced.
[FFN up replay receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-ffn-up-schedule-20261113-bd3094dd/summary.json).

## 2026-11-13 - Matched-input FFN up dot trace

Both native schedules reproduced their retained FFN up outputs. The
source reconstruction matched both row-16 witnesses and traced the
differences to F32 accumulation. [FFN up dot trace](D:/Compact_Evidence/candle/runs/lfm2-moe-ffn-up-dot-trace-20261113-bd3094dd/summary.json).

## 2026-11-13 - Composed layer-0 ShortConv replay

The verified executable reproduced the stock controls and matched the
reference at all seven stages through FFN RMSNorm. All seven row-16
Q8_K scale blocks recovered. The earlier attempts stopped at build and
post-build admission. [First stop](D:/Compact_Evidence/candle/runs/lfm2-moe-layer0-composed-replay-20261112-5648414d/summary.json),
[post-build stop](D:/Compact_Evidence/candle/runs/lfm2-moe-layer0-composed-replay-20261113-bd3094dd/summary.json),
[completed replay](D:/Compact_Evidence/candle/runs/lfm2-moe-layer0-composed-replay-20261113-bd3094dd-reuse/summary.json).

## 2026-11-12 - Qualification packet source binding

The packet now binds the working-tree bytes of shared build sources and
Cargo inputs alongside the MoE overlay. Regression coverage verifies local
edits, new source files, deletions, and exclusion of unrelated scratch files.
[QUALIFICATION](QUALIFICATION.md) records the source-binding contract.

## 2026-11-12 - ShortConv output projection schedule

Applied the proven GEMM and GEMV schedules to the retained reference
gate. The isolated candidate matched all 37 output rows and the
row-16 channel-557 witness. [STATUS](STATUS.md) links the comparison.

## 2026-11-12 - ShortConv three-tap accumulation replay

Matched the three tap products at row 2, channel 2. The two F32 reduction
orders reproduced each native convolution across all 37 rows and its
row-16 gate. [STATUS](STATUS.md) links the trace and both controls.

## 2026-11-12 - Corrected ShortConv mixer replay

Replayed the corrected projection on 37 frozen rows. The two engines
matched at `B × X` and first differed in convolution. The fixed-input
output projection confirmed the remaining dot-kernel contribution at
row 16, channel 557. [STATUS](STATUS.md) links the measurements.

## 2026-11-12 - ShortConv GEMV tail replay

The isolated candidate applied the pinned block-level reduction on row 36
and matched all 6,144 values. The 36 GEMM rows and the channel-306 witnesses
still matched. [STATUS](STATUS.md) links the full-projection evidence.

## 2026-11-12 - Combined ShortConv input projection

The isolated Candle candidate matched the pinned llama.cpp projection on all
36 GEMM rows and recovered every channel-306 witness. The comparison covered
all 227,328 outputs and confined the remaining differences to the final GEMV
row. [STATUS](STATUS.md) links the source-bound run.

## 2026-11-12 - ShortConv accumulation replay

Replayed integer scale/min folding and separate F32 accumulation on the
matched operands. The combined schedule matched all reference B, X and
`B × X` values at channel 306, rows 14–16. [STATUS](STATUS.md) links the
four-way comparison.

## 2026-11-12 - Matched Q4_K ShortConv dot trace

Both native packed probes reproduced the retained projections. The
reconstruction matched all six B/X values at channel 306 and localized the
first differences to F32 accumulation in block 0. [STATUS](STATUS.md) links
the detailed trace and the scoped CPU admission record.

## 2026-11-12 - Matched-input ShortConv input projection replay

Both engines reproduced their native projections on 37 normalized rows.
Q8_K scales, bytes and sums agreed before the dot. Dot arithmetic reproduced
the channel-306 `B × X` changes at rows 15 and 16, while row 14's product
matched. [STATUS](STATUS.md) links the complete replay.

## 2026-11-12 - Shared-input ShortConv convolution replay

Reused the pinned llama.cpp probe and replayed both retained `B × X` tensors
through llama.cpp and Candle. Both controls reproduced bit for bit. Incoming
input drift accounted for the larger part of the row-16 channel-306 gap.
[STATUS](STATUS.md) links the measurements and admission receipts.

## 2026-11-12 - Crossed ShortConv input replay

Crossed both C slices and convolution outputs through one pinned dot kernel.
The convolution swap had the larger row-16 channel-557 effect on either
background. Both swaps affected scale bits, and all four cases preserved
row-16 quant bytes and block sums. [STATUS](STATUS.md) links the receipt.

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
AI-edited: 2026-11-13 | agent=Codex/root | task=lfm2-moe-layer1-review | change=record the native stop and completed offline checkpoints
