# LFM2-MoE current state

## Current checkpoint

Production packet `d91a095a` consumed its attempt on the CPU arithmetic
comparison. The sampled prefill logits differed by up to 1.402465. The
first differing top token followed forced IDs `49, 1252` at position 2
(`1113` reference, `5141` Candle). CUDA was never reached.
[Terminal receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-maintained-layer2-20261115-30e48252/qualification/summary.json).

On the same 37 frozen rows, retained embeddings and reconstructed maintained
RMSNorm matched bitwise. An offline `GenericTiled` replay located two layer-0
ShortConv `in_proj` differences. Row 0, channel 2 diverged in the Q4_K dot
on identical Q8_K operands. The first activation rounding witness was row 2,
block 2, lane 119 (`-38.5` scaled, `-39` Candle, `-38` reference). Ties-to-even
reproduced all pinned Q8_K bytes, including the five changed sums. The production
logs contain no checkpoints or logits for `1113` and `5141`, so the path from
these local differences to the top-token flip remains open.
[Offline trace](D:/Compact_Evidence/candle/runs/lfm2-moe-d91a095a-retained-trace-20261115/trace.json).

The maintained MoE CPU Q4_K ShortConv input projection now matches all 296
Q8_K blocks and 227,328 F32 outputs on the 37 frozen rows. Rounding and dot
schedule controls were run separately. Both changes are needed for bitwise
parity. [Projection receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-shortconv-in-proj-maintained-20261115/summary.json).

The single offline connected replay under the original limits matched
embeddings, RMSNorm, input Q8_K, `in_proj` and B × X. It stopped at the first
convolution mismatch, row 2, channel 2 (`38fb8ee6` Candle, `38fb8ee4` llama.cpp).
The retained three-tap proof explains those same bits. The Job released its
PID. The maintained CPU tap-order port then matched all 75,776 convolution
values from Candle's B × X in one offline attempt with the authorized 16 GiB
RAM and commit floors. Its Job released the PID. [Convolution receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer0-ordered-shortconv-20261116/summary.json).

The maintained CPU layer-0 path matches through dense FFN gate/up on 37 rows.
Pre-dot Q8_K and all 435,712 values in each projection matched bitwise.
[Ordered replay](D:/Compact_Evidence/candle/runs/lfm2-moe-layer0-dense-ffn-gate-up-ordered-20261116/summary.json).

The dense CPU SiLU port matched all 435,712 SiLU values and SwiGLU products
on 37 rows. All 496,984 bytes of FFN-down Q8_K matched.
[SwiGLU receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer0-dense-swiglu-ordered-20261116/summary.json).
The [prior dense Q6_K replay](D:/Compact_Evidence/candle/runs/lfm2-moe-layer0-dense-ffn-down-20261116/completion.json)
stopped at row 0, channel 0. With the tiled F32 schedule, the source-bound
offline replay passed all 19 stages on the 37 rows, including 75,776 bitwise
matches each for dense FFN down and `layer_out`. Both Jobs released their PIDs.
[Layer-0 receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer0-dense-ffn-down-ordered-20261116/summary.json).

The maintained layer-1 packet binds that exact output and 16 checks. An
active Cargo build stopped admission before dispatch. The attempt is unused.
[Admission receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer1-maintained-20261116/summary.json).

Packet `9c8006a5` consumed its one authorized connected CPU attempt under
the original limits. Fresh admission and the whole GGUF hash passed. The
37 frozen rows passed all 27 declared comparisons through `layer_out`.
Entry through Q/K RoPE and ordered top-K were bit-exact. Downstream stages
passed the declared controls and per-engine formulas. `layer_out` has
72,211 classified cross-engine differences among its 75,776 values.
The Job exited with code 0, the probe PID was absent, and the post-run
census passed. [Terminal receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-published-snapshot-20261115/terminal.json).

Earlier inert packet reviews are recorded in [HISTORY](HISTORY.md#2026-11-15---inert-swiglu-and-expert-down-successor).

The maintained GGUF CPU candidate applies the connected diagnostic's
RMSNorm and RoPE arithmetic, Q4_K/Q6_K attention projections, F32 router
projection, and the indexed Q4_K gate/up and Q6_K down schedules with
reference-order SiLU. Its grouped weighting and expert-order accumulation
reuse the existing implementations. Native and GGUF routing retain their
separate normalization formulas. CUDA, dense LFM2 and other expert dtype
mixes keep their existing dispatch. CPU dense Q6_K down uses the tiled F32
schedule.
Focused synthetic and retained-operand checks cover the maintained CPU
operators. A connected layer-1 replay is next in [TODO](TODO.md).
[Inert candidate evidence](D:/Compact_Evidence/candle/runs/lfm2-moe-maintained-layer2-20261115-30e48252/summary.json).

For the clean MoE/LFM3 snapshot, use the rebound packet identified in the
[snapshot receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-published-snapshot-20261115/summary.json).
The snapshot preserves the LFM3 test additions and the prior inert packet.

The [up-projection replay](HISTORY.md#2026-11-15---layer-2-up-projection-replay)
explains the diagnostic difference in consumed packet `f007a0a9`.
The generic quantized fallback uses `GenericTiled`.

Maintained integration, the production position-2 failure and Edge
acceptance remain open. Standalone LFM3 CPU/CUDA text
qualification and the LTS support contract are separate remaining gates.
[TODO](TODO.md) owns their order. [Production qualification](QUALIFICATION.md)
binds two frozen cases and eight teacher-forced positions each.

## Implementation and retained proof

The November 10 capability was published as Candle
`95df99a0bda2f6ac64dbeb8475850f826bd92ca9` from base `5f094821`.
The first Q4 parity attempt stopped during CPU comparison. The comparator
repair exposes a remaining CPU numerical disagreement. Edge qualification and
adoption follow supplier parity. D1 stays closed.

The [frozen continuation check](HISTORY.md#2026-11-12---frozen-reference-continuation-check)
records the position-2 failure (`1113` vs `6986`) and the requirement
for a newly bound reference binary.

Native `models::lfm2_moe::{Config, Model, Cache}` loads the official per-expert
safetensors layout and exposes token/embedding forward, hidden states, logits
projection and cache reset. The pinned native index confirms 7,610 tensor names.
Shared attention/convolution math is reused through private mixers; dense LFM2
public behavior is preserved. Routing follows pinned sigmoid/bias/top-K semantics.

`models::quantized_lfm2_moe::{ModelWeights, inspect_gguf_metadata}` validates all
required roles, dimensions, storage types, offsets/alignment, non-overlap and
payload EOF. Q4_K/Q6_K/Q8_0 experts stay quantized; F32/F16/BF16 storage has
fixture coverage. CPU/CUDA share grouped execution. Input IDs and context are
checked before embedding allocation, and failed forwards clear both caches.
Memory estimation includes widened storage, auxiliary tensors, full input,
prefill scratch and KV/convolution cache. No silent device fallback is used.

Retained implementation proof includes eight CPU MoE tests and ten with CUDA enabled,
covering the pinned F32 oracle, routing, native untied output/missing experts,
chunking/decode/reset, malformed GGUF, lower-precision storage, retained quantized
matrices and recovery. CUDA production weights have not been loaded. Native
F32 tolerance is `atol=1e-5`, `rtol=1e-4`; quantized tolerance is separately
recorded in [DECISIONS](DECISIONS.md).

The preceding implementation gate passed 617 Rust tests with 14 ignored,
197 release assertions on each PowerShell host, library/example checks,
Clippy and existing model-free controls. These are retained source-bound results,
not new full-matrix or CUDA runs from the review/cleanup.
Overlay registration has seven independent owners and its 12 regression cases
pass. Summary Bank and the independent existing manifests remain verifiable.

Edge has an explicit `lfm2-moe-local` backend/headless CLI/API feature, new
`edgesymbio.local.lfm25-24b-a2b.q4km.v1` candidate, exact artifact admission,
48 KiB / 12,288 total / 2,048 generated limits, seeded reasoning parsing and
existing validated tool/JSON permission paths. Six focused model-free tests
pass with the local CUDA overlay. Exact rendered bytes and token IDs match four
pinned official tokenizer cases without loading model payloads. Edge's broader
backend regression and CLI/API checks are retained in the run record.

The candidate remains unapproved/held pending production qualification. Local
Cargo overlays restore original Edge locks byte-for-byte; its ordinary pin is
still `38d90519`. GenEye's `81e1aa28`, other consumer pins, frozen releases and
all original owner edits are preserved. The current publication covers Candle
only; it does not grant Edge pin/lock adoption or production execution.

[HISTORY](HISTORY.md) retains the initial model inspection and packet receipts.

The review/recheck compared all 48 Candle and 16 Edge scoped source hashes with
the retained final inventory at the unchanged Candle/Edge heads. Previous token,
context, routing, memory and consumer-identity fixes remain closed on recheck.
Cleanup repaired two new defects: the comparator previously admitted matching
shortened/wrong-reference records, and the example discarded normalized EOS
metadata by requiring a U32 encoding again. Comparisons now bind the frozen case,
reference revision, full continuation and exact count; the example consumes the
inspector's checked EOS. Eleven model-free protocol tests and five quantized CPU
tests pass, including EOS acceptance/refusal. Required checks/Clippy and their
exact commands belong to the review receipt under the run's `review-20261110/`.
The main run summary and all older proof remain unchanged.

Packet `00ab6c5e` consumed its one attempt on November 11. Both CPU reference
children and Candle's CPU arithmetic child exited successfully and released
their Jobs. The first comparison failed while decoding a clipped UTF-8 tokenizer
diagnostic. CPU Unicode and both RTX 4090 children were skipped. The host returned
to its pre-run resource range. The [attempt receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-qualification-20261111-00ab6c5e/failure-review.json)
binds the three original logs and cleanup evidence.

The comparator reads JSON lines from byte logs and reports the first
top-1 mismatch. Thirteen protocol tests passed. Offline replay of the original
arithmetic pair reached the numerical gate and rejected zero-based position 2
with reference token `1113` and Candle token `6986`.
The [cleanup receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-review-cleanup-20261111-95df99a0/summary.json)
records the source and replay identities.

The retained pair already differs at the output of the 37-token prefill. Its
seven sampled logits have a maximum absolute delta of `0.70701456` at position
0, before either engine consumes a forced token. Both still select `207` there.
The original packet omitted intermediate tensors and both competing logits at
position 2. The paired trace localized the first unequal operation, and the
later diagnostic measured the rank margin for both controls.

The pinned reference log identifies layer 0 ShortConv input projection as
repacked `Q4_K_8x8`. Candle's generic and packed x86 Q4_K paths use
`BlockQ8K::from_float`,
which rounds activation half-ties away from zero. The pinned reference's scalar
and AVX2 conversions round half-ties to even. A controlled 256-value block
demonstrates the difference with identical operands. The same arithmetic
comparison found equal quantization for four existing synthetic fixture rows.
Both prebuilt synthetic prefill tests passed. This establishes a source-level
difference and a fixture coverage gap. The production cause is unresolved.

The paired CPU trace used the same 37 frozen arithmetic IDs. The reference ran
one prefill with tensor callbacks. The Candle diagnostic invoked the same
embedding, RMSNorm and Q4_K projection operators through its retained release
libraries and read only their required GGUF tensors. All 75,776 embedding values
matched. Layer-0 RMSNorm was first to differ, with a maximum absolute delta of
`1.4305115e-6`. Candle uses a float sum and per-element division. The pinned
reference accumulates float products in double and multiplies by a reciprocal.
The ShortConv input projection differed by up to `0.00034119189`.

Replaying the reference-normalized rows through both activation converters
produced identical scale bits in all 296 Q8_K blocks. Five signed quant bytes
differed at exact half-ties on packed rows 2, 9 and 30. The scalar tail matched.
For example, the scaled operand `104.5` became `104` in llama.cpp and `105` in
Candle. With common normalized inputs, the projection still differed by up to
`0.00048783049`. That replay includes the rounding difference and cannot isolate
dot arithmetic with identical quantized bytes. The trace establishes two early
numerical differences that the CPU diagnostic then tested independently.

The [paired trace receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-paired-trace-20261111-42deff05/summary.json)
binds the operators, frozen input and both Job exits. The follow-up offline replay
matched both native RMSNorm outputs and both captured common-input Q8_K buffers
byte for byte. The RMS denominator differed on 33 of 37 rows. Using only the
reference sum precision or only its reciprocal multiply did not reproduce the
reference normalization. Both changes together did.

The projection replay held Q4_K weights and the F64 reduction fixed. Compared
with the arm using both reference rules, the maximum absolute deltas were
`3.411e-4` for baseline, `4.878e-4` for RMSNorm alone and `2.176e-6` for ties-to-even
alone. Ties-to-even changed two activation bytes with Candle normalization and
five with reference normalization. The largest residual between a retained
native projection and its own operand-matched oracle was `1.556e-6`. This is
an arithmetic reconstruction of layer 0. The [offline replay receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-cpu-arithmetic-20261112-42deff05/summary.json)
records the data and limits.

The isolated CPU diagnostic built from `42deff05` with the frozen lockfile.
The pinned reference and Candle baseline passed fresh admission and reproduced
all eight retained top-1 selections and the original sampled logits. At
zero-based position 2, the `1113` minus `6986` logit margin was `+0.307556`
for the reference and `-0.638268` for Candle. Prefill sampled logits differed
by up to `0.707015`. Both Jobs exited successfully and released their processes.
The post-baseline quiet-host check saw a Python process and stopped the attempt
before any variant launched. The terminal census was quiet.
[Control receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-cpu-variants-20261112-42deff05/summary.json).

The authorized continuation reused the verified diagnostic and both controls.
All three CPU variants passed fresh admission and validation on the same 37
tokens and eight teacher-forced positions. The 180-second, 32 GiB child limits,
900-second sequence limit and two CPU threads were retained. Every Job exited
and the final quiet-host check passed.

| CPU run | Position 2 top token | `1113` minus `6986` logit | Prefill max sampled logit delta vs reference |
| --- | ---: | ---: | ---: |
| Pinned reference | `1113` | +0.307556 | 0 |
| Candle baseline | `6986` | -0.638268 | 0.707015 |
| Reference RMSNorm only | `1113` | +0.170671 | 0.848389 |
| Q8_K ties-to-even only | `6986` | -0.262487 | 0.826451 |
| Both | `6986` | -0.619778 | 1.496321 |

RMSNorm alone matched all eight reference selections and reproduced the
layer-0 normalized rows byte for byte. Rounding alone brought the layer-0
ShortConv input projection within `2.6226044e-6` of the reference. Both changes
reproduced the normalized rows and reduced that projection delta further to
`1.4305115e-6`, while increasing downstream sampled prefill drift.
[Three-variant receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-cpu-variants-resume-20261112-42deff05/summary.json).

The paired layer-0 trace was prepared in an isolated source snapshot. Quiet
host and memory admission passed for the reference build. The bounded launcher
refused its `cmd.exe` wrapper because other sessions used that executable. The
attempt stopped before compilation, leaving the layer-0 output unmeasured.
[First build stop](D:/Compact_Evidence/candle/runs/lfm2-moe-layer0-paired-20261112-42deff05/summary.json).

The direct retry bound VS 18 `cl.exe` and SDK `10.0.26100.0`. Fresh preflight
found five Python processes. Physical RAM and commit headroom exceeded the
original floors, but the quiet-host gate stopped the build before dispatch.
[Direct build admission receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer0-paired-vs18-20261112-42deff05/summary.json).

The completed pair used the pinned VS 18 compiler directly and the locked
Candle CPU build from `42deff05`. For this diagnostic, admission recorded Python
activity and enforced the original 40 GiB RAM and commit floors and absence of
competing model or build processes. Both engines ran one prefill and reproduced
their retained prefix checkpoints on all 37 frozen tokens. Both Candle
arithmetic switches were enabled. Both bounded Jobs exited and released their
processes.

The ShortConv mixer output differed by at most `1.0430813e-7` and SwiGLU by
`2.0861626e-7`. The largest increase came at the dense FFN down projection,
with a maximum delta of `1.4526770e-5` at zero-based prompt row 16, channel
1649, token `355`. The same delta reached the complete `37 × 2048` layer-0
output. Both engines' captured tensors reconstructed the final residual
exactly. [Completed trace receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer0-paired-bounded-20261112-42deff05/summary.json).

The matched-input replay used the single Q6_K `blk.0.ffn_down.weight` tensor
and all 37 retained SwiGLU rows in their original order. Before any dot work,
the engines agreed on every Q8_K scale bit, quant byte and block sum on each
identical input. Switching between the two retained inputs changed 1,473
scale bit patterns and seven quant bytes. At row 16, block 38, lane 250, the
F32 scaled operand crossed `9.5` (`9.4999971` to `9.5000019`) and both
quantizers changed `9` to `10`.

Both native dot replays reproduced the retained down projections. With
identical Q8_K bytes, the maximum cross-engine dot delta was `2.2351742e-8`.
At row 16, channel 1649, changing only the input in Candle contributed
`-1.4528632e-5`, and switching dot kernels on the same reference bytes
contributed `+1.8626451e-9`. Their sum reproduces the original
`-1.4526770e-5`. [Matched-input replay receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-ffn-down-replay-20261112-42deff05/summary.json).

The upstream SwiGLU replay preserved the `37 × 11776` layout. Both engines
reproduced their retained SwiGLU outputs byte for byte. With either identical
gate input, Candle SiLU was one ULP higher at feature 9978 and one ULP lower
at feature 9835, with signed differences of `+7.4505806e-9` and
`-7.4505806e-9`. Given identical SiLU and up operands, multiplication
matched bit for bit over the full row-16 block in all eight combinations.

The retained up operands differed by `+2.9802322e-8` at feature 9978 and
`+5.9604645e-8` at feature 9835. Holding the gate input and SiLU engine fixed,
switching from the reference up tensor to Candle's moved the scaled lane
above `9.5` in all four combinations. With the reference up tensor every
combination rounded to `9`. With Candle's every combination rounded to `10`.
[SwiGLU replay receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-swiglu-replay-20261112-42deff05/summary.json).

The matched-input `blk.0.ffn_up.weight` replay used the original Q4_K
weights, the `37 × 2048` FFN-normalized rows, llama.cpp's four-row
`q4_K_8x8` activation groups and scalar tail, and Candle's 16-row tiles.
Both quantizers agreed on all scale bits, quant bytes and block sums for
each shared input before either dot ran. Switching between the retained
inputs changed 212 scale bit patterns, seven in row 16, with equal quant
bytes throughout. Each native dot reproduced its retained up projection.

| Row-16 feature | Dot change on reference Q8_K | Input change with Candle dot | Retained Candle minus reference |
| --- | ---: | ---: | ---: |
| `9978` | `+2.2351742e-8` | `+7.4505806e-9` | `+2.9802322e-8` |
| `9835` | `+2.9802322e-8` | `+2.9802322e-8` | `+5.9604645e-8` |

An offline reconstruction fed each of the four up-projection combinations
through the retained gate and SiLU tensors at row 16. Only reference dot
with reference normalized input yielded `9` at the downstream witness lane.
Changing either factor on its own yielded `10` for all four gate/SiLU
combinations.
[FFN up replay receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-ffn-up-replay-20261112-42deff05/summary.json).

The matched-residual FFN RMSNorm replay reconstructed the post-mixer input
from the retained embedding and ShortConv mixer outputs in F32. It used the
frozen `blk.0.ffn_norm.weight` and epsilon bits `0x3727c5ac`. Both engines
reproduced their retained normalized tensors, and their outputs matched bit
for bit on each shared residual with Candle's reference RMSNorm switch.

On either shared residual the Q8_K converters agreed on every scale, quant
byte and block sum. Changing only the residual reproduced all 212 original
scale differences with no quant-byte differences. The seven row-16 changes
reproduced the exact scale bit patterns at blocks `0, 2, 3, 4, 5, 6, 7` with
each RMSNorm and converter combination. The retained embeddings were
identical, so the input difference enters through the ShortConv mixer output.
[FFN RMSNorm replay receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-ffn-norm-replay-20261112-42deff05/summary.json).

The [ShortConv output replay receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-shortconv-out-replay-20261112-42deff05-proof/summary.json)
binds both native `37 × 2048` mixer outputs bit for bit. Candle `AdaptedLinear`
used the `Auto` x86 AVX-512 VNNI kernel with four-row activation tiles.
llama.cpp used `q4_K_8x8` four-row groups and a scalar tail. On either shared
F32 input, the Q8_K scale bits, quant bytes and block sums matched exactly.
Switching the retained inputs changed 263 scale bits, including all eight
blocks at row 16, with zero changes to quant bytes.

At row 16, channel 557, the input switch reproduced the full
`-1.043081283569336e-7` mixer difference. The two kernels also differed on
matched Q8_K operands elsewhere, by up to `8.940696716308594e-8` at the tail.
An offline four-way propagation reproduced both retained FFN RMSNorm and Q8_K
controls bit for bit. Of the seven retained row-16 FFN scale changes, blocks
0 and 5 respond only to the input switch. Blocks 2, 3, 4, 6 and 7 also
respond to the dot switch on at least one input. Production parity remains open.

The [crossed ShortConv replay](D:/Compact_Evidence/candle/runs/lfm2-moe-shortconv-crossed-inputs-20261112-5648414d/summary.json)
reused one pinned llama.cpp `q4_K_8x8` kernel on all 37 rows. Both original
F32 products, Q8_K buffers and fixed-kernel outputs reproduced bit for bit.
Swapping C changed row-16 scale bits in blocks 0, 1, 2, 4 and 5 on either
convolution background. Swapping the convolution changed blocks 0, 1, 3, 5,
6 and 7 on either C background, plus block 2 with reference C. All four
combinations kept identical row-16 quant bytes and block sums. The crossed
reference-C/Candle-convolution input changed one quant byte at row 17,
block 6, lane 110 (`3 → 2`).

At channel 557, the C-only and convolution-only swaps from the reference
control moved the output by `-1.4901161e-8` and `-4.4703484e-8`.
Swapping both moved it by `-1.0430813e-7`, including a
`-4.4703484e-8` interaction. The convolution swap moved the channel more
on either background. All four bounded children passed fresh admission and
released their processes.

The [shared-input convolution attempt](D:/Compact_Evidence/candle/runs/lfm2-moe-shortconv-common-bx-20261112-5648414d/summary.json)
prepared both 37-row `B × X` tensors and the frozen F32 convolution weight.
The pinned llama.cpp probe compiled and its Job released. Admission for the
Candle build measured 39.57 GiB available RAM against the 40 GiB floor.
That attempt stopped before either replay.

The [resumed convolution replay](D:/Compact_Evidence/candle/runs/lfm2-moe-shortconv-common-bx-20261112-5648414d-resume/summary.json)
reused the reference probe, both inputs and the frozen three-tap F32 weight.
Both native controls matched bit for bit with two leading causal zeros. On
either shared input, the maximum operator difference across all 37 rows was
`2.9802322e-8`. At row 16, channel 306, switching inputs in llama.cpp
contributed `+2.0861626e-7` and switching to Candle on that input contributed
`-2.9802322e-8`, reproducing the retained `+1.7881393e-7` gap. The retained
row-14 input at channel 306 matched, while rows 15 and 16 differed. All three
Jobs passed fresh admission and released their processes.

The [ShortConv input-projection attempt](D:/Compact_Evidence/candle/runs/lfm2-moe-shortconv-in-proj-20261112-5648414d/summary.json)
verified that the two retained normalized `37 × 2048` inputs are identical
bit for bit and bound the frozen Q4_K projection weight. Fresh admission
found active CMake and Ninja workloads with 40.96 GiB available RAM and
44.25 GiB commit headroom. The attempt stopped before either build. A
[fresh recheck](D:/Compact_Evidence/candle/runs/lfm2-moe-shortconv-in-proj-20261112-5648414d-resume/summary.json)
found no competing model or build processes. Available RAM was 37.37 GiB
and commit headroom was 38.31 GiB, both below the original 40 GiB floors.
The recheck dispatched no native children.

The [completed input-projection replay](D:/Compact_Evidence/candle/runs/lfm2-moe-shortconv-in-proj-20261112-5648414d-recheck/summary.json)
passed fresh admission for all six bounded jobs. Both native projection
controls and their downstream `B × X` tensors matched the retained bytes.
Before the dot, both engines produced identical Q8_K scales, quant bytes and
block sums on all 37 rows. The Candle AVX-512 VNNI and llama.cpp `q4_K_8x8`
dots on those matched operands differed by up to `1.4305115e-6` in the
projection. At channel 306 the `B × X` products matched at row 14, differed
by `-7.1525574e-7` at row 15 and by `-5.7220459e-6` at row 16 (Candle
minus llama.cpp). At row 16, swapping either B or X from the reference
control shifted the product by `-2.8610229e-6`. This attributes the incoming
causal-window drift to dot arithmetic for these normalized inputs.

The [matched-dot trace admission](D:/Compact_Evidence/candle/runs/lfm2-moe-shortconv-dot-trace-20261112-5648414d/summary.json)
verified the retained binaries and operands, then stopped before the first
child. Available RAM was 37.85 GiB and commit headroom was 38.44 GiB against
the original 40 GiB floors. No competing model or build process was found.
The [fresh trace recheck](D:/Compact_Evidence/candle/runs/lfm2-moe-shortconv-dot-trace-20261112-5648414d-recheck/summary.json)
also stopped before dispatch. It measured 34.19 GiB available RAM and
37.24 GiB commit headroom with no competing model or build processes.
The [following trace recheck](D:/Compact_Evidence/candle/runs/lfm2-moe-shortconv-dot-trace-20261112-5648414d-recheck2/summary.json)
stopped with 38.87 GiB available RAM and active Cargo and Rust compiler
processes. Commit headroom passed at 40.14 GiB. No native child ran.
The [last recheck at the original admission floors](D:/Compact_Evidence/candle/runs/lfm2-moe-shortconv-dot-trace-20261112-5648414d-recheck3/summary.json)
found the model and build queues clear. It stopped before dispatch at
37.74 GiB available RAM and 38.99 GiB commit headroom.

The [completed matched-dot trace](D:/Compact_Evidence/candle/runs/lfm2-moe-shortconv-dot-trace-20261112-5648414d-recheck4/summary.json)
used the operator's relaxed 16 GiB RAM and commit floors for this retained
CPU replay. Both probes passed the workload checks and retained the original
grouping, job memory cap and time limits. Peak Job memory was under 20 MiB
per probe and both processes exited cleanly.

Both probes reproduced the complete retained projection tensors. The
source-derived trace reproduced all six B/X values at channel 306,
rows 14–16, bit for bit. The 384 reconstructed subblock integer dot products
and sums agreed. llama.cpp applies the scale and minimum codes in integer
space, then updates separate F32 accumulators per subblock pair and subtracts
at the end. Candle applies each subblock's scale and minimum terms in one
F32 accumulator. The first unequal net appears in block 0 for all six
witnesses. At row 16, both final lanes shift by `+9.5367432e-7` (Candle minus
llama.cpp). This proves the local arithmetic difference on the matched
operands. End-to-end CPU/CUDA parity remains open.

The [arithmetic replay](D:/Compact_Evidence/candle/runs/lfm2-moe-shortconv-accumulation-replay-20261112-5648414d/summary.json)
varied integer scale/min folding and separate F32 accumulation on the same
retained operands. Both controls reproduced the retained B, X and product
bits at channel 306, rows 14–16. Integer folding alone matched two of six
reference lane values and two of three products. Separate F32 accumulation
alone matched one lane and zero products. The combined schedule matched
all six lane values and all three products. The single replay passed the
revised 16 GiB RAM and commit floors and its Job exited cleanly.

The [isolated combined-schedule candidate](D:/Compact_Evidence/candle/runs/lfm2-moe-shortconv-combined-projection-20261112-5648414d/summary.json)
reproduced the retained stock Candle projection and all Q8_K bytes. With
Candle's original `4 × 16` tiling, it matched all 221,184 values in rows
0–35 and all six B/X values and three products at channel 306, rows 14–16.
The remaining 4,799 differences were confined to row 36, with maximum
absolute error `4.7683716e-7` at column 1767. Stock Candle differed at 205,556
values across the full projection. Both bounded Jobs passed the revised
16 GiB RAM and commit floors and released their processes.

Pinned llama.cpp dispatches row 36 to GEMV. That kernel sums all four
integer pair terms within each 256-value block before the two F32 FMAs.
The [isolated GEMV replay](D:/Compact_Evidence/candle/runs/lfm2-moe-shortconv-gemv-tail-20261112-5648414d/summary.json)
used that reduction only on row 36. All 6,144 tail values matched, and
rows 0–35 matched both the previous candidate and pinned llama.cpp.
The full projection matched all 227,328 values, including all six B/X
values and three products at channel 306. The stock projection and Q8_K
controls also reproduced. Both Jobs passed the revised 16 GiB RAM and
commit floors and released their processes.

The [corrected-projection mixer replay](D:/Compact_Evidence/candle/runs/lfm2-moe-shortconv-mixer-replay-20261112-5648414d-recheck/summary.json)
reproduced both retained mixer controls and compared all 75,776 values per
stage. `B × X` matched bit for bit. Convolution first differed at row 2,
channel 2 and had a maximum absolute gap of `2.9802322e-8` across 19,513
values. The full output projection differed at 69,123 values, with a
maximum absolute gap of `8.9406967e-8`.

At row 16, channel 557, `B × X`, convolution and `C × conv` matched.
Across that row, the changed convolution and gate moved the pinned
output projection at channel 557 by `+1.4901161e-8`. Switching to Candle
on that same gated tensor contributed `-4.4703484e-8`, for a net gap of
`-2.9802322e-8` (Candle minus llama.cpp). The original stock gap at that
coordinate was `-1.0430813e-7`. The first attempt stopped at a mistyped
comparator hash before dispatch. The recheck verified the corrected
binding, passed the revised admission and released all five Jobs.
Production CPU/CUDA parity and adoption stay open.

The [three-tap replay](D:/Compact_Evidence/candle/runs/lfm2-moe-shortconv-tap-replay-20261112-5648414d/summary.json)
ran both native convolutions on the same retained `B × X` with two causal
zeros. At row 2, channel 2, the reconstructed tap products matched at
`390a3ca3`, `3927934a` and `b934087a`. Candle's horizontal reduction added
taps 0 and 2 first, then tap 1, yielding `38fb8ee6`. Pinned llama.cpp
accumulated taps in order 0, 1, 2 with separate F32 multiply and add,
yielding `38fb8ee4`. Candle was two ULPs higher (`+1.4551915e-11`).

The reconstructed schedules matched all 75,776 convolution values and
all 2,048 row-16 gate values for each engine. Across engines, the row-16
gate differed at 509 channels, with maximum absolute difference
`5.9604645e-8`. Channel 557 matched at `b88b2755`. All three Jobs passed
fresh admission under the revised limits and released their processes.

The [matched-gate output projection replay](D:/Compact_Evidence/candle/runs/lfm2-moe-shortconv-out-proj-schedule-20261112-5648414d/summary.json)
applied the proven Q4_K schedules to the retained reference gate. The
isolated Candle candidate matched pinned llama.cpp on all 75,776 values,
including 73,728 GEMM values in rows 0–35 and 2,048 GEMV values in row 36.
Row 16, channel 557 matched at `3e522893`. The stock Candle control on
that gate reproduced `3e522891` and differed from llama.cpp at 69,079
values overall. Both engines' retained Q8_K bytes matched, and fresh
Candle quantization reproduced them. The bounded build and replay passed
the revised admission and released both Jobs.

The qualification packet generator now hashes the shared local CPU/CUDA
build inputs along with the MoE overlay. The inventory captures current
working-tree bytes and deleted paths so a shared-kernel change is visible
before packet review. The prior receipts remain the controls for the
composed mixer and FFN normalization replay.

The composed layer-0 replay reused the verified executable after fresh
admission. The stock controls reproduced and all seven composed stages
matched bit for bit across the 37 frozen rows. FFN RMSNorm matched all
75,776 values and the full 86,432-byte Q8_K buffer. Row-16 scale blocks
`0, 2, 3, 4, 5, 6, 7` all recovered the reference bits. The single replay
process exited and released its Job Object.
[Composed replay receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer0-composed-replay-20261113-bd3094dd-reuse/summary.json).

The FFN up dot trace replayed both native CPU schedules on the recovered
reference Q8_K input. Both reproduced their retained outputs exactly.
The integer scale and minimum sums agreed. At feature `9978`, the first
completed-block difference was the F32 minimum term in block 0. At
`9835`, the first minimum term difference appeared in the second subblock
pair of block 0 and canceled by that block's end. Both F32 terms differed
at the end of block 1. [FFN up dot trace](D:/Compact_Evidence/candle/runs/lfm2-moe-ffn-up-dot-trace-20261113-bd3094dd/summary.json).

The isolated FFN up projection matched all 435,712 llama.cpp output values
bit for bit on the matched Q8_K rows. This covered the 36 GEMM rows and
the GEMV tail. Row 16 recovered `3d271bf2` at feature `9978` and
`3eb6b243` at feature `9835`. The stock Candle and Q8_K controls
reproduced, and both bounded Jobs exited and released their processes.
[FFN up projection receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-ffn-up-schedule-20261113-bd3094dd/summary.json).

The corrected-input SwiGLU replay used the same reference gate and corrected
up tensor for both CPU paths across 37 rows. SiLU was the first differing
stage. On row 16, Candle minus llama.cpp was `+7.4505806e-9` at feature
`9978` and `-7.4505806e-9` at `9835`. The product difference was
`+4.6566129e-10` at `9978`, and the products matched at `9835`.
Both multiply paths agreed across row 16's block 38 when given identical
SiLU outputs. Both quantizers agreed on each shared product tensor.
Across the two product tensors, 470 Q8_K scales differed, and all quant
bytes and block sums matched. At row 16, block 38, lane 250, both scaled
operands stayed below `9.5` and encoded byte `9` with scale bits
`b98d4480`. All six bounded processes exited.
[Corrected-input SwiGLU receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-swiglu-corrected-up-20261113-bd3094dd/summary.json).

The matched-gate SiLU trace reproduced both native witness values. At row 16,
features `9978` and `9835`, the first difference was `exp(-x)`. The
reconstructed Candle exponent was one ULP higher at each feature than the
pinned AVX2/FMA approximation. Reference SiLU with the corrected up tensor
matched all 435,712 reference products. Both quantizers recovered all 470
scales and matched the entire 496,984-byte Q8_K buffer. Row 16, block 38,
lane 250 remained byte `9`.
[Reference SiLU receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-silu-reference-20261113-bd3094dd/summary.json).

The Q6_K down-dot trace stopped at the first reference control. The
recovered Q8_K input passed binding and admission. PowerShell left
`GGML_CPU_TILED_MM` present with an empty value when the controller tried
to clear it. The launcher passed that value through, disabling tiled
matmul in the pinned reference. The reference process exited cleanly.
Its output differed from the retained control at 67,485 values. The
Candle child and term comparison did not run.
[Stopped attempt receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-ffn-down-dot-trace-20261113-bd3094dd/summary.json).

The recheck with `GGML_CPU_TILED_MM=1` reproduced both native 37-row
controls bit for bit. At row 16, channel `1649`, llama.cpp returned
`3c08db15` and Candle returned `3c08db17`. The reconstruction returned
`3c08db14` for llama.cpp, so the controller stopped before accepting a
first differing term. Disassembly of the pinned tiled object shows
separate F32 multiply and add instructions where the reconstruction used
FMA. Both native processes exited and released their Jobs.
[Tiled recheck receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-ffn-down-dot-trace-20261113-bd3094dd-tiled/summary.json).
The offline reconstruction verified the retained 37-row native and
prequantized controls and matched both native values at row 16, channel
`1649`: llama.cpp `3c08db15` and Candle `3c08db17`. All 46 integer block
sums agreed. In block 0 both engines summed to `-142899`. The first F32
contribution was `b8d307b6` for llama.cpp and `b8d307b8` for Candle.
The pinned llama.cpp kernel reduces the integers before applying the
weight and activation scales, with separate F32 multiply and add. Candle
combines the scales first and accumulates eight FMA lanes before reduction.
The offline reconstruction covered that one output coordinate.
[Offline dot receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-ffn-down-dot-offline-20261113-bd3094dd/summary.json).
The isolated Candle Q6_K replay reproduced all 75,776 stock control
values and matched all 75,776 llama.cpp outputs bit for bit on the
recovered Q8_K rows. Row 16, channel `1649` matched `3c08db15`.
All 2,048 values in row 36 matched using the same tiled F32 schedule.
The single attempt passed fresh admission for both build and replay.
Both Jobs exited and released their processes.
[Full projection receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-ffn-down-schedule-20261113-bd3094dd/summary.json).

One offline F32 addition of the corrected down projection and the
retained post-mixer residual matched llama.cpp `layer_out` bit for bit
across all 75,776 values, including all channels in rows 16 and 36.
The complete output digest matched the retained capture.
[Final residual receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer0-output-offline-20261113-bd3094dd/summary.json).

The single-path candidate bound the 37 frozen token IDs and passed
fresh build admission. The build stopped at Rust `E0284`. Four calls
to `pack_weights` omitted the output width const argument for the
ShortConv input and output projections and the FFN gate and up
projections. The compiler Job released its process. The replay and
stage comparisons did not run.
[Stopped build receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer0-single-path-20261113-bd3094dd/summary.json).

With explicit output widths at the four Q4_K packing calls, the next
single admitted replay started from the same 37 frozen IDs and read
the selected layer-0 weights and embedding rows from the frozen GGUF.
All 15 F32 stage outputs and four Q8_K buffers matched the retained
llama.cpp controls. The stock Candle FFN down control reproduced.
The final `layer_out` matched all 75,776 values, including every
channel in rows 16 and 36. Both Jobs exited and released their
processes. This closes the isolated layer-0 comparison.
[Layer-0 replay receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer0-single-path-20261113-bd3094dd-widths/summary.json).
The isolated full-model CPU prefill candidate bound the 37 frozen IDs,
the verified layer-0 control, and the retained pinned llama.cpp prefill
record. Fresh build admission detected Cargo PID 41144 with 43.25 GiB
available RAM and 44.30 GiB commit headroom. The controller stopped
before dispatching a compiler or model process.
[Prefill admission receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-full-cpu-prefill-20261113-bd3094dd/summary.json).

The bound candidate passed fresh admission for the build and one
full-model CPU prefill on the 37 frozen IDs. All 75,776 layer-0 output
values matched bit for bit before the computed tensor entered the stock
Candle layers. All 128,000 final-position Candle logits were finite.
Both engines selected token `207` at position 36. Across the nine
frozen probes, the maximum absolute logit delta was `1.313808` at token
`440`, and RMS was `0.612766`. The replay Job peaked at 15,206,772,736
bytes and both Jobs released their processes. This run covers one
prefill position on one frozen prompt.
[Full CPU prefill receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-full-cpu-prefill-20261113-bd3094dd-recheck/summary.json).
The review recomputed the token and nine deltas from the retained F32
logits and verified the source, binary and Job receipts. The successful
diagnostic used a separate layer-0 implementation and ran stock Candle
from layer 1 through the output head. The production position-2
comparison remains open.

The layer-1 CPU packet `8060130a` at published Candle `72a53981` consumed
its native attempt. All 75,776 layer-0 entry values matched pinned llama.cpp.
The trace stopped at layer-1 weighted RMSNorm, row 0, channel 0
(`0xbcbdce5d` versus `0xbcbdce5a`). All four Jobs released their processes.
[Native terminal](D:/Compact_Evidence/candle/runs/lfm2-moe-layer1-inert-20261113-72a53981/terminal.json).

The offline stock replay reproduced the first mismatch. Candle and llama.cpp
formed identical F32 squares, then accumulated them in F32 and F64,
respectively. Using llama.cpp's RMS with Candle's direct division still left
one ULP at the witness. F64 accumulation and reciprocal multiplication
matched all 37 rows. [Stock trace](D:/Compact_Evidence/candle/runs/lfm2-moe-layer1-rmsnorm-stock-20261113-8060130a/summary.json).
[RMSNorm replay](D:/Compact_Evidence/candle/runs/lfm2-moe-layer1-rmsnorm-f64-reciprocal-20261113-8060130a/summary.json).

The retained offline replays match all ten layer-1 checkpoints, from
weighted RMSNorm through ShortConv, dense FFN and `l_out-1`. The final
F32 residual addition matched all 75,776 values, including rows 16 and 36.
The review checked the retained tensors against all eleven reference files
and recomputed both F32 residual additions. [Final offline checkpoint](D:/Compact_Evidence/candle/runs/lfm2-moe-layer1-output-offline-20261113-8060130a/summary.json).

One composed offline replay at Candle `60ce38d4` verified the exact
`l_out-0` entry and carried its own computed intermediates through all
ten layer-1 checkpoints. All 75,776 `l_out-1` values matched pinned
llama.cpp, including rows 16 and 36. Both admitted Windows Jobs exited
and released their processes. The computed `l_out-1` is the entry control
for the inert layer-2 diagnostic. The recheck confirmed the retained bytes
and terminal receipts. [Composed replay](D:/Compact_Evidence/candle/runs/lfm2-moe-layer1-composed-offline-20261113-60ce38d4/summary.json).

Layer-2 CPU packet `0c99a403` was prepared against `9cc7c853`. It binds the
same 37 frozen IDs, exact entry, 13 weight ranges and 27 typed checkpoints.
Model-free native checks passed for strided attention, ordered integer
experts, invalid contracts, normalization, early stop and the signed KV head
metadata in the frozen header. Six controller tests passed, including
input/source drift and a synthetic admission refusal.
[Preparation proof](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-inert-20261113-9cc7c853/summary.json).

The review superseded `0c99a403` with packet `623dc38e`. It uses
the same two probes, frozen inputs, checkpoint order and limits. The
controller binds the run directory, holds read handles on the verified
runtime files and captures through dispatch, and records a post-child
census even when a launcher receipt is missing. Seven controller tests passed
on synthetic inputs. [Review proof](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-controller-review-20261113-9cc7c853/summary.json).

Packet `623dc38e` is consumed. The first owner census refused admission
while Python and Cargo were active. After those processes cleared, the next
owner census passed and the controller wrote the attempt marker. After model
identity verification, two Python processes blocked reference admission. The
census measured 40.75 GiB available RAM and 41.66 GiB of commit headroom.
The terminal record shows zero dispatches and zero Jobs. Both probes are absent.
[Admission](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-controller-review-20261113-9cc7c853/admission-reference.json)
[Terminal](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-controller-review-20261113-9cc7c853/terminal.json).

Layer-2 successor `0e236955` passed fresh admission and consumed its one CPU
attempt on the 37 frozen IDs. The layer-1 entry and layer-2 weighted RMSNorm
each matched all 75,776 F32 values bit for bit. The first mismatch was the Q4_K
query projection at token 0, head 0, channel 0. Candle produced `0x3c2a4502`
and pinned llama.cpp produced `0x3c2a4506`, a difference of four F32 ULPs.
The reference captured all 27 stages. Candle stopped at `q`, and both
Windows Jobs released their processes. [Trace receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-successor-20261113-4df21393/trace-summary.json).

One offline replay reproduced both engines' 75,776 query values on the
37 matched RMSNorm rows. All 296 Q8_K scales, 75,776 signed quant bytes
and 4,736 sums agreed before the dot. At token 0, head 0, channel 0, the
integer products and scale/min folds agreed. The first F32 difference
appeared after the first subblock pair of block 0. Candle accumulated
scale and minimum contributions per subblock in one F32 accumulator.
llama.cpp folded the integer terms by pair and accumulated the two F32
terms separately. Reconstructing both schedules reproduced the native
witnesses exactly. [Offline replay](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-query-offline-20261113-0e236955/summary.json).

Isolated query packet `0e822a25` built the GEMM/GEMV candidate, then stopped
before replay when four Python processes appeared. Admission showed 50.23 GiB
available RAM and 52.82 GiB commit headroom. The build Job released its
process. [Build and admission receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-query-schedule-20261113-031af7c7/summary.json).

Replay-only packet `97dc98a2` passed fresh admission and ran executable
`0e2e0bb2` once. The stock control and matched Q8_K bytes reproduced. The
isolated schedules matched all 75,776 query values bit for bit, including
73,728 GEMM values in rows 0–35 and all 2,048 GEMV values in row 36. Token 0,
head 0, channel 0 matched `0x3c2a4506`. The Job released the probe and the
post-run census passed. [Replay receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-query-replay-20261113-0e2e0bb2/summary.json).

Layer-2 key replay `cdf58e7d` reused the query schedules with the frozen
512-wide `blk.2.attn_k.weight` slice. All 296 Q8_K blocks reproduced and
all 18,944 key values matched pinned llama.cpp bit for bit. Token 0, head 0,
channel 0 matched `0x3a3453bc`, and all 512 values in the row-36 GEMV tail
matched. Both bounded Jobs released their processes and passed their final
censuses. [Key replay receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-key-replay-20261113-97dc98a2/summary.json).

Layer-2 value packet `27fd6f53` passed admission, then stopped at the
isolated probe build with Rust `E0433`. Its `require` helper uses `io::Error`
where the file imports `IoError`. The compiler Job released its process.
The packet consumed one attempt with zero replay dispatches.
[Value build-stop receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-value-replay-20261114-cdf58e7d/summary.json).

Corrected packet `9b0d5544` ran one offline replay. All 296 Q8_K blocks and
all 18,944 value outputs matched pinned llama.cpp bit for bit, including
token 0, head 0, channel 0 (`0x3c02a987`) and all 512 values in row 36.
Both Jobs released their processes.
[Value replay receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-value-replay-20261114-7432e492/summary.json).

Offline packet `a2f11de3` matched all 75,776 query values and 18,944 key
values after per-head weighted RMSNorm and before RoPE. The isolated
replay used frozen 64-channel weights and epsilon `0x3727c5ac`,
F64 accumulation of F32 squares, and F32 reciprocal and weight
multiplication. Both Jobs exited and released their processes.
[Q/K replay receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-qk-norm-20261114-9b0d5544/summary.json).

RoPE packet `4ddbcf69` stopped before the build because Cargo, rustc and
Python were active. Available RAM and commit headroom cleared the original
floors. The read-only recheck verified all 32 file bindings, the matched
RMSNorm inputs, and zero dispatches.
[RoPE admission receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-qk-rope-20261114-a2f11de3/summary.json).

Successor `a4279483` verified 37 bindings at published `2acafc05`.
The scoped implementation bytes match the prior packet. Cargo,
cargo-clippy and Python blocked admission before the build. RAM and
commit headroom cleared the original floors. The controller recorded zero
dispatches and left the offline attempt unconsumed at that stop.
[Successor receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-qk-rope-20261114-4ddbcf69/summary.json).

Packet `b67d23be` used preliminary admission before sealing and fresh
admission before both dispatches. The isolated build and one offline
replay matched all 75,776 query and 18,944 key RoPE values bit for bit
against pinned llama.cpp. It reused the matched inputs, frozen positions
0–36 and the original NeoX rotary configuration. Both Jobs released
their processes.
[RoPE replay receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-qk-rope-20261114-a4279483/summary.json).

Offline packet `d62aa83b` replayed causal attention on the matched Q/K/V
tensors. The isolated graph reused the pinned GGML CPU libraries, the F32
non-flash cache layout, and 256 padded KV slots. All 75,776 values of
`attn_values.f32` matched bit for bit, including all 2,048 in row 36.
Preliminary admission passed before sealing, and both bounded Jobs passed
their final censuses and released their processes.
This result establishes the reference attention control. Proof of
Candle's native [quantized attention path](../../candle-transformers/src/models/quantized_lfm2.rs)
on these operands remains open. The connected-layer gate must exercise
Candle's own matmuls, scaling, causal mask, softmax and cache handling.
[Attention replay receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-attn-values-20261114-b67d23be/summary.json).

Offline packet `6dbeea9c` matched Q8_K scales, bytes and sums before the
layer-2 Q4_K output dot. The pinned llama.cpp replay and isolated Candle
matched the entire 37-row `mixer.f32`, including the row-36 GEMV tail.
The host was quiet before sealing and each bounded child passed fresh
admission and released its Job.
[Output projection receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-attn-output-20261114-d62aa83b/summary.json).

Offline packet `73fb1e0b` added the matched mixer to the exact layer-1
output in F32 and retained the resulting residual. Its weighted
FFN RMSNorm matched pinned llama.cpp on all 75,776 values, including
row 36. The probe used epsilon `0x3727c5ac`, F64 accumulation of
F32 squares, and separate F32 reciprocal and weight multiplication.
Quiet preflight preceded sealing. Both Jobs passed fresh admission and
released their processes. These matched rows supplied the layer-2 F32
router projection replay in packet `47f523cd`.
[RMSNorm replay receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-ffn-norm-20261114-6dbeea9c/summary.json).

Router packet `a886d883` passed the pre-seal quiet check and reference
build admission at Candle `431342b2`. MSVC built the adapter and its Job
released the process. Python workloads appeared at the post-build quiet
census, so the controller stopped with one dispatch and zero retries.
The attempt is consumed. The read-only recheck verified the retained
reference object's digest, x64 COFF format, `router_reference` symbol,
and link to the successful MSVC Job.
[Stop receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-router-20261114-73fb1e0b/summary.json).

Offline successor `47f523cd` at Candle `67e7cbc0` passed quiet and
resource admission and ran once on the frozen 37 rows. Isolated Candle
and pinned llama.cpp both matched all 2,368 retained pre-sigmoid router
logits bit for bit, including all 64 in row 36. Both Jobs released their
processes and the final census was quiet. The attempt is consumed with zero
model loads and zero retries.
[Comparison](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-router-closeout-20261114-d0075c3b/comparison.json)
and [terminal receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-router-closeout-20261114-d0075c3b/terminal.json).

Offline sigmoid packet `996fe7b0` passed quiet and resource admission.
Candle's F32 CPU sigmoid and pinned llama.cpp `582eab71` reproduced all
2,368 retained `ffn_moe_probs-2` values bit for bit, including all 64
in row 36. The two isolated builds and the single replay released their
Jobs, and the final census was quiet.
[Comparison](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-router-sigmoid-20261114-47f523cd/comparison.json)
and [terminal receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-router-sigmoid-20261114-47f523cd/terminal.json).

Offline packet `17d27ea3` added the frozen F32 expert bias to the 37 matched
probability rows. Candle `broadcast_add` and pinned llama.cpp `ggml_add`
both reproduced all 2,368 retained `ffn_moe_probs_biased-2` values bit
for bit, including all 64 in row 36. Both builds and the single replay
passed fresh admission and released their Jobs. The final census was
quiet.
[Comparison](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-router-bias-20261114-996fe7b0/comparison.json)
and [terminal receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-router-bias-20261114-996fe7b0/terminal.json).

Offline packet `91a27e18` replayed the native CPU top-four selections at
Candle `53de1f7c`. Both engines reproduced all 148 ordered IDs across
37 rows. Row 36 matched `[54, 37, 30, 20]`. All seven quiet and resource
checks passed, and both builds and the replay released their Jobs.
The attempt is consumed with zero retries and zero model loads. Those
ordered IDs supplied the unbiased weight gather in packet `62ab7c06`.
[Comparison](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-router-top4-20261114-17d27ea3/comparison.json)
and [terminal receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-router-top4-20261114-17d27ea3/terminal.json).

Offline gather packet `62ab7c06` initially stopped before sealing when
the census found five Python processes. The earlier stop dispatched zero
children. The read-only review revalidated all 218 bindings. At Candle
`53de1f7c`, fresh admission passed and the original packet ran its one
offline attempt. Candle `gather` and pinned llama.cpp `ggml_get_rows`
matched all 148 unbiased F32 weights in order, including all four in
row 36. All seven admission checks passed and all three Jobs exited
cleanly. The attempt is consumed with zero retries and zero model loads.
[Pre-dispatch stop](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-router-gather-20261114-91a27e18/summary.json),
[comparison](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-router-gather-20261114-91a27e18/comparison.json),
and [terminal receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-router-gather-20261114-91a27e18/terminal.json).

Raw-sum packet `6db2c79c` at Candle `9be1b04d` passed quiet and
resource admission and ran once offline under the original limits.
Pinned GGML reproduced all 37 retained row sums, including row 36
`0x3f9d6164`. Candle's F32 CPU `sum_keepdim` first differed at row 1
with `0x3fb72528` versus GGML's `0x3fb72527`. All three Jobs released
their processes. The attempt is consumed with zero retries and zero
model loads. The follow-up reconstruction matched both saved native outputs
across all 37 rows. The engines differ on 11 rows. The rounded partial
sums first diverge at the fourth addition of row 1.

The local quantized CPU router now widens the reduction to F64 and rounds
before adding the Transformers epsilon. The focused MoE gate passed
12 tests, including an exact regression for the row-1 sum and denominator.
Fresh admission for the maintained-source replay found Cargo, rustc and
Python processes and stopped before sealing with zero dispatches.
[Reconstruction and admission](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-router-reduction-20261114-c01376ed/summary.json).

With the owner's authorization to allow concurrent build and Python
workloads, the maintained CPU replay ran once under the original numeric
limits. All 37 raw sums matched bit for bit, including row 36
`0x3f9d6164`. The first denominator difference was row 0. The common
raw sum was `0x3f87e358`, Candle's epsilon gave `0x3f87e360`, and
GGML's clamp gave `0x3f87e358`. Both outputs matched their respective
formulas. The replay stopped there and both Jobs released their processes.
[Completion receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-router-reduction-20261114-c01376ed/completion.json).
[First-difference receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-router-sum-20261114-62ab7c06/summary.json).

One offline normalized-weight replay used the matched 37×4 inputs and
the maintained Candle CPU reduction. Both engines matched all 148
values under their own formulas, and GGML reproduced all 148 values in
the pinned capture. The denominator formulas explain all 148
cross-engine differences. The largest absolute delta was
`5.066394806e-7`. Both builds and the replay released their Jobs.
[Normalized-weight receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-router-normalized-20261114-c01376ed/summary.json).

The first offline gate/up attempt verified the frozen FFN input, expert IDs
and both Q4_K ranges. It stopped during the harness build with `E0599`.
The harness used the GGUF spelling `Q4_K` for Candle's Rust enum variant
`GgmlDType::Q4K`. The build Job released its process. The attempt is
consumed with zero projection dispatches. Gate/up parity remains open.
[Compile stop](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-expert-gate-up-20261114-c01376ed/summary.json).

The corrected `GgmlDType::Q4K` harness built and ran once with the
original grouping and numeric limits. Before weighting, the first gate
mismatch in maintained dispatch was expert 0, token 28, slot 0,
channel 0. The group had one row under `GenericTiled`. Candle returned
`0x3ca52840` and pinned llama.cpp returned `0x3ca52844`. The attempt
stopped there. Both Jobs released their processes. Up and weighting
remain open. [First-difference receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-expert-gate-up-20261114-q4k/summary.json).

The one-row expert-0 dot trace verified the frozen input and Q4_K
operands and built the pinned GGML adapter. The bounded launcher refused
the Candle build because the same pinned `rustc.exe` was already running.
The reference Job released its process. The attempt is consumed with zero
replay dispatches. The reference object remains bound for the successor.
[Trace stop](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-expert0-gate-dot-20261114-q4k/summary.json).

The successor reused the verified GGML object and froze the expert-0
one-row trace before dispatch. The isolated native quantizers matched
all eight Q8_K scales, 2,048 quantized bytes and 128 block sums. Candle's
quad dot and GGML's direct dot both gave `0x3ca52840`. The direct GGML
dot differed from the retained indexed projection `0x3ca52844`, so the
probe stopped. Both Jobs released their processes. The actual indexed
reference dispatch and its operands still need proof.
[Trace receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-expert0-gate-dot-20261114-resume/summary.json).

The pinned indexed `mul_mat_id` replay used the same frozen 37 FFN rows,
ordered IDs and one-row expert-0 group. Its `CPU_REPACK` buffer selected
`q4_K_8x8` and reproduced `0x3ca52844` at token 28, slot 0, channel 0.
The plain CPU indexed buffer gave `0x3ca52840`, matching Candle's
prior quad dot. Both indexed scratch buffers matched the retained
Q8_K row in all 8 scales, 2,048 bytes and 128 sums. Both Jobs released
their processes. The buffer and kernel choice account for the observed
difference.
[Indexed gate receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-expert0-indexed-20261114-c01376ed/summary.json).

One bounded offline trace reproduced the packed GEMV output `0x3ca52844`
and the plain AVX2 output `0x3ca52840`. The first difference is in the
zero-index block 0 scale and signed minimum contributions. The packed and
plain scale terms are `0x3ce80ddc` and `0x3ce80ddd`. Their minimum terms are
`0xbcdf2299` and `0xbcdf229a`. Both kernels agree on the integer terms.
The packed kernel folds the integer lanes before F32 fused accumulation.
Plain AVX2 accumulates distinct F32 lanes before horizontal reduction.
The combined prefix first differs after block 2 at `0xbc118958` versus
`0xbc11895c`. All 48 native prefix checks passed and both Jobs released
their processes.
[Arithmetic trace receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-expert0-repack-arithmetic-20261114-c01376ed/summary.json).

The isolated Candle replay preserved the 35 expert groups and all 148 ordered
selections on the 37 frozen FFN rows. The pinned indexed path uses
`q4_K_8x8` GEMV for each selected row, including multi-row groups.
Integer folding and separate F32 fused scale and minimum accumulation matched
all 227,328 gate and 227,328 up values bit for bit before weighting. Token 28
expert-0 gate reproduced `0x3ca52844` with the captured Q8_K scales, bytes
and sums. Row 36 matched in all four slots. One offline attempt passed under
the original limits and both Jobs released their processes.
[Expert gate/up receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-expert-gate-up-repack-20261114-c01376ed/summary.json).

The isolated SwiGLU replay applied the verified AVX2/FMA SiLU arithmetic and
Candle's F32 multiply to the matched gate and up tensors. All 227,328 values
matched pinned llama.cpp bit for bit in token, slot and channel order.
Token 28, slot 0, channel 0 gave `0xba6c8f0f`, and all four slots in row 36
matched. Fresh admission passed, both Jobs released their processes, and the
single offline attempt stayed within the original limits.
[Expert SwiGLU receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-expert-swiglu-20261114-c01376ed/summary.json).

The isolated Q6_K expert down replay preserved all 35 groups. The pinned
native quantizer and Candle candidate matched the scales, quant bytes and
sums in all 888 Q8_K blocks before the dot. Experts 30, 37 and 54 used the
indexed tiled schedule. The remaining 32 groups used the plain AVX2 schedule.
All 303,104 unweighted values matched pinned llama.cpp bit for bit, including
all four slots in row 36. Token 28, slot 0, channel 0 gave `0x3a3c36f8`.
The single attempt passed fresh admission under the original limits, and all
four Jobs released their processes.
[Expert down receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-expert-q6k-down-20261114-c01376ed/summary.json).

The isolated expert weighting replay used the existing Candle F32 broadcast
multiply in the original 35 groups. All 303,104 products with the pinned
llama.cpp weights matched the reference. All 303,104 products with Candle's
weights matched an independent F32 multiplication. Changing only the verified
weights accounted for all output differences, with a maximum absolute delta
of `1.043081283569e-7`. Both Jobs released their processes after one offline
attempt under the original limits. The verified Candle tensor is retained
for the sparse expert merge.
[Expert weighting receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-expert-weighting-20261114-c01376ed/summary.json).

The offline sparse expert merge closed the isolated layer-2 sparse FFN
checkpoint. The llama.cpp slot-order replay matched all 75,776 pinned
values. Candle's existing expert-order `index_add` matched 75,776
independent F32 sums with each of the two verified weighted tensors.
Holding the operands fixed, addition order changed the result by at most
`7.4505806e-9`. Holding the order fixed, the verified weight difference
changed it by at most `1.7881393e-7`. Both Jobs exited and released under
the original limits. The Candle sparse FFN output is retained.

The maintained candidate starts with Candle's own causal attention on the
bound Q/K/V operands and positions. The earlier matching attention receipt
used a pinned GGML graph. The merge receipt binds that comparison target and
the layer-1 entry for the later connected layer-2 run.
[Expert merge receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-expert-merge-20261114-c01376ed/summary.json).

An active llama-server stopped the first Candle attention probe before
build, with both memory floors met and zero dispatches.
[Candle attention admission receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-candle-attention-baseline-20261114-c01376ed/summary.json).

The admitted successor matched all 2,048 values at token 0 and stopped
at token 1, head 0, channel 0 (`0x3b7aeaeb` vs `0x3b7aeaed`). Both Jobs
released under the original limits.
[Candle attention replay receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-candle-attention-baseline-20261114-c01376ed-resume/summary.json).

Active compilers blocked the paired trace at `c01376ed` before dispatch.
[Pre-seal refusal](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-attention-first-term-20261114-c01376ed/summary.json). [Prepared trace and admission record](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-attention-first-term-20261114-c01376ed-resume/readiness.json).

The completed [attention witness traces](HISTORY.md#2026-11-15---layer-2-attention-score-and-value-witnesses)
bind the Q·K score arithmetic, connected packet 93cc154a and the channel-8
value accumulation difference.

Packet `96e27811` closed the [isolated value-reduction comparison](HISTORY.md#2026-11-15---layer-2-attention-score-and-value-witnesses).

Connected packet `ab33318c` passed the whole-file hash and admission. One
attempt matched 530,432 values through ten stages, including Candle's
full native attention control. Mixer token 1/channel 0 stopped the run
at `0xbc149902` versus `0xbc149906`. The Job released and temporary view
was removed. The packet is consumed. FFN and residual proof remain open.
[Attempt receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-connected-successor-20261115-ab705d07/connected-attempt-20261115.json).

The paired mixer replay matched Q8_K and all 75,776 retained mixer values
per input in both engines. Across native inputs, 226 scales, two quant bytes
and two sums changed. Token 1 changed scales only, in blocks 1, 5, 6 and 7.
Both engines reproduced `0xbc149902` on Candle input and `0xbc149906` on
llama.cpp input at channel 0. Both Jobs released. The mixer gap follows
the attention inputs on this fixture. Connected FFN and residual remain open.
[Mixer replay receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-mixer-cross-20261115-ab705d07-recheck-121900/cross-input-result.json).

Packet `d6ebb030` verified both mixer controls and the 75,776 retained
llama.cpp residuals. Successor `51f32dae` reused its executable and
limits. Build and Python activity alone no longer blocks this diagnostic.
Whole-model hash and admission passed. The single attempt matched 606,208
values through the mixer and stopped at FFN RMSNorm token 1/channel 0
(`0xbe3e5437` vs `0xbe3e543a`). The Job released. The host was quiet during
the run. The concurrent-workload admission rule passed predicate tests.
[Attempt receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-connected-admission-20261115-67b14101/connected-attempt-20261115.json).

The cross-input replay independently reconstructed both 37 × 2,048 F32
post-mixer residuals from the retained layer-1 output and mixers. The
isolated Candle diagnostic and pinned llama.cpp GGML CPU graph matched
all 75,776 weighted FFN RMSNorm values on each shared residual. At token
1/channel 0, the Candle residual `0xbc914caa` produced `0xbe3e5437` and
the llama.cpp residual `0xbc914cac` produced `0xbe3e543a` in both engines.
This reproduces the connected witness by changing the residual input.
The original limits and revised admission held. One attempt completed
with four Jobs, all released, and no model load.
[Replay result](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-ffn-norm-cross-20261115-67b14101/result.json).
[Terminal receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-ffn-norm-cross-20261115-67b14101/terminal.json).

Packet `e73f4816` passed revised admission and the whole-model hash,
matched 681,984 values through FFN RMSNorm and stopped at router token
1/expert 1. The sole connected attempt released its Job.
[Connected attempt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-connected-norm-20261115-67b14101/connected-attempt-20261115.json).

Both engines matched all 2,368 logits on each shared norm input and
reproduced all 1,723 input-dependent differences. Both Jobs released.
[Router replay](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-router-cross-20261115-67b14101/result.json).

The fixture batch published at `60ce38d4` gives canonical channel-major
weights precedence for square ShortConv kernels. The nonsymmetric fixture
reproduced the old failure and checks prefill and cached decode with exact
expected values.
The 24B checkpoint uses distinct channel and kernel dimensions. New MoE
controls compare weighted top-K-four accumulation on identical quantized
expert weights and compare 127/128/129-token prefill, irregular chunks,
tokenwise execution and their continuation. The RTX 4090 synthetic fixture
now completes decode, rejects a stale continuation after invalid input
and verifies prefill plus decode on recovery.
The earlier Windows CPU fixture gate passed 11 MoE and 10 shared quantized
LFM2 tests. The closeout reran both filters with CUDA enabled and passed
13 MoE and 11 shared quantized LFM2 tests, including CUDA ShortConv history.
The MoE example check and all 15 protocol tests passed.

The packet inventory recheck found that `candle-examples/build.rs` and
its inputs were outside the bound paths. The repair published at `72a53981`
adds the build script, download module and CUDA kernel directory to
future packet inventories. Existing packet receipts remain historical.

The comparator and prefill isolation closeout is recorded in its
[guarded publication receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-comparator-publish-20261111-95df99a0/summary.json).
The receipt binds the verified source change to its commit and remote main.
Production runs, package assembly, installation, activation, release creation and
consumer repinning remain held. Edge runtime/CLI/API acceptance and reviewed adoption stay
open after the supplier gate. Full native/Q8 production, Metal and 131k context
qualification remain outside this delivery.

Active source: the two new model modules, private shared mixers, local example,
synthetic fixtures, reference/qualification tools and their owning records.
Known unsupported native configuration: `conv_bias=true`, non-default RoPE.
The two review findings are closed by the scoped repair. Retained failed
implementation attempts remain in the original run record. Native lower-
precision activation parity is not claimed by the CPU-F32 numerical gate.

---
AI-edited: 2026-11-13 | agent=Codex/root | task=lfm2-moe-layer2-attempt | change=record consumed packet and admission stop before dispatch
AI-edited: 2026-11-13 | agent=Codex/root | task=lfm2-moe-layer2-successor | change=record inert successor and admission-order proof
AI-edited: 2026-11-13 | agent=Codex/root | task=lfm2-moe-layer2-cpu-trace | change=record the exact entry and first query projection mismatch
AI-edited: 2026-11-13 | agent=Codex/root | task=lfm2-moe-layer2-query-offline | change=record exact Q8_K operands and isolate F32 query accumulation
AI-edited: 2026-11-13 | agent=Codex/root | task=lfm2-moe-layer2-query-schedule | change=record build-only progress and the held replay
AI-edited: 2026-11-13 | agent=Codex/root | task=lfm2-moe-layer2-query-replay | change=record isolated query parity and the next key projection gate
AI-edited: 2026-11-13 | agent=Codex/root | task=lfm2-moe-layer2-key-replay | change=record isolated key parity and the next value projection gate
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-value-replay | change=record the value probe compile stop and bounded correction
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-value-recovery | change=record full value projection parity
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-qk-norm | change=record exact pre-RoPE query and key outputs
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-rope-review | change=record the verified pre-dispatch admission stop
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-rope-successor | change=record current source binding and zero dispatches
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-rope-replay | change=record admitted rotary replay and next attention checkpoint
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-attn-values | change=record exact attention values and the output projection handoff
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-attn-output | change=record mixer parity and bounded normalization handoff
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-ffn-norm | change=record weighted FFN RMSNorm parity and router handoff
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-router-review | change=clarify the full-batch F32 router handoff
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-router-work | change=record the verified router inputs and pre-seal quiet-host refusal
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-router-admission-stop | change=record the reference build and post-build admission stop
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-router-successor-review | change=record the verified reference object and successor
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-router-successor | change=record the verified inert router successor and pending comparison
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-router-replay | change=record verified layer-2 F32 router parity and terminal cleanup
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-router-sigmoid | change=record layer-2 router sigmoid parity and terminal cleanup
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-router-sigmoid-review | change=replace the stale router projection pointer after sigmoid review
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-router-bias | change=record the exact expert selection scores and clean terminal result
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-router-top4 | change=record the top-four replay and route the next unbiased gather
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-router-gather | change=record the unbiased gather admission stop and unused packet
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-gather-review | change=record the read-only gather recheck and timestamped quiet census
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-gather-replay | change=record successful gather admission, parity and cleanup after the prior refusal
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-raw-sum | change=record the raw sum stop and keep native Candle attention proof open
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-maintained-reduction | change=record scoped CPU reduction and pending exact replay
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-router-replay | change=record exact raw sums and the classified denominator
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-router-normalized | change=record exact engine-specific normalization and queue expert projections
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-expert-projection | change=record the consumed harness compile stop
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-gate-witness | change=record the first grouped gate mismatch and queue the exact dot trace
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-expert0-gate-dot | change=record the compiler admission stop and retain the reference object
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-expert0-dot-resume | change=record exact Q8_K match and route the indexed reference discrepancy
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-expert0-indexed | change=record native CPU repack witness and matching Q8_K scratch
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-expert0-repack-arithmetic | change=record both native controls and the first rounded terms
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-expert-gate-up-repack | change=record bitwise gate and up parity and terminal cleanup
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-expert-swiglu | change=record exact split SwiGLU parity and terminal cleanup
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-expert-q6k-down | change=record exact indexed quantization and unweighted down parity
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-expert-weighting | change=record both exact product checks and their classified differences
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-expert-merge | change=close the isolated sparse FFN checkpoint and bind the maintained attention inputs
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-candle-attention-baseline | change=record the pre-dispatch attention admission stop
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-candle-attention-resume | change=record the first Candle attention difference and cleanup
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-attention-review | change=clarify the scope of the first attention mismatch trace
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-attention-first-term | change=retain the staged trace and pre-seal compiler refusal
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-closeout | change=record the prepared controller's revision binding
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-attention-qk | change=record the first raw dot difference and bounded cleanup
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-qk-dot | change=classify the native dot schedules and record the rounded prefixes
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-shared-scores | change=record exact downstream witness parity with shared scores
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-connected-prep | change=record native model-free proof and inert connected packet bindings
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-connected-admission | change=record whole-file identity and two pre-dispatch refusals
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-connected-attempt | change=record exact entry through RoPE and the channel-1 attention stop
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-attention-channel1 | change=classify the connected channel-1 attention witness
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-attention-all-shared | change=record full native controls and the first shared-score output stop
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-value-reduction-channel8 | change=record the first fused versus separate accumulation difference
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-value-reduction-all | change=retain the full reduction preparation and admission refusal
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-value-reduction-all | change=compact the attention history while preserving its receipts
AI-edited: 2026-11-15 | agent=Codex/root | task=lfm2-moe-layer2-value-reduction-review | change=record read-only binding and admission recheck
AI-edited: 2026-11-15 | agent=Codex/root | task=lfm2-moe-layer2-value-reduction-work | change=record both builds and terminal admission refusal
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-replay-only-prep | change=record verified inert packet and original build identity
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-replay-closeout | change=bind successor across the documentation checkpoint
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-value-replay | change=record full bitwise comparisons and cleanup
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-connected-prep | change=record connected successor and retained control
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-connected-attempt | change=record exact attention and terminal mixer stop
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-mixer-replay | change=record same-input parity and pre-dot cross-input differences
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-ffn-norm-stop | change=record packet-local admission and first downstream mismatch
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-ffn-norm-cross | change=record both residual controls and exact matched-input RMSNorm
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-connected-norm-prep | change=record the sealed connected packet and zero dispatches
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-connected-router-stop | change=record the admitted run and first raw router logit mismatch
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-router-cross | change=record exact same-input router projections and the explained witness
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-connected-router-prep | change=record the sealed router successor and zero probe launches
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-connected-router-review | change=record the inert recheck and held execution gate
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-connected-sigmoid-stop | change=record the consumed router packet and the first sigmoid mismatch
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-sigmoid-cross | change=record the exact same-input sigmoid controls and explained witness
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-sigmoid-dropin | change=lead with the consumed checkpoint and inert successor; archive preparation detail
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-connected-bias-stop | change=record the consumed sigmoid successor and selection score witness
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-selection-bias-cross | change=record exact same-input bias addition and the explained witness
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-bias-recheck | change=record the read-only rank and unbiased gather witnesses
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-connected-up-seal | change=record the sealed gate-plus-up packet and zero dispatches
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-connected-up-review | change=record the inert recheck and downstream input-lane finding
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-swiglu-cross | change=record both verified SwiGLU controls and cleanup
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-down-cross | change=record both expert-down controls and cleanup
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-connected-down-seal | change=record the verified inert SwiGLU and down successor
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-down-seal-review | change=record the inert recheck and remaining execution grant
