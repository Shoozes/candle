# LFM2-MoE current state

The November 10 capability was published as Candle
`95df99a0bda2f6ac64dbeb8475850f826bd92ca9` from base `5f094821`.
The first Q4 parity attempt stopped during CPU comparison. The comparator
repair exposes a remaining CPU numerical disagreement. Edge qualification and
adoption follow supplier parity. D1 stays closed.

The review reproduced an acceptance gap in the comparator at `42deff05`.
A reference record with a wrong same-length forced continuation could pass.
The comparator requires the exact frozen schedule from both engines. The
archived CPU result still reproduces at position 2 (`1113` vs `6986`) using the
comparator from `42deff05`. A successor packet needs a newly bound reference
binary. The 14 comparator tests and pinned VS 18 probe syntax check passed.

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

The existing official Q4_K_M file remains at `C:/llamacpp/models/`, unchanged and
unloaded. Header inspection admits all 428 tensors and estimates 16,035,996,672
bytes at context 12,288/chunk 128. That is an estimate, not observed peak memory
or runtime acceptance. The pinned llama.cpp CPU probe is built separately from
`582eab71`; the installed `9df0b73c` runtime is untouched. Its no-model version
probe passed and the bounded build Job released all children.

Run record: `D:/Compact_Evidence/candle/runs/lfm2-moe-20261110-5f094821/summary.json`.
[TODO](TODO.md) owns remaining Done When gates. [QUALIFICATION](QUALIFICATION.md)
freezes two prompts/eight teacher-forced positions before testing. The prior inert
packet `1e4449ab` at `inert-packet-final/packet.json` remains preserved as historical
preparation. It no longer matches the reviewed source and is not admissible.

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

The model-free artifact refresh is complete. A fresh CUDA-enabled release binary
was built locked/offline at
`D:/CodexStorage/build/candle-lfm2-moe-refresh-20261110-5f094821/release/examples/lfm2-moe.exe`,
SHA-256 `7ee3969b125feffd96d1df2b2b78f767dafca78cf4e978c73fb6f34a3e8a82e3`.
All 788 build-input hashes stayed stable. Header inspection and four exact
tokenizer cases passed without loading weight payloads, and the retained CPU
reference reported its pinned revision. The original model's full size/hash was
freshly reverified in place. The build emitted an LNK4098 library-conflict warning;
the successful build and no-model checks do not prove production execution.

The completed refresh's inert packet is
`D:/Compact_Evidence/candle/runs/lfm2-moe-artifact-refresh-20261110-5f094821/packet/packet.json`.
Its owning run's `summary.json`, `build-binding.json` and
`binding-verification.json` bind the binary, stable build inputs, finalized
candidate inventory and all four `--inputs`/`--case` comparisons. Original limits,
old packet/binaries, frozen inputs and consumer pins are preserved. All bounded
build/no-model children exited and released their Jobs. Production dispatches
remain zero. [HISTORY](HISTORY.md) owns the completed refresh details. This
`1717f529` packet remains immutable preparation for its pre-publication source.

The source closeout's commit, exact clean-commit native gate and confirmed remote
result belong to
`D:/Compact_Evidence/candle/runs/lfm2-moe-publish-20261110-5f094821/summary.json`.
Only the guarded helper's confirmed receipt establishes publication. Its
`packet/packet.json` rebinds preparation to the final committed source inventory
without rebuilding unchanged Rust inputs or changing the binary/model identity.
Its `binding-verification.json` records the source bridge and original limits.
No historical packet or binary is replaced.

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
The [next task](TODO.md) is query and key RoPE on the matched outputs.
[Q/K replay receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-qk-norm-20261114-9b0d5544/summary.json).

RoPE packet `4ddbcf69` stopped before the build because Cargo, rustc and
Python were active. Available RAM and commit headroom cleared the original
floors. The read-only recheck verified all 32 file bindings, the matched
RMSNorm inputs, and zero dispatches. The [next task](TODO.md) uses a fresh
run directory and retains the original admission and terminal receipts.
[RoPE admission receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-qk-rope-20261114-a2f11de3/summary.json).

The position-2 teacher-forced failure and production qualification remain
open. Standalone LFM3 text proof and an LTS support contract remain later work.

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
