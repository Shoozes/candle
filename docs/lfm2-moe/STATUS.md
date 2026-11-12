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
respond to the dot switch on at least one input. This isolates the ShortConv
output projection contribution. The next task in TODO separates the two
F32 operands that formed the input. Production parity remains open.

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
AI-edited: 2026-11-12 | agent=Codex/root | task=lfm2-moe-shortconv-out-replay | change=record the native matched-operand result and downstream scale effects
