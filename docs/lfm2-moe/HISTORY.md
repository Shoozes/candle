# LFM2-MoE completed work

## 2026-11-18 - Candle frozen CPU and RTX 4090 qualification

Shared-input replays proved the Q4_K expert down dispatch and the single-row
V projection, attention value and router folds. The model retains its
configured mixture epsilon and expert-ID accumulation order. CPU packet
`c8157131` and RTX 4090 packet `4d2b97b1` matched all 32 frozen selections
against pinned CPU on one executable. All four Windows Jobs released.

| Lane | Unicode | Arithmetic | Maximum sampled absolute logit delta |
| --- | ---: | ---: | --- |
| Windows CPU | 8/8 | 8/8 | 0.602364 Unicode, 1.361142 arithmetic |
| RTX 4090 | 8/8 | 8/8 | 0.601479 Unicode, 1.384670 arithmetic |

The audit contains both historical pinned CPU Job receipts, the four
comparison and Job receipts, resource peaks and post-exit observations.
[Qualification audit](D:/Compact_Evidence/candle/runs/lfm2-moe-arithmetic-mixture-ablation-20261118-remaining/qualification-audit.json) ·
[Harness pitfalls](../lfm2-vl/FAILURE_LOG.md#f-0057-a-copied-comparator-count-consumed-a-diagnostic-attempt).

## 2026-11-18 - Ordered CUDA attention and indexed experts

Equal frozen Q/K/V exposed independent score, softmax and value drift in
stock CUDA. The compiled ordered attention matched all frozen values and
restored the layer-2 residual. With that input, wide routing accumulation
matched every raw sum. Indexed Q4_K and Q6_K with ordered SiLU matched every
gate, up, SwiGLU and down value on 34 Unicode and 35 arithmetic expert groups.
The same-binary packet `5568f52b` passed 8/8 Unicode positions, including
`10966` at position 2, and matched 7/8 arithmetic positions. Both Jobs released.
On shared arithmetic prefill operands, the earliest remaining change was
the mixture denominator. The pinned clamp and slot-order controls reproduce
the pinned sparse FFN. Subsequent full-model controls retained the original
arithmetic mismatch and isolated the next layer to inspect.
[Four-control audit](D:/Compact_Evidence/candle/runs/lfm2-moe-arithmetic-mixture-ablation-20261118-remaining/ablation-audit.json).
[Operator proof](D:/Compact_Evidence/candle/runs/lfm2-moe-ordered-attention-repair-20261118/crossed-cases-proof.json) ·
[Terminal audit](D:/Compact_Evidence/candle/runs/lfm2-moe-ordered-attention-repair-20261118/production/terminal-audit.json).

## 2026-11-18 - Unicode position 2 and single-row dense Q6_K

On the frozen Unicode prefix and forced IDs `597, 4695`, pinned CPU
reported logits 26.781305 and 26.956537 for IDs `355` and `10966`. The
RTX 4090 reported 26.939594 and 26.852806. Both output heads agreed on
each shared hidden state. The pinned diagnostic reproduced all recorded
logits and captured identical layer-0 operands through SwiGLU. Its
wrapper expected an FFN down callback absent from the bias-free graph,
so the separate operator audit verified the named operands and final
layer residual against the retained reference.

The compiled single-row Q6_K fold reproduced the pinned layer-0 residual
bit for bit on both devices. The old fold differed in 1,785 and 1,736
values on the two forced steps, and layer 1 amplified the difference
before matching bitwise on common inputs. Multirow dense FFN retains the
verified tiled fold. A shared layer-2 route and probability replay found
eight divergent stock F32 prefill sums affecting 27 normalized weights.
Wide accumulation matched all sums. The two decode rows matched either fold.

Packet `f3d564fa` audited all eight Unicode positions and matched seven.
The position-2 ranking remained unresolved. Peak sampled VRAM was
15,282 MiB, peak Job memory was 15,946,031,104 bytes and the Job released.
[Operator proof](D:/Compact_Evidence/candle/runs/lfm2-moe-unicode-position2-20261118/operator-proof.json) ·
[Terminal audit](D:/Compact_Evidence/candle/runs/lfm2-moe-unicode-position2-20261118/production/terminal-audit.json).

## 2026-11-18 - Layer-2 CUDA Q/K/V projection

The bounded replay reproduced the earlier stock projection captures bitwise
and confirmed identical block-normalized inputs for the two devices. On
the shared 37 prefill rows and two single-row decode inputs, Q8_K activation
packing with the existing reference-order folds matched the CPU Q4_K Q/K
and Q6_K V projections bitwise across 119,808 outputs. Stock CUDA differed
by up to 0.00114620. The isolated launcher exited successfully, released its
PID and sampled 1,297,285,120 bytes of peak Job memory.
[Replay](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-qkv-dropin-20261118/summary.json)
and [Job receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-qkv-dropin-20261118/replay.job.json).

Packet `515de8df` consumed one arithmetic attempt with seven exact selections.
Position 2 was `6986` on CUDA versus `1113` on pinned CPU. Peak sampled device
memory was 15,324 MiB and the Job released its PID.
[Terminal audit](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-qkv-dropin-20261118/production/terminal-audit.json).

On the repaired Q/K/V inputs, ordered CUDA Q/K RMSNorm matched every CPU
value. Feeding the CPU context directly into the CUDA output projection
reproduced the stock mismatch and verified bitwise equality with the Q8_K
path across 79,872 outputs. Common RoPE tables reduced the rotary gap to
4.77e-7; that remaining difference did not justify changing table storage.
[Shared-input controls](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-qkv-dropin-20261118/shared-rope-output/summary.json).

Packet `d37538ce` then checked the integrated attention repairs. All eight
positions produced finite probe values; seven selections matched and position 2
still differed. Peak sampled device memory was 15,269 MiB; the Job released.
[Terminal audit](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-qkv-dropin-20261118/attention-candidate/production/terminal-audit.json).

Connected trace packet `1bec28f3` verified its corrected CUDA build and pinned
CPU reference. First admission stopped before either lane at 39.45 GiB available
RAM. [Admission receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-qkv-dropin-20261118/connected/admission-first-refusal.json).

## 2026-11-18 - CUDA router sigmoid and weight sum

With the same layer-2 logits and held route, stock CUDA differed at
540 of 2,496 sigmoid values and 30 of 156 selected probabilities.
The F32 exponent differed at 766 values. The reciprocal matched on
shared denominators. Evaluating the CUDA exponent in F64 and narrowing
before the F32 add and reciprocal reproduced every retained probability
bitwise. The integrated GGUF route selected the exact reference IDs.

On the same selected probabilities, the CUDA F32 reduction differed
on 11 of 39 rows and F64 reduction and rounding matched all 39.
Sharing numerator and denominator reproduced the division bitwise.
The sigmoid repair with the stock sum still differed at 40 normalized
weights. Both bounded Jobs released.
[Replay](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-sigmoid-cuda-20261118/replay-summary.json).

Exact-grant packet `2f1ea476` passed eight of eight RTX 4090 arithmetic
positions, including `1113` at position 2. All 56 sampled deltas were
finite, with maximum 1.15764 and RMS 0.38331. Sampled device use peaked
at 15,356 MiB, and the Job released its PID.
[Arithmetic audit](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-sigmoid-cuda-20261118/production-verified/terminal-audit.json).

The first Unicode packet `f1c9e45a` stopped before launch at 35.54 GiB
available RAM and 37.42 GiB commit headroom.
[Refusal receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-sigmoid-unicode-20261118/first-admission-receipt.json).

Exact-grant Unicode packet `d12d407f` used 34 GiB RAM, 36 GiB commit,
18 GiB free VRAM and the 32 GiB Job cap. It reused the arithmetic
candidate executable SHA-256
`593bbc72db998125495065cb970a48c70355bbedabb90c3f6b760bbf7736c9b5`.
The retained Candle CPU Unicode comparison against pinned CPU passed
eight of eight. On the RTX 4090, seven of eight selections matched.
Position 2 chose `355` versus pinned CPU's `10966` after forced IDs
`597, 4695`. All 56 sampled logits were finite. Peak sampled GPU use was
15,210 MiB. The Job released and the post-run scan showed 23,263 MiB free.
[Unicode audit](D:/Compact_Evidence/candle/runs/lfm2-moe-sigmoid-unicode-ram34-commit36-20261118/terminal-audit.json).

## 2026-11-18 - Sparse FFN router and expert 37

On the same normalized residual and router weights, stock CUDA changed
1,901 of 2,496 F32 router logits. The peak difference was 1.91e-6. Expert
IDs stayed identical. Sharing CPU logits reduced the prefill weight
difference from 115 to 59 values. On the shared logits and route, the
CUDA sigmoid changed 30 of 156 selected probabilities. Using the same
probabilities isolated 11 raw-sum differences, all in prefill. Division
matched bitwise with the same numerator and denominator. The first
repair ports the CPU eight-lane router dot schedule to CUDA.
[Mixture controls](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-sparse-20261118/mixture.log).

Expert 37 had sixteen prefill rows and one first-decode row. On the
held CPU route and rows the explicit Q8_K pack matched 39,712 bytes,
including scales, quants and partial sums. The stock gate/up prefill
peaks were 0.00830/0.00729. The tiled fold reduced both to 1.79e-7.
Replaying the whole-block indexed fold reproduced all 52,224 gate/up
values bitwise. No expert projection dispatch changed in this batch.
[Isolation](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-sparse-20261118/isolation.json).

The compiled CUDA router matched all 2,496 retained logits bitwise.
Exact-grant packet `7521a633` checked all eight arithmetic positions
under the 36 GiB RAM, 38 GiB commit, 18 GiB VRAM and 32 GiB Job limits.
Seven selections matched. Position 2 remained `6986` versus `1113`.
Its largest sampled logit delta fell from 0.71215 to 0.33599. Device use
peaked at 15,264 MiB and the Job released its PID.
[Kernel proof](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-sparse-20261118/router-proof/summary.json) ·
[Terminal audit](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-sparse-20261118/production-verified/terminal-audit.json).

## 2026-11-18 - Post-SiLU crossed layer 2

Both devices produced identical layer-1 outputs on the retained prefill and
two decode positions. Q/K/V and ordered Q/K normalization matched exactly.
Stock CUDA RoPE differed by up to 3.22e-6 in Q and 2.72e-6 in K. Matching
the CPU tables reduced the error; rounding both products before the
add/subtract removed it and matched the complete K and V caches.

On crossed inputs, the prefill attention residual still differed by up to
9.17e-6. Feeding the same residual to both sparse FFNs gave bitwise FFN norm
and identical expert IDs. Router logits and weights differed slightly.
Expert 37 gate and up projections differed by up to 0.00830 and 0.00729,
and the merged FFN difference was 0.0009764 in prefill. The first repair
is MoE-specific RoPE; those downstream differences remain available for
the next causal replay. [Crossed replay](D:/Compact_Evidence/candle/runs/lfm2-moe-post-silu-layer2-20261118/crossed-summary.json).

The compiled RoPE kernel matched the CPU Q/K and both K/V caches
bitwise at positions 0, 37, and 38. Both decode appends preserved their
prefixes and all bounded replay Jobs released.
[Kernel proof](D:/Compact_Evidence/candle/runs/lfm2-moe-post-silu-layer2-20261118/kernel-summary.json).
Packet `4e3addd8` checked all eight frozen positions. Seven matched, with
position 2 still at `6986` versus `1113`. Its largest sampled position-2
logit delta was 0.71215, down from 1.10894 in `d744a493`. Sampled GPU
use peaked at 15,173 MiB and the Job released.
[Terminal audit](D:/Compact_Evidence/candle/runs/lfm2-moe-post-silu-layer2-20261118/production/terminal-audit.json).

## 2026-11-18 - Connected cache and dense SiLU

RAM-only successor `7c36e274` stopped before launch at 39.64 GiB of commit
headroom. Exact-grant successor `0d24096f` used 36 GiB available RAM,
38 GiB commit, 18 GiB VRAM and the original 32 GiB Job cap. Both lanes
reproduced their retained endpoints and released their Jobs in 32.4 seconds;
peak sampled device memory was 15,110 MiB. Position-2 logits for
`1113`, `6986`, `5141` were CPU 25.978117 / 25.670561 / 25.225178 and
CUDA 25.606670 / 25.699162 / 25.007568.
[Trace](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-qkv-dropin-20261118/connected-ram36-commit38/position2-comparison.json).

Position-2 RMS drift grew from 3.62e-9 after layer 1 to 3.77e-5 after
layer 2. Reusing the actual captured prefix showed inherited K/V error:
the layer-2 attention residual gap was 1.36e-5 RMS on connected inputs
and 1.59e-9 on common CPU inputs.
[Layer-2 replay](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-qkv-dropin-20261118/connected-layer2-attention/summary.json).

The crossed layer-1 replay reproduced both prefill outputs bitwise.
At row 7, the layer-0 input difference first grew at the ShortConv input
projection. The Q8_K fold simulation reproduced 67 versus 66 at block 3,
lane 5 (scaled 66.500015 versus 66.499992). Layer-0 controls matched
both prefill endpoints bitwise and identified dense SiLU as the first
difference on a common gate. Replacing only that operation with the
CPU result made the entire layer-0 prefill bitwise equal and removed
layer 1's projection gap. The diagnostic substitution used CPU execution;
the maintained repair ports the arithmetic onto CUDA.
[Layer-1 replay](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-qkv-dropin-20261118/connected-layer1-prefill/summary.json) ·
[Q8_K witness](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-qkv-dropin-20261118/connected-layer1-prefill/q8-k-activation-witness.json) ·
[SiLU substitution](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-qkv-dropin-20261118/connected-layer1-reference-silu/summary.json).

The production CUDA SiLU kernel matched 918,528 retained gate outputs
bitwise. Exact-grant packet `d744a493` then verified all eight arithmetic
positions under 36 GiB available RAM, 38 GiB commit, 18 GiB VRAM and the 32 GiB
Job cap. Seven selections matched. Position 2 remained `6986` versus
`1113`; the Job released its PID and device use peaked at 15,247 MiB.
[Terminal audit](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-qkv-dropin-20261118/silu-candidate/production/terminal-audit.json).

## 2026-11-17 - Crossed layer-1 cache and ordered CUDA RMSNorm

The crossed replay fed both repaired layer-0 output streams into
layer 1. On shared input the first difference was block RMSNorm.
The ordered CUDA implementation matched CPU bitwise on all 78 crossed
rows and reproduced the layer-1 ShortConv cache and mixer residual
through prefill and both forced decode steps on shared inputs.
[Repair proof](C:/Compact_Evidence/candle/runs/lfm2-moe-layer1-crossed-20261117/layer1-rmsnorm-proof.json).

Layer 2 is the first attention block. Both K/V cache updates preserved
all previous entries. At position 2, substituting identical Q/K/V
shrunk the CPU/CUDA context gap from 0.00040295 to 1.49e-8. The Q/K/V
projections already differed on identical block-normalized inputs.
[Attention and cache proof](C:/Compact_Evidence/candle/runs/lfm2-moe-layer1-crossed-20261117/attention/attention-proof.json).

Exact-grant packet `128a69e71dd951940a2d74ef41fe908cd998051934852547db00897a47b57916`
passed admission with 45.36 GiB of available RAM, 46.16 GiB of commit
headroom, and 23,253 MiB free on the RTX 4090. It recorded 56 finite
probe logits across eight positions. Seven selections matched, with
position 2 selecting `6986` on CUDA and `1113` on pinned CPU. The
Job released its PID. Sampled device use peaked at 15,257 MiB.
[All-eight terminal audit](C:/Compact_Evidence/candle/runs/lfm2-moe-layer1-crossed-20261117/production/terminal-audit.json).

## 2026-11-17 - Dense Q6_K FFN down on shared SwiGLU rows

On local baseline `a6dc2271`, the D4 control reproduced the retained
CUDA down projection bitwise on 37 shared CPU SwiGLU rows. The largest
CPU/CUDA difference was 0.00079674. An independent calculation verified
all 1,702 Q8_K blocks. Fixing the F64 reduction while changing activations
isolated the activation-format effect. Holding the Q8_K bytes fixed
while fusing the scale and fold changed 49,681 output values. The
separately rounded tiled fold and the rebuilt Candle CUDA dispatch
each matched all 75,776 CPU values bitwise.
[Replay evidence](D:/Compact_Evidence/candle/runs/lfm2-moe-dense-q6k-down-20261117/dense-q6k-down-proof.json).

The isolated layer-0 replay carried its ShortConv cache from the
37-token prefill through the two forced decode tokens. At decode
position 2 the largest CUDA layer-0 output gap against pinned CPU
fell from 0.00088233 to 1.49e-8. The isolated CPU decode control also
differed by up to 1.49e-8. Prefill reproduced pinned CPU bitwise on
CPU, with a maximum CUDA difference of 0.00011072 on native inputs.
[Layer-0 replay](D:/Compact_Evidence/candle/runs/lfm2-moe-dense-q6k-down-20261117/layer0/layer0-proof.json).

Exact-grant packet `293dbe4c47910d8dac2ed04638e21399513babf1009a5e08a28665b41bb82dc2`
passed resource admission with Cargo, rustc, and Python observed.
It recorded eight positions and 56 finite logit values. Seven top
selections matched. Position 2 remained `6986` on RTX 4090 versus
`1113` on pinned CPU. The Windows Job released its PID. Sampled peak
total GPU use was 15,310 MiB.
[Terminal audit](D:/Compact_Evidence/candle/runs/lfm2-moe-dense-q6k-down-20261117/production/terminal-audit.json).

## 2026-11-17 - Dense Q4_K FFN gate and up on shared inputs

On local baseline `a6dc2271`, both CUDA DS4 controls reproduced the
retained gate and up projections bitwise on the same 37 CPU-normalized
rows. The CPU/CUDA maximum difference was 0.00429238 for gate and
0.00302982 for up. An independent calculation verified all 296 Q8_K
blocks, including their scales and group sums. A fixed F64 reduction
showed the activation format changed each selected dot. Forcing one
fold for all rows on unchanged Q8_K bytes changed output bits in the
grouped rows or in the tail. The correct schedule matched all 435,712
values per projection. Dense gate and up now share ShortConv's Q8_K
dispatch.
[Replay and controls](D:/Compact_Evidence/candle/runs/lfm2-moe-dense-gate-up-20261117/dense-gate-up-proof.json).

The rebuilt projection replay still found a maximum Q6_K down
difference of 0.00079674 on bitwise identical CPU SwiGLU inputs.
[Shared SwiGLU control](D:/Compact_Evidence/candle/runs/lfm2-moe-dense-gate-up-20261117/shared-swiglu/shared-swiglu-proof.json).

Exact-grant packet `9de6a1be470730121fc8e16c22ac304cfb1d8c80afc0c0838508d90640b75812`
passed resource-based admission with Cargo, rustc, and Python present.
The candidate recorded eight positions and 56 finite logits. Seven
top selections matched. Position 2 remained `6986` on CUDA versus
`1113` on pinned CPU. The Windows Job released its PID. Sampled peak
total GPU memory was 15,581 MiB.
[Terminal audit](D:/Compact_Evidence/candle/runs/lfm2-moe-dense-gate-up-20261117/production/terminal-audit.json).

## 2026-11-17 - Corrected CUDA position-2 trace

Packet `d786c8e7` ran the pinned llama.cpp CPU reference followed by
the fast-DS4-corrected RTX 4090 on the 37-token prompt with forced
decode inputs `49, 1252`. Both Jobs exited cleanly and released their
PIDs. Both lanes reproduced their retained top selections and seven
original logits at all three observations. The new position-2 logits were:

| Token | Pinned CPU | Corrected RTX 4090 |
| --- | ---: | ---: |
| 1113 | 25.978117 | 25.484024 |
| 6986 | 25.670561 | 26.169796 |
| 5141 | 25.225178 | 25.133598 |

Using the pinned CPU head on the CUDA final-normalized row kept `6986`
ahead of `1113`. Using the CUDA head on the CPU row kept `1113` ahead.
The embeddings matched bitwise at all three checkpoints, and the
first captured divergence was layer 0 at each checkpoint.
[Paired checkpoint comparison](D:/Compact_Evidence/candle/runs/lfm2-moe-corrected-position2-20261117/position2-comparison.json).

On the common retained layer-0 RMSNorm operands, the existing DS4
Q8_1 input projection had a 0.01320948 maximum absolute gap. The
CUDA Q8_K activation and Q4_K group/tail dot schedule reproduced all
227,328 CPU projection values bitwise. The MoE ShortConv path now selects
that schedule for Q4_K input and output projections on CUDA.
[Shared-input proof](D:/Compact_Evidence/candle/runs/lfm2-moe-corrected-position2-20261117/candidate/q4-q8-final-proof.json).

The rebuilt CUDA projection replay matched the CPU capture bitwise
through the layer-0 ShortConv residual on all 37 shared prefill rows.
The first remaining captured difference in that replay was the
post-FFN layer output, with a maximum gap of 0.00097196.
[Rebuilt layer-0 replay](D:/Compact_Evidence/candle/runs/lfm2-moe-corrected-position2-20261117/candidate/repaired-layer0-proof.json).

The crossed FFN replay substituted the CPU RMSNorm row on the same
residual. The CUDA gate, up, down, and layer output stayed bitwise
identical to the CUDA-normalized arm. The first substantial common-input
FFN gap was in the Q4_K gate projection (maximum 0.00429238); up differed
by up to 0.00302982. The native release build, targeted core regression,
four-crate CPU check, and MoE suite (19 passed, 2 ignored) passed.
[Dense FFN replay](D:/Compact_Evidence/candle/runs/lfm2-moe-corrected-position2-20261117/candidate/dense-ffn/dense-ffn-proof.json).

Qualification packet `094fb1364b84d5e21e36b2ed07d2440b42eeb5caada55007a94f99d2b8303851`
ran the rebuilt candidate once under resource-based admission. It recorded
all eight positions with 56 finite sampled logits and seven exact top
selections. At position 2 the CUDA winner was still `6986` versus
the CPU reference's `1113`. The Windows Job released its PID, and
peak sampled total GPU use was 15,672 MiB.
[Terminal audit](D:/Compact_Evidence/candle/runs/lfm2-moe-corrected-position2-20261117/candidate/production/terminal-audit.json).

## 2026-11-17 - Fast Q4_K MMQ DS4 replay

On local baseline `a6dc2271`, the native RTX 4090 driver replayed the
stock fast MMQ DS4 quantizer and Q4_K projection on the retained 37
shared RMSNorm rows and identical packed weights. All 227,328 F32
projection outputs matched stock bitwise. The Q4_K-only DS4 launcher
then used the activation scale times the signed quantized-byte sum.
Every quantized byte and FP16 scale matched stock, and all 2,368 sums
matched an independent calculation. The maximum CPU/CUDA projection
gap fell from 0.03963518 to 0.01320948, and RMS fell from 0.00286023 to
0.00159023.
[Kernel proof](D:/Compact_Evidence/candle/runs/lfm2-moe-layer0-shortconv-crossed-20261117/fast-ds4/ds4-proof.json).

The rebuilt Candle dispatch reproduced the corrected projection bitwise.
On the shared input, maximum and RMS gaps against CPU decreased at every
captured step from `in_proj` through `layer_out`. The layer output maximum
fell from 0.00317325 to 0.00242137. The replay Job released its PID.
The native CUDA release build, targeted Q8_1 test, MoE fixture suite
(18 passed, 2 ignored), and locked offline four-crate CPU check passed.
[Downstream proof](D:/Compact_Evidence/candle/runs/lfm2-moe-layer0-shortconv-crossed-20261117/fast-ds4/downstream-proof.json).

Exact grant `7e69189723e3ef9d748daad5092226b8cc1e18a21c063e445b254a59f8f3b934`
authorized one RTX 4090 arithmetic execution. Admission verified the full
GGUF SHA-256 and observed Cargo and rustc. Physical RAM available was
44.90 GiB, commit headroom was 44.93 GiB, and GPU free memory was
23,072 MiB. The Job exited with code 0 and released its PID. Peak Job
memory was 15,989,567,488 bytes and peak sampled total GPU use was
15,489 MiB.

The audit verified all eight selections and 56 finite sampled logits.
Seven selections matched. The frozen comparator failed at zero-based
position 2, where the corrected CUDA run selected `6986`, stock CUDA
selected `5141`, and pinned llama.cpp CPU selected `1113`. This
consumed the one permitted attempt. The frozen logit probes exclude
all three competing token IDs.
[Terminal audit](D:/Compact_Evidence/candle/runs/lfm2-moe-layer0-shortconv-crossed-20261117/fast-ds4/production-verified/terminal-audit.json).

## 2026-11-17 - Resource-gated RTX 4090 arithmetic replay

Packet `d7c0cdb3` tested the corrected CUDA fallback executable on
all eight frozen arithmetic positions. Model-free admission checks
admitted Cargo, rustc and Python and refused competing model or
other build workloads and low RAM, commit or VRAM. The real host
passed admission after cargo-tauri exited. The whole GGUF SHA-256
verified and the Windows Job exited with code 0 and released its PID.
Sampled peak total device memory was 15,266 MiB.

Seven of eight selections matched pinned llama.cpp CPU. At position 2,
the candidate selected `5141` and the reference selected `1113`.
All eight selections and all 56 sampled logits equalled the previous
CUDA run. The contiguous 37-row F32 Q4_K prefill is eligible for
`fast_mmq::try_fwd`, which precedes the corrected fallback. Its
`mmq_quantize.cu` DS4 path computes the minimum term from the original
float sum. The next bounded check should reproduce and isolate the
fast MMQ behavior on the retained operands.
[Terminal audit](D:/Compact_Evidence/candle/runs/lfm2-moe-layer0-shortconv-crossed-20261117/resource-gated-arithmetic/terminal-audit.json) ·
[Stock comparison](D:/Compact_Evidence/candle/runs/lfm2-moe-layer0-shortconv-crossed-20261117/resource-gated-arithmetic/stock-candidate-comparison.json).

## 2026-11-17 - Crossed ShortConv replay and Q4_K CUDA sum correction

The four-arm replay used the same GGUF ranges and exchanged the two
retained RMSNorm outputs. Both native arms reproduced all 75,776 values
of their captured layer-0 outputs bitwise. Swapping in CPU RMSNorm
left the CUDA ShortConv projection output and layer output bitwise
unchanged. On common normalized input the first backend divergence
was Q4_K ShortConv input projection. Its maximum delta was
`0.0396351814` and its RMS delta was `0.0028602261`.

The Q4_K CUDA matrix-matrix minimum term used the raw F32 input sum
from Q8_1 metadata. On the 37 frozen rows the isolated corrected
kernel matched the scaled quantized sum for all 2,368 activation
blocks. Quantized values and scales matched the direct baseline.
The maximum CPU/CUDA projection delta in the directly built pair
fell from `0.04009461` to `0.01313008`, with RMS falling from
`0.00285666` to `0.00158600`. The directly built baseline is an
independent kernel control and differs from the retained Candle CUDA
projection by up to `0.00085629`. All bounded replay Jobs released.
[Crossed replay](D:/Compact_Evidence/candle/runs/lfm2-moe-layer0-shortconv-crossed-20261117/summary.json) ·
[Sum proof](D:/Compact_Evidence/candle/runs/lfm2-moe-layer0-shortconv-crossed-20261117/quantized-sum-proof.json).

## 2026-11-17 - Completed CPU and RTX 4090 arithmetic prefill trace

Packet `529909b3` ran once at `a6dc2271` with the retained isolated
`e9015f91` diagnostic executable. Fresh quiet/resource admission and
whole GGUF SHA-256 verification passed. Both Jobs exited successfully
and their PIDs were absent after cleanup. Both device-specific
position-0 controls reproduced, with 37 frozen IDs, zero decode
steps and all 43 ordered checkpoints retained per lane.

The complete embedding tensors had the same SHA-256. On those
common inputs the first differing captured checkpoint was layer 0
output, row 0, channel 0 (CPU `0xba418b90`, RTX 4090 `0xba2b612c`).
All 75,776 layer-0 output values had nonzero absolute deltas, with a
maximum of `0.00317324697971344`. The trace ends at a layer output
boundary. The next diagnostic must locate the responsible operation
inside layer 0 before any arithmetic change.

[Terminal receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-rtx-arithmetic-prefill-publish-20261117/attempt/terminal.json) ·
[Comparison](D:/Compact_Evidence/candle/runs/lfm2-moe-rtx-arithmetic-prefill-publish-20261117/attempt/comparison.json) ·
[Capture audit](D:/Compact_Evidence/candle/runs/lfm2-moe-rtx-arithmetic-prefill-publish-20261117/attempt/capture-audit.json) ·
[Bound packet](D:/Compact_Evidence/candle/runs/lfm2-moe-rtx-arithmetic-prefill-publish-20261117/packet.json) ·
[Invocation](D:/Compact_Evidence/candle/runs/lfm2-moe-rtx-arithmetic-prefill-publish-20261117/invoke-reviewed-trace.ps1).

## 2026-11-17 - Bound RTX 4090 arithmetic prefill probe

Packet `95d7534e` binds one CUDA-enabled Candle executable for both trace
lanes at `e9015f91`. The executable SHA-256 is
`bd0f3837e9f6807b7674753e2e74838b514814b43724cc5b6e2f20dfee613858`.
The isolated source was verified against 1,516 exact Git blobs before
the three-path patch. The locked-offline build passed, and all 1,517
patched source files matched before and after the build.

The binary's CUDA/MSVC imports resolved to the pinned runtime. The
model-free help and missing-grant checks passed and released their Jobs.
The packet preserves 37 frozen IDs, both position-0 controls, 43
checkpoints per lane, zero decode steps and the original resource limits.
The single paired trace remains unused.
[Bound trace](D:/Compact_Evidence/candle/runs/lfm2-moe-rtx-arithmetic-prefill-inert-20261117/future-build/bound/packet.json).

## 2026-11-17 - RTX 4090 arithmetic prefill evidence and inert trace

The three retained arithmetic logs from `18e27010` bind the same 37
prompt IDs, seven probe IDs and eight forced IDs. All three choose `207`
at position 0. All seven Candle CPU and RTX 4090 sampled logits differ
there, before decode. For token `207`, Candle CPU records `0x41d5e5c8`,
RTX 4090 records `0x41e5bf6e`, and pinned llama.cpp records `0x41d61660`.
The first top-token difference is at position 2 after `49, 1252`. Neither
`1113` nor `5141` is in the retained logit sample. The retained result
localizes the first observable drift to the prefill endpoint.

Inert packet `b61f6e19` binds both endpoint controls, the original production
inputs and a diagnostic-only patch at `e9015f91`. A future isolated
CUDA-enabled Candle executable will capture one prefill per device.
Each endpoint must reproduce its own retained control. The offline
comparison scans all 37 embedding and layer rows, then the last-token
normalization and head, and stops at the first differing checkpoint.
There are 43 checkpoints per lane and no decode steps. Patch applicability,
Rust syntax and model-free comparison controls passed. Builds and runs
remain held.
[Paired inert trace and retained witnesses](D:/Compact_Evidence/candle/runs/lfm2-moe-rtx-arithmetic-prefill-inert-20261117/packet.json).

## 2026-11-17 - Production attempt stopped on RTX 4090 arithmetic

Packet `18e27010` consumed its one attempt at checkout `47e68166` with the
Candle probe built from `e9015f91`. Fresh admission and the stable full GGUF
SHA-256 passed. Candle CPU matched all eight teacher-forced selections on
both frozen cases. Maximum sampled CPU logit deltas were 0.898526 for
arithmetic and 0.916007 for Unicode.

RTX 4090 arithmetic stopped the packet at position 2 after forced IDs
`49, 1252`. Pinned llama.cpp and Candle CPU selected `1113`; RTX 4090
selected `5141`. The retained Candle CPU and RTX 4090 logs already differ
at position 0, with a 1.981274 sampled logit delta at token `207`. The first
divergent layer or operation remains open.

All five child Jobs exited 0 and verified process release. The RTX 4090
lane peaked at 15,990,063,104 Job bytes and 15,106 MiB device-wide GPU usage
in the samples. The post-run check was quiet with 23,397 MiB GPU free.
The stop left the CUDA Unicode lane unreached.
[Terminal receipt and comparison results](D:/Compact_Evidence/candle/runs/lfm2-moe-prod-binary-docs-publish-20261117/qualification/summary.json).

## 2026-11-17 - Binary-bound Unicode production successor

Packet `a9fafed1` binds the Windows CPU/RTX 4090 Candle probe built at
published `e9015f91` and the retained llama.cpp CPU reference. The Candle
executable SHA-256 is
`f6f9c8d2e7eb771844c1e02f66dcabe6721764465f4111650bbe11347c4e77f1`.
Both frozen cases, all four comparison commands, the 27-check Unicode
receipt and the original production limits carry forward from `e725408b`.

The first CUDA build stopped when nvcc could not resolve `cl.exe`. A
model-free kernel compile verified the pinned host compiler with a reduced
process-local `PATH`. The corrected locked-offline build passed in 349.5
seconds, with all 530 source bindings verified before and after. PE imports
resolved to the bound CUDA/MSVC DLLs. The model-free `--help` check passed,
all owned Jobs released their processes, and the runner refused dispatch
without a grant. The production attempt is unused.
[Binary-bound packet](D:/Compact_Evidence/candle/runs/lfm2-moe-prod-unicode-successor-inert-20261117-e9015f91/future-qualification/packet.json).
[Build and guard proof](D:/Compact_Evidence/candle/runs/lfm2-moe-prod-unicode-successor-inert-20261117-e9015f91/future-build/verification.json).

## 2026-11-17 - Inert Unicode production successor

Packet `e725408b` supersedes consumed production packet `88c5952f` at
published Candle `e9015f91`. It binds the `f7ef7c65` 27-check layer-2
receipt, both frozen prompts and continuations, the pinned llama.cpp
CPU binary, and the locked-offline build handoff for one Candle executable
shared by the CPU and RTX 4090 lanes.

The packet preserves six serial children, four comparisons and the original
40 GiB host and commit floors, 18 GiB CUDA floor, 32 GiB Job cap,
180-second child and 900-second wall limits. Preparation validated both cases
and rejected six invalid packet variants. The Candle binary is pending
separate build authorization. Builds and production dispatches are zero.
[Inert successor](D:/Compact_Evidence/candle/runs/lfm2-moe-prod-unicode-successor-inert-20261117-e9015f91/packet.json).

## 2026-11-17 - Connected maintained Unicode layer-2 CPU trace

Packet `f7ef7c65` consumed its one CPU attempt at published source
`e9015f91`. Both pre-dispatch admissions passed under the original 16 GiB
physical and commit floors. The whole-file Q4_K_M SHA-256 matched the
frozen artifact. The probe used exact retained layer-1 input and pinned
llama.cpp captures on all 32 Unicode rows.

All 27 ordered checks passed through Candle-computed `layer_out`.
Twenty-two checkpoints were bitwise equal, including Q/K RoPE, attention
values, mixer, ordered expert IDs, raw weight sums and unweighted expert
down. The two normalization stages, weighted experts, sparse merge and
layer output passed each engine's own formulas. The final output differed
from llama.cpp at 61,824 of 65,536 values under those checks.

The Windows Job exited 0 in 1,009 ms, peaked at 339,132,416 bytes and
released the child. The first post-run sample detected foreign Cargo
activity, with 43.15 GiB available and 43.64 GiB of commit headroom.
No second dispatch occurred. Production parity and Edge adoption remain open.
[Terminal and stage results](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-unicode-maintained-attempt-20261116/terminal.json).

## 2026-11-16 - Bound maintained Unicode layer-2 CPU probe

Successor `f7ef7c65` binds the isolated Windows CPU executable with SHA-256
`c0fca5b20ed284aec7917d5984f6e347839407db46509f99e1f291e3d41e4adb`
to inert packet `d7be3780`. The parent recheck validated 77 bindings, the
order of 27 comparison stages and four rejection controls. The successor
preserves the 32 Unicode IDs, exact entry and reference captures, both routing
formulas and the original limits. Its compiler arguments differ only in the
output path.

Pinned `rustc.exe` and MSVC linked the probe in a bounded Windows Job.
The build emitted linker warning `LNK4098`. The executable reported the
expected version and refused an unguarded trace with sentinel paths before
opening inputs. All three children exited and released their PIDs.
The model dispatch and attempt counts are zero.
[Bound packet and Job receipts](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-unicode-maintained-build-20261116/packet.json).

## 2026-11-16 - Inert maintained Unicode layer-2 packet

Packet `d7be3780` binds the exact 32-row layer-1 entry, 27 ordered
reference captures, maintained mixer sources and original limits. Its
trace adapter uses the maintained CPU attention route and feeds its
computed tensors through the final residual. Each engine's routing
formulas are checked independently. The pre-change Candle layer_out
remains historical evidence. The comparator writes the computed output.

Preparation verified 77 bindings and parsed seven Rust sources. Four
negative controls rejected execution enablement, a lower memory floor,
a missing attention binding and legacy Candle output acceptance.
Builds, model dispatches and attempt consumption are zero. The next
gate reviews the inert packet and builds and binds its isolated CPU
binary under separate authorization.
[Packet](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-unicode-maintained-inert-20261116/packet.json).
[Preparation checks](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-unicode-maintained-inert-20261116/preparation.json).

## 2026-11-16 - Maintained quantized MoE CPU attention

The private quantized MoE mixer now selects reference-order F32 attention
for supported AVX2/FMA inputs. RoPE, cache assembly and output projection
remain in the shared layer. Dense LFM2, CUDA and other CPU inputs use the
stock path. Routing formulas are unchanged.

The maintained kernel matched 16,896 raw scores, 16,896 probabilities
and 65,536 values on the bound 32-row Unicode regression. It verified the
input and control hashes. The neighboring 28 tests passed, covering dense
decode, batched GQA, rectangular prefill, cached decode and recovery.
Library and example checks, Clippy, formatting and overlay checks passed.
[Source-bound verification](D:/Compact_Evidence/candle/runs/lfm2-moe-maintained-cpu-attention-20261116/summary.json).

The next gate starts from exact retained layer-1 output. The maintained
layer must compute its own layer-2 Q/K/V, attention and residual. Bind
the pinned and retained controls and preserve both routing formulas.
Production Unicode selection and CPU/RTX 4090 qualification are open.

## 2026-11-16 - Unicode connected CPU attention

The value-order pass-down was already satisfied by the 32-row reference
replay. The successor connected the three proven schedules. One admitted
offline probe started from frozen Q/K/V and carried its computed scores
and probabilities forward. It matched 16,896 visible raw Q·K scores,
16,896 probabilities and 65,536 values bitwise. The Job exited 0 and
released. Channel 8 matched `0xbac911ed`.
[Replay receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-unicode-attention-connected-20261116/terminal.json).
[Inputs and layouts](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-unicode-attention-connected-20261116/inputs.json).

This supports a private reference-order implementation for supported
quantized MoE CPU F32 attention. The shared AttentionLayer should keep
the existing cache, projection and rotary flow. Its MoE CPU caller should
select the proven score, softmax and value schedules after projection
and RoPE. Derive mask, grouping and lengths from the caller. Keep the
existing dense, CUDA and unsupported CPU paths. Prove the maintained
route with the frozen attention regression and neighboring prefill,
cached decode and reset checks before the connected Unicode layer trace.
CPU/CUDA production selection, standalone LFM3, Edge acceptance and the
LTS contract remain open.

## 2026-11-16 - Unicode CPU reference QK order

One admitted standalone replay applied the pinned llama.cpp AVX2/FMA dot
order on the exact matched Q/K RoPE inputs. It kept the causal mask and
the original 32-key Candle and 256-key GGML score layouts. All 16,896
visible raw F32 scores matched, including the three earlier witnesses.
The score capture is byte-identical to the pinned control. The Job released.
[Q·K replay](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-unicode-qk-reference-order-20261116/terminal.json).

The subsequent connected replay used the same frozen Q/K/V. The isolated
controls and replay inputs are bound below.
[Q·K inputs and layouts](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-unicode-qk-reference-order-20261116/inputs.json).
[Value inputs and control](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-unicode-value-reference-order-20261116/inputs.json).

## 2026-11-16 - Unicode CPU reference value order

One admitted standalone F32 replay used the `96e27811` eight-lane FMA and
horizontal sum order on all 32 rows. It kept the causal mask, the 32-key
Candle and 256-key padded GGML score layouts, and the 12,288-stride V cache.
All 65,536 outputs matched pinned llama.cpp bitwise, including token 1,
head 0, channel 8 at `0xbac911ed`. The Job released after exit 0.
[Replay and bindings](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-unicode-value-reference-order-20261116/terminal.json).

The Q·K dot trace reproduced three scores on the retained Q/K RoPE inputs.
The subsequent full Q·K replay extended that proof to all visible scores.
[Q·K trace](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-unicode-qk-dot-20261116/result.json).

## 2026-11-16 - Unicode CPU value reduction

One admitted Candle replay reproduced all 65,536 native control values.
With matched probabilities and frozen V, the first 2,056 outputs matched
llama.cpp, including token 1/channel 0 at `0x3b7aeaed`. The next output
at token 1/head 0/channel 8 differed by one ULP (`0xbac911ee` Candle,
`0xbac911ed` llama.cpp). The probe stopped there and released its Job.
[Comparison](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-unicode-value-reduction-20261116/comparison.json).
[Terminal](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-unicode-value-reduction-20261116/terminal.json).

Both probability words and both V words match the earlier channel-8 trace
bitwise. That trace reproduced these bits with fused accumulation in
Candle and separate F32 multiplication and addition at key 1 in llama.cpp.
Packet `96e27811` proved the full reference order on the 37-row case.
The reference-order successor kept the same probabilities, frozen V, native
layouts and limits.
[Consumed replay inputs](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-unicode-value-reduction-20261116/inputs.json).
[Channel-8 trace](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-value-reduction-channel8-20261114-53d89669/result.json).
[Reference order proof](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-value-reduction-replay-only-20261115-851b5cc0-docs/result.json).

## 2026-11-16 - Unicode CPU softmax replay

The isolated replay used Candle `50b81e4b` and pinned llama.cpp `582eab71`.
Stock Candle reproduced its native control. Reciprocal-only matched token
2/head 0/key 0 at `0x3db318f2` and differed at 10,399 visible values.
The AVX2/FMA exponent, reduction and reciprocal schedule matched all 16,896
reference probabilities bitwise. All masked and padded lanes stayed zero.
One admitted offline child exited and released its Job.
[Comparison](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-unicode-softmax-schedule-run-20261116/comparison.json).
[Terminal](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-unicode-softmax-schedule-run-20261116/terminal.json).

The shared-score manifest binds the reference probabilities, frozen V and
the 65,536-value reference reduction that already matched on those inputs.
Its Candle lane stopped at softmax. Use the native attention packet's Candle
probabilities and Candle value output together for that engine's control.
The shared-score replay keeps the causal mask, Candle's 32-key layout,
llama.cpp's 256-key padded layout and the 16 GiB RAM/commit floors.
[Shared-score manifest](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-unicode-shared-scores-20261116/inputs.json).
[Original controls and layouts](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-unicode-attention-build-20261116/packet.json).

## 2026-11-16 - Production successor stopped on CPU Unicode

Packet `88c5952f` passed fresh admission and whole GGUF identity, then ran
both reference and both Candle CPU children. Arithmetic matched all eight
selections, with maximum sampled logit delta `1.145143`. Unicode stopped
at position 2 after forced IDs `597, 4695` (`10966` reference, `355` Candle).
All four Jobs released their processes. CUDA was unreached.
[Terminal receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-prod-todo-rebind-20261116/qualification/summary.json).

## 2026-11-16 - Production packet review

The inert successor binds both frozen cases, the `50b81e4b` binaries,
and the maintained CPU layer-2 proof. The runner preserves prior receipts
and verifies Job command and log identity before comparison. Model-free
checks passed. Production parity remains open. [Review receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-prod-reviewed-20261116-50b81e4b/review.json).

## 2026-11-16 - Maintained CPU layer-2 checkpoint

At `50b81e4b`, packet `2aa2ed81` verified the whole GGUF hash and passed
all 27 checks on the 37 frozen rows. Entry through RoPE and the 148 ordered
expert IDs matched exactly. The other stages matched the retained controls
or each engine's formula. The final Candle output reproduced `9c8006a5`
bitwise. Both Jobs released their PIDs. [Terminal receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-maintained-20261116-50b81e4b/terminal.json).

## 2026-11-16 - Maintained CPU layer-1 checkpoint

The single admitted offline replay at `50b81e4b` used the exact layer-0
output and passed all 16 checks, including 75,776 bitwise `layer_out` values.
Both Jobs released their processes. [Layer-1 receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer1-committed-20261116/completion.json).

## 2026-11-16 - Maintained CPU layer-0 checkpoint

On the 37 frozen rows, the maintained CPU schedules passed all 19 offline
comparisons, including 75,776 bitwise matches each for dense Q6_K FFN down
and the F32 `layer_out`. Both Jobs released their processes. This output
became the exact entry for the maintained layer-1 replay.
[Layer-0 receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer0-dense-ffn-down-ordered-20261116/summary.json).

## 2026-11-15 - Connected layer-2 CPU diagnostic

On published `30e48252`, packet `9c8006a5` passed fresh admission and the
whole-model SHA-256, then ran the verified binary once. The run covered
all 37 frozen rows and 27 declared stages. The nine entry-through-RoPE
stages and ordered top-K matched exactly. Candle's downstream controls
and both engines' weighting, merge and residual formulas passed. The
303,104-byte connected output has SHA-256
`f6345e1826cac7895ec15d0deb9093a502f66549c84917159209efdd2df21241`.
The Job recorded 51.7 MiB peak job memory, exit code 0 and PID cleanup.
The post-run census passed. The diagnostic is complete. Maintained
integration and production qualification follow in TODO.
[Job receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-published-snapshot-20261115/connected-job.json).
[Comparison log](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-published-snapshot-20261115/connected.log).

## 2026-11-15 - Inert SwiGLU and expert-down successor

Packet `2a47477d` adds the exact Candle SwiGLU and down controls to unused
parent `078aa139`. The other 25 comparisons, frozen inputs and limits
carry forward. The source still computes Candle's gate, up, SwiGLU, down
and residual. The offline audit reproduced pinned reference weighting,
slot-order merge and final residual using its own captured operands.
Candle's audit used its own norm-derived weights, expert order and mixer.

The guarded compiler built once after the pinned executable cleared.
Its Job released, the correct seal verified, a wrong digest failed,
and both packets had zero probe dispatches. [Readiness receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-connected-down-20261115-630691ed/readiness.json).

The inert review of `2a47477d` confirmed 486 bindings and 27 checkpoint
definitions. [Review receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-connected-down-20261115-630691ed/review.json).

Inert successor `cc7662ea` resolved the local checkout binding for review.
Within the diagnostic scope, the canonical HEAD had no committed delta from
the compiled source. The two dirty LFM3 files appended test functions only.
The successor bound those exact bytes, preserved the predecessor's 486
bindings, and passed read-only verification of 492 bindings, 27 stages and
13 GGUF ranges. The review created no attempt marker or computed output.
[Successor verification](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-checkout-20261115-ca2a29b7/summary.json).

The local review of `cc7662ea` confirmed the inherited executable, arguments,
comparison definitions, environment and limits. The `2a47477d` verifier
rejected the checkout. The successor's full 14,526,258,464-byte model SHA-256
matched, and the host census cleared the unchanged conditions. Both attempts
were unused at review. Execution required an exact grant and fresh checks.
[Successor review](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-checkout-20261115-ca2a29b7/review.json).

## 2026-11-15 - Layer-2 Q6_K expert down on both input lanes

The pinned GGML indexed graph used the ordinary CPU buffer with the tiled
path for experts 30, 37 and 54 and plain AVX2 for the other 32 groups.
On each verified SwiGLU input it matched Candle on all 888 pre-dot Q8_K
blocks and 303,104 unweighted outputs. The reference lane also reproduced
its archived Q8_K and down controls, including row 36.

Changing the input lane changed 682 Q8_K blocks and 250,333 down values.
The first output difference was token 1, slot 0, expert 37, channel 0
(`0x3ba325b7` Candle, `0x3ba325b8` llama.cpp). The guarded Windows attempt
built only the paired diagnostic and exercised the frozen weight range.
All three Jobs released under the original limits. [Down replay receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-down-cross-20261115-62579e67/summary.json).

## 2026-11-15 - Layer-2 SwiGLU controls on both input lanes

Inert review of `078aa139` found that the next checkpoint compares
SwiGLU directly with llama.cpp even when the sealed Candle up tensor
differs. Reconstruction matched every pinned reference value and
predicted 194,626 cross-input differences. The first is token 1, slot 0,
expert 37, channel 1, with the same gate and an up difference of one ULP.
[Review receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-connected-up-20261115-cf15b2fc/review-swiglu.json).

One bounded Windows CPU attempt reused the original binary for two
sequential Jobs. All 227,328 outputs per lane matched their controls,
including row 36. Both Jobs released after passing quiet/resource
admission. The retained Candle tensor is verified for the next down
replay. [SwiGLU receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-swiglu-cross-20261115-62579e67/summary.json).

## 2026-11-15 - Inert gate-plus-up connected layer-2 packet

Packet `078aa139` supersedes consumed `f007a0a9` and binds both frozen
FFN norm lanes to the full gate and up controls. It keeps 26 comparisons
from the parent and adds the exact Candle-side control at `expert_up`.
The guarded build ran once after compiler contention cleared and released
its Job. The seal passed inert verification, a wrong digest was rejected,
and the attempt state was absent.
[Readiness receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-connected-up-20261115-cf15b2fc/readiness.json). [Preparation record](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-connected-up-20261115-cf15b2fc/closeout-preparation.json).

## 2026-11-15 - Layer-2 up-projection replay

Packet `f007a0a9` passed fresh admission and the whole-model hash, then
used its single connected attempt on 37 frozen rows. It passed the declared
comparisons for 916,934 values across 21 stages through `expert_gate`,
with exact layer-1 entry, Q/K RoPE and top-K. It stopped at `expert_up`,
token 1, slot 0, expert 37, channel 1 (`0x3e0e29a6` vs `0x3e0e29a7`,
one ULP). The Job released and the post-run census passed.
[Attempt receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-connected-gate-20261115-10448a6b/connected-attempt-20261115.json).

The offline Q4_K up replay matched all pre-dot Q8_K operands and
227,328 outputs per shared FFN norm in both engines across all 35 groups.
Changing the norm reproduced all 170,062 connected differences, starting
at expert 37's channel-1 witness. All Jobs released after one attempt
under the original limits. This proof covers the indexed diagnostic
schedule. Maintained CPU execution still uses `GenericTiled`.
[Full up replay](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-up-cross-20261115-cf15b2fc/result.json).

## 2026-11-15 - Connected layer-2 whole-gate control

Packet `f007a0a9` sealed both full gate controls with exact entry, Q/K RoPE
and top-K order. Inert verification passed after one compiler contention
refusal and a successful guarded retry.
[Readiness receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-connected-gate-20261115-10448a6b/readiness.json).

The connected trace computed gate and up from Candle's own FFN norm.
All 227,328 connected gate values matched the bound Candle control.
The first up-projection difference consumed the attempt.
[Attempt receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-connected-gate-20261115-10448a6b/connected-attempt-20261115.json).

## 2026-11-15 - Connected layer-2 score and gather controls

Packet `05899599` passed revised admission and whole-model verification.
It matched 686,720 values against the bound controls through sigmoid across
14 stages. The first selection-score discrepancy was token 1/expert 3.
The Job released and the post-run census passed.
[Connected attempt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-connected-sigmoid-20261115-0be31b31/connected-attempt-20261115.json).

Both engines matched all 2,368 selection scores per shared input. Switching
inputs reproduced all 1,661 differences, starting with `0x3e358d0e` on Candle
probabilities and `0x3e358d0f` on llama.cpp probabilities at token 1/expert 3.
One offline attempt passed revised admission and both Jobs released.
[Selection-bias replay](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-selection-bias-cross-20261115-0be31b31/result.json).

Packet `f7a9f73f` bound both score and gather inputs and preserved exact
entry, Q/K RoPE and expert order. The prepared comparator checks each
engine's raw sum using F64 accumulation and one F32 rounding. Its
connected attempt validated all 37 raw sums per engine, including 28
input-driven differences, and reached normalized weights before the gate stop.
[Readiness receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-connected-gather-20261115-10448a6b/readiness.json).

The single connected attempt accepted 689,606 values across 20 stages
after fresh admission and whole-model identity verification. It stopped at
expert 37's gate at token 1, slot 0, channel 2 (`0xbc065340` vs
`0xbc065330`, 16 ULPs). The Job released and post-run admission passed.
[Attempt receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-connected-gather-20261115-10448a6b/connected-attempt-20261115.json).

Expert 37's offline replay matched all Q8_K operands and 24,576 gate outputs
per shared input in both engines. Changing FFN norms changed three Q8_K scales
at token 1, with identical quantized bytes and sums, and reproduced both
channel-2 values. All Jobs released.
[Replay receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-expert37-gate-cross-20261115-10448a6b/result.json).

The full offline gate replay matched all pre-dot Q8_K fields and 227,328
outputs per shared FFN norm in both engines across the 35 original groups.
Changing inputs reproduced all 169,497 connected gate differences. All
Jobs released after one attempt under the original limits.
[Full gate replay](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-gate-cross-20261115-10448a6b/result.json).

## 2026-11-15 - Layer-2 shared-input unbiased gather

The first attempt passed the memory checks, then the launcher refused an
active pinned compiler. Its marker was consumed with zero dispatches.
[Compiler stop](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-gather-cross-20261115-0be31b31/terminal.json).

The next attempt checked the bound compiler before marking the attempt,
built one probe and ran one replay per frozen probability input. Both engines
matched all 148 controls per input, and all three Jobs released.
Switching inputs reproduced 110 differences, first at token 1/slot 1/expert 53.
[Gather proof](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-gather-cross-20261115-10448a6b/result.json).

## 2026-11-15 - Layer-2 shared-input sigmoid proof

Packet `a21893d2` matched 684,352 values through logits and stopped at
sigmoid token 1/expert 3.
[Connected attempt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-connected-router-20261115-67b14101/connected-attempt-20261115.json).

Both engines matched all 2,368 sigmoid probabilities per shared 37-row
input. Switching inputs reproduced all 1,680 differences. The first was
`0x3e17402f` on Candle logits and `0x3e174030` on llama.cpp logits at token
1/expert 3. Both Jobs released.
[Sigmoid replay](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-sigmoid-cross-20261115-0be31b31/result.json).

## 2026-11-15 - Layer-2 attention score and value witnesses

The rebound trace at `53d89669` reproduced the channel-0 output controls
on 37 rows and stopped at the two raw F32 Q·K scores. Candle gave
`0x420f017a,0x423ebead` and llama.cpp gave `0x420f017b,0x423ebeac`.
The receipt binds the operands, layouts and dispatches. All Jobs released
after one offline attempt.
[Paired attention trace receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-attention-first-term-20261114-53d89669/summary.json).

The 64-lane reconstruction reproduced both score pairs. Candle first
uses a different FMA accumulator at channel 1. The first different rounded
prefix is channel 1 for key 0 (`0x3dfac4dd` vs `0x3dfac4de`) and channel 2
for key 1 (`0x3d35b375` vs `0x3d35b376`). Both replay Jobs released.
[Q·K arithmetic receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-qk-dot-20261114-53d89669/summary.json).

At token 1, head 0, shared raw scores and frozen V matched through
scaling, softmax, the two products and channel-0 output `0x3b7aeaed`.
The raw score schedule explains this witness. All Jobs released.
[Shared-score receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-attention-shared-scores-20261114-53d89669/summary.json).

Connected CPU packet `93cc154a` binds 37 rows, 13 GGUF ranges and 27
captures at `53d89669`. Native self-tests and wrong-digest refusal passed.
[Inert connected packet](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-connected-prep-20261114-53d89669/packet.json).

Two identity checks matched the whole-model hash, then build activity
blocked final admission with zero dispatches and both memory floors met.
[Admission receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-connected-prep-20261114-53d89669/connected-admission-20261114.json).

Packet `93cc154a` passed the whole-file hash and admission. Its sole
attempt matched 454,656 values through Q/K RoPE and stopped at attention
token 1/head 0/channel 1 (`0x3baac798` vs `0x3baac794`, four ULPs).
The Job released, the post-run census passed, and the packet is consumed.
[Connected attempt receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-connected-prep-20261114-53d89669/connected-attempt-20261114.json).

The channel-1 replay reproduced both native values. Shared raw scores
and frozen V matched through softmax, products and final `0x3baac794`.
The upstream score difference explains the four ULPs. All Jobs released.
[Channel-1 replay receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-attention-channel1-20261114-53d89669/summary.json).

The 37-row shared-score replay reproduced all 75,776 native values per
engine. With 22,496 shared raw scores, Candle matched 2,056 reference
values and stopped at token 1/head 0/channel 8 (`0xbac911ee` vs
`0xbac911ed`). Probabilities and rounded products matched. All Jobs
released. This receipt is in C: artifacts after a D: allocation failure.
[Full replay receipt](../../artifacts/lfm2-moe/layer2-attention-all-shared-20261114-53d89669-resume/summary.json).

At channel 8, the differing key-1 accumulation is fused in Candle and
separately rounded in llama.cpp. The retained probabilities and V
reproduced both bits. Both Jobs released with zero model loads.
[Channel-8 arithmetic receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-value-reduction-channel8-20261114-53d89669/summary.json).

The full value-reduction probes bind 22,496 probabilities and frozen V.
A pre-seal compiler refusal consumed no attempt. Both memory floors passed.
[Admission receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-value-reduction-all-20261114-53d89669/summary.json). [Prepared successor](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-value-reduction-all-20261114-53d89669-resume/readiness.json).

Both builds passed and released their Jobs. Compiler PID 16624 blocked
post-build admission. The attempt is consumed with zero replays.
[Terminal receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-value-reduction-all-20261114-53d89669-resume/summary.json).

Packet `96e27811` matched all 75,776 values per engine against its own
order on shared probabilities and V. Both controls passed. Accumulation
orders explain 46,278 cross-engine bit differences. Both Jobs released.
[Terminal receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-value-reduction-replay-only-20261115-851b5cc0-docs/summary.json).

## 2026-11-14 - Layer-2 raw expert-weight sum replay stopped at row 1

Packet `6db2c79c` fed both native CPU reductions the matched weights
from `62ab7c06`. Pinned GGML `ggml_sum_rows` reproduced all 37
retained raw sums. Candle `sum_keepdim` matched row 0 and stopped at
row 1 with `0x3fb72528` versus GGML's `0x3fb72527`. The four input
bits were `0x3f171661`, `0x3ea27484`, `0x3e8ba988` and
`0x3e8049ce`, in expert-slot order. All seven admission checks passed,
and all three Jobs released their processes. The single attempt ended
before normalization with zero retries and zero model loads.
[First-difference receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-router-sum-20261114-62ab7c06/summary.json).

## 2026-11-14 - Layer-2 unbiased expert-weight gather parity

Packet `62ab7c06` reused `996fe7b0` probabilities and `91a27e18` IDs.
Both CPU engines reproduced all 148 retained unbiased weights bit for
bit. For row 36, experts `[54, 37, 30, 20]` yielded `0x3eb91b2e`,
`0x3ebc59e1`, `0x3eb410b2` and `0x3e17ff9c` in that order. Fresh
admission passed and all three Jobs exited cleanly. The earlier
pre-seal refusal is retained, and the single offline attempt used
zero retries and zero model loads.
[Comparison](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-router-gather-20261114-91a27e18/comparison.json)
and [terminal receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-router-gather-20261114-91a27e18/terminal.json).

## 2026-11-14 - Layer-2 ordered top-four expert parity

Packet `91a27e18` ran the native Candle CPU sort and pinned llama.cpp
`ggml_argsort_top_k` on the exact 37×64 selection scores from `17d27ea3`.
Both reproduced all 148 IDs in order, including `[54, 37, 30, 20]` in
row 36. The isolated builds and the one replay passed admission and
released all three Jobs. The attempt used zero retries and zero model loads.
[Comparison](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-router-top4-20261114-17d27ea3/comparison.json)
and [terminal receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-router-top4-20261114-17d27ea3/terminal.json).

## 2026-11-14 - Layer-2 F32 expert selection bias parity

Packet `17d27ea3` used a verified 256-byte bias slice from the frozen GGUF
and the exact 37×64 probabilities from `996fe7b0`. Both CPU engines matched
all 2,368 scores, including row 36 with endpoint bits `0x3e3ae9bb` and
`0x3e08647b`. Seven quiet/resource checks passed, all three Jobs released
their processes, and the attempt used zero model loads and zero retries.
[Comparison](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-router-bias-20261114-996fe7b0/comparison.json)
and [terminal receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-router-bias-20261114-996fe7b0/terminal.json).

## 2026-11-14 - Layer-2 F32 router sigmoid parity

Packet `996fe7b0` replayed Candle's CPU sigmoid and the pinned llama.cpp
CPU graph on the 37×64 exact logits from `47f523cd`. Both reproduced all
2,368 retained probabilities, including row 36 with endpoint bits
`0x3e16c884` and `0x3dec25cc`. Both isolated builds and the single replay
passed their admission and cleanup checks. The packet used zero model
loads and zero retries.
[Comparison](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-router-sigmoid-20261114-47f523cd/comparison.json)
and [terminal receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-router-sigmoid-20261114-47f523cd/terminal.json).

## 2026-11-14 - Layer-2 F32 router projection parity

Packet `47f523cd` reused the pinned llama.cpp adapter and built the isolated
Candle probe once at `67e7cbc0`. Both engines reproduced every one of the
2,368 pre-sigmoid logits, including the 64 row-36 values. The row-36 endpoint
bits were `0xbfe0cfc2` and `0xc00268d2`. Both bounded Jobs released their
processes and passed their final quiet/resource census.
[Comparison](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-router-closeout-20261114-d0075c3b/comparison.json)
and [terminal receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-router-closeout-20261114-d0075c3b/terminal.json).

## 2026-11-14 - Layer-2 post-mixer residual and weighted FFN RMSNorm parity

The offline probe added the matched mixer and layer-1 output in F32, then
replayed weighted RMSNorm. All 75,776 normalized values matched pinned
llama.cpp bit for bit, including all 2,048 in row 36. The build and replay
passed fresh admission and released both Jobs.
[Receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-ffn-norm-20261114-6dbeea9c/summary.json).

## 2026-11-14 - Layer-2 Q4_K attention output projection parity

One admitted offline replay matched the 296 Q8_K scales, 75,776 signed
quant bytes and 4,736 integer sums before the dot. Both the pinned
llama.cpp replay and isolated Candle matched all 75,776 retained
`mixer.f32` values. Row 36 matched all 2,048 values through the GEMV
schedule, with endpoint bits `0xb95b0980` and `0xba4507b0`. Both builds
and the replay released their Jobs. [Receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-attn-output-20261114-d62aa83b/summary.json).

## 2026-11-14 - Layer-2 causal attention values parity

One admitted offline replay matched all 75,776 values of
`attn_values.f32` on the 37 matched Q/K/V rows, including all 2,048
values in row 36. The probe used the pinned GGML CPU graph with 256
padded KV slots. Both Jobs released their processes.
[Receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-attn-values-20261114-b67d23be/summary.json).

## 2026-11-14 - Layer-2 query and key RoPE parity

After the competing verifier and build cleared, preliminary admission passed
before packet `b67d23be` was sealed. The build and offline replay each passed
fresh admission and released their Jobs. All 94,720 rotated values matched
the pinned captures at the 37 frozen positions.
[Receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-qk-rope-20261114-a4279483/summary.json).

## 2026-11-14 - Layer-2 query and key weighted RMSNorm parity

One admitted offline replay matched all 94,720 pre-RoPE values on the
37 frozen rows. At token 0, head 0, channel 0, query matched `0x40075525`
and key matched `0x3d5c3210`. Both Jobs released their processes.
[Receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-qk-norm-20261114-9b0d5544/summary.json).

## 2026-11-14 - Layer-2 Q6_K value projection parity

The `IoError` correction passed the isolated build. One offline replay
matched all 18,944 values across the 37 retained rows using the proven
Q6_K F32 schedule. All 512 values in row 36 matched. Both Jobs released
their processes. [Receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-value-replay-20261114-7432e492/summary.json).

## 2026-11-14 - Layer-2 value replay build stop

Packet `27fd6f53` passed admission, then stopped when the isolated probe
failed to compile with `E0433`. The copied `require` helper refers to
`io::Error`, while the probe imports that type as `IoError`. The build
Job released the compiler. The attempt ended with zero replay dispatches.
[Receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-value-replay-20261114-cdf58e7d/summary.json).

## 2026-11-13 - Layer-2 key GEMM/GEMV parity

Packet `cdf58e7d` built the isolated 512-wide probe and ran one offline replay
on the retained 37 Q8_K rows. The replay reproduced all 296 Q8_K blocks and
matched 18,432 GEMM values in rows 0–35 and 512 GEMV values in row 36 bit for
bit with pinned llama.cpp. Token 0, head 0, channel 0 matched `0x3a3453bc`.
Both Jobs released their processes. [Receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-key-replay-20261113-97dc98a2/summary.json).

## 2026-11-13 - Layer-2 query GEMM/GEMV parity

Packet `97dc98a2` reused the isolated Candle executable `0e2e0bb2` for one
offline replay. The stock Candle control and all 296 matched Q8_K blocks
reproduced. The corrected output matched pinned llama.cpp bit for bit across
73,728 GEMM values in rows 0–35 and all 2,048 GEMV values in row 36. Token 0,
head 0, channel 0 matched `0x3c2a4506`. Admission passed before and after
the single Job, which released the probe. [Receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-query-replay-20261113-0e2e0bb2/summary.json).

## 2026-11-13 - Layer-2 query schedule build and admission stop

Packet `0e822a25` verified 187 bindings and built the isolated GEMM/GEMV
probe. The build exited zero and its Job released the process. Four Python
processes appeared before replay admission, so the attempt ended after the
build with zero replay dispatches. The binary is retained for a separately
selected replay-only packet. [Receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-query-schedule-20261113-031af7c7/summary.json).

## 2026-11-13 - Offline layer-2 query replay

Reproduced both native `blk.2.attn_q.weight` outputs on all 37 retained
RMSNorm rows. Every Q8_K scale, signed byte and block sum agreed. The
token-0, head-0, channel-0 dot trace reproduced both native outputs and
localized the first F32 difference to the first subblock pair of block 0.
Two bounded probe builds and one bounded component replay completed with
zero model loads. [Replay receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-query-offline-20261113-0e236955/summary.json).

## 2026-11-13 - Bounded layer-2 CPU trace

Packet `0e236955` wrote its one-attempt marker after final reference
admission. Pinned llama.cpp captured all 27 checkpoints and exited zero.
Candle matched all 75,776 F32 values at each of entry and weighted RMSNorm,
then exited one at the query projection. The first witness was token 0,
head 0, channel 0, with Candle `0x3c2a4502` and llama.cpp `0x3c2a4506`.
Both Jobs released their processes and their post-exit censuses passed.
The reference captures and the mismatching Candle query tensor are retained.
[Trace receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-successor-20261113-4df21393/trace-summary.json).

## 2026-11-13 - Inert layer-2 CPU successor

Sealed [packet `0e236955`](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-successor-20261113-4df21393/packet.json)
at checkout `4df21393` with the original probe source scope at `9cc7c853`.
The controller binds the consumed predecessor and its zero-dispatch terminal
receipt. Preparation rechecked the GGUF header and all 13 selected tensor
ranges, 27 checkpoint definitions and 259 file bindings. Eleven controller
tests covered both admission stages, the model hash check, the one-shot marker,
capture locks and cleanup after failure. The real packet remained unconsumed
with zero dispatches. [Preparation receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-successor-20261113-4df21393/summary.json).

## 2026-11-13 - Layer-2 CPU packet stopped before launch

Packet `623dc38e` passed owner admission after an earlier refusal and
verified the model identity. The controller had already recorded its one
attempt when two Python processes blocked reference admission. It
stopped with zero model dispatches and retained the marker and all
admission results. [Terminal receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-controller-review-20261113-9cc7c853/terminal.json).

## 2026-11-13 - Inert layer-2 CPU packet

Prepared packet `0c99a403` from Candle `9cc7c853`, pinned llama.cpp `582eab71`
and the verified 37-token `l_out-1` entry. The packet binds 27 typed checkpoints
and all 13 layer-2 weight ranges. Native model-free controls exercised strided
F32 and integer captures, wrong contracts, the two normalization formulas,
first-failure stopping and the signed KV head metadata in the frozen header.
The controller checks rejected source and input drift,
an invalid grant, a consumed attempt and failed admission. Preparation stopped
with zero model runs and the attempt unconsumed.
[Verification](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-inert-20261113-9cc7c853/summary.json).
Review superseded the original controller with packet `623dc38e` before
either packet ran a model. [Review proof](D:/Compact_Evidence/candle/runs/lfm2-moe-layer2-controller-review-20261113-9cc7c853/summary.json).

## 2026-11-13 - Composed offline layer-1 CPU replay

One isolated CPU run at Candle `60ce38d4` started from the exact retained
`l_out-0` and computed each of the ten layer-1 checkpoints in order.
The complete `l_out-1` matched all 75,776 reference values, including rows
16 and 36. The source, executable, frozen weights and epsilon, stage
hashes, and build and replay Job receipts are bound in the terminal record.
[Composed replay](D:/Compact_Evidence/candle/runs/lfm2-moe-layer1-composed-offline-20261113-60ce38d4/summary.json).

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

Offline replay with comparator `42deff05` reproduced the position-2
mismatch (`1113` vs `6986`). The production successor needs a newly
bound reference binary. The 14 comparator tests and pinned VS 18 probe
syntax check passed.

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
binary in the separately owned D: build directory. The historical path was
`D:/CodexStorage/build/candle-lfm2-moe-refresh-20261110-5f094821/release/examples/lfm2-moe.exe`
with SHA-256 `7ee3969b125feffd96d1df2b2b78f767dafca78cf4e978c73fb6f34a3e8a82e3`.
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

The initial [implementation receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-20261110-5f094821/summary.json)
retains packet `1e4449ab`. It was superseded by the refresh. The model
was inspected at `C:/llamacpp/models/` with a CPU probe built from `582eab71`
apart from the installed `9df0b73c` runtime.

The [refresh receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-artifact-refresh-20261110-5f094821/summary.json)
and [inert packet](D:/Compact_Evidence/candle/runs/lfm2-moe-artifact-refresh-20261110-5f094821/packet/packet.json)
retain the exact commands, binary/build bindings, and pre-publication packet
`1717f529`. The subsequent source closeout binds the committed inventory.
The shared catalog protects the D: build output while this packet depends on it;
review its retention only after the packet is closed or replaced. [TODO](TODO.md)
owns packet review, separately granted finite production qualification, Edge
acceptance and separately authorized publication/adoption. D1 stays closed.

---
AI-edited: 2026-11-13 | agent=Codex/root | task=lfm2-moe-layer2-attempt | change=record the layer-2 admission stop and zero dispatches
AI-edited: 2026-11-13 | agent=Codex/root | task=lfm2-moe-layer2-successor | change=retain preparation and controller proof
AI-edited: 2026-11-13 | agent=Codex/root | task=lfm2-moe-layer2-cpu-trace | change=retain the consumed attempt and query witness
AI-edited: 2026-11-13 | agent=Codex/root | task=lfm2-moe-layer2-query-offline | change=retain the matching pre-dot operands and query witness
AI-edited: 2026-11-13 | agent=Codex/root | task=lfm2-moe-layer2-query-schedule | change=retain the verified isolated build and replay admission stop
AI-edited: 2026-11-13 | agent=Codex/root | task=lfm2-moe-layer2-query-replay | change=retain full query GEMM and GEMV parity on matched operands
AI-edited: 2026-11-13 | agent=Codex/root | task=lfm2-moe-layer2-key-replay | change=retain full key GEMM and GEMV parity on matched operands
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-value-replay | change=retain the compiler failure and zero-dispatch result
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-value-recovery | change=retain the corrected build and parity proof
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-qk-norm | change=retain per-head weighted RMSNorm parity
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-rope-replay | change=retain exact isolated rotary output proof
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-attn-values | change=retain exact causal attention values proof
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-attn-output | change=retain the exact projection and row-36 proof
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-ffn-norm | change=retain the exact FFN normalization parity proof
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-router-replay | change=retain the consumed router packet and its exact comparison
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-router-sigmoid | change=retain the exact full-row router sigmoid proof
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-router-bias | change=retain the completed F32 expert bias replay
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-router-top4 | change=retain the ordered top-four CPU parity and cleanup proof
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-gather-replay | change=retain bitwise unbiased gather parity and the row-36 witness
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-layer2-raw-sum | change=retain the raw sum mismatch, exact operands and terminal cleanup
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-ffn-norm-cross | change=retain the completed layer-2 attention trace details
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-connected-sigmoid-stop | change=archive completed value-reduction attempt details
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-sigmoid-dropin | change=retain the historical preparation paths and release evidence
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-connected-up-seal | change=retain build cleanup and inert successor proof
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-swiglu-cross | change=retain the native two-lane SwiGLU proof
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-down-cross | change=retain exact down controls and group dispatch proof
AI-edited: 2026-11-15 | agent=Codex/root | task=moe-connected-down-seal | change=retain the seal, lane audit and build cleanup proof
AI-edited: 2026-11-17 | agent=Codex/root | task=lfm3-moe-qualification | change=record LFM3 admission and the completed MoE trace
AI-edited: 2026-11-17 | agent=Codex/root | task=moe-cuda-shortconv | change=bind the crossed replay and Q4_K activation sum correction
AI-edited: 2026-11-17 | agent=Codex/root | task=moe-resource-replay | change=record all eight RTX selections and the fast MMQ lead
AI-edited: 2026-11-18 | agent=Codex/root | task=portfolio-dropin | change=retain stock controls and shared-input Q8_K comparison
AI-edited: 2026-11-18 | agent=Codex/root | task=moe-unicode-position2 | change=archive the shared-input Q6_K and weight-sum controls with the eight-position replay
AI-edited: 2026-11-18 | agent=Codex/root | task=moe-ordered-attention | change=archive ordered CUDA operator proof and both production outcomes
AI-edited: 2026-11-18 | agent=Codex/root | task=moe-frozen-qualification | change=retain both final packets and their exact frozen results
