# LFM2-MoE completed work

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
