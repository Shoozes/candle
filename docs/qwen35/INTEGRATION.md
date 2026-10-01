# Exact 4B Qwen3.5 consumer handoff

This is a local Candle interface and latency handoff, not an EdgeSymbio
activation receipt.
The handoff was captured before the local overlay checkpoints, from `main`
baseline `0c0bc3ecb7eed3eb22c8fa08a190741c84f2bb56`. The inspected EdgeSymbio
checkout was local `37d475efa873841cd31751a29b3e29d7804197fa`; the
symbio-code checkout was local `f27dea8bed9cfee4309f52fb7de770ae9056abf7`.
Recheck those identities and each checkout's dirty/worker state before editing.

## Candle interface and qualified boundary

`candle_transformers::models::qwen35::admission::{Artifact,
AdmissionLimits}` opens a caller-bounded GGUF only after its complete SHA-256
and tensor directory pass. Use the exact Q8_0/F32 4B model identity:

| Field | Value |
| --- | --- |
| File | `Darkidol-Ballad-4B.Q8_0.gguf` |
| Bytes | `4,482,403,200` |
| SHA-256 | `fd6dcc1ea357b483ae2517b482bf328181e5b371b65661c734aa133407ae2572` |
| Replacement source | `mradermacher/Darkidol-Ballad-4B-GGUF` at `0fc1207385c1655f27e2bb93c72115ee8ac12369` |

`qwen35::cpu::CpuModel::load(artifact, max_positions, max_state_bytes)` is the
native CPU reference. With Candle's `cuda` feature,
`qwen35::cpu::HybridGpuModel::load(artifact, max_positions, max_state_bytes,
device)` requires a CUDA device. Both provide token-ID `prefill`, `decode`,
`reset`, and `position`; the hybrid also provides `prefill_with_cancel` with
token-boundary cancellation and state clearing. Projection weights reside on
CUDA while recurrent FIFO/matrix state, attention KV, and scalar operations
remain on CPU. Admission is configuration driven, but only this artifact,
short context, and the pinned trace have acceptance evidence. There is no
tokenizer, chat-template, application resource broker, receipt schema, or
end-to-end product loader in this Candle module.

The native strict CUDA trace passed 49-token prefill, cached decode
`[19, 248046]`, EOS, identical top-10 sets, maximum raw-logprob error
`0.00005269050598144531` against `0.01`, exact reset, and post-cancellation
replay. Short-session CUDA free memory returned to the same observed value
`19,646,119,936` bytes. Evidence and commands are in `PERFORMANCE.md` and
ignored `artifacts/qwen35/cuda-failclosed-parity-1.*`. These are sampled short
proofs, not a 20 GB Edge product ceiling or long-context qualification.

## Current consumer handback and next owner gate

The Edge owner has since built a distinct local Qwen4B short-lab CPU/CUDA
profile and passed short application gates. A symbio-code diagnostic consumer
run passed two 36-token requests with linked receipts. The owner handbacks are
`C:\Users\jc816\Documents\Codex\2026-09-30\task-6\edge-qwen-final-handoff.json`
and `edge-qwen-candle-latency-handback.json`. These are read-only inputs in
this Candle task; no Edge or symbio-code files were edited here. The Qwen
profile does not relabel the maintained LFM Q8 identity.

Edge's first 1,024-token native context replay matched its own tokens and
logits, but took 176.392 s generation versus the existing 90 s Responses
deadline. Model load was recorded separately at 11.310 s. Its exact prompt
token file and gate receipt hashes, plus the Candle replays on the optimized
source, are in `PERFORMANCE.md`. The local Candle release proof now takes
50.62-50.75 s prefill plus about 0.07 s cached decode on that same input,
with unchanged selected IDs/top-10 and checked logprob error below 0.01.
The long reset/cancellation/replay and sampled VRAM checks passed. These
timings were measured in Candle's test binary under unrelated GPU work; they
do not establish Edge's whole Responses latency or normal provider readiness.

The Edge owner should rebuild its native backend against these local Candle
changes and rerun the sealed 1,024-token Responses gate under the existing
90-second deadline, preserving the prompt hash and exact artifact identity.
Record load, prefill, cached decode, application overhead, token/logprob
parity, cancellation, cleanup, RAM/VRAM and receipt links separately. Keep the
short and context lab profiles distinct and the normal alias disabled until
the provider owner reviews the resulting end-to-end gate. Tool bridge,
coding tasks and reserved-output capacity remain separate Edge qualifications.

The symbio-code owner should then replay its diagnostic consumer contract
against the newly qualified Edge build and verify correlation, model,
tokenizer, runtime/build, capacity, cancellation and cleanup receipt fields.
Do not change its maintained `gpt-5.4`/LFM mapping by inference from this
candidate. Exact cross-repository file ownership belongs to the Edge and
symbio-code workers before their concurrent edits or heavy tests.

The qualification work edited only Candle files and activated no product
route. The owner subsequently authorized a source closeout and publication
round. Local checkpoint verification is recorded in STATUS; source publication
awaits a supported secure authentication route. The Edge application gate
remains independently owned.
