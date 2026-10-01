# LFM2.5-VL Current Status

## Current phase and source

Upstream stabilization: integrated pinned Hugging Face main
`aebc405d2b4bf42808387e0ca597bf7dad9b565f` while retaining the fork's public
contracts. Starting clean local/remote main:
`8f27ddfbee47957c274341fd6d32ccabd4767f9f`.
The sync inventory and exact verification results are owned by
`UPSTREAM_SYNC.md`.

Canonical checkout is `C:\DevStuff\candle`, native Windows main. Publish only
with `.tools/gitpush.ps1 -Yes`; secrets remain ignored and unread by the agent.
WSL is a secondary verification lane and never owns Git main.

Compatibility baseline remains Candle 0.11.0
`31f35b147389700ed2a178ee66a91c3cc25cc80d`. Frozen 0.2.0 receipt base
`6f74e7c390c717f8fd34f23ce02aceb058173370` and immutable first-MVP tag
`lfm2-vl-mvp-0.1.0` are unchanged. Historical receipt identities are not
rewritten to the sync revision.

## Proven behavior and ownership

- Config-driven LFM2.5 text, embedding prefill, cached decode, and reset.
  Fallible try_into_config is maintained; deprecated into_config retains its
  original compatibility behavior and requires caller validation.
- SigLIP2 NaFlex, bounded image processing, prompt expansion, pixel unshuffle,
  projection, and multi-image feature replacement.
- Native safetensors, split dense MMProj, direct GGUF MMProj and CPU F32
  native Q8_0 MMProj with deterministic fixtures and cache-reset proof.
- Generic SDXL three-component LoRA transactions/rollback, residual validation,
  and opt-in pooled-text/time-ID conditioning.
- Experimental GPT-OSS short parity and 2,080-token performance acceptance
  remain tied to original source `4a4699981ed55bb11e857c58923c67133559145d`
  and `docs/gpt-oss/STATUS.md`. Larger-context diagnostics are not acceptance.
  The sync does not establish new production parity or change the Q8 default.
- Retained ug public APIs/backend wiring and onig tokenizer selection are
  explicit compatibility differences, independently registered in
  `docs/fork-compat/MOD_MANIFEST.md`.
- Complete ownership is enforced by `docs/FORK_OVERLAYS.md` and its verifier;
  inherited upstream files are not claimed as overlay additions.

## Last green verification and active files

The starting maintained native gate and merged core, nn, transformer, VLM,
test-utils rollback, example, ONNX, strict CPU/CUDA Clippy and bounded GPU
regressions passed. Exact commands, evidence and platform limits are in
`UPSTREAM_SYNC.md`. The guarded helper replays the maintained gate on the
clean delivery commit and records the exact remote identity in ignored
`artifacts/publication/last-push.json`.

For the current local text-adapter candidate, the native release
`quantized_lfm2` test filter passed 3/3 and `lfm2_vl` passed 27/27 on
2026-10-01. Locked/offline CPU checks passed for candle-core, candle-nn,
candle-transformers, candle-vlm, and the quantized-LFM2 example; the bounded
report is `artifacts/qwen35/cuda-lfm-lora-final-cpu-check.json`.
`cargo fmt --all -- --check`, summary-bank, LFM manifest, and rolling overlay
verifiers passed. The first compile failed on an implementation borrow error
and was corrected; the initial non-elevated D: build could not open Cargo's
lock under the sandbox, then the approved D: build passed. The first CPU
check script invocation omitted its required filter and started no job.

Active LFM2-VL local candidate: `candle-transformers/src/models/quantized_lfm2.rs`
and `quantized_lfm2/lora.rs`, with owned manifest, decision, status, backlog,
history, summary-bank, and shared overlay records. A bounded synthetic text
LoRA path retains the GGUF QMatMul base, admits canonical linear targets and
hash-bound base identity, and clears convolution/attention caches on successful
switch. The accepted 2.6B Q8_0 base file at Edge was rehashed locally as
`1e22128dfa128bdfb684da167e74e072d0a056baa7d06d9f280291e2839b0fc9`
(2,874,779,648 bytes); this is an input identity check, not a trained-adapter
or full-model LoRA proof. The independent Qwen3.5 compatibility overlay currently
owns `candle-transformers/src/models/qwen35/`, its artifact-specific fixture,
and the shared registry/overlay records; see `docs/qwen35/STATUS.md` for its
exact 4B CPU/CUDA trace and sealed 1,024-token context evidence. Edge's normal
provider gate remains independently owned. The compatibility manifest accounts
for retained APIs, the new cast regression and the strict-Clippy CUDA
launch-count correction.

## 2026-10-01 closeout verification

The native locked/offline `.tools/verify-before-push.ps1` gate passed on the
complete candidate source: maintained core/nn/transformer/VLM and all three
example checks, strict transformer Clippy, maintained tests (138 transformer
tests passed, five external-model tests ignored), 33 LFM2-VL example tests,
formatting, summary-bank, module-layout, overlay union, and Git whitespace.
The focused LFM manifest separately passed 164 paths (17 fork modifications,
147 additions); the rolling overlay gate passed at the pre-overlay baseline.
The log is retained outside Git as
`%TEMP%/codex-candle-closeout-20261001-gate-final.log`.

Local checkpoint preparation preserves the original model/CUDA receipt source
identities. This closeout ran no production model or CUDA proof. Strict Clippy
required only private observer aliases and equivalent iterator spelling in
the independent Qwen overlay. Source publication awaits a supported secure
authentication route for the repository-owned publication helper.

## Known limitations and blockers

- WSL Codex-Compat lacks its Cargo registry cache (accelerate-src offline
  resolution failed); no current Linux replay claim. Metal/AArch64 are unrun.
- Retained ug needs process-local CUDA 13.0 API selection with installed CUDA
  13.3 libraries; exact build settings and rejected alternatives are documented
  in UPSTREAM_SYNC. Maintained non-ug CUDA passed with normal detection.
- Native 3B and official 400M Q8 MMProj production acceptance remains Gated
  pending separately authorized external artifacts/oracles and cleanup proof.
  The locked revisions and exact contracts remain in TODO/PARITY.
- Consumer repinning, production downloads, release tagging, GPT-OSS 32k,
  broader model/backend support, and post-release inherited panic/stub cleanup
  remain separate tasks. No consumer mutation follows implicitly from sync.
- No trained, immutable LFM2.5-2.6B text adapter and held-out competence
  evidence are available. Candle's adapter SHA label is caller supplied; Edge
  must independently hash the retained serialized adapter bytes and own its
  provenance, session lease, receipt, and recovery checks. CUDA LoRA and
  real-model adapter numerical parity are unrun.

## Exact next task

The focused native text-adapter gate is complete. GenUni and Edge next own
the real immutable adapter, held-out quality, and independently admitted
serving contract in TODO. Do not download new inputs or repin consumers.

---
AI-edited: 2026-10-01T17:47:56+00:00 | agent=Codex/candle_edge_closeout | model=unknown | effort=unknown | task=closeout | change=recorded native gate and remaining publication boundary
