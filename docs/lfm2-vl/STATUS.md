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

Active implementation files: none. Feature-model source and fixture bytes are
unchanged. The compatibility manifest accounts for retained APIs, the new cast
regression and the strict-Clippy CUDA launch-count correction.

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

## Exact next task

Select a separately authorized task from TODO; do not automatically
download production inputs or repin consumers.

---
AI-edited: 2026-09-27 | agent=Codex | model=unknown | effort=unknown | task=upstream-sync | change=reconciled active state and preserved historical receipt boundaries
