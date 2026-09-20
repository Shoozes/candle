# GPT-OSS Experimental Candle Status

## Assignment state

- Boundary: C0 overlay registration/model-free rejection, C1a packed MXFP4
  storage/loading plus CPU expert reference, and C1b synthetic forward/cache
  proof.
- Baseline: `e87851e4608d4770e3c5aaa0a9d7a7c0447a4345` on local `main`.
- Source revision: `6a43f5289f98f135d3406a3b957e1d94a22c3cae`.
- Delivery: C0, C1a, and C1b are committed. Repository policy, manifests,
  guarded publication tooling, and this source-bound status follow in the
  later delivery commit; no real-checkpoint or CUDA claim is added.
- Acknowledgement: prepared for independent EdgeSymbio source-graph,
  compatibility, consumer, and canonical review.

## Proven evidence

- Synthetic fixture/reference ID: `synthetic-gpt-oss-c1a-v1`.
  Deterministic in-source packed blocks/scales, CPU nibble/scale decode,
  selected matrix multiply, expert routing, SwiGLU, and residual parity.
- Synthetic forward/cache ID: `synthetic-gpt-oss-c1b-cache-v1`.
  One tiny 1-layer/2-expert CPU model proves uncached final-token logits equal
  prompt-plus-incremental cached logits and that reset replay is identical.
- Safetensors identity: generated in-memory test file only; no artifact or
  checkpoint identity is claimed.
- Config identity: tiny synthetic config in `model.rs` tests only; no external
  `config.json` was loaded.
- Tokenizer identity: none; no tokenizer was loaded.
- Real-checkpoint evidence: none.
- CUDA evidence: none; the implementation is CPU reference-only.

## Public boundary

`GptOssCheckpoint` rejects missing/model-free local directories. `GptOssConfig`
validates architecture dimensions. `PackedMxfp4` retains U8 blocks/scales and
does not expose a dense-dequant residency override. `Mxfp4ExpertOperation`
performs the CPU reference operation. `GptOssModel` provides synthetic
forward/cache proof with bounded append-only rollback on failed forwards; no
production model loader is exposed at this boundary.

## Limitations and next dependency

The production safetensors weight inventory, tokenizer/chat format, sharding,
model loader, real CPU parity, memory receipt, and CUDA kernels remain
unimplemented and unclaimed. The next dependency is an owner-admitted,
hash-pinned GPT-OSS checkpoint/config/tokenizer snapshot plus independent
reference traces. Q8 remains the maintained product default; GPT-OSS remains
experimental.

## Verification

Native Windows/MSVC with locked offline dependencies:

- `cargo test --locked --offline -p candle-transformers gpt_oss`: 14/14 passed.
- `cargo test --locked --offline -p candle-transformers --lib`: 105/105
  passed, including the existing LFM2/Q8 tests.
- `cargo check --locked --offline -p candle-core -p candle-nn
  -p candle-transformers -p candle-vlm`: passed.
- `cargo check --locked --offline -p candle-examples --example lfm2`,
  `quantized-lfm2`, and `lfm2-vl`: all passed.
- Warnings-denied `cargo clippy --locked --offline -p candle-transformers
  --lib -- -D warnings`, `cargo fmt --all -- --check`, and `git diff --check`:
  passed.
- Native Git-for-Windows `scripts/gpt-oss/verify-mod-manifest.sh` and
  `scripts/verify-fork-overlays.sh`: passed; PowerShell summary-bank verifier:
  passed.
- Candle `gitpush.ps1` integration fixtures: 4/4 passed for verifier failure,
  verifier drift, remote race, and exact successful publication/receipt.

No model download, hosted execution, real-checkpoint run, or CUDA execution
was used.

---

AI-edited: 2026-09-19; agent=Codex; task=guarded-publication; change=bound C0/C1 source revision and publication proof
