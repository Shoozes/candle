# LFM2.5-VL Status

## Baseline

- Upstream: Hugging Face Candle
- Base version: 0.11.0
- Baseline commit: `31f35b147389700ed2a178ee66a91c3cc25cc80d`
- Working branch: `feat/lfm2-vl-mmproj`
- Bootstrap checkpoint: `4a6b30a124abb32b4b275ea8c343ce7ef3ac8be7`
- Source-lock checkpoint: `f007daec10e5751e5899676a7c58098183ec1256`
- Tags: `lfm2-vl-baseline-candle-0.11.0`, `lfm2-vl-phase-0-bootstrap`

## Current Phase

- Phase: 0 — Reference baseline and harness
- Task: Implement config-only, deterministic tiny-random, and explicitly guarded production modes
- Scope: `tools/lfm2_vl/reference/`, the committed tiny fixture, and this status handoff; no Candle Rust, Cargo manifest, source-lock pin, or production tensor payload changes
- Status: phase gate green; checkpoint commit and tag pending

## Source-Lock Results

- Transformers: `fd12552d770f745fdbe41031ff4daa688f5ed57e`
- LiquidAI 450M: `fc6221ca597f3315e4f82fc2df606783267b34ba`
- LiquidAI 1.6B: `919fde3d022e3f90a4716006f993938ee8c2eb97`
- mistral.rs: `8010b6a0578e416120b590ed72fd46ed5f24ee85`
- llama.cpp: `74ce15741b420b8d6f12e720398458b576c51c2c`
- MLX-VLM: `ffd7aeff0bd213c31534a969e0003d49451eef39`
- Transformers.js: `353007be131c2e44d16d46ba49b9a56f2955dfd8`
- Official safetensors metadata: 349 tensors for 450M and 589 for 1.6B; header-only Range reads; zero tensor payload bytes
- Production weights or GGUF files downloaded: none

## Source-Lock Verification

- Date: 2026-08-09 23:11 EDT (`2026-08-10T03:11:06Z` to `2026-08-10T03:11:09Z`)
- Environment: fresh detached Linux-home worktree `/home/workbench/code/candle-lfm2-vl-source-lock-verify`; WSL2 `NVIDIA-Workbench`; CPU-only lane
- Verification HEAD: `4a6b30a124abb32b4b275ea8c343ce7ef3ac8be7` with exactly the six source-lock paths staged
- Command: from `/tmp`, with the existing Linux build cache selected through `CARGO_TARGET_DIR`, `bash /home/workbench/code/candle-lfm2-vl-source-lock-verify/scripts/lfm2-vl/verify-baseline.sh`
- Results: passed `cargo fmt --all -- --check`; locked/offline checks for `candle-core`, `candle-nn`, `candle-transformers`, `lfm2`, and `quantized-lfm2`; staged and unstaged diff checks
- JSON: PowerShell semantic validation and Linux `python3 -m json.tool` passed
- Local-only lockfile SHA-256: `4e059ffe6035520ca6553303932173eba562f4985f82931a90129eea9849ce54`
- Retained log: `artifacts/verification/source-lock/baseline-final.log`; SHA-256 `563a6c97ebf416a7f85ba77296cdc73464b79a73d09238f9dc521d197d94eb6a`

## Reference Harness Verification

- Date: 2026-08-09 23:53–23:58 EDT
- Environment: detached Linux-home worktree `/home/workbench/code/candle-lfm2-vl-reference-verify`; WSL2 `NVIDIA-Workbench`; Python 3.10.12 virtual environment; CPU-only
- Direct pins: Torch `2.8.0+cpu`, safetensors `0.8.0`, Transformers `5.15.0.dev0` from `fd12552d770f745fdbe41031ff4daa688f5ed57e`, huggingface-hub `1.5.0`, tokenizers `0.22.2`, regex `2025.10.22`, Pillow `11.3.0`, pytest `8.4.1`
- Dependency lock: `requirements-reference.txt` contains the complete resolved environment and matches `python -m pip freeze --all` exactly after comments and index directives are removed
- Python tests: `.venv/bin/python -m pytest -q tools/lfm2_vl/reference/test_reference_tools.py` passed, 9 tests in 12.23 seconds
- Config-only proof: both CLIs passed under bare `/usr/bin/python3` with user packages disabled; the inspector also normalized the real pinned 450M `config.json` and `processor_config.json` without downloading weights
- Fixture proof: two independent seed-1234 exports were byte-identical; 87 tensors; raw source-image SHA-256 `08359b108fa567f5dcf319fa3434da6abbc1d595f426372666447f09cc5a87dc`
- Fixture files: `tensors.safetensors` 61,072 bytes, SHA-256 `d4ccbd62ebd8afdecb6207fe341a0880e74c5dda8deea680a08366ece4ec96c3`; `metadata.json` 2,422 bytes, SHA-256 `3add6fa29206fe2b404f3e0959d3c51d69046eacab3f1ebb19dac43964cb0199`; `manifest.json` 8,485 bytes, SHA-256 `c5461dadb0edfd920b20f308650c59676977110a1cc2f199e317dea7d75bdd7b`
- Fixture coverage: official class construction, padded packed patches, exact resized positions, both vision encoder layers, post-LN, factor-2 pixel unshuffle, optional projector LayerNorm, both projector linears, image-placeholder replacement, multimodal prefill logits, and three cached decode steps
- Rust baseline: locked/offline CPU verification passed from `2026-08-10T03:53:09Z` to `2026-08-10T03:53:12Z`, including formatting, core crates, dense LFM2, quantized LFM2, and staged/unstaged diff checks
- Retained logs: `artifacts/verification/reference-harness/python-final.log`, SHA-256 `fb46403e3315b49dcfb424b07fc062f0b2366f45ec627aa1bdd64b006bcf2c93`; `artifacts/verification/reference-harness/baseline-final.log`, SHA-256 `b08863fe6c0e6a33d86812e8d95d601e51654bcea65d4f51f6aefadb59e1d3d8`

## Bootstrap Proof

- Date: 2026-08-09 22:35 EDT (`2026-08-10T02:35:09Z` to `2026-08-10T02:35:12Z`)
- Environment: WSL2 `NVIDIA-Workbench`; Ubuntu 22.04.5 LTS; Linux `6.6.87.2-microsoft-standard-WSL2`; CPU-only lane
- Verification HEAD: `31f35b147389700ed2a178ee66a91c3cc25cc80d` with bootstrap paths staged in the detached Linux verification worktree
- Command: from `/tmp`, `bash /home/workbench/code/candle-lfm2-vl-verify/scripts/lfm2-vl/verify-baseline.sh`
- Results: passed formatting; locked/offline checks for `candle-core`, `candle-nn`, `candle-transformers`, `lfm2`, and `quantized-lfm2`; staged and unstaged diff checks
- Local-only lockfile SHA-256: `4e059ffe6035520ca6553303932173eba562f4985f82931a90129eea9849ce54`
- Retained baseline log SHA-256: `a4f77d1b007eb267865be01ef1c239754ac0e093dd1c27ad457d77242b614f22`
- Retained environment log SHA-256: `5f4fd70b4dd5ca6a956c9678d386598ca2ff6bcdb2e75ef3ba3aa6a10775e4d8`

## Environment Snapshot

- WSL2 distribution: `NVIDIA-Workbench`
- Linux home: `/home/workbench`
- Rust compiler: `rustc 1.97.1 (8bab26f4f 2026-07-14)`
- Cargo: `cargo 1.97.1 (c980f4866 2026-06-30)`
- Python: `Python 3.10.12`
- CMake: `cmake 3.22.1`
- Ninja: missing; optional for current gates
- Reference Python: isolated `.venv` with a complete checked-in CPU-lane dependency lock

## Proven

- The untouched Candle 0.11 baseline and existing dense/quantized LFM2 examples compile in the locked, offline CPU lane.
- Every external implementation and model reference is pinned to an immutable revision with path, purpose, authority, license, and adaptation boundary.
- Both official checkpoint tensor namespaces and representative shapes were read from safetensors headers without reading tensor payloads.
- The 450M effective FFN width is 4,608 and the 1.6B width is 8,192; the production headers confirm both.
- Both checkpoints omit `lm_head.weight`, confirming the required tied-output loading path.
- The source-lock patch changes no Candle Rust source, Cargo manifest, lockfile policy, or runtime dependency.
- The pinned official Transformers LFM2, SigLIP2, and LFM2-VL classes execute deterministically in the tiny-random harness, including multimodal prefill and incremental cache reuse.
- Config-only mode is stdlib-only and production model loading remains guarded by explicit production, load, and download flags.
- The tiny fixture deliberately omits the tied `lm_head.weight` duplicate, preserving the production checkpoints' missing-head loading contract.

## Known Conflicts

- Official config context is 128,000 while model cards advertise 32,768; construction follows config and production policy remains unresolved.
- Numeric IDs for image wrapper, row/column, and thumbnail marker strings must be exported by the tokenizer harness; only image placeholder ID 396 is config-explicit.
- llama.cpp PR #25524 for reading LFM2 tiling parameters from GGUF metadata is open and unmerged; official processor config remains authoritative.
- Physical GGUF tensor orientation beyond the converter-defined patch reshape awaits header-only inspection of a pinned GGUF.

## Blockers

- None for the reference-harness phase.

## Active Files

- `tools/lfm2_vl/reference/README.md`
- `tools/lfm2_vl/reference/requirements-reference.in`
- `tools/lfm2_vl/reference/requirements-reference.txt`
- `tools/lfm2_vl/reference/export_fixtures.py`
- `tools/lfm2_vl/reference/inspect_config.py`
- `tools/lfm2_vl/reference/manifest.py`
- `tools/lfm2_vl/reference/tensor_dump.py`
- `tools/lfm2_vl/reference/test_reference_tools.py`
- `tests/fixtures/lfm2_vl_tiny/README.md`
- `tests/fixtures/lfm2_vl_tiny/metadata.json`
- `tests/fixtures/lfm2_vl_tiny/manifest.json`
- `tests/fixtures/lfm2_vl_tiny/tensors.safetensors`
- `docs/lfm2-vl/STATUS.md`

## Next Task

Phase 1 — repair text-only dense and quantized LFM2 configuration, add embedding-driven forward paths, and prove normalized FFN widths, prefill, incremental decode, and legacy compatibility before starting vision work.

---
AI-edited: 2026-08-09T23:58:47-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=reference-harness | change=recorded green deterministic fixture and local verification proof
