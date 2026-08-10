# LFM2.5-VL Status

## Baseline

- Upstream: Hugging Face Candle
- Base version: 0.11.0
- Baseline commit: `31f35b147389700ed2a178ee66a91c3cc25cc80d`
- Working branch: `feat/lfm2-vl-mmproj`
- Bootstrap checkpoint: `4a6b30a124abb32b4b275ea8c343ce7ef3ac8be7`
- Source-lock checkpoint: `f007daec10e5751e5899676a7c58098183ec1256`
- Reference-harness checkpoint: `a9594101c97589f6deabe7a2dddaaffeb5471a94`
- Phase 1 checkpoint: `f660b8e3f2b4560f133356864e012be83f29d9c0`
- Phase 3 checkpoint: `37264b49cf74d0cf7697317eda0183f084db6ff8`
- Tags: `lfm2-vl-baseline-candle-0.11.0`, `lfm2-vl-phase-0-bootstrap`, `lfm2-vl-phase-0-reference`, `lfm2-vl-phase-1-text`, `lfm2-vl-phase-2-siglip2`, `lfm2-vl-phase-3-native-composite`

## Current Phase

- Phase: 4 — Rust Processor and Prompt Expander
- Task: Remove the Python runtime preprocessing dependency by turning raw images and user-positioned image sentinels into the packed tensors, exact token IDs, and one placeholder span per crop consumed by the proven native composite model
- Scope: the new `candle-vlm` workspace crate, dynamic processor config and precedence, RGB conversion, TorchVision-compatible antialiased resize, smart resize, tiling, thumbnails, normalization, patchification/padding, exact metadata, tokenizer-backed prompt expansion, multiple images, and controlled malformed-input failures; no GGUF loader, production model download, CUDA-specific optimization, or CLI
- Status: The Rust processor/prompt suite is green at 24/24, all 12 official processor cases match, all 5 prompt cases match exact text/IDs/per-crop spans, all 10 real-dimension metadata oracles are consumed, the pinned Python oracle is green at 9/9, `candle-transformers` remains green at 37/37, and the final staged baseline is green. The Phase 4 checkpoint is pending.

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

## Text Compatibility Verification

- Date: 2026-08-10 00:31–00:32 EDT
- Environment: detached Linux-home worktree `/home/workbench/code/candle-lfm2-vl-text-verify`; WSL2 `NVIDIA-Workbench`; CPU-only, locked, offline lane; staged Phase 1 candidate based on `a9594101c97589f6deabe7a2dddaaffeb5471a94`
- Implementation scope: config aliases and normalization, checked `try_into_config` plus the preserved `into_config` API, standalone/nested dense roots, tied/explicit output heads, dense hidden/logit APIs, quantized embedding-driven forwarding, and cache clearing
- Focused command: `cargo test --locked --offline -p candle-transformers lfm2 -- --nocapture` passed all 5 focused tests plus filtered integration binaries
- Focused proof: official 450M/1.6B effective FFN widths, legacy aliases/fallback/precedence, standalone and nested roots, explicit and tied heads, dense token-ID versus embedding forwarding, committed-fixture merged prefill parity, three cached decode steps, cache-reset determinism, and quantized embedding-driven equivalence
- Maximum absolute errors: token embeddings `0`; token-ID versus embedding-driven output `0`; prefill hidden states `2.38418579e-7`; prefill logits `2.98023224e-8`; cached decode steps `1.86264515e-8`, `2.98023224e-8`, and `1.49011612e-8`; reset prefill hidden states `2.38418579e-7`; reset decode `1.86264515e-8`
- Focused log: `artifacts/verification/text-compatibility/focused-tests.log`; SHA-256 `022e7e7fed20f0b255424d04334108c419b9aee49e722132a6563dff4b67d034`
- Broader library command: `cargo test --locked --offline -p candle-transformers --lib` passed all 18 tests; retained log `artifacts/verification/text-compatibility/candle-transformers-lib.log`; SHA-256 `cc8faae1c192569ba3952e14e9bbcbc2a5ad3a24132e0cf20424eb12b0606db3`
- Full baseline: `scripts/lfm2-vl/verify-baseline.sh` passed from `2026-08-10T04:32:23Z` to `2026-08-10T04:32:51Z`, including formatting, `candle-core`, `candle-nn`, `candle-transformers`, the `lfm2` and `quantized-lfm2` examples, and staged/unstaged diff checks
- Baseline log: `artifacts/verification/text-compatibility/baseline-final.log`; SHA-256 `c72eccd8b77689878689f7e720c46a040c26f3cee8060b17727392f392862f46`
- Local-only lockfile SHA-256: `4e059ffe6035520ca6553303932173eba562f4985f82931a90129eea9849ce54`; no production weights or GGUF files were downloaded

## SigLIP2 NaFlex Verification

- Environment: manager Linux-home WSL2 `NVIDIA-Workbench`; CPU-only F32 lane
- Focused command: `cargo test --locked --offline -p candle-transformers siglip2 -- --nocapture`
- Result: 7/7 focused tests passed
- Broader library command: `cargo test --locked --offline -p candle-transformers --lib`; 25 passed, 0 failed
- Broader library log: `artifacts/verification/siglip2/candle-transformers-lib.log`; SHA-256 `b4c91d4bd6a0c1850a66d9cc27d61776b5ec96c152783e6c13f23a0cfcdf5197`
- Focused log: `artifacts/verification/siglip2/focused-tests.log`; SHA-256 `d09ec6bdf1d110711f347b89bba353eeb8ebb6172e8a45d19d3edd1aeb254645`
- Full baseline: passed from `2026-08-10T05:14:56Z` to `2026-08-10T05:15:08Z`, including `cargo fmt`, `candle-core`, `candle-nn`, `candle-transformers`, `lfm2`, `quantized-lfm2`, and both diff checks
- Full baseline log: `artifacts/verification/siglip2/baseline-final.log`; SHA-256 `727e0d8a029f121a7225d3d35a53addd480791323b6e0c501576408cc6460d52`
- Phase 2 checkpoint/tag: complete at commit `74e109aec5f9801cfead3eeb27fe3f93ac646b84`, annotated tag `lfm2-vl-phase-2-siglip2`
- Gate: positional max absolute error `<=2e-5`; vision cosine similarity `>=0.99999`
- Stage evidence: patch projection max abs `5.960464478e-8`, cosine `0.999999940`; resized positions `2.980232239e-8`, cosine `0.999999940`; embedding sum `1.192092896e-7`, cosine `1.000000119`; encoder layer 0 `4.768371582e-7`, cosine `0.999999881`; encoder layer 1 `1.192092896e-6`, cosine `0.999999881`
- Final evidence: returned post-LN `7.152557373e-7`, cosine `1.000000119`; post-LN hook matched the same result; padding-key isolation max abs `0`, cosine `1`
- Proven behavior: packed patch projection with bias, CPU F32 separable antialiased positional interpolation and per-shape cache, bidirectional key masking, F32 score/softmax with original-dtype value matmul, configured encoder activation, post-LN, checked malformed-input handling, and controlled exclusion of the vision pooling head
- Production weights or GGUF files downloaded: none

## Phase 3 Verification

- Phase 2 checkpoint: complete at commit `74e109aec5f9801cfead3eeb27fe3f93ac646b84`, annotated tag `lfm2-vl-phase-2-siglip2`
- Implementation proven in scope: dynamic top-level config, factor-N official pixel-unshuffle, optional projector LayerNorm, linear/GELU/linear projection, crop unpadding/ranges/order, strict one-span-per-image exact-length merge, multimodal prefill, ordinary cached decode, cache reset, and `EncodedImages`
- Focused Phase 3 gate: 11/11 passed; retained log `artifacts/verification/native-composite/focused-tests.log`; SHA-256 `7d727e1b8558f1f242ce940c8af36d44a3e292f4ffa023d1ff124ccf2cc13638`
- Maximum absolute errors: projector stages `<=5.960464478e-8`; encoded and merged embeddings `<=6.519258022e-9`; prefill logits `<=4.470348358e-8`; cached decode `<=2.980232239e-8`
- SigLIP2 repeated-crop regression: 8/8 passed; retained log `artifacts/verification/native-composite/siglip2-regression.log`; SHA-256 `5684568b060c6338f3e5d8bc94361d37bc64ddf84584ad4a5e05915acc275f38`
- Runtime defect resolved: batched SigLIP2 attention received a non-contiguous transposed left-hand operand and failed with `MatMulUnexpectedStriding`; `split_heads` now materializes a contiguous tensor, protected by the repeated-crop regression
- `candle-transformers` library gate: 37/37 passed; retained log `artifacts/verification/native-composite/candle-transformers-lib.log`; SHA-256 `0f36d6a8d54f77abfe9c5031075b7174cff83859315d0997f60a1a399f475497`
- Full locked/offline CPU baseline: passed `2026-08-10T05:48:07Z`–`2026-08-10T05:48:10Z` against pre-Phase-3-checkpoint HEAD `74e109aec5f9801cfead3eeb27fe3f93ac646b84`; retained log `artifacts/verification/native-composite/baseline-final.log`; SHA-256 `47d984dd3afe7b92b6a72bcdb93e7d9da99bd8673e5c1067b8f1fac7ed2b8b45`
- Cargo.lock SHA-256: `4e059ffe6035520ca6553303932173eba562f4985f82931a90129eea9849ce54`
- Phase 3 checkpoint/tag: complete at commit `37264b49cf74d0cf7697317eda0183f084db6ff8`, annotated tag `lfm2-vl-phase-3-native-composite`
- Not claimed: production-checkpoint parity, CUDA, GGUF, raw-image preprocessing, tokenizer/chat template, or CLI support

## Phase 4 Verification

- Phase 3 checkpoint/tag: complete at commit `37264b49cf74d0cf7697317eda0183f084db6ff8`, annotated tag `lfm2-vl-phase-3-native-composite`
- Implementation proven in scope: parsed processor JSON, `explicit > processor JSON > GGUF metadata > model config > defaults` precedence, fixed padding including explicit `max_num_patches`, RGB/grayscale/RGBA conversion, checked smart resize and tile selection, row-major crops, optional thumbnail, TorchVision-compatible byte resize, normalization, patchification, masks/shapes, image/crop metadata, tokenizer-resolved special tokens, sentinel preservation, exact per-crop spans, multiple images, context checks, and controlled invalid-input errors
- Rust processor/prompt gate: 24/24 passed; retained log `artifacts/verification/processor-prompt/candle-vlm-tests.log`; SHA-256 `ddf2b1b311ccd849a9491342bb1c5a82b5c528fac3bde826e6cebd6ffcc66702`
- Processor fixture: all 12 required cases passed; integer masks, spatial shapes, image grids/sizes, crop ranges/order/kinds, and prompt IDs/spans were exact; worst pixel-value maximum absolute error `1.192092896e-7`, cosine similarity `1.0`
- Direct resize regression: the complete 7×5 RGB to 8×4 byte output matches pinned TorchVision in all 96 channel values; boundary cases and unrepresentable allocation requests are also covered
- Real-dimension gate: all 10 pinned cases assert smart dimensions, large-image decision, selected grid, tile canvas, and whole/tile/thumbnail order
- Fixture reproduction: a fresh pinned-oracle export is byte-identical to the checked-in fixture. Manifest SHA-256 `2fb787e378f5fd1ddfa147913aadccd07add9a1045b8bb0f693ca2c2f564959c`; metadata `aca7f4d5e5e4ef0e4872adeb227b56cf3960d87b353c40162af97660783f2327`; tensors `a25932fc57f3e78f48a1a8f558216521c7ae3e8659fcf0a389cd0a4ebe0ab3f6`
- Fixture logs: regeneration SHA-256 `fbb7a0bbb247d29f8bd8725d8b48cd971ed1a3768a2dacb75300b19f3696b955`; checked-in/regenerated hash comparison `84aa570f6da7995977e424ae8d61491ec07cdfbdad0af7847d77604d74a492eb`
- Pinned Python reference tests: 9/9 passed in 11.83 seconds; retained log SHA-256 `00f0fac4bd730862e95945ef357d402ee88473b61fce0fff6b6614930eb615ee`
- `candle-transformers` regression: 37/37 passed, including exact encoded image/crop range-union validation; retained log SHA-256 `78fa7ede014c72ca5d7defd6833c8152be2e3b0455d635ad79b122788452b560`
- Independent worker re-audit: five actionable findings were resolved and the initially reported resize mismatch was withdrawn after exact reproduction showed it used no matching source/coordinate convention
- Full locked/offline CPU baseline: passed `2026-08-10T08:35:14Z`–`2026-08-10T08:35:25Z` against pre-Phase-4-checkpoint HEAD `37264b49cf74d0cf7697317eda0183f084db6ff8` with exactly the Phase 4 candidate staged; retained log `artifacts/verification/processor-prompt/baseline-final.log`; SHA-256 `fb16481302c9bdf15f4b04df0250e45bf8c9a2126b92b09a787f5360cc3a3140`
- Verifier-only Cargo.lock SHA-256 after adding `candle-vlm`: `a4f4379f73d38db1a148f96538e6b868d1ef148a8417069807bae451c9766fb4`
- Phase 4 checkpoint: pending manager commit/tag
- Not claimed: production-checkpoint numerical parity, GGUF/mmproj loading, CUDA, generated-caption parity, or CLI support

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
- The complete Phase 1 text gate passes in the Linux-home CPU/offline lane: all 5 focused LFM2 tests, all 18 `candle-transformers` library tests, both existing LFM2 examples, and the full locked/offline baseline are green.
- Phase 1 is checkpointed at `f660b8e3f2b4560f133356864e012be83f29d9c0` and tagged `lfm2-vl-phase-1-text`.
- The focused Phase 2 SigLIP2 gate is green: all 7 tests pass with the exact stage errors recorded above; the Phase 2 checkpoint/tag is complete at `74e109aec5f9801cfead3eeb27fe3f93ac646b84` / `lfm2-vl-phase-2-siglip2`.
- The Phase 3 focused gate is green at 11/11, the SigLIP2 repeated-crop regression is green at 8/8, and the `candle-transformers` library gate is green at 37/37. Phase 3 is checkpointed at `37264b49cf74d0cf7697317eda0183f084db6ff8` and tagged `lfm2-vl-phase-3-native-composite`.
- The Phase 4 Rust-native raw-image and prompt path is green at 24/24 against all required pinned fixtures. Packed integer metadata, exact prompt strings/IDs/spans, crop ordering, and fixture regeneration are exact; normalized pixel values differ by at most `1.192092896e-7`.
- Tiny-fixture dense parity is within `2.38418579e-7` for hidden states and `2.98023224e-8` for logits; production-checkpoint and GGUF numerical parity remain unclaimed.

## Known Conflicts

- Official config context is 128,000 while model cards advertise 32,768; construction follows config and production policy remains unresolved.
- Numeric IDs for image wrapper, row/column, and thumbnail marker strings must be exported by the tokenizer harness; only image placeholder ID 396 is config-explicit.
- llama.cpp PR #25524 for reading LFM2 tiling parameters from GGUF metadata is open and unmerged; official processor config remains authoritative.
- Physical GGUF tensor orientation beyond the converter-defined patch reshape awaits header-only inspection of a pinned GGUF.

## Blockers

- None. The Phase 4 checkpoint remains pending manager commit/tag.

## Active Files

- `candle-transformers/src/models/lfm2.rs`
- `candle-transformers/src/models/quantized_lfm2.rs`
- `candle-transformers/src/models/siglip2.rs`
- `candle-transformers/src/models/lfm2_vl/`
- `candle-vlm/`
- `candle-transformers/src/models/mod.rs`
- `candle-examples/examples/lfm2/main.rs`
- `tests/fixtures/lfm2_vl_processor_tiny/`
- `tools/lfm2_vl/reference/export_processor_fixture.py`
- `docs/lfm2-vl/DECISIONS.md`
- `docs/lfm2-vl/PARITY.md`
- `docs/lfm2-vl/STATUS.md`

## Next Task

Create the Phase 4 checkpoint/tag. Then begin Phase 5 hybrid loading without combining it with direct GGUF mmproj loading. Keep production-checkpoint, CUDA, generated-caption, and CLI claims separately labeled.

---
AI-edited: 2026-08-10T04:35:42-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=lfm2-vl-phase-4-docs | change=recorded Rust processor, prompt, fixture, audit, and final baseline evidence
