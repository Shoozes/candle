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
- Phase 2 checkpoint: `74e109aec5f9801cfead3eeb27fe3f93ac646b84`
- Phase 3 checkpoint: `37264b49cf74d0cf7697317eda0183f084db6ff8`
- Phase 4 checkpoint: `8d1bbe471404848730685c98e7dd56b13a457eb4`
- Phase 5 checkpoint: `1535a0a5fef09f243811b83553b9c75baad78ee2`
- Phase 6 checkpoint: `dc6321cde3a9b8a019c1f0fd82780d90afa046df`
- Phase 7 checkpoint: `1a9cf291ba9a1bbac1de83029f9d8f057aca00b5`
- Tags: `lfm2-vl-baseline-candle-0.11.0`, `lfm2-vl-phase-0-bootstrap`, `lfm2-vl-phase-0-reference`, `lfm2-vl-phase-1-text`, `lfm2-vl-phase-2-siglip2`, `lfm2-vl-phase-3-native-composite`, `lfm2-vl-phase-4-native-e2e`, `lfm2-vl-phase-5-hybrid`, `lfm2-vl-phase-6-gguf`, `lfm2-vl-phase-7-q8`

## Current Phase

- Phase: Post-Phase 7 Stabilization — Bounded Oracle Complete; Official 450M CPU-F32 Next
- Task: Publish the locally verified NR-4/NR-5A checkpoint for review, then execute official 450M native CPU-F32 component parity
- Scope: deterministic native/split/direct inference evidence, immutable llama.cpp bundle identity, suspended Job Object containment, and local-only verification; no new model execution, hosted verification, GitHub Actions, PR, or secret inspection in NR-5A
- Status: The runner implementation, exact file-level native/split/direct provenance, GGUF EOS metadata, 26/26 focused example tests, 32/32 focused transformer/LFM2 tests, full core/transformer/VLM tests, strict scoped Clippy, full baseline, exact mod/fork classification, local same-artifact decoded output, and the harmless bounded-oracle process-tree smoke suite are green. The exact pinned llama.cpp b10335 owner build passes a bounded no-model identity probe. Official-base payload parity remains unclaimed. The designated independent NR-4 worker could not start because its Codex agent loop died; the manager audit closed the direct-provenance coverage gap, but no independent NR-4 verdict is claimed.

## Source-Lock Results

- Transformers: `fd12552d770f745fdbe41031ff4daa688f5ed57e`
- LiquidAI 450M: `fc6221ca597f3315e4f82fc2df606783267b34ba`
- LiquidAI 1.6B: `919fde3d022e3f90a4716006f993938ee8c2eb97`
- mistral.rs: `8010b6a0578e416120b590ed72fd46ed5f24ee85`
- llama.cpp: `74ce15741b420b8d6f12e720398458b576c51c2c`
- MLX-VLM: `ffd7aeff0bd213c31534a969e0003d49451eef39`
- Transformers.js: `353007be131c2e44d16d46ba49b9a56f2955dfd8`
- Official safetensors metadata: 349 tensors for 450M and 589 for 1.6B; header-only Range reads; zero tensor payload bytes
- Official 450M MMProj GGUF: `166cd80bbe157dc86d65f964eb8cc6a2cede62ca`; F16 and Q8_0 headers each 12,736 bytes, 32 metadata records, 201 tensors, zero retained tensor payload bytes
- Production weight payloads or complete GGUF files downloaded by this project: none. NR-4 downloaded only the pinned official 450M `tokenizer.json`; the complete GGUF files under `C:\llamacpp` pre-existed this worktree.

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
- Phase 4 checkpoint/tag: complete at commit `8d1bbe471404848730685c98e7dd56b13a457eb4`, annotated tag `lfm2-vl-phase-4-native-e2e`
- Not claimed: production-checkpoint numerical parity, GGUF/mmproj loading, CUDA, generated-caption parity, or CLI support

## Phase 5 Verification

- Phase 4 checkpoint/tag: complete at commit `8d1bbe471404848730685c98e7dd56b13a457eb4`, annotated tag `lfm2-vl-phase-4-native-e2e`
- Split artifact contract: deterministic `mmproj.safetensors`, versioned `mmproj.json`, and canonical `processor_config.json`; exact config-derived inventory; immutable source model/revision; SHA-256 coverage; atomic per-file writes; overwrite refusal; no production weights
- Loader contract: manifest, processor, architecture, text width/layer count, vision layer inventory, patch/factor, tokenizer image ID, tensor names/shapes/dtypes/byte counts, header size, tensor count, offsets, overlaps, gaps, and payload coverage are checked before tensor construction
- File identity and allocation: the weights file is opened and buffered once; the same bytes are hashed, inspected, and consumed by `VarBuilder`; the maximum allocation is derived with checked arithmetic from the validated manifest payload plus the bounded header and prefix
- Quantized path: deterministic real GGUF bytes are written from the committed tiny text tensors, hash-pinned as `8fbd510aeea4715547c57975a7adcb91c148a8bc5e8d869d9617b69af6a006b1`, parsed through `gguf_file::Content::read`, and loaded through `ModelWeights::from_gguf`; block-aligned matrices use Q8_0 and small unalignable tensors remain F32
- Numerical evidence: split versus unified image features max abs `0`; quantized hybrid prefill logits max abs `4.457309842e-5`; cached decode steps `2.650916576e-5`, `2.175569534e-5`, and `1.309439540e-5`; cache reset max abs `0`
- Python reference/exporter suite: 19/19 passed from `2026-08-10T10:01:59Z` to `2026-08-10T10:02:12Z`; retained log `artifacts/verification/hybrid-mmproj/python-final.log`; SHA-256 `96e17862334d3fc7877596cafb3daa743c58904dcca11a8b7e205138d40dab85`
- `candle-transformers`: 42/42 library tests plus 5 generation and 8 NMS integration tests passed from `2026-08-10T10:02:23Z` to `2026-08-10T10:02:26Z`; retained log `artifacts/verification/hybrid-mmproj/candle-transformers-tests.log`; SHA-256 `aba483c8bc2c01b4fd6ecc6c511dfc93831280933596b0ffad9660a3bd22b529`
- `candle-vlm` and example: 25/25 tests and `cargo check -p candle-examples --example lfm2-vl` passed from `2026-08-10T10:02:39Z` to `2026-08-10T10:02:48Z`; retained log `artifacts/verification/hybrid-mmproj/vlm-example-final.log`; SHA-256 `b431fa113f047ec0981f93e9a3ce581dc3ac9638b0185f4b57fa81aab0780eb7`
- Clippy: transformer, VLM, and example gates passed with only the recorded pre-existing newer-Clippy allowances; retained log `artifacts/verification/hybrid-mmproj/clippy-final.log`; SHA-256 `d2e79ade9bab2e9115a4ce526c1ed16f72622f3230f98939132c5e82124752d6`
- Distinct-device lane: source-complete CUDA-vision/CPU-text test verifies that raw packed inputs remain caller-owned, projected features stay on the vision device until merge, logits return on the text device, and prefill agrees within `1e-4`; local execution was skipped at the `cudarc` build because `nvcc` is absent despite an RTX 4090 driver at compute capability 8.9; retained log `artifacts/verification/hybrid-mmproj/cuda-distinct-device-skipped.log`; SHA-256 `6c16f84a925c9bb2d856b18919cc9fae45a69b38bcf7d4db65b44396f011eb97`
- Independent worker re-audit: all nine implementation findings are resolved; the worker confirmed no code blocker remains and classified the CUDA result as an environment-owned evidence gap
- Full locked/offline staged CPU baseline: passed `2026-08-10T10:07:11Z`–`2026-08-10T10:07:19Z` against pre-Phase-5-checkpoint HEAD `8d1bbe471404848730685c98e7dd56b13a457eb4` with exactly the Phase 5 candidate staged; retained log `artifacts/verification/hybrid-mmproj/baseline-final.log`; SHA-256 `594932158f4a99702cedd47ced1fe0ffd8e4aa18835346e65a0f9003963bc369`
- Verifier-only Cargo.lock SHA-256: `acd9419056b786da820b5120db8e78be06902721689c39b55f29445abdddaffc`
- Phase 5 checkpoint/tag: complete at commit `1535a0a5fef09f243811b83553b9c75baad78ee2`, annotated tag `lfm2-vl-phase-5-hybrid`
- Not claimed at Phase 5: production-checkpoint numerical parity, production GGUF numerical parity, direct GGUF mmproj compatibility, executed CUDA parity, generated-caption parity, or quantized vision execution

## Phase 6 Verification

- Phase 5 checkpoint/tag: complete at commit `1535a0a5fef09f243811b83553b9c75baad78ee2`, annotated tag `lfm2-vl-phase-5-hybrid`
- Official header evidence: `LiquidAI/LFM2.5-VL-450M-GGUF@166cd80bbe157dc86d65f964eb8cc6a2cede62ca`; exact range `bytes=0-12735`; header end 12,708; aligned tensor-data offset 12,736; 32 metadata records; 201 tensors; tensor-name SHA-256 `45e3f6cf0b51dc9f5e458b8af3375d368cc59daff70b79e2938c7490a94df828`; zero retained payload bytes
- Official dtype evidence: F16 file 75 F16 plus 126 F32 tensors, prefix SHA-256 `338099d49dd803963c9496cfbba56ab46a425ca7895c5edf59010337ae4436ac`; Q8_0 file 74 Q8_0 plus 127 F32 tensors, prefix SHA-256 `7a4f0f1e168d52b70a03f2773f0f20b9f65d1692f8e973aa0cf9ecee25e43d1c`
- Loader contract: one stable handle; required `clip/mmproj/lfm2` metadata; exact config-derived 201-tensor production inventory; paired optional input LayerNorm/projector biases; supported target dtypes F32/F16/BF16; no Phase 7 quantized operator
- Security boundary: caller-specific GGUF limits apply before allocation (16,384 tensor, metadata, and array records; 1 MiB strings; 16 MiB aligned header), followed by checked 8 GiB file, retained-dense, and conservative peak-allocation bounds plus alignment/offset/overlap/truncation validation
- Orientation: official headers prove every non-patch matrix is already in Candle `[out,in]`; only patch `[V,3,P,P]` is converted with `permute(0,2,3,1)`, contiguous, and reshape to `[V,3P²]`
- Processor/prompt boundary: absent GGUF tiling keys retain pinned official architecture defaults (2–10 tiles, thumbnail, 512 tile size, 64–256 image tokens, effective 1,024 packed patches); direct GGUF uses image markers and resolves image token ID through the tokenizer
- Deterministic dense MMProj GGUF SHA-256: `7361b57e6d9dbf2d7809d4f446944fdc7325b368e4444fee2bc3497376695256`
- Numerical evidence: dense direct/native image features max abs `0`; Q8_0-dequantized/dense image features `8.463021368e-5`; direct prefill `4.457309842e-5`; cached decode `2.650916576e-5`, `2.175569534e-5`, and `1.309439540e-5`; cache reset `0`
- Full Python reference suite: 23/23 passed `2026-08-10T11:46:49Z`–`2026-08-10T11:47:03Z`; retained log `artifacts/verification/gguf-mmproj/python-final.log`; SHA-256 `508b999476bf3dac595479f29d41f3831698c23395ab68f02f17c43c537ae998`
- Full Rust gate: `cargo test --locked --offline -p candle-core -p candle-transformers -p candle-vlm` passed `2026-08-10T11:43:47Z`–`2026-08-10T11:44:06Z`; this includes 21/21 core library tests, all core integration/doc tests, 47/47 transformer library tests, 5/5 generation, 8/8 NMS, and 26/26 VLM tests; retained log `artifacts/verification/gguf-mmproj/rust-tests-final.log`; SHA-256 `e8aa97169506331362d5a0446de9d5d8837f78e9963fca4636f87fd9a277b9ec`
- Strict scoped Clippy: core/transformer/VLM libraries and the `lfm2-vl` example passed with `-D warnings` plus the five recorded pre-existing Rust 1.97 allowances (`useless_borrows_in_formatting`, `manual_filter`, `manual_is_multiple_of`, `needless_range_loop`, `manual_contains`); library log SHA-256 `4c994e3ab472ac4b39abac4750304ca1c74da31581d65c3cdf8530d114f5adc6`; example log SHA-256 `96ff6e913ed4421d0e5111f4311c605409b5b978250ef0cb2eb21ac2dbc50c4d`
- Full locked/offline staged CPU baseline: passed `2026-08-10T11:51:12Z`–`2026-08-10T11:51:24Z` against pre-Phase-6-checkpoint HEAD `1535a0a5fef09f243811b83553b9c75baad78ee2` with exactly the 19 Phase 6 paths staged and no unstaged delta; retained log `artifacts/verification/gguf-mmproj/baseline-final.log`; SHA-256 `1f4c755e4271da48ea7906f4804181e75a5a0b4b61e5e0db37cdc1fd95bdedd3`
- Audit: the initial assigned-worker audit found no P0; processor markers/tokenizer ID, pre-allocation header bounds, transient memory accounting, official lock coverage, range negatives, required `general.type`, and CLI ID compatibility were addressed. The final bounded re-audit found no remaining P0/P1 defect; exact dtype-distribution assertions were added from its only actionable test-polish note.
- Phase 6 checkpoint/tag: complete at commit `dc6321cde3a9b8a019c1f0fd82780d90afa046df`, annotated tag `lfm2-vl-phase-6-gguf`
- Not claimed: production-checkpoint numerical parity, production MMProj payload execution, llama.cpp runtime numerical parity, executed CUDA parity, generated-caption parity, or native quantized vision execution

## Phase 7 Verification

- Phase 6 checkpoint/tag: complete at commit `dc6321cde3a9b8a019c1f0fd82780d90afa046df`, annotated tag `lfm2-vl-phase-6-gguf`
- Operator boundary: `LinearOp` keeps the existing dense `candle_nn::Linear` path and stores native Q8_0 weights directly as `QMatMul::QTensor`; the constructor deliberately bypasses the environment-sensitive eager-dequantization helper; a unit test pattern-matches the retained Q8_0 storage
- Tensor roles: native Q8_0 is accepted only for vision Q/K/V/out, vision MLP up/down, and projector linear 1/2; patch projection, positional embeddings, LayerNorms, and all biases remain dense; dense eligible matrices remain supported for mixed-width checkpoints
- Loader boundary: `from_gguf`/`load_gguf` remain the explicit Phase 6 dense compatibility APIs; `from_gguf_q8`/`load_gguf_q8` require at least one valid Q8_0 linear; `*_auto` selects native Q8 on F32 Q8 artifacts and propagates invalid Q8 role/alignment errors rather than silently dequantizing
- Dtype/device boundary: native Q8 activation dtype is currently F32; automatic F16/BF16 loading uses the dense compatibility path and the CLI prints the selected execution mode plus native-Q8 tensor count; native CUDA execution remains unverified and unclaimed
- Comprehensive two-layer fixture: all 14 eligible attention/MLP/projector matrices are Q8_0; GGUF SHA-256 `241f59dc92c033c9877654261cf538dc107087eab5834920bd4b0e52cbdcc056`; native versus dequantized-Q8 operator max abs `3.734588623e-3`; native versus dense max abs `5.300968885e-3`; cosine `0.999923348`
- Committed hybrid fixture: Q8_0 MMProj GGUF SHA-256 `225241e57bc84c62d097aab6daa9466a75e920dbb858daf4cba4cc18ef8bb3f0`; native-Q8 image-feature max abs `1.533385366e-4`; prefill max abs `1.650899649e-4`; cached decode `7.853843272e-5`, `6.113573909e-5`, and `4.052370787e-5`; cache reset `0`
- Negative coverage: dense-only strict-Q8 request, BF16 strict-Q8 activation, Q4_0, Q8_0 dense roles, Q8_0 patch role, non-block-aligned Q8_0 input width, malformed metadata/inventory/ranges/payload, and cross-artifact pairing mismatches return controlled errors
- Full Python reference suite: 23/23 passed at 2026-08-10 08:40 EDT; retained log `artifacts/verification/q8-mmproj/python-final.log`; SHA-256 `798ecb8a3cdc1cd63b635e249b230b9d536aa55e1b5fddcab632366e3d7cfd24`
- Full Rust gate: `cargo test --locked --offline -p candle-core -p candle-transformers -p candle-vlm` passed 2026-08-10 08:40 EDT; this includes 21/21 core library tests, all core integration/doc tests, 49/49 transformer library tests, 5/5 generation, 8/8 NMS, and 26/26 VLM tests; retained log `artifacts/verification/q8-mmproj/rust-tests-final.log`; SHA-256 `1638e2f86728770a4a46da4fc39d4b5cca9dcd2af165408189a18d7de1e2f68e`
- Strict scoped Clippy: core/transformer/VLM libraries and the `lfm2-vl` example passed with `-D warnings` plus the five recorded pre-existing Rust 1.97 allowances; library log SHA-256 `efc623c3b62b1114ffbfe3dd08ae83ef03a9d7baf17a24887580fc73451bd910`; example log SHA-256 `c24c4549a12dabcabd8f67e75b4349881b84bd1aa8778a9ed1fbdfb19f598238`
- Full locked/offline staged CPU baseline: passed `2026-08-10T12:49:32Z`–`2026-08-10T12:49:53Z` against pre-Phase-7-checkpoint HEAD `dc6321cde3a9b8a019c1f0fd82780d90afa046df` with exactly the 12 Phase 7 paths staged and no unstaged delta; retained log `artifacts/verification/q8-mmproj/baseline-final.log`; SHA-256 `ff46cc0b23a28050ffe856be2cb81ef7144667977587021f1d3cd221e00ed330`
- Verifier-only Cargo.lock SHA-256: `acd9419056b786da820b5120db8e78be06902721689c39b55f29445abdddaffc`
- Audit: the assigned worker verified that eligible weights remain `QMatMul::QTensor`, dense Phase 6 APIs remain intact, F32 auto-selection propagates validation failures, CLI diagnostics are explicit, the two-layer fixture covers 14 linears, and no P0/P1 defect remains in the initial CPU-F32 scope
- Phase 7 checkpoint/tag: complete at commit `1a9cf291ba9a1bbac1de83029f9d8f057aca00b5`, annotated tag `lfm2-vl-phase-7-q8`
- Not claimed: production-checkpoint numerical parity, production MMProj payload execution, llama.cpp runtime numerical parity, executed native-Q8 CUDA parity, generated-caption parity, or lower-bit native vision execution

## Vision Safety Limits Verification

- Candidate base HEAD: `f14a46a6967c38e84d99c08801234fd98aa2203a`; WSL2 `NVIDIA-Workbench`; CPU-only, locked, offline lane; no network, model payload, hosted runner, commit, tag, push, or PR
- Shared contract: default and hard ceilings are 67,108,864 pixels per source/derived image surface, 16 images, 11 crops per image, 64 total crops, 1,024 patches per crop, and 65,536 projected tokens; configuration can only tighten them
- Pre-allocation order: raw images are checked before RGB conversion/resizing/cropping/patchification; external prompt batches are revalidated before expansion; packed shapes, ranges, resized surfaces, spatial values, masks, projected counts, and vision batch size are checked before MMProj device transfer
- Focused transformer command: `cargo test --locked --offline -p candle-transformers lfm2_vl -- --nocapture`; 27/27 passed
- Focused VLM command: `cargo test --locked --offline -p candle-vlm lfm2_vl -- --nocapture`; 24/24 passed; all existing processor fixture tensors retain maximum absolute error `1.192092896e-7`
- Full Rust command: `cargo test --locked --offline -p candle-core -p candle-transformers -p candle-vlm`; passed 21/21 core library tests plus all core integration/doc tests, 53/53 transformer library tests, 5/5 generation tests, 8/8 NMS tests, and 29/29 VLM tests
- Checks: locked/offline `cargo check` passed for `candle-core`, `candle-nn`, `candle-transformers`, and `candle-vlm`; the `lfm2`, `quantized-lfm2`, and `lfm2-vl` examples all passed
- Strict scoped Clippy: core/transformer/VLM libraries and the `lfm2-vl` example passed with `-D warnings` plus the five recorded pre-existing Rust 1.97 allowances
- Full baseline: `bash scripts/lfm2-vl/verify-baseline.sh` passed from `2026-08-10T14:11:01Z` to `2026-08-10T14:12:00Z`, including formatting, affected crates, all three examples, and staged/unstaged diff checks; verifier-only Cargo.lock SHA-256 `acd9419056b786da820b5120db8e78be06902721689c39b55f29445abdddaffc`
- Independent re-audit: the assigned GPT-5.6 Luna worker found no remaining P0/P1 or concrete P2; it confirmed hard ceilings, resized-surface coverage, the non-exhaustive pre-release API boundary, full pre-transfer validation, and removal of the merge coverage allocation
- Not claimed: pre-decode image-file header enforcement, production-payload parity, llama.cpp runtime parity, executed CUDA parity, generated-caption parity, or lower-bit native vision execution

## Example Execution Policy Verification

- Candidate base HEAD: `f14a46a6967c38e84d99c08801234fd98aa2203a`; WSL2 `NVIDIA-Workbench`; CPU-only, locked, offline lane; no network, model payload, hosted runner, commit, tag, push, or PR
- Compatibility: both original positional split-MMProj loading and explicit path flags remain accepted; `--processor-config`, `--cpu`, and `--vision-cpu` retain their prior meanings
- Dtype policy: absent `--dtype` remains F32 on CPU and BF16 on CUDA; all canonical and long-form F32/BF16/F16 spellings are covered; diagnostics distinguish requested/defaulted from resolved dtype
- Execution policy: split input resolves dense; direct GGUF auto/dense/Q8 maps to `load_gguf_auto`/`load_gguf`/`load_gguf_q8`; strict Q8 rejects split input and non-F32 activations before any model/tokenizer file is opened
- Focused command: `cargo test --locked --offline -p candle-examples --example lfm2-vl`; 10/10 parser, placement-policy, routing-matrix, pre-I/O, help, and controlled-error tests passed
- Affected check: `cargo check --locked --offline -p candle-examples --example lfm2-vl`; passed
- Strict scoped Clippy: the `lfm2-vl` example passed with `-D warnings` plus the five recorded pre-existing Rust 1.97 allowances
- Full baseline: `bash scripts/lfm2-vl/verify-baseline.sh` passed from `2026-08-10T14:39:59Z` to `2026-08-10T14:40:33Z`, including formatting, all required library and example checks, and staged/unstaged diff checks; verifier-only Cargo.lock SHA-256 `acd9419056b786da820b5120db8e78be06902721689c39b55f29445abdddaffc`
- Independent final re-audit: the assigned GPT-5.6 Luna worker confirmed requested/defaulted and resolved dtype diagnostics, all six dtype spellings, the tested device policy consumed by `main`, loader routing, and pre-I/O Q8 rejection; no P0/P1/P2 defect remains
- Not claimed: successful production-file loading, production numerical parity, executed CUDA behavior, or llama.cpp runtime parity

## Native Unified Checkpoint Verification

- Candidate base HEAD: `f14a46a6967c38e84d99c08801234fd98aa2203a`; WSL2 `NVIDIA-Workbench`; local-only, locked, offline CPU lane; no network, model payload, hosted runner, commit, tag, push, or PR
- File contract: require exactly one `model.safetensors` or `model.safetensors.index.json`; canonicalize every file under the model directory; bound index/header/file/aggregate sizes, shard/tensor counts, shapes, offsets, overlaps, gaps, and payload coverage before memory mapping
- Model contract: derive the complete expected inventory from normalized configuration; accept the official `model.vision_tower.vision_model` root and committed-fixture direct root; support tied embeddings or explicit `lm_head`; reject missing, unexpected, or shape-incompatible tensors before payload construction
- Pairing contract: require local `config.json`, `processor_config.json`, and `tokenizer.json`; apply explicit processor override precedence; require processor patch/downsample compatibility and exact tokenizer/model image-token ID agreement
- Placement contract: explicit dtype applies to both native components; default dtype resolves independently per text and vision device, and distinct dtype builders are not silently shared
- Focused command: `cargo test --locked --offline -p candle-examples --example lfm2-vl`; 19/19 passed, covering actual tiny single/sharded safetensors, canonical/direct roots, tied/explicit heads, exact official 349/589 inventories, independent component dtypes, wrong index mappings, bad `total_size`, duplicate shard tensors, traversal, missing files, and pairing failures
- Official header contract: bounded pinned Range reads consumed 46,864 and 82,400 header bytes and zero payload bytes; canonical sorted name/BF16/shape SHA-256 values are `08f544b4495804ed842a37acf0936544ec88aa5d947bef8304a47816fee5b1a7` for 450M and `24728d0ed10229e788c5b9baf25e0cc6c92c93b9cdb12ebb252a3c140a861703` for 1.6B; the test also asserts raw FFN 6,656/12,288 normalization to 4,608/8,192
- Full Rust command: `cargo test --locked --offline -p candle-core -p candle-transformers -p candle-vlm`; passed all core library/integration/doc tests, 53/53 transformer library tests, 5/5 generation tests, 8/8 NMS tests, and 29/29 VLM tests
- Strict scoped Clippy: core/transformer/VLM libraries and the `lfm2-vl` example passed with `-D warnings` plus the five recorded pre-existing Rust 1.97 allowances
- Full baseline: `bash scripts/lfm2-vl/verify-baseline.sh` passed from `2026-08-10T15:42:32Z` to `2026-08-10T15:43:14Z`, including formatting, all required library/example checks, and both diff gates; verifier-only Cargo.lock SHA-256 `7292957b78b688fe2d8d0f61ba5987b92638d6138a0faa9a13db014d09b06a26`
- Launcher note: the first non-login WSL invocation at `2026-08-10T15:29:39Z` stopped before the formatting step because `cargo` was absent from that shell's `PATH`; the corrected login-shell command above is the verification result
- Integrity boundary: checkpoint files are an immutable local snapshot from header inspection through the returned model lifetime, as required by memory-mapped safetensors
- CUDA source boundary: the feature-gated native test covers distinct-device construction/loading only; the earlier hybrid test owns projected-feature transfer and forward coverage. Executed native CUDA inference is not claimed.
- Independent final re-audit: the assigned GPT-5.6 Luna worker confirmed the full official inventory digests, correct raw-to-effective FFN normalization, independent dtype policy, indexed-shard defenses, roots, tied/explicit heads, pairing, pre-payload rejection, CLI routing, mmap precondition, and honest CUDA scope; no P0/P1/P2 finding remains
- Reference-suite note: the ad hoc system-Python discovery command could not import `pytest`, and the previously documented repo `.venv` is not present in this checkout. No dependency was installed; the changed lock JSON was parsed directly and the exact digests are enforced by the green Rust test.
- Not claimed: production-payload construction or numerical parity, generated output, local llama.cpp runtime parity, executed native CUDA inference, or lower-bit native vision execution

## Deterministic Runner and Local llama.cpp Verification

- Candidate base HEAD: `f14a46a6967c38e84d99c08801234fd98aa2203a`; local-only CPU execution; no hosted runner, commit, tag, push, PR, production-weight download, or secret inspection
- Runner contract: `candle-lfm2-vl-inference-v1`; bounded prompt/image/generation inputs; deterministic greedy selection with lower-token-ID tie breaking; finite-logit enforcement; full F32-logit SHA-256 plus top-5 per step; exact expanded prompt, token IDs, image spans, crop metadata, packed tensor shapes, EOS provenance, decoded forms, and two-run cache-reset equality
- Artifact evidence: native, split, and direct loaders provide the exact config, tokenizer, processor, index/shard, manifest, and weight files they consumed. The runner canonicalizes, deduplicates, sizes, and SHA-256 hashes each regular file and rejects directory-only evidence. Input files must remain immutable from loader open through report emission.
- Focused command: `cargo test --locked --offline -p candle-examples --example lfm2-vl`; 26/26 passed. The real hybrid runner regression constructs deterministic text GGUF bytes from the committed tiny tensors, loads the committed split MMProj bundle, processes a generated 8x4 PNG, performs prefill plus three cached decode steps twice, resolves EOS from GGUF metadata, hashes all five consumed files, and serializes one-line JSON. A pure source-list regression asserts exact split, direct-GGUF, bundled-processor deduplication, and explicit-override inputs.
- Focused transformer command: `cargo test --locked --offline -p candle-transformers lfm2 -- --nocapture`; 32/32 focused tests passed with the established deterministic GGUF hashes and numerical tolerances unchanged. Optional `tokenizer.ggml.eos_token_id` parsing and validation are covered.
- Affected check: `cargo check --locked --offline -p candle-examples --example lfm2-vl`; passed in the WSL2 locked/offline lane.
- Full Rust command: `cargo test --locked --offline -p candle-core -p candle-transformers -p candle-vlm`; passed all library, integration, and doc-test lanes with 53/53 transformer library tests, 5/5 generation tests, 8/8 NMS tests, and 29/29 VLM tests.
- Strict scoped Clippy: core/transformer/VLM libraries and the `lfm2-vl` example passed with `-D warnings` plus the five recorded pre-existing Rust 1.97 allowances. The final example pass followed a small `GenerationInputs` grouping fix and the 26/26 test replay.
- Full baseline: `bash scripts/lfm2-vl/verify-baseline.sh` passed from `2026-08-10T19:42:13Z` to `2026-08-10T19:42:46Z`, including formatting, every required library/example check, and both diff gates; verifier-only Cargo.lock SHA-256 `7292957b78b688fe2d8d0f61ba5987b92638d6138a0faa9a13db014d09b06a26`.
- Provenance classification: relative to untouched Candle `31f35b147389700ed2a178ee66a91c3cc25cc80d`, the current 72-path delta is exactly nine fork-origin modifications plus 63 mod-owned additions. There are no unexpected baseline edits and no changed paths absent from `MOD_MANIFEST.md`.
- Independent audit boundary: the designated GPT-5.6 Luna task failed to start with `agent loop died unexpectedly`. No replacement user task was created. The manager's local audit added the missing direct-GGUF source-list regression and found no remaining P0/P1/P2 defect in the NR-4 scope.
- Resource incident: the completed local llama.cpp proof left PID 32000 resident with up to `131,549,319,168` private bytes. PowerShell, exact `taskkill`, Task Manager, and native termination attempts failed, timed out, or were denied. The PID later disappeared, but host performance and memory pressure did not recover sufficiently; an operator restart was required. WER recorded `RADAR_PRE_LEAK_64` and Windows recorded low-virtual-memory events. The legacy b9981 bundle is coherent and Defender/Code Integrity checks found no llama-related block. Exact root cause remains unproven; F-0008 records the evidence and closes the operational mystery with containment rather than a speculative attribution.
- Bounded owner proof: `pwsh -NoProfile -NonInteractive -File scripts/lfm2-vl/test-bounded-oracle.ps1` passes harmless normal-exit, timeout/descendant, owner-exit, concurrent-name-refusal, suspended-start, assign-before-resume, and exact-PID-absence cases. The wrapper defaults to a 24 GiB ceiling, rejects limits above 75% of physical RAM, defaults CUDA graphs off, and writes atomic JSON evidence.
- Installed oracle: `C:\llamacpp\llama-mtmd-cli.exe`, build b9981 / `(34558825a)`, 82,944 bytes, SHA-256 `01e191f9dd389b6e3b091eeaa8b6142784bd0e1b0e19ed7c67039afc6626ae1d`
- Manual current comparison: `C:\llamacpp\tools-b10344\llama-mtmd-cli.exe`, 84,480 bytes, SHA-256 `78ef208334fec62d62068cfd242a0b7358c602211125ceead3dd82b1347f717c`; inventoried read-only and not used as the pinned authority.
- Pinned owner oracle: ignored bundle `artifacts/llama-oracle/74ce1574-cuda-sm89/bundle`, exact source `74ce15741b420b8d6f12e720398458b576c51c2c`, CUDA 13.3/SM89/MSVC 19.33, executable SHA-256 `848e638069699149210b70945bdbb422494d7d03b8a18d7fb31a240d10e8abd0`. The parallel-1 build exited 0 with peak Job Object memory `7,889,661,952` bytes. Its bounded no-model probe reports `version: 10335 (74ce15741)`, peaks at `291,172,352` Job bytes, and verifies suspended assignment, exit 0, and PID absence. `bundle-manifest.json` records the complete EXE/DLL closure and hashes; no model was loaded.
- Text GGUF: local fine-tuned SFT derivative, 219,310,432 bytes, SHA-256 `84540fa23696ab9000f4a670b72e3405962264a920c3b7582d0e5a38b978abae`; exact bounded header ends at byte 2,387,296, contains 27 metadata records and 148 tensors, and reports LFM2 hidden width 1,024, 16 layers, vocabulary 65,536, context 128,000, image token 396, and EOS 7
- MMProj GGUF: 102,815,168 bytes, SHA-256 `ebfc428baa37efad8bae93864f914b2634a09009f91ad59f974fe1a1565d8561`. Its size and complete-file hash exactly match `LiquidAI/LFM2.5-VL-450M-GGUF@166cd80bbe157dc86d65f964eb8cc6a2cede62ca`, proving the local Q8_0 MMProj is byte-for-byte official.
- Tokenizer: pinned `LiquidAI/LFM2.5-VL-450M@fc6221ca597f3315e4f82fc2df606783267b34ba/tokenizer.json`, 4,733,040 bytes, SHA-256 `f3910942aa907c48b0cc20ec426ee38bfa8dcda8feecf035ced981918cb30f14`; image token 396. The unrelated local 2.6B tokenizer was rejected because its image token 124,907 is outside this model's 65,536-token vocabulary.
- Image: `candle-examples/examples/yolo-v8/assets/bike.jpg`, 182,991 bytes, SHA-256 `317e4a9d2d2be7859ba0ab8726a526f5ece9d77daf92857f3e93fb7b367824c1`, source dimensions 800x556
- Aligned structure: llama.cpp and Candle used the same text/MMProj/tokenizer/image, deterministic greedy settings, 4,096-token context, eight requested tokens, and equivalent chat framing. Both produced 608x416 vision input, 247 projected tokens, and 268 prompt tokens; Candle recorded the image span as `[5, 252)`.
- Exact output agreement: both runtimes decoded `A group of cyclists race on a road`. Candle generated IDs `[542, 2514, 803, 62480, 7736, 884, 768, 6671]` and tokens `["A", "Ġgroup", "Ġof", "Ġcyclists", "Ġrace", "Ġon", "Ġa", "Ġroad"]`; cache reset replay was exact. Candle prefill full-logit SHA-256 was `b2f21e55e855d162ecb3a4a91fdda102728c76c5e39c489377fb6ddae66d287b`.
- Claim boundary: this proves same-artifact preprocessing structure and greedy decoded-sequence/output behavior for the local fine-tuned text GGUF plus official MMProj. It does not prove official-base text parity, installed-build identity with the pinned llama.cpp commit, or component/logit equality because `llama-mtmd-cli` exposes no stable intermediate/logit dump.

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
- Phase 5 is checkpointed at `1535a0a5fef09f243811b83553b9c75baad78ee2` / `lfm2-vl-phase-5-hybrid`; split/native image features are exact and deterministic quantized-text hybrid prefill/decode remains within `4.457309842e-5`.
- The Phase 6 direct GGUF dense compatibility loader has exact dense/native image features, Q8_0-dequantized feature error `8.463021368e-5`, and direct-hybrid prefill/decode equal to the Phase 5 deterministic bounds.
- Official F16 and Q8_0 MMProj physical shapes, dtype placement, metadata, and names are locked from exact zero-payload header ranges; only the patch tensor requires a layout inverse.
- The Phase 7 native-Q8 path retains eligible GGUF weights as `QMatMul::QTensor`; the two-layer all-linear fixture reaches cosine `0.999923348`, and the committed hybrid fixture stays within `1.650899649e-4` prefill drift with exact cache reset.
- Shared request-wide vision limits now reject zero, one-over, overflow, above-hard-ceiling, malformed packed metadata, and oversized derived surfaces before expensive allocation or MMProj device transfer while preserving every existing tiny fixture result.
- The `lfm2-vl` example now exposes explicit dtype and MMProj execution intent, preserves original path and device flags, rejects invalid strict-Q8 policy before file I/O, and reports requested versus resolved policy.
- The example now loads an unmodified local unified Hugging Face directory from single or indexed safetensors, validates the entire normalized tensor contract before mapping, honors tied output weights, pairs processor/tokenizer/config inputs, and resolves text/vision dtype independently by device.
- The deterministic runner covers native and hybrid image prefill, cached decode, exact reset replay, finite full-logit hashes, stable top-k/token evidence, one-line JSON, and bounded external image/prompt/model-file diagnostics.
- Native, split-MMProj, and direct-GGUF inference evidence identifies and hashes every exact consumed file; directory paths are not accepted as artifact identity.
- The local fine-tuned text GGUF and byte-for-byte official Q8_0 MMProj produce the exact same eight-token caption under aligned deterministic Candle and llama.cpp execution.
- The Windows bounded-oracle smoke suite proves suspended Job Object assignment before resume, timeout and owner-exit tree cleanup, concurrency refusal, and exact PID absence without loading a model.
- The pinned llama.cpp b10335 CUDA/SM89 bundle is source-, build-, and file-identified and passes a 512 MiB/30-second bounded no-model identity probe with no residual process.
- Tiny-fixture dense parity is within `2.38418579e-7` for hidden states and `2.98023224e-8` for logits; production-checkpoint and GGUF numerical parity remain unclaimed.

## Known Conflicts

- Official config context is 128,000 while model cards advertise 32,768; construction follows config and production policy remains unresolved.
- Numeric IDs for image wrapper, row/column, and thumbnail marker strings must be exported by the tokenizer harness; only image placeholder ID 396 is config-explicit.
- llama.cpp PR #25524 for reading LFM2 tiling parameters from GGUF metadata is open and unmerged; official processor config remains authoritative.
- The official MMProj headers omit all three tiling metadata keys; direct loading therefore depends on pinned architecture defaults or an explicit processor document.
- The local WSL verifier exposes an RTX 4090 through the driver but has no Linux CUDA toolkit or `nvcc`; the committed distinct-device test remains an owner-scoped execution gap.
- The legacy `C:\llamacpp` runtime is build `b9981` / `(34558825a)` and the manual `tools-b10344` bundle is a newer current-master comparison. Neither substitutes for the exact pinned b10335 owner build.
- The only local text GGUF is a fine-tuned game-QA SFT derivative, not the official base checkpoint. Its pairing with the pinned official tokenizer and byte-identical official MMProj is proven for same-artifact execution, but it cannot serve as official-base text parity evidence.
- `llama-mtmd-cli` exposes deterministic sampling controls but no stable logits or intermediate-tensor dump contract, so local llama.cpp can currently prove same-artifact prompt/token/output behavior, not component-tensor equality by itself.
- A completed `llama-mtmd-cli` run remained attached under Codex with approximately 131.5 GB private memory, while normal exact-PID termination was denied or timed out. PID disappearance did not restore usable host performance; restart was required. WER leak evidence, virtual-memory pressure, related upstream CUDA/MTMD reports, and a possible Codex token/job boundary are recorded, but none is proven as the unique cause; see F-0008.

## Blockers

- None for starting official 450M native CPU-F32 work or the pinned llama.cpp lane: containment and the no-model identity probe are green. Git publication still needs an accessible `origin`; this worktree currently has none, and local `gh` is unauthenticated. Official-base production payload parity, the 1.6B checkpoint, executed native-Q8 CUDA, and lower-bit evidence remain incomplete and unclaimed.

## Active Files

- `candle-transformers/src/models/lfm2.rs`
- `candle-transformers/src/models/quantized_lfm2.rs`
- `candle-transformers/src/models/siglip2.rs`
- `candle-transformers/src/models/lfm2_vl/`
- `candle-transformers/src/models/lfm2_vl/gguf.rs`
- `candle-transformers/src/models/lfm2_vl/linear.rs`
- `candle-core/src/quantized/gguf_file.rs`
- `candle-vlm/`
- `candle-vlm/src/lfm2_vl/config.rs`
- `candle-vlm/src/lfm2_vl/processor.rs`
- `candle-vlm/src/lfm2_vl/prompt.rs`
- `candle-examples/examples/lfm2-vl/`
- `candle-examples/examples/lfm2-vl/runner.rs`
- `tools/lfm2_vl/reference/inspect_gguf_header.py`
- `tools/lfm2_vl/reference/test_gguf_header.py`
- `tools/export_lfm2_vl_mmproj.py`
- `scripts/lfm2-vl/run-bounded-oracle.ps1`
- `scripts/lfm2-vl/test-bounded-oracle.ps1`
- `tests/fixtures/lfm2_vl_mmproj_tiny/`
- `candle-transformers/src/models/mod.rs`
- `candle-examples/examples/lfm2/main.rs`
- `tests/fixtures/lfm2_vl_processor_tiny/`
- `tools/lfm2_vl/reference/export_processor_fixture.py`
- `docs/lfm2-vl/DECISIONS.md`
- `docs/lfm2-vl/PARITY.md`
- `docs/lfm2-vl/MOD_MANIFEST.md`
- `docs/lfm2-vl/FAILURE_LOG.md`
- `docs/lfm2-vl/STATUS.md`

## Next Task

NR-5B — obtain the pinned official `LiquidAI/LFM2.5-VL-450M@fc6221ca597f3315e4f82fc2df606783267b34ba` native files only through the guarded production path. Record repository, revision, filename, size, and SHA-256; preflight host commit/physical/GPU memory; run CPU F32 first; and compare selected processor, vision, projector, merge, prefill, and cached-decode tensors against the pinned Transformers oracle. Use the bounded llama.cpp owner only for a later same-artifact comparison. Done when every selected production tensor is within the specified tolerance, the deterministic trace replays exactly, all consumed files are identified, the process tree is absent, and post-run host memory is healthy. Do not start 1.6B or CUDA inference before this gate is green.

---
AI-edited: 2026-08-10T15:41:00-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=lfm2-vl-nr5a | change=closed the memory incident operationally, proved the bounded pinned oracle, and advanced the exact next task to official 450M CPU-F32 parity
