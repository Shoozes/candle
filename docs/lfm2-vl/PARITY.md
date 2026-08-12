# LFM2.5-VL Parity

## Current State

The deterministic fixture phases, official 450M native Windows CPU-F32 component gate, and official 1.6B native Windows CPU-F32 component gate are green. Candle and pinned llama.cpp also executed the same official Q4_0 text GGUF plus Q8_0 MMProj and produced the exact decoded text `The image features` under bounded CPU runs. The 1.6B comparison uses one pinned regular-file snapshot, one deterministic image, the official rendered prompt, 51 processor/vision/projector/merge/language tensors, three cached decode steps, exact reset replay, and before/after resource evidence. Native CUDA, lower-bit production MMProj execution, llama.cpp component/logit equality, and the optional WSL portability replay remain unclaimed.

## Required Gates

| Gate | Required evidence | Phase status |
| --- | --- | --- |
| Native Windows workspace baseline | Locked CPU-only Candle checks under MSVC plus diff/manifest gates | Green for the current source tree: bounded offline core/transformer/VLM and example tests/checks passed; read-only WSL Git replay now passes `git diff --check` and the 94-path mod-manifest gate. |
| WSL portability baseline | Locked CPU-only Candle checks and diff/manifest gates from Linux home | Secondary replay green; Phase 7 staged log SHA-256 `ff46cc0b23a28050ffe856be2cb81ef7144667977587021f1d3cd221e00ed330` and current consolidated baseline green |
| Reference fixture | Deterministic pinned-Python export with component and multimodal tensors | Green; 87 tensors, byte-identical independent exports; manifest SHA-256 `c5461dadb0edfd920b20f308650c59676977110a1cc2f199e317dea7d75bdd7b` |
| LFM2 text configuration | 450M effective FFN width `4608`; 1.6B width `8192` | Green in config tests and header evidence |
| Dense text forwarding | Token-ID and embedding-driven prefill plus incremental decode agree | Green on the committed fixture; maximum hidden-state error `2.38418579e-7`, maximum logit error `2.98023224e-8` |
| Quantized text forwarding | Token-ID and embedding-driven paths agree and cache can be reset | API/equivalence gate green; official-base GGUF decoded output is green at stable cross-runtime fields, while llama.cpp component/logit equality remains unavailable and unclaimed |
| SigLIP2 | Component tensor comparisons against pinned fixtures | Phase 2 checkpoint complete; Phase 3 repeated-crop regression 8/8 green |
| Projector | Exact pixel-unshuffle and stage-level comparisons | Phase 3 focused gate green; 11/11 total |
| Processor | Exact resize, crop, patch, mask, shape, and token metadata | Green on all 12 pinned cases; 24/24 crate tests; worst pixel max abs `1.192092896e-7`; integer/crop metadata exact |
| Prompt expander | Exact expanded strings, tokenizer IDs, marker placement, and one span per crop | Green on all 5 pinned prompt cases, including multiple images and tiled thumbnail markers |
| Composite model | Image-span replacement and prefill/decode parity | Phase 3 focused gate green; 11/11 total |
| Phase 3 library gate | Locked/offline `candle-transformers` library tests | 37/37 passed |
| Phase 4 fixture reproduction | Fresh pinned-oracle export matches checked-in bytes | Green; manifest, metadata, and safetensors hashes match exactly |
| Split MMProj artifact | Exact versioned inventory, hashes, immutable provenance, and processor pairing | Green on the deterministic 43-tensor fixture; exporter/reference suite 19/19 |
| Hybrid GGUF text + dense MMProj | Real GGUF parse/load, split/native image-feature equivalence, prefill/decode/cache comparison | Green on the committed deterministic fixture; image features exact, hybrid text-logit max abs `4.457309842e-5` |
| Direct GGUF MMProj dense compatibility | Strict metadata/inventory/range load, patch inverse, dequantization, image-feature and hybrid execution comparison | Green on deterministic GGUF fixtures; dense image features exact, Q8_0 dequantized max abs `8.463021368e-5`, direct hybrid errors equal Phase 5 |
| Native Q8_0 GGUF MMProj | Eligible weights remain QTensor, all vision/projector linear roles execute through QMatMul, dense fallback remains intact, and hybrid prefill/decode/cache stay within documented drift | Green on CPU F32 deterministic fixtures; 14/14 two-layer linear roles quantized, feature cosine `0.999923348`, prefill max abs `1.650899649e-4`, cache reset exact |
| Unified native checkpoint loading | Unmodified single/indexed Hugging Face directory, exact official inventories, tied output, config/processor/tokenizer pairing, and pre-map diagnostics | Green on real tiny safetensors; 19/19 focused tests; pinned 450M/1.6B name/BF16/shape digests and counts 349/589 exact |
| Deterministic inference evidence | Bounded prompt/image/generation inputs, exact consumed-file hashes, native and hybrid prefill/decode, full-logit hashes, greedy top-k/token trace, and exact reset replay | Green; 29/29 focused example tests, including real split-MMProj hybrid execution, exact direct/split/override source lists, one-line JSON, and native trace no-clobber publication |
| Local same-artifact oracle | Identical text GGUF, MMProj, tokenizer, processor policy, image, prompt framing, context, and deterministic decode settings in Candle and llama.cpp | Green for the local fine-tuned text GGUF plus byte-identical official Q8_0 MMProj: 608x416, 247 image tokens, 268 prompt tokens, and exact eight-token output agreement |
| Official MMProj header contract | Pinned F16/Q8_0 metadata, names, physical shapes, dtype placement, and zero-payload evidence | Green; 32 metadata records, 201 tensors, tensor-data offset 12,736, no retained payload bytes |
| Official text GGUF identity | Immutable official source, full-file size/SHA-256, payload-free bounded header, text/tokenizer metadata, quantization placement, and separation from the local derivative | Green; 219,311,264 bytes, SHA-256 `6d2757dd0f0b98aea7dc90477bb5b3a0df1089be85ef92943f8cecb05121ccbf`, 39 metadata records, 148 tensors, exact physical/declared extent |
| Distinct devices | Vision and text may differ; only projected image features cross at merge | Public `--text-cpu` route and official 450M CPU-text/CUDA-vision F32 parity are green; optional WSL replay remains secondary |
| Official 450M production checkpoint | Pinned Transformers versus native Candle processor, vision, projector, merge, prefill, cached-decode, artifact, replay, and cleanup evidence | Green on native Windows CPU F32; 36/36 tensors, zero failures, exact input tensors and reset, comparison SHA-256 `caaae9ad159ec8370007169bd7c486ccff96f8b547ea6a113685f0c8703bbbac` |
| Remaining production checkpoints and GGUF | Same-artifact official-base GGUF, 1.6B, CUDA, and lower-bit numerical validation | Official-base GGUF identity and both bounded runtime replays are green at every shared stable field; 1.6B artifact/config/load, Python/native traces, and phase-specific comparison are green; 450M CUDA F32/BF16 parity is green; lower-bit production CUDA remains |

## Native Windows 450M Load-Only Evidence

The pinned local Hugging Face snapshot `LiquidAI/LFM2.5-VL-450M@fc6221ca597f3315e4f82fc2df606783267b34ba` was inspected without downloading. Its `model.safetensors` blob is 897,484,568 bytes; the snapshot also contains the exact processor, tokenizer, generation, template, license, and model configuration files listed below. The snapshot uses external Windows symlinks, which the native loader correctly rejects because its immutable inventory contract requires regular files inside the supplied model directory. A disposable regular-file copy outside the repository was therefore used for the proof and removed afterward.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `.gitattributes` | 1,519 | `11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361` |
| `chat_template.jinja` | 3,836 | `309e586e2ed3d7f2db1e2a045bfb07f4c83798b23f7ac587954426302d508e9` |
| `config.json` | 2,373 | `ab0de32d57b83b8b0cbb4526e096cf1e8adc1d8b6a09cb55df38597866eae53f` |
| `generation_config.json` | 136 | `40d17f9ec64c97e8fd5400540960b6a9761ed6d6acc0b1ab6a6656055e4755b3` |
| `LICENSE` | 10,574 | `4d28ca14dedc0b3d0fcc2b3339f0e79931faa33874f3d24f522183a8fc70068c` |
| `model.safetensors` | 897,484,568 | `2f6deb5dd43707de5cfe3c59470d3bccf4c3112a810a74570499f4728d412eea` |
| `processor_config.json` | 828 | `622b75b531b3f49b1cdf4f90626c34e5ffb4f8bba2b8637807af0462398ae718` |
| `README.md` | 13,444 | `666cc6b49fcdec9ddd378081c47df5aa11679c12a4a05cbbb436a3107f04ee3b` |
| `tokenizer_config.json` | 829 | `aed83606e95db808fc4d5312bf117605360e770bfe5a6028c348c3981ce143a` |
| `tokenizer.json` | 4,733,040 | `f3910942aa907c48b0cc20ec426ee38bfa8dcda8feecf035ced981918cb30f14` |

Native Windows load-only command: `CARGO_NET_OFFLINE=true CARGO_BUILD_JOBS=2 cargo run --locked --offline -p candle-examples --example lfm2-vl -- --model-dir <regular-file-copy> --cpu`. It passed from `2026-08-10T21:06:37-04:00` through `2026-08-10T21:06:42-04:00` with text `16x1024`, 12 vision layers, patch 16, factor 2, image token 396, processor max patches 1024, tied output embeddings, 349 tensors, one shard, and F32 CPU placement for both towers. No prompt or image was supplied, so no generated caption or production tensor comparison is claimed; the disposable copy was removed and no `llama*` process remained.

## Native Windows 450M Component Evidence

NR-5B used the external regular-file snapshot identified by artifact-manifest SHA-256 `659c8421530586b6cc28c094cfcdc69719ea8626f2abc0efd9eec4ac2a68a984` and the 256x256 input image SHA-256 `f902f8d2e47e53eafac86831cfc692001dc15870eb81d57abc3128f048d2efca`. The Python oracle manifest is `41f97daf914bd2c3eea81065ca87f1b002e869dd0dcedf010bba229646529d06`; the final native manifest is `286bc3c453188de38ac12a9553e60515a17aad61a57d03086c350b0f2d013345`.

The comparison report SHA-256 `caaae9ad159ec8370007169bd7c486ccff96f8b547ea6a113685f0c8703bbbac` records `passed=true`, 36 compared tensors, and zero failures. Input IDs, attention masks, image bytes, pixel masks, spatial shapes, projector patch ranges, and decode IDs are exact. Every vision encoder layer plus projector, merged-embedding, prefill, hidden-state, and cached-decode tensor passes its stage-specific CPU-F32 allclose contract; cache reset is exact. The largest maximum absolute delta is `0.0189208984375` at vision encoder layer 11. Native peak Job memory was 2,120,413,184 bytes under an 8 GiB ceiling, its PID was absent after exit, and the post-run census recovered to 46,049,075,200 available physical bytes and 23,420 MiB GPU memory free with no model or build process present.

## Phase 2 Focused Evidence

The manager's Linux-home WSL2 CPU F32 verifier passed all 7 SigLIP2 tests. The exact maximum absolute errors and cosine similarities were: patch projection `5.960464478e-8` / `0.999999940`; resized positions `2.980232239e-8` / `0.999999940`; embedding sum `1.192092896e-7` / `1.000000119`; encoder layer 0 `4.768371582e-7` / `0.999999881`; encoder layer 1 `1.192092896e-6` / `0.999999881`; returned post-LN `7.152557373e-7` / `1.000000119`; and the post-LN hook matched the returned result. Padding-key isolation was exact: max absolute error `0`, cosine `1`.

This focused proof covers packed patch projection, CPU F32 separable antialiased positional interpolation with per-shape caching, bidirectional key masking, encoder stages, post-LN, malformed-input rejection, and the required no-pooling-head boundary. The Phase 2 checkpoint is complete at commit `74e109aec5f9801cfead3eeb27fe3f93ac646b84`, annotated tag `lfm2-vl-phase-2-siglip2`. The Phase 2-era broader and baseline logs remain historical; the final Phase 3 gate and pre-checkpoint baseline are recorded below.

## Phase 3 Focused Evidence

Phase 3 implements dynamic top-level configuration, factor-N official pixel-unshuffle, optional projector LayerNorm, linear/GELU/linear projection, crop unpadding/ranges/order, strict one-span-per-image exact-length merge, multimodal prefill, ordinary cached decode, cache reset, and `EncodedImages`.

The focused Phase 3 gate passed 11/11. Retained log: `artifacts/verification/native-composite/focused-tests.log`; SHA-256 `7d727e1b8558f1f242ce940c8af36d44a3e292f4ffa023d1ff124ccf2cc13638`. Maximum absolute errors were projector stages `<=5.960464478e-8`, encoded/merged embeddings `<=6.519258022e-9`, prefill `<=4.470348358e-8`, and cached decode `<=2.980232239e-8`.

The SigLIP2 repeated-crop regression passed 8/8. Retained log: `artifacts/verification/native-composite/siglip2-regression.log`; SHA-256 `5684568b060c6338f3e5d8bc94361d37bc64ddf84584ad4a5e05915acc275f38`. It protects a real runtime defect found during multi-crop execution: batched attention received a non-contiguous transposed left-hand operand and failed with `MatMulUnexpectedStriding`; `split_heads` now materializes a contiguous tensor.

The locked/offline `candle-transformers` library gate passed 37/37. Retained log: `artifacts/verification/native-composite/candle-transformers-lib.log`; SHA-256 `0f36d6a8d54f77abfe9c5031075b7174cff83859315d0997f60a1a399f475497`.

The full locked/offline CPU baseline passed `2026-08-10T05:48:07Z`–`2026-08-10T05:48:10Z` against pre-Phase-3-checkpoint HEAD `74e109aec5f9801cfead3eeb27fe3f93ac646b84`. Retained log: `artifacts/verification/native-composite/baseline-final.log`; SHA-256 `47d984dd3afe7b92b6a72bcdb93e7d9da99bd8673e5c1067b8f1fac7ed2b8b45`. Cargo.lock SHA-256 remains `4e059ffe6035520ca6553303932173eba562f4985f82931a90129eea9849ce54`.

The Phase 3 checkpoint is complete at `37264b49cf74d0cf7697317eda0183f084db6ff8`, tagged `lfm2-vl-phase-3-native-composite`. These results do not claim production-checkpoint parity, CUDA, GGUF, raw-image preprocessing, tokenizer/chat-template behavior, or CLI support.

## Phase 4 Processor and Prompt Evidence

The new `candle-vlm` crate implements the processor configuration precedence contract, RGB conversion, checked smart resize, TorchVision-compatible antialiased byte resize, tile-grid selection, row-major crops, optional thumbnails, fused rescale/normalization, patchification, fixed padding, masks/shapes, image/crop metadata, tokenizer-resolved special markers, sentinel-position preservation, and exact per-crop span recording.

The locked/offline `candle-vlm` suite passed 24/24. All 12 required image cases compare packed pixel tensors, masks, spatial shapes, image grids/sizes, crop ranges/order/kinds, and projected token counts. Maximum normalized pixel error was `1.192092896e-7` with cosine similarity `1.0`; all integer and structural metadata was exact. A direct 7×5 to 8×4 regression compares all 96 RGB channel bytes against pinned TorchVision output.

All 5 prompt oracles match exact expanded strings, tokenizer IDs, placeholder counts, and one span per crop, including tiled row/column markers, thumbnail markers, multiple images, image-first positioning, and images across turns. Controlled-error tests cover missing tokens, sentinel/image mismatch, projected-token mismatch, context overflow, malformed row-major crop metadata, inconsistent empty batches, packed allocation overflow, and encoded image ranges that split crop ranges.

All 10 real-dimension oracles assert smart dimensions, large-image classification, selected grid, tile canvas, and whole/tile/thumbnail order. A fresh pinned Python export reproduced the checked-in fixture byte-for-byte: manifest `2fb787e378f5fd1ddfa147913aadccd07add9a1045b8bb0f693ca2c2f564959c`, metadata `aca7f4d5e5e4ef0e4872adeb227b56cf3960d87b353c40162af97660783f2327`, and tensors `a25932fc57f3e78f48a1a8f558216521c7ae3e8659fcf0a389cd0a4ebe0ab3f6`.

The pinned Python reference tests passed 9/9, the full `candle-transformers` library regression remained green at 37/37, and the final staged baseline passed formatting, all required package/example checks, and both diff gates. Phase 4 is checkpointed at `8d1bbe471404848730685c98e7dd56b13a457eb4`, tagged `lfm2-vl-phase-4-native-e2e`. Production-checkpoint, GGUF/mmproj, CUDA, generated-caption, and CLI parity remained unclaimed at that gate.

## Phase 5 Hybrid MMProj Evidence

The split exporter emits only the config-derived canonical SigLIP2/projector inventory into `mmproj.safetensors`, plus a versioned manifest and canonical processor JSON. It rejects missing, unexpected, shape-incompatible, non-dense, duplicate-normalized, or incomplete tensors before writing, requires an immutable source revision, and produces byte-identical output from the committed tiny unified fixture.

The Rust loader validates the manifest and processor pair, exact tensor inventory, shape/dtype/byte counts, bounded safetensors header and tensor count, offsets, overlaps, gaps, payload coverage, and hashes. A single fallibly allocated buffer—bounded from the validated manifest payload and maximum header—is used for hash, inspection, and construction, removing path-replacement ambiguity. GGUF metadata also rejects malformed present RoPE values and bounds rotary-table allocation before construction.

The deterministic hybrid proof writes real GGUF bytes from committed text tensors, pins SHA-256 `8fbd510aeea4715547c57975a7adcb91c148a8bc5e8d869d9617b69af6a006b1`, parses them with `gguf_file::Content::read`, and loads them through `ModelWeights::from_gguf`. Q8_0 is used where tiny matrix widths meet its block constraint; small unalignable tensors remain F32. Split and unified image features are exact. Relative to the native dense model, maximum absolute errors are prefill `4.457309842e-5`, cached decode `2.650916576e-5`, `2.175569534e-5`, and `1.309439540e-5`, with exact cache reset.

Final retained evidence: Python 19/19, `candle-transformers` 42/42 plus its integration tests, `candle-vlm` 25/25, the `lfm2-vl` example check, scoped Clippy gates, and the staged locked/offline baseline all pass. The CUDA-vision/CPU-text test is source-complete and asserts device residency and `1e-4` prefill agreement, but local execution is truthfully skipped because WSL exposes the RTX 4090 driver without a Linux CUDA toolkit or `nvcc`. The assigned worker confirmed all nine audit findings resolved and no remaining code blocker. Production models and GGUF files were not downloaded.

## Phase 6 Direct GGUF MMProj Evidence

The direct loader opens one stable GGUF handle, applies phase-specific parser limits before allocation, validates exact metadata and tensor inventory, checks dtypes, element counts, alignment, offsets, overlaps, truncation, retained dense bytes, and conservative peak bytes, then dequantizes into the already proven native SigLIP2/projector path. It requires `general.type=mmproj`; optional projector LayerNorm and bias tensors must be complete pairs. The only layout transform is the header-proven inverse for `v.patch_embd.weight`.

Official header-only evidence at `LiquidAI/LFM2.5-VL-450M-GGUF@166cd80bbe157dc86d65f964eb8cc6a2cede62ca` fixes the 201-tensor name set, physical shapes, F16/F32 and Q8_0/F32 placement, and absent preprocessing keys. Both exact 12,736-byte prefixes end at the tensor-data boundary and contain zero payload bytes. The direct path therefore retains official processor defaults and resolves the image placeholder ID from the tokenizer rather than inventing GGUF metadata.

The deterministic dense GGUF has SHA-256 `7361b57e6d9dbf2d7809d4f446944fdc7325b368e4444fee2bc3497376695256` and matches native image features exactly. The Q8_0 compatibility fixture dequantizes with maximum image-feature error `8.463021368e-5`. Paired with the deterministic quantized text GGUF, direct-MMProj prefill max abs is `4.457309842e-5`; cached decode is `2.650916576e-5`, `2.175569534e-5`, and `1.309439540e-5`; cache reset is exact. These are deterministic fixture results, not production-payload or llama.cpp runtime parity.

Final local evidence is green: pinned Python 23/23; the complete offline core/transformer/VLM test command, including all integrations and doc tests; strict scoped Clippy with five documented pre-existing Rust 1.97 allowances; and the exact staged locked/offline baseline. Retained historical hashes are recorded in `HISTORY.md`; `STATUS.md` carries only the latest gate. The assigned worker's final static re-audit found no remaining P0/P1 defect. No production model or MMProj payload was downloaded.

## Phase 7 Native Q8 MMProj Evidence

`LinearOp` now covers every vision attention projection, both vision MLP linears, and both projector linears. Dense construction still stores `candle_nn::Linear`; native Q8 construction stores `QMatMul::QTensor` directly and adds the dense bias afterward. Patch projection, positions, LayerNorm parameters, and biases remain dense. Mixed checkpoints may retain dense eligible matrices, while explicit native mode rejects lower-bit weights, Q8 dense roles, and non-block-aligned Q8 input widths.

The two-layer block-aligned fixture quantizes all 14 eligible linears and has GGUF SHA-256 `241f59dc92c033c9877654261cf538dc107087eab5834920bd4b0e52cbdcc056`. Native versus dequantized-Q8 operator max abs is `3.734588623e-3`; native versus the dense source is `5.300968885e-3` with cosine `0.999923348`. This is a documented quantized drift gate, distinct from the earlier dense CPU-F32 target of cosine `>=0.99999`.

The committed hybrid fixture's native-Q8 GGUF SHA-256 is `225241e57bc84c62d097aab6daa9466a75e920dbb858daf4cba4cc18ef8bb3f0`. Its image-feature max abs is `1.533385366e-4`; multimodal prefill is `1.650899649e-4`; three cached decode comparisons are `7.853843272e-5`, `6.113573909e-5`, and `4.052370787e-5`; cache reset is exact. The full local gate passes 23 Python tests, all core/transformer/VLM tests, and strict scoped Clippy. The assigned worker's final audit reports no P0/P1 defect in the initial CPU-F32 scope.

The example automatically selects native Q8 for valid F32 Q8 artifacts and reports the selected execution mode/count. F16/BF16 automatic loading deliberately stays on the Phase 6 dense path. No production payload or llama.cpp runtime was used, so official-file numerical parity, top-k/token agreement, and native-Q8 CUDA remain evidence gaps rather than claims.

## Native Unified Loader Evidence

The local native loader accepts the official unified namespace without file renaming and supports one safetensors file or an indexed shard set. It bounds and validates the complete header/index inventory before memory mapping, resolves tied versus explicit output weights, pairs processor and tokenizer semantics with model configuration, and reports roots, shards, bytes, dtypes, devices, and all inventory defects. Its 19 focused tests construct real tiny checkpoint files and compare the generated official 450M/1.6B inventories against canonical sorted name/BF16/shape digests from zero-payload pinned header reads. This is loader proof, not production numerical parity. The feature-gated native CUDA test is construction-only; native CUDA inference remains unclaimed.

## Local llama.cpp Oracle Boundary

The official P2 text artifact is locked separately from the earlier local comparison: `LiquidAI/LFM2.5-VL-450M-GGUF@166cd80bbe157dc86d65f964eb8cc6a2cede62ca/LFM2.5-VL-450M-Q4_0.gguf`, 219,311,264 bytes, SHA-256 `6d2757dd0f0b98aea7dc90477bb5b3a0df1089be85ef92943f8cecb05121ccbf`. Its exact payload-free 2,388,128-byte header has SHA-256 `bdb33b992b136a77b4d807b84319a7daa43ebac15144e6336c0d9b9ef1e8ed2e`, 39 metadata records, and 148 tensors. The physical extent matches the header declaration. Its paired Q8_0 MMProj is the byte-identical official file already present under `C:\llamacpp`, SHA-256 `ebfc428baa37efad8bae93864f914b2634a09009f91ad59f974fe1a1565d8561`.

P2 runtime execution is green. Candle and pinned llama.cpp build 10335 consumed those exact files plus the same deterministic 256x256 image and equivalent official chat framing, then decoded exactly `The image features` for three greedy steps. Candle retained generated IDs `[1098, 4646, 5251]`, 64 projected image tokens, prefill-logit SHA-256 `aa2e0aa2132cb67fc33cb57523e73dee1c0cabac9362d7adbb22ea1a871d5280`, and exact cache reset. llama.cpp does not expose stable forms of those fields, so the cross-runtime claim is exact artifact/prompt/decoded-output/cleanup agreement rather than token-ID, preprocessing, or component-tensor parity. The machine report is 7,026 bytes with SHA-256 `2c54cd790aef5ddcf8b053923a7ebb18ef055e9b06b6b580abd2a1eb9b92f6fd` and records `passed=true` with a bounded 128,000-versus-4,096 context-ceiling difference; the actual sequence uses only 83 positions.

P3.1 artifact admission is green. The pinned official 1.6B snapshot contains eight direct regular files totaling 3,198,084,631 bytes; an independent local pass rehashed the 3,193,334,216-byte `model.safetensors` as `7fc7458e4382fc6e558cfdda45857fbf9ab5b40a8bf199c9cd073003b14ac26d` and matched every acquisition/artifact record. The manifest SHA-256 is `a080891c8d1099d58a01377af258ef04898f808eed0fcf4fbe718d4698f4b732`; publication was atomic, the bounded process tree exited, and no model was loaded at acquisition time. The projected 51-tensor trace remains about 182.53 MB. Stage-specific 16/24/12 GiB Job ceilings derive from measured 450M peaks, the exact 3.558093732 model-byte ratio, and a 1.35 safety factor. Subsequent load, trace, and parity evidence is recorded below.

P3.2 non-load admission is green. The fresh hash-only artifact audit at `C:\DevStuff\candle-oracle\evidence\p3-1.6b-artifact-audit-20260811T162000Z.json` reports the pinned 1.6B revision, eight files, and 3,198,084,631 bytes. The stdlib-only config audit at `C:\DevStuff\candle-oracle\evidence\p3-1.6b-config-audit-20260811T162000Z.json` has SHA-256 `39fac83ce04986a2c14ea2e3b423eb81d34197db817a59e9b02b9d7ccfeee596` and proves the 589-tensor inventory, 2,048/1,152 text/vision widths, effective FFN 8,192, marker ID contract, patch size 16, factor 2, and tied output. The exact 42-distribution Windows reference environment verifier passed with no lock or test mismatches. These were admission checks only; the load-only, trace, and numerical claims are closed by the subsequent sections below.

P3.2 Python dry-load is green. The pinned Windows environment loaded the exact external regular-file snapshot under a 16 GiB Job Object without inference or tensor serialization. The external production metadata reports `weights_loaded=true`, `tensor_payload_generated=false`, the pinned model/revision, and the external snapshot mode. Owner evidence records exit 0, 19,289 ms, peak Job memory 7,475,851,264 bytes, and exact PID absence. This proves Python loadability for that stage; native load, component traces, and numerical parity are recorded in the completed sections below.

The read-only local runtime at `C:\llamacpp` reports build b9981 / `(34558825a)` and executable SHA-256 `01e191f9dd389b6e3b091eeaa8b6142784bd0e1b0e19ed7c67039afc6626ae1d`. That build is not proven identical to the pinned implementation-reference commit `74ce15741b420b8d6f12e720398458b576c51c2c`.

The local text GGUF is a fine-tuned game-QA SFT derivative with SHA-256 `84540fa23696ab9000f4a670b72e3405962264a920c3b7582d0e5a38b978abae`; it is not the official base checkpoint. The local Q8_0 MMProj SHA-256 `ebfc428baa37efad8bae93864f914b2634a09009f91ad59f974fe1a1565d8561` and size exactly match the pinned official LiquidAI file. The tokenizer is pinned to the official 450M revision and hashes to `f3910942aa907c48b0cc20ec426ee38bfa8dcda8feecf035ced981918cb30f14`.

With identical artifacts, image, prompt framing, 4,096-token context, and greedy settings, both runtimes produced 608x416 preprocessing, 247 projected image tokens, 268 prompt tokens, and `A group of cyclists race on a road`. Candle IDs were `[542, 2514, 803, 62480, 7736, 884, 768, 6671]` with exact reset replay. This is same-artifact runtime behavior parity, not official-base or component-tensor parity. `llama-mtmd-cli` still exposes no stable logits or intermediate-tensor dump; unavailable stages remain explicitly unavailable.

After that comparison, the Windows `llama-mtmd-cli` process remained resident with approximately 131.5 GB private memory and normal user/task-manager termination was denied or timed out. The PID later disappeared, but host pressure and degraded performance persisted until an operator restart. WER recorded `RADAR_PRE_LEAK_64`; Defender and bundle-coherence checks did not support a security block or mixed DLL set. `FAILURE_LOG.md` F-0008 records the evidence without assigning an unproven root cause.

The owner boundary is now locally proven without loading a model. `scripts/lfm2-vl/test-bounded-oracle.ps1` passes normal exit, timeout plus descendant cleanup, owner-exit cleanup, concurrent-name refusal, suspended creation, assignment before resume, and exact PID absence. The wrapper enforces per-process/per-job memory ceilings, kill-on-close, timeout, a 75%-of-physical-RAM admission maximum, and CUDA-graph disablement by default.

Three bundles remain deliberately separate: legacy b9981 incident evidence; user-supplied `tools-b10344` as a current-master comparison; and the exact pinned b10335 owner build at `74ce15741b420b8d6f12e720398458b576c51c2c`. The pinned CUDA 13.3/SM89 bundle has executable SHA-256 `848e638069699149210b70945bdbb422494d7d03b8a18d7fb31a240d10e8abd0`, a complete local dependency manifest, and a green bounded `--version` probe reporting `10335 (74ce15741)`. This proves artifact identity and containment only, not model parity.

## P3.2 Load-Only Addendum (2026-08-11)

The earlier P3.2 admission paragraphs above were written before the native run and are superseded by this addendum. The bounded native Candle load-only proof is green: the immediate pre-load artifact rehash preserved the eight-file snapshot and manifest SHA-256 `b8d582c40214a1a8df82f21ece21fb683a5e5377c7c03b4fba0e97feb865e585`; executable SHA-256 was `338ebcbf02dbac13fabf6ce9115bdb3a91fc3316a84a9c23e1ad304fbd900d9a`; PID 15792 exited 0 in 2,264 ms under the 12 GiB bound with peak Job memory 6,433,579,008 bytes and exact cleanup. The loader reported 589 tensors, one shard, CPU F32 for vision/text, expected roots, tied output, and tokenizer image token 396. No inference or trace payload was generated. External owner evidence is `C:\DevStuff\candle-oracle\evidence\p3-1.6b-native-load-owner-20260811T204250Z.json`; combined log SHA-256 is `8c8395c2da88d76848fc66830a50c42bfee02b88e291bb27592808ae8acaee3e`.

P3.1 through P3.5 are green; CUDA, lower-bit production MMProj, llama.cpp component/logit equality, and optional WSL replay remain unclaimed.

## P3.3 Official 1.6B Python Component Trace

The immediate pre-trace artifact rehash at `C:\DevStuff\candle-oracle\evidence\p3-1.6b-python-trace-artifact-20260811T210231Z.json` preserved the eight-file, 3,198,084,631-byte snapshot and manifest SHA-256 `b8d582c40214a1a8df82f21ece21fb683a5e5377c7c03b4fba0e97feb865e585`. The exact deterministic image is `C:\DevStuff\candle-oracle\inputs\trace-gradient-256.png` with SHA-256 `f902f8d2e47e53eafac86831cfc692001dc15870eb81d57abc3128f048d2efca`; the user text is `Describe this image.` and the pinned processor rendered the official sentinel-bearing prompt.

The bounded owner used the pinned Python executable SHA-256 `b2c836c52cdf063180b9ee76f67ac42946101b79ac457f3494035a67c090d961`, CPU F32, three cached decode steps, a 24 GiB Job Object, and a 7,200-second timeout. PID 28560 exited 0 in 28,505 ms; peak Job memory was 14,482,644,992 bytes and exact PID cleanup was recorded in `C:\DevStuff\candle-oracle\evidence\p3-1.6b-python-trace-owner-20260811T210231Z.json`. The combined log SHA-256 is `a85229de763b4ac459100d03fdbd6165a5fa99a2247eb52ec6ff1bc8c6ba973c`.

The external bundle `C:\DevStuff\candle-oracle\evidence\python-trace-1.6b-20260811T210231Z` validates through the pinned reference validator: 51 tensors, 182,528,392 safetensors bytes, payload SHA-256 `184d62de07a1b72c8e6a0190b05ef15ff7361c2a029fe5fc2c04a0e17ebbb2f2`, 80 input tokens, 64 projected image tokens, exact cache reset (`max_abs=0`), unchanged artifact manifest, and `weights_serialized=false`. Pinned reference tests passed 81/81.

The clean postflight at `C:\DevStuff\candle-oracle\evidence\p3-1.6b-python-trace-postflight-clean-20260811T210659Z.json` (SHA-256 `fdc6034e85a208a170016c39bae18f52ba258fe9aed378901eeb156e40289853`) recorded zero model/build families, 43.5 GiB physical memory, 47.4 GiB commit headroom, and 23,438 MiB GPU free. The native P3.4/P3.5 gate is recorded below.

## P3.4–P3.5 Official 1.6B Native CPU-F32 Component Parity

The corrected native trace used the exact Python oracle artifact, deterministic image SHA-256 `f902f8d2e47e53eafac86831cfc692001dc15870eb81d57abc3128f048d2efca`, official rendered prompt, CPU F32, single crop, and three decode steps. The release executable SHA-256 is `1f21125cdfe107a42a608920703755c499c7c75cae637b834724d78b175887e0`. Owner evidence is `C:\DevStuff\candle-oracle\evidence\p3-1.6b-native-trace-f32-corrected-owner-20260811T214952Z.json`: PID 4788 exited 0 in 29,486 ms, peak Job memory `6,845,521,920` bytes under the 12 GiB ceiling, and exact PID cleanup. The combined log SHA-256 is `8da2c7137c0f5234bd5e46ca9621dbf5b0f6db75e220eb7cad2b69dc224991ac`.

The native bundle has 51 tensors and 182,528,392 safetensors bytes, exact integer/input identity, 80 input tokens, 64 projected image tokens, exact reset replay, and generated IDs `[1098, 4646, 40027]` (`The image depicts`). The phase-contract comparator report `C:\DevStuff\candle-oracle\evidence\comparison-1.6b-contract-v3-20260811T220300Z.json` has SHA-256 `9a0b16256a222678f9dce1282660e49fc6d19103cc6dd6a53c824bb58a6412c0`, `passed=true`, 51/51 tensors, and zero failures.

The comparator preserves exact integer/input checks; applies the `<=2e-5` resized-position and `<=1e-3` prefill-logit CPU-F32 bounds; uses allclose-or-cosine for vision/projector/hidden-state stages with a `>=0.99999` cosine floor; and keeps structural pixel-unshuffle on allclose. Prefill max abs is `0.0009407997131347656`; vision layer 26's `0.022125244` elementwise drift is accepted by cosine after allclose fails, while layer 9 remains accepted by allclose. Projector, final post-LN, decode logits, output IDs, and reset pass. This is a phase-contract correction for cross-kernel reduction drift, not an unbounded tolerance change.

The final clean postflight `C:\DevStuff\candle-oracle\evidence\p3-close-final-postflight-clean-20260811T221143Z.json` has SHA-256 `1f4399ed6bfbbbf6c6b400054c0cbfebac6fcc8c28ef4d204fdcefbb6fdc4030`, zero tracked model/build processes, 44,067,688,448 available physical bytes, 49,131,601,920 bytes commit headroom, and 23,463 MiB GPU free. The official 1.6B CPU-F32 gate is green; no CUDA or llama.cpp component-tensor claim follows from it.

## Evidence Rules

- Plausible captions are not parity evidence.
- Component tensors and exact metadata must be compared at the applicable phase gate.
- Fixture evidence must not be described as production-checkpoint or production-GGUF parity.
- No result may be marked green until the exact command and result are recorded in `STATUS.md`; move it to `HISTORY.md` when a newer gate supersedes it.

## P4.1 Public Device-Policy Gate

The public example now exposes the complete four-way placement matrix without
changing the existing flags: default accelerator/accelerator,
`--vision-cpu` accelerator/CPU, `--text-cpu` CPU/accelerator, and `--cpu`
CPU/CPU. `--cpu` is authoritative when combined with either component flag.
The existing loader/model path already accepts distinct devices and reports
both resolved placements; this task only made the CPU-text route selectable and
documented. Focused parser/policy tests pass 12/12, the example check passes,
and the trace-only CPU lane rejects `--text-cpu` without `--cpu` as intended.
No CUDA runtime or production checkpoint was used for this gate.

## P4.2 Native CUDA/CPU Distinct-Device Fixture

The native Windows CUDA toolchain is locally available: `nvcc` 13.3.33,
Cargo/rustc 1.91.0, MSVC target, and an RTX 4090 with driver `32.0.16.1088`.
CUDA 13.3's CCCL headers require MSVC's conforming preprocessor, so
`candle-kernels/build.rs` passes `-Xcompiler /Zc:preprocessor` through both
PTX and static-library builders. The first bounded compile failed at that
known requirement before any test ran; the corrected build passed.

The example's CUDA-gated `loads_native_vision_cuda_text_cpu_on_distinct_devices`
test passed 1/1 under a 16 GiB bounded Cargo owner. The transformer
`split_vision_cuda_text_cpu_transfers_only_projected_features` test also passed
1/1: vision remained on CUDA, text/cache/prefill remained on CPU, only the
projected image features crossed devices, and hybrid prefill max abs was
`4.456564784e-5`. The owner records, log hashes, peak Job memory, and exact
cleanup are retained in `STATUS.md` and `HISTORY.md`.

The final postflight recorded zero tracked/llama processes, 43,408,338,944
available physical bytes, 47,523,815,424 bytes commit headroom, and 23,421 MiB
GPU free. This is a tiny fixture/toolchain proof, not official production
checkpoint CUDA parity.

## P4.3 Official 450M CUDA Parity

The final native release executable is `C:\DevStuff\candle-mods\target\release\examples\lfm2-vl.exe`, 65,032,192 bytes, SHA-256 `5b147767e5c45074035d884eaa0b1111ee0ebc6dbf5ed098ee8f120539a8a669`. The admitted checkpoint is `LiquidAI/LFM2.5-VL-450M@fc6221ca597f3315e4f82fc2df606783267b34ba`; artifact-manifest SHA-256 is `659c8421530586b6cc28c094cfcdc69719ea8626f2abc0efd9eec4ac2a68a984`; image SHA-256 is `f902f8d2e47e53eafac86831cfc692001dc15870eb81d57abc3128f048d2efca`; and the rendered prompt is identical to the CPU-F32 baseline.

Sequential bounded runs prove the two supported F32 placements:

- All-CUDA F32 exited 0, peaked at 3,474,706,432 Job bytes, and left PID 14684 absent. Owner SHA-256 is `346925d6be44621b5b03d5c10ca1b69d06c5b279bfb63fabd0d8351ebc82de77`; log SHA-256 is `2991fedc58de944dbe7065cf0a824ec8e6460dce68a4d2ee601a17768c30c076`.
- CPU-text/CUDA-vision F32 exited 0, peaked at 3,241,332,736 Job bytes, and left PID 28720 absent. Owner SHA-256 is `42340a9659dbfa1b889715fd9abd94556c2b1e55b9f2b38d7635e7fb7912d63a`; log SHA-256 is `f92322fd8ce307701220767da0da0c974bcfe62d0d38f63c811b973ac33d05b0`.

Both routes generated `[1098, 4646, 5251]`, expanded the same prompt, projected 64 image tokens, reset the cache exactly, and matched the CPU baseline's top-k IDs at all three steps. Maximum displayed top-k drift was approximately `3.960059e-5` all-CUDA and `2.660059e-5` CPU-text/CUDA-vision. All-CUDA BF16 also exited 0 with peak Job memory 2,783,182,848 bytes (owner SHA-256 `b01977d99ea6dd5fb64ad8d552bc4a932d3e6654356fc706a755024f4a174e94`; log SHA-256 `c143193787e869864e8c071abc9fa72b4a580ea93d3d75657dfa7e040cec8764`). Explicit BF16 on a CPU component is intentionally rejected before model load; guarded exit-1 evidence owner SHA-256 is `2317d3bc44650f0b35862ed139eee4d86be10cc07f5541e58d7eb8b8c194f465`, log SHA-256 `e1a4285ea5bcc186d869a214defc8b04754414ba1714a7dddbf53c5a21e6f78`.

The gate required two source fixes: CUDA `I32 -> F32` cast registration for packed masks and contiguous materialization before dense CUDA matmul. The public `--text-cpu` path now constructs CUDA vision independently instead of inheriting the CPU text device. P4.3 is green for the admitted native 450M placements; P4.4 now owns measured CUDA optimization.

## P4.4 Diagnostic Timing Baseline

The release executable was rebuilt with the opt-in `--timings` diagnostic. The
timing-series artifact was 65,035,264 bytes with SHA-256
`b25984aac5332f2655ba91478dec5a46b4fe6e538760891c2a6591816c54d81a`. After
reverting the unproven candidate, the current source-matching rebuild is the
same size with SHA-256
`9cd51ffefbae6e5c80907629817e0a27a854fc3325e61aa041f79eec9c7998c8`; build
hashes are retained per owner because native release rebuilds are not
byte-identical. The
bounded build owner exited 0, peaked at 890,847,232 Job bytes, and left its
PID absent; owner SHA-256 is
`7345c2bf2401c126ea828abdfb60f845ff50eab2f07e9d53e8aff44265f956fa` and log
SHA-256 is `755aac9083bf0eafce60c8ef7da83cf8f0fbc599815968f6b505878549c596d3`.

Three sequential all-CUDA F32 runs over the exact 450M checkpoint and
deterministic 256px fixture exited 0, left their PIDs absent, and peaked at
3,475,668,992, 3,474,984,960, and 3,476,611,072 Job bytes. The reported
stage ranges were model load 435.885–452.501 ms, processor 0.980–1.051 ms,
prompt 20.216–20.727 ms, vision 38.037–39.186 ms, first generation
446.959–469.142 ms, cache-reset replay 419.065–444.422 ms, and total
inference 1,388.056–1,462.027 ms. The three owner/log pairs are retained
externally under `C:\DevStuff\candle-oracle\evidence` with owner/log hashes:
`d5a7a5622cbd9f867f14d39f3632f780faf17f00fa7a22cdf191e85734d99a85` /
`25f49f9c5b89789c2ae70b94888723b87f93e4dc37a205861c574e7473b3c21f`,
`89b3535a3bafd84e839bbd4f7bd3faa351161ba1563fdeda910d097c67f1def7` /
`a1527db845ada92acc5a08e48b8fdceb1f88d1ed76b4facecea29dbfc1760405`, and
`795ab08a33c5556e4c6cd9785915791497bb03db45c5aee7baab169fbfa03aad` /
`6ec4f82dc8a0cbe30ace447c963345dbee7757c78d4d9cdfe18904d3c2cb9f0a`.

The timing-only probe does not change the versioned JSON evidence contract.
Generation is the largest measured stage, but token-count narrowing showed
warm-up and runtime variance, so no optimization is claimed yet. The next
P4.4 action is a decode/cache microbenchmark with repeated warm-up and an
explicit variance bound before changing one generation hot path.

## Next Parity Task

NR-5B, the official-base GGUF same-artifact comparison, P3.1–P3.5, P4.1,
P4.2, and P4.3 are green. P4.4's stage-timed baseline is captured; next,
measure the decode/cache hot path with repeated warm-up, change one measured
bottleneck, and replay CPU/CUDA parity. Lower-bit CUDA and the optional WSL
replay remain later lanes.

---
AI-edited: 2026-08-11T19:30:00-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=p4-3 | change=recorded official 450M CUDA F32/BF16 parity and advanced the active gate to measured optimization
