# LFM2-MoE authorities and inputs

| Authority | Exact revision | Use |
| --- | --- | --- |
| LiquidAI/LFM2.5-24B-A2B | `3dc4cd09c9a2a99e1e819693e79d3aba5e9bb4f2` | Official configuration, checkpoint names, tokenizer and template |
| LiquidAI/LFM2.5-24B-A2B-GGUF | `7be3a8c77b010e612c42089d0dc082962cecb3a6` | GGUF names, metadata and mixed weight types |
| huggingface/transformers 5.12.1 | `ddb849abe009d1089e6c691bfc897f27211c663c` | Numerical oracle, configuration and cache behavior |
| ggml-org/llama.cpp | `582eab718608b99792286729890aac1675547c29` | Independent GGUF and production comparison reference |

Transformers files `src/transformers/models/lfm2_moe/configuration_lfm2_moe.py`
and `modeling_lfm2_moe.py` are Apache-2.0 references. The Rust implementation is
Candle-native; the fixture generator invokes the exact local source with eager
attention/experts, without checkpoint weights. Its manifest binds source-file
hashes, dependencies, deterministic seed, dtype/device and output file hashes.

The official native checkpoint stores per-expert `w1/w2/w3`; Transformers 5.12.1
uses packed expert parameters in memory. Fixture export reverses that packing
explicitly into the published checkpoint names.
The locked native shard index confirms 7,610 published names, including the
per-expert weights and expert bias. `source-lock.json` records its exact bytes.

The task-local CPU reference probe builds against the pinned llama.cpp source
archive (SHA-256 `4916b62d2b836d003f79c03924d77a4ad904ea1b4e98a3d906d1e953eefacc83`).
The archive source is externally retained and not vendored into Candle. The
small probe is fresh code using llama.cpp's public API; its MIT reference and
the bundled nlohmann JSON dependency retain their upstream notices in that
external tree. It is a qualification tool, not a Candle production dependency.

The maintained quantized CPU schedules use the connected `9c8006a5`
diagnostic as their arithmetic authority. The normal-range SiLU polynomial
adapts `ggml/src/ggml-cpu/vec.h` at `582eab71` with its MIT notice in
`quantized_lfm2_moe/cpu.rs`. The dense CUDA port in `candle-kernels/src/reduce.cu`
uses the same arithmetic and source notice. Packed scales are decoded with the already
workspace-pinned `half` dependency. GGUF weights remain quantized.

The quantized MoE CPU attention order comes from pinned llama.cpp `582eab71`.
The reference uses `ggml/src/ggml-cpu/vec.cpp` and `simd-mappings.h` for Q·K,
`vec.cpp` and `vec.h` for softmax, and `llamafile/sgemm.cpp` under the same CPU
directory for value reduction. The adapted exponent retains the ggml MIT
notice and Arm attribution in `quantized_lfm2_moe/cpu_attention.rs`.
The GGUF MoE CUDA attention kernels in `candle-kernels/src/reduce.cu`
port those same documented AVX2 folds and exceptional exponent split.

The single-row dense Q6_K FFN fold follows the existing Candle CPU
plain dot. Its reference is pinned llama.cpp `582eab71`, MIT, in
`ggml/src/ggml-cpu/arch/x86/quants.c` (`ggml_vec_dot_q6_K_q8_K`) and
`ggml/src/ggml-cpu/llamafile/sgemm.cpp` (the single-row dispatch).

The Q4_K CUDA matrix-matrix Q8_1 sum follows the quantized-byte sum
in pinned llama.cpp's MIT-licensed
[ggml/src/ggml-cuda/quantize.cu](https://github.com/ggml-org/llama.cpp/blob/582eab718608b99792286729890aac1675547c29/ggml/src/ggml-cuda/quantize.cu#L459-L477).
The Candle kernel retains its existing grouping and uses this source for
the sum formula.

The explicit MoE ShortConv CUDA projection follows the maintained
MoE CPU Q8_K activation and Q4_K GEMM/GEMV schedule. Its verification
uses the same pinned CPU reference and retained 37-row projection
operands from the layer-0 comparison.

The verified external file is
`C:/llamacpp/models/LFM2.5-24B-A2B-Q4_K_M.gguf`, 14,526,258,464 bytes, SHA-256
`d3334405b96228eb9908cdebb5c6194908f2da64282bda852afc3a3910c64e7d`.
It has 428 tensors: 207 F32, 195 Q4_K and 26 Q6_K. Its 8,250,144-byte header has
SHA-256 `4145b9cf5c17e0173d658cda7357273db34aca23368cedd957737acc01ec645f`.
Header/hash inspection is not runtime qualification. LFM Open License v1.0 owns
model assets; production weights and upstream Python source are not committed.

---
AI-edited: 2026-11-10 | agent=Codex/root | task=lfm2-moe | change=pin numerical, checkpoint, tokenizer and GGUF authorities
AI-edited: 2026-11-17 | agent=Codex/root | task=moe-cuda-shortconv | change=bind the crossed replay and Q4_K activation sum correction
AI-edited: 2026-11-18 | agent=Codex/root | task=moe-unicode-position2 | change=attribute the dense Q6_K decode fold and dispatch
