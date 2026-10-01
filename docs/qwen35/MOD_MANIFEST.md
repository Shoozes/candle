# Qwen3.5 GGUF Compatibility Overlay

This overlay owns Candle-side Qwen3.5 text support. It does not change the
maintained Q8 product default or activate an EdgeSymbio model. The first
artifact-specific fixture is kept separate from the config-driven model code.

## Owned additions

- `docs/qwen35/MOD_MANIFEST.md`
- `docs/qwen35/DECISIONS.md`
- `docs/qwen35/SOURCES.md`
- `docs/qwen35/STATUS.md`
- `docs/qwen35/PERFORMANCE.md`
- `docs/qwen35/INTEGRATION.md`
- `docs/qwen35/HISTORY.md`
- `candle-transformers/src/models/qwen35/mod.rs`
- `candle-transformers/src/models/qwen35/admission.rs`
- `candle-transformers/src/models/qwen35/cpu.rs`
- `candle-core/src/quantized/k_quants.rs`
- `candle-core/src/quantized/cuda.rs`
- `candle-core/src/quantized/fast_mmvq.rs`
- `candle-core/src/quantized/mod.rs`
- `candle-core/src/quantized/repack_x86.rs`
- `candle-kernels/src/mmvq_gguf.cu`
- `scripts/qwen35/run-bounded-parity.ps1`
- `scripts/qwen35/run-bounded-performance.ps1`
- `scripts/qwen35/run-bounded-cuda.ps1`
- `scripts/qwen35/probe-llama-batch.ps1`
- `scripts/qwen35/llama-layer-probe.cpp`
- `scripts/qwen35/run-bounded-layer-probe.ps1`
- `tests/fixtures/qwen35_codename_b/README.md`
- `tests/fixtures/qwen35_codename_b/inspection.json`
- `tests/fixtures/qwen35_codename_b/reference.json`

## Shared paths

- `.gitattributes`
- `candle-kernels/src/ffi.rs`
- `candle-transformers/src/models/mod.rs`
- `docs/FORK_OVERLAYS.md`
- `docs/lfm2-vl/STATUS.md`
- `docs/lfm2-vl/TODO.md`
- `scripts/verify-fork-overlays.sh`
- `summary_bank.json`

All shared paths are listed in the root registry. The LFM2-VL and GPT-OSS
overlays retain independent implementation and proof ownership.
