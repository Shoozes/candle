# LFM2-MoE overlay

This independent capability owns native and GGUF MoE support. Historical
release inventories, D1 proof and consumer pins remain unchanged.

## Owned additions

- `candle-examples/examples/lfm2-moe/README.md`
- `candle-examples/examples/lfm2-moe/main.rs`
- `candle-transformers/src/models/lfm2_moe/config.rs`
- `candle-transformers/src/models/lfm2_moe/mod.rs`
- `candle-transformers/src/models/lfm2_moe/model.rs`
- `candle-transformers/src/models/lfm2_moe/routing.rs`
- `candle-transformers/src/models/lfm2_moe/tests.rs`
- `candle-transformers/src/models/quantized_lfm2/mixer.rs`
- `candle-transformers/src/models/quantized_lfm2_moe/metadata.rs`
- `candle-transformers/src/models/quantized_lfm2_moe/mod.rs`
- `candle-transformers/src/models/quantized_lfm2_moe/model.rs`
- `candle-transformers/src/models/quantized_lfm2_moe/tests.rs`
- `docs/lfm2-moe/DECISIONS.md`
- `docs/lfm2-moe/HISTORY.md`
- `docs/lfm2-moe/MOD_MANIFEST.md`
- `docs/lfm2-moe/QUALIFICATION.md`
- `docs/lfm2-moe/SOURCES.md`
- `docs/lfm2-moe/STATUS.md`
- `docs/lfm2-moe/TODO.md`
- `docs/lfm2-moe/source-lock.json`
- `tests/fixtures/lfm2_moe_tiny/README.md`
- `tests/fixtures/lfm2_moe_tiny/manifest.json`
- `tests/fixtures/lfm2_moe_tiny/native.json`
- `tests/fixtures/lfm2_moe_tiny/native.safetensors`
- `tests/fixtures/lfm2_moe_tiny/qualification_inputs.json`
- `tests/fixtures/lfm2_moe_tiny/quantized.json`
- `tests/fixtures/lfm2_moe_tiny/quantized.safetensors`
- `tests/fixtures/lfm2_moe_tiny/tokenizer_reference.json`
- `tools/lfm2_moe/CMakeLists.txt`
- `tools/lfm2_moe/compare_probes.py`
- `tools/lfm2_moe/export_fixture.py`
- `tools/lfm2_moe/export_tokenizer_fixture.py`
- `tools/lfm2_moe/llama_probe.cpp`
- `tools/lfm2_moe/prepare_packet.py`
- `tools/lfm2_moe/prepare_qualification.py`
- `tools/lfm2_moe/test_compare_probes.py`
- `tools/lfm2_moe/test_prepare_packet.py`
- `tools/lfm2_moe/verify_tokenizer.py`

## Shared hunk ownership

- `.gitignore`
- `.gitattributes`
- `candle-transformers/src/models/lfm2/layers.rs`
- `candle-transformers/src/models/quantized_lfm2.rs`
- `candle-transformers/src/models/mod.rs`
- `docs/FORK_OVERLAYS.md`
- `docs/lfm2-vl/STATUS.md`
- `docs/lfm2-vl/TODO.md`
- `docs/lfm2-vl/DECISIONS.md`
- `scripts/verify-fork-overlays.sh`
- `scripts/tests/test-verify-fork-overlays.sh`
- `summary_bank.json`

Only crate-private mixer reuse, model registry entries, byte-stable fixtures
and current capability routing/ownership are changed on shared paths.

---
AI-edited: 2026-11-12 | agent=Codex/root | task=lfm2-moe-review | change=register the qualification packet inventory regression test
