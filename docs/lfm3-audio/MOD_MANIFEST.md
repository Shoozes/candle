# LFM3-Audio overlay

This independent overlay owns reusable native LFM3-Audio operations and their
finite example. D1, diffusion, GPT, Qwen and fork compatibility retain their
source-bound behavior and acceptance. Current work does not alter released
MVP inventories or consumer pins.

## Owned additions

- `docs/lfm3-audio/MOD_MANIFEST.md`
- `docs/lfm3-audio/STATUS.md`
- `docs/lfm3-audio/TODO.md`
- `docs/lfm3-audio/SOURCES.md`
- `docs/lfm3-audio/HISTORY.md`
- `candle-transformers/src/models/lfm3/mod.rs`
- `candle-transformers/src/models/lfm3/config.rs`
- `candle-transformers/src/models/lfm3/layers.rs`
- `candle-transformers/src/models/lfm3/tests.rs`
- `candle-transformers/src/models/lfm3_audio/mod.rs`
- `candle-transformers/src/models/lfm3_audio/encoder.rs`
- `candle-transformers/src/models/lfm3_audio/features.rs`
- `candle-transformers/src/models/liquid_codec/mod.rs`
- `candle-transformers/src/models/liquid_codec/signal.rs`
- `candle-transformers/src/models/liquid_codec/stream.rs`
- `candle-examples/examples/lfm3-audio/main.rs`
- `candle-examples/examples/lfm3-audio/asr.rs`
- `candle-examples/examples/lfm3-audio/README.md`

## Shared paths and hunk ownership

- `Cargo.lock`
- `candle-transformers/Cargo.toml`
- `candle-transformers/src/models/mod.rs`
- `candle-transformers/src/models/lfm2.rs`
- `candle-transformers/src/models/lfm2/cache.rs`
- `candle-transformers/src/models/lfm2/model.rs`
- `docs/FORK_OVERLAYS.md`
- `docs/lfm2-vl/STATUS.md`
- `docs/lfm2-vl/TODO.md`
- `scripts/verify-fork-overlays.sh`
- `scripts/tests/test-verify-fork-overlays.sh`
- `summary_bank.json`

This overlay owns only the RustFFT direct dependency edge, new model registry
entries and LFM3 routing/ownership additions. Existing security pins, behavior,
source identities and all released evidence remain independently owned.
The feature-only LFM2 body and private uncached window mask serve the released
audio encoder; ordinary LFM2 constructors and cache defaults are unchanged.

---
AI-edited: 2026-10-28 | agent=Codex/root | task=lfm3-audio | change=register independent native audio source ownership
