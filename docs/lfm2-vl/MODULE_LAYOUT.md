# LFM2-VL Module Layout

This document records the C1 production-source split. The split is intentionally behavior-neutral: public APIs, serialization contracts, evidence schemas, fixture locations, and test modules remain in their original wrapper modules. Production responsibilities move into bounded same-module source units through `include!`, avoiding a new abstraction layer or visibility churn.

## Design Rules

- Wrapper files retain module documentation, imports, constants, and tests.
- Included production files share the wrapper's module scope.
- A wrapper may contain at most 900 lines.
- A production source part may contain at most 500 lines.
- Tests never move into included production files.
- `scripts/lfm2-vl/verify-module-layout.py` enforces the include inventory and size limits.

## Before and After

| Original file | Before | Wrapper after | Largest production part | Split responsibility |
| --- | ---: | ---: | ---: | --- |
| `candle-transformers/src/models/lfm2.rs` | 1,484 | 490 | 405 | Configuration, cache, layers, model API |
| `candle-transformers/src/models/siglip2.rs` | 1,286 | 370 | 256 | Configuration, embeddings, encoder, model, interpolation |
| `candle-transformers/src/models/lfm2_vl/gguf.rs` | 1,952 | 869 | 398 | Types, loading, metadata, inventory, metadata values |
| `candle-transformers/src/models/lfm2_vl/weights.rs` | 1,889 | 733 | 453 | Manifest/pairing, runtime, safetensors inspection |
| `candle-transformers/src/models/lfm2_vl/model.rs` | 1,537 | 535 | 363 | Types, runtime, image encoding, merge/validation, config extension |
| `candle-vlm/src/lfm2_vl/processor.rs` | 1,324 | 603 | 289 | Types, entry/resize, budgets, crop construction, helpers |
| `candle-vlm/src/lfm2_vl/prompt.rs` | 1,303 | 583 | 242 | Types, token resolution, expansion, validation, image blocks, helpers |
| `candle-examples/examples/lfm2-vl/runner.rs` | 1,555 | 544 | 260 | Report types, runtime adapters, orchestration, generation, evidence |
| `candle-examples/examples/lfm2-vl/native_loading.rs` | 1,209 | 704 | 310 | Load/report types, construction, inventory validation |

## Why Same-Module Source Units

These files already expose proven APIs and have broad private-item coupling. Converting every seam into a nested Rust module would require visibility changes unrelated to product behavior. Same-module source units reduce context and merge pressure while preserving exact name resolution. A future abstraction change must be justified by reuse or measured coupling, not by file length alone.

## Verification Contract

C1 remains green only when:

1. `python3 scripts/lfm2-vl/verify-module-layout.py` passes.
2. Formatting, focused tests, example tests, and scoped Clippy pass.
3. Existing fixture and production trace contracts remain unchanged.
4. `summary_bank.json` routes each responsibility to its bounded source files.
5. `MOD_MANIFEST.md` contains every added source unit.

---
AI-edited: 2026-08-11T11:30:00-04:00 | agent=ChatGPT | model=gpt-5.6-pro | task=C1 | change=recorded bounded source split and verification contract
