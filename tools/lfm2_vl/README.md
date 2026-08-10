# LFM2-VL Tools

This directory contains the reference and fixture tooling described by the LFM2.5-VL execution plan. The implementation lives under `reference/` and is intentionally separate from Candle runtime code.

Future tools must keep these boundaries:

- Configuration-only inspection must not import heavy packages or download weights.
- Production model loading requires `--allow-production`; Hub downloads require the separate `--allow-download` flag and loaded tensors are never serialized by this harness.
- Tiny deterministic fixtures may be committed under `tests/fixtures/lfm2_vl_tiny/`.
- Local caches and reference outputs belong to ignored paths.
- Reference revisions, package versions, image hashes, dtype, device, and seed must be recorded with generated outputs.

Run the stdlib-only config path with `python3 tools/lfm2_vl/reference/inspect_config.py --model 450m`. The exact CPU-lane setup, official Transformers tiny-random oracle, production guard, and manager-resolution-pending requirements are documented in `tools/lfm2_vl/reference/README.md`.

---
AI-edited: 2026-08-09T23:21:10-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=reference-harness | change=connected tool entrypoint to guarded official-class reference harness
