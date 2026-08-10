# LFM2-VL Tiny Fixture

This directory contains the committed deterministic CPU fixture produced by the pinned official Transformers LFM2, SigLIP2, and LFM2-VL classes:

```bash
python tools/lfm2_vl/reference/export_fixtures.py \
  --mode tiny-random \
  --seed 1234 \
  --output tests/fixtures/lfm2_vl_tiny \
  --overwrite
```

- `tensors.safetensors` contains 87 sorted input, weight, vision, projector, multimodal-prefill, and three-step cached-decode tensors. The tied `lm_head.weight` duplicate is deliberately omitted to preserve the official checkpoints' missing-head loading contract.
- `metadata.json` records the immutable source revisions, exact reference packages, synthetic image hash, official class inventory, dimensions, dtype, device, and seed.
- `manifest.json` records the tensor inventory and SHA-256 hashes used by `tensor_dump.validate_bundle()`.

Two independent exports were byte-identical in the locked Python 3.10.12 Linux x86_64 CPU environment. The fixture contains no production model weights, user content, access tokens, or Hugging Face cache data.

---
AI-edited: 2026-08-09T23:47:42-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=reference-harness | change=documented deterministic tiny fixture inventory and provenance
