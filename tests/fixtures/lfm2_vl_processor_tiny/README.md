# LFM2-VL Processor Fixture

This deterministic, no-weight bundle was exported by the pinned official
`Lfm2VlImageProcessor` and `Lfm2VlProcessor` classes:

```bash
/home/workbench/code/candle-lfm2-vl-reference-verify/.venv/bin/python \
  tools/lfm2_vl/reference/export_processor_fixture.py \
  --output tests/fixtures/lfm2_vl_processor_tiny --overwrite
```

It covers RGB, grayscale, RGBA, square/rectangular/odd/upscaled inputs,
tiled crops with thumbnails, multiple images, real-dimension metadata, and
official prompt expansions with per-crop placeholder spans. `metadata.json`
records source byte hashes, exact expanded strings, IDs, ranges, shapes, and
the pinned CPU package/oracle revision. `manifest.json` validates the sorted
safetensors inventory and hashes.

The fixture contains no production weights, credentials, or user content.

---
AI-edited: 2026-08-10T00:00:00-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=lfm2-vl-processor | change=added official raw-image and prompt parity fixture
