# LFM2-VL Split MMProj Fixture

This no-production-weight bundle was extracted byte-for-byte from the pinned
tiny-random LFM2-VL safetensors fixture. It contains only the canonical
`model.vision_tower.vision_model.*` and `model.multi_modal_projector.*`
namespaces.

Regenerate it with the stdlib-only exporter:

```bash
python3 tools/export_lfm2_vl_mmproj.py \
  --input tests/fixtures/lfm2_vl_tiny/tensors.safetensors \
  --model-config tests/fixtures/lfm2_vl_mmproj_tiny/source_model_config.json \
  --processor-config tests/fixtures/lfm2_vl_mmproj_tiny/processor_config.json \
  --output-dir tests/fixtures/lfm2_vl_mmproj_tiny \
  --source-model LiquidAI/LFM2.5-VL-450M-tiny-random \
  --source-revision fc6221ca597f3315e4f82fc2df606783267b34ba \
  --source-prefix weights. \
  --overwrite
```

Pinned SHA-256 values:

- `mmproj.safetensors`: `9ef641ccc2d1587b6c6499ca2a9dee874d89f1aa5f53e2576d591c70414e930a`
- `mmproj.json`: `b932d4e6c58224d6d97182b0aa969c701beafb0130e2f6031bba189cf9d04f39`
- `processor_config.json`: `97b79ebfc8eae3a5bcbeb8f1494c1decdbade5d20d3204739143d17b460906f2`

The fixture contains no credentials, user content, production tensors, or
GGUF payloads.

---
AI-edited: 2026-08-10T04:42:00-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=lfm2-vl-phase-5 | change=added deterministic split dense mmproj fixture and regeneration contract
