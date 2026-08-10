# LFM2-VL Reference Harness

This harness has three explicit modes:

| Mode | Inputs | Output |
| --- | --- | --- |
| `config-only` | Checked-in `reference-lock.json`, optionally local small JSON files | Normalized dimensions and revision metadata; never imports Torch or Transformers and never reads weights |
| `tiny-random` | Deterministic CPU seed and reduced configs | Safetensors plus stable JSON from the official pinned Transformers LFM2, SigLIP2, and LFM2-VL classes |
| `production` | Explicit model revision and an explicit safety flag | Metadata by default; `--load-model` can load the pinned model locally or with a second explicit download flag, but this tool never writes production tensor payloads |

The source lock is `tools/lfm2_vl/reference-lock.json`. The tiny model uses the locked Transformers commit `fd12552d770f745fdbe41031ff4daa688f5ed57e` and records both official LiquidAI model revisions in its metadata. It exercises packed linear patches, resized learned positions with antialias semantics, bidirectional masked vision attention, post LayerNorm, factor-2 pixel unshuffle, optional projector LayerNorm, both projector linears, image-placeholder replacement, and the official LFM2 attention/short-convolution layer classes.

## Config-only

This path works with stdlib-only Python:

```bash
python3 tools/lfm2_vl/reference/inspect_config.py --model 450m
python3 tools/lfm2_vl/reference/export_fixtures.py --mode config-only --model 1.6b
```

Small local `config.json` and `processor_config.json` files can be supplied with `--config` and `--processor-config`. The files must be JSON and are bounded to a small configuration size; weight files are not accepted.

## Manager setup and tiny fixture export

Use the pinned requirements in a project-local environment. Installation is intentionally not performed by the harness task:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r tools/lfm2_vl/reference/requirements-reference.txt
python tools/lfm2_vl/reference/export_fixtures.py \
  --mode tiny-random \
  --seed 1234 \
  --output artifacts/lfm2-vl/reference/tiny-1234
python -m pytest tools/lfm2_vl/reference
```

The exporter refuses an existing output directory unless `--overwrite` is supplied. Tiny output is deterministic for a fixed seed and package set; the manifest hashes both JSON metadata and safetensors. The synthetic raw RGB image is the exact byte source for `source_image_sha256`, and the packed patch tensor is derived from those same bytes. No access token, tokenizer text, model cache, or production weight is serialized.

## Production guard

Production mode requires `--allow-production`, requires an output directory outside this repository, and is metadata-only unless model loading is explicitly requested:

```bash
python tools/lfm2_vl/reference/export_fixtures.py \
  --mode production \
  --model 450m \
  --allow-production \
  --output /tmp/lfm2-vl-production-metadata
```

To exercise the actual pinned model loader, add `--load-model`. Hub access is local-cache-only unless the separate `--allow-download` flag is also supplied:

```bash
python tools/lfm2_vl/reference/export_fixtures.py \
  --mode production \
  --model 450m \
  --allow-production \
  --load-model \
  --output /tmp/lfm2-vl-production-model-check
```

The loader calls the pinned `Lfm2VlForConditionalGeneration.from_pretrained` path and never serializes the loaded tensors. Keep any user-fetched model artifacts outside Git and outside the tiny fixture directory.

## Validation

`requirements-reference.in` is the direct CPU-lane intent. `requirements-reference.txt` is the fully resolved Python 3.10.12 / Linux x86_64 CPU verification lock. `tensor_dump.validate_bundle()` checks stable JSON, safetensors SHA-256, tensor names, shapes, and dtypes. The focused tests cover config-only behavior, official tiny construction, deterministic regeneration, overwrite refusal, production opt-in, mocked production loading, and hash failure.

---
AI-edited: 2026-08-09T23:21:10-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=reference-harness | change=added pinned official-class reference modes and deterministic bundle contract
