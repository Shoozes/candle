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

## Split dense MMProj export

`tools/export_lfm2_vl_mmproj.py` is a separate stdlib-only development tool. It accepts a local safetensors file plus local model and processor JSON, streams only the canonical `model.vision_tower.*` and `model.multi_modal_projector.*` payloads, and emits `mmproj.safetensors`, `mmproj.json`, and `processor_config.json`. It validates source offsets and byte sizes, refuses non-dense MMProj tensors, writes atomically, and never downloads a model.

The committed `tests/fixtures/lfm2_vl_mmproj_tiny/` bundle is derived byte-for-byte from the no-production-weight tiny fixture. `test_mmproj_exporter.py` proves deterministic regeneration, the exact 43-tensor namespace, hashes and version fields, overwrite refusal, processor/model mismatch diagnostics, and controlled failure when the requested source namespace is absent.

## Bounded GGUF header inspection

`inspect_gguf_header.py` is a stdlib-only parser for a local bounded GGUF prefix. It validates magic, version, counts, string/array lengths, dimensions, dtypes, alignment, and header completeness, then reports both raw GGUF dimensions and Candle logical shapes. It never fetches a URL itself.

For the pinned official 450M F16 and Q8_0 MMProj files, the complete aligned header is exactly bytes `0-12735` and tensor data starts at byte `12736`. Keep the temporary prefix outside the repository and request only that exact range:

```bash
curl --fail --location \
  --header 'Range: bytes=0-12735' \
  --output /tmp/mmproj-header.gguf \
  'https://huggingface.co/LiquidAI/LFM2.5-VL-450M-GGUF/resolve/166cd80bbe157dc86d65f964eb8cc6a2cede62ca/mmproj-LFM2.5-VL-450m-Q8_0.gguf'

python3 tools/lfm2_vl/reference/inspect_gguf_header.py \
  /tmp/mmproj-header.gguf \
  --source-revision 166cd80bbe157dc86d65f964eb8cc6a2cede62ca \
  --byte-range bytes=0-12735 \
  --summary-only
```

Omit `--summary-only` for the complete metadata and tensor inventory, or use `--output` to write JSON outside the repository. The result must report `contains_tensor_payload=false`; do not retain or commit production prefixes. Exact official hashes and parsed facts live in `reference-lock.json`, and the unit test checks their zero-payload boundary.

## Validation

`requirements-reference.in` is the direct CPU-lane intent. `requirements-reference.txt` is the fully resolved Python 3.10.12 / Linux x86_64 CPU verification lock. `tensor_dump.validate_bundle()` checks stable JSON, safetensors SHA-256, tensor names, shapes, and dtypes. The focused tests cover config-only behavior, official tiny construction, deterministic regeneration, overwrite refusal, production opt-in, mocked production loading, and hash failure.

---
AI-edited: 2026-08-10T07:31:00-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=lfm2-vl-phase-6 | change=documented bounded zero-payload GGUF header inspection and locked official ranges
