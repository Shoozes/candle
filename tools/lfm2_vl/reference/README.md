# LFM2-VL Reference Harness

This harness has three explicit modes:

| Mode | Inputs | Output |
| --- | --- | --- |
| `config-only` | Checked-in `reference-lock.json`, optionally local config, processor, and tokenizer JSON files | Normalized dimensions, revision metadata, and image-marker token IDs; never imports Torch or Transformers and never reads weights |
| `tiny-random` | Deterministic CPU seed and reduced configs | Safetensors plus stable JSON from the official pinned Transformers LFM2, SigLIP2, and LFM2-VL classes |
| `production` | Explicit model revision and an explicit safety flag | Metadata by default; `--load-model` can load the pinned model locally or with a second explicit download flag, but this tool never writes production tensor payloads |

The source lock is `tools/lfm2_vl/reference-lock.json`. The tiny model uses the locked Transformers commit `fd12552d770f745fdbe41031ff4daa688f5ed57e` and records both official LiquidAI model revisions in its metadata. It exercises packed linear patches, resized learned positions with antialias semantics, bidirectional masked vision attention, post LayerNorm, factor-2 pixel unshuffle, optional projector LayerNorm, both projector linears, image-placeholder replacement, and the official LFM2 attention/short-convolution layer classes. Production loading checks the locked Python/package versions and Transformers VCS commit before importing the model.

## Config-only

This path works with stdlib-only Python:

```bash
python3 tools/lfm2_vl/reference/inspect_config.py --model 450m
python3 tools/lfm2_vl/reference/inspect_config.py --model 1.6b --tokenizer /path/to/tokenizer.json
python3 tools/lfm2_vl/reference/inspect_config.py --model 3b --tokenizer /path/to/3b/tokenizer.json
python3 tools/lfm2_vl/reference/export_fixtures.py --mode config-only --model 1.6b
```

Small local `config.json`, `processor_config.json`, and `tokenizer.json` files can be supplied with `--config`, `--processor-config`, and `--tokenizer`. The files must be JSON and are bounded to a small configuration size; weight files are not accepted. Tokenizer inspection verifies the model's image placeholder ID, requires at least one row/column marker, and reports distinct in-vocabulary wrapper, thumbnail, and grid marker IDs without importing `tokenizers`. Standalone `--output` publication refuses an existing or racing destination unless the command exposes and receives an explicit `--overwrite` flag.

The lock aliases are `450m`, `1.6b`, and `3b`. The 3B config-only path
checks the actual 30-layer/2,048-width text model, 27-layer/1,152-width
SigLIP2 tower, 16-pixel patch geometry, 4,608-wide projector input, image
token `124907`, processor tile/patch limits, and tokenizer marker IDs. The
exact 3B snapshot currently contains no model Python code and has an empty
`auto_map`; its locked policy therefore does not enable `trust_remote_code`.
If a future exact snapshot needs custom code, every `.py` file must be listed
and hash-matched in the external artifact manifest before admission.

## Guarded snapshot acquisition

`acquire_snapshot.py --plan` validates the pinned revision, exact file list,
source identities, destination boundaries, and disk admission without importing
Hub code, using the network, or creating a path. The output, cache, and manifest
must be external to this repository, non-nested, and beneath existing regular
directories:

```powershell
$Snapshot = "C:\DevStuff\candle-oracle\lfm2-vl-1.6b-919fde3d"
$Cache = "C:\DevStuff\candle-oracle\.hf-cache-1.6b"
$Manifest = "C:\DevStuff\candle-oracle\evidence\p3-1.6b-artifact-manifest.json"

python tools\lfm2_vl\reference\acquire_snapshot.py `
  --model 1.6b --output-dir $Snapshot --cache-dir $Cache `
  --manifest $Manifest --plan
```

The multi-gigabyte write remains a separate owner-approved action. After a
green plan, run the exact Python executable through the existing Job Object
owner; invoke the wrapper directly in the current PowerShell process so the
argument array crosses no extra shell boundary:

```powershell
$AcquisitionArgs = @(
  "tools\lfm2_vl\reference\acquire_snapshot.py",
  "--model", "1.6b",
  "--output-dir", $Snapshot,
  "--cache-dir", $Cache,
  "--manifest", $Manifest,
  "--allow-production-download"
)

& scripts\lfm2-vl\run-bounded-oracle.ps1 `
  -FilePath (Resolve-Path ".\.venv\Scripts\python.exe").Path `
  -ArgumentList $AcquisitionArgs `
  -TimeoutSeconds 7200 `
  -MaxJobMemoryBytes 2147483648 `
  -ConcurrencyScope Executable `
  -WorkingDirectory (Get-Location).Path `
  -LogPath "C:\DevStuff\candle-oracle\evidence\p3-1.6b-acquisition.log" `
  -EvidencePath "C:\DevStuff\candle-oracle\evidence\p3-1.6b-acquisition-owner.json"
```

The 2 GiB ceiling is for transfer only and must not be raised automatically.
Timeout or limit termination leaves the caller-owned cache resumable but cannot
make a snapshot admissible: any stale snapshot or manifest staging path, final
output without its matching manifest, or manifest without final output blocks
retry for operator inspection. Acquisition uses the exact immutable revision,
downloads the eight files serially with
`token=False`, retains resumable data only in the caller-owned Hub cache,
requires every returned source to resolve inside that cache, streams each file
into a clean staging directory, verifies byte counts and locked Git/LFS
identities, then atomically publishes the regular-file snapshot without
replacing a destination that appeared after planning. The manifest is written
and flushed to a sibling temporary file, then published through an atomic
no-clobber hard link; a racing owner file is preserved and snapshot publication
rolls back.
Before importing Hub, it forces `HF_HUB_DISABLE_XET=1`; this keeps
the large file on Hub's resumable HTTP path instead of the installed Xet
backend's parallel chunk transfer. A process that already imported Hub with Xet
enabled is refused. It never loads the model. The download path checks only the
direct `huggingface-hub==1.5.0` dependency and its ordinary transfer
dependencies; the full CPU oracle environment is required later for model
loading, tensor traces, and comparison. The production acquisition API exposes
no alternate downloader or artifact-verifier callback; unit tests replace
private module boundaries without creating a shipped bypass.
Downloader failures retain only the pinned filename and exception class; the
provider exception cause is suppressed so a signed transfer URL cannot leak
through a programmatic traceback.

Acquisition evidence schema 2 distinguishes permission from observation. A
plan records `network_policy=disabled` and `network_used=false`. Execution
records `network_policy=permitted-cache-aware` and `network_used=null`, because
an immutable-revision Hub call may return a complete cached pointer without a
request; no uninstrumented inference is promoted to fact.

## Pinned local artifact manifest

Before a production parity run, create a hash-only identity record from the
same regular-file snapshot that both lanes will consume:

```bash
python3 tools/lfm2_vl/reference/inspect_artifact.py \
  --model 450m \
  --model-dir /path/to/regular-file-snapshot \
  --output /tmp/lfm2-vl-450m-artifact.json \
  --allow-production
```

This command is stdlib-only, local-only, and does not download, load, or copy
weights. It requires explicit production opt-in, rejects repository-local
model/output paths and symlinks, streams bounded hashes for the pinned config,
processor, tokenizer, license, template, generation, and safetensors files,
and records only repository, revision, filename, byte size, purpose, and
SHA-256 metadata. Use a disposable regular-file copy when a Hugging Face cache
exposes external symlinks. The native and Python lanes are not admitted until
their manifests identify the same files and hashes.

## Owner-managed setup and tiny fixture export

Use an isolated environment; installation is intentionally not performed by
the harness. Native Windows uses the last official Python 3.10 binary release,
3.10.11, and the proven `requirements-reference-windows.txt` lock. The fully
resolved
`requirements-reference.txt` is retained evidence for the Linux x86_64
Python 3.10.12 lane and must not be presented as a Windows lock.

Native Windows PowerShell:

```powershell
$ReferencePython = "C:\path\to\python-3.10.11\python.exe"
& $ReferencePython -m venv .venv
& .\.venv\Scripts\python.exe -m pip install `
  -r tools\lfm2_vl\reference\requirements-reference-windows.txt
& .\.venv\Scripts\python.exe `
  tools\lfm2_vl\reference\verify_environment.py `
  --require-tests --verify-lock
& .\.venv\Scripts\python.exe -m pytest tools\lfm2_vl\reference
```

Linux/WSL2:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r tools/lfm2_vl/reference/requirements-reference.txt
python tools/lfm2_vl/reference/verify_environment.py \
  --require-tests --verify-lock
python -m pytest tools/lfm2_vl/reference
```

After the pin guard is green, a tiny export can be generated outside the
committed fixture tree:

```bash
python tools/lfm2_vl/reference/export_fixtures.py \
  --mode tiny-random \
  --seed 1234 \
  --output artifacts/lfm2-vl/reference/tiny-1234
```

### Why these packages exist

This environment belongs to the independent Python oracle only. It is not a
runtime dependency of Candle, and the Rust binaries do not import or launch
Python. The pins keep the reference tensors reproducible and prevent a CUDA
wheel or a moving Transformers implementation from silently changing the
parity authority.

| Package | Role in the reference lane | Used by |
| --- | --- | --- |
| Python 3.10.11 on Windows / 3.10.12 on Linux | Exact per-platform interpreter contract. Python 3.10.12 has no official Windows binary, while the resolved Linux lock and committed tiny fixtures retain their proven 3.10.12 identity. | All oracle commands |
| `torch==2.8.0+cpu` | CPU-only F32 model execution, deterministic vision/projector/language tensors, and tensor comparisons. | Tiny and production traces; comparator |
| `torchvision==0.23.0+cpu` | CPU-only vision-stack compatibility pin for the Transformers image path; prevents an accidental CUDA companion wheel. | Transformers/vision imports |
| `transformers` at the locked Git commit | Numerical authority for LFM2, SigLIP2, LFM2-VL, processor behavior, and model loading. | Tiny fixtures and production oracle |
| `safetensors==0.8.0` | Writes/reads external tensor bundles and validates their inventories and hashes. | Fixture exporter, trace comparator, tests |
| `tokenizers==0.22.2` | Builds the tiny tokenizer and reproduces processor tokenization and image-token expansion. | Processor fixture and Transformers |
| `huggingface-hub==1.5.0` | Acquires exact pinned files and resolves local model/processor revisions; downloads remain opt-in. | Guarded acquisition and Transformers loading |
| `Pillow==11.3.0` | Decodes the deterministic source image and applies the oracle image-processor path. | Processor and production trace |
| `regex==2025.10.22` | Compatibility pin for the Transformers/tokenization text-processing stack. | Transformers/tokenizers |
| `pytest==8.4.1` | Runs the reference-tool regression suite; it is test-only and is excluded from the production runtime guard. | `test_*.py` |

The resolved lock also contains transitive packages (for example `numpy`,
`fsspec`, and `filelock`). They are installed because the pinned packages
declare them, not because Candle needs them. `config-only`, header inspection,
acquisition planning, and most bundle checks remain stdlib-only. Actual
acquisition needs only the pinned Hub package and its dependencies. Although
Hub declares `hf-xet` on supported machines, this workflow disables that
parallel backend before import. Only tiny/production oracle work and tensor
comparison require the heavier environment.

The exporter refuses an existing output directory unless `--overwrite` is supplied. Tiny output is deterministic for a fixed seed and package set; the manifest hashes both JSON metadata and safetensors. The synthetic raw RGB image is the exact byte source for `source_image_sha256`, and the packed patch tensor is derived from those same bytes. No access token, tokenizer text, model cache, or production weight is serialized.

The tiny and processor exporters enforce the same Python/package/VCS pins as production. On an unpinned machine, those official-fixture tests skip or the exporter fails before importing model code; config-only inspection and bundle/comparator validation remain available without the oracle runtime.

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
  --model-dir /path/to/regular-file-snapshot \
  --allow-production \
  --load-model \
  --output /tmp/lfm2-vl-production-model-check
```

The loader calls the pinned `Lfm2VlForConditionalGeneration.from_pretrained` path and never serializes the loaded tensors. When `--model-dir` is supplied, it hashes the external regular-file snapshot first and loads by that path with `local_files_only=True`; `--allow-download` is refused. Keep any user-fetched model artifacts outside Git and outside the tiny fixture directory.

### Production CPU-F32 trace

To export a bounded production component trace, use the owner-managed pinned CPU environment. The trace requires both production and model-load opt-in, a local image outside the repository, a non-empty prompt, and an output directory outside the repository:

```bash
python tools/lfm2_vl/reference/export_fixtures.py \
  --mode production \
  --model 450m \
  --model-dir /path/to/regular-file-snapshot \
  --allow-production \
  --load-model \
  --trace \
  --image /path/to/deterministic-image.png \
  --prompt "Describe this image." \
  --max-new-tokens 8 \
  --output /tmp/lfm2-vl-production-trace
```

The trace requires `--model-dir` so the Python oracle and native replay consume the same identified external regular-file snapshot. It hashes that snapshot into `artifact_manifest`, then loads it by path with `local_files_only=True`; `--allow-download` is refused. After inference it hashes the snapshot again and refuses to write evidence if any record changed. The Python `--prompt` is user text; the pinned processor applies the official multimodal chat template. Oracle metadata records the original text as `user_prompt` and the exact templated text as both `prompt` and `rendered_prompt`. Pass `metadata.json`'s `prompt` value—not the original user text—to native Candle so both lanes tokenize the same sentinel-bearing string. The trace forces CPU F32, one Torch thread, deterministic algorithms, a 4,096-token input bound, a 1,024-patch/64-crop bound, a 64 MiB/16-million-pixel source-image bound, and at most 32 decode steps. The first parity contract deliberately requires one non-tiled crop so native and oracle stage shapes are unambiguous; tiled or multi-crop inputs fail closed. It records processor tensors, vision hooks, projector input/output, merged embeddings, prefill logits, exactly aligned greedy cached-decode logits, input/image hashes, and bit-exact cache-reset evidence. It writes no model weights. Run the Python process through `scripts/lfm2-vl/run-bounded-oracle.ps1`; for native replay, build first and wrap only the resulting executable, not `cargo run`.

Once both external bundles exist, compare them without loading model weights:

```bash
python tools/lfm2_vl/reference/compare_traces.py \
  --oracle /path/to/lfm2-vl-production-trace \
  --native /path/to/lfm2-vl-native-trace \
  --output /tmp/lfm2-vl-trace-comparison.json
```

The comparator validates each bundle's trace schema/mode, matching metadata/manifest contract, no-weights claim, manifest, metadata hash, safetensors hash, canonical dtype names, shapes, and tensor names before loading one pair at a time. It requires both lanes to record successful post-inference input revalidation, then requires the native trace's consumed `config.json`, `processor_config.json`, `tokenizer.json`, safetensors index, and weight-shard evidence to match the oracle artifact manifest by direct filename, byte count, and SHA-256; missing, extra, duplicate, or content-mismatched native inputs fail before tensor comparison. Required input IDs, attention masks, processor tensors, projector input-patch ranges, and decode IDs are exact. The complete `stage.*` inventory must also match, so an optional configured stage such as projector LayerNorm cannot silently disappear; vision, projector, and language floats then use recorded CPU-F32 tolerances. Exit 0 means the report has `passed=true`, exit 1 means a valid comparison contains one or more failed tensors, and exit 2 means the bundles or invocation are invalid. A nonzero result is a failed gate, not a rounded-caption success claim.

Direct-GGUF production evidence uses the same native trace command with
`--model-file`, `--mmproj-file`, tokenizer, processor, image, prompt, and
`--trace-output`. A direct GGUF output is a separate `hybrid-trace` bundle;
split MMProj output remains ineligible for this evidence mode. Produce one
dense/dequantized bundle from the official F16 MMProj and one native-Q8 bundle
from the official Q8_0 MMProj, then compare them with:

```bash
python tools/lfm2_vl/reference/compare_traces.py \
  --dense /external/evidence/lfm2-vl-3b-f16 \
  --q8 /external/evidence/lfm2-vl-3b-q8 \
  --output /external/evidence/lfm2-vl-3b-dense-vs-q8.json
```

The hybrid comparator requires identical text/tokenizer/processor/prompt/image
identity, exact generated IDs and cache-reset replay, positive native Q8
retention, projector cosine `>=0.9999`, and language-logit max absolute drift
`<=2e-2`. These are comparison tolerances for the bounded Q8 lane, not a
production claim until both official artifact manifests and cleanup receipts
are present.

## Split dense MMProj export

`tools/export_lfm2_vl_mmproj.py` is a separate stdlib-only development tool. It accepts a local safetensors file plus local model and processor JSON, streams only the canonical `model.vision_tower.*` and `model.multi_modal_projector.*` payloads, and emits `mmproj.safetensors`, `mmproj.json`, and `processor_config.json`. It validates source offsets and byte sizes, refuses non-dense MMProj tensors, writes atomically, and never downloads a model.

The committed `tests/fixtures/lfm2_vl_mmproj_tiny/` bundle is derived byte-for-byte from the no-production-weight tiny fixture. `test_mmproj_exporter.py` proves deterministic regeneration, the exact 43-tensor namespace, hashes and version fields, existing/racing-output preservation, processor/model mismatch diagnostics, and controlled failure when the requested source namespace is absent.

## Bounded GGUF header inspection

`inspect_gguf_header.py` is a stdlib-only parser for a local bounded GGUF prefix. It validates magic, version, counts, string/array lengths, dimensions, dtypes, alignment, and header completeness, then reports both raw GGUF dimensions and Candle logical shapes. It never fetches a URL itself. For an already-local complete GGUF, `--full-file` memory-maps the file but bounds parser access to 4 MiB and hashes only the exact prefix through `tensor_data_offset`; the result separately reports physical file bytes and whether they match the declared tensor extent. JSON printed to a Windows console is ASCII-escaped; use `--output <path> --quiet` to retain the full UTF-8 report without duplicating a large tokenizer inventory into wrapper logs.

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

Omit `--summary-only` for the complete metadata and tensor inventory, or use `--output` to write JSON outside the repository. Add `--full-file` only for an already-local complete GGUF and pair a full report with `--quiet`. The result must report `contains_tensor_payload=false`; do not retain or commit production prefixes. Exact official hashes and parsed facts live in `reference-lock.json`, and the unit test checks their zero-payload boundary.

## Validation

`requirements-reference.in` is the shared direct CPU-lane intent.
`requirements-reference-windows.txt` is the fully resolved Python 3.10.11 /
Windows x86_64 lock, while `requirements-reference.txt` retains the fully
resolved Python 3.10.12 / Linux x86_64 lane. Native Windows selects Python
3.10.11 because Python.org did not publish Windows installers for 3.10.12. Run
`verify_environment.py --require-tests --verify-lock` to inspect exact pins,
the Transformers VCS revision, and the complete platform freeze without
importing Torch or the model. The production guard performs the same
runtime-only check before any official model or processor import; pytest is
required only when `--require-tests` is selected. Snapshot acquisition instead
checks only the pinned Hub distribution because it neither imports nor executes
the numerical oracle. `tensor_dump.validate_bundle()`
checks stable JSON objects, safetensors SHA-256, tensor names, shapes, dtypes,
and direct regular-file identity inside the bundle root; manifest paths cannot
escape through traversal, absolute names, nested paths, or symlinks.
`inspect_artifact.py` also rejects malformed lock entries, duplicate pinned
paths, NUL-containing names, invalid indexed tensor names, and shard path
traversal before hashing. The focused tests cover config-only behavior,
official tiny construction, deterministic regeneration, overwrite refusal,
production opt-in, site-packages-free acquisition planning, public/no-token
call arguments, Xet-enabled pre-import refusal, atomic rollback, mocked
production loading, snapshot/manifest publication races, stale manifest
staging, duplicate verifier output, hash failure, manifest path escape,
malformed manifest JSON, artifact identity ambiguity, exact oracle/native
artifact matching, and platform-specific environment selection.

---
AI-edited: 2026-08-21T12:40:00-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=ultra | task=lfm2-3b-q8-proof-gap | change=documented 3B locking, custom-code admission, and hybrid comparison commands
