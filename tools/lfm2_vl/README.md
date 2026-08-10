# LFM2-VL Tools

This directory contains the reference and fixture tooling described by the LFM2.5-VL execution plan. The implementation lives under `reference/` and is intentionally separate from Candle runtime code.

Future tools must keep these boundaries:

- Configuration-only inspection must not import heavy packages or download weights.
- Production model loading requires `--allow-production`; Hub downloads require the separate `--allow-download` flag and loaded tensors are never serialized by this harness.
- Tiny deterministic fixtures may be committed under `tests/fixtures/lfm2_vl_tiny/`.
- Local caches and reference outputs belong to ignored paths.
- Reference revisions, package versions, image hashes, dtype, device, and seed must be recorded with generated outputs.
- Official safetensors inventory digests use header-only HTTP Range reads and zero payload bytes. Canonical input is one UTF-8 line per tensor, sorted by name: `name<TAB>dtype<TAB>comma-separated-shape<LF>`.

Run the stdlib-only config path with `python3 tools/lfm2_vl/reference/inspect_config.py --model 450m`. The exact CPU-lane setup, official Transformers tiny-random oracle, production guard, and manager-resolution-pending requirements are documented in `tools/lfm2_vl/reference/README.md`.

## Deterministic runtime evidence

The `lfm2-vl` example remains load-only when `--prompt` is absent. A prompt enables the versioned `candle-lfm2-vl-inference-v1` runner:

```text
cargo run --locked -p candle-examples --example lfm2-vl --release -- \
  --model-dir <pinned-native-checkpoint> --cpu --dtype f32 \
  --prompt '<image>Describe the image.' --image <image> \
  --max-new-tokens 8 --json
```

For direct GGUF MMProj evidence, replace `--model-dir` with `--model-file`, `--mmproj-file`, and `--tokenizer`; add `--processor-config` when the artifact does not carry complete preprocessing policy. Split MMProj uses `--mmproj-dir`.

JSON mode emits one timing-free record with exact consumed-file paths/sizes/SHA-256 values, original and expanded prompt data, input IDs and image spans, processor/crop/packed shapes, EOS provenance, full-logit hashes, stable top-k values, generated IDs/tokens, decoded output, and exact two-run cache replay. Each image requires one literal `<image>` sentinel. Generation and vision limits are enforced before unbounded work.

Treat every model, tokenizer, processor, and MMProj input as immutable from loader open through report emission. Do not run `llama-mtmd-cli` directly for new parity work; `FAILURE_LOG.md` F-0008 requires the bounded owner wrapper and a verified child-process exit before another large oracle run.

## Bounded Windows llama.cpp execution

First prove the wrapper without loading a model:

```powershell
pwsh -NoProfile -NonInteractive -File `
  scripts/lfm2-vl/test-bounded-oracle.ps1
```

Then invoke an exact versioned executable through the wrapper:

```powershell
pwsh -NoProfile -NonInteractive -File `
  scripts/lfm2-vl/run-bounded-oracle.ps1 `
  -FilePath "C:\path\to\versioned-bundle\llama-mtmd-cli.exe" `
  -ArgumentList @("--version") `
  -WorkingDirectory "C:\path\to\versioned-bundle" `
  -EvidencePath "C:\path\to\existing-evidence-dir\version.json"
```

The default ceiling is 24 GiB, the default timeout is 900 seconds, and any requested ceiling above 75% of physical RAM is rejected. The child is created suspended, assigned to a kill-on-close Job Object before resume, and the wrapper refuses an already-running process with the same name. Timeout, memory pressure, wrapper-owner exit, and setup failure terminate the complete assigned process tree. Evidence records the executable hash, limits, timing, peak process/job memory, exit/termination result, suspended-assignment facts, and exact PID cleanup.

Windows CUDA graphs are disabled by default with `GGML_CUDA_DISABLE_GRAPHS=1`; `-AllowCudaGraphs` is an explicit, recorded override. Use `-RedactArguments` whenever an argument could disclose sensitive data, and never place credentials in a command line. A green wrapper smoke test is containment proof only; it is not model or numerical parity evidence.

---
AI-edited: 2026-08-10T15:34:55-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=bounded-llamacpp | change=documented the suspended Job Object wrapper, evidence, and CUDA-graph safety default
