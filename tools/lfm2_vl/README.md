# LFM2-VL Tools

This directory contains the reference and fixture tooling described by the LFM2.5-VL execution plan. The implementation lives under `reference/` and is intentionally separate from Candle runtime code.

Future tools must keep these boundaries:

- Configuration-only inspection must not import heavy packages or download weights.
- Snapshot acquisition is a separate external action: `acquire_snapshot.py --plan` is stdlib-only and read-only; `--allow-production-download` checks only the pinned Hub package, disables Xet before Hub import, downloads exact files serially through resumable HTTP without a token, verifies them, publishes snapshot and manifest without clobbering racing paths, and never loads a model.
- Production model metadata/loading requires `--allow-production`; any Hub access requires the separate `--allow-download` flag and loaded tensors are never serialized by this harness. The component trace is stricter: it requires an external regular-file `--model-dir`, hashes that snapshot first, loads locally, and refuses downloads.
- Tiny deterministic fixtures may be committed under `tests/fixtures/lfm2_vl_tiny/`.
- Local caches and reference outputs belong to ignored paths.
- Reference revisions, package versions, image hashes, dtype, device, and seed must be recorded with generated outputs.
- Official safetensors inventory digests use header-only HTTP Range reads and zero payload bytes. Canonical input is one UTF-8 line per tensor, sorted by name: `name<TAB>dtype<TAB>comma-separated-shape<LF>`.
- Standalone JSON and split-bundle outputs publish from flushed sibling temporaries and refuse existing or racing destinations by default; replacement requires the command's explicit `--overwrite` option.

Run the stdlib-only config path with `python3 tools/lfm2_vl/reference/inspect_config.py --model 450m`; add `--tokenizer <tokenizer.json>` to validate and record distinct in-vocabulary image wrapper, thumbnail, and row/column marker IDs without importing `tokenizers`. For a missing pinned snapshot, run `tools/lfm2_vl/reference/acquire_snapshot.py ... --plan`; the real download requires a separate `--allow-production-download` action and only the exact Hub package, not Torch. Before a production parity run, use `tools/lfm2_vl/reference/inspect_artifact.py --model <450m|1.6b> --model-dir <regular-file-snapshot> --output <external-manifest.json> --allow-production` to record the pinned repository/revision and exact local file hashes; it is local-only and never serializes weights. Both trace lanes rehash their inputs after inference. The comparator rejects a native bundle unless its consumed config, processor, tokenizer, index, and weight evidence matches that oracle manifest by filename, size, and SHA-256. The Python lane applies the official chat template; native parity must consume the resulting oracle `metadata.json.prompt`, not the untemplated user text. The exact CPU-lane setup, including Python 3.10.11 on native Windows, Python 3.10.12 in the resolved Linux lock, and the import-light environment verifier, is documented in `tools/lfm2_vl/reference/README.md`.

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

## Bounded Windows inference execution

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
  -ConcurrencyScope Name `
  -WorkingDirectory "C:\path\to\versioned-bundle" `
  -LogPath "C:\path\to\existing-evidence-dir\version.log" `
  -EvidencePath "C:\path\to\existing-evidence-dir\version.json"
```

The default ceiling is 24 GiB, the default timeout is 900 seconds, and any requested ceiling above 75% of physical RAM is rejected. The child is created suspended and assigned to a kill-on-close Job Object before resume. `-ConcurrencyScope Name` refuses any same-name process and is appropriate for unique model tools; `Executable` compares canonical executable paths and is required when a generic interpreter such as Python may have unrelated same-name workers. If a matching process path cannot be inspected, the wrapper fails closed. Timeout, memory pressure, wrapper-owner exit, and setup failure terminate the complete assigned process tree. `-LogPath` retains combined stdout/stderr and its hash so a failed child remains diagnosable. Evidence records the executable hash, concurrency policy, limits, timing, 64-bit peak process/job memory, log identity, exit/termination result, suspended-assignment facts, and exact PID cleanup.

The wrapper obtains total physical RAM from `Win32_ComputerSystem` when that
query is available and falls back to the Windows `GlobalMemoryStatusEx` API
when managed or restricted hosts deny CIM access. Evidence records the source
used as `physical_memory_source`; it never guesses a ceiling when both probes
fail. The harmless smoke suite passes under both Windows PowerShell 5.1 and
PowerShell 7; its test-only process-launch/cleanup compatibility paths avoid
APIs unavailable on the older .NET Framework.

Before any real model or llama.cpp run, take a read-only host census:

```powershell
pwsh -NoProfile -NonInteractive -File `
  scripts/lfm2-vl/preflight.ps1 -AsJson
```

The preflight reports Git identity (or an unusable linked-worktree error),
physical and committed-memory probes, repository-drive space plus its probe
source, optional NVIDIA memory/process data, and matching `llama`/Python/build-tool PIDs with path,
parent identity, start time, and memory fields. It deliberately omits command
lines and never inspects secrets. `review` means the physical and committed
memory probes are complete and a human still must approve the run; `blocked`
means a llama process or required physical/commit-memory probe is
present/incomplete. Use `-OutputPath
<existing-dir\\report.json>` for an atomic evidence file; replacement requires
`-ForceOutput`. The harmless cross-version contract test is:

```powershell
pwsh -NoProfile -NonInteractive -File `
  scripts/lfm2-vl/test-preflight.ps1
```

The general process list is capped at 64 records for compact evidence, but the
`llama_processes` collection is derived before that cap and therefore includes
every matching llama process. A low-memory model server cannot be hidden by a
large compiler fan-out.

Windows CUDA graphs are disabled by default with `GGML_CUDA_DISABLE_GRAPHS=1`; `-AllowCudaGraphs` is an explicit, recorded override. Use `-RedactArguments` whenever an argument could disclose sensitive data, and never place credentials in a command line. A green wrapper smoke test is containment proof only; it is not model or numerical parity evidence.

---
AI-edited: 2026-08-11T09:15:59-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=docs | change=clarified strict marker and no-clobber tool contracts
