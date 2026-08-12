# candle-lfm2-vl

This example loads LFM2.5-VL from either an unmodified native Hugging Face checkpoint or a hybrid GGUF text model plus split/GGUF MMProj. It can stop after validated loading or run deterministic image-conditioned generation.

## Build and Help

Use the repository's local lockfile and start on CPU:

```bash
cargo run --locked --offline -p candle-examples --example lfm2-vl --release -- --help
```

The command does not download model files. Supply existing local paths whose repository revision, size, and SHA-256 have been recorded.

## Loading Modes

Native unified safetensors directory:

```bash
cargo run --locked --offline -p candle-examples --example lfm2-vl --release -- \
  --model-dir /path/to/LFM2.5-VL-450M \
  --cpu
```

Hybrid GGUF text plus split dense MMProj:

```bash
cargo run --locked --offline -p candle-examples --example lfm2-vl --release -- \
  --model-file /path/to/text.gguf \
  --mmproj-dir /path/to/split-mmproj \
  --tokenizer /path/to/tokenizer.json \
  --cpu
```

Hybrid GGUF text plus GGUF MMProj:

```bash
cargo run --locked --offline -p candle-examples --example lfm2-vl --release -- \
  --model-file /path/to/text.gguf \
  --mmproj-file /path/to/mmproj.gguf \
  --tokenizer /path/to/tokenizer.json \
  --dtype f32 \
  --mmproj-execution auto \
  --cpu
```

Without `--prompt`, all three forms validate, load, and report the resolved model/processor/device policy without generating tokens.

## Inference

Each `--image` requires exactly one literal `<image>` sentinel in the prompt:

```bash
cargo run --locked --offline -p candle-examples --example lfm2-vl --release -- \
  --model-dir /path/to/LFM2.5-VL-450M \
  --prompt 'Describe <image> precisely.' \
  --image /path/to/image.png \
  --max-new-tokens 32 \
  --vision-batch-size 1 \
  --json \
  --cpu
```

`--json` emits the deterministic `candle-lfm2-vl-inference-v1` report, including exact consumed-file evidence, processor/crop/token-span metadata, logits hashes/top-k entries, generated IDs, stop reason, and cache-reset replay result.

To keep vision on the selected accelerator while running the language model on
CPU, add `--text-cpu`:

```bash
cargo run --locked --offline -p candle-examples --example lfm2-vl --release -- \
  --model-dir /path/to/LFM2.5-VL-450M \
  --prompt 'Describe <image> precisely.' \
  --image /path/to/image.png \
  --text-cpu
```

For an official native component-parity lane, add `--trace-output <external-directory>` to a native `--cpu` run. The trace requires one image, explicit CPU/F32 execution, and at most 32 generated tokens; tiled or multi-crop inputs fail closed. It hashes every consumed model input before inference, hashes them again after deterministic replay, and refuses evidence output if they changed. The external directory contains `tensors.safetensors`, `metadata.json`, and `manifest.json` with exact input IDs/attention masks, processor, vision, projector, merge, prefill, and exactly aligned cached-decode tensors. Publication is atomic and no-clobber on Windows and Linux, so a destination that appears during the run is preserved. The trace never writes model weights or repository artifacts.

When replaying a Python oracle trace, pass the exact `prompt` value from the oracle's `metadata.json` to this native command. That value is the official rendered chat-template text containing `<image>`; the Python command's original user-text `--prompt` is not token-equivalent.

## Execution Policy

- CPU defaults to F32; CUDA defaults to BF16 unless `--dtype` is explicit.
- The four device arrangements are accelerator/accelerator (default), accelerator/CPU (`--vision-cpu`), CPU/accelerator (`--text-cpu`), and CPU/CPU (`--cpu`). `--cpu` is authoritative and keeps both components on CPU even when either component-specific flag is also present.
- CPU components are F32-only. Explicit BF16 or F16 on a resolved CPU component is rejected before model loading because Candle CPU matmul does not support those dtypes; use F32 for CPU text or vision, or keep that component on CUDA. The guard follows resolved devices, including accelerator-helper fallback to CPU.
- Official 450M evidence proves CPU/CPU F32, all-CUDA F32/BF16/F16, CPU-text/CUDA-vision F32, and CUDA-text/CPU-vision F32. Any resolved CPU component remains F32-only.
- `--timings` prints model-load, image-load, processor, prompt, vision, first-generation, cache-reset replay, and total inference durations to stderr. Timed device stages synchronize CUDA work and append `sync=cuda-device-complete`; evidence and JSON contracts are unchanged.
- `--benchmark-generation` is a separate quiet-host evidence lane. It requires at least two generated tokens, rejects `--timings` and `--trace-output`, performs 10 warm-ups plus 30 measured direct prefill/cached-decode iterations with device synchronization, requires exact generated-ID replay and at most 5% relative median absolute deviation, and writes one `candle-lfm2-vl-generation-benchmark-v1` JSON record to stderr without changing inference JSON.
- `--mmproj-execution auto` selects the supported path from the artifact. `dense` forces compatibility/dequantized execution.
- `--mmproj-execution q8` requires a GGUF MMProj file and F32 vision dtype. Native safetensors and split MMProj bundles are dense and reject this flag.
- `--max-new-tokens` is bounded to 0 through 1,024; `--vision-batch-size` is bounded to 1 through 64; prompt and image counts are bounded before model execution.
- Loader inputs must remain immutable from open through report emission. Missing, extra, malformed, shape-incompatible, or mismatched model/processor/tokenizer inputs fail with an error; there is no silent text-only fallback.

| Text device | Vision device | F32 | BF16 | F16 |
| --- | --- | --- | --- | --- |
| CPU | CPU | Proven | Rejected | Rejected |
| CPU | CUDA | Proven | Rejected | Rejected |
| CUDA | CPU | Proven | Rejected | Rejected |
| CUDA | CUDA | Proven | Proven | Proven |

Production evidence covers unmodified native 450M/1.6B CPU-F32 checkpoints,
the complete native 450M matrix above, and same-artifact GGUF/MMProj decoded
output. Deterministic tiny fixtures additionally protect split/direct MMProj,
native Q8_0, malformed inputs, prompt expansion, and cache behavior. Lower-bit
vision execution, video, true text batching, WebGPU, and a generic VLM layer
are unsupported/deferred rather than implicit release promises.

For production parity, follow `docs/lfm2-vl/START_HERE.md`: record a host/GPU/PID census, run the 450M CPU-F32 gate first, serialize large-model work, and verify cleanup before another run. Never invoke a llama.cpp oracle directly; use the bounded owner described in `docs/lfm2-vl/FAILURE_LOG.md` F-0008.

---
AI-edited: 2026-08-11T23:12:00-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=release-closeout | change=documented the final support matrix and benchmark contract
