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
- Explicit BF16 on either CPU component is rejected before model loading because Candle CPU matmul does not support it; use F32 for CPU text or vision, or keep that component on CUDA.
- `--timings` prints model-load, image-load, processor, prompt, vision, first-generation, cache-reset replay, and total inference durations to stderr; it does not alter JSON evidence.
- `--mmproj-execution auto` selects the supported path from the artifact. `dense` forces compatibility/dequantized execution.
- `--mmproj-execution q8` requires a GGUF MMProj file and F32 vision dtype. Native safetensors and split MMProj bundles are dense and reject this flag.
- `--max-new-tokens` is bounded to 0 through 1,024; `--vision-batch-size` is bounded to 1 through 64; prompt and image counts are bounded before model execution.
- Loader inputs must remain immutable from open through report emission. Missing, extra, malformed, shape-incompatible, or mismatched model/processor/tokenizer inputs fail with an error; there is no silent text-only fallback.

For production parity, follow `docs/lfm2-vl/START_HERE.md`: record a host/GPU/PID census, run the 450M CPU-F32 gate first, serialize large-model work, and verify cleanup before another run. Never invoke a llama.cpp oracle directly; use the bounded owner described in `docs/lfm2-vl/FAILURE_LOG.md` F-0008.

---
AI-edited: 2026-08-11T09:15:59-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=docs | change=documented cross-platform no-clobber native trace publication
