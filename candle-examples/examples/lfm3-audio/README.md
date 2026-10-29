# Native LFM3-Audio CPU example

This example executes the released model with Rust/Candle and explicit local
safetensors. TTS produces mono 24 kHz PCM16 WAV. ASR consumes 16 kHz PCM16 WAV;
multiple channels are mean-reduced and resampling is explicit caller work.
There are no downloads, Python calls, conversions of model weights, ORT or
llama.cpp execution. Existing consumer pins and engine defaults are unchanged.

Read [the current proof and limitations](../../../docs/lfm3-audio/STATUS.md)
before treating a source path as qualified. The example's resource/process
policy belongs to the existing native Job launcher; reusable model operations
are exposed from `candle_transformers::models::{lfm3, lfm3_audio, liquid_codec}`.

## Build and inspect

```powershell
$env:CARGO_BUILD_JOBS = '2'
$env:CARGO_INCREMENTAL = '0'
cargo build --locked --offline --release -p candle-examples --example lfm3-audio
& ./target/release/examples/lfm3-audio.exe --model-dir <verified-snapshot> --inspect
```

The snapshot must contain its original `config.json`, `model.safetensors`,
`tokenizer.json`, `audio_tokenizer/config.json` and
`audio_tokenizer/model.safetensors`, plus an explicit `asset-manifest.json`:

```json
{"repo":"LiquidAI/LFM3-Audio-2B","revision":"9d49701c372d6011d900f6abb40c6b296fb706dd","files":[{"path":"config.json","bytes":6243,"sha256":"1eb2a8349a1ec9eda2ed8e8bb1a8788007ae2ae9baa38a7d959743c5379d30d8"}]}
```

The abbreviated example shows the record shape; execution requires exactly
one correct identity for every consumed file. The existing acquired snapshot
already has the complete manifest. All consumed assets are checked before and
after inference. Output/report paths must be fresh. A report is completed only
after inference, input revalidation and output checks succeed.

## Finite execution

After fresh host/resource admission, launch the built executable through
`scripts/lfm2-vl/run-bounded-oracle.ps1`, with an explicit fresh evidence/log
path, an 8 GiB Job ceiling and 180-second timeout. Pass these child arguments
through `-ArgumentList`:

```text
TTS:
--model-dir <verified-snapshot> --text "Hello."
--language "American English" --voice female
--max-frames 100 --context 1024 --seed 42
--output <fresh.wav> --report <fresh.json> --verify-codec-chunks

ASR:
--model-dir <verified-snapshot> --input <16k-pcm16.wav>
--language English --max-new-tokens 128 --context 1024 --report <fresh.json>
```

TTS uses the official audio-processor ChatML/BOS envelope and task prompt,
prefilling `<|audio_start|>`. Every generated frame uses the configured MaskGIT
schedule; any EOS code ends the speech segment and that terminal frame is
excluded from synthesis. `frame_limit` is reported separately from natural
`audio_eos`. Rust `StdRng` replay does not promise PyTorch RNG compatibility.
The default audio repetition window is 4096 preceding frames per codebook.
The public sampler also applies its configured history window when called
directly; zero disables repetition history. Top-k filtering retains every
candidate tied at the cutoff; greedy decoding resolves ties by token index.

The codec chunk check uses seven-code-frame chunks and an empty final-tail
flush. It requires equal output length, terminal finalization, no retained tail
and maximum absolute F32 PCM error at most `1e-4`. This is a local codec replay
criterion; it changes no consumer/backend acceptance threshold.
`--verify-codec-chunks` requires TTS execution and conflicts with `--input`
and `--inspect`; neither mode silently skips a requested codec check.

ASR expands one input-audio placeholder to exactly the encoder token count,
replaces only those embeddings and performs greedy text decoding with speech
entry/placeholder tokens suppressed. The input ceiling is 160,000 samples
(10 seconds), with at most 512 generated tokens and 4096 context positions.
The frontend rejects wrong sample rates and nonfinite/out-of-range floats.
WAV admission also rejects missing chunk padding; valid padded metadata before
or after sample data is skipped without changing decoded PCM.

## Reusable operations and remaining scope

- `lfm3::Backbone`: F32 embedding forwarding with caller-bounded append-only
  caches, full/sliding attention, causal convolution and uncached depth mode.
- `lfm3_audio::Model`: text prefill, validated frame embeddings, depth logits,
  seeded MaskGIT frame sampling and audio-frame advancement.
- `lfm3_audio::features` / `encoder::Encoder`: released log-mel frontend and
  finite input encoder; `merge_audio_embeddings` enforces exact placeholder
  counts and preserves text positions.
- `liquid_codec::Decoder`: finite segment synthesis and transactional chunk
  synthesis with bounded attention/convolution state and joint audio/envelope
  overlap retention. Finalized caches refuse further chunks.

Current scope is single-sequence CPU F32 with the released artifact namespace
and default RoPE. GPU/lower precision, native GGUF loading, broader language or
quality qualification, streaming input frontend/encoder, interleaved live
conversation, tool policies, microphone/playback UI and product provisioning
are separate work. Apps own request scheduling, cancellation and resource
admission; reusable model operations do not start child processes.

---
AI-edited: 2026-10-29 | agent=Codex/root | task=lfm3-wav-review | change=document strict RIFF chunk padding admission and preserved PCM
