# LFM3-Audio completion history

## 2026-10-28: scoped source closeout and publication

The paired Closeout/Publish selects the 30-file reviewed native audio batch
for Candle `main`: reusable LFM3/audio/codec operations, finite TTS/ASR example,
the private LFM2 encoder-body reuse, existing RustFFT dependency edge, ownership
and Summary Bank routes, and the three reviewed sampler/CLI control repairs.
All candidate hashes matched the review receipt before documentation was updated
for this delivery. There were no staged owner edits or unrelated outgoing
commits; refreshed `origin/main` matched starting baseline `87e13046`.

The existing model-free controls and limited native TTS/ASR proof retain their
exact source identities. This publication introduces no new model execution,
release executable rebuild, model conversion, package build, installation,
activation, consumer repin or release asset. The clean-commit guarded gate adds
the repository's local software checks, including model-free Python layout
and receipt controls; the audio implementation remains native Rust/Candle.

Exact staged paths, candidate identities, command/results, resource disposition
and the guarded helper's confirmed remote result are retained in
`D:/Compact_Evidence/candle/runs/lfm3-audio-publish-20261028-87e13046/`.
Its `publication.json` alone establishes the final delivery commit/tree and
remote confirmation. Broader model parity and consumer usefulness remain
unqualified; source publication does not change their acceptance conditions.
The completed D1 milestone, frozen inputs, consumer pins and earlier immutable
proof remain preserved. The old disk-deletion batch is not part of this save.

## 2026-10-28: review, recheck and control cleanup

The read-only phases compared the 30-file local candidate at main baseline
`87e1304638344bb4644d35ab20284824fa826c77` with the previous source-bound
receipt. All hashes, the qualified executable and software archive matched.
Summary Bank's audio group was read directly, including implementation,
CLI consumers, tests and current records; no saved context pack was generated.
The prior model-frame history bound, dimension overflow guards and codec
finalization findings were verified closed. Full reference-model tensor parity
remains unqualified, as already documented; it was not selected by this pass.

Three new defects were repaired after reproducing them with failing tests:

- Top-k truncated to exactly k indices, excluding tied scores at its cutoff.
  Filtering now retains every cutoff tie, following the official warper
  contract recorded in [SOURCES](SOURCES.md). Greedy tie ordering and fixed-seed
  replay remain covered.
- The direct public sampler applied all supplied history despite its configured
  window. It now trims before validation/penalty; zero disables history, stale
  out-of-vocabulary entries outside the window are ignored, and active invalid
  entries still fail. The model's existing per-frame bound remains in place.
- ASR and inspection accepted `--verify-codec-chunks` without running that
  check. Clap now rejects both combinations before reading files or weights;
  ordinary TTS verification, ASR and inspection remain admitted.

Stale current-tense D1 publication instructions were made historical. The
completed D1 milestone, model inputs, locks, consumer pins, immutable evidence
and owner files were preserved. No cleanup deletion, foreign-process stop,
real-model run, package operation, installation, activation, commit or push ran.

Current native Windows/MSVC Rust 1.97.1 verification used locked/offline Cargo,
two jobs, incremental compilation disabled and the existing owned TEMP/TMP:

- Six focused audio tests passed after two new sampling tests failed before
  repair. The full transformer library suite passed 199 tests, six ignored.
- Four example tests passed after the new codec-admission test failed before
  repair. The tests cover actual clap parsing without model or filesystem use.
- Four CPU-library checks, the example check, strict Clippy and
  `cargo fmt --all -- --check` passed.
- Summary Bank, six-overlay ownership (349 paths, 35 shared), the unchanged
  LFM2 manifest (215 paths) and whitespace checks passed. The twelve prior
  isolated overlay controls were not rerun; their source remains unchanged.

Exact commands, red/green logs, final source identities and documentation
verification are retained in
`D:/Compact_Evidence/candle/runs/lfm3-review-cleanup-20261028-87e13046/summary.json`.
The previous six native Jobs and WAV/ASR results below remain bound to their
original candidate; this model-free cleanup does not renew runtime qualification.
TODO retains only the proposed consumer-owned adapter step, not another
automatic Candle implementation or model attempt.

## 2026-10-28: finite native CPU-F32 TTS and ASR

The selected work implemented the pinned released model directly with
Rust/Candle, from main baseline `87e1304638344bb4644d35ab20284824fa826c77`.
No commit or publication was requested. D1 and immutable consumer packages
remain closed/frozen.

Completed criteria:

- LFM3 width-seven gated convolution, full/sliding attention, per-head output
  gates, sandwich norms and zero-tail partial rotary frequencies.
- Bounded append-only language caches, reset/limit rejection, chunk/full and
  causal/bidirectional controls, with transactional failure handling.
- Shared speech input embedding table, independent tied depth heads,
  bidirectional codebook slots, configured three-pass MaskGIT sampling,
  bounded per-codebook repetition history and natural audio-EOS handling.
- LiquidCodec inverse FSQ, native decoder LFM, exact integer frame repetition,
  F32 inverse FFT and joint audio/envelope overlap-add, including bounded
  transactional streaming state and terminal empty-chunk finalization.
- Exact configured 16 kHz Hann/Slaney log-mel frontend, released three-stage
  downsampler, private feature-only LFM2 body with windowed causal attention,
  exact audio-placeholder embedding replacement and finite greedy ASR.
- Explicit local assets, before/after hashes, official TTS/ASR task templates,
  fresh output/report paths, finite CLI controls and source-bound receipts.

### Native results

Evidence: `D:/Compact_Evidence/candle/runs/lfm3-audio-20261028-87e13046/`.
The final executable SHA-256 is
`c16b9e83864fb0151964b389bd448ce4004171f496c425a3f48f3170f2143836`.
`runtime-source-closeout.json` binds its uncommitted source files and dependency
inputs. Job records retain exact child arguments, limits, times, hashes and
cleanup results.

| Proof | Observed result |
| --- | --- |
| First TTS | 66 frames, 126,720 mono 24 kHz samples, audio EOS, 26,275 ms Job, exit 0 |
| Seeded TTS replay | Identical codes and PCM16 WAV, 27,171 ms Job, exit 0 |
| Codec full/chunk replay | Equal 126,720 samples; max absolute F32 PCM delta `2.4400651e-6`, below preselected local `1e-4`; no final tail/reuse |
| Independent reference ASR | 76,800 input samples at 16 kHz; 479 mel frames/60 encoder tokens; text EOS, 13,972 ms Job, exit 0 |
| Candle round-trip ASR | 84,480 input samples at 16 kHz; 527 mel frames/66 encoder tokens; text EOS, 13,204 ms Job, exit 0 |
| Final guarded-binary TTS | Identical codes/WAV and codec chunk delta; audio EOS, 25,072 ms Job, exit 0 |
| Final guarded-binary ASR | Identical transcript/generated IDs; text EOS, 11,079 ms Job, exit 0 |

The TTS prompt was "Hello. This is SymbioVoice testing local audio without
Python." with American English/female, seed 42, Rust StdRng, audio temperature
1/top-k 128/repetition penalty 1.1 and a 4096-frame history window.
All model Jobs used four CPU threads, context 1024, at most 100 audio frames
or 128 ASR text tokens, an 8 GiB Job ceiling and 180-second wall bound.
All six exited normally and their PIDs were absent after cleanup. Peak Job
memory was at most 7,609,962,496 bytes; peak working set including read-only
model mappings reached 11,146,944,512 bytes. Healthy final physical and commit
headroom was 46,844,669,952 and 42,444,251,136 bytes respectively.

The independent reference transcript was "Hello, this is Simbeo voice testing
local audio without Python." Candle's own waveform transcribed as "Hello,
this is Symbio Voice testing local audio without Python." These are synthetic
short checks, not listening acceptance or independent speech-quality/accuracy
benchmarks. Different precision/seed/runtime outputs are not bit-exact targets.

The source-bound model files and tokenizer were downloaded once, with their
published hashes verified. Every consumed input was rehashed after inference.
Native FFmpeg explicitly resampled the two retained 24 kHz WAVs to 16 kHz;
`resample.json` binds its runtime and each input/output hash. No model conversion
or Python execution occurred. The operator's explicit Python-process override
is recorded; other admission checks remained enforced and both foreign PIDs
were untouched.

### Local software proof

All commands used native Windows/MSVC Rust 1.97.1, locked/offline Cargo,
two build jobs and `CARGO_INCREMENTAL=0`.

- `cargo test --locked --offline -p candle-transformers --lib`: 197 passed,
  six ignored. The initial broader run had 196 pass/one failure due to sandbox
  temp access in an unchanged GPT safetensor fixture. Routing TEMP/TMP to an
  owned workspace scratch directory passed the full suite without source repair.
- `cargo test --locked --offline -p candle-examples --example lfm3-audio`:
  three passed, including template and malformed/rate/extent/channel controls.
- Four-library locked/offline check and new-example check passed.
- Strict Clippy on the transformer library and new example passed.
- `cargo fmt --all -- --check`, explicit rustfmt checks for the two changed
  include files, and `git diff --check` passed.
- Summary Bank passed; the audio group remains below 256 KiB. The union overlay
  gate and twelve isolated-repository controls passed; the unchanged LFM2
  allowlist remained 215 paths (20 fork-origin/195 additions), with 19 text and
  six binary fixture attribute checks.

Independent controls compare partial RoPE against scalar frequencies,
Hann/Slaney energy against an analytic one-bin spectrum, inverse FFT against a
direct inverse DFT, FSQ against mixed-radix values, and cache/codec chunking
against full execution. Read-only independent source comparison found no
numerical or stream-finalization mismatch in the inspected graph.
Final review also added checked tensor-byte products and rejection controls
for malformed convolution/codec dimensions. The rebuilt binary repeated the
same native TTS/WAV and ASR IDs. Three actual CLI rejection controls refused
zero frame allowance, wrong-rate ASR input and an existing output path before
loading weights; no new output/report was created and the old WAV was unchanged.
No Python release/oracle gate, WSL lane, GPU run, consumer package test or
publication ran. See STATUS for the exact qualification boundary.

---
AI-edited: 2026-10-28 | agent=Codex/root | task=lfm3-audio-publish | change=record scoped source delivery with separate runtime and consumer proof boundaries
