# LFM3-Audio implementation

The selected outcome is reusable native Rust/Candle LFM3-Audio inference,
starting with finite TTS, then audio input/ASR. Baseline is
`87e1304638344bb4644d35ab20284824fa826c77`. Source publication and runtime
qualification remain separate.
This is a separate model capability; the completed D1 milestone stays closed.

Status: Complete with verified limitations for finite native CPU-F32 TTS/ASR.
The LFM3 backbone, released audio encoder, codebook embeddings, three-pass
MaskGIT depth decoder and LiquidCodec synthesis are implemented with Candle
operations. Existing LFM2 APIs/defaults and consumer revisions are preserved.

The pre-review native candidate produced 66 nonterminal code frames and
126,720 mono 24 kHz samples (5.28 seconds), ending at audio EOS. Its final replay
returned identical codes
and the same PCM16 WAV SHA-256:
`742f55e94551895119741b410a0fc82a34ebe68298aaf143bd889bb25221016e`.
Seven-frame codec chunks plus final overlap flush differed from full synthesis
by at most `2.4400651e-6` F32 PCM (local limit `1e-4`), with equal lengths and
terminal finalization. Peak amplitude was 0.6436264, with no saturated samples.

Native ASR ended at text EOS on both the independent native reference WAV and
Candle's own WAV. The Candle round trip transcribed:
"Hello, this is Symbio Voice testing local audio without Python."
This is finite synthetic-speech proof, not a speech-quality or ASR benchmark.

The review/cleanup model-free gate passed 199 library tests, with six ignored,
and all four example tests. The four CPU libraries and new example checked
locked/offline; strict Clippy, formatting, whitespace, Summary Bank, six-overlay
ownership and the unchanged LFM2 manifest passed. Twelve isolated overlay
controls were retained from the prior candidate; their verifier source is
unchanged and those controls were not rerun.
Independent analytic RoPE, Hann/Slaney, inverse-DFT/overlap and mixed-radix
controls supplement source inspection; no Python oracle ran.

Pinned official native assets are retained once under
`D:/Models/LiquidAI/LFM3-Audio-2B/9d49701c372d6011d900f6abb40c6b296fb706dd/`.
The asset manifest binds 4,169,809,496 bytes including weights, tokenizer,
configs, template and license. All three downloaded asset hashes matched the
official file inventory. All twelve released depth output tables were checked
byte-for-byte equal to their tied input embedding tables.

Exact commands, source/binary identities, input hashes, native Job records,
WAVs and results are retained under
`D:/Compact_Evidence/candle/runs/lfm3-audio-20261028-87e13046/`.
[HISTORY](HISTORY.md) records this completed slice; [TODO](TODO.md) names the
next bounded consumer handoff. [The example](../../candle-examples/examples/lfm3-audio/README.md)
documents usable APIs and commands. No further Candle implementation is
automatically selected by consumer work remaining.

Read-only review/recheck first matched all 30 candidate files, the native
executable and compact proof archive to that prior receipt. Earlier model-level
history-window, checked-dimension and stream-finalization findings were closed.
Cleanup then fixed three newly demonstrated control defects: top-k cutoff ties
remain eligible, direct sampler calls respect their repetition window, and
`--verify-codec-chunks` conflicts with ASR and inspection. Regression tests
failed before the fixes and passed afterwards. Historical publication wording
was corrected without reopening D1 or changing the consumer pins.

The current cleanup receipt is
`D:/Compact_Evidence/candle/runs/lfm3-review-cleanup-20261028-87e13046/summary.json`.
It binds the reviewed working-tree bytes and current model-free commands.
No real-model run or release executable rebuild ran in this cleanup pass.
Earlier native WAV/ASR results qualify their original candidate only; consumer
adoption needs a fresh receipt for the source it actually selects.

The operator-selected Closeout/Publish saves the reviewed source and records
on Candle `main` through `.tools/gitpush.ps1`. Its exact delivery commit/tree,
fresh clean-commit gate and confirmed remote result belong to
`D:/Compact_Evidence/candle/runs/lfm3-audio-publish-20261028-87e13046/publication.json`.
Only that receipt establishes publication. The gate includes model-free Python
module-layout and typed-receipt checks; no Python model runtime is introduced.

All six model Jobs used four CPU threads, context 1024, an 8 GiB Job memory
ceiling and a 180-second wall limit. Every child exited 0 and was absent after
cleanup. Peak Job memory was at most 7,609,962,496 bytes; peak working set was
11,146,944,512 bytes including read-only file mappings. The Job limit is not an
8 GiB resident-memory guarantee. Final physical/commit headroom was
46,844,669,952 / 42,444,251,136 bytes. The final rebuilt binary replayed the
same TTS codes/WAV and ASR token IDs after malformed-dimension guards were added.
The operator explicitly allowed existing foreign Python processes to remain
("continue, we have plenty of mem"); fresh admission enforced all other checks.
The two foreign processes were preserved; the native implementation and review
tasks executed no Python.

Verified limits: one sequence/clip, CPU F32, default RoPE and the released
safetensors namespace. GPU, lower precision, native GGUF, full reference-model
tensor parity, broad language/quality evaluation, live input frontend/encoder,
interleaved conversation, tools and product deployment are unqualified here.

No real-model execution, model conversion, package build/installation,
activation or consumer pin/default change is selected by this publication.
The old disk cleanup batch remains unapproved and untouched.

---
AI-edited: 2026-10-28 | agent=Codex/root | task=lfm3-audio-publish | change=bind scoped source delivery separately from native qualification and consumer adoption
