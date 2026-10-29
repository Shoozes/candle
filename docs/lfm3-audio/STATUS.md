# LFM3-Audio implementation

The selected outcome is reusable native Rust/Candle LFM3-Audio inference,
starting with finite TTS, then audio input/ASR. Baseline is
`87e1304638344bb4644d35ab20284824fa826c77`. Source publication and runtime
qualification remain separate.
This is a separate model capability; the completed D1 milestone stays closed.

Reviewed implementation checkpoint: `a213f04f1b404c5b404172c84223ca1776136ec2`, the published
WAV-padding repair after `6a46737c`. Review/recheck matched all seven delivery
files to the passing publication receipt and found no new code regression in
the changed parser, fixture controls or adjacent admission path. The current
reconciliation changes records only; the implementation baseline above is historical.

The five-file documentation delivery has its own exact commit/tree, clean-commit
software gate and confirmed remote result in
`D:/Compact_Evidence/candle/runs/lfm3-handoff-docs-publish-20261029-a213f04f/publication.json`.
That receipt alone establishes its publication; it does not replace the
implementation checkpoint or admit any new model, package or consumer result.

The four-record portable-evidence follow-up has its own delivery identity,
clean-commit software gate and confirmed remote result in
`D:/Compact_Evidence/candle/runs/lfm3-evidence-docs-publish-20261029-7630e31b/publication.json`.
That receipt establishes documentation publication only. The earlier missing
consumer summary was a historical review observation; the later read-only
recheck below closes that part without renewing runtime qualification.

Publication of the catalog-only reconciliation is established only by its
guarded helper's exact commit/tree, software gate and confirmed remote result in
`D:/Compact_Evidence/candle/runs/lfm3-catalog-docs-publish-20261029-2e96f6e3/publication.json`.
That delivery preserves the preceding consumer observation and conditional
handoff without rechecking live consumer state or changing qualification.

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

The October 28 review/cleanup model-free gate passed 199 library tests, with six
ignored, and all four example tests. The four CPU libraries and new example checked
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
automatically selected by consumer work remaining. Read-only consumer source
inspection supersedes the old adapter proposal: SymbioVoice already implements
it and retains Candle `6a46737c`. Its owner records local TTS/app acceptance;
portable recipient qualification remains consumer-owned. During review of
documentation commit `7630e31b`, the consumer reported local portable CPU
qualification and retained bundle/native artifacts, but its documented top-level
summary and portable catalog entry were unavailable. That review did not admit
that claim.

Review of `2e96f6e3` now finds the canonical consumer summary at
`D:/Compact_Evidence/symbio-voice/runs/portable-cpu-20261029/summary.json`.
All 18 published-file hashes match consumer commit
`303dfa1073771069c13528aeea2b6171ce894ed7`; all 69 bundle build-input hashes
match its current source. ZIP, manifest, executable and six DLL hashes match
the retained receipt. A newer owner edit to its TODO is separate from that
published identity. These checks verify the binding of the recorded Windows
x64 CPU proof; no model, browser or recipient test was rerun. Manifest, lock and
runtime evidence retain Candle `6a46737c`. Catalog registration alone was pending
behind an existing `.index.lock`, which that review left untouched. TODO narrows the
handoff to consumer-owned catalog repair after lock ownership is resolved;
the consumer's own TODO remains authoritative for live status, without dispatch,
repin or another Candle implementation task. Clean OS VM, other Windows builds,
GPU, installer/public release and Edge/MOSS qualification remain separate.

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

The October 28 Closeout/Publish saved the reviewed source and records
on Candle `main` through `.tools/gitpush.ps1`. Its exact delivery commit/tree,
fresh clean-commit gate and confirmed remote result belong to
`D:/Compact_Evidence/candle/runs/lfm3-audio-publish-20261028-87e13046/publication.json`.
Only that receipt establishes publication. The gate includes model-free Python
module-layout and typed-receipt checks; no Python model runtime is introduced.

The October 29 review kept the prior sampling and codec-admission findings
closed, then reproduced a newly found WAV defect: an odd-sized terminal chunk
could omit its pad byte while satisfying the declared RIFF length. The parser
now rejects that truncated extent. Properly padded metadata before and after
the data chunk preserves exact PCM. All five example tests, four CPU-library
checks, the example check, strict example Clippy and formatting passed in this
pass. Current proof and documentation-verifier results are retained in
`D:/Compact_Evidence/candle/runs/lfm3-wav-review-20261029-6a46737c/summary.json`.
That review retained the prior 609-test publication gate without rerunning it,
and performed no model run or publication.

The separate WAV-repair Closeout/Publish used `.tools/gitpush.ps1` to save only
the seven reviewed files at `a213f04f`. Its clean-commit software gate, exact commit/tree
and confirmed remote result belong to
`D:/Compact_Evidence/candle/runs/lfm3-wav-publish-20261029-6a46737c/publication.json`.
That receipt confirms 609 Rust tests passed, 14 ignored, five audio example
tests and 197 release assertions on each PowerShell host. These are retained
results, not new tests run by this recheck. They do not renew native model
qualification or change consumer acceptance.

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

This documentation reconciliation changes no model/runtime behavior, package,
installation, activation or consumer pin/default. Native model qualification
and full portable-recipient acceptance retain their existing limits.
The old disk cleanup batch remains unapproved and untouched.

---
AI-edited: 2026-10-29 | agent=Codex/root | task=lfm3-catalog-docs-publish | change=bind documentation publication separately from consumer qualification
