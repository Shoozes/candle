# LFM3-Audio implementation

The shared LFM3 backbone has an independent CPU F32 ShortConv fixture.
PyTorch generated 18 outputs for biased, width-seven causal convolution.
Candle matched them across cached chunks of 4, 4, and 1 tokens within 1e-7.
An independent PyTorch F32 attention fixture also checks all 80 sliding
and full attention outputs across cached chunks within 2e-7. Standalone
text has a 95-logit PyTorch F32 fixture covering full prefill, irregular
chunks, single-token decode, and cache reset within 2e-6. Actual checkpoint
qualification on CPU and RTX 4090 remains open.

The offline `lfm3-text` example accepts explicit local config and safetensors
paths, input IDs and probe IDs. It reports ordered top IDs and probe logits
for teacher-forced execution. The diagnostic limits are 4,096 context
tokens, 128 tokens per chunk and 64 probe columns. For actual checkpoint
qualification, bind the files and binary to the run record and invoke the
example through the bounded host launcher.
The Windows CPU smoke test loaded a synthetic safetensors checkpoint.
Full versus chunked prefill matched all five top IDs and 15 probe logits
within 6.1e-8. An over-budget chunk was rejected.

The retained Windows CPU probe also loaded a synthetic BF16 safetensors
checkpoint and ran in F32. All 95 logits matched exactly against an F32
checkpoint made by widening the same BF16 values. Chunked BF16-input
execution retained all five top IDs and stayed within 6.1e-8 of one-shot
prefill. [Storage conversion receipt](D:/Compact_Evidence/candle/runs/lfm3-reference-e7200cd9/bf16-storage-smoke/summary.json).

The same probe parsed and validated the released standalone config after its
size and SHA-256 matched the pinned source record. An oversized chunk was
rejected before weight access. [Config preflight receipt](D:/Compact_Evidence/candle/runs/lfm3-reference-e7200cd9/standalone-config-smoke.json).

The published standalone Python implementation ran offline on the same tiny
BF16 fixture in F32. All five top IDs matched in both lanes. Across 95
logits per lane, maximum absolute error was `2.24e-7` for one-shot prefill
and `1.79e-7` for cached chunks of 2, 2 and 1. The actual standalone
checkpoint remains to be qualified on Windows CPU and RTX 4090.
[Reference replay](D:/Compact_Evidence/candle/runs/lfm3-reference-e7200cd9/tiny-replay-result.log).

The parameterized offline CPU reference probe reads BF16 safetensors one
tensor at a time into the F32 model. It takes a frozen case and emits one
JSON report compatible with the Candle comparator. With the tiny fixture
and cached chunks it matched five top IDs and 95 logits within `1.76e-7`.
The report was checked with `tools/lfm3/compare_probes.py`.
[Reference probe](D:/Compact_Evidence/candle/runs/lfm3-reference-e7200cd9/reference_probe.py).
[Comparison receipt](D:/Compact_Evidence/candle/runs/lfm3-reference-e7200cd9/reference-probe-smoke/comparison.json).

The Candle text probe releases its source tensor map after model
construction and device synchronization, before prefill. This shortens the
lifetime of the BF16 source tensors but retains the construction-time peak.
A fresh locked/offline Windows build matched the same five top IDs and all
95 synthetic logits within `1.76e-7` of the pinned CPU reference.
[Rebuilt probe comparison](D:/Compact_Evidence/candle/runs/lfm3-reference-e7200cd9/reference-probe-smoke/comparison-release-map.json).

Locked, offline native Windows CUDA check and build passed for `lfm3-text`.
The linker emitted `LNK4098` for `LIBCMT`. The RTX 4090 tiny BF16 fixture
matched all five top IDs and 95 F32 logits against the pinned CPU reference
within `2.21e-7`. No probe process remained. The tested binary SHA-256 was
`1215e8935dedefc71f95751835668bb9fca2b584731c0cdb93ef866be57194d5`.
[CUDA fixture comparison](D:/Compact_Evidence/candle/runs/lfm3-reference-e7200cd9/reference-probe-smoke/comparison-cuda.json).

The released standalone tokenizer was size- and SHA-256-verified. The inert
case inventory freezes a 33-token prompt and a 354-token prompt that crosses
the 256-token sliding window, with 11 probe columns per position. Both
candidate devices use the same IDs and probe columns, 16-token chunks and
a 384-token context. The proposed absolute logit tolerance is `0.001` and
needs review before execution. The tokenizer exposes 125,033 token IDs
within the checkpoint's 128,000 embedding rows. All four case files passed
the comparator's schema and token bounds checks.
[Frozen case inventory](D:/Compact_Evidence/candle/runs/lfm3-reference-e7200cd9/standalone-frozen-cases/inventory.json).

The inert standalone qualification draft binds the tested CUDA-enabled
probe, pinned reference sources, launch/preflight tools, comparator and four
case files. It specifies six serial model invocations and four comparisons
with one attempt and no retries. The exact checkpoint is now present on D:.
Its full SHA-256, size and safetensors header matched the pinned records.
The checkpoint binding includes the runtime inventory and the six verified
extractor commands. The review packet is sealed and inert. It requires the
exact owner grant, whole-checkpoint re-verification and fresh quiet/resource
admission before execution. Admission floors are 24 GiB available physical
memory and commit headroom, plus 14 GiB free dedicated memory for RTX 4090
stages. They give margin over the 9.66 GiB of simultaneous BF16 and F32
parameter storage during Candle load. Production peaks remain unmeasured.
Acceptance requires exact top IDs and a fixed `0.001` absolute error ceiling
on the probed logits. No production run occurred.
[Qualification draft](D:/Compact_Evidence/candle/runs/lfm3-reference-e7200cd9/standalone-production-draft.json).
[Checkpoint binding](D:/Compact_Evidence/candle/runs/lfm3-reference-e7200cd9/standalone-checkpoint-binding.json).
[Inert qualification packet](D:/Compact_Evidence/candle/runs/lfm3-reference-e7200cd9/standalone-production-packet.json)
SHA-256 `f243359f37bc3b58d0157386586897eaeeb3792f9a7c0a99b013af94d149eb06`.
The prior packet remains preserved. The [snapshot receipt](D:/Compact_Evidence/candle/runs/lfm3-standalone-clean-snapshot-20261115/summary.json)
identifies the active inert packet bound to the clean MoE/LFM3 snapshot.
The read-only verifier checked the full checkpoint, 24 draft bindings,
3,457 runtime files, extractor identity, canonical HEAD and absence of
the production output directory. It rejected an incorrect packet digest.
[Verifier](D:/Compact_Evidence/candle/runs/lfm3-reference-e7200cd9/verify_standalone_packet.py).
[Verification receipt](D:/Compact_Evidence/candle/runs/lfm3-reference-e7200cd9/standalone-packet-verification.json).

The packet controller defaults to read-only verification and left the
production output path absent in that mode. Its held execution path applies
the per-stage quiet and resource gates, aggregate deadline, bounded launcher,
GPU sampling, report extraction and stop-on-failure comparisons. The
controller's production execution path has not run.
[Controller](D:/Compact_Evidence/candle/runs/lfm3-reference-e7200cd9/invoke_standalone_packet.ps1)
SHA-256 `422d18b312f7cb42a006371913679672fcf8cdca77c155be63a57f1957371586`.
The same controller bytes passed a separate three-stage synthetic packet
on the reference CPU, Candle CPU and RTX 4090. It captured a GPU memory
sample, recorded a passing terminal receipt, and rejected reuse of the
consumed fixture packet. The production output path remained absent.
[Controller fixture receipt](D:/Compact_Evidence/candle/runs/lfm3-reference-e7200cd9/controller-tiny-smoke/results/terminal.json).

The reference runtime inventory binds the Python executable, the isolated
wheel files, the extracted Transformers source tree, and version and
`RECORD` identities for nine Python distributions. Base package `RECORD`
hashes identify their installed manifests. The inventory records individual
file hashes for the isolated wheels and Transformers tree.
[Runtime inventory](D:/Compact_Evidence/candle/runs/lfm3-reference-e7200cd9/standalone-reference-runtime.json).

The Windows Job launcher ran both CPU probes against the tiny BF16 fixture.
The extractor verified the combined-log hash, successful job exit and PID
cleanup before validating and writing each report. The comparator matched
five top IDs and 95 logits within `1.76e-7`. The extractor rejected duplicate
reports and a receipt without confirmed PID cleanup.
[Extractor](D:/Compact_Evidence/candle/runs/lfm3-reference-e7200cd9/extract_bounded_report.py).
[Bounded comparison](D:/Compact_Evidence/candle/runs/lfm3-reference-e7200cd9/bounded-report-smoke/comparison.json).

The same Windows Job launcher and extractor passed the RTX 4090 tiny
fixture with the packet-bound executable. The job exited and its PID was
absent after cleanup. All five top IDs matched the bounded CPU reference,
with a maximum absolute difference of `2.21e-7` across 95 logits.
[Bounded CUDA comparison](D:/Compact_Evidence/candle/runs/lfm3-reference-e7200cd9/bounded-cuda-smoke/comparison.json).

Use `tools/lfm3/compare_probes.py --case CASE --reference REF --candidate GOT`
for frozen report comparison. The case fixes `vocab_size`, `context`,
ordered `tokens` and `probes`, `max_abs`, and each side's `device` and
`chunk`. Each report must match the runner's F32 JSON fields. The
comparator requires exact top IDs, checks every selected logit, and reports
max absolute and RMS differences with the three input hashes.
The retained Transformers 5.12.1 tree cannot import the pinned LFM3 module
because it lacks `use_kernel_func_from_hub_with_fallback`.

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

The read-only recheck at clean `main` `ea026436` matched all four publication
file hashes, the tree and guarded-log digest in that receipt. Implementation,
tests, lock and Summary Bank remain unchanged from `a213f04f`; no new Candle
implementation defect was found. The 609-pass/14-ignored software gate and
197 release assertions on each PowerShell host are retained results, not new
runs in this recheck. At that review the consumer summary digest was unchanged;
its catalog entry was absent and its existing lock was retained. That observation
preceded the consumer-owned catalog closure recorded below.

The subsequent two-file documentation delivery is established only by the
guarded helper's exact commit/tree, clean-commit gate and confirmed remote result in
`D:/Compact_Evidence/candle/runs/lfm3-catalog-recheck-publish-20261029-ea026436/publication.json`.
It preserves the reviewed consumer observation without another live consumer
check, new supplier implementation or model/package/deployment operation.

Catalog closure is verified at Candle `969beab2`: the consumer catalog contains
exactly one relative `runs/portable-cpu-20261029` entry resolving to the retained
canonical receipt. Ownership was resolved from completed historical catalog
commands and an exclusive old-file handle; the task lock is released. The 12
prior entries and retention flags were preserved. Only receipt catalog metadata
changed; published consumer `303dfa10`, Candle `6a46737c`, qualification fields,
package bytes and owner TODO edits remain unchanged. [HISTORY](HISTORY.md)
records the exact receipt/catalog digests and ownership evidence.

No Candle implementation or catalog task remains in this audio slice.
[TODO](TODO.md) identifies the next consumer-owned read-only voice gap/source
matrix and model-free preflight plan, subject to its voice-specific owner return.
Existing Edge/MOSS qualification remains open. This reconciliation selects no
consumer execution or publication; prior native and software results remain
retained source-bound proof, not new runs.

The separately authorized five-file documentation publication is established
only by the guarded helper's exact commit/tree, clean-commit gate and confirmed
remote result in
`D:/Compact_Evidence/candle/runs/lfm3-catalog-closure-publish-20261029-969beab2/publication.json`.
It saves the closure and conditional planning handoff without changing frozen
inputs, consumer pins or any model/package/deployment state.

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

Review of `2e96f6e3` found the canonical consumer summary at
`D:/Compact_Evidence/symbio-voice/runs/portable-cpu-20261029/summary.json`.
All 18 published-file hashes match consumer commit
`303dfa1073771069c13528aeea2b6171ce894ed7`; all 69 bundle build-input hashes
match its current source. ZIP, manifest, executable and six DLL hashes match
the retained receipt. A newer owner edit to its TODO is separate from that
published identity. These checks verify the binding of the recorded Windows
x64 CPU proof; no model, browser or recipient test was rerun. Manifest, lock and
runtime evidence retain Candle `6a46737c`. Catalog registration alone was pending
behind an existing `.index.lock`, which that review left untouched. The later
consumer-owned registration closes that historical traceability gap. The
consumer's own TODO remains authoritative for live status, without automatic
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
AI-edited: 2026-10-29 | agent=Codex/root | task=lfm3-catalog-closure-publish | change=bind the documentation delivery separately from consumer acceptance
