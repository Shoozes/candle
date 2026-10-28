# LFM3-Audio next bounded handoff

No Candle implementation remains in the selected finite CPU-F32 TTS/ASR
slice or its review/recheck/cleanup batch. Completed criteria and exact proof are in
[HISTORY](HISTORY.md) and [STATUS](STATUS.md). The paired Closeout/Publish
selects this reviewed source snapshot for guarded delivery to Candle `main`;
the delivery receipt in STATUS owns the exact remote result. Consumer adoption
requires separate selection and evidence.

## Proposed consumer step; not dispatched

- What: Wire an explicitly selected Candle LFM3 TTS adapter into SymbioVoice's
  existing voice-engine boundary, preserving current defaults and pins until
  separately admitted.
- Why: A real Candle model/WAV proof does not implement app selection,
  playback, cancellation, readiness or product error handling.
- When: The consumer owner selects the adapter and reviews a source/pin
  adoption plan. This handoff does not commit or repin any repository. Earlier
  native receipts predate the sampler/CLI cleanup and do not qualify its bytes.
- Where: SymbioVoice owns orchestration. Candle's reusable operations and
  finite example remain the reference integration surface.
- How: Admit the existing pinned native snapshot by hash, use finite generation
  and codec state, and preserve request-scoped cancellation/resource ownership.
  Start with one TTS request and validated PCM events, then actual playback/save
  and failure cleanup. Keep ASR input contracts separate from text synthesis.
- Done when: An opt-in consumer request produces verified 24 kHz PCM and
  playback/save, rejects missing/changed assets, respects stop/deadline limits,
  and releases all owned resources without changing default engines.
- Verification: A fresh local consumer receipt binds Candle source, runtime,
  asset hashes, request/stop behavior and output. Existing llama.cpp/ORT,
  Kitten/MOSS, Edge approval and D1 package acceptance stay independent.
  Any renewed model execution needs explicit selection and fresh resource
  admission; no prior allowance is reused by this TODO.

Other capabilities require separate selection: GPU/lower precision/GGUF,
full reference-model tensor parity, broader quality/language evaluation,
streaming audio input, live interleaving/tools and model distribution.
None reopens the completed D1 milestone or changes its frozen packages.

---
AI-edited: 2026-10-28 | agent=Codex/root | task=lfm3-audio-publish | change=save reviewed progress while retaining the separate consumer acceptance contract
