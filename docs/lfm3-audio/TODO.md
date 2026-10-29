# LFM3-Audio next bounded handoff

No Candle implementation remains in the selected finite CPU-F32 TTS/ASR slice
or the October 29 WAV-padding repair. Completed criteria and proof are in
[HISTORY](HISTORY.md) and [STATUS](STATUS.md). The WAV repair was published at
`a213f04f1b404c5b404172c84223ca1776136ec2`; review/recheck found no new code
regression in that diff. The old proposal to create a SymbioVoice adapter is
superseded: its source already implements the opt-in adapter and retains Candle
`6a46737c`. Its owner records local acceptance at `075439d`; later owner edits
and portable delivery retain separate proof requirements. Do not repin it to
`a213f04f` automatically. The five-record documentation delivery has its separate
receipt in STATUS and changes no model/package/deployment state or consumer pin.

## Next bounded consumer handoff: SV-PKG-001 preparation; not dispatched

- What: Prepare a model-free portable layout and asset/runtime-manifest plan
  for the consumer's already selected Windows x64 CPU Kitten+LFM3 bundle.
- Why: The adapter exists, but development-checkout and local app acceptance do
  not prove a standalone executable/DLL/model delivery to a clean recipient.
- When/owner: SymbioVoice selects this preparatory pass under its `SV-PKG-001`.
  Candle edits no consumer files and dispatches no work through this handoff.
- How: Preserve Candle `6a46737c`, consumer defaults, admitted model inputs and
  owner edits. Identify the candidate after the consumer's admitted-root repair;
  describe package/config-relative roots and manifest/rejection rules using
  existing admission and lifecycle contracts. Compiled DLL closure and fresh
  recipient/model proof remain explicitly held, unverified gates.
- Done when: One reviewable plan binds the candidate and existing pins, names
  the proposed layout/manifest contract and model-free path/error fixtures, and
  lists the remaining build/native/recipient gates without claiming them passed.
- Verification: Consumer-owned source inspection and model-free manifest/path
  checks only. This preparation does not close the full `SV-PKG-001` Done When,
  authorize a package build or renew native/listening acceptance. Edge, MOSS,
  ASR/conversation and frozen D1 package acceptance remain independent.

Other capabilities require separate selection: GPU/lower precision/GGUF,
full reference-model tensor parity, broader quality/language evaluation,
streaming audio input, live interleaving/tools and model distribution.
None reopens the completed D1 milestone or changes its frozen packages.

---
AI-edited: 2026-10-29 | agent=Codex/root | task=lfm3-handoff-docs-publish | change=save reconciled bounded handoff without granting consumer execution or changing pins
