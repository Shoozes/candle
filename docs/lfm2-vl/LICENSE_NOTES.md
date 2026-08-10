# LFM2.5-VL License Notes

## Scope

This is a provenance and implementation-boundary record, not legal advice. Exact source revisions and file URLs are in `SOURCES.md` and `tools/lfm2_vl/reference-lock.json`.

No external implementation code or production model artifact is included by the Source Lock Phase. The current repository changes are authored documentation and machine-readable references only.

## Repository License

Candle is distributed under the repository's existing Apache-2.0 or MIT choice. This extension must preserve that upstream licensing structure and must not silently introduce source or artifacts whose terms are incompatible with it.

## Source Register

| Source | Locked license | Project use | Direct adaptation boundary |
| --- | --- | --- | --- |
| Hugging Face Transformers | Apache-2.0 | Numerical oracle and fixture generator | Reference-only by project policy. If this changes, preserve notices, identify modified files, and record the exact source file and commit. |
| mistral.rs | MIT; copyright Eric Buehler | Primary Rust donor | Narrow model-math or processor ports are allowed only with explicit provenance and the applicable MIT notice. Do not copy its surrounding pipeline abstractions. |
| llama.cpp | MIT; copyright the ggml authors | GGUF conversion/naming and parity reference | Reference-only unless a future change explicitly identifies a narrow derived portion and retains the MIT notice. |
| MLX-VLM | MIT; copyright Prince Canuma | Secondary independent cross-check | Reference-only. |
| Transformers.js | Apache-2.0 | Secondary browser-runtime cross-check | Reference-only. |
| LiquidAI LFM2.5-VL model repositories | LFM Open License v1.0 | Official configs, processor/tokenizer inputs, optional user-fetched production weights | Do not vendor production model artifacts. Config and artifact use must comply with the pinned model license. |

## LiquidAI Artifact Boundary

The pinned license grants use, modification, and redistribution subject to its conditions. Its redistribution section requires the license, change notices for modified files, and retention of relevant notices. Its commercial-use grant is conditioned on a USD 10 million annual-revenue threshold, with the license text defining the applicable entity and exceptions.

Accordingly:

- This repository does not redistribute LiquidAI weights, GGUF files, tokenizer files, chat templates, or copied configs in Source Lock Phase.
- Production downloads remain explicit, local, ignored, and user-initiated.
- Tiny-random fixtures must be generated from deterministic synthetic weights, not extracted production tensors.
- A future distribution that includes LiquidAI artifacts needs a separate license-compliance review against the pinned `LICENSE` file.
- Runtime support must not imply that every downstream user is eligible to use a particular checkpoint commercially.

## Required Provenance for Future Ports

Any future file that directly adapts external implementation code must record:

1. source repository and immutable commit;
2. exact source path;
3. license;
4. what was adapted and materially changed;
5. the retained notice location; and
6. the parity test that proves behavior against the official Transformers oracle.

Mathematical facts, tensor dimensions, public API names, and independently implemented behavior should still cite `SOURCES.md`, but they are not to be mislabeled as copied source.

## Current Adaptation Inventory

None. No Candle Rust source changed in this phase.

---
AI-edited: 2026-08-09T22:56:01-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=source-lock | change=recorded licenses and future adaptation boundaries
