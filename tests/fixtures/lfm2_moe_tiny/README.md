# Synthetic LFM2-MoE fixtures

These seed-371 CPU/F32 weights contain no production checkpoint weights.
`manifest.json` records the pinned Transformers source, environment, source
hashes and payload hashes. The exporter uses the pinned eager attention and
expert implementation without changing that source. Native checkpoint tensors
are normalized to the official per-expert `w1/w2/w3` layout.

`native` covers four convolution/attention layers, dense-to-MoE transition,
biased and unbiased sigmoid routing, scaling, normalization and cached decode.
`quantized` uses 256-wide matrices for real Q4_K/Q6_K/Q8_0 block alignment. Rust
constructs GGUFs in memory from these weights and checks F32 oracle behavior,
quantized execution, cache recovery and malformed inputs. No duplicate GGUF
fixture payloads are retained.

`tokenizer_reference.json` records exact rendered bytes and IDs from the pinned
official tokenizer/template, including Unicode, old reasoning and tool-related
messages. Only these small derived cases are tracked, not the tokenizer asset.
The input control-token escaping is explicit Edge policy before the official
template is applied.

Regenerate with `tools/lfm2_moe/export_fixture.py` and
`export_tokenizer_fixture.py`, using the locked local reference tree described
in `docs/lfm2-moe/SOURCES.md`. JSON checkout bytes are fixed to LF.

---
AI-edited: 2026-11-10 | agent=Codex/root | task=lfm2-moe | change=document deterministic fixture provenance and scope
