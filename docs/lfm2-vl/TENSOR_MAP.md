# LFM2.5-VL Tensor Map

## Bootstrap State

This is an honest bootstrap placeholder. No external tensor inventory, name mapping, shape table, orientation transform, or GGUF metadata has been inspected or locked in this phase. The table below must not be treated as a source-lock claim.

## Required Mapping Work

| Mapping concern | Hugging Face native name | Candle target name | llama.cpp GGUF name | 450M shape | 1.6B shape | Orientation transform | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Text embedding | Pending source lock | Pending implementation | Pending GGUF inventory | Not recorded | Not recorded | Not recorded | Pending |
| LFM2 attention and convolution | Pending source lock | Pending implementation | Pending GGUF inventory | Not recorded | Not recorded | Not recorded | Pending |
| SigLIP2 patch embedding | Pending source lock | Pending implementation | Pending GGUF inventory | Not recorded | Not recorded | Not recorded | Pending |
| SigLIP2 position table | Pending source lock | Pending implementation | Pending GGUF inventory | Not recorded | Not recorded | Not recorded | Pending |
| Projector linear 1 and 2 | Pending source lock | Pending implementation | Pending GGUF inventory | Not recorded | Not recorded | Not recorded | Pending |
| Output head | Pending source lock | Pending implementation | Pending GGUF inventory | Not recorded | Not recorded | Not recorded | Pending |

## Rules for Population

The Source Lock Phase must populate this map from pinned files and real tensor inventories. It must record both checkpoint sizes, required reshape or transpose operations, tied-output behavior, and any GGUF layout reversal. Do not fill this table from model-name assumptions or unpinned moving branches.

---
AI-edited: 2026-08-09T22:35:40-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=docs | change=established bootstrap tensor mapping contract
