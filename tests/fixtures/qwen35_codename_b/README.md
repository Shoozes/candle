# Qwen3.5 first CPU reference

`reference.json` is byte-for-byte copied from EdgeSymbio
`37d475efa873841cd31751a29b3e29d7804197fa`, path
`tests/fixtures/candle-qwen35-codename-b-reference.v1.json` (SHA-256
`0c5315e7289e719a9500c94a0faaa4ef7d42c8a4429d1a702ca88429ead121a9`).
It is an artifact-specific black-box target, not a generic Qwen3.5 model
configuration or a full-vector/component numerical oracle. The external GGUF
is never stored here. See `docs/qwen35/STATUS.md` for its identity and gate.

`inspection.json` is the retained Edge tensor inventory, copied byte-for-byte
from `runtime-data/proofs/candle-qwen35-20260930/codename-b.inspection.json`
at the pinned checkout (SHA-256
`a9de899468afd3b407c3091680cf80a0403902f5e3388a435350330db9343d1c`).
The admission test reconstructs its 426-tensor directory without requiring the
external GGUF bytes.
