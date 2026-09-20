# GPT-OSS Experimental Source Record

The implementation is a fresh Candle-native reference. The pinned behavioral
source is OpenAI's `gpt-oss` repository at commit
`7b583341fe16729127f6d5b94a7b09ccae97e1a1` (the `main` tip resolved on
2026-09-19), under Apache-2.0. The relevant files are:

- `gpt_oss/torch/weights.py` for the U8 `blocks`/`scales` representation,
  FP4 lookup, E8M0 exponent bias, and packed tensor shapes.
- `gpt_oss/torch/model.py` for GPT-OSS configuration defaults, rotary
  parameters, sink attention, SwiGLU, and selected-expert MoE ordering.

No source block is copied. This record is a behavior and format reference
only; production compatibility still requires a pinned checkpoint and an
independent numerical receipt.

The reference description states that each MXFP4 block contains 32 FP4 values
packed into 16 U8 bytes, with a matching scale along the final tensor
dimension. The Candle tests use deterministic synthetic payloads and do not
identify or load a production model.

---

AI-edited: 2026-09-19; agent=Codex; task=gpt-oss-c0-c1; change=pinned official source identity

