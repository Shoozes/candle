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

The GGUF wire-format and converter behavior are pinned to `ggml-org/llama.cpp`
commit `f072b103714dfa1eee531f80b24512faf38e3dd2`, resolved on 2026-09-20,
under MIT. The relevant files are:

- [`conversion/gpt_oss.py`](https://github.com/ggml-org/llama.cpp/blob/f072b103714dfa1eee531f80b24512faf38e3dd2/conversion/gpt_oss.py)
  for MXFP4 repacking and fused-to-split expert tensor naming.
- [`gguf-py/gguf/constants.py`](https://github.com/ggml-org/llama.cpp/blob/f072b103714dfa1eee531f80b24512faf38e3dd2/gguf-py/gguf/constants.py)
  for GGML MXFP4 type 39 and its 17-byte/32-value block size.
- [`src/llama-arch.cpp`](https://github.com/ggml-org/llama.cpp/blob/f072b103714dfa1eee531f80b24512faf38e3dd2/src/llama-arch.cpp)
  for the `gpt-oss` architecture and canonical GGUF tensor names.

The Candle loader uses these files as format behavior references only. It
retains the GGUF MXFP4 bytes without applying the converter's nibble transform;
that execution concern is intentionally deferred to the later executor gate.

---

AI-edited: 2026-09-19; agent=Codex; task=gpt-oss-c0-c1; change=pinned official source identity
