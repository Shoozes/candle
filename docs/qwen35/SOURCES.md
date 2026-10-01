# Qwen3.5 Sources

| Source | Revision and file | Use | License |
| --- | --- | --- | --- |
| llama.cpp / ggml | `b49650adb31f2e49a0d76113aeb1792134fd8413`, `src/models/qwen35.cpp`, `src/models/delta-net-base.cpp`, `conversion/qwen.py`, `ggml/src/ggml-cpu/vec.h`, `ggml/src/ggml-cpu/vec.cpp`, `ggml/src/ggml-cpu/simd-mappings.h`, `ggml/src/ggml-cpu/ops.cpp` | Pinned graph, recurrent and attention arithmetic, GGUF conversion layout, and adapted Zen4 F32 exponential arithmetic | MIT; copyright (c) 2023-2026 The ggml authors |
| EdgeSymbio | `37d475efa873841cd31751a29b3e29d7804197fa`, `docs/dev/handoffs/2026-09-30-candle-qwen35-compatibility.md`, `tests/fixtures/candle-qwen35-codename-b-reference.v1.json` | Owner-provided artifact identity, resource envelope, and first CPU black-box trace | Internal fixture and handoff |
| Local retained Edge evidence | `runtime-data/proofs/candle-qwen35-20260930` at the pinned Edge checkout | Independently retained GGUF tensor inventory and reference execution receipts; see `STATUS.md` for hashes | Internal evidence |
| Published GGUF replacement | `mradermacher/Darkidol-Ballad-4B-GGUF` at immutable revision `0fc1207385c1655f27e2bb93c72115ee8ac12369`, file `Darkidol-Ballad-4B.Q8_0.gguf`, 4,482,403,200 bytes, LFS SHA-256 `fd6dcc1ea357b483ae2517b482bf328181e5b371b65661c734aa133407ae2572` | Owner-approved local reacquisition of the exact bytes for CPU parity | Apache-2.0 metadata |

The replacement file's published revision and LFS identity are verified. The
originally deleted local file's download history is unknown; identical SHA-256
proves byte identity, not its original acquisition provenance. The scalar
exponential in `cpu.rs` adapts the pinned ggml Zen4 polynomial and operation
order. All other Candle code is freshly written.

The adapted ggml arithmetic is covered by this MIT notice: Copyright (c)
2023-2026 The ggml authors. Permission is hereby granted, free of charge, to
any person obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following
conditions: The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software. THE SOFTWARE
IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR
A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
