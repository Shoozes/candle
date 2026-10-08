# D1 policy and session fixtures

These are synthetic inputs; no production weights are committed.

`tokenizer.json` is an authored 32-token WordLevel tokenizer with the tiny image
markers, d1 turn markers, answer aliases and rubric digits. `policy.json` was
exported from the hash-verified official d1 `prompt.py` at
`051bcc464b01b9f92942b364d9586b0ef5912432`, using tokenizers 0.22.2. It pins prompts,
ordered Unicode/float JSON, token groups/IDs, probabilities, empty/null criteria
and ties. The generator is `tools/lfm2_vl/d1/export_policy_goldens.py`.

`bicubic.json` stores byte-exact RGB downscaling results from Pillow 12.3.0. The
numerical contract is pinned to Pillow 11.3.0 `Resample.c` at
`89f1f4626a2aaf5f3d5ca6437f41def2998fbe09`: a=-0.5, 22-bit coefficients and byte
rounding after each separable pass. The byte results were replayed exactly under
the pinned Pillow 11.3.0 environment.

`torchvision-bicubic.json` is generated under pinned Torch 2.8.0+cpu and
TorchVision 0.23.0+cpu using the HF backend's `transforms.v2.functional.resize`
with native uint8 CPU input. It covers up/down/one-axis resizing, edges and
samples of the full study geometry. Its signed fixed-point coefficients and
byte intermediate differ from the v1 floating convenience path. Reproduce it
with `tools/lfm2_vl/d1/export_resize_goldens.py`; the compact JSON keeps this
deterministic numeric payload inside the Summary Bank's context budget.

Session tests author a deterministic block-aligned 32-wide Q8 text tower using
the existing two-layer conv/attention tiny GGUF metadata, and adapt the tiny
projector's final width to 32 in memory. The original 12-wide text fixture cannot
represent all major Q8 rows. No production derivative is created. These tests
prove cache/question isolation, actual counters, image reuse and partial errors
on CPU and CUDA. The ignored retained-image slice requires explicit input/output
environment paths and runs no model forward. The projector tests additionally
compare contiguous, permuted and offset native-Q8 inputs to F32 dense operands.
File hashes are in `manifest.json`; JSON and Markdown use LF checkout bytes.

---
AI-edited: 2026-10-08 | agent=Codex/root | model=unknown | effort=unknown | task=lfm2-d1 | change=documented deterministic policy and native-Q8 fixtures
