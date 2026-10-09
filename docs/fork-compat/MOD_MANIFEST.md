# Retained Upstream Compatibility Manifest

Live upstream baseline: `c68b24997319b8d76e5a4e775dab197c7f26322a`.
These intentional differences retain our public API and tokenizer contract
after upstream removed ug and selected fancy-regex. This overlay is independent
of LFM2-VL, diffusion, and experimental GPT-OSS model ownership.

Retained ug 0.5 APIs include the public candle-ug crate, core ug feature,
UgIOp1, CUDA/Metal device compilation, error conversion, and backend wiring.
Core and examples retain onig; candle-vlm remains owned by LFM2-VL and uses
the same workspace tokenizer version/backend. No Metal execution claim is made.

Verified native CUDA compatibility build on this host:
`$env:CUDARC_CUDA_VERSION='13000'` with installed CUDA 13.3 libraries, then
`cargo test --locked --offline -j 2 -p candle-core --features cuda,ug --test custom_op_tests --no-run`.
Run the resulting binary through the existing bounded oracle wrapper. Normal
non-ug builds do not need the API-version override. See UPSTREAM_SYNC for the
failed detection/link alternatives and bounded runtime receipts.

## Owned paths

- `candle-core/src/quantized/cpu_matmul_mode.rs`
- `candle-core/src/quantized/k_quants.rs`
- `candle-core/src/quantized/mod.rs`
- `candle-core/tests/cpu_quantized_matmul_tests.rs`
- `candle-core/tests/quantized_tests.rs`
- `docs/fork-compat/MOD_MANIFEST.md`
- `Cargo.toml`
- `candle-core/Cargo.toml`
- `candle-core/src/cuda_backend/device.rs`
- `candle-core/src/cuda_backend/mod.rs`
- `candle-core/src/custom_op.rs`
- `candle-core/src/error.rs`
- `candle-core/src/lib.rs`
- `candle-core/src/metal_backend/device.rs`
- `candle-core/tests/custom_op_tests.rs`
- `candle-examples/Cargo.toml`
- `candle-ug/Cargo.toml`
- `candle-ug/src/lib.rs`

Shared hunks: Cargo.toml retains ug registration; core custom-op tests retain
ug_op alongside GPT-OSS's I32 cast regression; examples retains onig alongside
the existing model examples. CUDA cast/kernel build additions retain their
existing feature-overlay owners. The CUDA backend uses div_ceil for the
upstream small-reduction launch count to pass the pinned strict Clippy gate
without overflow-prone addition. Other inherited upstream files are not owned here.

CPU-option hunks own scoped Auto/GenericTiled/GenericRowwise selection and
checked generic buffer/scratch boundaries. Quantized tests adapt upstream's
undersized-output panic to a controlled error and prove deterministic mode
parity. Qwen rounding and LFM2-VL native Q8/F32 hunks retain their owners.

---
AI-edited: 2026-10-09 | agent=Codex/root | model=unknown | effort=unknown | task=upstream-cpu-options | change=integrated pinned upstream and verified scoped CPU execution
