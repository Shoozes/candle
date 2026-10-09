# Upstream integration

## Current integration — 2026-10-09

Starting clean local/remote `main`: `f93a4111ae41683b548758225940f80de6cd37e8`.
Pinned official upstream: `c68b24997319b8d76e5a4e775dab197c7f26322a`.
Merge commit: `e445e88d8ff803c5889eedd895e7c689b2ad4115`. Both histories are
preserved. The two upstream commits restore normalization tensor lookup
(`5ba5d5b4`) and tile generic CPU quantized prefill (`c68b2499`). The
normalization implementation now matches upstream exactly; its former overlay
entry is replaced by the quantized-LFM2 example option entry.

### CPU execution options

`candle_core::quantized::{CpuQuantizedMatmulMode, with_cpu_quantized_matmul_mode}`
select synchronous execution on the calling thread. Nested scopes restore the
previous mode on success, error and unwind; child threads start with Auto.

| Mode | CPU behavior for F32/BF16 activations |
| --- | --- |
| `Auto` (default) | Existing strict native Q8 or optimized repacks first, then upstream generic 16-row tiling |
| `GenericTiled` | Bypass repacks and use generic 16-row tiles |
| `GenericRowwise` | Bypass repacks and use the legacy generic row scheduler |

```rust
use candle_core::quantized::{with_cpu_quantized_matmul_mode, CpuQuantizedMatmulMode};
let output = with_cpu_quantized_matmul_mode(CpuQuantizedMatmulMode::GenericTiled, || {
    model.forward(&input)
})?;
```

The `quantized-lfm2` example accepts
`--cpu-quantized-matmul auto|generic-tiled|generic-rowwise`. Non-default modes
require the selected device to be CPU and are checked before model/tokenizer
resolution. Existing model choices and defaults remain intact. Explicit
generic modes reject F16 activations and conflict with strict native CPU Q8;
Auto preserves d1's Q8 weight/F32 activation policy. Dense, CUDA and Metal
operations are unchanged. The selector adds no environment variable or dependency.

Before allocation or worker writes, generic execution checks dimensions,
block alignment, input lengths, output capacity and scratch sizes. Short
buffers return errors rather than upstream's panic. Empty output dimensions
perform no writes; zero-width multiplication fills only the required output
with zeros. Single-row scheduling and per-dot arithmetic are unchanged.

### Current verification and retention

Native Windows/MSVC Rust 1.97.1 verification passed:

- `cargo test --locked --offline -j 2 -p candle-nn --test layer_norm`: 4 passed.
- `cargo test --locked --offline -j 2 -p candle-core --test quantized_tests`:
  54 passed, 1 benchmark ignored; the final maintained gate reran this suite.
- `cargo test --locked --offline -j 2 -p candle-core --test cpu_quantized_matmul_tests`:
  4 passed, 1 benchmark ignored; 600 deterministic type/shape/activation
  combinations agree exactly across schedulers and single-row references.
- `cargo test --locked --offline -j 2 -p candle-core --lib quantized::cpu_matmul_mode`:
  1 passed, including nesting, error/unwind restoration and thread isolation.
- `cargo test --locked --offline -j 2 -p candle-examples --example quantized-lfm2`:
  1 passed, covering parsing and device validation.
- `pwsh -NoProfile -File .tools/verify-before-push.ps1`: 512 Rust tests passed,
  12 explicit ignores, 6 Python receipt-control tests passed. Formatting,
  locked/offline maintained libraries and four examples, required Clippy,
  Summary Bank, module layout, overlay union and whitespace checks passed.
- `cargo clippy --locked --offline -j 2 -p candle-core -p candle-nn -p candle-transformers -p candle-vlm --lib -p candle-examples --example quantized-lfm2 -- -D warnings`:
  passed. The same core/NN/transformers library Clippy gate with `--features cuda`
  passed. `cargo check --locked --offline -j 2 -p candle-core --features ug` passed.
- Core `--features cuda --lib --test cpu_quantized_matmul_tests --no-run`:
  built; bounded execution passed 5 CPU-option tests and 3 native Q8 tests,
  with 2 timing tests ignored. Ordinary and strict native CUDA outputs and
  counters remain exact under both explicit CPU modes.
- Core `--features cuda,ug --test custom_op_tests --no-run`, with process-local
  `CUDARC_CUDA_VERSION=13000` and installed CUDA 13.3 libraries: built; bounded
  execution passed all 7 tests, including ug and I32/BF16 CUDA casts.
- All three overlay-specific manifests passed; LFM2-VL remains 203 paths
  (20 modifications, 183 additions). Union: 276 paths, 5 overlays, 27 shared.
  `scripts/tests/test-verify-fork-overlays.sh` passed its 12 checks after repairing
  the pre-existing four-overlay fixture to include the registered Qwen overlay.
- Core tokenizer feature resolution retains only onig. Lock, workspace pins,
  toolchain and ug sources are unchanged against the starting published head.

Task evidence root: `artifacts/upstream-sync-20261009-cpu-options/`; retain
`native-gate.log`, `native-clippy.log`, `cuda-clippy.log`, `overlay-regression.log`,
`tokenizers-features.log` and the bounded `cuda-{options,native,ug}.json` receipts.
All bounded children exited zero and their PIDs were absent after Job cleanup.
The WSL `Codex-Compat` offline replay returned exit 101: its cache lacks
`accelerate-src`. No dependency was fetched and no WSL pass is claimed.
CUDA linking emitted the existing nonfatal MSVC LIBCMT conflict warning.
Metal and AArch64 execution are unrun on this host.

Implementation source: `34830c195df9143fe1c997d7682fc4e0cd5bdc93`.
Source/evidence binding: `proof.json`, SHA-256
`b4c7c1d46e5003c719d465f37a53dbe6196182fbcbe482fea1553e57cb6c8e3e`.
The final closeout source and successful clean-head gate are bound by
`publication-receipt.json` in the same root after guarded helper success.

### Frozen generic CPU component timing

Authored inputs, Q4K/Q8_0, `n=2040`, `k=2048`, 16 worker threads, three
warmups per mode, five alternating-order samples and ten calls per sample
were fixed in `benchmark-protocol.json` before timing. The release test
`benchmark_cpu_generic_modes` ran through the bounded Job helper with a
600-second/8-GiB limit. It passed and released its PID. Host: Ryzen 9 7950X.

| Weights | Rows | Tiled median ms | Row-wise median ms | Row-wise / tiled |
| --- | ---: | ---: | ---: | ---: |
| Q4K | 1 | 0.014450 | 0.014470 | 1.001 |
| Q4K | 32 | 0.348040 | 0.464230 | 1.334 |
| Q4K | 128 | 1.427650 | 1.807600 | 1.266 |
| Q8_0 | 1 | 0.025030 | 0.025280 | 1.010 |
| Q8_0 | 32 | 0.752260 | 0.906840 | 1.205 |
| Q8_0 | 128 | 2.971740 | 3.263330 | 1.098 |

These warm synthetic generic-kernel measurements do not measure optimized
repacks, d1's strict native CPU path, cold loading or real-model latency.
The benchmark performs 636 authored matrix operations and zero model forwards.
No performance pass threshold was invented. Exact samples and executable
identity are retained in `cpu-benchmark.log` and `cpu-benchmark.json`.

Production receipts retain their original source identities and are not
new model parity evidence. No new production inference is admitted in this batch.

## Historical integration — 2026-09-27

## Source identities and scope

- Starting clean local/remote main: `8f27ddfbee47957c274341fd6d32ccabd4767f9f`.
- Prior integrated upstream: `7c2e89295dad4aeebc6ef7a92c255360b6957c2c`.
- Pinned merge target: `aebc405d2b4bf42808387e0ca597bf7dad9b565f`.
- Integration uses a non-rewriting merge on canonical Windows main. The target
  must be an ancestor of the delivered commit; publication uses only the
  Candle-owned guarded helper.
- Historical 0.2.0 receipt base `6f74e7c390c717f8fd34f23ce02aceb058173370`,
  release tags, and model receipt identities remain immutable.

The 17 upstream commits bring CPU quantized repacking, broadcast batch and
noncontiguous indexing corrections, CUDA vectorization/reduction/copy fixes,
ONNX LogSoftmax default-axis correction, Mimi rotary correction, dependency
updates, and CI security/cache changes. Ordinary inherited upstream changes
are not fork-overlay paths.

## Deliberate fork differences

The fourth overlay, `docs/fork-compat/MOD_MANIFEST.md`, retains the public
candle-ug crate, ug feature, UgIOp1, error conversion, device compile methods,
and CUDA/Metal wiring removed upstream. Core/examples/candle-vlm retain onig,
with workspace tokenizers 0.23.1 resolving to 0.23.2. This avoids silently
changing the tokenizer contract for consumers.

The I32-to-F32 CUDA cast, MSVC conforming-preprocessor flags, GPT-OSS static
kernel, tracked lock, Rust 1.97.1 pin, and private-runner fork restriction remain.
Upstream action SHAs, per-job permissions, and caching are integrated while
retaining the pinned compiler and locked Cargo CI commands.

Cargo resolution changed 17 package entries: cudarc 0.19.10, cutile 0.3.1,
Parquet/Arrow 60 and their required compression dependencies. No broad update,
production dependency addition, or toolchain change was needed.

LFM2-VL/MMProj, SDXL LoRA/conditioning, and experimental GPT-OSS retain their
independent owners. GPT-OSS acceptance remains bound to source
`4a4699981ed55bb11e857c58923c67133559145d` and its observed-safe 2,080-token
receipt. This integration does not relabel that receipt or prove new model
parity, consumer acceptance, longer context, or production performance.

## Verification

Native PowerShell 7 / MSVC Rust 1.97.1 verification (all commands below passed):

- Starting baseline: `pwsh -NoProfile -File .tools/verify-before-push.ps1`.
- `cargo check --locked --offline -j 2 -p candle-core -p candle-nn -p candle-transformers -p candle-vlm`.
- `cargo test --locked --offline -j 2 -p candle-core -p candle-nn -p candle-transformers -p candle-vlm --features candle-transformers/test-utils`.
  Includes quantized repack/GEMV, stride-zero broadcast, noncontiguous indexing,
  LFM2/SigLIP2/projector/processor fixtures, SDXL conditioning and rollback.
- `cargo check --locked --offline -j 2 -p candle-core --features ug`.
- `cargo check --locked --offline -j 2 -p candle-examples --example lfm2 --example quantized-lfm2 --example lfm2-vl`.
- `cargo test --locked --offline -j 2 -p candle-examples --example lfm2-vl`:
  33 passed, including dense/direct-Q8 and native fixture cache replay.
- `cargo clippy --locked --offline -j 2 -p candle-core -p candle-nn -p candle-transformers -p candle-vlm --lib -- -D warnings`.
- `cargo clippy --locked --offline -j 2 -p candle-examples --example lfm2 --example quantized-lfm2 --example lfm2-vl -- -D warnings`.
- `cargo check --locked --offline -j 2 -p candle-datasets` verifies the updated
  Parquet/Arrow graph without the live-network dataset tests.
- `cargo clippy --locked --offline -j 2 -p candle-core -p candle-nn -p candle-transformers --features cuda --lib -- -D warnings`.
  Upstream's small-reduction launch division was changed to div_ceil to satisfy
  the strict pinned lint without overflow-prone addition; owned by compatibility.
- ONNX: `cargo test --locked --offline -j 2 --manifest-path candle-onnx/Cargo.toml --test ops test_logsoftmax`: 1 passed.
  PROTOC used cached protoc-bin-vendored-win32 3.2.0 (libprotoc 31.1).
  The excluded crate's generated lock is retained only in ignored task evidence.
- `cargo fmt --all -- --check`, `git diff --check HEAD`,
  `pwsh scripts/lfm2-vl/verify-summary-bank.ps1`,
  `python scripts/lfm2-vl/verify-module-layout.py`.
- Git for Windows Bash: `scripts/tests/test-verify-fork-overlays.sh`
  (12 checks); each of `scripts/{lfm2-vl,snapflash,gpt-oss}/verify-mod-manifest.sh`.
- Eight manifest-bound fixture SHA-256 checks passed; all fixture paths are
  byte-unchanged against the starting head. Exact prompt IDs passed in VLM tests.
  Targeted native tokenizer feature resolution has only onig; the full workspace
  also has existing WASM/dataset fancy-regex consumers.

CUDA binaries were built with `cargo test --locked --offline -j 2` and
`--no-run`, then run through `scripts/lfm2-vl/run-bounded-oracle.ps1`
with 180-second/8-GiB process limits and `--test-threads=1`:

- Core `--features cuda,ug --test custom_op_tests --test matmul_tests --test tensor_tests`:
  7 / 22 / 90 passed. Includes ug_op, I32-to-F32, new BF16 offset/tail
  casts, narrow-source scatter, strided operations and small reductions.
- Transformers `--features cuda --lib`, filter `models::gpt_oss::cuda::tests`:
  8 passed (synthetic parity, narrow views, cancellation and recovery).
- Receipts/logs: ignored `artifacts/upstream-sync-20260927/`. All wrapper
  receipts assert child exit 0 and PID absent after cleanup. No runtime retained.
- Nonfatal MSVC LIBCMT conflict warnings were emitted during CUDA test linking;
  test execution and strict CUDA library Clippy passed.

Publication requires clean-main replay by `.tools/gitpush.ps1 -Yes`.
The complete exact upstream delta is 210 paths across four registered overlays;
feature inventories remain 163 LFM2-VL, 20 diffusion, and 35 GPT-OSS. The helper's
default union gate now checks the pinned upstream exact delta, not the old rolling
checkpoint. Both histories must remain ancestors of the published delivery.

## Platform limits

- Native Windows/MSVC is authoritative. Retained ug's cudarc 0.17.8 rejects
  automatic CUDA 13.3 detection. CUDA 13.0 libraries also fail to link the newer
  cudarc's cublasLt emulation symbols. The verified compatibility build uses
  installed CUDA 13.3 libraries with process-local
  `$env:CUDARC_CUDA_VERSION='13000'` (CUDA 13.0 API selection) for cuda,ug.
  Maintained non-ug CUDA builds use normal CUDA 13.3 detection. No global
  environment setting or dependency pin was changed.
- Codex-Compat WSL offline replay could not resolve accelerate-src from its
  empty registry cache. Rustup auto-installed pinned Rust 1.97.1 on the first
  invocation; no further WSL provisioning is authorized by this sync.
- No production model downloads/runs, consumer repinning, release tags, hosted
  verification, Metal execution, or AArch64 execution are part of this task.

---
AI-edited: 2026-10-09 | agent=Codex/root | model=unknown | effort=unknown | task=upstream-cpu-options | change=integrated pinned upstream and verified scoped CPU execution
