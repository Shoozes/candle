# Upstream integration — 2026-09-27

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
AI-edited: 2026-09-27 | agent=Codex | model=unknown | effort=unknown | task=upstream-sync | change=recorded pinned integration and retained compatibility
