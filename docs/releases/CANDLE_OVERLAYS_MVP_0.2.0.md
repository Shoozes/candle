# Candle Overlays MVP 0.2.0 Release Contract

This document freezes the combined LFM2-VL and SnapFlash-derived Candle
snapshot. It is a candidate contract, not evidence that a tag or GitHub release
already exists. Publication must use the annotated tag
`candle-overlays-mvp-0.2.0`; the external release receipt records the exact
commit and tree without creating a self-referential follow-up commit. The
existing `lfm2-vl-mvp-0.1.0` tag remains immutable.

## Reproducible inputs

- Candidate parent: `dca9849584e377cebc1da40de966d050733f3bbf`.
- Upstream Candle base: `6f74e7c390c717f8fd34f23ce02aceb058173370`.
- Rust: `rustc 1.97.1 (8bab26f4f 2026-07-14)`.
- Cargo: `cargo 1.97.1 (c980f4866 2026-06-30)`.
- Root `Cargo.lock` SHA-256:
  `9b7aa15899ae8acf7b1a09b951ddba2f16462137eee2fed0db863a9d84707175`.
- Required workflow matrix: Ubuntu latest, Ubuntu 24.04 x64, Ubuntu 24.04
  ARM, Windows latest, and macOS latest checks; Ubuntu, Windows, and macOS
  tests; Rustfmt and warnings-denied Clippy.
- Local release authority: native Windows/MSVC locked/offline verification.
  The checked-in workflow is a pinned portability contract, not authorization
  to invoke or treat hosted runners as release evidence.

The release receipt must reject a candidate if the compiler, Cargo version,
lock hash, tag target, tree, or overlay inventories differ from these inputs
and the final local proof.

## Included overlays

- LFM2-VL/MMProj: config-driven dense and quantized LFM2, SigLIP2 NaFlex,
  native and GGUF MMProj, Rust preprocessing and prompt expansion, public
  hybrid assembly, deterministic fixtures, and bounded local proof tooling.
- SnapFlash-derived diffusion: three-component SDXL LoRA transactions,
  dev-only downstream rollback injection, exact additional residual admission,
  opt-in SDXL `text_time` conditioning, lower-precision cast order, and a
  controlled unsupported flash-attention error.
- Expected manifest gate after this closeout: 156 LFM2-VL paths
  (16 fork modifications and 140 additions), 20 SnapFlash-derived paths
  (8 fork modifications and 12 additions), and 167 union paths with 13 shared
  release or framework paths.

## Compatibility

SnapFlash `0.1.0` remains pinned to Candle
`1660f9fca8d6c8eb70937791e796203527f7be26`. This newer Candle snapshot does
not silently upgrade that proven runtime. Repinning requires a separate
SnapFlash lock update and rollback, no-model, LoRA, ControlNet, inpainting, and
installed-runtime requalification.

## Proven support boundary

The retained project evidence covers native Windows CPU-F32 official 450M and
1.6B component parity, direct Q8_0 MMProj execution, official GGUF decoded
output agreement, the complete advertised 450M CPU/CUDA placement and dtype
matrix, deterministic hybrid loader construction, and the focused diffusion
transaction/conditioning fixtures. The exact current candidate must still
pass the local release commands below before tagging.

No new model, API, queue type, training system, editor feature, or generic
framework abstraction is admitted to this release. Specifically excluded are
native Candle training, batch generation, multi-LoRA, prompt embedding cache,
additional ControlNet types, semantic face identity, generic VLM traits,
SnapFlash repinning, broad disabled-feature panic cleanup, and speculative
performance work.

## Local acceptance

Run with the pinned root toolchain and committed lock:

```powershell
$env:CARGO_NET_OFFLINE = 'true'
$env:CARGO_BUILD_JOBS = '2'
$env:PYO3_NO_PYTHON = '1'
cargo fmt --all -- --check
cargo check --locked --offline -j 2 --workspace
cargo clippy --locked --offline -j 2 --workspace --all-targets -- -D warnings
cargo test --locked --offline -j 2 --workspace --exclude candle-datasets --exclude candle-pyo3
```

The network-backed `candle-datasets` smoke is owner-scoped and must be reported
separately. `candle-pyo3` runtime tests require a matching Python 3.13 import
library; when it is absent, report that lane as an environment skip. Both
crates still participate in locked workspace check and warnings-denied Clippy.
Also run the focused LFM2-VL, LoRA rollback, overlay, summary-bank,
module-layout, formatting, and diff gates documented in `STATUS.md`.

After the owner-authorized clean candidate is published to `main` and the
annotated tag is published with the guarded local helper, write the external
identity receipt:

```powershell
pwsh -NoProfile -File .\scripts\release\write-candle-overlays-receipt.ps1 `
  -ExpectedHead <forty-character-candidate-commit> `
  -ExpectedTree <forty-character-candidate-tree> `
  -ReceiptPath <outside-repository-directory>\candle-overlays-mvp-0.2.0-receipt.json
```

The receipt command is read-only except for its explicit external output. It
requires a clean named `main`, explicit expected commit and tree, exact
local/remote branch and annotated-tag
identity, the unchanged historical tag, pinned Windows Rust/Cargo versions,
the lock hash, and the frozen 156/20/167 overlay inventories. It writes
atomically without overwrite and emits no local filesystem paths. Because it
reads public remote refs, do not run it before publication authority exists.

The hermetic no-network contract test exercises the complete success receipt,
path-free JSON, no-overwrite behavior, remote-main mismatch, and temporary-file
cleanup under both supported PowerShell parsers:

```powershell
pwsh -NoProfile -File .\scripts\release\test-write-candle-overlays-receipt.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\release\test-write-candle-overlays-receipt.ps1
```

## Publication boundary

Do not tag, push, create a GitHub release, change repository rules, or delete
the historical feature branch without explicit owner authorization. When
authorized, the guarded publisher must first prove local `main`, remote `main`,
the annotated tag target, and the release receipt all identify the same exact
candidate. Release protection must disallow force pushes, branch deletion, and
release-tag update or deletion while preserving the direct-main workflow.

---
AI-edited: 2026-08-13T20:08:58-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=ultra | task=repo-integrity | change=reconciled the exact offline workspace gate with the cleaned local Python environment
