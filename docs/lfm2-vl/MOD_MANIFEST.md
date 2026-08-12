# LFM2-VL Mod File Manifest

This manifest separates the LFM2-VL mod overlay from the integrated Candle fork. It is the publication allowlist authority; files not represented here must not be staged merely because they exist in the worktree.

## Classification Rule

- Model and compatibility baseline: Candle 0.11.0 at `31f35b147389700ed2a178ee66a91c3cc25cc80d`.
- Current publication baseline: Candle main at `6f74e7c390c717f8fd34f23ce02aceb058173370`, the exact `origin/main` tip integrated before this direct-main release.
- Historical mod checkpoint: `c9b60f0b906fa8fe70423295e2e1164648a8fa53` on `feat/lfm2-vl-mmproj`; that branch is retained as evidence, not used as a second publication line.
- Current `main` overlay: 141 paths, exactly 14 fork-origin modifications and 127 mod-owned additions. The 29 upstream paths added or changed between Candle 0.11.0 and the publication baseline are inherited fork state and are intentionally outside this overlay.
- A **fork-origin modification** is a path that exists in the current publication baseline and is intentionally changed by this mod.
- A **mod-owned addition** is a path absent from the current publication baseline and created for this project.
- “Mod-owned” describes repository provenance, not third-party authorship. External source and license provenance remains authoritative in `SOURCES.md` and `LICENSE_NOTES.md`.
- Every other tracked path is inherited unchanged from the Candle fork.

## Fork-Origin Files Intentionally Modified

Exactly these fourteen baseline files contain mod changes:

| Path | LFM2-VL reason |
| --- | --- |
| `.github/workflows/ci_cuda.yaml` | Limit the inherited private AWS CUDA runner job to the upstream repository so fork pull requests skip instead of failing before checkout. |
| `.gitignore` | Exclude local reference environments, caches, downloads, production models, and generated artifacts. |
| `README.md` | Add the discoverable LFM2.5-VL example and support-boundary entry for this fork. |
| `Cargo.toml` | Register the new `candle-vlm` workspace crate and its shared dependencies. |
| `candle-core/src/quantized/gguf_file.rs` | Add bounded GGUF directory parsing and validation used by direct MMProj loading. |
| `candle-examples/Cargo.toml` | Wire the new LFM2-VL example to its runtime and fixture dependencies. |
| `candle-examples/examples/lfm2/main.rs` | Preserve and prove current LFM2.5 text configuration compatibility. |
| `candle-core/tests/custom_op_tests.rs` | Prove the CUDA I32-to-F32 cast used by packed vision masks. |
| `candle-kernels/build.rs` | Pass MSVC's conforming-preprocessor switch through nvcc for CUDA 13.3/CCCL builds. |
| `candle-kernels/src/cast.cu` | Add the CUDA I32-to-F32 cast required by packed vision mask validation. |
| `candle-transformers/Cargo.toml` | Add dependencies required by LFM2-VL loading and verification. |
| `candle-transformers/src/models/lfm2.rs` | Normalize LFM2.5 configuration and add dense embedding-driven forwarding/cache support. |
| `candle-transformers/src/models/mod.rs` | Register the new SigLIP2 and LFM2-VL model modules. |
| `candle-transformers/src/models/quantized_lfm2.rs` | Add validated GGUF metadata, tied-output handling, embedding-driven forwarding, and cache support for hybrid execution. |

No other file from the integrated Candle publication baseline is part of the mod delta.

## Mod-Owned Runtime and Example Additions

### LFM2-VL example

- `candle-examples/examples/lfm2-vl/README.md`
- `candle-examples/examples/lfm2-vl/args.rs`
- `candle-examples/examples/lfm2-vl/loading.rs`
- `candle-examples/examples/lfm2-vl/main.rs`
- `candle-examples/examples/lfm2-vl/native_checkpoint.rs`
- `candle-examples/examples/lfm2-vl/native_loading.rs`
- `candle-examples/examples/lfm2-vl/native_loading/types.rs`
- `candle-examples/examples/lfm2-vl/native_loading/load.rs`
- `candle-examples/examples/lfm2-vl/native_loading/inventory.rs`
- `candle-examples/examples/lfm2-vl/runner.rs`
- `candle-examples/examples/lfm2-vl/runner/types.rs`
- `candle-examples/examples/lfm2-vl/runner/runtime.rs`
- `candle-examples/examples/lfm2-vl/runner/run.rs`
- `candle-examples/examples/lfm2-vl/runner/generation.rs`
- `candle-examples/examples/lfm2-vl/runner/benchmark.rs`
- `candle-examples/examples/lfm2-vl/runner/evidence.rs`
- `candle-examples/examples/lfm2-vl/trace.rs`

### Transformer models and loaders

- `candle-transformers/src/models/lfm2/config.rs`
- `candle-transformers/src/models/lfm2/cache.rs`
- `candle-transformers/src/models/lfm2/layers.rs`
- `candle-transformers/src/models/lfm2/model.rs`
- `candle-transformers/src/models/siglip2.rs`
- `candle-transformers/src/models/siglip2/config.rs`
- `candle-transformers/src/models/siglip2/embeddings.rs`
- `candle-transformers/src/models/siglip2/encoder.rs`
- `candle-transformers/src/models/siglip2/model.rs`
- `candle-transformers/src/models/siglip2/interpolation.rs`
- `candle-transformers/src/models/lfm2_vl/config.rs`
- `candle-transformers/src/models/lfm2_vl/gguf.rs`
- `candle-transformers/src/models/lfm2_vl/gguf/types.rs`
- `candle-transformers/src/models/lfm2_vl/gguf/loading.rs`
- `candle-transformers/src/models/lfm2_vl/gguf/metadata.rs`
- `candle-transformers/src/models/lfm2_vl/gguf/inventory.rs`
- `candle-transformers/src/models/lfm2_vl/gguf/metadata_values.rs`
- `candle-transformers/src/models/lfm2_vl/linear.rs`
- `candle-transformers/src/models/lfm2_vl/mod.rs`
- `candle-transformers/src/models/lfm2_vl/model.rs`
- `candle-transformers/src/models/lfm2_vl/model/types.rs`
- `candle-transformers/src/models/lfm2_vl/model/runtime.rs`
- `candle-transformers/src/models/lfm2_vl/model/encoding.rs`
- `candle-transformers/src/models/lfm2_vl/model/merge.rs`
- `candle-transformers/src/models/lfm2_vl/model/config_ext.rs`
- `candle-transformers/src/models/lfm2_vl/projector.rs`
- `candle-transformers/src/models/lfm2_vl/weights.rs`
- `candle-transformers/src/models/lfm2_vl/weights/manifest.rs`
- `candle-transformers/src/models/lfm2_vl/weights/runtime.rs`
- `candle-transformers/src/models/lfm2_vl/weights/safetensors.rs`

### Rust-native vision-language processing crate

- `candle-vlm/Cargo.toml`
- `candle-vlm/README.md`
- `candle-vlm/src/lib.rs`
- `candle-vlm/src/image.rs`
- `candle-vlm/src/lfm2_vl/config.rs`
- `candle-vlm/src/lfm2_vl/mod.rs`
- `candle-vlm/src/lfm2_vl/processor.rs`
- `candle-vlm/src/lfm2_vl/processor/types.rs`
- `candle-vlm/src/lfm2_vl/processor/entry.rs`
- `candle-vlm/src/lfm2_vl/processor/budget.rs`
- `candle-vlm/src/lfm2_vl/processor/crops.rs`
- `candle-vlm/src/lfm2_vl/processor/helpers.rs`
- `candle-vlm/src/lfm2_vl/prompt.rs`
- `candle-vlm/src/lfm2_vl/prompt/types.rs`
- `candle-vlm/src/lfm2_vl/prompt/tokens.rs`
- `candle-vlm/src/lfm2_vl/prompt/expand.rs`
- `candle-vlm/src/lfm2_vl/prompt/validation.rs`
- `candle-vlm/src/lfm2_vl/prompt/image_block.rs`
- `candle-vlm/src/lfm2_vl/prompt/helpers.rs`
- `candle-vlm/src/lfm2_vl/types.rs`

## Mod-Owned Project Control and Evidence

### Repository control and design documents
- `AGENTS.md`
- `docs/lfm2-vl/DECISIONS.md`
- `docs/lfm2-vl/FAILURE_LOG.md`
- `docs/lfm2-vl/HISTORY.md`
- `docs/lfm2-vl/LICENSE_NOTES.md`
- `docs/lfm2-vl/MOD_MANIFEST.md`
- `docs/lfm2-vl/MODULE_LAYOUT.md`
- `docs/lfm2-vl/PARITY.md`
- `docs/lfm2-vl/SOURCES.md`
- `docs/lfm2-vl/SPEC.md`
- `docs/lfm2-vl/START_HERE.md`
- `docs/lfm2-vl/STATUS.md`
- `docs/lfm2-vl/TENSOR_MAP.md`
- `docs/lfm2-vl/TODO.md`
- `docs/lfm2-vl/history/BOOTSTRAP_AND_PHASE_GUIDE.md`
- `summary_bank.json`

### Local verification scripts

- `scripts/lfm2-vl/env-report.sh`
- `scripts/lfm2-vl/preflight.ps1`
- `scripts/lfm2-vl/run-bounded-oracle.ps1`
- `scripts/lfm2-vl/test-bounded-oracle.ps1`
- `scripts/lfm2-vl/test-preflight.ps1`
- `scripts/lfm2-vl/verify-baseline.sh`
- `scripts/lfm2-vl/verify-mod-manifest.sh`
- `scripts/lfm2-vl/verify-module-layout.py`
- `scripts/lfm2-vl/verify-summary-bank.ps1`

### Committed deterministic fixtures

- `tests/fixtures/lfm2_vl_tiny/README.md`
- `tests/fixtures/lfm2_vl_tiny/manifest.json`
- `tests/fixtures/lfm2_vl_tiny/metadata.json`
- `tests/fixtures/lfm2_vl_tiny/tensors.safetensors`
- `tests/fixtures/lfm2_vl_processor_tiny/README.md`
- `tests/fixtures/lfm2_vl_processor_tiny/manifest.json`
- `tests/fixtures/lfm2_vl_processor_tiny/metadata.json`
- `tests/fixtures/lfm2_vl_processor_tiny/tensors.safetensors`
- `tests/fixtures/lfm2_vl_mmproj_tiny/README.md`
- `tests/fixtures/lfm2_vl_mmproj_tiny/mmproj.json`
- `tests/fixtures/lfm2_vl_mmproj_tiny/mmproj.safetensors`
- `tests/fixtures/lfm2_vl_mmproj_tiny/processor_config.json`
- `tests/fixtures/lfm2_vl_mmproj_tiny/source_model_config.json`

### Export, inspection, and reference tools

- `tools/export_lfm2_vl_mmproj.py`
- `tools/lfm2_vl/README.md`
- `tools/lfm2_vl/reference-lock.json`
- `tools/lfm2_vl/reference/README.md`
- `tools/lfm2_vl/reference/acquire_snapshot.py`
- `tools/lfm2_vl/reference/export_fixtures.py`
- `tools/lfm2_vl/reference/export_processor_fixture.py`
- `tools/lfm2_vl/reference/production_trace.py`
- `tools/lfm2_vl/reference/compare_traces.py`
- `tools/lfm2_vl/reference/inspect_artifact.py`
- `tools/lfm2_vl/reference/inspect_config.py`
- `tools/lfm2_vl/reference/inspect_gguf_header.py`
- `tools/lfm2_vl/reference/manifest.py`
- `tools/lfm2_vl/reference/requirements-reference.in`
- `tools/lfm2_vl/reference/requirements-reference.txt`
- `tools/lfm2_vl/reference/requirements-reference-windows.txt`
- `tools/lfm2_vl/reference/tensor_dump.py`
- `tools/lfm2_vl/reference/verify_environment.py`
- `tools/lfm2_vl/reference/test_acquire_snapshot.py`
- `tools/lfm2_vl/reference/test_gguf_header.py`
- `tools/lfm2_vl/reference/test_mmproj_exporter.py`
- `tools/lfm2_vl/reference/test_reference_tools.py`

## Never Publish From the Local Worktree

- `.tools/` and every descendant, including secret material.
- The ignored verifier-only `Cargo.lock` unless the repository policy is explicitly changed.
- `.venv/`, `artifacts/`, `downloads/`, `models/`, Hugging Face caches, or generated reference outputs.
- Production model weights, authentication material, or ad hoc local logs.

Publication must use an explicit path allowlist derived from this manifest, followed by `git diff --cached --name-status`, `git diff --cached --check`, and a staged secret/name audit. Broad staging commands are prohibited.

---
AI-edited: 2026-08-11T22:55:00-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=merge-review | change=merged modular source layout with verified CUDA provenance
