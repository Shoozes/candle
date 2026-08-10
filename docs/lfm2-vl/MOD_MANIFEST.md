# LFM2-VL Mod File Manifest

This manifest separates the LFM2-VL mod from its Candle 0.11.0 fork base. It is the publication allowlist authority; files not represented here must not be staged merely because they exist in the worktree.

## Classification Rule

- Untouched fork baseline: Candle commit `31f35b147389700ed2a178ee66a91c3cc25cc80d`.
- Current committed mod history: through `f14a46a6967c38e84d99c08801234fd98aa2203a` on `feat/lfm2-vl-mmproj`.
- A **fork-origin modification** is a path that exists in the untouched baseline and is intentionally changed by this mod.
- A **mod-owned addition** is a path absent from the untouched baseline and created for this project.
- “Mod-owned” describes repository provenance, not third-party authorship. External source and license provenance remains authoritative in `SOURCES.md` and `LICENSE_NOTES.md`.
- Every other tracked path is inherited unchanged from the Candle fork.

## Fork-Origin Files Intentionally Modified

Exactly these nine baseline files contain mod changes:

| Path | LFM2-VL reason |
| --- | --- |
| `.gitignore` | Exclude local reference environments, caches, downloads, production models, and generated artifacts. |
| `Cargo.toml` | Register the new `candle-vlm` workspace crate and its shared dependencies. |
| `candle-core/src/quantized/gguf_file.rs` | Add bounded GGUF directory parsing and validation used by direct MMProj loading. |
| `candle-examples/Cargo.toml` | Wire the new LFM2-VL example to its runtime and fixture dependencies. |
| `candle-examples/examples/lfm2/main.rs` | Preserve and prove current LFM2.5 text configuration compatibility. |
| `candle-transformers/Cargo.toml` | Add dependencies required by LFM2-VL loading and verification. |
| `candle-transformers/src/models/lfm2.rs` | Normalize LFM2.5 configuration and add dense embedding-driven forwarding/cache support. |
| `candle-transformers/src/models/mod.rs` | Register the new SigLIP2 and LFM2-VL model modules. |
| `candle-transformers/src/models/quantized_lfm2.rs` | Add validated GGUF metadata, tied-output handling, embedding-driven forwarding, and cache support for hybrid execution. |

No other Candle-baseline source file is part of the mod delta.

## Mod-Owned Runtime and Example Additions

### LFM2-VL example

- `candle-examples/examples/lfm2-vl/args.rs`
- `candle-examples/examples/lfm2-vl/loading.rs`
- `candle-examples/examples/lfm2-vl/main.rs`
- `candle-examples/examples/lfm2-vl/native_checkpoint.rs`
- `candle-examples/examples/lfm2-vl/native_loading.rs`
- `candle-examples/examples/lfm2-vl/runner.rs`

### Transformer models and loaders

- `candle-transformers/src/models/siglip2.rs`
- `candle-transformers/src/models/lfm2_vl/config.rs`
- `candle-transformers/src/models/lfm2_vl/gguf.rs`
- `candle-transformers/src/models/lfm2_vl/linear.rs`
- `candle-transformers/src/models/lfm2_vl/mod.rs`
- `candle-transformers/src/models/lfm2_vl/model.rs`
- `candle-transformers/src/models/lfm2_vl/projector.rs`
- `candle-transformers/src/models/lfm2_vl/weights.rs`

### Rust-native vision-language processing crate

- `candle-vlm/Cargo.toml`
- `candle-vlm/src/lib.rs`
- `candle-vlm/src/image.rs`
- `candle-vlm/src/lfm2_vl/config.rs`
- `candle-vlm/src/lfm2_vl/mod.rs`
- `candle-vlm/src/lfm2_vl/processor.rs`
- `candle-vlm/src/lfm2_vl/prompt.rs`
- `candle-vlm/src/lfm2_vl/types.rs`

## Mod-Owned Project Control and Evidence

### Repository control and design documents

- `AGENTS.md`
- `docs/lfm2-vl/DECISIONS.md`
- `docs/lfm2-vl/FAILURE_LOG.md`
- `docs/lfm2-vl/LICENSE_NOTES.md`
- `docs/lfm2-vl/MOD_MANIFEST.md`
- `docs/lfm2-vl/PARITY.md`
- `docs/lfm2-vl/SOURCES.md`
- `docs/lfm2-vl/SPEC.md`
- `docs/lfm2-vl/START_HERE.md`
- `docs/lfm2-vl/STATUS.md`
- `docs/lfm2-vl/TENSOR_MAP.md`

### Local verification scripts

- `scripts/lfm2-vl/env-report.sh`
- `scripts/lfm2-vl/run-bounded-oracle.ps1`
- `scripts/lfm2-vl/test-bounded-oracle.ps1`
- `scripts/lfm2-vl/verify-baseline.sh`

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
- `tools/lfm2_vl/reference/export_fixtures.py`
- `tools/lfm2_vl/reference/export_processor_fixture.py`
- `tools/lfm2_vl/reference/inspect_config.py`
- `tools/lfm2_vl/reference/inspect_gguf_header.py`
- `tools/lfm2_vl/reference/manifest.py`
- `tools/lfm2_vl/reference/requirements-reference.in`
- `tools/lfm2_vl/reference/requirements-reference.txt`
- `tools/lfm2_vl/reference/tensor_dump.py`
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
AI-edited: 2026-08-10T15:34:55-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=bounded-oracle | change=retained the hardened Windows bounded-oracle owner and smoke-test scripts in the mod-owned allowlist
