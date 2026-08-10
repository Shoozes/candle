# Project Instructions

## Mission

Extend Candle 0.11.0 with correct, tested, config-driven LFM2.5-VL support, including:

- LFM2.5 text-model compatibility.
- Embedding-driven LFM2 prefill.
- SigLIP2 NaFlex packed-patch vision encoding.
- LFM2-VL pixel-unshuffle multimodal projection.
- Image placeholder expansion and embedding replacement.
- Native safetensors loading.
- Quantized GGUF text with split dense mmproj.
- Direct llama.cpp-compatible GGUF mmproj loading.
- CPU and CUDA verification after CPU parity is proven.

The complete technical specification is:

`docs/lfm2-vl/SPEC.md`

The execution sequence is:

`docs/lfm2-vl/START_HERE.md`

Read both before planning or editing.

## Authority Order

When sources disagree, use this order:

1. The pinned official Hugging Face Transformers implementation.
2. The pinned official LiquidAI model and processor files.
3. Golden tensor fixtures generated from the pinned reference environment.
4. Existing Candle 0.11 architecture and compatibility requirements.
5. mistral.rs as the primary Rust implementation reference.
6. llama.cpp as the GGUF, preprocessing, and independent parity reference.
7. MLX-VLM and Transformers.js as secondary independent references.
8. The written specification.

Document every material conflict in `docs/lfm2-vl/DECISIONS.md`.

## Baseline

- Base revision: Candle 0.11.0.
- Working branch: `feat/lfm2-vl-mmproj`.
- First checkpoint: `LiquidAI/LFM2.5-VL-450M`.
- Second checkpoint: `LiquidAI/LFM2.5-VL-1.6B`.
- First backend: CPU F32.
- CUDA is optional until CPU parity passes.
- Native safetensors precede all GGUF work.
- Text-only LFM2 compatibility must remain intact.

## Required Execution Order

Work only in this sequence:

1. Bootstrap and baseline verification.
2. Reference-source lock and fixture harness.
3. LFM2 text configuration and `forward_embeds`.
4. SigLIP2 NaFlex from preprocessed tensors.
5. Pixel unshuffle, projector, and composite native model.
6. Rust image processor and prompt expansion.
7. Quantized text plus split dense mmproj.
8. Direct GGUF mmproj loading.
9. Quantized mmproj execution.
10. CUDA optimization and broader stabilization.

Do not implement later phases early.

## Engineering Rules

- Prefer the smallest correct change over broad architectural rewrites.
- Do not create a generic VLM framework before LFM2-VL works.
- Do not hardcode checkpoint names, hidden widths, token counts, layer counts, or image dimensions.
- Derive model behavior from normalized configuration.
- Use checked arithmetic for external dimensions and allocations.
- Return actionable errors for malformed images, configs, weights, and token spans.
- Do not silently truncate, pad, duplicate, or discard image features to force a count match.
- Do not silently fall back to text-only behavior.
- Do not use `unwrap`, `expect`, or unchecked indexing on external input paths.
- Preserve current Candle public behavior unless an explicit compatibility change is documented.
- Keep CPU builds functional without CUDA, TensorRT, or NVIDIA libraries.
- Add optional accelerator behavior behind existing or explicit feature gates.
- Do not add production dependencies without documenting why they are required.
- No placeholder implementations, fake output, skeleton production paths, or unproven success claims.
- Generated captions are not proof. Component tensor parity is required.
- Keep comments short and aligned with actual behavior.
- Keep source files focused. Split modules when responsibilities become distinct.
- Do not modify unrelated Candle architectures.
- Do not reformat unrelated code.

## Source and Licensing Rules

- Treat external implementations primarily as references.
- Prefer a fresh Candle-native implementation over copying large code blocks.
- When code is directly adapted, preserve required copyright and license notices.
- Record source repository, file, commit, and license in `docs/lfm2-vl/SOURCES.md`.
- Pin external revisions before using them as parity authorities.
- Never use an unpinned moving branch as a golden reference.

## Workflow Before Editing

Before every task:

1. Read `AGENTS.md`.
2. Read `docs/lfm2-vl/SPEC.md`.
3. Read `docs/lfm2-vl/STATUS.md`.
4. Inspect the relevant current Candle code.
5. State the task boundary and expected files.
6. Run the narrowest existing verification command that establishes the starting state.
7. Identify the exact acceptance gate.

When a safe ambiguity exists, choose the simplest path consistent with the specification and record the decision. Do not stop for cosmetic or naming ambiguity.

## Workflow After Editing

After every task:

1. Run `cargo fmt --all -- --check`.
2. Run targeted tests for the changed module.
3. Run targeted `cargo check` for affected crates and examples.
4. Run `git diff --check`.
5. Inspect the complete diff.
6. Update `docs/lfm2-vl/STATUS.md`.
7. Update `docs/lfm2-vl/DECISIONS.md` when architecture or compatibility decisions changed.
8. Report exact commands, pass/fail status, blockers, and remaining work.

Do not report a test as passing unless it was executed in the current task.

## Verification Policy

Use focused verification during development and broader verification at phase gates.

Minimum Rust checks:

```bash
cargo fmt --all -- --check
cargo check --locked -p candle-core
cargo check --locked -p candle-nn
cargo check --locked -p candle-transformers
git diff --check
```

At relevant gates also check:

```bash
cargo check --locked -p candle-examples --example lfm2
cargo check --locked -p candle-examples --example quantized-lfm2
```

Do not hide pre-existing failures. Record them separately from failures caused by the current change.

## Model and Fixture Rules

- Do not commit production model weights.
- Do not commit Hugging Face caches.
- Do not download full production checkpoints during bootstrap.
- Every production download must record repository, revision, filename, size, and hash.
- Tiny deterministic fixtures belong under `tests/fixtures/lfm2_vl_tiny/`.
- Generated runtime output belongs under ignored `artifacts/`.
- Reference manifests must record package versions, model revision, processor revision, dtype, device, seed, and source image hash.

## Git Rules

- Never use destructive Git commands unless explicitly instructed.
- Do not reset, clean, force-push, rewrite, or discard unrelated work.
- Do not commit unless the current task explicitly requests a commit.
- Keep commits limited to one proven responsibility.
- Do not open a pull request during early implementation phases.
- Create a checkpoint commit after every green phase gate.
- Review staged files before committing.

## Codex Task Scope

One Codex task should prove one focused result.

Good task:

- Normalize LFM2.5 feed-forward dimensions and add tests.

Bad task:

- Implement all LFM2-VL and GGUF support.

Subagents may perform read-only source comparison or test planning. Do not allow multiple agents to edit overlapping files concurrently.

## Status Handoff

`docs/lfm2-vl/STATUS.md` must always state:

- Current phase.
- Current baseline commit.
- Last green verification.
- Files currently under active work.
- Proven behavior.
- Known failures.
- Blockers.
- Exact next task.

The next Codex session must be able to continue from this file without reconstructing project history from chat logs.
